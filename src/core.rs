// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! MicroContinualCore v0.4.0 (Единая архитектура: Frozen SO(3) + KerrKernel + 2L Fast-Slow Cayley-RTRL + Page-Hinkley):
//! - 100% #[axiom_zero], zero-alloc горячий путь, полная SEU-защита, 0% exp/ln/sin/cos/tanh.
//! - Полоса 0 («Замороженные часы»): опциональный фиксированный ортогональный ротор Кэли SO(3)^S + KerrKernelBank.
//! - Полоса 1 («Пластичный Fast/Mid Cayley-RTRL», `rtrl_sectors > 0`):
//!   * точный O(S) онлайн-RTRL по (alpha_s, omega_s, w_{s,j}) + алгебраический градиентный RMS;
//!   * опциональная развязанная сетка `(частота x Q)` (`rtrl_grid_mode = 1`).
//! - Полоса 2 («Иерархический Slow Cayley-RTRL», `rtrl_slow_sectors > 0`):
//!   * медленные огибающие SO(3), принимающие `[raw_x, pooled_fast(Band1)]`;
//!   * вместе с Полосой 1 и skip-каналом хранятся ВНУТРИ каждого банка `PageHinkley`!

use axiom_zero::{
    axiom_zero, sanitize_bounded_f32, sanitize_f32, total_div_f32, total_get_copy,
    total_rem_usize, total_set, Presence,
};

use crate::belief::{CayleyBeliefState, EligibilityTraceRing, PageHinkleyDetector};
use crate::math::{
    fast_dot, fast_saxpy, fast_scale, rand_signed_f32, rand_unit_f32, rational_probs_and_psi_into,
    sanitize_and_normalize_unit, splitmix64, SpinorEncoder,
};
use crate::memory::KerrKernelBank;
use crate::rtrl::{PlasticCayleyParams, PlasticCayleyState};

#[axiom_zero]
#[inline]
fn update_normed_vec(w: &mut [f32], dir: &[f32], step: f32, keep: f32, max_norm: f32) {
    if w.is_empty() || dir.is_empty() {
        return;
    }
    if keep < 1.0 {
        fast_scale(w, keep);
    }
    fast_saxpy(w, dir, step);
    let nsq = fast_dot(w, w);
    let lim = sanitize_f32(max_norm * max_norm);
    if nsq > lim {
        fast_scale(w, total_div_f32(max_norm, nsq.sqrt(), 1.0));
    }
}

#[axiom_zero]
#[inline]
fn accum_trace_vec(e: &mut [f32], feat: &[f32], decay: f32, weight: f32) {
    if e.is_empty() || feat.is_empty() {
        return;
    }
    fast_scale(e, decay);
    fast_saxpy(e, feat, weight);
    let nsq = fast_dot(e, e);
    if nsq > 16.0 {
        fast_scale(e, total_div_f32(4.0, nsq.sqrt(), 1.0));
    }
}

#[axiom_zero]
#[inline]
fn update_actor_band(
    actor_w: &mut [f32],
    e_actor: &[f32],
    n_act: usize,
    fdim: usize,
    step_lr: f32,
    keep: f32,
) {
    if fdim == 0 || actor_w.is_empty() {
        return;
    }
    for a in 0..n_act {
        let off = a.saturating_mul(fdim);
        if let (Some(w), Some(ef)) = (
            actor_w.get_mut(off..off + fdim),
            e_actor.get(off..off + fdim),
        ) {
            update_normed_vec(w, ef, step_lr, keep, 6.0);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MicroCoreConfig {
    pub half_sectors: usize,
    pub sem_slots_pow2: usize,
    pub num_actions: usize,
    pub max_regimes: usize,
    pub obs_bank_capacity: usize,
    pub bank_capacity: usize,
    pub trace_capacity: usize,
    pub belief_lambda: f32,
    pub belief_gamma: f32,
    pub rotor_norm: f32,
    pub novelty_gate: f32,
    pub belief_tau: f32,
    pub belief_coalesce: f32,
    pub trace_gamma: f32,
    pub explore_ucb: f32,
    pub intrinsic_curiosity: f32,
    pub linear_lr: f32,
    pub linear_mix: f32,
    pub linear_wd: f32,
    pub standardize_vec: bool,
    pub delta_obs_scale: f32,
    pub actor_critic_lr: f32,
    pub critic_lr: f32,
    pub entropy_beta: f32,
    /// Число пластичных 3D-секторов Band 1 (Fast/Mid) Cayley-RTRL (0 = выключено).
    pub rtrl_sectors: usize,
    /// Число пластичных 3D-секторов Band 2 (Slow Envelope) Cayley-RTRL (0 = выключено).
    pub rtrl_slow_sectors: usize,
    /// Число сырых входов для пластичной полосы RTRL.
    pub rtrl_max_in: usize,
    /// Скорость онлайн-RTRL обучения внутренних параметров (alpha_s, omega_s, w_{s,j}).
    pub rtrl_lr: f32,
    /// Режим сетки RTRL: 0 = сопряжённая, 1 = развязанная (частота x добротность Q).
    pub rtrl_grid_mode: u8,
    /// Включить прямой skip-канал 0.5*x_t в считывание Actor/Critic.
    pub rtrl_use_skip_x: bool,
    /// Масштаб рациональной нормализации Norm_kappa в RTRL.
    pub rtrl_kappa: f32,
    pub ph_delta: f32,
    pub ph_threshold: f32,
    pub ph_warmup: usize,
    /// Параллельный read-only скоринг всех зрелых банков при дрейфе за O(1) окно (8 шагов)
    /// вместо последовательного перебора O(R) банков (для многорежимных потоков R > 2).
    pub parallel_bank_probe: bool,
    pub seed: u64,
}

impl Default for MicroCoreConfig {
    fn default() -> Self {
        Self {
            half_sectors: 32,
            sem_slots_pow2: 10,
            num_actions: 12,
            max_regimes: 6,
            obs_bank_capacity: 32,
            bank_capacity: 112,
            trace_capacity: 24,
            belief_lambda: 0.85,
            belief_gamma: 1.25,
            rotor_norm: 0.09,
            novelty_gate: 0.28,
            belief_tau: 0.48,
            belief_coalesce: 0.78,
            trace_gamma: 0.90,
            explore_ucb: 0.48,
            intrinsic_curiosity: 0.18,
            linear_lr: 0.0,
            linear_mix: 0.0,
            linear_wd: 0.0,
            standardize_vec: false,
            delta_obs_scale: 0.0,
            actor_critic_lr: 0.0,
            critic_lr: 0.0,
            entropy_beta: 0.0,
            rtrl_sectors: 0,
            rtrl_slow_sectors: 0,
            rtrl_max_in: 4,
            rtrl_lr: 0.0,
            rtrl_grid_mode: 0,
            rtrl_use_skip_x: false,
            rtrl_kappa: 0.5,
            ph_delta: 0.22,
            ph_threshold: 3.20,
            ph_warmup: 45,
            parallel_bank_probe: false,
            seed: 0x4D49_4352_4F56_0001,
        }
    }
}

impl MicroCoreConfig {
    pub fn numeric_stream(num_classes: usize, seed: u64) -> Self {
        Self {
            half_sectors: 20,
            sem_slots_pow2: 0,
            num_actions: num_classes.clamp(1, 32),
            max_regimes: 4,
            obs_bank_capacity: 0,
            bank_capacity: 28,
            trace_capacity: 4,
            belief_lambda: 0.80,
            belief_gamma: 0.0,
            rotor_norm: 0.10,
            novelty_gate: 0.0,
            belief_tau: 0.30,
            belief_coalesce: 0.84,
            trace_gamma: 0.90,
            explore_ucb: 0.0,
            intrinsic_curiosity: 0.0,
            linear_lr: 0.09,
            linear_mix: 0.88,
            linear_wd: 0.0022,
            standardize_vec: true,
            delta_obs_scale: 0.0,
            actor_critic_lr: 0.0,
            critic_lr: 0.0,
            entropy_beta: 0.0,
            rtrl_sectors: 0,
            rtrl_slow_sectors: 0,
            rtrl_max_in: 4,
            rtrl_lr: 0.0,
            rtrl_grid_mode: 0,
            rtrl_use_skip_x: false,
            rtrl_kappa: 0.5,
            ph_delta: 0.14,
            ph_threshold: 3.6,
            ph_warmup: 50,
            parallel_bank_probe: false,
            seed,
        }
    }
}

#[derive(Clone)]
pub struct RegimeBank {
    pub obs_kernel: KerrKernelBank,
    pub belief_kernel: KerrKernelBank,
    pub linear_w: Vec<f32>,
    pub linear_bias: Vec<f32>,
    pub critic_w: Vec<f32>,
    pub critic_bias: f32,
    /// Полоса 1 (Fast/Mid Cayley-RTRL) + головки Actor/Critic.
    pub rtrl_params: PlasticCayleyParams,
    pub rtrl_actor_w: Vec<f32>,  // [num_actions * rtrl_dim]
    pub rtrl_critic_w: Vec<f32>, // [rtrl_dim]
    /// Полоса 2 (Slow Envelope Cayley-RTRL, 0 байт при rtrl_slow_sectors == 0).
    pub rtrl_slow_params: PlasticCayleyParams,
    pub rtrl_slow_actor_w: Vec<f32>,  // [num_actions * rtrl_slow_dim]
    pub rtrl_slow_critic_w: Vec<f32>, // [rtrl_slow_dim]
    /// Прямой skip-канал 0.5*raw_x (0 байт при !rtrl_use_skip_x).
    pub rtrl_skip_actor_w: Vec<f32>,  // [num_actions * skip_dim]
    pub rtrl_skip_critic_w: Vec<f32>, // [skip_dim]
    pub dim: usize,
    pub rtrl_dim: usize,
    pub rtrl_slow_dim: usize,
    pub skip_dim: usize,
    pub num_actions: usize,
    pub ph: PageHinkleyDetector,
    pub cooldown_until: usize,
    pub age_eps: usize,
    pub solved_eps: usize,
    pub ret_ema: f32,
}

#[axiom_zero]
fn fill_small_rand(buf: &mut [f32], seed: u64, scale: f32) {
    if buf.is_empty() {
        return;
    }
    let mut rng = splitmix64(seed);
    for w in buf.iter_mut() {
        *w = sanitize_f32(rand_signed_f32(&mut rng) * scale);
    }
}

#[axiom_zero]
fn scrub_vec_seu(buf: &mut [f32], limit: f32, replacement: f32) -> u64 {
    let mut faults = 0u64;
    for v in buf.iter_mut() {
        if !v.is_finite() || v.abs() > limit {
            *v = replacement;
            faults = faults.saturating_add(1);
        }
    }
    faults
}

#[axiom_zero]
impl RegimeBank {
    pub fn new(cfg: &MicroCoreConfig, dim: usize, half_dim: usize, bank_idx: usize) -> Self {
        let na = cfg.num_actions.clamp(1, 64);
        let has_frozen = cfg.half_sectors > 0;
        let use_lin = has_frozen && (cfg.linear_lr > 0.0 || cfg.actor_critic_lr > 0.0);
        let use_ac = cfg.actor_critic_lr > 0.0;
        let use_bias = cfg.linear_lr > 0.0 || use_ac;
        let need_rtrl_critic = use_ac || na >= 2;

        let rtrl_seed = cfg.seed.wrapping_add((bank_idx as u64).wrapping_mul(0x9E37_79B9));
        let slow_sec = if cfg.rtrl_sectors > 0 { cfg.rtrl_slow_sectors } else { 0 };
        let slow_in = if slow_sec > 0 {
            cfg.rtrl_max_in.clamp(1, 256).saturating_add(4).min(260)
        } else {
            1
        };

        let rtrl_params = PlasticCayleyParams::new_with_grid(
            cfg.rtrl_sectors,
            cfg.rtrl_max_in,
            if slow_sec > 0 { rtrl_seed ^ 0xFA57_0001 } else { rtrl_seed },
            cfg.rtrl_grid_mode,
        );
        let rtrl_slow_params = PlasticCayleyParams::new_with_grid(
            slow_sec,
            slow_in,
            rtrl_seed ^ 0x510B_0002,
            if cfg.rtrl_grid_mode > 0 { 2 } else { 0 },
        );

        let rdim = rtrl_params.dim();
        let r2dim = rtrl_slow_params.dim();
        let skip_dim = if cfg.rtrl_sectors > 0 && cfg.rtrl_use_skip_x {
            cfg.rtrl_max_in.clamp(1, 256)
        } else {
            0
        };

        let w_init = if slow_sec > 0 { 0.05_f32 } else { 0.08_f32 };
        let mut rtrl_actor_w = if rdim > 0 { vec![0.0_f32; na.saturating_mul(rdim)] } else { Vec::new() };
        let mut rtrl_critic_w = if rdim > 0 && need_rtrl_critic { vec![0.0_f32; rdim] } else { Vec::new() };
        fill_small_rand(&mut rtrl_actor_w, rtrl_seed ^ 0xA4C7_0011_55AA_33CC, w_init);
        fill_small_rand(&mut rtrl_critic_w, rtrl_seed ^ 0xC4C7_0011_55AA_33DD, w_init);

        let mut rtrl_slow_actor_w = if r2dim > 0 { vec![0.0_f32; na.saturating_mul(r2dim)] } else { Vec::new() };
        let mut rtrl_slow_critic_w = if r2dim > 0 && need_rtrl_critic { vec![0.0_f32; r2dim] } else { Vec::new() };
        fill_small_rand(&mut rtrl_slow_actor_w, rtrl_seed ^ 0xB5D8_1122_66BB_44EE, w_init);
        fill_small_rand(&mut rtrl_slow_critic_w, rtrl_seed ^ 0xD5D8_1122_66BB_44FF, w_init);

        let mut rtrl_skip_actor_w = if skip_dim > 0 { vec![0.0_f32; na.saturating_mul(skip_dim)] } else { Vec::new() };
        let mut rtrl_skip_critic_w = if skip_dim > 0 && need_rtrl_critic { vec![0.0_f32; skip_dim] } else { Vec::new() };
        fill_small_rand(&mut rtrl_skip_actor_w, rtrl_seed ^ 0x5119_0011_77CC_8899, w_init);
        fill_small_rand(&mut rtrl_skip_critic_w, rtrl_seed ^ 0x5119_0011_77CC_88AA, w_init);

        let need_kernels = has_frozen && cfg.actor_critic_lr <= 0.0 && cfg.linear_mix < 0.99;
        let obs_cap = if need_kernels { cfg.obs_bank_capacity } else { 0 };
        let bel_cap = if need_kernels { cfg.bank_capacity } else { 0 };
        Self {
            obs_kernel: KerrKernelBank::new(dim, half_dim, na, obs_cap, 0.72),
            belief_kernel: KerrKernelBank::new(dim, half_dim, na, bel_cap, cfg.belief_coalesce),
            linear_w: if use_lin { vec![0.0_f32; na.saturating_mul(dim)] } else { Vec::new() },
            linear_bias: if use_bias { vec![0.0_f32; na] } else { Vec::new() },
            critic_w: if has_frozen && use_ac { vec![0.0_f32; dim] } else { Vec::new() },
            critic_bias: 0.0,
            rtrl_params,
            rtrl_actor_w,
            rtrl_critic_w,
            rtrl_slow_params,
            rtrl_slow_actor_w,
            rtrl_slow_critic_w,
            rtrl_skip_actor_w,
            rtrl_skip_critic_w,
            dim,
            rtrl_dim: rdim,
            rtrl_slow_dim: r2dim,
            skip_dim,
            num_actions: na,
            ph: PageHinkleyDetector::new(cfg.ph_delta, cfg.ph_threshold, cfg.ph_warmup),
            cooldown_until: 0,
            age_eps: 0,
            solved_eps: 0,
            ret_ema: 0.0,
        }
    }

    pub fn clear_all(&mut self, cfg: &MicroCoreConfig, bank_idx: usize) {
        self.obs_kernel.clear();
        self.belief_kernel.clear();
        self.linear_w.fill(0.0);
        self.linear_bias.fill(0.0);
        self.critic_w.fill(0.0);
        self.critic_bias = 0.0;
        let rtrl_seed = cfg.seed.wrapping_add((bank_idx as u64 + 7).wrapping_mul(0x9E37_79B9));
        let slow_sec = if cfg.rtrl_sectors > 0 { cfg.rtrl_slow_sectors } else { 0 };
        self.rtrl_params.reset_with_grid(
            if slow_sec > 0 { rtrl_seed ^ 0xFA57_0001 } else { rtrl_seed },
            cfg.rtrl_grid_mode,
        );
        self.rtrl_slow_params.reset_with_grid(
            rtrl_seed ^ 0x510B_0002,
            if cfg.rtrl_grid_mode > 0 { 2 } else { 0 },
        );
        let w_init = if slow_sec > 0 { 0.05_f32 } else { 0.08_f32 };
        fill_small_rand(&mut self.rtrl_actor_w, rtrl_seed ^ 0xA4C7_0011_55AA_33CC, w_init);
        fill_small_rand(&mut self.rtrl_critic_w, rtrl_seed ^ 0xC4C7_0011_55AA_33DD, w_init);
        fill_small_rand(&mut self.rtrl_slow_actor_w, rtrl_seed ^ 0xB5D8_1122_66BB_44EE, w_init);
        fill_small_rand(&mut self.rtrl_slow_critic_w, rtrl_seed ^ 0xD5D8_1122_66BB_44FF, w_init);
        fill_small_rand(&mut self.rtrl_skip_actor_w, rtrl_seed ^ 0x5119_0011_77CC_8899, w_init);
        fill_small_rand(&mut self.rtrl_skip_critic_w, rtrl_seed ^ 0x5119_0011_77CC_88AA, w_init);
        self.ph = PageHinkleyDetector::new(cfg.ph_delta, cfg.ph_threshold, cfg.ph_warmup);
        self.age_eps = 0;
        self.solved_eps = 0;
        self.ret_ema = 0.0;
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.obs_kernel.memory_footprint_bytes()
            + self.belief_kernel.memory_footprint_bytes()
            + self.rtrl_params.memory_footprint_bytes()
            + self.rtrl_slow_params.memory_footprint_bytes()
            + (self.linear_w.len()
                + self.linear_bias.len()
                + self.critic_w.len()
                + self.rtrl_actor_w.len()
                + self.rtrl_critic_w.len()
                + self.rtrl_slow_actor_w.len()
                + self.rtrl_slow_critic_w.len()
                + self.rtrl_skip_actor_w.len()
                + self.rtrl_skip_critic_w.len())
            .saturating_mul(4)
            + 64
    }

    pub fn scrub_weight_seu(&mut self) -> u64 {
        let mut faults = self.rtrl_params.scrub_weight_seu();
        faults = faults.saturating_add(self.rtrl_slow_params.scrub_weight_seu());
        faults = faults.saturating_add(scrub_vec_seu(&mut self.rtrl_actor_w, 6.1, 0.0));
        faults = faults.saturating_add(scrub_vec_seu(&mut self.rtrl_slow_actor_w, 6.1, 0.0));
        faults = faults.saturating_add(scrub_vec_seu(&mut self.rtrl_skip_actor_w, 6.1, 0.0));
        faults = faults.saturating_add(scrub_vec_seu(&mut self.linear_w, 6.1, 0.0));
        faults = faults.saturating_add(scrub_vec_seu(&mut self.rtrl_critic_w, 16.1, 0.0));
        faults = faults.saturating_add(scrub_vec_seu(&mut self.rtrl_slow_critic_w, 16.1, 0.0));
        faults = faults.saturating_add(scrub_vec_seu(&mut self.rtrl_skip_critic_w, 16.1, 0.0));
        faults = faults.saturating_add(scrub_vec_seu(&mut self.critic_w, 16.1, 0.0));
        faults = faults.saturating_add(scrub_vec_seu(&mut self.linear_bias, 16.1, 0.0));
        if !self.critic_bias.is_finite() || self.critic_bias.abs() > 16.1 {
            self.critic_bias = 0.0;
            faults = faults.saturating_add(1);
        }
        faults
    }

    #[inline]
    pub fn linear_score(&self, action: usize, b_vec: &[f32]) -> f32 {
        let bias = total_get_copy(&self.linear_bias, action, 0.0);
        if self.linear_w.is_empty() {
            return sanitize_f32(bias);
        }
        let off = action.saturating_mul(self.dim);
        let w = self.linear_w.get(off..off + self.dim).unwrap_or(&[]);
        sanitize_f32(fast_dot(w, b_vec) + bias)
    }

    #[inline]
    pub fn split_band_logit(
        &self,
        action: usize,
        frozen_b: &[f32],
        plastic_b: &[f32],
        slow_b: &[f32],
        skip_x: &[f32],
    ) -> f32 {
        let mut s = self.linear_score(action, frozen_b);
        if self.rtrl_dim > 0 {
            let r_scale = total_div_f32(1.0, (self.rtrl_params.sectors.max(1) as f32).sqrt(), 0.25);
            let roff = action.saturating_mul(self.rtrl_dim);
            let rw = self.rtrl_actor_w.get(roff..roff + self.rtrl_dim).unwrap_or(&[]);
            s = sanitize_f32(s + r_scale * fast_dot(rw, plastic_b));
        }
        if self.rtrl_slow_dim > 0 {
            let s_scale = total_div_f32(1.0, (self.rtrl_slow_params.sectors.max(1) as f32).sqrt(), 0.25);
            let soff = action.saturating_mul(self.rtrl_slow_dim);
            let sw = self.rtrl_slow_actor_w.get(soff..soff + self.rtrl_slow_dim).unwrap_or(&[]);
            s = sanitize_f32(s + s_scale * fast_dot(sw, slow_b));
        }
        if self.skip_dim > 0 {
            let xoff = action.saturating_mul(self.skip_dim);
            let xw = self.rtrl_skip_actor_w.get(xoff..xoff + self.skip_dim).unwrap_or(&[]);
            s = sanitize_f32(s + fast_dot(xw, skip_x));
        }
        s
    }

    #[inline]
    pub fn critic_value(&self, frozen_b: &[f32], plastic_b: &[f32], slow_b: &[f32], skip_x: &[f32]) -> f32 {
        let mut v = self.critic_bias;
        if !self.critic_w.is_empty() {
            v = sanitize_f32(v + fast_dot(&self.critic_w, frozen_b));
        }
        if !self.rtrl_critic_w.is_empty() {
            let r_scale = total_div_f32(1.0, (self.rtrl_params.sectors.max(1) as f32).sqrt(), 0.25);
            v = sanitize_f32(v + r_scale * fast_dot(&self.rtrl_critic_w, plastic_b));
        }
        if !self.rtrl_slow_critic_w.is_empty() {
            let s_scale = total_div_f32(1.0, (self.rtrl_slow_params.sectors.max(1) as f32).sqrt(), 0.25);
            v = sanitize_f32(v + s_scale * fast_dot(&self.rtrl_slow_critic_w, slow_b));
        }
        if !self.rtrl_skip_critic_w.is_empty() {
            v = sanitize_f32(v + fast_dot(&self.rtrl_skip_critic_w, skip_x));
        }
        v
    }

    pub fn linear_delta_update(&mut self, action: usize, b_vec: &[f32], target: f32, lr: f32, wd: f32) {
        if action >= self.num_actions || lr <= 0.0 || self.linear_w.is_empty() {
            return;
        }
        let pred = self.linear_score(action, b_vec);
        let err = sanitize_bounded_f32(target - pred, 2.5);
        let step = sanitize_f32(lr * err);
        let keep = (1.0 - sanitize_f32(wd).clamp(0.0, 0.05)).clamp(0.95, 1.0);
        let off = action.saturating_mul(self.dim);
        if let Some(w) = self.linear_w.get_mut(off..off + self.dim) {
            update_normed_vec(w, b_vec, step, keep, 6.0);
        }
        let old_b = total_get_copy(&self.linear_bias, action, 0.0);
        let _ = total_set(
            &mut self.linear_bias,
            action,
            sanitize_bounded_f32(old_b * keep + step * 0.25, 3.0),
        );
    }

    pub fn critic_update(&mut self, b_vec: &[f32], target_v: f32, lr: f32) {
        if self.critic_w.is_empty() || lr <= 0.0 {
            return;
        }
        let v = self.critic_value(b_vec, &[], &[], &[]);
        let err = sanitize_bounded_f32(target_v - v, 3.0);
        let step = sanitize_f32(lr * err);
        update_normed_vec(&mut self.critic_w, b_vec, step, 1.0, 16.0);
        self.critic_bias = sanitize_bounded_f32(self.critic_bias + step * 0.25, 25.0);
    }

    pub fn actor_ac_update(
        &mut self,
        chosen_act: usize,
        b_vec: &[f32],
        probs: &[f32],
        psis: &[f32],
        adv: f32,
        lr: f32,
        ent_beta: f32,
        wd: f32,
    ) {
        if lr <= 0.0 || self.linear_w.is_empty() {
            return;
        }
        let na = self.num_actions;
        let a_clamped = sanitize_bounded_f32(adv, 2.5);
        let mut sum_p2 = 0.0_f32;
        for a in 0..na {
            let p = total_get_copy(probs, a, 0.0).max(0.0);
            sum_p2 = sanitize_f32(sum_p2 + p * p);
        }
        let mut sum_s = 0.0_f32;
        let mut sum_h = 0.0_f32;
        for a in 0..na {
            let p = total_get_copy(probs, a, 0.0).max(0.0);
            let psi = total_get_copy(psis, a, 1.0);
            let ind = if a == chosen_act { 1.0_f32 } else { 0.0_f32 };
            sum_s = sanitize_f32(sum_s + psi * (ind - p));
            sum_h = sanitize_f32(sum_h + p * psi * (sum_p2 - p));
        }
        let mean_s = total_div_f32(sum_s, na.max(1) as f32, 0.0);
        let mean_h = total_div_f32(sum_h, na.max(1) as f32, 0.0);

        let keep = (1.0 - sanitize_f32(wd).clamp(0.0, 0.05)).clamp(0.95, 1.0);
        for a in 0..na {
            let p = total_get_copy(probs, a, 0.0).max(0.0);
            let psi = total_get_copy(psis, a, 1.0);
            let ind = if a == chosen_act { 1.0_f32 } else { 0.0_f32 };
            let d_pg = sanitize_f32(a_clamped * (psi * (ind - p) - mean_s));
            let d_ent = sanitize_f32(ent_beta * (p * psi * (sum_p2 - p) - mean_h));
            let step = sanitize_f32(lr * (d_pg + d_ent));
            let off = a.saturating_mul(self.dim);
            if let Some(w) = self.linear_w.get_mut(off..off + self.dim) {
                update_normed_vec(w, b_vec, step, keep, 6.0);
            }
            let old_b = total_get_copy(&self.linear_bias, a, 0.0);
            let _ = total_set(
                &mut self.linear_bias,
                a,
                sanitize_bounded_f32(old_b * keep + step * 0.25, 3.0),
            );
        }
    }
}

#[derive(Clone)]
pub struct CoreScratch {
    pub obs_vec: Vec<f32>,
    pub obs_trace: Vec<f32>,
    pub tmp_sem: Vec<f32>,
    pub tmp_key: Vec<f32>,
    pub sims: Vec<f32>,
    pub weights: Vec<f32>,
    pub q_obs: Vec<f32>,
    pub v_obs: Vec<f32>,
    pub q_bel: Vec<f32>,
    pub v_bel: Vec<f32>,
    pub ac_logits: Vec<f32>,
    pub ac_probs: Vec<f32>,
    pub ac_psi: Vec<f32>,
    pub feat_mean: Vec<f32>,
    pub feat_var: Vec<f32>,
    pub feat_count: u64,
    pub prev_raw_obs: Vec<f32>,
    pub has_prev_obs: bool,
    pub vec_aug_buf: Vec<f32>,
    pub ep_slots_unused: Vec<(usize, f32)>,
    pub intra_ep_fatigue: Vec<f32>,
    // Следы и буферы для Frozen + Band 1 (Fast) + Band 2 (Slow) + Skip
    pub td_e_actor_f: Vec<f32>,
    pub td_e_actor_b: Vec<f32>,
    pub td_e_critic_f: Vec<f32>,
    pub td_e_critic_b: f32,
    pub td_e_actor_r: Vec<f32>,
    pub td_e_critic_r: Vec<f32>,
    pub rtrl_fb_buf: Vec<f32>,
    pub td_e_actor_r2: Vec<f32>,
    pub td_e_critic_r2: Vec<f32>,
    pub rtrl_slow_fb_buf: Vec<f32>,
    pub rtrl_slow_in_buf: Vec<f32>,
    pub td_e_actor_skip: Vec<f32>,
    pub td_e_critic_skip: Vec<f32>,
    pub rtrl_skip_buf: Vec<f32>,
    pub has_pending_td: bool,
    pub pending_v: f32,
    pub pending_r: f32,
    pub td_mean_ema: f32,
    pub td_var_ema: f32,
    pub pending_active_in: usize,
    pub parallel_bank_hits: Vec<u16>,
    pub parallel_bank_margins: Vec<f32>,
    pub last_reg_pred: f32,
}

#[axiom_zero]
impl CoreScratch {
    pub fn new(
        dim: usize,
        half_dim: usize,
        bank_cap: usize,
        trace_cap: usize,
        num_actions: usize,
        rtrl_dim: usize,
        rtrl_slow_dim: usize,
        slow_in_dim: usize,
        skip_dim: usize,
        max_in_dim: usize,
        use_frozen: bool,
        need_sem: bool,
        need_kernels: bool,
        need_obs_trace: bool,
    ) -> Self {
        let cap = if need_kernels { bank_cap.max(32) } else { 0 };
        let has_r1 = rtrl_dim > 0;
        let has_r2 = rtrl_slow_dim > 0;
        let has_skip = skip_dim > 0;
        let feat_cap = max_in_dim.clamp(32, 256);
        Self {
            obs_vec: if use_frozen { vec![0.0_f32; dim] } else { Vec::new() },
            obs_trace: if need_obs_trace {
                vec![0.0_f32; trace_cap.clamp(4, 256).saturating_mul(dim)]
            } else {
                Vec::new()
            },
            tmp_sem: if need_sem { vec![0.0_f32; half_dim] } else { Vec::new() },
            tmp_key: if need_kernels { vec![0.0_f32; dim] } else { Vec::new() },
            sims: if need_kernels { vec![0.0_f32; cap] } else { Vec::new() },
            weights: if need_kernels { vec![0.0_f32; cap] } else { Vec::new() },
            q_obs: if need_kernels { vec![0.0_f32; num_actions] } else { Vec::new() },
            v_obs: if need_kernels { vec![0.0_f32; num_actions] } else { Vec::new() },
            q_bel: if need_kernels { vec![0.0_f32; num_actions] } else { Vec::new() },
            v_bel: if need_kernels { vec![0.0_f32; num_actions] } else { Vec::new() },
            ac_logits: vec![0.0_f32; num_actions],
            ac_probs: vec![0.0_f32; num_actions],
            ac_psi: vec![1.0_f32; num_actions],
            feat_mean: vec![0.0_f32; feat_cap],
            feat_var: vec![1.0_f32; feat_cap],
            feat_count: 0,
            prev_raw_obs: vec![0.0_f32; feat_cap],
            has_prev_obs: false,
            vec_aug_buf: vec![0.0_f32; feat_cap.saturating_mul(2)],
            ep_slots_unused: if need_sem { Vec::with_capacity(16) } else { Vec::new() },
            intra_ep_fatigue: vec![0.0_f32; num_actions],
            td_e_actor_f: if has_r1 && use_frozen && num_actions >= 2 { vec![0.0_f32; num_actions.saturating_mul(dim)] } else { Vec::new() },
            td_e_actor_b: if has_r1 && num_actions >= 2 { vec![0.0_f32; num_actions] } else { Vec::new() },
            td_e_critic_f: if has_r1 && use_frozen && num_actions >= 2 { vec![0.0_f32; dim] } else { Vec::new() },
            td_e_critic_b: 0.0,
            td_e_actor_r: if has_r1 && num_actions >= 2 { vec![0.0_f32; num_actions.saturating_mul(rtrl_dim)] } else { Vec::new() },
            td_e_critic_r: if has_r1 && num_actions >= 2 { vec![0.0_f32; rtrl_dim] } else { Vec::new() },
            rtrl_fb_buf: if has_r1 { vec![0.0_f32; rtrl_dim] } else { Vec::new() },
            td_e_actor_r2: if has_r2 && num_actions >= 2 { vec![0.0_f32; num_actions.saturating_mul(rtrl_slow_dim)] } else { Vec::new() },
            td_e_critic_r2: if has_r2 && num_actions >= 2 { vec![0.0_f32; rtrl_slow_dim] } else { Vec::new() },
            rtrl_slow_fb_buf: if has_r2 { vec![0.0_f32; rtrl_slow_dim] } else { Vec::new() },
            rtrl_slow_in_buf: if has_r2 { vec![0.0_f32; slow_in_dim] } else { Vec::new() },
            td_e_actor_skip: if has_skip && num_actions >= 2 { vec![0.0_f32; num_actions.saturating_mul(skip_dim)] } else { Vec::new() },
            td_e_critic_skip: if has_skip && num_actions >= 2 { vec![0.0_f32; skip_dim] } else { Vec::new() },
            rtrl_skip_buf: if has_skip { vec![0.0_f32; skip_dim] } else { Vec::new() },
            has_pending_td: false,
            pending_v: 0.0,
            pending_r: 0.0,
            td_mean_ema: 0.0,
            td_var_ema: 0.25,
            pending_active_in: 0,
            parallel_bank_hits: Vec::new(),
            parallel_bank_margins: Vec::new(),
            last_reg_pred: 0.0,
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        (self.obs_vec.len()
            + self.obs_trace.len()
            + self.tmp_sem.len()
            + self.tmp_key.len()
            + self.sims.len()
            + self.weights.len()
            + self.q_obs.len()
            + self.v_obs.len()
            + self.q_bel.len()
            + self.v_bel.len()
            + self.ac_logits.len()
            + self.ac_probs.len()
            + self.ac_psi.len()
            + self.feat_mean.len()
            + self.feat_var.len()
            + self.prev_raw_obs.len()
            + self.vec_aug_buf.len()
            + self.intra_ep_fatigue.len()
            + self.td_e_actor_f.len()
            + self.td_e_actor_b.len()
            + self.td_e_critic_f.len()
            + self.td_e_actor_r.len()
            + self.td_e_critic_r.len()
            + self.rtrl_fb_buf.len()
            + self.td_e_actor_r2.len()
            + self.td_e_critic_r2.len()
            + self.rtrl_slow_fb_buf.len()
            + self.rtrl_slow_in_buf.len()
            + self.td_e_actor_skip.len()
            + self.td_e_critic_skip.len()
            + self.rtrl_skip_buf.len()
            + self.parallel_bank_margins.len())
        .saturating_mul(4)
            + self.parallel_bank_hits.len().saturating_mul(2)
    }

    pub fn clear_td_traces(&mut self) {
        self.td_e_actor_f.fill(0.0);
        self.td_e_actor_b.fill(0.0);
        self.td_e_critic_f.fill(0.0);
        self.td_e_critic_b = 0.0;
        self.td_e_actor_r.fill(0.0);
        self.td_e_critic_r.fill(0.0);
        self.td_e_actor_r2.fill(0.0);
        self.td_e_critic_r2.fill(0.0);
        self.td_e_actor_skip.fill(0.0);
        self.td_e_critic_skip.fill(0.0);
        self.rtrl_skip_buf.fill(0.0);
        self.rtrl_slow_in_buf.fill(0.0);
        self.has_pending_td = false;
        self.pending_v = 0.0;
        self.pending_r = 0.0;
    }
}

#[derive(Clone)]
pub struct MicroContinualCore {
    pub config: MicroCoreConfig,
    pub encoder: SpinorEncoder,
    pub belief: CayleyBeliefState,
    pub rtrl_state: PlasticCayleyState,
    pub rtrl_slow_state: PlasticCayleyState,
    pub rtrl_pool_w: Vec<f32>,
    pub trace: EligibilityTraceRing,
    pub banks: Vec<RegimeBank>,
    pub active_regime: usize,
    pub episode_counter: usize,
    pub probe_eps_left: usize,
    pub probe_return_sum: f32,
    pub probe_return_cnt: usize,
    pub parallel_probe_left: usize,
    pub parallel_probe_prev: usize,
    pub total_switches: usize,
    pub rng_state: u64,
    pub scratch: CoreScratch,
    pub seu_fault_count: u64,
}

#[axiom_zero]
impl MicroContinualCore {
    pub fn new(config: MicroCoreConfig) -> Self {
        let encoder = SpinorEncoder::new(config.half_sectors, config.sem_slots_pow2);
        let d = encoder.dim;
        let hd = encoder.half_dim;
        let use_frozen = config.half_sectors > 0;
        let mut belief = CayleyBeliefState::new(
            d,
            hd,
            config.belief_lambda,
            config.belief_gamma,
            config.rotor_norm,
            config.seed,
        );
        belief.novelty_gate = sanitize_f32(config.novelty_gate).clamp(0.0, 0.9);

        let kap = config.rtrl_kappa;
        let rtrl_state = PlasticCayleyState::new(config.rtrl_sectors, config.rtrl_max_in, kap);
        let r1_dim = rtrl_state.sectors.saturating_mul(3);

        let slow_sec = if rtrl_state.sectors > 0 { config.rtrl_slow_sectors } else { 0 };
        let slow_in = if slow_sec > 0 {
            config.rtrl_max_in.clamp(1, 256).saturating_add(4).min(260)
        } else {
            1
        };
        let rtrl_slow_state = PlasticCayleyState::new(slow_sec, slow_in, kap);
        let r2_dim = rtrl_slow_state.sectors.saturating_mul(3);

        let mut rtrl_pool_w = if r2_dim > 0 {
            vec![0.0_f32; 4usize.saturating_mul(r1_dim)]
        } else {
            Vec::new()
        };
        if !rtrl_pool_w.is_empty() && r1_dim > 0 {
            let p_scale = total_div_f32(1.0, (r1_dim.max(1) as f32).sqrt(), 0.2);
            fill_small_rand(&mut rtrl_pool_w, config.seed ^ 0x41E2_CA71_E700, p_scale);
            // Modified Gram-Schmidt (2 passes) so the 4 rows of rtrl_pool_w are orthonormal (P P^T = I_4).
            for r in 0..4usize {
                let roff = r.saturating_mul(r1_dim);
                for _pass in 0..2usize {
                    for prev in 0..r {
                        let poff = prev.saturating_mul(r1_dim);
                        let mut dot = 0.0_f32;
                        for i in 0..r1_dim {
                            dot = sanitize_f32(
                                dot + total_get_copy(&rtrl_pool_w, roff + i, 0.0)
                                    * total_get_copy(&rtrl_pool_w, poff + i, 0.0),
                            );
                        }
                        for i in 0..r1_dim {
                            let nv = sanitize_f32(
                                total_get_copy(&rtrl_pool_w, roff + i, 0.0)
                                    - dot * total_get_copy(&rtrl_pool_w, poff + i, 0.0),
                            );
                            let _ = total_set(&mut rtrl_pool_w, roff + i, nv);
                        }
                    }
                }
                let mut n2 = 0.0_f32;
                for i in 0..r1_dim {
                    let v = total_get_copy(&rtrl_pool_w, roff + i, 0.0);
                    n2 = sanitize_f32(n2 + v * v);
                }
                let norm = n2.sqrt();
                if norm > 1e-6 {
                    let inv_n = total_div_f32(1.0, norm, 0.0);
                    for i in 0..r1_dim {
                        let nv = sanitize_f32(total_get_copy(&rtrl_pool_w, roff + i, 0.0) * inv_n);
                        let _ = total_set(&mut rtrl_pool_w, roff + i, nv);
                    }
                } else {
                    for i in 0..r1_dim {
                        let _ = total_set(&mut rtrl_pool_w, roff + i, 0.0);
                    }
                }
            }
        }

        let skip_dim = if rtrl_state.sectors > 0 && config.rtrl_use_skip_x {
            config.rtrl_max_in.clamp(1, 256)
        } else {
            0
        };

        let n_act = config.num_actions.clamp(1, 64);
        let use_trace = !(config.actor_critic_lr > 0.0 && r1_dim > 0);
        let trace_cap = if use_trace && !(n_act == 1 && config.actor_critic_lr <= 0.0) {
            config.trace_capacity
        } else {
            0
        };
        let trace_na = if config.actor_critic_lr > 0.0 && r1_dim == 0 { n_act } else { 0 };
        let trace = EligibilityTraceRing::with_actions(d, trace_cap, trace_na);

        let n_reg = config.max_regimes.clamp(1, 64);
        let mut banks = Vec::with_capacity(n_reg);
        for bidx in 0..n_reg {
            banks.push(RegimeBank::new(&config, d, hd, bidx));
        }

        let need_kernels = use_frozen && config.actor_critic_lr <= 0.0 && config.linear_mix < 0.99;
        let need_obs_trace = need_kernels && config.obs_bank_capacity > 0;
        let mut scratch = CoreScratch::new(
            d,
            hd,
            config.bank_capacity.max(config.obs_bank_capacity),
            config.trace_capacity,
            n_act,
            r1_dim,
            r2_dim,
            slow_in,
            skip_dim,
            config.rtrl_max_in,
            use_frozen,
            encoder.num_slots > 0,
            need_kernels,
            need_obs_trace,
        );
        if config.parallel_bank_probe {
            scratch.parallel_bank_hits = vec![0u16; n_reg];
            scratch.parallel_bank_margins = vec![0.0_f32; n_reg];
        }

        Self {
            config,
            encoder,
            belief,
            rtrl_state,
            rtrl_slow_state,
            rtrl_pool_w,
            trace,
            banks,
            active_regime: 0,
            episode_counter: 0,
            probe_eps_left: 0,
            probe_return_sum: 0.0,
            probe_return_cnt: 0,
            parallel_probe_left: 0,
            parallel_probe_prev: 0,
            total_switches: 0,
            rng_state: splitmix64(config.seed ^ 0x9e37_79b9_7f4a_7c15),
            scratch,
            seu_fault_count: 0,
        }
    }

    pub fn total_seu_faults(&self) -> u64 {
        self.seu_fault_count
            .saturating_add(self.rtrl_state.seu_fault_count)
            .saturating_add(self.rtrl_slow_state.seu_fault_count)
    }

    /// Прямая инъекция побитового переворота IEEE-754 в живой регистр состояния
    /// (симметрично по Band 1 `rtrl_state.u` и Band 2 `rtrl_slow_state.u`, либо в `belief.state`).
    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let nf = self.rtrl_state.u.len();
        let ns = self.rtrl_slow_state.u.len();
        let tot = nf.saturating_add(ns);
        if tot > 0 {
            let r = total_rem_usize(reg_idx, tot, 0);
            if r < nf {
                self.rtrl_state.inject_register_bit_flip(r, bit);
            } else {
                self.rtrl_slow_state.inject_register_bit_flip(r.saturating_sub(nf), bit);
            }
        } else if !self.belief.state.is_empty() {
            let idx = total_rem_usize(reg_idx, self.belief.state.len(), 0);
            let cur = total_get_copy(&self.belief.state, idx, 0.0);
            let flipped = f32::from_bits(cur.to_bits() ^ (1u32 << (bit & 31)));
            let _ = total_set(&mut self.belief.state, idx, flipped);
        }
    }

    /// Прямая инъекция побитового переворота IEEE-754 равномерно по ВСЕМ живым обучаемым весам
    /// активного банка (физическая модель SRAM SEU пропорционально размеру каждого тензора).
    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let mask = 1u32 << (bit & 31);
        if let Some(bank) = self.banks.get_mut(self.active_regime) {
            let mut slices: [&mut [f32]; 14] = [
                &mut bank.rtrl_params.alpha,
                &mut bank.rtrl_params.omega,
                &mut bank.rtrl_params.in_w,
                &mut bank.rtrl_slow_params.alpha,
                &mut bank.rtrl_slow_params.omega,
                &mut bank.rtrl_slow_params.in_w,
                &mut bank.rtrl_actor_w,
                &mut bank.rtrl_slow_actor_w,
                &mut bank.rtrl_skip_actor_w,
                &mut bank.linear_w,
                &mut bank.rtrl_critic_w,
                &mut bank.rtrl_slow_critic_w,
                &mut bank.rtrl_skip_critic_w,
                &mut bank.critic_w,
            ];
            let mut total_len = 0usize;
            for sl in slices.iter() {
                total_len = total_len.saturating_add(sl.len());
            }
            if total_len == 0 {
                return;
            }
            let hashed = reg_idx.wrapping_mul(2_654_435_761usize);
            let mut rem = total_rem_usize(hashed, total_len, 0);
            for sl in slices.iter_mut() {
                let len = sl.len();
                if rem < len {
                    let mut cur = total_get_copy(sl, rem, 1.25);
                    if cur.is_finite() && (cur.abs() < 1.001 || cur.abs() >= 2.0) {
                        cur = if cur < 0.0 { -1.25 } else { 1.25 };
                    }
                    let _ = total_set(sl, rem, f32::from_bits(cur.to_bits() ^ mask));
                    return;
                }
                rem = rem.saturating_sub(len);
            }
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        let banks_b: usize = self.banks.iter().map(|b| b.memory_footprint_bytes()).sum();
        self.encoder.memory_footprint_bytes()
            + self.belief.memory_footprint_bytes()
            + self.rtrl_state.memory_footprint_bytes()
            + self.rtrl_slow_state.memory_footprint_bytes()
            + self.rtrl_pool_w.len().saturating_mul(4)
            + self.trace.memory_footprint_bytes()
            + banks_b
            + self.scratch.memory_footprint_bytes()
    }

    pub fn unlabeled_cloze_adapt(&mut self, docs: &[String], epochs: usize) {
        self.encoder.adapt_unlabeled_cloze(docs, epochs);
        self.encoder.freeze_to_i8();
    }

    pub fn start_episode(&mut self) {
        self.belief.reset();
        self.rtrl_state.reset();
        self.rtrl_slow_state.reset();
        self.trace.clear();
        self.scratch.clear_td_traces();
        self.scratch.intra_ep_fatigue.fill(0.0);
        self.scratch.has_prev_obs = false;
    }

    pub fn decide_text(
        &mut self,
        obs_text: &str,
        train_mode: bool,
        seu_nan: bool,
        seu_spikes: bool,
    ) -> Presence<(usize, f32)> {
        if self.config.half_sectors > 0 && !self.scratch.obs_vec.is_empty() {
            self.encoder.encode_text_into(
                obs_text,
                &mut self.scratch.obs_vec,
                &mut self.scratch.ep_slots_unused,
                &mut self.scratch.tmp_sem,
            );
        }
        self.decide_from_encoded_obs(train_mode, true, seu_nan, seu_spikes, 0)
    }

    fn encode_and_advance_bands(
        &mut self,
        features: &[f32],
        train_mode: bool,
        seu_nan: bool,
        seu_spikes: bool,
    ) -> usize {
        let n_in = features.len().min(self.scratch.feat_mean.len());
        let mut aug_len = n_in;

        let mut input_corrupted = false;
        for i in 0..n_in {
            let rv = total_get_copy(features, i, 0.0);
            if !rv.is_finite() || rv.abs() > 1e5 {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                input_corrupted = true;
            }
        }

        if self.config.standardize_vec && n_in > 0 {
            let update_stats = !input_corrupted && (train_mode || self.config.actor_critic_lr <= 0.0);
            if update_stats {
                self.scratch.feat_count = self.scratch.feat_count.saturating_add(1);
                let alpha = if self.scratch.feat_count < 40 {
                    total_div_f32(1.0, self.scratch.feat_count as f32, 0.05).clamp(0.025, 0.5)
                } else {
                    0.008_f32
                };
                let keep = 1.0 - alpha;
                for i in 0..n_in {
                    let raw = sanitize_bounded_f32(total_get_copy(features, i, 0.0), 1e5);
                    let mu = total_get_copy(&self.scratch.feat_mean, i, 0.0);
                    let var = total_get_copy(&self.scratch.feat_var, i, 1.0);
                    let diff = raw - mu;
                    let new_mu = sanitize_f32(mu + alpha * diff);
                    let new_var = sanitize_f32(keep * var + alpha * diff * (raw - new_mu)).max(1e-3);
                    let _ = total_set(&mut self.scratch.feat_mean, i, new_mu);
                    let _ = total_set(&mut self.scratch.feat_var, i, new_var);
                    let std = new_var.sqrt().max(0.05);
                    let z = if self.scratch.feat_count <= 2 {
                        raw.clamp(-4.0, 4.0)
                    } else {
                        total_div_f32(raw - new_mu, std, 0.0).clamp(-4.0, 4.0)
                    };
                    let _ = total_set(&mut self.scratch.vec_aug_buf, i, z);
                }
            } else {
                for i in 0..n_in {
                    let raw = sanitize_bounded_f32(total_get_copy(features, i, 0.0), 1e5);
                    let mu = total_get_copy(&self.scratch.feat_mean, i, 0.0);
                    let var = total_get_copy(&self.scratch.feat_var, i, 1.0);
                    let std = var.sqrt().max(0.05);
                    let z = if self.scratch.feat_count <= 2 {
                        raw.clamp(-4.0, 4.0)
                    } else {
                        total_div_f32(raw - mu, std, 0.0).clamp(-4.0, 4.0)
                    };
                    let _ = total_set(&mut self.scratch.vec_aug_buf, i, z);
                }
            }
        } else {
            for i in 0..n_in {
                let raw = sanitize_bounded_f32(total_get_copy(features, i, 0.0), 1e5);
                let _ = total_set(&mut self.scratch.vec_aug_buf, i, raw);
            }
        }

        if self.config.delta_obs_scale > 0.0 && n_in > 0 {
            let dscale = sanitize_f32(self.config.delta_obs_scale).clamp(0.1, 50.0);
            for i in 0..n_in {
                let cur = total_get_copy(&self.scratch.vec_aug_buf, i, 0.0);
                let prev = if self.scratch.has_prev_obs {
                    total_get_copy(&self.scratch.prev_raw_obs, i, cur)
                } else {
                    cur
                };
                let delta = sanitize_bounded_f32((cur - prev) * dscale, 4.0);
                let _ = total_set(&mut self.scratch.prev_raw_obs, i, cur);
                let _ = total_set(&mut self.scratch.vec_aug_buf, n_in + i, delta);
            }
            self.scratch.has_prev_obs = true;
            aug_len = n_in.saturating_mul(2);
        }

        // Обновляем Полосу 1 и (если включена) Полосу 2 Cayley-RTRL + skip-канал (с защитой от SEU!)
        if self.rtrl_state.sectors > 0 {
            let r_in = aug_len.min(self.rtrl_state.max_in);
            if seu_nan && r_in > 0 {
                let _ = total_set(&mut self.scratch.vec_aug_buf, 0, f32::NAN);
            }
            if seu_spikes && r_in > 1 {
                let _ = total_set(&mut self.scratch.vec_aug_buf, 1, 1e28_f32);
            }
            if !self.scratch.rtrl_skip_buf.is_empty() {
                self.scratch.rtrl_skip_buf.fill(0.0);
                let sk_n = r_in.min(self.scratch.rtrl_skip_buf.len());
                for i in 0..sk_n {
                    let v = sanitize_bounded_f32(total_get_copy(&self.scratch.vec_aug_buf, i, 0.0), 8.0);
                    let _ = total_set(&mut self.scratch.rtrl_skip_buf, i, sanitize_f32(v * 0.5));
                }
            }
            if let Some(bank) = self.banks.get_mut(self.active_regime) {
                self.seu_fault_count = self.seu_fault_count.saturating_add(bank.scrub_weight_seu());
                {
                    let r_slice = self.scratch.vec_aug_buf.get(0..r_in).unwrap_or(&[]);
                    self.rtrl_state.step_forward(&bank.rtrl_params, r_slice);
                }
                if self.rtrl_slow_state.sectors > 0 {
                    let slow_len = self.scratch.rtrl_slow_in_buf.len();
                    let d_raw = self.rtrl_state.max_in.min(slow_len.saturating_sub(4));
                    self.scratch.rtrl_slow_in_buf.fill(0.0);
                    for i in 0..r_in.min(d_raw) {
                        let v = sanitize_bounded_f32(total_get_copy(&self.scratch.vec_aug_buf, i, 0.0), 8.0);
                        let _ = total_set(&mut self.scratch.rtrl_slow_in_buf, i, v);
                    }
                    let r1_dim = self.rtrl_state.b.len();
                    for c in 0..4usize {
                        if d_raw + c < slow_len {
                            let poff = c.saturating_mul(r1_dim);
                            let pw = self.rtrl_pool_w.get(poff..poff + r1_dim).unwrap_or(&[]);
                            let proj = sanitize_bounded_f32(fast_dot(pw, &self.rtrl_state.b), 4.0);
                            let _ = total_set(&mut self.scratch.rtrl_slow_in_buf, d_raw + c, proj);
                        }
                    }
                    self.rtrl_slow_state
                        .step_forward(&bank.rtrl_slow_params, &self.scratch.rtrl_slow_in_buf);
                }
            }
        } else if let Some(bank) = self.banks.get_mut(self.active_regime) {
            self.seu_fault_count = self.seu_fault_count.saturating_add(bank.scrub_weight_seu());
        }

        if self.config.half_sectors > 0 && !self.scratch.obs_vec.is_empty() {
            let slice = self.scratch.vec_aug_buf.get(0..aug_len).unwrap_or(&[]);
            self.encoder.encode_vec_into(slice, &mut self.scratch.obs_vec);
        }
        aug_len
    }

    pub fn decide_vec(
        &mut self,
        features: &[f32],
        train_mode: bool,
        seu_nan: bool,
        seu_spikes: bool,
    ) -> Presence<(usize, f32)> {
        let aug_len = self.encode_and_advance_bands(features, train_mode, seu_nan, seu_spikes);
        self.decide_from_encoded_obs(train_mode, false, seu_nan, seu_spikes, aug_len)
    }

    /// Непрерывный регрессионный шаг поверх Полосы 1 (Fast/Mid Cayley-RTRL) + Полосы 2 (Slow Envelope)
    /// + прямого skip-канала (+ опционально Полосы 0). 100% #[axiom_zero], 0% exp, 0 heap allocs.
    pub fn decide_reg_vec(
        &mut self,
        features: &[f32],
        train_mode: bool,
        seu_nan: bool,
        seu_spikes: bool,
    ) -> Presence<f32> {
        let aug_len = self.encode_and_advance_bands(features, train_mode, seu_nan, seu_spikes);
        self.scratch.pending_active_in = aug_len;
        let use_frozen = self.config.half_sectors > 0 && !self.scratch.obs_vec.is_empty();
        if use_frozen {
            if seu_nan {
                let _ = total_set(&mut self.scratch.obs_vec, 0, f32::NAN);
                let _ = total_set(&mut self.scratch.obs_vec, 3, f32::INFINITY);
            }
            if seu_spikes {
                let _ = total_set(&mut self.scratch.obs_vec, 1, 1e28_f32);
                let _ = total_set(&mut self.scratch.obs_vec, 2, -1e30_f32);
            }
            sanitize_and_normalize_unit(&mut self.scratch.obs_vec, 8.0);
            if self.belief.state.iter().any(|&v| !v.is_finite() || v.abs() > 8.0) {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            let _ = self.belief.update(&self.scratch.obs_vec);
        }
        let Some(bank) = self.banks.get(self.active_regime) else {
            self.scratch.last_reg_pred = 0.0;
            return Presence::boundary(0.0);
        };
        let pred = bank.split_band_logit(
            0,
            &self.belief.state,
            &self.rtrl_state.b,
            &self.rtrl_slow_state.b,
            &self.scratch.rtrl_skip_buf,
        );
        self.scratch.last_reg_pred = pred;
        Presence::new(pred)
    }

    #[inline]
    pub fn last_reg_pred(&self) -> f32 {
        self.scratch.last_reg_pred
    }

    /// Нормированная энергия медленной огибающей Полосы 2 (`||b_slow||_2 / sqrt(S_2)`), read-only.
    #[inline]
    pub fn slow_envelope_norm(&self) -> f32 {
        let s2 = self.rtrl_slow_state.sectors;
        if s2 == 0 || self.rtrl_slow_state.b.is_empty() {
            return 0.0;
        }
        let n2 = fast_dot(&self.rtrl_slow_state.b, &self.rtrl_slow_state.b).max(0.0);
        total_div_f32(n2.sqrt(), (s2 as f32).sqrt(), 0.0)
    }

    fn bank_suspect_drift(&self, reg_id: usize) -> bool {
        self.banks
            .get(reg_id)
            .map(|b| b.ph.cusum > b.ph.threshold * 0.42 && b.ph.mu_bank >= 0.62)
            .unwrap_or(false)
    }

    /// Онлайн пошаговое TD(lambda) обновление голов Actor/Critic и обеих полос Cayley-RTRL.
    fn apply_online_rtrl_td(&mut self, td_err: f32) {
        let reg_id = self.active_regime;
        if self.bank_suspect_drift(reg_id) || self.probe_eps_left > 0 {
            return;
        }

        let raw_d = sanitize_bounded_f32(td_err, 6.0);
        let crit_err = sanitize_bounded_f32(raw_d, 3.0);
        let centered = sanitize_f32(raw_d - self.scratch.td_mean_ema);
        self.scratch.td_mean_ema = sanitize_f32(0.99 * self.scratch.td_mean_ema + 0.01 * raw_d);
        self.scratch.td_var_ema =
            sanitize_f32(0.98 * self.scratch.td_var_ema + 0.02 * centered * centered).max(0.08);
        let adv_err = total_div_f32(centered, self.scratch.td_var_ema.sqrt(), 0.0).clamp(-2.5, 2.5);

        let a_lr = self.config.actor_critic_lr;
        let c_lr = if self.config.critic_lr > 0.0 {
            self.config.critic_lr.max(0.04)
        } else {
            sanitize_f32(a_lr * 3.5).max(0.04)
        };
        let r_lr = self.config.rtrl_lr;
        let wd = self.config.linear_wd;
        let keep = (1.0 - sanitize_f32(wd).clamp(0.0, 0.05)).clamp(0.95, 1.0);
        let n_act = self.config.num_actions.clamp(2, 64);
        let d = if self.config.half_sectors > 0 { self.encoder.dim } else { 0 };
        let r1_dim = self.rtrl_state.sectors.saturating_mul(3);
        let r2_dim = self.rtrl_slow_state.sectors.saturating_mul(3);
        let skip_dim = self.scratch.rtrl_skip_buf.len();

        if let Some(bank) = self.banks.get_mut(reg_id) {
            // 1. Critic (все активные полосы) — по нецентрированному crit_err
            update_normed_vec(&mut bank.critic_w, &self.scratch.td_e_critic_f, c_lr * crit_err, 1.0, 16.0);
            update_normed_vec(&mut bank.rtrl_critic_w, &self.scratch.td_e_critic_r, c_lr * crit_err, 1.0, 16.0);
            update_normed_vec(&mut bank.rtrl_slow_critic_w, &self.scratch.td_e_critic_r2, c_lr * crit_err, 1.0, 16.0);
            update_normed_vec(&mut bank.rtrl_skip_critic_w, &self.scratch.td_e_critic_skip, c_lr * crit_err, 1.0, 16.0);
            bank.critic_bias = sanitize_bounded_f32(
                bank.critic_bias + 0.25 * c_lr * crit_err * self.scratch.td_e_critic_b,
                25.0,
            );

            // 2. Actor (все активные полосы) — по центрированному adv_err
            let actor_step = sanitize_f32(a_lr * adv_err);
            update_actor_band(&mut bank.linear_w, &self.scratch.td_e_actor_f, n_act, d, actor_step, keep);
            update_actor_band(&mut bank.rtrl_actor_w, &self.scratch.td_e_actor_r, n_act, r1_dim, actor_step, keep);
            update_actor_band(&mut bank.rtrl_slow_actor_w, &self.scratch.td_e_actor_r2, n_act, r2_dim, actor_step, keep);
            update_actor_band(&mut bank.rtrl_skip_actor_w, &self.scratch.td_e_actor_skip, n_act, skip_dim, actor_step, keep);

            for a in 0..n_act {
                let eb = total_get_copy(&self.scratch.td_e_actor_b, a, 0.0);
                let ob = total_get_copy(&bank.linear_bias, a, 0.0);
                let _ = total_set(
                    &mut bank.linear_bias,
                    a,
                    sanitize_bounded_f32(ob * keep + 0.25 * actor_step * eb, 3.0),
                );
            }

            // 3. Точный онлайн-RTRL по внутренним параметрам Band 1 и Band 2
            if r_lr > 0.0 && r1_dim > 0 {
                self.rtrl_state.apply_td_update(
                    &mut bank.rtrl_params,
                    adv_err,
                    r_lr,
                    self.scratch.pending_active_in,
                );
                if r2_dim > 0 {
                    self.rtrl_slow_state.apply_td_update(
                        &mut bank.rtrl_slow_params,
                        adv_err,
                        r_lr,
                        self.rtrl_slow_state.max_in,
                    );
                }
            }
        }
    }

    /// Безусловная онлайн-регуляризация полиномиальной энтропии Цаллиса H_2(p) для Ветви А (0% ln).
    fn apply_online_entropy_step(&mut self) {
        let ent_beta = self.config.entropy_beta;
        let a_lr = self.config.actor_critic_lr;
        let reg_id = self.active_regime;
        if ent_beta <= 0.0 || a_lr <= 0.0 || self.bank_suspect_drift(reg_id) || self.probe_eps_left > 0 {
            return;
        }
        let n_act = self.config.num_actions.clamp(2, 64);
        let d = if self.config.half_sectors > 0 { self.encoder.dim } else { 0 };
        let r1_dim = self.rtrl_state.sectors.saturating_mul(3);
        let r2_dim = self.rtrl_slow_state.sectors.saturating_mul(3);
        let skip_dim = self.scratch.rtrl_skip_buf.len();
        let sc1 = total_div_f32(1.0, (self.rtrl_state.sectors.max(1) as f32).sqrt(), 0.25);
        let sc2 = total_div_f32(1.0, (self.rtrl_slow_state.sectors.max(1) as f32).sqrt(), 0.25);

        let mut sum_p2 = 0.0_f32;
        for a in 0..n_act {
            let p = total_get_copy(&self.scratch.ac_probs, a, 0.0).max(0.0);
            sum_p2 = sanitize_f32(sum_p2 + p * p);
        }
        let mut sum_h = 0.0_f32;
        for a in 0..n_act {
            let p = total_get_copy(&self.scratch.ac_probs, a, 0.0).max(0.0);
            let psi = total_get_copy(&self.scratch.ac_psi, a, 1.0);
            sum_h = sanitize_f32(sum_h + p * psi * (sum_p2 - p));
        }
        let mean_h = total_div_f32(sum_h, n_act.max(1) as f32, 0.0);

        if let Some(bank) = self.banks.get_mut(reg_id) {
            for a in 0..n_act {
                let p = total_get_copy(&self.scratch.ac_probs, a, 0.0).max(0.0);
                let psi = total_get_copy(&self.scratch.ac_psi, a, 1.0);
                let d_ent = sanitize_f32(ent_beta * (p * psi * (sum_p2 - p) - mean_h));
                let step = sanitize_f32(a_lr * d_ent);

                if d > 0 {
                    let off = a.saturating_mul(d);
                    if let Some(w) = bank.linear_w.get_mut(off..off + d) {
                        update_normed_vec(w, &self.belief.state, step, 1.0, 6.0);
                    }
                }
                if r1_dim > 0 {
                    let roff = a.saturating_mul(r1_dim);
                    if let Some(rw) = bank.rtrl_actor_w.get_mut(roff..roff + r1_dim) {
                        update_normed_vec(rw, &self.rtrl_state.b, sanitize_f32(step * sc1), 1.0, 6.0);
                    }
                }
                if r2_dim > 0 {
                    let soff = a.saturating_mul(r2_dim);
                    if let Some(sw) = bank.rtrl_slow_actor_w.get_mut(soff..soff + r2_dim) {
                        update_normed_vec(sw, &self.rtrl_slow_state.b, sanitize_f32(step * sc2), 1.0, 6.0);
                    }
                }
                if skip_dim > 0 {
                    let xoff = a.saturating_mul(skip_dim);
                    if let Some(xw) = bank.rtrl_skip_actor_w.get_mut(xoff..xoff + skip_dim) {
                        update_normed_vec(xw, &self.scratch.rtrl_skip_buf, step, 1.0, 6.0);
                    }
                }
                let ob = total_get_copy(&bank.linear_bias, a, 0.0);
                let _ = total_set(
                    &mut bank.linear_bias,
                    a,
                    sanitize_bounded_f32(ob + step * 0.25, 3.0),
                );
            }
        }
    }

    fn decide_from_encoded_obs(
        &mut self,
        train_mode: bool,
        use_sem_fallback: bool,
        seu_nan: bool,
        seu_spikes: bool,
        active_in: usize,
    ) -> Presence<(usize, f32)> {
        self.scratch.pending_active_in = active_in;
        let use_frozen = self.config.half_sectors > 0 && !self.scratch.obs_vec.is_empty();
        let novelty = if use_frozen {
            if seu_nan {
                let _ = total_set(&mut self.scratch.obs_vec, 0, f32::NAN);
                let _ = total_set(&mut self.scratch.obs_vec, 3, f32::INFINITY);
            }
            if seu_spikes {
                let _ = total_set(&mut self.scratch.obs_vec, 1, 1e28_f32);
                let _ = total_set(&mut self.scratch.obs_vec, 2, -1e30_f32);
            }
            sanitize_and_normalize_unit(&mut self.scratch.obs_vec, 8.0);
            if self.belief.state.iter().any(|&v| !v.is_finite() || v.abs() > 8.0) {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.belief.update(&self.scratch.obs_vec)
        } else {
            1.0_f32
        };

        let reg_id = self.active_regime;
        let n_act = self.config.num_actions.clamp(2, 64);
        let in_probe = self.probe_eps_left > 0 || self.parallel_probe_left > 0;
        let r1_dim = self.rtrl_state.sectors.saturating_mul(3);
        let r2_dim = self.rtrl_slow_state.sectors.saturating_mul(3);
        let skip_dim = self.scratch.rtrl_skip_buf.len();

        // Ветвь А: Единый онлайн TD(lambda) + 1L/2L Cayley-RTRL внутри PageHinkley-банков
        if self.config.actor_critic_lr > 0.0 && r1_dim > 0 {
            let v_cur = self
                .banks
                .get(reg_id)
                .map(|b| {
                    b.critic_value(
                        &self.belief.state,
                        &self.rtrl_state.b,
                        &self.rtrl_slow_state.b,
                        &self.scratch.rtrl_skip_buf,
                    )
                })
                .unwrap_or(0.0);

            if train_mode && !in_probe && self.scratch.has_pending_td {
                let gamma = self.config.trace_gamma.clamp(0.0, 0.99);
                let td_err = sanitize_f32(self.scratch.pending_r + gamma * v_cur - self.scratch.pending_v);
                self.apply_online_rtrl_td(td_err);
            }

            if let Some(bank) = self.banks.get(reg_id) {
                for a in 0..n_act {
                    let lg = bank.split_band_logit(
                        a,
                        &self.belief.state,
                        &self.rtrl_state.b,
                        &self.rtrl_slow_state.b,
                        &self.scratch.rtrl_skip_buf,
                    );
                    let _ = total_set(&mut self.scratch.ac_logits, a, lg);
                }
            }
            rational_probs_and_psi_into(
                self.scratch.ac_logits.get(0..n_act).unwrap_or(&[]),
                self.scratch.ac_probs.get_mut(0..n_act).unwrap_or(&mut []),
                self.scratch.ac_psi.get_mut(0..n_act).unwrap_or(&mut []),
            );

            let best_a = if train_mode && !in_probe {
                let u = rand_unit_f32(&mut self.rng_state);
                let mut cdf = 0.0_f32;
                let mut picked = n_act - 1;
                for a in 0..n_act {
                    cdf = sanitize_f32(cdf + total_get_copy(&self.scratch.ac_probs, a, 0.0));
                    if u <= cdf {
                        picked = a;
                        break;
                    }
                }
                picked
            } else {
                let mut ba = 0usize;
                let mut bp = -1.0_f32;
                for a in 0..n_act {
                    let p = total_get_copy(&self.scratch.ac_probs, a, 0.0);
                    if p > bp {
                        bp = p;
                        ba = a;
                    }
                }
                ba
            };

            if train_mode && !in_probe {
                let decay = sanitize_f32(self.config.trace_gamma.clamp(0.0, 0.99) * 0.80);
                let d = if use_frozen { self.encoder.dim } else { 0 };
                let sc1 = total_div_f32(1.0, (self.rtrl_state.sectors.max(1) as f32).sqrt(), 0.25);
                let sc2 = total_div_f32(1.0, (self.rtrl_slow_state.sectors.max(1) as f32).sqrt(), 0.25);

                if d > 0 {
                    accum_trace_vec(&mut self.scratch.td_e_critic_f, &self.belief.state, decay, 1.0);
                }
                accum_trace_vec(&mut self.scratch.td_e_critic_r, &self.rtrl_state.b, decay, sc1);
                accum_trace_vec(&mut self.scratch.td_e_critic_r2, &self.rtrl_slow_state.b, decay, sc2);
                accum_trace_vec(&mut self.scratch.td_e_critic_skip, &self.scratch.rtrl_skip_buf, decay, 1.0);
                self.scratch.td_e_critic_b = sanitize_bounded_f32(decay * self.scratch.td_e_critic_b + 1.0, 5.0);

                let mut sum_s = 0.0_f32;
                for a in 0..n_act {
                    let p = total_get_copy(&self.scratch.ac_probs, a, 0.0).max(0.0);
                    let psi = total_get_copy(&self.scratch.ac_psi, a, 1.0);
                    let ind = if a == best_a { 1.0_f32 } else { 0.0_f32 };
                    sum_s = sanitize_f32(sum_s + psi * (ind - p));
                }
                let mean_s = total_div_f32(sum_s, n_act.max(1) as f32, 0.0);

                self.scratch.rtrl_fb_buf.fill(0.0);
                self.scratch.rtrl_slow_fb_buf.fill(0.0);
                for a in 0..n_act {
                    let p = total_get_copy(&self.scratch.ac_probs, a, 0.0).max(0.0);
                    let psi = total_get_copy(&self.scratch.ac_psi, a, 1.0);
                    let ind = if a == best_a { 1.0_f32 } else { 0.0_f32 };
                    let score_a = sanitize_f32(psi * (ind - p) - mean_s);

                    if d > 0 {
                        let off = a.saturating_mul(d);
                        if let Some(ef) = self.scratch.td_e_actor_f.get_mut(off..off + d) {
                            accum_trace_vec(ef, &self.belief.state, decay, score_a);
                        }
                    }
                    let roff = a.saturating_mul(r1_dim);
                    if let Some(er) = self.scratch.td_e_actor_r.get_mut(roff..roff + r1_dim) {
                        accum_trace_vec(er, &self.rtrl_state.b, decay, score_a * sc1);
                    }
                    if r2_dim > 0 {
                        let soff = a.saturating_mul(r2_dim);
                        if let Some(er2) = self.scratch.td_e_actor_r2.get_mut(soff..soff + r2_dim) {
                            accum_trace_vec(er2, &self.rtrl_slow_state.b, decay, score_a * sc2);
                        }
                    }
                    if skip_dim > 0 {
                        let xoff = a.saturating_mul(skip_dim);
                        if let Some(ex) = self.scratch.td_e_actor_skip.get_mut(xoff..xoff + skip_dim) {
                            accum_trace_vec(ex, &self.scratch.rtrl_skip_buf, decay, score_a);
                        }
                    }
                    let ob = total_get_copy(&self.scratch.td_e_actor_b, a, 0.0);
                    let _ = total_set(
                        &mut self.scratch.td_e_actor_b,
                        a,
                        sanitize_bounded_f32(decay * ob + score_a, 5.0),
                    );

                    if let Some(bank) = self.banks.get(reg_id) {
                        if let Some(rw) = bank.rtrl_actor_w.get(roff..roff + r1_dim) {
                            fast_saxpy(&mut self.scratch.rtrl_fb_buf, rw, score_a * sc1);
                        }
                        if r2_dim > 0 {
                            let soff = a.saturating_mul(r2_dim);
                            if let Some(sw) = bank.rtrl_slow_actor_w.get(soff..soff + r2_dim) {
                                fast_saxpy(&mut self.scratch.rtrl_slow_fb_buf, sw, score_a * sc2);
                            }
                        }
                    }
                }
                if let Some(bank) = self.banks.get(reg_id) {
                    fast_saxpy(&mut self.scratch.rtrl_fb_buf, &bank.rtrl_critic_w, 0.15 * sc1);
                    if r2_dim > 0 {
                        fast_saxpy(&mut self.scratch.rtrl_slow_fb_buf, &bank.rtrl_slow_critic_w, 0.15 * sc2);
                    }
                }
                self.rtrl_state
                    .accumulate_traces(&self.scratch.rtrl_fb_buf, decay, active_in);
                if r2_dim > 0 {
                    self.rtrl_slow_state.accumulate_traces(
                        &self.scratch.rtrl_slow_fb_buf,
                        decay,
                        self.rtrl_slow_state.max_in,
                    );
                }

                self.apply_online_entropy_step();

                let v_for_pending = self
                    .banks
                    .get(reg_id)
                    .map(|b| {
                        b.critic_value(
                            &self.belief.state,
                            &self.rtrl_state.b,
                            &self.rtrl_slow_state.b,
                            &self.scratch.rtrl_skip_buf,
                        )
                    })
                    .unwrap_or(v_cur);
                self.scratch.pending_v = v_for_pending;
                self.scratch.pending_r = 0.0;
                self.scratch.pending_active_in = active_in;
                self.scratch.has_pending_td = true;
            }

            let p_sel = total_get_copy(&self.scratch.ac_probs, best_a, 0.5);
            return Presence::new((best_a, p_sel));
        }

        let Some(bank) = self.banks.get(reg_id) else {
            return Presence::boundary((0, 0.0));
        };

        // Ветвь Б: Эпизодический 3-факторный Actor-Critic (без RTRL)
        if self.config.actor_critic_lr > 0.0 {
            let v_pred = bank.critic_value(&self.belief.state, &[], &[], &[]);
            for a in 0..n_act {
                let lg = bank.linear_score(a, &self.belief.state);
                let _ = total_set(&mut self.scratch.ac_logits, a, lg);
            }
            rational_probs_and_psi_into(
                self.scratch.ac_logits.get(0..n_act).unwrap_or(&[]),
                self.scratch.ac_probs.get_mut(0..n_act).unwrap_or(&mut []),
                self.scratch.ac_psi.get_mut(0..n_act).unwrap_or(&mut []),
            );

            let best_a = if train_mode && !in_probe {
                let u = rand_unit_f32(&mut self.rng_state);
                let mut cdf = 0.0_f32;
                let mut picked = n_act - 1;
                for a in 0..n_act {
                    cdf = sanitize_f32(cdf + total_get_copy(&self.scratch.ac_probs, a, 0.0));
                    if u <= cdf {
                        picked = a;
                        break;
                    }
                }
                picked
            } else {
                let mut ba = 0usize;
                let mut bp = -1.0_f32;
                for a in 0..n_act {
                    let p = total_get_copy(&self.scratch.ac_probs, a, 0.0);
                    if p > bp {
                        bp = p;
                        ba = a;
                    }
                }
                ba
            };

            let p_sel = total_get_copy(&self.scratch.ac_probs, best_a, 0.5);
            self.trace.push_full(
                &self.belief.state,
                best_a,
                novelty,
                v_pred,
                self.scratch.ac_probs.get(0..n_act).unwrap_or(&[]),
                self.scratch.ac_psi.get(0..n_act).unwrap_or(&[]),
            );
            return Presence::new((best_a, p_sel));
        }

        // Ветвь В: KerrKernelBank + Delta-Rule (+ опциональный RTRL для потоков)
        let use_lin = self.config.linear_lr > 0.0 && self.config.linear_mix > 0.0;
        let lin_mix = if use_lin {
            sanitize_f32(self.config.linear_mix).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let need_kernels = use_frozen && lin_mix < 0.99;

        let sim_obs = if need_kernels && bank.obs_kernel.active_count > 0 {
            bank.obs_kernel.query_obs_gated_into(
                &self.scratch.obs_vec,
                4,
                0.35,
                use_sem_fallback,
                &mut self.scratch.sims,
                &mut self.scratch.weights,
                &mut self.scratch.q_obs,
                &mut self.scratch.v_obs,
            )
        } else {
            self.scratch.q_obs.fill(0.0);
            self.scratch.v_obs.fill(0.0);
            0.0
        };

        let sim_bel = if need_kernels && bank.belief_kernel.active_count > 0 {
            bank.belief_kernel.query_into(
                &self.belief.state,
                4,
                self.config.belief_tau,
                use_sem_fallback,
                &mut self.scratch.sims,
                &mut self.scratch.weights,
                &mut self.scratch.q_bel,
                &mut self.scratch.v_bel,
            )
        } else {
            self.scratch.q_bel.fill(0.0);
            self.scratch.v_bel.fill(0.0);
            0.0
        };

        let mut max_q_cur = -1e9_f32;
        for a in 0..n_act {
            let qo = total_get_copy(&self.scratch.q_obs, a, 0.0);
            let qb = total_get_copy(&self.scratch.q_bel, a, 0.0);
            let vb = total_get_copy(&self.scratch.v_bel, a, 0.0);
            let q_k = if vb >= 0.7 {
                sanitize_f32(0.85 * qb + 0.15 * qo)
            } else {
                sanitize_f32(qb + qo)
            };
            let q_l = if use_lin {
                bank.split_band_logit(
                    a,
                    &self.belief.state,
                    &self.rtrl_state.b,
                    &self.rtrl_slow_state.b,
                    &self.scratch.rtrl_skip_buf,
                )
            } else {
                0.0
            };
            let q_eff = sanitize_f32((1.0 - lin_mix) * q_k + lin_mix * q_l);
            if q_eff > max_q_cur {
                max_q_cur = q_eff;
            }
        }
        if !use_lin && sim_obs.max(sim_bel) < 0.35 || max_q_cur <= -1e8 {
            max_q_cur = 0.0;
        }
        self.trace.set_prev_next_value(max_q_cur);

        let fat_decay = if novelty > 0.55 { 0.15_f32 } else { 0.65_f32 };
        for f in self.scratch.intra_ep_fatigue.iter_mut() {
            *f = sanitize_f32(*f * fat_decay);
        }

        let ucb_c = if train_mode && !in_probe { self.config.explore_ucb } else { 0.0 };

        let mut best_a = 0usize;
        let mut best_s = -1e9_f32;
        let mut second_s = -1e9_f32;

        for a in 0..n_act {
            let qo = total_get_copy(&self.scratch.q_obs, a, 0.0);
            let qb = total_get_copy(&self.scratch.q_bel, a, 0.0);
            let vo = total_get_copy(&self.scratch.v_obs, a, 0.0).max(0.0);
            let vb = total_get_copy(&self.scratch.v_bel, a, 0.0).max(0.0);

            let q_k = if vb >= 0.7 {
                sanitize_f32(0.85 * qb + 0.15 * qo)
            } else if qo < -0.10 && vo >= 2.0 {
                qo
            } else if qo > 0.12 && vo >= 1.0 {
                qo
            } else {
                qb
            };
            let q_l = if use_lin {
                bank.split_band_logit(
                    a,
                    &self.belief.state,
                    &self.rtrl_state.b,
                    &self.rtrl_slow_state.b,
                    &self.scratch.rtrl_skip_buf,
                )
            } else {
                0.0
            };
            let q_est = sanitize_f32((1.0 - lin_mix) * q_k + lin_mix * q_l);

            let ucb_scale = if qo < -0.10 && vo >= 2.0 && vb < 0.7 { 0.10_f32 } else { 1.0_f32 };
            let bonus = if ucb_c > 0.0 {
                total_div_f32(ucb_c * ucb_scale, (1.0 + vb).sqrt(), 0.0)
            } else {
                0.0
            };

            let fat = if train_mode && ucb_c > 0.0 {
                total_get_copy(&self.scratch.intra_ep_fatigue, a, 0.0)
            } else {
                0.0
            };
            let jitter = if train_mode && !in_probe {
                rand_signed_f32(&mut self.rng_state) * (if use_lin { 0.08 } else { 0.03 })
            } else {
                0.0
            };

            let sc = sanitize_f32(q_est + bonus - 0.16 * fat + jitter);
            if sc > best_s {
                second_s = best_s;
                best_s = sc;
                best_a = a;
            } else if sc > second_s {
                second_s = sc;
            }
        }

        let prev_fat = total_get_copy(&self.scratch.intra_ep_fatigue, best_a, 0.0);
        let _ = total_set(&mut self.scratch.intra_ep_fatigue, best_a, (prev_fat + 0.55).min(2.0));

        if need_kernels && !self.scratch.obs_trace.is_empty() {
            let step_slot = if self.trace.len < self.trace.capacity {
                self.trace.len
            } else {
                let d = self.encoder.dim;
                for k in 1..self.trace.capacity {
                    let dst_off = (k - 1).saturating_mul(d);
                    let src_off = k.saturating_mul(d);
                    for j in 0..d {
                        let v = total_get_copy(&self.scratch.obs_trace, src_off + j, 0.0);
                        let _ = total_set(&mut self.scratch.obs_trace, dst_off + j, v);
                    }
                }
                self.trace.capacity - 1
            };
            let d = self.encoder.dim;
            let off = step_slot.saturating_mul(d);
            if let Some(dst) = self.scratch.obs_trace.get_mut(off..off + d) {
                for (dc, &oc) in dst.iter_mut().zip(self.scratch.obs_vec.iter()) {
                    *dc = oc;
                }
            }
        }

        self.trace.push(&self.belief.state, best_a, novelty);

        if second_s <= -1e8 {
            second_s = 0.0;
        }
        let margin = sanitize_f32((best_s - second_s).max(0.0));
        if margin < 0.03 {
            Presence::boundary((best_a, margin))
        } else {
            Presence::new((best_a, margin))
        }
    }

    #[inline]
    pub fn observe_step_reward(&mut self, step_reward: f32) {
        let r = sanitize_f32(step_reward);
        if self.rtrl_state.sectors > 0 && self.config.actor_critic_lr > 0.0 {
            self.scratch.pending_r = r;
        } else {
            self.trace.set_last_reward(r);
        }
    }

    pub fn observe_step_bandit_reward(&mut self, action: usize, reward: f32) {
        let reg_id = self.active_regime;
        let lr = self.config.linear_lr;
        let wd = self.config.linear_wd;
        if lr <= 0.0 {
            return;
        }
        if let Some(bank) = self.banks.get_mut(reg_id) {
            bank.linear_delta_update(action, &self.belief.state, reward, lr, wd);
        }
    }

    pub fn end_episode(&mut self, final_return: f32, train_mode: bool) -> bool {
        self.end_episode_inner(final_return, false, false, false, train_mode)
    }

    pub fn end_episode_with_solved(&mut self, final_return: f32, solved: bool, train_mode: bool) -> bool {
        self.end_episode_inner(final_return, true, solved, false, train_mode)
    }

    pub fn end_episode_truncated(&mut self, final_return: f32, truncated: bool, train_mode: bool) -> bool {
        self.end_episode_inner(final_return, false, false, truncated, train_mode)
    }

    fn end_episode_inner(
        &mut self,
        final_return: f32,
        use_solved_override: bool,
        solved_flag: bool,
        truncated: bool,
        train_mode: bool,
    ) -> bool {
        if !train_mode {
            return false;
        }
        self.episode_counter = self.episode_counter.saturating_add(1);
        let ret = sanitize_bounded_f32(final_return, 10.0);
        let ac_lr = self.config.actor_critic_lr;
        let r_norm = if use_solved_override {
            if solved_flag { 1.0_f32 } else { 0.0_f32 }
        } else if ac_lr > 0.0 {
            sanitize_f32((ret + 1.0) * 0.5).clamp(0.0, 1.0)
        } else if ret >= 0.75 {
            1.0_f32
        } else {
            sanitize_f32((ret + 1.0) * 0.25).clamp(0.0, 0.35)
        };
        let reg_id = self.active_regime;

        if self.probe_eps_left > 0 {
            self.probe_return_sum = sanitize_f32(self.probe_return_sum + r_norm);
            self.probe_return_cnt = self.probe_return_cnt.saturating_add(1);
            self.probe_eps_left -= 1;

            if self.probe_eps_left == 0 {
                let avg = total_div_f32(
                    self.probe_return_sum,
                    self.probe_return_cnt.max(1) as f32,
                    0.0,
                );
                let accept_th = if ac_lr > 0.0 { 0.38_f32 } else { 0.45_f32 };
                if avg >= accept_th {
                    if let Some(b) = self.banks.get_mut(reg_id) {
                        b.ph.reset_cusum();
                    }
                    return false;
                } else {
                    if let Some(b) = self.banks.get_mut(reg_id) {
                        b.cooldown_until = self.episode_counter.saturating_add(48);
                    }
                    return self.switch_to_next_or_fresh_bank();
                }
            }
            return false;
        }

        // Ветвь А: закрываем последний терминальный шаг онлайн TD(lambda) + 1L/2L Cayley-RTRL
        if ac_lr > 0.0 && self.rtrl_state.sectors > 0 {
            if self.scratch.has_pending_td {
                let gamma = self.config.trace_gamma.clamp(0.0, 0.99);
                let v_next = if truncated {
                    self.scratch.pending_v
                } else {
                    0.0_f32
                };
                let td_err = sanitize_f32(self.scratch.pending_r + gamma * v_next - self.scratch.pending_v);
                self.apply_online_rtrl_td(td_err);
                self.scratch.has_pending_td = false;
            }
            if let Some(bank) = self.banks.get_mut(reg_id) {
                bank.age_eps = bank.age_eps.saturating_add(1);
                if r_norm >= 0.5 {
                    bank.solved_eps = bank.solved_eps.saturating_add(1);
                }
            }
            let drift_detected = if let Some(bank) = self.banks.get_mut(reg_id) {
                bank.ph.observe(r_norm)
            } else {
                false
            };
            if drift_detected {
                if let Some(bank) = self.banks.get_mut(reg_id) {
                    bank.ph.reset_cusum();
                    bank.cooldown_until = self.episode_counter.saturating_add(48);
                }
                return self.switch_to_next_or_fresh_bank();
            }
            return false;
        }

        let t_len = self.trace.len.min(256);
        let gamma = self.config.trace_gamma.clamp(0.5, 0.99);

        // Ветвь Б: Эпизодический 3-факторный Actor-Critic (GAE + RMS)
        if ac_lr > 0.0 && t_len > 0 {
            let bank_suspect_drift = self
                .banks
                .get(reg_id)
                .map(|b| b.ph.cusum > b.ph.threshold * 0.35 && b.ph.mu_bank >= 0.50)
                .unwrap_or(false);

            if !bank_suspect_drift {
                let mut gae = 0.0_f32;
                let gae_lam = 0.90_f32;
                let mut next_v = 0.0_f32;
                let mut advs = [0.0_f32; 256];
                let mut v_tgts = [0.0_f32; 256];
                let mut t = t_len;
                while t > 0 {
                    t -= 1;
                    let r_t = total_get_copy(&self.trace.rewards, t, 0.0);
                    let v_t = total_get_copy(&self.trace.values, t, 0.0);
                    let delta = sanitize_f32(r_t + gamma * next_v - v_t);
                    gae = sanitize_f32(delta + gamma * gae_lam * gae);
                    let _ = total_set(&mut advs, t, gae);
                    let _ = total_set(&mut v_tgts, t, sanitize_f32(v_t + gae));
                    next_v = v_t;
                }

                if t_len >= 4 {
                    let mut sq_a = 0.0_f32;
                    for s in 0..t_len {
                        let a = total_get_copy(&advs, s, 0.0);
                        sq_a = sanitize_f32(sq_a + a * a);
                    }
                    let rms_a = total_div_f32(sq_a, t_len as f32, 1.0).sqrt().max(0.25);
                    for s in 0..t_len {
                        let a = total_get_copy(&advs, s, 0.0);
                        let _ = total_set(&mut advs, s, total_div_f32(a, rms_a, 0.0).clamp(-2.5, 2.5));
                    }
                }

                let c_lr = self.config.critic_lr.max(0.05);
                let ent_b = self.config.entropy_beta.max(0.0);
                let wd = self.config.linear_wd;
                let step_scale = total_div_f32(1.0, 1.0 + 0.004 * (t_len as f32), 0.5);
                if let Some(bank) = self.banks.get_mut(reg_id) {
                    bank.age_eps = bank.age_eps.saturating_add(1);
                    if r_norm >= 0.5 {
                        bank.solved_eps = bank.solved_eps.saturating_add(1);
                    }
                    for s in 0..t_len {
                        let act = total_get_copy(&self.trace.actions, s, 0);
                        let adv = total_get_copy(&advs, s, 0.0);
                        let vt = total_get_copy(&v_tgts, s, 0.0);
                        let b_vec = self.trace.belief_at(s);
                        let probs = self.trace.probs_at(s);
                        let psis = self.trace.psi_at(s);
                        bank.critic_update(b_vec, vt, c_lr * step_scale);
                        bank.actor_ac_update(act, b_vec, probs, psis, adv, ac_lr * step_scale, ent_b, wd);
                    }
                }
            }

            let drift_detected = if let Some(bank) = self.banks.get_mut(reg_id) {
                bank.ph.observe(r_norm)
            } else {
                false
            };
            if drift_detected {
                if let Some(bank) = self.banks.get_mut(reg_id) {
                    bank.ph.reset_cusum();
                    bank.cooldown_until = self.episode_counter.saturating_add(48);
                }
                return self.switch_to_next_or_fresh_bank();
            }
            return false;
        }

        // Ветвь В: KerrKernelBank + Eligibility Traces (+ Delta-Rule)
        let mut has_intermediate_r = false;
        if t_len > 1 {
            for t in 0..(t_len - 1) {
                if total_get_copy(&self.trace.rewards, t, 0.0).abs() > 1e-6 {
                    has_intermediate_r = true;
                    break;
                }
            }
        }

        let mut targets = [0.0_f32; 256];
        if t_len > 0 {
            if has_intermediate_r {
                let mut g_next = 0.0_f32;
                let mut t = t_len;
                while t > 0 {
                    t -= 1;
                    let r_t = total_get_copy(&self.trace.rewards, t, 0.0);
                    let v_next = if t + 1 < t_len {
                        total_get_copy(&self.trace.next_values, t, 0.0)
                    } else {
                        0.0
                    };
                    let td_1 = sanitize_f32(r_t + gamma * v_next);
                    let mc_t = sanitize_f32(r_t + gamma * g_next);
                    let blended = if r_t >= 0.25 {
                        sanitize_f32(r_t + gamma * v_next.max(0.0).max(g_next.max(0.0)))
                    } else {
                        sanitize_f32(0.55 * td_1 + 0.45 * mc_t)
                    };
                    let _ = total_set(&mut targets, t, blended);
                    g_next = blended;
                }
            } else {
                let cur_w = self.config.intrinsic_curiosity.clamp(0.0, 0.4);
                let mut running = ret;
                let mut t = t_len;
                while t > 0 {
                    t -= 1;
                    let is_last = t + 1 == t_len;
                    let v_next = if !is_last {
                        total_get_copy(&self.trace.next_values, t, 0.0)
                    } else {
                        0.0
                    };
                    let r_int = if !is_last && cur_w > 0.0 {
                        let next_nov = total_get_copy(&self.trace.novelties, t + 1, 0.5);
                        sanitize_f32(cur_w * (next_nov - 0.50))
                    } else {
                        0.0
                    };
                    let r_step = if is_last { ret } else { r_int };
                    let mc_val = if is_last { ret } else { sanitize_f32(r_int + gamma * running) };
                    let td_val = sanitize_f32(r_step + gamma * v_next);
                    let tgt = if ret > 0.0 {
                        mc_val
                    } else {
                        sanitize_f32(0.55 * td_val + 0.45 * mc_val)
                    };
                    let _ = total_set(&mut targets, t, tgt);
                    running = tgt;
                }
            }
        }

        let bank_suspect_drift = self
            .banks
            .get(reg_id)
            .map(|b| b.ph.cusum > b.ph.threshold * 0.55 && b.ph.mu_bank >= 0.55)
            .unwrap_or(false);

        let lin_lr = self.config.linear_lr;
        let lin_wd = self.config.linear_wd;
        let need_kernels = self.config.half_sectors > 0 && self.config.linear_mix < 0.99;

        if !bank_suspect_drift {
            let d = self.encoder.dim;
            if let Some(bank) = self.banks.get_mut(reg_id) {
                bank.age_eps = bank.age_eps.saturating_add(1);
                if ret > 0.25 {
                    bank.solved_eps = bank.solved_eps.saturating_add(1);
                }
                let mut trace_w = 1.0_f32;
                let mut t = t_len;
                let rpe = sanitize_bounded_f32(ret - bank.ret_ema, 1.5);
                bank.ret_ema = sanitize_f32(bank.ret_ema + 0.08 * (ret - bank.ret_ema));
                while t > 0 {
                    t -= 1;
                    let tgt = total_get_copy(&targets, t, 0.0);
                    let act = total_get_copy(&self.trace.actions, t, 0);
                    let b_vec = self.trace.belief_at(t);

                    if lin_lr > 0.0 {
                        let lin_tgt = if has_intermediate_r {
                            sanitize_bounded_f32(0.35 * tgt + 0.85 * rpe, 2.0)
                        } else {
                            tgt
                        };
                        bank.linear_delta_update(
                            act,
                            b_vec,
                            lin_tgt,
                            sanitize_f32(lin_lr * trace_w),
                            lin_wd,
                        );
                    }

                    if need_kernels {
                        if !self.scratch.obs_trace.is_empty() && bank.obs_kernel.capacity > 0 {
                            let o_off = t.saturating_mul(d);
                            let o_vec = self.scratch.obs_trace.get(o_off..o_off + d).unwrap_or(&[]);
                            bank.obs_kernel.update_action_value(
                                o_vec,
                                act,
                                tgt,
                                trace_w * 0.6,
                                &mut self.scratch.tmp_key,
                            );
                        }
                        bank.belief_kernel.update_action_value(
                            b_vec,
                            act,
                            tgt,
                            trace_w,
                            &mut self.scratch.tmp_key,
                        );
                    }

                    trace_w = sanitize_f32(trace_w * 0.94);
                }
            }
        }

        let drift_detected = if let Some(bank) = self.banks.get_mut(reg_id) {
            bank.ph.observe(r_norm)
        } else {
            false
        };

        if drift_detected {
            if let Some(bank) = self.banks.get_mut(reg_id) {
                bank.ph.reset_cusum();
                bank.cooldown_until = self.episode_counter.saturating_add(48);
            }
            return self.switch_to_next_or_fresh_bank();
        }
        false
    }

    pub fn end_supervised_step(&mut self, chosen: usize, true_label: usize) -> bool {
        self.episode_counter = self.episode_counter.saturating_add(1);
        let correct = chosen == true_label;
        let r_norm = if correct { 1.0_f32 } else { 0.0_f32 };
        let reg_id = self.active_regime;

        if self.parallel_probe_left > 0 {
            let n_act = self.config.num_actions.clamp(2, 64);
            let n_banks = self.banks.len();
            for bi in 0..n_banks {
                if bi == self.parallel_probe_prev {
                    continue;
                }
                if let Some(b) = self.banks.get(bi) {
                    if b.age_eps < 20 {
                        continue;
                    }
                    let mut best_a = 0usize;
                    let mut best_l = -1e30_f32;
                    let mut logit_y = 0.0_f32;
                    let mut best_other = -1e30_f32;
                    for a in 0..n_act {
                        let lg = b.split_band_logit(
                            a,
                            &self.belief.state,
                            &self.rtrl_state.b,
                            &self.rtrl_slow_state.b,
                            &self.scratch.rtrl_skip_buf,
                        );
                        if lg > best_l {
                            best_l = lg;
                            best_a = a;
                        }
                        if a == true_label {
                            logit_y = lg;
                        } else if lg > best_other {
                            best_other = lg;
                        }
                    }
                    if best_a == true_label {
                        let old_h = total_get_copy(&self.scratch.parallel_bank_hits, bi, 0);
                        let _ = total_set(
                            &mut self.scratch.parallel_bank_hits,
                            bi,
                            old_h.saturating_add(1),
                        );
                    }
                    let m = sanitize_bounded_f32(logit_y - best_other, 2.0);
                    let old_m = total_get_copy(&self.scratch.parallel_bank_margins, bi, 0.0);
                    let _ = total_set(
                        &mut self.scratch.parallel_bank_margins,
                        bi,
                        sanitize_f32(old_m + m),
                    );
                }
            }
            self.parallel_probe_left -= 1;
            let mut best_b = self.parallel_probe_prev;
            let mut best_h = 0u16;
            let mut best_m = -1e30_f32;
            let mut found = false;
            for bi in 0..n_banks {
                if bi == self.parallel_probe_prev {
                    continue;
                }
                if let Some(b) = self.banks.get(bi) {
                    if b.age_eps < 20 {
                        continue;
                    }
                    let h = total_get_copy(&self.scratch.parallel_bank_hits, bi, 0);
                    let m = total_get_copy(&self.scratch.parallel_bank_margins, bi, -1e30);
                    if !found || h > best_h || (h == best_h && m > best_m) {
                        found = true;
                        best_b = bi;
                        best_h = h;
                        best_m = m;
                    }
                }
            }
            if self.parallel_probe_left > 0 {
                // Строго каузально переключаем прогноз на текущего лидера (без обновления весов до конца окна!)
                if found {
                    self.active_regime = best_b;
                }
                return false;
            }
            if found && best_h >= 4 {
                self.active_regime = best_b;
                self.probe_eps_left = 0;
                if let Some(b) = self.banks.get_mut(best_b) {
                    b.ph.reset_cusum();
                }
                self.total_switches = self.total_switches.saturating_add(1);
                return true;
            }
            self.active_regime = self.parallel_probe_prev;
            return self.allocate_fresh_or_evict_bank();
        }

        if self.probe_eps_left > 0 {
            self.probe_return_sum = sanitize_f32(self.probe_return_sum + r_norm);
            self.probe_return_cnt = self.probe_return_cnt.saturating_add(1);
            self.probe_eps_left -= 1;

            if self.probe_eps_left == 0 {
                let avg = total_div_f32(
                    self.probe_return_sum,
                    self.probe_return_cnt.max(1) as f32,
                    0.0,
                );
                if avg >= 0.60 {
                    if let Some(b) = self.banks.get_mut(reg_id) {
                        b.ph.reset_cusum();
                    }
                    return false;
                } else {
                    if let Some(b) = self.banks.get_mut(reg_id) {
                        b.cooldown_until = self.episode_counter.saturating_add(48);
                    }
                    return self.switch_to_next_or_fresh_bank();
                }
            }
            return false;
        }

        let bank_suspect_drift = self
            .banks
            .get(reg_id)
            .map(|b| b.ph.cusum > b.ph.threshold * 0.55 && b.ph.mu_bank >= 0.60)
            .unwrap_or(false);

        let lin_lr = self.config.linear_lr;
        let lin_wd = self.config.linear_wd;
        let need_kernels = self.config.half_sectors > 0 && self.config.linear_mix < 0.99;
        let n_act = self.config.num_actions.clamp(2, 64);
        let r1_dim = self.rtrl_state.sectors.saturating_mul(3);
        let r2_dim = self.rtrl_slow_state.sectors.saturating_mul(3);
        let skip_dim = self.scratch.rtrl_skip_buf.len();

        if !bank_suspect_drift {
            let r1_scale = total_div_f32(1.0, (self.rtrl_state.sectors.max(1) as f32).sqrt(), 0.25);
            let r2_scale = total_div_f32(1.0, (self.rtrl_slow_state.sectors.max(1) as f32).sqrt(), 0.25);
            let r_lr = self.config.rtrl_lr;
            let keep = (1.0 - sanitize_f32(lin_wd).clamp(0.0, 0.05)).clamp(0.95, 1.0);
            if r1_dim > 0 {
                self.scratch.rtrl_fb_buf.fill(0.0);
                self.scratch.rtrl_slow_fb_buf.fill(0.0);
            }

            if let Some(bank) = self.banks.get_mut(reg_id) {
                bank.age_eps = bank.age_eps.saturating_add(1);
                if correct {
                    bank.solved_eps = bank.solved_eps.saturating_add(1);
                }
                let b_vec = &self.belief.state;
                if lin_lr > 0.0 {
                    if r1_dim == 0 {
                        for a in 0..n_act {
                            let tgt = if a == true_label { 1.0_f32 } else { -1.0_f32 };
                            bank.linear_delta_update(a, b_vec, tgt, lin_lr, lin_wd);
                        }
                    } else {
                        for a in 0..n_act {
                            let pred = bank.split_band_logit(
                                a,
                                b_vec,
                                &self.rtrl_state.b,
                                &self.rtrl_slow_state.b,
                                &self.scratch.rtrl_skip_buf,
                            );
                            let tgt = if a == true_label { 1.0_f32 } else { -1.0_f32 };
                            let err = sanitize_bounded_f32(tgt - pred, 2.5);
                            let step = sanitize_f32(lin_lr * err);

                            if !bank.linear_w.is_empty() {
                                let off = a.saturating_mul(bank.dim);
                                if let Some(w) = bank.linear_w.get_mut(off..off + bank.dim) {
                                    update_normed_vec(w, b_vec, step, keep, 6.0);
                                }
                            }
                            let roff = a.saturating_mul(r1_dim);
                            if let Some(rw) = bank.rtrl_actor_w.get(roff..roff + r1_dim) {
                                fast_saxpy(&mut self.scratch.rtrl_fb_buf, rw, err * r1_scale);
                            }
                            if let Some(rw) = bank.rtrl_actor_w.get_mut(roff..roff + r1_dim) {
                                update_normed_vec(rw, &self.rtrl_state.b, sanitize_f32(step * r1_scale), keep, 6.0);
                            }
                            if r2_dim > 0 {
                                let soff = a.saturating_mul(r2_dim);
                                if let Some(sw) = bank.rtrl_slow_actor_w.get(soff..soff + r2_dim) {
                                    fast_saxpy(&mut self.scratch.rtrl_slow_fb_buf, sw, err * r2_scale);
                                }
                                if let Some(sw) = bank.rtrl_slow_actor_w.get_mut(soff..soff + r2_dim) {
                                    update_normed_vec(sw, &self.rtrl_slow_state.b, sanitize_f32(step * r2_scale), keep, 6.0);
                                }
                            }
                            if skip_dim > 0 {
                                let xoff = a.saturating_mul(skip_dim);
                                if let Some(xw) = bank.rtrl_skip_actor_w.get_mut(xoff..xoff + skip_dim) {
                                    update_normed_vec(xw, &self.scratch.rtrl_skip_buf, step, keep, 6.0);
                                }
                            }
                            let ob = total_get_copy(&bank.linear_bias, a, 0.0);
                            let _ = total_set(
                                &mut bank.linear_bias,
                                a,
                                sanitize_bounded_f32(ob * keep + step * 0.25, 3.0),
                            );
                        }
                        if r_lr > 0.0 {
                            let sup_decay = sanitize_f32(self.config.trace_gamma.clamp(0.0, 0.90));
                            self.rtrl_state.accumulate_traces(
                                &self.scratch.rtrl_fb_buf,
                                sup_decay,
                                self.scratch.pending_active_in,
                            );
                            self.rtrl_state.apply_td_update(
                                &mut bank.rtrl_params,
                                1.0,
                                r_lr,
                                self.scratch.pending_active_in,
                            );
                            if r2_dim > 0 {
                                self.rtrl_slow_state.accumulate_traces(
                                    &self.scratch.rtrl_slow_fb_buf,
                                    sup_decay,
                                    self.rtrl_slow_state.max_in,
                                );
                                self.rtrl_slow_state.apply_td_update(
                                    &mut bank.rtrl_slow_params,
                                    1.0,
                                    r_lr,
                                    self.rtrl_slow_state.max_in,
                                );
                            }
                        }
                    }
                }
                if need_kernels {
                    bank.belief_kernel.update_action_value(
                        b_vec,
                        true_label,
                        1.0,
                        1.0,
                        &mut self.scratch.tmp_key,
                    );
                    if !correct {
                        bank.belief_kernel.update_action_value(
                            b_vec,
                            chosen,
                            -1.0,
                            1.0,
                            &mut self.scratch.tmp_key,
                        );
                    }
                }
            }
        }

        let drift_detected = if let Some(bank) = self.banks.get_mut(reg_id) {
            bank.ph.observe(r_norm)
        } else {
            false
        };

        if drift_detected {
            if let Some(bank) = self.banks.get_mut(reg_id) {
                bank.ph.reset_cusum();
                bank.cooldown_until = self.episode_counter.saturating_add(48);
            }
            if self.config.parallel_bank_probe {
                let mut mature_candidates = 0usize;
                for (bi, b) in self.banks.iter().enumerate() {
                    if bi != reg_id && b.age_eps >= 20 {
                        mature_candidates = mature_candidates.saturating_add(1);
                    }
                }
                if mature_candidates >= 2 {
                    self.scratch.parallel_bank_hits.fill(0);
                    self.scratch.parallel_bank_margins.fill(0.0);
                    self.parallel_probe_prev = reg_id;
                    self.parallel_probe_left = 6;
                    return false;
                }
                return self.switch_to_next_or_fresh_bank();
            }
            return self.switch_to_next_or_fresh_bank();
        }
        false
    }

    /// Онлайн MSE-обновление непрерывного регрессионного контура (Полоса 0/1/2 + Skip + RTRL + Page-Hinkley).
    /// 100% #[axiom_zero], 0% exp, 0 heap allocs. Возвращает `true`, если PH переключил банк.
    pub fn end_reg_step(&mut self, target: f32) -> bool {
        self.episode_counter = self.episode_counter.saturating_add(1);
        let tgt = sanitize_bounded_f32(target, 16.0);
        let reg_id = self.active_regime;
        let pred = self
            .banks
            .get(reg_id)
            .map(|b| {
                b.split_band_logit(
                    0,
                    &self.belief.state,
                    &self.rtrl_state.b,
                    &self.rtrl_slow_state.b,
                    &self.scratch.rtrl_skip_buf,
                )
            })
            .unwrap_or(0.0);
        let err_raw = sanitize_f32(tgt - pred);
        let err = sanitize_bounded_f32(err_raw, 2.5);
        // Рациональная оценка качества в (0, 1] без exp: q = 1 / (1 + 4 * err^2)
        let q_norm = total_div_f32(1.0, sanitize_f32(1.0 + 4.0 * err * err), 0.0);

        if self.probe_eps_left > 0 {
            self.probe_return_sum = sanitize_f32(self.probe_return_sum + q_norm);
            self.probe_return_cnt = self.probe_return_cnt.saturating_add(1);
            self.probe_eps_left -= 1;
            if self.probe_eps_left == 0 {
                let avg = total_div_f32(
                    self.probe_return_sum,
                    self.probe_return_cnt.max(1) as f32,
                    0.0,
                );
                if avg >= 0.65 {
                    if let Some(b) = self.banks.get_mut(reg_id) {
                        b.ph.reset_cusum();
                    }
                    return false;
                } else {
                    if let Some(b) = self.banks.get_mut(reg_id) {
                        b.cooldown_until = self.episode_counter.saturating_add(48);
                    }
                    return self.switch_to_next_or_fresh_bank();
                }
            }
            return false;
        }

        let multi_bank = self.banks.len() > 1;
        let bank_suspect_drift = multi_bank
            && self
                .banks
                .get(reg_id)
                .map(|b| b.ph.cusum > b.ph.threshold * 0.55 && b.ph.mu_bank >= 0.60)
                .unwrap_or(false);

        let lin_lr = self.config.linear_lr;
        let lin_wd = self.config.linear_wd;
        let r1_dim = self.rtrl_state.sectors.saturating_mul(3);
        let r2_dim = self.rtrl_slow_state.sectors.saturating_mul(3);
        let skip_dim = self.scratch.rtrl_skip_buf.len();

        if !bank_suspect_drift {
            let r1_scale = total_div_f32(1.0, (self.rtrl_state.sectors.max(1) as f32).sqrt(), 0.25);
            let r2_scale = total_div_f32(1.0, (self.rtrl_slow_state.sectors.max(1) as f32).sqrt(), 0.25);
            let r_lr = self.config.rtrl_lr;
            let keep = (1.0 - sanitize_f32(lin_wd).clamp(0.0, 0.05)).clamp(0.95, 1.0);
            if r1_dim > 0 {
                self.scratch.rtrl_fb_buf.fill(0.0);
                self.scratch.rtrl_slow_fb_buf.fill(0.0);
            }

            if let Some(bank) = self.banks.get_mut(reg_id) {
                bank.age_eps = bank.age_eps.saturating_add(1);
                if lin_lr > 0.0 {
                    let step = sanitize_f32(lin_lr * err);
                    if !bank.linear_w.is_empty() {
                        if let Some(w) = bank.linear_w.get_mut(0..bank.dim) {
                            update_normed_vec(w, &self.belief.state, step, keep, 6.0);
                        }
                    }
                    if r1_dim > 0 {
                        if let Some(rw) = bank.rtrl_actor_w.get(0..r1_dim) {
                            fast_saxpy(&mut self.scratch.rtrl_fb_buf, rw, sanitize_f32(err * r1_scale));
                        }
                        if let Some(rw) = bank.rtrl_actor_w.get_mut(0..r1_dim) {
                            update_normed_vec(rw, &self.rtrl_state.b, sanitize_f32(step * r1_scale), keep, 6.0);
                        }
                    }
                    if r2_dim > 0 {
                        if let Some(sw) = bank.rtrl_slow_actor_w.get(0..r2_dim) {
                            fast_saxpy(&mut self.scratch.rtrl_slow_fb_buf, sw, sanitize_f32(err * r2_scale));
                        }
                        if let Some(sw) = bank.rtrl_slow_actor_w.get_mut(0..r2_dim) {
                            update_normed_vec(sw, &self.rtrl_slow_state.b, sanitize_f32(step * r2_scale), keep, 6.0);
                        }
                    }
                    if skip_dim > 0 {
                        if let Some(xw) = bank.rtrl_skip_actor_w.get_mut(0..skip_dim) {
                            update_normed_vec(xw, &self.scratch.rtrl_skip_buf, step, keep, 6.0);
                        }
                    }
                    let ob = total_get_copy(&bank.linear_bias, 0, 0.0);
                    let _ = total_set(
                        &mut bank.linear_bias,
                        0,
                        sanitize_bounded_f32(ob + step * 0.5, 4.0),
                    );

                    if r_lr > 0.0 && r1_dim > 0 {
                        let sup_decay = sanitize_f32(self.config.trace_gamma.clamp(0.0, 0.90));
                        self.rtrl_state.accumulate_traces(
                            &self.scratch.rtrl_fb_buf,
                            sup_decay,
                            self.scratch.pending_active_in,
                        );
                        self.rtrl_state.apply_td_update(
                            &mut bank.rtrl_params,
                            1.0,
                            r_lr,
                            self.scratch.pending_active_in,
                        );
                        if r2_dim > 0 {
                            self.rtrl_slow_state.accumulate_traces(
                                &self.scratch.rtrl_slow_fb_buf,
                                sup_decay,
                                self.rtrl_slow_state.max_in,
                            );
                            self.rtrl_slow_state.apply_td_update(
                                &mut bank.rtrl_slow_params,
                                1.0,
                                r_lr,
                                self.rtrl_slow_state.max_in,
                            );
                        }
                    }
                }
            }
        }

        let drift_detected = if multi_bank {
            if let Some(bank) = self.banks.get_mut(reg_id) {
                bank.ph.observe(q_norm)
            } else {
                false
            }
        } else {
            false
        };

        if drift_detected {
            if let Some(bank) = self.banks.get_mut(reg_id) {
                bank.ph.reset_cusum();
                bank.cooldown_until = self.episode_counter.saturating_add(48);
            }
            return self.switch_to_next_or_fresh_bank();
        }
        false
    }

    fn allocate_fresh_or_evict_bank(&mut self) -> bool {
        if self.banks.len() <= 1 {
            return false;
        }
        self.scratch.td_mean_ema = 0.0;
        self.scratch.td_var_ema = 0.25;
        let prev = self.active_regime;
        let n = self.banks.len();
        for r in 0..n {
            if r != prev {
                if let Some(b) = self.banks.get(r) {
                    if b.age_eps == 0 && self.episode_counter >= b.cooldown_until {
                        self.active_regime = r;
                        self.probe_eps_left = 0;
                        self.total_switches = self.total_switches.saturating_add(1);
                        return true;
                    }
                }
            }
        }
        let next_r = total_rem_usize(prev.saturating_add(1), n, 0);
        if let Some(b) = self.banks.get_mut(next_r) {
            b.clear_all(&self.config, next_r);
        }
        self.active_regime = next_r;
        self.probe_eps_left = 0;
        self.total_switches = self.total_switches.saturating_add(1);
        true
    }

    fn switch_to_next_or_fresh_bank(&mut self) -> bool {
        if self.banks.len() <= 1 {
            return false;
        }
        self.scratch.td_mean_ema = 0.0;
        self.scratch.td_var_ema = 0.25;
        let prev = self.active_regime;
        let n = self.banks.len();
        for r in 0..n {
            if r != prev {
                if let Some(b) = self.banks.get(r) {
                    if b.age_eps > 0 && self.episode_counter >= b.cooldown_until {
                        self.active_regime = r;
                        self.probe_eps_left = 6;
                        self.probe_return_sum = 0.0;
                        self.probe_return_cnt = 0;
                        self.total_switches = self.total_switches.saturating_add(1);
                        return true;
                    }
                }
            }
        }
        for r in 0..n {
            if r != prev {
                if let Some(b) = self.banks.get(r) {
                    if b.age_eps == 0 && self.episode_counter >= b.cooldown_until {
                        self.active_regime = r;
                        self.probe_eps_left = 0;
                        self.total_switches = self.total_switches.saturating_add(1);
                        return true;
                    }
                }
            }
        }
        let next_r = total_rem_usize(prev.saturating_add(1), n, 0);
        if let Some(b) = self.banks.get_mut(next_r) {
            b.clear_all(&self.config, next_r);
        }
        self.active_regime = next_r;
        self.probe_eps_left = 0;
        self.total_switches = self.total_switches.saturating_add(1);
        true
    }
}
