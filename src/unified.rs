// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! UnifiedCayleyCore (v0.5.0, оптимизированная архитектура):
//! - Единое универсальное непрерывное ядро (`One-Config` + `Unitary-Limit`) без backprop/BPTT.
//! - 100% `#[axiom_zero]`, 0% `exp/ln/sin/cos/tanh`, 0% `.sqrt()`, 0 аллокаций на горячем шаге.
//! - Ключевые оптимизации (`step131`):
//!   * Общий временной хребет (`shared_backbone`: один набор `B1`/`B2` в `slots[0]`, лёгкие
//!     банки режимов `1..max_banks` хранят только головки, полосовые вентили, 4-band IDBD и PH);
//!   * Ленивая аллокация буферов селективности и локальной ОС (0 байт при выключенных флагах);
//!   * 4-скалярный полосовой рациональный IDBD (`[f32; 4]` вместо поэлементного);
//!   * Опциональная Полоса 0 (`use_band0`, 0 байт и 0 вычислений при `false`);
//!   * Единые `#[axiom_zero]` алгебраические примитивы из `crate::algebra`.

use axiom_zero::{
    axiom_zero, sanitize_bounded_f32, sanitize_f32, total_div_f32, total_div_unit_shifted_f32,
    total_get_copy, total_rem_usize, total_set,
};

use crate::algebra::{
    fill_p4_logprob_grads as algebra_fill_p4_logprob_grads, inv_sqrt_select, nlms_step, norm3,
    rmsnorm_band_backward, rmsnorm_band_into, rmsprop_step, rot3_apply, rot3_step_and_deriv,
    scrub_slice_seu,
};
use crate::math::{
    fast_dot, fast_saxpy, fast_scale, inv_sqrt_newton_2step_f32, inv_sqrt_newton_f32,
    rand_signed_f32, rand_unit_f32, rational_pos_and_deriv, rational_probs_p4_into, splitmix64,
};

pub const UNIFIED_MAX_RAW_IN: usize = 24;
pub const UNIFIED_MAX_EFF_IN: usize = 48;
pub const UNIFIED_B0_SECTORS: usize = 8;
pub const UNIFIED_B0_DIM: usize = UNIFIED_B0_SECTORS * 3;
pub const UNIFIED_B1_SLOTS: usize = 16;
pub const UNIFIED_B1_DIM: usize = UNIFIED_B1_SLOTS * 2;
pub const UNIFIED_B2_SLOTS: usize = 8;
pub const UNIFIED_B2_DIM: usize = UNIFIED_B2_SLOTS * 3;
pub const UNIFIED_MAX_BANKS: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct UnifiedCayleyConfig {
    pub raw_in_dim: usize,
    pub num_actions: usize,
    pub seed: u64,
    // Переключатели абляции (Suite B: B1 -> B2 -> B3):
    pub enable_selectivity: bool,
    pub enable_local_feedback: bool,
    pub enable_bartlett_ph: bool,
    pub enable_idbd: bool,
    pub enable_surprise_rl: bool,
    pub enable_bank_switching: bool,
    // Архитектура и динамика:
    pub use_band0: bool,
    pub shared_backbone: bool,
    pub use_delta_channel: bool,
    pub delta_scale: f32,
    pub min_alpha_sq: f32, // 1e-4 => max lambda = 0.9999 (нет потолка 0.995!)
    pub unitary_limit: bool,
    pub learn_norm: bool,
    pub exact_inv_sqrt: bool,
    // Скорости обучения и RL:
    pub ro_lr: f32,
    pub rtrl_lr: f32,
    pub critic_lr: f32,
    pub weight_decay: f32,
    pub gamma: f32,
    pub trace_lambda: f32,
    pub entropy_beta: f32,
    pub idbd_lr: f32,
    // Bartlett PH детектор:
    pub ph_delta: f32,
    pub ph_threshold: f32,
    pub ph_warmup: usize,
    pub max_banks: usize,
}

impl UnifiedCayleyConfig {
    /// Чекпоинт B1: Master Eq без 0.995 + 2D/3D + Newton RMSNorm + Band Gates + безопасный вход var+1.
    pub fn checkpoint_b1(num_actions: usize, raw_in_dim: usize, seed: u64) -> Self {
        Self {
            raw_in_dim: raw_in_dim.clamp(1, UNIFIED_MAX_RAW_IN),
            num_actions: num_actions.clamp(1, 32),
            seed,
            enable_selectivity: false,
            enable_local_feedback: false,
            enable_bartlett_ph: false,
            enable_idbd: false,
            enable_surprise_rl: false,
            enable_bank_switching: false,
            use_band0: true,
            shared_backbone: true,
            use_delta_channel: true,
            delta_scale: 2.0,
            min_alpha_sq: 1e-4,
            unitary_limit: false,
            learn_norm: true,
            exact_inv_sqrt: false,
            ro_lr: 0.032,
            rtrl_lr: 0.006,
            critic_lr: 0.08,
            weight_decay: 0.0001,
            gamma: 0.90,
            trace_lambda: 0.65,
            entropy_beta: 0.006,
            idbd_lr: 0.0,
            ph_delta: 0.14,
            ph_threshold: 2.2,
            ph_warmup: 28,
            max_banks: 1,
        }
    }

    /// Чекпоинт B2: B1 + обучаемая селективность beta_k + локальный Kerr/state вентиль z_{t,k}(u_{t-1}).
    pub fn checkpoint_b2(num_actions: usize, raw_in_dim: usize, seed: u64) -> Self {
        let mut c = Self::checkpoint_b1(num_actions, raw_in_dim, seed);
        c.enable_selectivity = true;
        c.enable_local_feedback = true;
        c
    }

    /// Финальный One-Config (B3; включает shared_backbone, use_band0=false, 4-скалярный IDBD и Bartlett PH):
    /// общий адаптивный Master-Eq хребет `B1+B2` (`shared_backbone`, с селективностью `beta_k` и
    /// локальной ОС `z_{t,k}(u_{t-1})`, без избыточной Полосы 0), 4-скалярный полосовой IDBD,
    /// 4 лёгких банка Bartlett PH.
    pub fn one_config(num_actions: usize, raw_in_dim: usize, seed: u64) -> Self {
        let mut c = Self::checkpoint_b2(num_actions, raw_in_dim, seed);
        c.use_band0 = false;
        c.shared_backbone = true;
        c.enable_bartlett_ph = true;
        c.enable_idbd = true;
        c.enable_surprise_rl = true;
        c.enable_bank_switching = true;
        c.idbd_lr = 0.025;
        c.max_banks = UNIFIED_MAX_BANKS;
        c
    }

    /// Track 2 (Unitary Limit, sigma=0): чистые ортогональные роторы с alpha_min = 0 (lambda = 1.0).
    pub fn unitary_limit_cfg(num_actions: usize, raw_in_dim: usize, seed: u64) -> Self {
        let mut c = Self::checkpoint_b1(num_actions, raw_in_dim, seed);
        c.use_band0 = false;
        c.min_alpha_sq = 0.0;
        c.unitary_limit = true;
        c.use_delta_channel = false;
        c.learn_norm = false;
        c.exact_inv_sqrt = true;
        c.ro_lr = 0.05;
        c.rtrl_lr = 0.01;
        c
    }
}

/// Строго каузальная Welford-нормализация (`var_i + 1.0`, `step122`, без `.sqrt()`)
/// с алгебраическим каналом скорости `dx_t`.
#[derive(Clone)]
pub struct CausalInputConditioner {
    pub raw_dim: usize,
    pub use_delta: bool,
    pub eff_dim: usize,
    pub delta_scale: f32,
    pub count: u64,
    pub mean: Vec<f32>,
    pub m2: Vec<f32>,
    pub prev_norm: Vec<f32>,
    pub has_prev: bool,
    pub out_feat: Vec<f32>,
}

#[axiom_zero]
impl CausalInputConditioner {
    pub fn new(raw_dim: usize, use_delta: bool, delta_scale: f32) -> Self {
        let d = raw_dim.clamp(1, UNIFIED_MAX_RAW_IN);
        let eff = if use_delta { d.saturating_mul(2) } else { d };
        Self {
            raw_dim: d,
            use_delta,
            eff_dim: eff,
            delta_scale: sanitize_f32(delta_scale).clamp(0.1, 12.0),
            count: 0,
            mean: vec![0.0_f32; d],
            m2: vec![0.0_f32; d],
            prev_norm: vec![0.0_f32; d],
            has_prev: false,
            out_feat: vec![0.0_f32; eff],
        }
    }

    pub fn reset_episode_delta(&mut self) {
        self.prev_norm.fill(0.0);
        self.has_prev = false;
    }

    pub fn hard_reset(&mut self) {
        self.count = 0;
        self.mean.fill(0.0);
        self.m2.fill(0.0);
        self.reset_episode_delta();
        self.out_feat.fill(0.0);
    }

    pub fn step_transform(&mut self, raw_x: &[f32], learn_stats: bool) {
        let inv_n_prev = if self.count > 1 {
            total_div_f32(1.0, (self.count - 1) as f32, 0.0)
        } else {
            0.0
        };
        let next_cnt = self.count.saturating_add(1);
        let inv_n_next = total_div_f32(1.0, next_cnt as f32, 1.0);

        for i in 0..self.raw_dim {
            let x = sanitize_bounded_f32(total_get_copy(raw_x, i, 0.0), 1e12);
            let mu = total_get_copy(&self.mean, i, 0.0);
            let m2i = total_get_copy(&self.m2, i, 0.0);
            let var = if self.count >= 2 {
                sanitize_f32(m2i * inv_n_prev).max(0.0)
            } else {
                0.0
            };
            let inv_sd = inv_sqrt_newton_2step_f32(sanitize_f32(var + 1.0));
            let xn = sanitize_bounded_f32((x - mu) * inv_sd, 12.0);
            let _ = total_set(&mut self.out_feat, i, xn);

            if self.use_delta {
                let prev = total_get_copy(&self.prev_norm, i, 0.0);
                let dx_raw = if self.has_prev {
                    sanitize_bounded_f32((xn - prev) * self.delta_scale, 32.0)
                } else {
                    0.0
                };
                let dx_sq = sanitize_f32(dx_raw * dx_raw);
                let dx_comp = sanitize_f32(dx_raw * inv_sqrt_newton_2step_f32(dx_sq + 1.0));
                let _ = total_set(&mut self.out_feat, self.raw_dim + i, dx_comp);
                let _ = total_set(&mut self.prev_norm, i, xn);
            }

            if learn_stats {
                let d1 = sanitize_bounded_f32(x - mu, 1e12);
                let new_mu = sanitize_bounded_f32(mu + d1 * inv_n_next, 1e12);
                let d2 = sanitize_bounded_f32(x - new_mu, 1e12);
                let new_m2 = sanitize_f32(m2i + d1 * d2).clamp(0.0, 1e24);
                let _ = total_set(&mut self.mean, i, new_mu);
                let _ = total_set(&mut self.m2, i, new_m2);
            }
        }
        self.has_prev = true;
        if learn_stats {
            self.count = next_cnt;
        }
    }

    pub fn memory_bytes(&self) -> usize {
        (self.mean.len() + self.m2.len() + self.prev_norm.len() + self.out_feat.len()) * 4
    }
}

/// Опциональная замороженная Полоса 0 (8 x 3D = 24D; при `enabled == false` занимает 0 байт динамики).
#[derive(Clone)]
pub struct Band0Clock {
    pub enabled: bool,
    pub eff_in: usize,
    pub lambdas: Vec<f32>,
    pub rot_c1: Vec<f32>,
    pub rot_c2: Vec<f32>,
    pub axis: Vec<f32>,
    pub in_dir: Vec<f32>,
    pub proj_w: Vec<f32>, // [8 * eff_in]
}

#[axiom_zero]
impl Band0Clock {
    pub fn new(eff_in: usize, seed: u64, enabled: bool) -> Self {
        let ein = eff_in.clamp(1, UNIFIED_MAX_EFF_IN);
        if !enabled {
            return Self {
                enabled: false,
                eff_in: ein,
                lambdas: Vec::new(),
                rot_c1: Vec::new(),
                rot_c2: Vec::new(),
                axis: Vec::new(),
                in_dir: Vec::new(),
                proj_w: Vec::new(),
            };
        }
        let mut lambdas = vec![0.0_f32; UNIFIED_B0_SECTORS];
        let mut rot_c1 = vec![0.0_f32; UNIFIED_B0_SECTORS];
        let mut rot_c2 = vec![0.0_f32; UNIFIED_B0_SECTORS];
        let mut axis = vec![0.0_f32; UNIFIED_B0_DIM];
        let mut in_dir = vec![0.0_f32; UNIFIED_B0_DIM];
        let mut proj_w = vec![0.0_f32; UNIFIED_B0_SECTORS * ein];

        let mut rng = splitmix64(seed ^ 0xB0C1_0C40_55AA_1001);
        let inv_root_in = inv_sqrt_newton_f32(ein as f32);

        for s in 0..UNIFIED_B0_SECTORS {
            let frac = total_div_f32(s as f32, (UNIFIED_B0_SECTORS - 1) as f32, 0.5);
            let alpha = 1.10 - 1.00 * frac;
            let lam = total_div_unit_shifted_f32(1.0, alpha * alpha);
            let _ = total_set(&mut lambdas, s, lam);

            let sign = if (s & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
            let omega = sign * (0.22 + 1.55 * frac);
            let g = 0.5 * omega;
            let inv_d = total_div_unit_shifted_f32(1.0, g * g);
            let _ = total_set(&mut rot_c1, s, 2.0 * g * inv_d);
            let _ = total_set(&mut rot_c2, s, 2.0 * g * g * inv_d);

            let (e0, e1, e2) = norm3(
                rand_signed_f32(&mut rng),
                rand_signed_f32(&mut rng),
                rand_signed_f32(&mut rng),
            );
            let off = s * 3;
            let _ = total_set(&mut axis, off, e0);
            let _ = total_set(&mut axis, off + 1, e1);
            let _ = total_set(&mut axis, off + 2, e2);

            let (d0, d1, d2) = norm3(
                rand_signed_f32(&mut rng),
                rand_signed_f32(&mut rng),
                rand_signed_f32(&mut rng),
            );
            let _ = total_set(&mut in_dir, off, d0);
            let _ = total_set(&mut in_dir, off + 1, d1);
            let _ = total_set(&mut in_dir, off + 2, d2);

            let woff = s * ein;
            for j in 0..ein {
                let r = rand_signed_f32(&mut rng) * inv_root_in;
                let _ = total_set(&mut proj_w, woff + j, sanitize_f32(r));
            }
        }

        Self {
            enabled: true,
            eff_in: ein,
            lambdas,
            rot_c1,
            rot_c2,
            axis,
            in_dir,
            proj_w,
        }
    }

    pub fn step_into(&self, feat_x: &[f32], state: &mut [f32]) {
        if !self.enabled || state.len() < UNIFIED_B0_DIM {
            return;
        }
        for s in 0..UNIFIED_B0_SECTORS {
            let lam = total_get_copy(&self.lambdas, s, 0.8);
            let c1 = total_get_copy(&self.rot_c1, s, 0.0);
            let c2 = total_get_copy(&self.rot_c2, s, 0.0);
            let off = s * 3;
            let e = (
                total_get_copy(&self.axis, off, 0.0),
                total_get_copy(&self.axis, off + 1, 0.0),
                total_get_copy(&self.axis, off + 2, 1.0),
            );
            let u = (
                total_get_copy(state, off, 0.0),
                total_get_copy(state, off + 1, 0.0),
                total_get_copy(state, off + 2, 0.0),
            );
            let r = rot3_apply(e, c1, c2, u);
            let woff = s * self.eff_in;
            let mut m = 0.0_f32;
            if let Some(ws) = self.proj_w.get(woff..woff + self.eff_in) {
                m = fast_dot(ws, feat_x);
            }
            let drv = sanitize_bounded_f32(m * 0.5, 8.0);
            let d0 = total_get_copy(&self.in_dir, off, 1.0);
            let d1 = total_get_copy(&self.in_dir, off + 1, 0.0);
            let d2 = total_get_copy(&self.in_dir, off + 2, 0.0);
            let _ = total_set(state, off, sanitize_bounded_f32(lam * r.0 + drv * d0, 64.0));
            let _ = total_set(state, off + 1, sanitize_bounded_f32(lam * r.1 + drv * d1, 64.0));
            let _ = total_set(state, off + 2, sanitize_bounded_f32(lam * r.2 + drv * d2, 64.0));
        }
    }

    pub fn memory_bytes(&self) -> usize {
        (self.proj_w.len()
            + self.lambdas.len()
            + self.rot_c1.len()
            + self.rot_c2.len()
            + self.axis.len()
            + self.in_dir.len())
            * 4
    }
}

/// Дисперсионно-замороженный Page-Hinkley с корректировкой Бартлетта (v0.5.0, без `sqrt`).
#[derive(Clone, Debug)]
pub struct BartlettPageHinkley {
    pub mu_bank: f32,
    pub var_bank: f32,
    pub cov1_bank: f32,
    pub prev_err: f32,
    pub has_prev_err: bool,
    pub cusum: f32,
    pub delta: f32,
    pub threshold: f32,
    pub min_warmup: usize,
    pub episodes_seen: usize,
}

#[axiom_zero]
impl BartlettPageHinkley {
    pub fn new(delta: f32, threshold: f32, min_warmup: usize) -> Self {
        Self {
            mu_bank: 0.0,
            var_bank: 0.04,
            cov1_bank: 0.0,
            prev_err: 0.0,
            has_prev_err: false,
            cusum: 0.0,
            delta: sanitize_f32(delta).clamp(0.02, 0.45),
            threshold: sanitize_f32(threshold).clamp(0.5, 1e6),
            min_warmup: min_warmup.max(6),
            episodes_seen: 0,
        }
    }

    pub fn reset_cusum(&mut self) {
        self.cusum = 0.0;
    }

    pub fn observe(&mut self, r_norm: f32, use_bartlett: bool) -> bool {
        let r = sanitize_f32(r_norm).clamp(0.0, 1.0);
        self.episodes_seen = self.episodes_seen.saturating_add(1);

        if self.episodes_seen == 1 {
            self.mu_bank = r;
            self.var_bank = 0.04;
            self.cov1_bank = 0.0;
            self.prev_err = 0.0;
            self.has_prev_err = false;
            self.cusum = 0.0;
            return false;
        }

        if self.episodes_seen <= self.min_warmup {
            let inv_n = total_div_f32(1.0, self.episodes_seen as f32, 0.1);
            let e = r - self.mu_bank;
            self.mu_bank = sanitize_f32(self.mu_bank + inv_n * e);
            let d = r - self.mu_bank;
            if self.has_prev_err {
                self.cov1_bank = sanitize_f32(0.92 * self.cov1_bank + 0.08 * e * self.prev_err);
            }
            self.var_bank = sanitize_f32(0.92 * self.var_bank + 0.08 * d * d).clamp(0.0025, 0.5);
            self.prev_err = e;
            self.has_prev_err = true;
            self.cusum = 0.0;
            return false;
        }

        if self.mu_bank < 0.42 {
            let e = r - self.mu_bank;
            self.mu_bank = sanitize_f32(self.mu_bank + 0.08 * e);
            self.prev_err = e;
            self.has_prev_err = true;
            self.cusum = 0.0;
            return false;
        }

        let drop_raw = sanitize_f32(self.mu_bank - r - self.delta);
        if use_bartlett {
            let rho_raw = total_div_f32(self.cov1_bank, self.var_bank.max(1e-4), 0.0);
            let rho = sanitize_f32(rho_raw).clamp(-0.8, 0.8);
            let kappa_bart = total_div_f32(1.0 + rho, (1.0 - rho).max(0.2), 1.0).clamp(0.111, 9.0);
            let lrv = sanitize_f32(self.var_bank * kappa_bart).clamp(0.025, 4.5);
            let inv_sigma_eff = inv_sqrt_newton_2step_f32(lrv);
            let z_drop = sanitize_f32(drop_raw * inv_sigma_eff * 0.25);
            self.cusum = sanitize_f32(self.cusum + z_drop).max(0.0);
        } else {
            self.cusum = sanitize_f32(self.cusum + drop_raw).max(0.0);
        }

        if self.cusum <= 1e-6 {
            let e = r - self.mu_bank;
            self.mu_bank = sanitize_f32(self.mu_bank + 0.06 * e);
            if self.has_prev_err {
                self.cov1_bank = sanitize_f32(0.95 * self.cov1_bank + 0.05 * e * self.prev_err);
            }
            self.var_bank = sanitize_f32(0.95 * self.var_bank + 0.05 * e * e).clamp(0.0025, 0.5);
            self.prev_err = e;
            self.has_prev_err = true;
        }

        self.cusum >= self.threshold
    }
}

/// Один банк параметров внутри `UnifiedCayleyCore`:
/// - При `shared_backbone = true` слот `0` владеет рекуррентным хребтом `B1`+`B2`,
///   а слоты `1..max_banks` хранят ТОЛЬКО головки, вентили полос, 4-band IDBD и Bartlett PH.
/// - Поля селективности и локальной ОС аллоцируются лениво (0 байт, если выключены).
#[derive(Clone)]
pub struct UnifiedRegimeSlot {
    pub eff_in: usize,
    pub num_actions: usize,
    pub feat_dim: usize,
    // Полоса 1 (16 x 2D — пустые при bank_idx > 0 && shared_backbone):
    pub b1_alpha: Vec<f32>,
    pub b1_omega: Vec<f32>,
    pub b1_beta: Vec<f32>,
    pub b1_bdelta: Vec<f32>,
    pub b1_qfb: Vec<f32>,
    pub b1_in_w: Vec<f32>,
    pub b1_sel_w: Vec<f32>,
    pub v_b1_alpha: Vec<f32>,
    pub v_b1_omega: Vec<f32>,
    pub v_b1_beta: Vec<f32>,
    pub v_b1_bdelta: Vec<f32>,
    pub v_b1_qfb: Vec<f32>,
    pub v_b1_in_w: Vec<f32>,
    pub v_b1_sel_w: Vec<f32>,
    // Полоса 2 (8 x 3D — пустые при bank_idx > 0 && shared_backbone):
    pub b2_alpha: Vec<f32>,
    pub b2_omega: Vec<f32>,
    pub b2_beta: Vec<f32>,
    pub b2_bdelta: Vec<f32>,
    pub b2_axis: Vec<f32>,
    pub b2_in_dir: Vec<f32>,
    pub b2_qfb: Vec<f32>,
    pub b2_in_w: Vec<f32>,
    pub b2_sel_w: Vec<f32>,
    pub v_b2_alpha: Vec<f32>,
    pub v_b2_omega: Vec<f32>,
    pub v_b2_beta: Vec<f32>,
    pub v_b2_bdelta: Vec<f32>,
    pub v_b2_qfb: Vec<f32>,
    pub v_b2_in_w: Vec<f32>,
    pub v_b2_sel_w: Vec<f32>,
    // Полосовые вентили [skip, B0, B1, B2]:
    pub gate_logits: [f32; 4],
    pub v_gates: [f32; 4],
    // Выходные головки (Class/Actor, Reg/Critic) + 4-скалярный полосовой IDBD (`step131`):
    pub actor_w: Vec<f32>,  // [num_actions * feat_dim]
    pub actor_b: Vec<f32>,  // [num_actions]
    pub critic_w: Vec<f32>, // [feat_dim]
    pub critic_b: f32,
    pub v_actor_w: Vec<f32>,
    pub v_actor_b: Vec<f32>,
    pub idbd_beta_actor: [f32; 4],
    pub idbd_h_actor: [f32; 4],
    pub idbd_beta_critic: [f32; 4],
    pub idbd_h_critic: [f32; 4],
    // Метаданные банка & Bartlett PH:
    pub ph: BartlettPageHinkley,
    pub ret_ema: f32,
    pub td_var_ema: f32,
    pub age_eps: usize,
    pub cooldown_until: usize,
    pub last_used_ep: usize,
}

#[axiom_zero]
impl UnifiedRegimeSlot {
    pub fn new(cfg: &UnifiedCayleyConfig, eff_in: usize, bank_idx: usize) -> Self {
        let ein = eff_in.clamp(1, UNIFIED_MAX_EFF_IN);
        let na = cfg.num_actions.clamp(1, 32);
        let b0_dim = if cfg.use_band0 { UNIFIED_B0_DIM } else { 0 };
        let feat_dim = ein + b0_dim + UNIFIED_B1_DIM + UNIFIED_B2_DIM;
        let has_bb = bank_idx == 0 || !cfg.shared_backbone;
        let has_sel = has_bb && cfg.enable_selectivity;
        let has_fb = has_sel && cfg.enable_local_feedback;

        let mut rng = splitmix64(
            cfg.seed
                ^ 0x5107_0000_0000_0001
                ^ ((bank_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        let inv_root_in = inv_sqrt_newton_f32(ein as f32);

        let mut b1_alpha = if has_bb { vec![0.25_f32; UNIFIED_B1_SLOTS] } else { Vec::new() };
        let mut b1_omega = if has_bb { vec![0.40_f32; UNIFIED_B1_SLOTS] } else { Vec::new() };
        let mut b1_in_w = if has_bb { vec![0.0_f32; UNIFIED_B1_SLOTS * ein] } else { Vec::new() };
        let mut b1_sel_w = if has_sel { vec![0.0_f32; UNIFIED_B1_SLOTS * ein] } else { Vec::new() };

        let mut b2_alpha = if has_bb { vec![0.25_f32; UNIFIED_B2_SLOTS] } else { Vec::new() };
        let mut b2_omega = if has_bb { vec![0.30_f32; UNIFIED_B2_SLOTS] } else { Vec::new() };
        let mut b2_axis = if has_bb { vec![0.0_f32; UNIFIED_B2_DIM] } else { Vec::new() };
        let mut b2_in_dir = if has_bb { vec![0.0_f32; UNIFIED_B2_DIM] } else { Vec::new() };
        let mut b2_in_w = if has_bb { vec![0.0_f32; UNIFIED_B2_SLOTS * ein] } else { Vec::new() };
        let mut b2_sel_w = if has_sel { vec![0.0_f32; UNIFIED_B2_SLOTS * ein] } else { Vec::new() };

        if has_bb {
            for k in 0..UNIFIED_B1_SLOTS {
                let frac = total_div_f32(k as f32, (UNIFIED_B1_SLOTS - 1) as f32, 0.5);
                let a0 = if cfg.unitary_limit {
                    0.0_f32
                } else {
                    sanitize_f32(0.04 + 0.91 * frac * frac)
                };
                let sign = if (k & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
                let w0 = sanitize_f32(sign * (0.28 + 2.00 * frac));
                let _ = total_set(&mut b1_alpha, k, a0);
                let _ = total_set(&mut b1_omega, k, w0);
                let woff = k * ein;
                for j in 0..ein {
                    let _ = total_set(
                        &mut b1_in_w,
                        woff + j,
                        sanitize_f32(rand_signed_f32(&mut rng) * 0.45 * inv_root_in),
                    );
                    let r_sel = rand_signed_f32(&mut rng);
                    if has_sel {
                        let _ = total_set(
                            &mut b1_sel_w,
                            woff + j,
                            sanitize_f32(r_sel * 0.25 * inv_root_in),
                        );
                    }
                }
            }

            for s in 0..UNIFIED_B2_SLOTS {
                let frac = total_div_f32(s as f32, (UNIFIED_B2_SLOTS - 1) as f32, 0.5);
                let a0 = if cfg.unitary_limit {
                    0.0_f32
                } else {
                    sanitize_f32(0.04 + 0.72 * frac * frac)
                };
                let sign = if (s & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
                let w0 = sanitize_f32(sign * (0.14 + 1.35 * frac));
                let _ = total_set(&mut b2_alpha, s, a0);
                let _ = total_set(&mut b2_omega, s, w0);

                let off = s * 3;
                let (e0, e1, e2) = norm3(
                    rand_signed_f32(&mut rng),
                    rand_signed_f32(&mut rng),
                    rand_signed_f32(&mut rng),
                );
                let _ = total_set(&mut b2_axis, off, e0);
                let _ = total_set(&mut b2_axis, off + 1, e1);
                let _ = total_set(&mut b2_axis, off + 2, e2);

                let (d0, d1, d2) = norm3(
                    rand_signed_f32(&mut rng),
                    rand_signed_f32(&mut rng),
                    rand_signed_f32(&mut rng),
                );
                let _ = total_set(&mut b2_in_dir, off, d0);
                let _ = total_set(&mut b2_in_dir, off + 1, d1);
                let _ = total_set(&mut b2_in_dir, off + 2, d2);

                let woff = s * ein;
                for j in 0..ein {
                    let _ = total_set(
                        &mut b2_in_w,
                        woff + j,
                        sanitize_f32(rand_signed_f32(&mut rng) * 0.45 * inv_root_in),
                    );
                    let r_sel = rand_signed_f32(&mut rng);
                    if has_sel {
                        let _ = total_set(
                            &mut b2_sel_w,
                            woff + j,
                            sanitize_f32(r_sel * 0.25 * inv_root_in),
                        );
                    }
                }
            }
        }

        let mut actor_w = vec![0.0_f32; na * feat_dim];
        let inv_root_f = inv_sqrt_newton_f32(feat_dim as f32);
        for w in actor_w.iter_mut() {
            *w = sanitize_f32(rand_signed_f32(&mut rng) * 0.08 * inv_root_f);
        }

        Self {
            eff_in: ein,
            num_actions: na,
            feat_dim,
            b1_alpha,
            b1_omega,
            b1_beta: if has_sel { vec![0.0_f32; UNIFIED_B1_SLOTS] } else { Vec::new() },
            b1_bdelta: if has_sel { vec![0.0_f32; UNIFIED_B1_SLOTS] } else { Vec::new() },
            b1_qfb: if has_fb { vec![0.0_f32; UNIFIED_B1_DIM] } else { Vec::new() },
            b1_in_w,
            b1_sel_w,
            v_b1_alpha: if has_bb { vec![0.04_f32; UNIFIED_B1_SLOTS] } else { Vec::new() },
            v_b1_omega: if has_bb { vec![0.04_f32; UNIFIED_B1_SLOTS] } else { Vec::new() },
            v_b1_beta: if has_sel { vec![0.04_f32; UNIFIED_B1_SLOTS] } else { Vec::new() },
            v_b1_bdelta: if has_sel { vec![0.04_f32; UNIFIED_B1_SLOTS] } else { Vec::new() },
            v_b1_qfb: if has_fb { vec![0.04_f32; UNIFIED_B1_DIM] } else { Vec::new() },
            v_b1_in_w: if has_bb { vec![0.04_f32; UNIFIED_B1_SLOTS * ein] } else { Vec::new() },
            v_b1_sel_w: if has_sel { vec![0.04_f32; UNIFIED_B1_SLOTS * ein] } else { Vec::new() },
            b2_alpha,
            b2_omega,
            b2_beta: if has_sel { vec![0.0_f32; UNIFIED_B2_SLOTS] } else { Vec::new() },
            b2_bdelta: if has_sel { vec![0.0_f32; UNIFIED_B2_SLOTS] } else { Vec::new() },
            b2_axis,
            b2_in_dir,
            b2_qfb: if has_fb { vec![0.0_f32; UNIFIED_B2_DIM] } else { Vec::new() },
            b2_in_w,
            b2_sel_w,
            v_b2_alpha: if has_bb { vec![0.04_f32; UNIFIED_B2_SLOTS] } else { Vec::new() },
            v_b2_omega: if has_bb { vec![0.04_f32; UNIFIED_B2_SLOTS] } else { Vec::new() },
            v_b2_beta: if has_sel { vec![0.04_f32; UNIFIED_B2_SLOTS] } else { Vec::new() },
            v_b2_bdelta: if has_sel { vec![0.04_f32; UNIFIED_B2_SLOTS] } else { Vec::new() },
            v_b2_qfb: if has_fb { vec![0.04_f32; UNIFIED_B2_DIM] } else { Vec::new() },
            v_b2_in_w: if has_bb { vec![0.04_f32; UNIFIED_B2_SLOTS * ein] } else { Vec::new() },
            v_b2_sel_w: if has_sel { vec![0.04_f32; UNIFIED_B2_SLOTS * ein] } else { Vec::new() },
            gate_logits: [-0.25, -0.25, 0.25, 0.25],
            v_gates: [0.04_f32; 4],
            actor_w,
            actor_b: vec![0.0_f32; na],
            critic_w: vec![0.0_f32; feat_dim],
            critic_b: 0.0,
            v_actor_w: vec![0.04_f32; na * feat_dim],
            v_actor_b: vec![0.04_f32; na],
            idbd_beta_actor: [0.0_f32; 4],
            idbd_h_actor: [0.0_f32; 4],
            idbd_beta_critic: [0.0_f32; 4],
            idbd_h_critic: [0.0_f32; 4],
            ph: BartlettPageHinkley::new(cfg.ph_delta, cfg.ph_threshold, cfg.ph_warmup),
            ret_ema: 0.5,
            td_var_ema: 0.25,
            age_eps: 0,
            cooldown_until: 0,
            last_used_ep: 0,
        }
    }

    pub fn scrub_seu(&mut self) -> u64 {
        let mut faults = 0u64;
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b1_alpha, 6.0, 0.25));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b1_omega, 4.5, 0.30));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b1_beta, 6.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b1_bdelta, 6.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b1_qfb, 4.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b1_in_w, 6.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b1_sel_w, 6.0, 0.0));

        faults = faults.saturating_add(scrub_slice_seu(&mut self.b2_alpha, 6.0, 0.25));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b2_omega, 4.5, 0.30));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b2_beta, 6.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b2_bdelta, 6.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b2_qfb, 4.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b2_in_w, 6.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b2_sel_w, 6.0, 0.0));

        faults = faults.saturating_add(scrub_slice_seu(&mut self.gate_logits, 6.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.actor_w, 8.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.actor_b, 8.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.critic_w, 8.0, 0.0));
        if !self.critic_b.is_finite() || self.critic_b.abs() > 8.0 {
            self.critic_b = 0.0;
            faults = faults.saturating_add(1);
        }
        faults = faults.saturating_add(scrub_slice_seu(&mut self.idbd_beta_actor, 4.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.idbd_h_actor, 4.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.idbd_beta_critic, 4.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.idbd_h_critic, 4.0, 0.0));
        faults
    }

    pub fn memory_bytes(&self) -> usize {
        let vec_floats = self.b1_alpha.len()
            + self.b1_omega.len()
            + self.b1_beta.len()
            + self.b1_bdelta.len()
            + self.b1_qfb.len()
            + self.b1_in_w.len()
            + self.b1_sel_w.len()
            + self.v_b1_alpha.len()
            + self.v_b1_omega.len()
            + self.v_b1_beta.len()
            + self.v_b1_bdelta.len()
            + self.v_b1_qfb.len()
            + self.v_b1_in_w.len()
            + self.v_b1_sel_w.len()
            + self.b2_alpha.len()
            + self.b2_omega.len()
            + self.b2_beta.len()
            + self.b2_bdelta.len()
            + self.b2_axis.len()
            + self.b2_in_dir.len()
            + self.b2_qfb.len()
            + self.b2_in_w.len()
            + self.b2_sel_w.len()
            + self.v_b2_alpha.len()
            + self.v_b2_omega.len()
            + self.v_b2_beta.len()
            + self.v_b2_bdelta.len()
            + self.v_b2_qfb.len()
            + self.v_b2_in_w.len()
            + self.v_b2_sel_w.len()
            + self.actor_w.len()
            + self.actor_b.len()
            + self.critic_w.len()
            + self.v_actor_w.len()
            + self.v_actor_b.len();
        vec_floats * 4 + 24 * 4 + 96
    }
}

/// Буферы чувствительностей RTRL + мгновенные и пригодные градиенты (выделяются ОДИН раз, 0 аллокаций на шаге).
#[derive(Clone)]
pub struct UnifiedRtrlWorkspace {
    pub eff_in: usize,
    pub b0_dim: usize,
    pub num_actions: usize,
    pub feat_dim: usize,
    // Состояния полос:
    pub b0_state: Vec<f32>,
    pub b1_state: [f32; UNIFIED_B1_DIM],
    pub b2_state: [f32; UNIFIED_B2_DIM],
    // RMSNorm промежуточные скаляры и векторы:
    pub norm_inv: [f32; 4],
    pub gate_val: [f32; 4],
    pub gate_deriv: [f32; 4],
    pub feat_normed: Vec<f32>,
    pub feat_gated: Vec<f32>,
    // RTRL sensitivity d(u)/d(theta) для Полосы 1 (16 x 2D):
    pub s_b1_alpha: [f32; UNIFIED_B1_DIM],
    pub s_b1_omega: [f32; UNIFIED_B1_DIM],
    pub s_b1_beta: Vec<f32>,
    pub s_b1_bdelta: Vec<f32>,
    pub s_b1_qfb: Vec<f32>,
    pub s_b1_inw: Vec<f32>,
    pub s_b1_selw: Vec<f32>,
    // RTRL sensitivity d(u)/d(theta) для Полосы 2 (8 x 3D):
    pub s_b2_alpha: [f32; UNIFIED_B2_DIM],
    pub s_b2_omega: [f32; UNIFIED_B2_DIM],
    pub s_b2_beta: Vec<f32>,
    pub s_b2_bdelta: Vec<f32>,
    pub s_b2_qfb: Vec<f32>,
    pub s_b2_inw: Vec<f32>,
    pub s_b2_selw: Vec<f32>,
    // Выходы шага и рабочие градиенты:
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub psis: Vec<f32>,
    pub critic_val: f32,
    pub grad_feat: Vec<f32>,
    pub grad_b1_u: [f32; UNIFIED_B1_DIM],
    pub grad_b2_u: [f32; UNIFIED_B2_DIM],
    pub g_gates: [f32; 4],
    // Поэлементные мгновенные градиенты для RTRL:
    pub g_b1_alpha: [f32; UNIFIED_B1_SLOTS],
    pub g_b1_omega: [f32; UNIFIED_B1_SLOTS],
    pub g_b1_beta: Vec<f32>,
    pub g_b1_bdelta: Vec<f32>,
    pub g_b1_qfb: Vec<f32>,
    pub g_b1_inw: Vec<f32>,
    pub g_b1_selw: Vec<f32>,
    pub g_b2_alpha: [f32; UNIFIED_B2_SLOTS],
    pub g_b2_omega: [f32; UNIFIED_B2_SLOTS],
    pub g_b2_beta: Vec<f32>,
    pub g_b2_bdelta: Vec<f32>,
    pub g_b2_qfb: Vec<f32>,
    pub g_b2_inw: Vec<f32>,
    pub g_b2_selw: Vec<f32>,
    // Eligibility traces для RL:
    pub e_actor_w: Vec<f32>,
    pub e_actor_b: Vec<f32>,
    pub e_critic_w: Vec<f32>,
    pub e_critic_b: f32,
    pub e_gates: [f32; 4],
    pub e_b1_alpha: [f32; UNIFIED_B1_SLOTS],
    pub e_b1_omega: [f32; UNIFIED_B1_SLOTS],
    pub e_b1_beta: Vec<f32>,
    pub e_b1_bdelta: Vec<f32>,
    pub e_b1_qfb: Vec<f32>,
    pub e_b1_inw: Vec<f32>,
    pub e_b1_selw: Vec<f32>,
    pub e_b2_alpha: [f32; UNIFIED_B2_SLOTS],
    pub e_b2_omega: [f32; UNIFIED_B2_SLOTS],
    pub e_b2_beta: Vec<f32>,
    pub e_b2_bdelta: Vec<f32>,
    pub e_b2_qfb: Vec<f32>,
    pub e_b2_inw: Vec<f32>,
    pub e_b2_selw: Vec<f32>,
}

#[axiom_zero]
impl UnifiedRtrlWorkspace {
    pub fn new(cfg: &UnifiedCayleyConfig, eff_in: usize, num_actions: usize) -> Self {
        let ein = eff_in.clamp(1, UNIFIED_MAX_EFF_IN);
        let na = num_actions.clamp(1, 32);
        let b0_dim = if cfg.use_band0 { UNIFIED_B0_DIM } else { 0 };
        let feat_dim = ein + b0_dim + UNIFIED_B1_DIM + UNIFIED_B2_DIM;
        let sel = cfg.enable_selectivity;
        let fb = sel && cfg.enable_local_feedback;
        Self {
            eff_in: ein,
            b0_dim,
            num_actions: na,
            feat_dim,
            b0_state: if b0_dim > 0 { vec![0.0_f32; b0_dim] } else { Vec::new() },
            b1_state: [0.0_f32; UNIFIED_B1_DIM],
            b2_state: [0.0_f32; UNIFIED_B2_DIM],
            norm_inv: [1.0_f32; 4],
            gate_val: [0.5_f32; 4],
            gate_deriv: [0.0_f32; 4],
            feat_normed: vec![0.0_f32; feat_dim],
            feat_gated: vec![0.0_f32; feat_dim],
            s_b1_alpha: [0.0_f32; UNIFIED_B1_DIM],
            s_b1_omega: [0.0_f32; UNIFIED_B1_DIM],
            s_b1_beta: if sel { vec![0.0_f32; UNIFIED_B1_DIM] } else { Vec::new() },
            s_b1_bdelta: if sel { vec![0.0_f32; UNIFIED_B1_DIM] } else { Vec::new() },
            s_b1_qfb: if fb { vec![0.0_f32; UNIFIED_B1_DIM * 2] } else { Vec::new() },
            s_b1_inw: vec![0.0_f32; UNIFIED_B1_SLOTS * ein * 2],
            s_b1_selw: if sel { vec![0.0_f32; UNIFIED_B1_SLOTS * ein * 2] } else { Vec::new() },
            s_b2_alpha: [0.0_f32; UNIFIED_B2_DIM],
            s_b2_omega: [0.0_f32; UNIFIED_B2_DIM],
            s_b2_beta: if sel { vec![0.0_f32; UNIFIED_B2_DIM] } else { Vec::new() },
            s_b2_bdelta: if sel { vec![0.0_f32; UNIFIED_B2_DIM] } else { Vec::new() },
            s_b2_qfb: if fb { vec![0.0_f32; UNIFIED_B2_DIM * 3] } else { Vec::new() },
            s_b2_inw: vec![0.0_f32; UNIFIED_B2_SLOTS * ein * 3],
            s_b2_selw: if sel { vec![0.0_f32; UNIFIED_B2_SLOTS * ein * 3] } else { Vec::new() },
            logits: vec![0.0_f32; na],
            probs: vec![0.0_f32; na],
            psis: vec![0.0_f32; na],
            critic_val: 0.0,
            grad_feat: vec![0.0_f32; feat_dim],
            grad_b1_u: [0.0_f32; UNIFIED_B1_DIM],
            grad_b2_u: [0.0_f32; UNIFIED_B2_DIM],
            g_gates: [0.0_f32; 4],
            g_b1_alpha: [0.0_f32; UNIFIED_B1_SLOTS],
            g_b1_omega: [0.0_f32; UNIFIED_B1_SLOTS],
            g_b1_beta: if sel { vec![0.0_f32; UNIFIED_B1_SLOTS] } else { Vec::new() },
            g_b1_bdelta: if sel { vec![0.0_f32; UNIFIED_B1_SLOTS] } else { Vec::new() },
            g_b1_qfb: if fb { vec![0.0_f32; UNIFIED_B1_DIM] } else { Vec::new() },
            g_b1_inw: vec![0.0_f32; UNIFIED_B1_SLOTS * ein],
            g_b1_selw: if sel { vec![0.0_f32; UNIFIED_B1_SLOTS * ein] } else { Vec::new() },
            g_b2_alpha: [0.0_f32; UNIFIED_B2_SLOTS],
            g_b2_omega: [0.0_f32; UNIFIED_B2_SLOTS],
            g_b2_beta: if sel { vec![0.0_f32; UNIFIED_B2_SLOTS] } else { Vec::new() },
            g_b2_bdelta: if sel { vec![0.0_f32; UNIFIED_B2_SLOTS] } else { Vec::new() },
            g_b2_qfb: if fb { vec![0.0_f32; UNIFIED_B2_DIM] } else { Vec::new() },
            g_b2_inw: vec![0.0_f32; UNIFIED_B2_SLOTS * ein],
            g_b2_selw: if sel { vec![0.0_f32; UNIFIED_B2_SLOTS * ein] } else { Vec::new() },
            e_actor_w: vec![0.0_f32; na * feat_dim],
            e_actor_b: vec![0.0_f32; na],
            e_critic_w: vec![0.0_f32; feat_dim],
            e_critic_b: 0.0,
            e_gates: [0.0_f32; 4],
            e_b1_alpha: [0.0_f32; UNIFIED_B1_SLOTS],
            e_b1_omega: [0.0_f32; UNIFIED_B1_SLOTS],
            e_b1_beta: if sel { vec![0.0_f32; UNIFIED_B1_SLOTS] } else { Vec::new() },
            e_b1_bdelta: if sel { vec![0.0_f32; UNIFIED_B1_SLOTS] } else { Vec::new() },
            e_b1_qfb: if fb { vec![0.0_f32; UNIFIED_B1_DIM] } else { Vec::new() },
            e_b1_inw: vec![0.0_f32; UNIFIED_B1_SLOTS * ein],
            e_b1_selw: if sel { vec![0.0_f32; UNIFIED_B1_SLOTS * ein] } else { Vec::new() },
            e_b2_alpha: [0.0_f32; UNIFIED_B2_SLOTS],
            e_b2_omega: [0.0_f32; UNIFIED_B2_SLOTS],
            e_b2_beta: if sel { vec![0.0_f32; UNIFIED_B2_SLOTS] } else { Vec::new() },
            e_b2_bdelta: if sel { vec![0.0_f32; UNIFIED_B2_SLOTS] } else { Vec::new() },
            e_b2_qfb: if fb { vec![0.0_f32; UNIFIED_B2_DIM] } else { Vec::new() },
            e_b2_inw: vec![0.0_f32; UNIFIED_B2_SLOTS * ein],
            e_b2_selw: if sel { vec![0.0_f32; UNIFIED_B2_SLOTS * ein] } else { Vec::new() },
        }
    }

    pub fn clear_state_and_traces(&mut self) {
        self.b0_state.fill(0.0);
        self.b1_state = [0.0_f32; UNIFIED_B1_DIM];
        self.b2_state = [0.0_f32; UNIFIED_B2_DIM];
        self.s_b1_alpha = [0.0_f32; UNIFIED_B1_DIM];
        self.s_b1_omega = [0.0_f32; UNIFIED_B1_DIM];
        self.s_b1_beta.fill(0.0);
        self.s_b1_bdelta.fill(0.0);
        self.s_b1_qfb.fill(0.0);
        self.s_b1_inw.fill(0.0);
        self.s_b1_selw.fill(0.0);
        self.s_b2_alpha = [0.0_f32; UNIFIED_B2_DIM];
        self.s_b2_omega = [0.0_f32; UNIFIED_B2_DIM];
        self.s_b2_beta.fill(0.0);
        self.s_b2_bdelta.fill(0.0);
        self.s_b2_qfb.fill(0.0);
        self.s_b2_inw.fill(0.0);
        self.s_b2_selw.fill(0.0);
        self.e_actor_w.fill(0.0);
        self.e_actor_b.fill(0.0);
        self.e_critic_w.fill(0.0);
        self.e_critic_b = 0.0;
        self.e_gates = [0.0_f32; 4];
        self.e_b1_alpha = [0.0_f32; UNIFIED_B1_SLOTS];
        self.e_b1_omega = [0.0_f32; UNIFIED_B1_SLOTS];
        self.e_b1_beta.fill(0.0);
        self.e_b1_bdelta.fill(0.0);
        self.e_b1_qfb.fill(0.0);
        self.e_b1_inw.fill(0.0);
        self.e_b1_selw.fill(0.0);
        self.e_b2_alpha = [0.0_f32; UNIFIED_B2_SLOTS];
        self.e_b2_omega = [0.0_f32; UNIFIED_B2_SLOTS];
        self.e_b2_beta.fill(0.0);
        self.e_b2_bdelta.fill(0.0);
        self.e_b2_qfb.fill(0.0);
        self.e_b2_inw.fill(0.0);
        self.e_b2_selw.fill(0.0);
    }

    pub fn decay_eligibility(&mut self, decay: f32) {
        if decay <= 1e-6 {
            self.e_actor_w.fill(0.0);
            self.e_actor_b.fill(0.0);
            self.e_critic_w.fill(0.0);
            self.e_critic_b = 0.0;
            self.e_gates = [0.0_f32; 4];
            self.e_b1_alpha = [0.0_f32; UNIFIED_B1_SLOTS];
            self.e_b1_omega = [0.0_f32; UNIFIED_B1_SLOTS];
            self.e_b1_beta.fill(0.0);
            self.e_b1_bdelta.fill(0.0);
            self.e_b1_qfb.fill(0.0);
            self.e_b1_inw.fill(0.0);
            self.e_b1_selw.fill(0.0);
            self.e_b2_alpha = [0.0_f32; UNIFIED_B2_SLOTS];
            self.e_b2_omega = [0.0_f32; UNIFIED_B2_SLOTS];
            self.e_b2_beta.fill(0.0);
            self.e_b2_bdelta.fill(0.0);
            self.e_b2_qfb.fill(0.0);
            self.e_b2_inw.fill(0.0);
            self.e_b2_selw.fill(0.0);
            return;
        }
        fast_scale(&mut self.e_actor_w, decay);
        fast_scale(&mut self.e_actor_b, decay);
        fast_scale(&mut self.e_critic_w, decay);
        self.e_critic_b = sanitize_f32(self.e_critic_b * decay);
        for v in self.e_gates.iter_mut() { *v = sanitize_f32(*v * decay); }
        for v in self.e_b1_alpha.iter_mut() { *v = sanitize_f32(*v * decay); }
        for v in self.e_b1_omega.iter_mut() { *v = sanitize_f32(*v * decay); }
        fast_scale(&mut self.e_b1_beta, decay);
        fast_scale(&mut self.e_b1_bdelta, decay);
        fast_scale(&mut self.e_b1_qfb, decay);
        fast_scale(&mut self.e_b1_inw, decay);
        fast_scale(&mut self.e_b1_selw, decay);
        for v in self.e_b2_alpha.iter_mut() { *v = sanitize_f32(*v * decay); }
        for v in self.e_b2_omega.iter_mut() { *v = sanitize_f32(*v * decay); }
        fast_scale(&mut self.e_b2_beta, decay);
        fast_scale(&mut self.e_b2_bdelta, decay);
        fast_scale(&mut self.e_b2_qfb, decay);
        fast_scale(&mut self.e_b2_inw, decay);
        fast_scale(&mut self.e_b2_selw, decay);
    }

    pub fn scrub_seu(&mut self) -> u64 {
        let mut faults = 0u64;
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b0_state, 128.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b1_state, 128.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.b2_state, 128.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.s_b1_alpha, 64.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.s_b1_omega, 64.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.s_b1_beta, 64.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.s_b1_bdelta, 64.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.s_b1_qfb, 64.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.s_b1_inw, 64.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.s_b1_selw, 64.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.s_b2_alpha, 64.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.s_b2_omega, 64.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.s_b2_beta, 64.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.s_b2_bdelta, 64.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.s_b2_qfb, 64.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.s_b2_inw, 64.0, 0.0));
        faults = faults.saturating_add(scrub_slice_seu(&mut self.s_b2_selw, 64.0, 0.0));
        faults
    }

    pub fn memory_bytes(&self) -> usize {
        let vfloats = self.b0_state.len()
            + self.feat_normed.len()
            + self.feat_gated.len()
            + self.s_b1_beta.len()
            + self.s_b1_bdelta.len()
            + self.s_b1_qfb.len()
            + self.s_b1_inw.len()
            + self.s_b1_selw.len()
            + self.s_b2_beta.len()
            + self.s_b2_bdelta.len()
            + self.s_b2_qfb.len()
            + self.s_b2_inw.len()
            + self.s_b2_selw.len()
            + self.logits.len()
            + self.probs.len()
            + self.psis.len()
            + self.grad_feat.len()
            + self.g_b1_beta.len()
            + self.g_b1_bdelta.len()
            + self.g_b1_qfb.len()
            + self.g_b1_inw.len()
            + self.g_b1_selw.len()
            + self.g_b2_beta.len()
            + self.g_b2_bdelta.len()
            + self.g_b2_qfb.len()
            + self.g_b2_inw.len()
            + self.g_b2_selw.len()
            + self.e_actor_w.len()
            + self.e_actor_b.len()
            + self.e_critic_w.len()
            + self.e_b1_beta.len()
            + self.e_b1_bdelta.len()
            + self.e_b1_qfb.len()
            + self.e_b1_inw.len()
            + self.e_b1_selw.len()
            + self.e_b2_beta.len()
            + self.e_b2_bdelta.len()
            + self.e_b2_qfb.len()
            + self.e_b2_inw.len()
            + self.e_b2_selw.len();
        let fixed_floats = UNIFIED_B1_DIM * 4
            + UNIFIED_B2_DIM * 4
            + UNIFIED_B1_SLOTS * 4
            + UNIFIED_B2_SLOTS * 4
            + 24;
        (vfloats + fixed_floats) * 4
    }
}

/// Обновление 4-скалярного полосового рационального мета-шага IDBD (`step131`, 0% `exp`).
#[axiom_zero]
#[inline]
fn update_band_idbd(
    beta_arr: &mut [f32; 4],
    h_arr: &mut [f32; 4],
    band: usize,
    mean_u: f32,
    mean_f2: f32,
    base_lr: f32,
    idbd_lr: f32,
) {
    let b_old = total_get_copy(beta_arr, band, 0.0);
    let h_old = total_get_copy(h_arr, band, 0.0);
    let (gp, _) = rational_pos_and_deriv(b_old);
    let step_v = sanitize_f32(base_lr * gp);
    let new_b = sanitize_f32(b_old + idbd_lr * mean_u * h_old).clamp(-3.5, 1.5);
    let keep_h = sanitize_f32(1.0 - step_v * mean_f2).clamp(0.0, 1.0);
    let new_h = sanitize_f32(h_old * keep_h + step_v * mean_u).clamp(-2.0, 2.0);
    let _ = total_set(beta_arr, band, new_b);
    let _ = total_set(h_arr, band, new_h);
}

#[derive(Clone)]
pub struct UnifiedCayleyCore {
    pub cfg: UnifiedCayleyConfig,
    pub in_cond: CausalInputConditioner,
    pub band0: Band0Clock,
    pub slots: Vec<UnifiedRegimeSlot>,
    pub active_slot: usize,
    pub ws: UnifiedRtrlWorkspace,
    pub rng: u64,
    pub total_episodes: usize,
    pub seu_faults_scrubbed: u64,
    pub last_action: usize,
    pub last_critic_val: f32,
    pub pending_reward: f32,
    pub has_pending_step: bool,
    pub has_pending_reward: bool,
    pub last_surp: f32,
    pub probe_eps_left: usize,
    pub probe_score_sum: f32,
    pub probe_score_cnt: usize,
    pub probe_origin_slot: usize,
    pub bank_switches: usize,
}

#[axiom_zero]
impl UnifiedCayleyCore {
    pub fn new(cfg: UnifiedCayleyConfig) -> Self {
        let in_cond = CausalInputConditioner::new(cfg.raw_in_dim, cfg.use_delta_channel, cfg.delta_scale);
        let eff_in = in_cond.eff_dim;
        let na = cfg.num_actions.clamp(1, 32);
        let band0 = Band0Clock::new(eff_in, cfg.seed, cfg.use_band0);
        let nb = cfg.max_banks.clamp(1, UNIFIED_MAX_BANKS);
        let mut slots = Vec::with_capacity(nb);
        for bi in 0..nb {
            slots.push(UnifiedRegimeSlot::new(&cfg, eff_in, bi));
        }
        let ws = UnifiedRtrlWorkspace::new(&cfg, eff_in, na);
        Self {
            cfg,
            in_cond,
            band0,
            slots,
            active_slot: 0,
            ws,
            rng: splitmix64(cfg.seed ^ 0xA1E7_5EED_0000_0001),
            total_episodes: 0,
            seu_faults_scrubbed: 0,
            last_action: 0,
            last_critic_val: 0.0,
            pending_reward: 0.0,
            has_pending_step: false,
            has_pending_reward: false,
            last_surp: 1.0,
            probe_eps_left: 0,
            probe_score_sum: 0.0,
            probe_score_cnt: 0,
            probe_origin_slot: 0,
            bank_switches: 0,
        }
    }

    #[inline]
    fn backbone_idx(&self, active: usize) -> usize {
        if self.cfg.shared_backbone { 0 } else { active }
    }

    pub fn start_episode(&mut self) {
        self.in_cond.reset_episode_delta();
        self.ws.clear_state_and_traces();
        self.pending_reward = 0.0;
        self.has_pending_step = false;
        self.has_pending_reward = false;
        self.last_critic_val = 0.0;
        self.last_action = 0;
    }

    pub fn inject_weight_bit30_seu(&mut self, slot_idx: usize) {
        let bb_idx = self.backbone_idx(slot_idx);
        if let Some(bb) = self.slots.get_mut(bb_idx) {
            if let Some(v) = bb.b1_alpha.get_mut(0) {
                *v = f32::from_bits(v.to_bits() ^ (1u32 << 30));
            }
            if let Some(v) = bb.b2_in_w.get_mut(0) {
                *v = f32::from_bits(v.to_bits() ^ (1u32 << 30));
            }
        }
        if let Some(slot) = self.slots.get_mut(slot_idx) {
            if let Some(v) = slot.actor_w.get_mut(0) {
                *v = f32::from_bits(v.to_bits() ^ (1u32 << 30));
            }
        }
        if let Some(v) = self.ws.b1_state.get_mut(0) {
            *v = f32::from_bits(v.to_bits() ^ (1u32 << 30));
        }
    }

    pub fn scrub_all_seu(&mut self) -> u64 {
        let mut f = self.ws.scrub_seu();
        for s in self.slots.iter_mut() {
            f = f.saturating_add(s.scrub_seu());
        }
        self.seu_faults_scrubbed = self.seu_faults_scrubbed.saturating_add(f);
        f
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        let mut sbytes = 0usize;
        for s in self.slots.iter() {
            sbytes = sbytes.saturating_add(s.memory_bytes());
        }
        self.in_cond.memory_bytes()
            + self.band0.memory_bytes()
            + sbytes
            + self.ws.memory_bytes()
            + 192
    }

    /// Продвигает вход, Полосы `B0 (опц.), B1, B2` и точные локальные трассы RTRL, вычисляет RMSNorm,
    /// полосовые вентили, логиты/вероятности и критик (без аллокаций).
    pub fn forward_and_rtrl_traces(&mut self, raw_x: &[f32], learn_norm: bool, update_traces: bool, seu_guard: bool) {
        if seu_guard {
            let _ = self.scrub_all_seu();
        }
        self.in_cond.step_transform(raw_x, learn_norm);
        if self.band0.enabled {
            self.band0.step_into(&self.in_cond.out_feat, &mut self.ws.b0_state);
        }

        let ein = self.ws.eff_in;
        let b0_dim = self.ws.b0_dim;
        let exact = self.cfg.exact_inv_sqrt;
        let sel_on = self.cfg.enable_selectivity;
        let fb_on = sel_on && self.cfg.enable_local_feedback;
        let min_a2 = self.cfg.min_alpha_sq;

        let active = self.active_slot.min(self.slots.len().saturating_sub(1));
        let bb_idx = self.backbone_idx(active);

        if let Some(bb) = self.slots.get(bb_idx) {
            // ---------- ПОЛОСА 1: 16 x 2D Master-Eq Cayley ----------
            for k in 0..UNIFIED_B1_SLOTS {
                let u_off = k * 2;
                let u0 = total_get_copy(&self.ws.b1_state, u_off, 0.0);
                let u1 = total_get_copy(&self.ws.b1_state, u_off + 1, 0.0);

                let a_k = total_get_copy(&bb.b1_alpha, k, 0.2);
                let w_k = total_get_copy(&bb.b1_omega, k, 0.3);
                let woff = k * ein;
                let mut m_in = 0.0_f32;
                if let Some(ws) = bb.b1_in_w.get(woff..woff + ein) {
                    m_in = sanitize_bounded_f32(fast_dot(ws, &self.in_cond.out_feat), 32.0);
                }

                let (delta, d_delta_darg, z_pre, beta_k, q0, q1, inv_r, inv_r3, uhat0, uhat1) = if sel_on {
                    let beta_k = total_get_copy(&bb.b1_beta, k, 0.0);
                    let b_d = total_get_copy(&bb.b1_bdelta, k, 0.0);
                    let mut sel_in = 0.0_f32;
                    if let Some(sw) = bb.b1_sel_w.get(woff..woff + ein) {
                        sel_in = sanitize_bounded_f32(fast_dot(sw, &self.in_cond.out_feat), 16.0);
                    }
                    let (q0, q1, inv_r, inv_r3, uhat0, uhat1, fb_term) = if fb_on {
                        let q0 = total_get_copy(&bb.b1_qfb, u_off, 0.0);
                        let q1 = total_get_copy(&bb.b1_qfb, u_off + 1, 0.0);
                        let r2 = sanitize_f32(u0 * u0 + u1 * u1 + 1.0);
                        let inv_r = inv_sqrt_select(r2, exact);
                        let inv_r3 = inv_r * inv_r * inv_r;
                        let uhat0 = u0 * inv_r;
                        let uhat1 = u1 * inv_r;
                        (q0, q1, inv_r, inv_r3, uhat0, uhat1, q0 * uhat0 + q1 * uhat1)
                    } else {
                        (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
                    };
                    let z = sanitize_bounded_f32(b_d + sel_in + fb_term, 10.0);
                    let arg = sanitize_bounded_f32(beta_k * z, 10.0);
                    let (del, ddel) = rational_pos_and_deriv(arg);
                    (del, ddel, z, beta_k, q0, q1, inv_r, inv_r3, uhat0, uhat1)
                } else {
                    (1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
                };

                let eff_a2 = a_k * a_k + min_a2;
                let lam = total_div_unit_shifted_f32(1.0, delta * eff_a2);
                let g_in = total_div_unit_shifted_f32(delta, 0.5 * delta);
                let theta = sanitize_bounded_f32(delta * w_k, 16.0);
                let t = 0.5 * theta;
                let inv_d = total_div_unit_shifted_f32(1.0, t * t);
                let c = (1.0 - t * t) * inv_d;
                let s = 2.0 * t * inv_d;

                let ru0 = sanitize_f32(c * u0 - s * u1);
                let ru1 = sanitize_f32(s * u0 + c * u1);
                let u_next0 = sanitize_bounded_f32(lam * ru0 + g_in * m_in, 128.0);
                let u_next1 = sanitize_bounded_f32(lam * ru1, 128.0);

                if update_traces {
                    let d_lam_da = -2.0 * delta * a_k * lam * lam;
                    let d_u_da0 = d_lam_da * ru0;
                    let d_u_da1 = d_lam_da * ru1;

                    let d_ru_dth0 = -ru1 * inv_d;
                    let d_ru_dth1 = ru0 * inv_d;
                    let d_u_dw0 = lam * delta * d_ru_dth0;
                    let d_u_dw1 = lam * delta * d_ru_dth1;

                    let mut j00 = lam * c;
                    let mut j01 = -lam * s;
                    let mut j10 = lam * s;
                    let mut j11 = lam * c;
                    let mut d_u_ddel0 = 0.0_f32;
                    let mut d_u_ddel1 = 0.0_f32;

                    if sel_on {
                        let d_lam_ddel = -eff_a2 * lam * lam;
                        let d_gin_ddel = total_div_f32(1.0, (1.0 + 0.5 * delta) * (1.0 + 0.5 * delta), 0.0);
                        d_u_ddel0 = d_lam_ddel * ru0 + lam * w_k * d_ru_dth0 + d_gin_ddel * m_in;
                        d_u_ddel1 = d_lam_ddel * ru1 + lam * w_k * d_ru_dth1;
                        if fb_on {
                            let scale_fb = d_delta_darg * beta_k;
                            let duh00 = inv_r - u0 * u0 * inv_r3;
                            let duh01 = -u0 * u1 * inv_r3;
                            let duh10 = -u1 * u0 * inv_r3;
                            let duh11 = inv_r - u1 * u1 * inv_r3;
                            let dg_du0 = scale_fb * (q0 * duh00 + q1 * duh10);
                            let dg_du1 = scale_fb * (q0 * duh01 + q1 * duh11);
                            j00 = sanitize_f32(j00 + d_u_ddel0 * dg_du0);
                            j01 = sanitize_f32(j01 + d_u_ddel0 * dg_du1);
                            j10 = sanitize_f32(j10 + d_u_ddel1 * dg_du0);
                            j11 = sanitize_f32(j11 + d_u_ddel1 * dg_du1);
                        }
                    }

                    macro_rules! step_tr2 {
                        ($arr:expr, $idx0:expr, $add0:expr, $add1:expr) => {{
                            let p0 = total_get_copy(&$arr, $idx0, 0.0);
                            let p1 = total_get_copy(&$arr, $idx0 + 1, 0.0);
                            let n0 = sanitize_bounded_f32(j00 * p0 + j01 * p1 + $add0, 64.0);
                            let n1 = sanitize_bounded_f32(j10 * p0 + j11 * p1 + $add1, 64.0);
                            let _ = total_set(&mut $arr, $idx0, n0);
                            let _ = total_set(&mut $arr, $idx0 + 1, n1);
                        }};
                    }

                    step_tr2!(self.ws.s_b1_alpha, u_off, d_u_da0, d_u_da1);
                    step_tr2!(self.ws.s_b1_omega, u_off, d_u_dw0, d_u_dw1);

                    for j in 0..ein {
                        let xj = total_get_copy(&self.in_cond.out_feat, j, 0.0);
                        let tr_off = (woff + j) * 2;
                        step_tr2!(self.ws.s_b1_inw, tr_off, g_in * xj, 0.0);
                        if sel_on {
                            let ddel_ds = d_delta_darg * beta_k * xj;
                            step_tr2!(self.ws.s_b1_selw, tr_off, d_u_ddel0 * ddel_ds, d_u_ddel1 * ddel_ds);
                        }
                    }

                    if sel_on {
                        let ddel_dbeta = d_delta_darg * z_pre;
                        step_tr2!(self.ws.s_b1_beta, u_off, d_u_ddel0 * ddel_dbeta, d_u_ddel1 * ddel_dbeta);
                        let ddel_dbd = d_delta_darg * beta_k;
                        step_tr2!(self.ws.s_b1_bdelta, u_off, d_u_ddel0 * ddel_dbd, d_u_ddel1 * ddel_dbd);
                        if fb_on {
                            let ddel_dq0 = d_delta_darg * beta_k * uhat0;
                            let ddel_dq1 = d_delta_darg * beta_k * uhat1;
                            let qtr_off = u_off * 2;
                            step_tr2!(self.ws.s_b1_qfb, qtr_off, d_u_ddel0 * ddel_dq0, d_u_ddel1 * ddel_dq0);
                            step_tr2!(self.ws.s_b1_qfb, qtr_off + 2, d_u_ddel0 * ddel_dq1, d_u_ddel1 * ddel_dq1);
                        }
                    }
                }

                let _ = total_set(&mut self.ws.b1_state, u_off, u_next0);
                let _ = total_set(&mut self.ws.b1_state, u_off + 1, u_next1);
            }

            // ---------- ПОЛОСА 2: 8 x 3D Non-Abelian Cayley SO(3)^8 ----------
            for s in 0..UNIFIED_B2_SLOTS {
                let off = s * 3;
                let u = (
                    total_get_copy(&self.ws.b2_state, off, 0.0),
                    total_get_copy(&self.ws.b2_state, off + 1, 0.0),
                    total_get_copy(&self.ws.b2_state, off + 2, 0.0),
                );
                let e = (
                    total_get_copy(&bb.b2_axis, off, 1.0),
                    total_get_copy(&bb.b2_axis, off + 1, 0.0),
                    total_get_copy(&bb.b2_axis, off + 2, 0.0),
                );
                let d_dir = (
                    total_get_copy(&bb.b2_in_dir, off, 0.0),
                    total_get_copy(&bb.b2_in_dir, off + 1, 1.0),
                    total_get_copy(&bb.b2_in_dir, off + 2, 0.0),
                );

                let a_s = total_get_copy(&bb.b2_alpha, s, 0.2);
                let w_s = total_get_copy(&bb.b2_omega, s, 0.3);
                let woff = s * ein;
                let mut m_in = 0.0_f32;
                if let Some(ws) = bb.b2_in_w.get(woff..woff + ein) {
                    m_in = sanitize_bounded_f32(fast_dot(ws, &self.in_cond.out_feat), 32.0);
                }

                let (delta, d_delta_darg, z_pre, beta_s, qfb, inv_r, inv_r3, uhat) = if sel_on {
                    let beta_s = total_get_copy(&bb.b2_beta, s, 0.0);
                    let b_d = total_get_copy(&bb.b2_bdelta, s, 0.0);
                    let mut sel_in = 0.0_f32;
                    if let Some(sw) = bb.b2_sel_w.get(woff..woff + ein) {
                        sel_in = sanitize_bounded_f32(fast_dot(sw, &self.in_cond.out_feat), 16.0);
                    }
                    let (qfb, inv_r, inv_r3, uhat, fb_term) = if fb_on {
                        let qfb = (
                            total_get_copy(&bb.b2_qfb, off, 0.0),
                            total_get_copy(&bb.b2_qfb, off + 1, 0.0),
                            total_get_copy(&bb.b2_qfb, off + 2, 0.0),
                        );
                        let r2 = sanitize_f32(u.0 * u.0 + u.1 * u.1 + u.2 * u.2 + 1.0);
                        let inv_r = inv_sqrt_select(r2, exact);
                        let inv_r3 = inv_r * inv_r * inv_r;
                        let uhat = (u.0 * inv_r, u.1 * inv_r, u.2 * inv_r);
                        let fb = qfb.0 * uhat.0 + qfb.1 * uhat.1 + qfb.2 * uhat.2;
                        (qfb, inv_r, inv_r3, uhat, fb)
                    } else {
                        ((0.0, 0.0, 0.0), 1.0, 1.0, (0.0, 0.0, 0.0), 0.0)
                    };
                    let z = sanitize_bounded_f32(b_d + sel_in + fb_term, 10.0);
                    let arg = sanitize_bounded_f32(beta_s * z, 10.0);
                    let (del, ddel) = rational_pos_and_deriv(arg);
                    (del, ddel, z, beta_s, qfb, inv_r, inv_r3, uhat)
                } else {
                    (1.0_f32, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), 1.0, 1.0, (0.0, 0.0, 0.0))
                };

                let eff_a2 = a_s * a_s + min_a2;
                let lam = total_div_unit_shifted_f32(1.0, delta * eff_a2);
                let g_in = total_div_unit_shifted_f32(delta, 0.5 * delta);
                let theta = sanitize_bounded_f32(delta * w_s, 16.0);
                let (ru, dr_dth, c1, c2) = rot3_step_and_deriv(e, theta, u);

                let un0 = sanitize_bounded_f32(lam * ru.0 + g_in * m_in * d_dir.0, 128.0);
                let un1 = sanitize_bounded_f32(lam * ru.1 + g_in * m_in * d_dir.1, 128.0);
                let un2 = sanitize_bounded_f32(lam * ru.2 + g_in * m_in * d_dir.2, 128.0);

                if update_traces {
                    let d_lam_da = -2.0 * delta * a_s * lam * lam;
                    let d_u_da = (d_lam_da * ru.0, d_lam_da * ru.1, d_lam_da * ru.2);
                    let d_u_dw = (lam * delta * dr_dth.0, lam * delta * dr_dth.1, lam * delta * dr_dth.2);

                    let mut d_u_ddel = (0.0_f32, 0.0_f32, 0.0_f32);
                    let mut dg_du0 = 0.0_f32;
                    let mut dg_du1 = 0.0_f32;
                    let mut dg_du2 = 0.0_f32;
                    if sel_on {
                        let d_lam_ddel = -eff_a2 * lam * lam;
                        let d_gin_ddel = total_div_f32(1.0, (1.0 + 0.5 * delta) * (1.0 + 0.5 * delta), 0.0);
                        d_u_ddel = (
                            d_lam_ddel * ru.0 + lam * w_s * dr_dth.0 + d_gin_ddel * m_in * d_dir.0,
                            d_lam_ddel * ru.1 + lam * w_s * dr_dth.1 + d_gin_ddel * m_in * d_dir.1,
                            d_lam_ddel * ru.2 + lam * w_s * dr_dth.2 + d_gin_ddel * m_in * d_dir.2,
                        );
                        if fb_on {
                            let scale_fb = d_delta_darg * beta_s;
                            let qu = qfb.0 * u.0 + qfb.1 * u.1 + qfb.2 * u.2;
                            dg_du0 = scale_fb * (qfb.0 * inv_r - u.0 * qu * inv_r3);
                            dg_du1 = scale_fb * (qfb.1 * inv_r - u.1 * qu * inv_r3);
                            dg_du2 = scale_fb * (qfb.2 * inv_r - u.2 * qu * inv_r3);
                        }
                    }

                    macro_rules! step_tr3 {
                        ($arr:expr, $base:expr, $add0:expr, $add1:expr, $add2:expr) => {{
                            let p = (
                                total_get_copy(&$arr, $base, 0.0),
                                total_get_copy(&$arr, $base + 1, 0.0),
                                total_get_copy(&$arr, $base + 2, 0.0),
                            );
                            let rp = rot3_apply(e, c1, c2, p);
                            let fb_proj = if fb_on {
                                dg_du0 * p.0 + dg_du1 * p.1 + dg_du2 * p.2
                            } else {
                                0.0
                            };
                            let _ = total_set(&mut $arr, $base, sanitize_bounded_f32(lam * rp.0 + d_u_ddel.0 * fb_proj + $add0, 64.0));
                            let _ = total_set(&mut $arr, $base + 1, sanitize_bounded_f32(lam * rp.1 + d_u_ddel.1 * fb_proj + $add1, 64.0));
                            let _ = total_set(&mut $arr, $base + 2, sanitize_bounded_f32(lam * rp.2 + d_u_ddel.2 * fb_proj + $add2, 64.0));
                        }};
                    }

                    step_tr3!(self.ws.s_b2_alpha, off, d_u_da.0, d_u_da.1, d_u_da.2);
                    step_tr3!(self.ws.s_b2_omega, off, d_u_dw.0, d_u_dw.1, d_u_dw.2);

                    for j in 0..ein {
                        let xj = total_get_copy(&self.in_cond.out_feat, j, 0.0);
                        let tr_off = (woff + j) * 3;
                        let gx = g_in * xj;
                        step_tr3!(self.ws.s_b2_inw, tr_off, gx * d_dir.0, gx * d_dir.1, gx * d_dir.2);
                        if sel_on {
                            let dds = d_delta_darg * beta_s * xj;
                            step_tr3!(self.ws.s_b2_selw, tr_off, d_u_ddel.0 * dds, d_u_ddel.1 * dds, d_u_ddel.2 * dds);
                        }
                    }

                    if sel_on {
                        let ddb = d_delta_darg * z_pre;
                        step_tr3!(self.ws.s_b2_beta, off, d_u_ddel.0 * ddb, d_u_ddel.1 * ddb, d_u_ddel.2 * ddb);
                        let ddbd = d_delta_darg * beta_s;
                        step_tr3!(self.ws.s_b2_bdelta, off, d_u_ddel.0 * ddbd, d_u_ddel.1 * ddbd, d_u_ddel.2 * ddbd);
                        if fb_on {
                            let ddq0 = d_delta_darg * beta_s * uhat.0;
                            step_tr3!(self.ws.s_b2_qfb, off * 3, d_u_ddel.0 * ddq0, d_u_ddel.1 * ddq0, d_u_ddel.2 * ddq0);
                            let ddq1 = d_delta_darg * beta_s * uhat.1;
                            step_tr3!(self.ws.s_b2_qfb, (off + 1) * 3, d_u_ddel.0 * ddq1, d_u_ddel.1 * ddq1, d_u_ddel.2 * ddq1);
                            let ddq2 = d_delta_darg * beta_s * uhat.2;
                            step_tr3!(self.ws.s_b2_qfb, (off + 2) * 3, d_u_ddel.0 * ddq2, d_u_ddel.1 * ddq2, d_u_ddel.2 * ddq2);
                        }
                    }
                }

                let _ = total_set(&mut self.ws.b2_state, off, un0);
                let _ = total_set(&mut self.ws.b2_state, off + 1, un1);
                let _ = total_set(&mut self.ws.b2_state, off + 2, un2);
            }
        }

        // ---------- Newton RMSNorm + Band Gates [skip, B0, B1, B2] + Головки из active слота ----------
        if let Some(slot) = self.slots.get(active) {
            let off_b0 = ein;
            let off_b1 = off_b0 + b0_dim;
            let off_b2 = off_b1 + UNIFIED_B1_DIM;
            let n0 = rmsnorm_band_into(&self.in_cond.out_feat, &mut self.ws.feat_normed, 0, 0.5, exact);
            let n1 = if b0_dim > 0 {
                rmsnorm_band_into(&self.ws.b0_state, &mut self.ws.feat_normed, off_b0, 0.05, exact)
            } else {
                1.0
            };
            let n2 = rmsnorm_band_into(&self.ws.b1_state, &mut self.ws.feat_normed, off_b1, 0.05, exact);
            let n3 = rmsnorm_band_into(&self.ws.b2_state, &mut self.ws.feat_normed, off_b2, 0.05, exact);
            let _ = total_set(&mut self.ws.norm_inv, 0, n0);
            let _ = total_set(&mut self.ws.norm_inv, 1, n1);
            let _ = total_set(&mut self.ws.norm_inv, 2, n2);
            let _ = total_set(&mut self.ws.norm_inv, 3, n3);

            for b in 0..4usize {
                let gl = total_get_copy(&slot.gate_logits, b, 0.0);
                let (gp, dgp) = rational_pos_and_deriv(gl);
                let _ = total_set(&mut self.ws.gate_val, b, 0.5 * gp);
                let _ = total_set(&mut self.ws.gate_deriv, b, 0.5 * dgp);
            }

            let g0 = total_get_copy(&self.ws.gate_val, 0, 0.5);
            let g1 = total_get_copy(&self.ws.gate_val, 1, 0.5);
            let g2 = total_get_copy(&self.ws.gate_val, 2, 0.5);
            let g3 = total_get_copy(&self.ws.gate_val, 3, 0.5);
            if let (Some(dst), Some(src)) = (self.ws.feat_gated.get_mut(0..ein), self.ws.feat_normed.get(0..ein)) {
                for (d, &s) in dst.iter_mut().zip(src.iter()) { *d = s * g0; }
            }
            if b0_dim > 0 {
                if let (Some(dst), Some(src)) = (self.ws.feat_gated.get_mut(off_b0..off_b1), self.ws.feat_normed.get(off_b0..off_b1)) {
                    for (d, &s) in dst.iter_mut().zip(src.iter()) { *d = s * g1; }
                }
            }
            if let (Some(dst), Some(src)) = (self.ws.feat_gated.get_mut(off_b1..off_b2), self.ws.feat_normed.get(off_b1..off_b2)) {
                for (d, &s) in dst.iter_mut().zip(src.iter()) { *d = s * g2; }
            }
            let fdim_all = off_b2 + UNIFIED_B2_DIM;
            if let (Some(dst), Some(src)) = (self.ws.feat_gated.get_mut(off_b2..fdim_all), self.ws.feat_normed.get(off_b2..fdim_all)) {
                for (d, &s) in dst.iter_mut().zip(src.iter()) { *d = s * g3; }
            }

            let na = slot.num_actions;
            let fdim = slot.feat_dim;
            for c in 0..na {
                let off = c * fdim;
                let mut dot = 0.0_f32;
                if let Some(wc) = slot.actor_w.get(off..off + fdim) {
                    dot = fast_dot(wc, &self.ws.feat_gated);
                }
                let lg = sanitize_bounded_f32(total_get_copy(&slot.actor_b, c, 0.0) + dot, 64.0);
                let _ = total_set(&mut self.ws.logits, c, lg);
            }
            rational_probs_p4_into(&self.ws.logits, &mut self.ws.probs, &mut self.ws.psis);

            let cv = sanitize_bounded_f32(slot.critic_b + fast_dot(&slot.critic_w, &self.ws.feat_gated), 64.0);
            self.ws.critic_val = cv;
        }
    }

    /// Обратный проход через головки, вентили полос, 2-шаговый Ньютон RMSNorm и проецирование на трассы RTRL.
    pub fn compute_instant_rtrl_grads(&mut self, logit_errs: &[f32], reg_err: f32) {
        let active = self.active_slot.min(self.slots.len().saturating_sub(1));
        let ein = self.ws.eff_in;
        let off_b0 = ein;
        let off_b1 = off_b0 + self.ws.b0_dim;
        let off_b2 = off_b1 + UNIFIED_B1_DIM;
        let sel_on = self.cfg.enable_selectivity;
        let fb_on = sel_on && self.cfg.enable_local_feedback;

        self.ws.grad_feat.fill(0.0);
        if let Some(slot) = self.slots.get(active) {
            let fdim = slot.feat_dim;
            for c in 0..slot.num_actions {
                let ec = sanitize_bounded_f32(total_get_copy(logit_errs, c, 0.0), 8.0);
                if ec.abs() > 1e-12 {
                    let off = c * fdim;
                    if let Some(wc) = slot.actor_w.get(off..off + fdim) {
                        fast_saxpy(&mut self.ws.grad_feat, wc, ec);
                    }
                }
            }
            let re = sanitize_bounded_f32(reg_err, 8.0);
            if re.abs() > 1e-12 {
                fast_saxpy(&mut self.ws.grad_feat, &slot.critic_w, re);
            }
        }

        let bounds = [(0usize, off_b0), (off_b0, off_b1), (off_b1, off_b2), (off_b2, self.ws.feat_dim)];
        for b in 0..4usize {
            let (lo, hi) = total_get_copy(&bounds, b, (0, 0));
            let dg = total_get_copy(&self.ws.gate_deriv, b, 0.0);
            let mut dot_g = 0.0_f32;
            for i in lo..hi {
                dot_g = sanitize_f32(
                    dot_g + total_get_copy(&self.ws.grad_feat, i, 0.0) * total_get_copy(&self.ws.feat_normed, i, 0.0),
                );
            }
            let _ = total_set(&mut self.ws.g_gates, b, sanitize_f32(dg * dot_g));
        }

        for b in 0..4usize {
            let (lo, hi) = total_get_copy(&bounds, b, (0, 0));
            let gv = total_get_copy(&self.ws.gate_val, b, 0.5);
            if let Some(sub) = self.ws.grad_feat.get_mut(lo..hi) {
                for p in sub.iter_mut() {
                    *p = sanitize_f32(*p * gv);
                }
            }
        }

        if let (Some(gn1), Some(n1)) = (self.ws.grad_feat.get(off_b1..off_b2), self.ws.feat_normed.get(off_b1..off_b2)) {
            rmsnorm_band_backward(gn1, n1, total_get_copy(&self.ws.norm_inv, 2, 1.0), &mut self.ws.grad_b1_u);
        }
        if let (Some(gn2), Some(n2)) = (self.ws.grad_feat.get(off_b2..self.ws.feat_dim), self.ws.feat_normed.get(off_b2..self.ws.feat_dim)) {
            rmsnorm_band_backward(gn2, n2, total_get_copy(&self.ws.norm_inv, 3, 1.0), &mut self.ws.grad_b2_u);
        }

        // Band 1
        for k in 0..UNIFIED_B1_SLOTS {
            let u_off = k * 2;
            let gb0 = total_get_copy(&self.ws.grad_b1_u, u_off, 0.0);
            let gb1 = total_get_copy(&self.ws.grad_b1_u, u_off + 1, 0.0);
            let _ = total_set(&mut self.ws.g_b1_alpha, k, gb0 * total_get_copy(&self.ws.s_b1_alpha, u_off, 0.0) + gb1 * total_get_copy(&self.ws.s_b1_alpha, u_off + 1, 0.0));
            let _ = total_set(&mut self.ws.g_b1_omega, k, gb0 * total_get_copy(&self.ws.s_b1_omega, u_off, 0.0) + gb1 * total_get_copy(&self.ws.s_b1_omega, u_off + 1, 0.0));
            let woff = k * ein;
            for j in 0..ein {
                let toff = (woff + j) * 2;
                let val_w = gb0 * total_get_copy(&self.ws.s_b1_inw, toff, 0.0) + gb1 * total_get_copy(&self.ws.s_b1_inw, toff + 1, 0.0);
                let _ = total_set(&mut self.ws.g_b1_inw, woff + j, val_w);
                if sel_on {
                    let val_s = gb0 * total_get_copy(&self.ws.s_b1_selw, toff, 0.0) + gb1 * total_get_copy(&self.ws.s_b1_selw, toff + 1, 0.0);
                    let _ = total_set(&mut self.ws.g_b1_selw, woff + j, val_s);
                }
            }
            if sel_on {
                let _ = total_set(&mut self.ws.g_b1_beta, k, gb0 * total_get_copy(&self.ws.s_b1_beta, u_off, 0.0) + gb1 * total_get_copy(&self.ws.s_b1_beta, u_off + 1, 0.0));
                let _ = total_set(&mut self.ws.g_b1_bdelta, k, gb0 * total_get_copy(&self.ws.s_b1_bdelta, u_off, 0.0) + gb1 * total_get_copy(&self.ws.s_b1_bdelta, u_off + 1, 0.0));
                if fb_on {
                    let qtr = u_off * 2;
                    let _ = total_set(&mut self.ws.g_b1_qfb, u_off, gb0 * total_get_copy(&self.ws.s_b1_qfb, qtr, 0.0) + gb1 * total_get_copy(&self.ws.s_b1_qfb, qtr + 1, 0.0));
                    let _ = total_set(&mut self.ws.g_b1_qfb, u_off + 1, gb0 * total_get_copy(&self.ws.s_b1_qfb, qtr + 2, 0.0) + gb1 * total_get_copy(&self.ws.s_b1_qfb, qtr + 3, 0.0));
                }
            }
        }

        // Band 2
        for s in 0..UNIFIED_B2_SLOTS {
            let off = s * 3;
            let gb0 = total_get_copy(&self.ws.grad_b2_u, off, 0.0);
            let gb1 = total_get_copy(&self.ws.grad_b2_u, off + 1, 0.0);
            let gb2 = total_get_copy(&self.ws.grad_b2_u, off + 2, 0.0);
            let dot3 = |arr: &[f32], b: usize| -> f32 {
                gb0 * total_get_copy(arr, b, 0.0) + gb1 * total_get_copy(arr, b + 1, 0.0) + gb2 * total_get_copy(arr, b + 2, 0.0)
            };
            let _ = total_set(&mut self.ws.g_b2_alpha, s, dot3(&self.ws.s_b2_alpha, off));
            let _ = total_set(&mut self.ws.g_b2_omega, s, dot3(&self.ws.s_b2_omega, off));
            let woff = s * ein;
            for j in 0..ein {
                let toff = (woff + j) * 3;
                let _ = total_set(&mut self.ws.g_b2_inw, woff + j, dot3(&self.ws.s_b2_inw, toff));
                if sel_on {
                    let _ = total_set(&mut self.ws.g_b2_selw, woff + j, dot3(&self.ws.s_b2_selw, toff));
                }
            }
            if sel_on {
                let _ = total_set(&mut self.ws.g_b2_beta, s, dot3(&self.ws.s_b2_beta, off));
                let _ = total_set(&mut self.ws.g_b2_bdelta, s, dot3(&self.ws.s_b2_bdelta, off));
                if fb_on {
                    for c_i in 0..3usize {
                        let qtr = (off + c_i) * 3;
                        let _ = total_set(&mut self.ws.g_b2_qfb, off + c_i, dot3(&self.ws.s_b2_qfb, qtr));
                    }
                }
            }
        }
    }

    fn apply_rmsprop_rtrl_scaled(&mut self, use_elig: bool, grad_scale: f32, lr_mult: f32) {
        let active = self.active_slot.min(self.slots.len().saturating_sub(1));
        let bb_idx = self.backbone_idx(active);
        let lr = sanitize_f32(self.cfg.rtrl_lr * lr_mult);
        let wd = self.cfg.weight_decay;
        let sel_on = self.cfg.enable_selectivity;
        let fb_on = sel_on && self.cfg.enable_local_feedback;
        let exact = self.cfg.exact_inv_sqrt;
        let u_lim = self.cfg.unitary_limit;
        let ein = self.ws.eff_in;

        // 1. Полосовые вентили принадлежат активному слоту режима
        if let Some(slot) = self.slots.get_mut(active) {
            for b in 0..4usize {
                let raw_g = if use_elig {
                    total_get_copy(&self.ws.e_gates, b, 0.0)
                } else {
                    total_get_copy(&self.ws.g_gates, b, 0.0)
                };
                if let (Some(p), Some(v)) = (slot.gate_logits.get_mut(b), slot.v_gates.get_mut(b)) {
                    rmsprop_step(p, v, raw_g * grad_scale, lr * 1.5, wd * 0.25, 4.0, exact);
                }
            }
        }

        // 2. Рекуррентный хребет B1+B2 обновляется в bb_idx (`0` при `shared_backbone`)
        if let Some(bb) = self.slots.get_mut(bb_idx) {
            for k in 0..UNIFIED_B1_SLOTS {
                if !u_lim {
                    let ga = if use_elig { total_get_copy(&self.ws.e_b1_alpha, k, 0.0) } else { total_get_copy(&self.ws.g_b1_alpha, k, 0.0) };
                    if let (Some(p), Some(v)) = (bb.b1_alpha.get_mut(k), bb.v_b1_alpha.get_mut(k)) {
                        rmsprop_step(p, v, ga * grad_scale, lr * 0.65, wd * 0.5, 4.0, exact);
                    }
                }
                let gw = if use_elig { total_get_copy(&self.ws.e_b1_omega, k, 0.0) } else { total_get_copy(&self.ws.g_b1_omega, k, 0.0) };
                if let (Some(p), Some(v)) = (bb.b1_omega.get_mut(k), bb.v_b1_omega.get_mut(k)) {
                    rmsprop_step(p, v, gw * grad_scale, lr, 0.0, 3.5, exact);
                }
                let woff = k * ein;
                for j in 0..ein {
                    let idx = woff + j;
                    let giw = if use_elig { total_get_copy(&self.ws.e_b1_inw, idx, 0.0) } else { total_get_copy(&self.ws.g_b1_inw, idx, 0.0) };
                    if let (Some(p), Some(v)) = (bb.b1_in_w.get_mut(idx), bb.v_b1_in_w.get_mut(idx)) {
                        rmsprop_step(p, v, giw * grad_scale, lr, wd, 4.0, exact);
                    }
                    if sel_on {
                        let gsw = if use_elig { total_get_copy(&self.ws.e_b1_selw, idx, 0.0) } else { total_get_copy(&self.ws.g_b1_selw, idx, 0.0) };
                        if let (Some(p), Some(v)) = (bb.b1_sel_w.get_mut(idx), bb.v_b1_sel_w.get_mut(idx)) {
                            rmsprop_step(p, v, gsw * grad_scale, lr * 0.85, wd * 2.0, 3.0, exact);
                        }
                    }
                }
                if sel_on {
                    let gbe = if use_elig { total_get_copy(&self.ws.e_b1_beta, k, 0.0) } else { total_get_copy(&self.ws.g_b1_beta, k, 0.0) };
                    if let (Some(p), Some(v)) = (bb.b1_beta.get_mut(k), bb.v_b1_beta.get_mut(k)) {
                        rmsprop_step(p, v, gbe * grad_scale, lr * 1.25, wd * 2.0, 3.0, exact);
                    }
                    let gbd = if use_elig { total_get_copy(&self.ws.e_b1_bdelta, k, 0.0) } else { total_get_copy(&self.ws.g_b1_bdelta, k, 0.0) };
                    if let (Some(p), Some(v)) = (bb.b1_bdelta.get_mut(k), bb.v_b1_bdelta.get_mut(k)) {
                        rmsprop_step(p, v, gbd * grad_scale, lr * 0.85, wd * 2.0, 3.0, exact);
                    }
                    if fb_on {
                        let qoff = k * 2;
                        for ci in 0..2usize {
                            let gq = if use_elig { total_get_copy(&self.ws.e_b1_qfb, qoff + ci, 0.0) } else { total_get_copy(&self.ws.g_b1_qfb, qoff + ci, 0.0) };
                            if let (Some(p), Some(v)) = (bb.b1_qfb.get_mut(qoff + ci), bb.v_b1_qfb.get_mut(qoff + ci)) {
                                rmsprop_step(p, v, gq * grad_scale, lr * 0.85, wd * 2.0, 2.5, exact);
                            }
                        }
                    }
                }
            }

            for s in 0..UNIFIED_B2_SLOTS {
                if !u_lim {
                    let ga = if use_elig { total_get_copy(&self.ws.e_b2_alpha, s, 0.0) } else { total_get_copy(&self.ws.g_b2_alpha, s, 0.0) };
                    if let (Some(p), Some(v)) = (bb.b2_alpha.get_mut(s), bb.v_b2_alpha.get_mut(s)) {
                        rmsprop_step(p, v, ga * grad_scale, lr * 0.65, wd * 0.5, 4.0, exact);
                    }
                }
                let gw = if use_elig { total_get_copy(&self.ws.e_b2_omega, s, 0.0) } else { total_get_copy(&self.ws.g_b2_omega, s, 0.0) };
                if let (Some(p), Some(v)) = (bb.b2_omega.get_mut(s), bb.v_b2_omega.get_mut(s)) {
                    rmsprop_step(p, v, gw * grad_scale, lr, 0.0, 3.5, exact);
                }
                let woff = s * ein;
                for j in 0..ein {
                    let idx = woff + j;
                    let giw = if use_elig { total_get_copy(&self.ws.e_b2_inw, idx, 0.0) } else { total_get_copy(&self.ws.g_b2_inw, idx, 0.0) };
                    if let (Some(p), Some(v)) = (bb.b2_in_w.get_mut(idx), bb.v_b2_in_w.get_mut(idx)) {
                        rmsprop_step(p, v, giw * grad_scale, lr, wd, 4.0, exact);
                    }
                    if sel_on {
                        let gsw = if use_elig { total_get_copy(&self.ws.e_b2_selw, idx, 0.0) } else { total_get_copy(&self.ws.g_b2_selw, idx, 0.0) };
                        if let (Some(p), Some(v)) = (bb.b2_sel_w.get_mut(idx), bb.v_b2_sel_w.get_mut(idx)) {
                            rmsprop_step(p, v, gsw * grad_scale, lr * 0.85, wd * 2.0, 3.0, exact);
                        }
                    }
                }
                if sel_on {
                    let gbe = if use_elig { total_get_copy(&self.ws.e_b2_beta, s, 0.0) } else { total_get_copy(&self.ws.g_b2_beta, s, 0.0) };
                    if let (Some(p), Some(v)) = (bb.b2_beta.get_mut(s), bb.v_b2_beta.get_mut(s)) {
                        rmsprop_step(p, v, gbe * grad_scale, lr * 1.25, wd * 2.0, 3.0, exact);
                    }
                    let gbd = if use_elig { total_get_copy(&self.ws.e_b2_bdelta, s, 0.0) } else { total_get_copy(&self.ws.g_b2_bdelta, s, 0.0) };
                    if let (Some(p), Some(v)) = (bb.b2_bdelta.get_mut(s), bb.v_b2_bdelta.get_mut(s)) {
                        rmsprop_step(p, v, gbd * grad_scale, lr * 0.85, wd * 2.0, 3.0, exact);
                    }
                    if fb_on {
                        let qoff = s * 3;
                        for ci in 0..3usize {
                            let gq = if use_elig { total_get_copy(&self.ws.e_b2_qfb, qoff + ci, 0.0) } else { total_get_copy(&self.ws.g_b2_qfb, qoff + ci, 0.0) };
                            if let (Some(p), Some(v)) = (bb.b2_qfb.get_mut(qoff + ci), bb.v_b2_qfb.get_mut(qoff + ci)) {
                                rmsprop_step(p, v, gq * grad_scale, lr * 0.85, wd * 2.0, 2.5, exact);
                            }
                        }
                    }
                }
            }
        }
    }

    fn refresh_heads(&mut self) {
        let active = self.active_slot.min(self.slots.len().saturating_sub(1));
        if let Some(slot) = self.slots.get(active) {
            let na = slot.num_actions;
            let fdim = slot.feat_dim;
            for c in 0..na {
                let off = c * fdim;
                let mut dot = 0.0_f32;
                if let Some(wc) = slot.actor_w.get(off..off + fdim) {
                    dot = fast_dot(wc, &self.ws.feat_gated);
                }
                let lg = sanitize_bounded_f32(total_get_copy(&slot.actor_b, c, 0.0) + dot, 64.0);
                let _ = total_set(&mut self.ws.logits, c, lg);
            }
            rational_probs_p4_into(&self.ws.logits, &mut self.ws.probs, &mut self.ws.psis);
            self.ws.critic_val = sanitize_bounded_f32(
                slot.critic_b + fast_dot(&slot.critic_w, &self.ws.feat_gated),
                64.0,
            );
        }
    }

    fn apply_rl_td_update(&mut self, td_err: f32) {
        let active = self.active_slot.min(self.slots.len().saturating_sub(1));
        let mut surp_mult = 1.0_f32;
        if let Some(slot) = self.slots.get_mut(active) {
            if self.cfg.enable_surprise_rl {
                let st = total_div_f32(td_err * td_err, slot.td_var_ema + 0.05, 1.0);
                surp_mult = total_div_f32(1.0 + 1.5 * st, 1.0 + 0.75 * st, 1.0).clamp(1.0, 2.0);
                slot.td_var_ema = sanitize_f32(0.98 * slot.td_var_ema + 0.02 * td_err * td_err).clamp(0.01, 4.0);
            }
        }
        self.last_surp = surp_mult;

        let feat_norm_sq = sanitize_f32(fast_dot(&self.ws.feat_gated, &self.ws.feat_gated));
        let inv_nlms = total_div_unit_shifted_f32(1.0, feat_norm_sq);
        let ro_lr = sanitize_f32(self.cfg.ro_lr * 4.5 * surp_mult * inv_nlms);
        let c_lr = sanitize_f32(self.cfg.critic_lr * 3.0 * inv_nlms);
        let wd = self.cfg.weight_decay;
        let na = self.ws.num_actions;
        let fdim = self.ws.feat_dim;
        let ent = self.cfg.entropy_beta;

        if let Some(slot) = self.slots.get_mut(active) {
            let inv_na = total_div_f32(1.0, na as f32, 0.5);
            for c in 0..na {
                let eb = total_get_copy(&self.ws.e_actor_b, c, 0.0);
                let psi = total_get_copy(&self.ws.psis, c, 0.0);
                let pr = total_get_copy(&self.ws.probs, c, 0.0);
                let ent_b = ent * psi * (inv_na - pr);
                if let Some(pb) = slot.actor_b.get_mut(c) {
                    nlms_step(pb, td_err * eb + ent_b, 0.5 * ro_lr, 0.0, 4.0);
                }
                let off = c * fdim;
                for i in 0..fdim {
                    let ew = total_get_copy(&self.ws.e_actor_w, off + i, 0.0);
                    let f_i = total_get_copy(&self.ws.feat_gated, i, 0.0);
                    let idx = off + i;
                    if let Some(pw) = slot.actor_w.get_mut(idx) {
                        nlms_step(pw, td_err * ew + ent_b * f_i, ro_lr, wd, 8.0);
                    }
                }
            }
            nlms_step(&mut slot.critic_b, td_err * self.ws.e_critic_b, 0.5 * c_lr, 0.0, 4.0);
            for i in 0..fdim {
                let ecw = total_get_copy(&self.ws.e_critic_w, i, 0.0);
                if let Some(pw) = slot.critic_w.get_mut(i) {
                    nlms_step(pw, td_err * ecw, c_lr, wd, 8.0);
                }
            }
        }
        self.apply_rmsprop_rtrl_scaled(true, td_err, surp_mult);
    }

    /// Точный градиент log p_target по всем логитам рационального p4 softmax (учитывает сдвиг на max_l).
    pub fn fill_p4_logprob_grads(logits: &[f32], probs: &[f32], psis: &[f32], target: usize, out: &mut [f32]) {
        algebra_fill_p4_logprob_grads(logits, probs, psis, target, out);
    }

    /// Потоковая классификация (ELEC2, Waves, Dyck-2). Возвращает `(argmax_pred, probs[pred])`.
    pub fn step_supervised_class(&mut self, raw_x: &[f32], target_class: usize, has_target: bool, train: bool, seu_guard: bool) -> (usize, f32) {
        let eff_train = train && (self.probe_eps_left == 0);
        self.forward_and_rtrl_traces(raw_x, self.cfg.learn_norm && eff_train, eff_train, seu_guard);

        let na = self.ws.num_actions;
        let mut pred = 0usize;
        let mut best_l = -1e30_f32;
        for c in 0..na {
            let lg = total_get_copy(&self.ws.logits, c, 0.0);
            if c == 0 || lg > best_l {
                best_l = lg;
                pred = c;
            }
        }
        let conf = total_get_copy(&self.ws.probs, pred, 0.0);

        if eff_train && has_target && target_class < na {
            let active = self.active_slot.min(self.slots.len().saturating_sub(1));
            let mut lerrs = [0.0_f32; 32];
            Self::fill_p4_logprob_grads(&self.ws.logits, &self.ws.probs, &self.ws.psis, target_class, &mut lerrs);
            self.compute_instant_rtrl_grads(lerrs.get(..na).unwrap_or(&[]), 0.0);

            let ro_lr = self.cfg.ro_lr;
            let wd = self.cfg.weight_decay;
            let exact = self.cfg.exact_inv_sqrt;
            let use_idbd = self.cfg.enable_idbd;
            let idbd_lr = self.cfg.idbd_lr;
            let fdim = self.ws.feat_dim;
            let off_b0 = self.ws.eff_in;
            let off_b1 = off_b0 + self.ws.b0_dim;
            let off_b2 = off_b1 + UNIFIED_B1_DIM;
            let bounds = [(0usize, off_b0), (off_b0, off_b1), (off_b1, off_b2), (off_b2, fdim)];

            if let Some(slot) = self.slots.get_mut(active) {
                let mut band_gains = [1.0_f32; 4];
                if use_idbd {
                    for b in 0..4usize {
                        let (gp, _) = rational_pos_and_deriv(total_get_copy(&slot.idbd_beta_actor, b, 0.0));
                        let _ = total_set(&mut band_gains, b, gp);
                    }
                }
                let mut band_u = [0.0_f32; 4];
                let mut band_f2 = [0.0_f32; 4];
                let mut band_cnt = [0usize; 4];

                for c in 0..na {
                    let ec = total_get_copy(&lerrs, c, 0.0);
                    let off = c * fdim;
                    for b in 0..4usize {
                        let (lo, hi) = total_get_copy(&bounds, b, (0, 0));
                        let eff_lr = sanitize_f32(ro_lr * total_get_copy(&band_gains, b, 1.0));
                        for i in lo..hi {
                            let f_i = total_get_copy(&self.ws.feat_gated, i, 0.0);
                            let g = ec * f_i;
                            let idx = off + i;
                            if use_idbd {
                                let v_cur = total_get_copy(&slot.v_actor_w, idx, 0.04);
                                let u_i = sanitize_bounded_f32(g * inv_sqrt_select(v_cur + 0.04, exact), 3.0);
                                let prev_u = total_get_copy(&band_u, b, 0.0);
                                let prev_f2 = total_get_copy(&band_f2, b, 0.0);
                                let prev_cnt = total_get_copy(&band_cnt, b, 0);
                                let _ = total_set(&mut band_u, b, prev_u + u_i);
                                let _ = total_set(&mut band_f2, b, prev_f2 + f_i * f_i);
                                let _ = total_set(&mut band_cnt, b, prev_cnt + 1);
                            }
                            if let (Some(pw), Some(vw)) = (slot.actor_w.get_mut(idx), slot.v_actor_w.get_mut(idx)) {
                                rmsprop_step(pw, vw, g, eff_lr, wd, 8.0, exact);
                            }
                        }
                    }
                    if let (Some(pb), Some(vb)) = (slot.actor_b.get_mut(c), slot.v_actor_b.get_mut(c)) {
                        rmsprop_step(pb, vb, ec, 0.35 * ro_lr, 0.0, 4.0, exact);
                    }
                }

                if use_idbd {
                    for b in 0..4usize {
                        let cnt = total_get_copy(&band_cnt, b, 0);
                        if cnt > 0 {
                            let inv_c = total_div_f32(1.0, cnt as f32, 0.0);
                            update_band_idbd(
                                &mut slot.idbd_beta_actor,
                                &mut slot.idbd_h_actor,
                                b,
                                total_get_copy(&band_u, b, 0.0) * inv_c,
                                total_get_copy(&band_f2, b, 0.0) * inv_c,
                                ro_lr,
                                idbd_lr,
                            );
                        }
                    }
                }
            }
            self.apply_rmsprop_rtrl_scaled(false, 1.0, 1.0);
        }

        (pred, conf)
    }

    /// Потоковая скалярная регрессия (NASA C-MAPSS RUL).
    pub fn step_supervised_reg(&mut self, raw_x: &[f32], target_y: f32, has_target: bool, train: bool, seu_guard: bool) -> f32 {
        let eff_train = train && (self.probe_eps_left == 0);
        self.forward_and_rtrl_traces(raw_x, self.cfg.learn_norm && eff_train, eff_train, seu_guard);
        let pred = self.ws.critic_val;

        if eff_train && has_target {
            let err = sanitize_bounded_f32(target_y - pred, 2.0);
            self.compute_instant_rtrl_grads(&[], err);

            let active = self.active_slot.min(self.slots.len().saturating_sub(1));
            let feat_norm_sq = sanitize_f32(fast_dot(&self.ws.feat_gated, &self.ws.feat_gated));
            let inv_nlms = total_div_unit_shifted_f32(1.0, feat_norm_sq);
            let eff_reg_lr = sanitize_f32(0.15 * self.cfg.critic_lr * inv_nlms);
            let wd = self.cfg.weight_decay;
            let use_idbd = self.cfg.enable_idbd;
            let idbd_lr = self.cfg.idbd_lr;
            let fdim = self.ws.feat_dim;
            let off_b0 = self.ws.eff_in;
            let off_b1 = off_b0 + self.ws.b0_dim;
            let off_b2 = off_b1 + UNIFIED_B1_DIM;
            let bounds = [(0usize, off_b0), (off_b0, off_b1), (off_b1, off_b2), (off_b2, fdim)];

            if let Some(slot) = self.slots.get_mut(active) {
                for b in 0..4usize {
                    let (lo, hi) = total_get_copy(&bounds, b, (0, 0));
                    if hi <= lo {
                        continue;
                    }
                    let gain = if use_idbd {
                        rational_pos_and_deriv(total_get_copy(&slot.idbd_beta_critic, b, 0.0)).0
                    } else {
                        1.0
                    };
                    let step_lr = sanitize_f32(eff_reg_lr * gain);
                    let mut sum_u = 0.0_f32;
                    let mut sum_f2 = 0.0_f32;
                    for i in lo..hi {
                        let f_i = total_get_copy(&self.ws.feat_gated, i, 0.0);
                        let g = err * f_i;
                        if use_idbd {
                            sum_u += sanitize_bounded_f32(g * 3.5, 3.0);
                            sum_f2 += f_i * f_i;
                        }
                        if let Some(pw) = slot.critic_w.get_mut(i) {
                            nlms_step(pw, g, step_lr, wd * 0.5, 8.0);
                        }
                    }
                    if use_idbd {
                        let inv_c = total_div_f32(1.0, (hi - lo) as f32, 0.0);
                        update_band_idbd(
                            &mut slot.idbd_beta_critic,
                            &mut slot.idbd_h_critic,
                            b,
                            sum_u * inv_c,
                            sum_f2 * inv_c,
                            eff_reg_lr,
                            idbd_lr,
                        );
                    }
                }
                nlms_step(&mut slot.critic_b, err, 0.5 * eff_reg_lr, 0.0, 4.0);
            }
            self.apply_rmsprop_rtrl_scaled(false, 1.0, 0.15);
        }
        pred
    }

    /// Безопасно считывает предсказание класса по `&self` (на стеке, без мутации и аллокаций, 100% каузально!).
    pub fn predict_class_readonly(&self, raw_seq: &[&[f32]]) -> usize {
        if raw_seq.is_empty() {
            return 0;
        }
        let active = self.active_slot.min(self.slots.len().saturating_sub(1));
        let bb_idx = self.backbone_idx(active);
        let slot = match self.slots.get(active) {
            Some(s) => s,
            _ => return 0,
        };
        let bb = match self.slots.get(bb_idx) {
            Some(s) => s,
            _ => return 0,
        };
        let mut prev_x = [0.0_f32; UNIFIED_MAX_RAW_IN];
        let mut eff_x = [0.0_f32; UNIFIED_MAX_EFF_IN];
        let mut b0 = [0.0_f32; UNIFIED_B0_DIM];
        let mut b1 = [0.0_f32; UNIFIED_B1_DIM];
        let mut b2 = [0.0_f32; UNIFIED_B2_DIM];
        let mut fnorm = [0.0_f32; UNIFIED_MAX_EFF_IN + UNIFIED_B0_DIM + UNIFIED_B1_DIM + UNIFIED_B2_DIM];

        let rdim = self.in_cond.raw_dim;
        let ein = self.ws.eff_in;
        let b0_dim = self.ws.b0_dim;
        let inv_cnt_m1 = if self.in_cond.count > 1 {
            total_div_f32(1.0, (self.in_cond.count - 1) as f32, 0.0)
        } else {
            0.0
        };
        let exact = self.cfg.exact_inv_sqrt;
        let sel_on = self.cfg.enable_selectivity;
        let fb_on = sel_on && self.cfg.enable_local_feedback;
        let min_a2 = self.cfg.min_alpha_sq;
        let mut has_prev = false;

        for step_x in raw_seq.iter() {
            for i in 0..rdim {
                let x = sanitize_bounded_f32(total_get_copy(step_x, i, 0.0), 1e12);
                let mu = total_get_copy(&self.in_cond.mean, i, 0.0);
                let m2i = total_get_copy(&self.in_cond.m2, i, 0.0);
                let var = if self.in_cond.count >= 2 { sanitize_f32(m2i * inv_cnt_m1).max(0.0) } else { 0.0 };
                let xn = sanitize_bounded_f32((x - mu) * inv_sqrt_newton_2step_f32(var + 1.0), 12.0);
                let _ = total_set(&mut eff_x, i, xn);
                if self.in_cond.use_delta {
                    let px = total_get_copy(&prev_x, i, 0.0);
                    let dxr = if has_prev { sanitize_bounded_f32((xn - px) * self.in_cond.delta_scale, 32.0) } else { 0.0 };
                    let dxc = sanitize_f32(dxr * inv_sqrt_newton_2step_f32(dxr * dxr + 1.0));
                    let _ = total_set(&mut eff_x, rdim + i, dxc);
                    let _ = total_set(&mut prev_x, i, xn);
                }
            }
            has_prev = true;
            let eff_slice = eff_x.get(..ein).unwrap_or(&[]);
            if self.band0.enabled {
                self.band0.step_into(eff_slice, &mut b0);
            }
            for k in 0..UNIFIED_B1_SLOTS {
                let u_off = k * 2;
                let u0 = total_get_copy(&b1, u_off, 0.0);
                let u1 = total_get_copy(&b1, u_off + 1, 0.0);
                let woff = k * ein;
                let mut m_in = 0.0_f32;
                if let Some(ws) = bb.b1_in_w.get(woff..woff + ein) {
                    m_in = sanitize_bounded_f32(fast_dot(ws, eff_slice), 32.0);
                }
                let delta = if sel_on {
                    let mut sel_in = 0.0_f32;
                    if let Some(sw) = bb.b1_sel_w.get(woff..woff + ein) {
                        sel_in = sanitize_bounded_f32(fast_dot(sw, eff_slice), 16.0);
                    }
                    let fb = if fb_on {
                        let inv_r = inv_sqrt_select(sanitize_f32(u0 * u0 + u1 * u1 + 1.0), exact);
                        (total_get_copy(&bb.b1_qfb, u_off, 0.0) * u0 + total_get_copy(&bb.b1_qfb, u_off + 1, 0.0) * u1) * inv_r
                    } else { 0.0 };
                    let z = sanitize_bounded_f32(total_get_copy(&bb.b1_bdelta, k, 0.0) + sel_in + fb, 10.0);
                    rational_pos_and_deriv(sanitize_bounded_f32(total_get_copy(&bb.b1_beta, k, 0.0) * z, 10.0)).0
                } else { 1.0 };
                let a = total_get_copy(&bb.b1_alpha, k, 0.2);
                let lam = total_div_unit_shifted_f32(1.0, delta * (a * a + min_a2));
                let g_in = total_div_unit_shifted_f32(delta, 0.5 * delta);
                let t = 0.5 * sanitize_bounded_f32(delta * total_get_copy(&bb.b1_omega, k, 0.3), 16.0);
                let inv_d = total_div_unit_shifted_f32(1.0, t * t);
                let c = (1.0 - t * t) * inv_d;
                let s = 2.0 * t * inv_d;
                let _ = total_set(&mut b1, u_off, sanitize_bounded_f32(lam * (c * u0 - s * u1) + g_in * m_in, 128.0));
                let _ = total_set(&mut b1, u_off + 1, sanitize_bounded_f32(lam * (s * u0 + c * u1), 128.0));
            }
            for s_idx in 0..UNIFIED_B2_SLOTS {
                let off = s_idx * 3;
                let u = (total_get_copy(&b2, off, 0.0), total_get_copy(&b2, off + 1, 0.0), total_get_copy(&b2, off + 2, 0.0));
                let e = (total_get_copy(&bb.b2_axis, off, 1.0), total_get_copy(&bb.b2_axis, off + 1, 0.0), total_get_copy(&bb.b2_axis, off + 2, 0.0));
                let dd = (total_get_copy(&bb.b2_in_dir, off, 0.0), total_get_copy(&bb.b2_in_dir, off + 1, 1.0), total_get_copy(&bb.b2_in_dir, off + 2, 0.0));
                let woff = s_idx * ein;
                let mut m_in = 0.0_f32;
                if let Some(ws) = bb.b2_in_w.get(woff..woff + ein) {
                    m_in = sanitize_bounded_f32(fast_dot(ws, eff_slice), 32.0);
                }
                let delta = if sel_on {
                    let mut sel_in = 0.0_f32;
                    if let Some(sw) = bb.b2_sel_w.get(woff..woff + ein) {
                        sel_in = sanitize_bounded_f32(fast_dot(sw, eff_slice), 16.0);
                    }
                    let fb = if fb_on {
                        let inv_r = inv_sqrt_select(sanitize_f32(u.0 * u.0 + u.1 * u.1 + u.2 * u.2 + 1.0), exact);
                        (total_get_copy(&bb.b2_qfb, off, 0.0) * u.0 + total_get_copy(&bb.b2_qfb, off + 1, 0.0) * u.1 + total_get_copy(&bb.b2_qfb, off + 2, 0.0) * u.2) * inv_r
                    } else { 0.0 };
                    let z = sanitize_bounded_f32(total_get_copy(&bb.b2_bdelta, s_idx, 0.0) + sel_in + fb, 10.0);
                    rational_pos_and_deriv(sanitize_bounded_f32(total_get_copy(&bb.b2_beta, s_idx, 0.0) * z, 10.0)).0
                } else { 1.0 };
                let a = total_get_copy(&bb.b2_alpha, s_idx, 0.2);
                let lam = total_div_unit_shifted_f32(1.0, delta * (a * a + min_a2));
                let g_in = total_div_unit_shifted_f32(delta, 0.5 * delta);
                let th = sanitize_bounded_f32(delta * total_get_copy(&bb.b2_omega, s_idx, 0.3), 16.0);
                let (ru, _, _, _) = rot3_step_and_deriv(e, th, u);
                let _ = total_set(&mut b2, off, sanitize_bounded_f32(lam * ru.0 + g_in * m_in * dd.0, 128.0));
                let _ = total_set(&mut b2, off + 1, sanitize_bounded_f32(lam * ru.1 + g_in * m_in * dd.1, 128.0));
                let _ = total_set(&mut b2, off + 2, sanitize_bounded_f32(lam * ru.2 + g_in * m_in * dd.2, 128.0));
            }
        }

        let off_b0 = ein;
        let off_b1 = off_b0 + b0_dim;
        let off_b2 = off_b1 + UNIFIED_B1_DIM;
        let _ = rmsnorm_band_into(eff_x.get(..ein).unwrap_or(&[]), &mut fnorm, 0, 0.5, exact);
        if b0_dim > 0 {
            let _ = rmsnorm_band_into(&b0, &mut fnorm, off_b0, 0.05, exact);
        }
        let _ = rmsnorm_band_into(&b1, &mut fnorm, off_b1, 0.05, exact);
        let _ = rmsnorm_band_into(&b2, &mut fnorm, off_b2, 0.05, exact);

        let g0 = 0.5 * rational_pos_and_deriv(total_get_copy(&slot.gate_logits, 0, 0.0)).0;
        let g1 = 0.5 * rational_pos_and_deriv(total_get_copy(&slot.gate_logits, 1, 0.0)).0;
        let g2 = 0.5 * rational_pos_and_deriv(total_get_copy(&slot.gate_logits, 2, 0.0)).0;
        let g3 = 0.5 * rational_pos_and_deriv(total_get_copy(&slot.gate_logits, 3, 0.0)).0;
        let scale_band = |buf: &mut [f32], lo: usize, hi: usize, g: f32| {
            if let Some(sub) = buf.get_mut(lo..hi) {
                for v in sub.iter_mut() {
                    *v = sanitize_f32(*v * g);
                }
            }
        };
        scale_band(&mut fnorm, 0, ein, g0);
        if b0_dim > 0 {
            scale_band(&mut fnorm, off_b0, off_b1, g1);
        }
        scale_band(&mut fnorm, off_b1, off_b2, g2);
        scale_band(&mut fnorm, off_b2, slot.feat_dim, g3);

        let mut best_c = 0usize;
        let mut best_l = -1e30_f32;
        for c in 0..slot.num_actions {
            let off = c * slot.feat_dim;
            let mut dot = 0.0_f32;
            if let (Some(wc), Some(fg)) = (slot.actor_w.get(off..off + slot.feat_dim), fnorm.get(..slot.feat_dim)) {
                dot = fast_dot(wc, fg);
            }
            let lg = total_get_copy(&slot.actor_b, c, 0.0) + dot;
            if c == 0 || lg > best_l {
                best_l = lg;
                best_c = c;
            }
        }
        best_c
    }

    /// Онлайн шаг RL-политики с точным `TD(lambda)` (CartPole, RepeatPrevious, T-Maze, Waves-RL, HardPOMDP).
    pub fn decide_rl(&mut self, raw_x: &[f32], train: bool, inject_seu: bool, seu_guard: bool) -> (usize, f32) {
        let eff_train = train && (self.probe_eps_left == 0);
        if inject_seu {
            self.inject_weight_bit30_seu(self.active_slot);
        }
        self.forward_and_rtrl_traces(raw_x, self.cfg.learn_norm && eff_train, eff_train, seu_guard);

        if eff_train && self.has_pending_step && self.has_pending_reward {
            let td_prev = sanitize_bounded_f32(
                self.pending_reward + self.cfg.gamma * self.ws.critic_val - self.last_critic_val,
                4.0,
            );
            self.apply_rl_td_update(td_prev);
            self.has_pending_reward = false;
            self.refresh_heads();
        }

        let na = self.ws.num_actions;
        let act = if eff_train {
            self.rng = splitmix64(self.rng);
            let u = rand_unit_f32(&mut self.rng);
            let mut cum = 0.0_f32;
            let mut chosen = na.saturating_sub(1);
            for c in 0..na {
                cum = sanitize_f32(cum + total_get_copy(&self.ws.probs, c, 0.0));
                if u <= cum {
                    chosen = c;
                    break;
                }
            }
            chosen
        } else {
            let mut chosen = 0usize;
            let mut best_l = -1e30_f32;
            for c in 0..na {
                let lg = total_get_copy(&self.ws.logits, c, 0.0);
                if c == 0 || lg > best_l {
                    best_l = lg;
                    chosen = c;
                }
            }
            chosen
        };

        self.last_action = act;
        self.last_critic_val = self.ws.critic_val;
        self.has_pending_step = eff_train;

        if eff_train {
            let decay = sanitize_f32(self.cfg.gamma * self.cfg.trace_lambda).clamp(0.0, 0.98);
            self.ws.decay_eligibility(decay);

            let fdim = self.ws.feat_dim;
            let mut lerrs = [0.0_f32; 32];
            Self::fill_p4_logprob_grads(&self.ws.logits, &self.ws.probs, &self.ws.psis, act, &mut lerrs);
            for c in 0..na {
                let ec = total_get_copy(&lerrs, c, 0.0);
                if let Some(p) = self.ws.e_actor_b.get_mut(c) {
                    *p = sanitize_f32(*p + ec);
                }
                let off = c * fdim;
                for i in 0..fdim {
                    let f_i = total_get_copy(&self.ws.feat_gated, i, 0.0);
                    if let Some(p) = self.ws.e_actor_w.get_mut(off + i) {
                        *p = sanitize_bounded_f32(*p + ec * f_i, 16.0);
                    }
                }
            }
            self.ws.e_critic_b = sanitize_f32(self.ws.e_critic_b + 1.0).clamp(0.0, 16.0);
            for i in 0..fdim {
                let f_i = total_get_copy(&self.ws.feat_gated, i, 0.0);
                if let Some(p) = self.ws.e_critic_w.get_mut(i) {
                    *p = sanitize_bounded_f32(*p + f_i, 16.0);
                }
            }

            self.compute_instant_rtrl_grads(lerrs.get(..na).unwrap_or(&[]), 0.35);
            let add_bounded = |dst: &mut [f32], src: &[f32]| {
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = sanitize_bounded_f32(*d + s, 16.0);
                }
            };
            add_bounded(&mut self.ws.e_gates, &self.ws.g_gates);
            add_bounded(&mut self.ws.e_b1_alpha, &self.ws.g_b1_alpha);
            add_bounded(&mut self.ws.e_b1_omega, &self.ws.g_b1_omega);
            add_bounded(&mut self.ws.e_b1_beta, &self.ws.g_b1_beta);
            add_bounded(&mut self.ws.e_b1_bdelta, &self.ws.g_b1_bdelta);
            add_bounded(&mut self.ws.e_b1_qfb, &self.ws.g_b1_qfb);
            fast_saxpy(&mut self.ws.e_b1_inw, &self.ws.g_b1_inw, 1.0);
            fast_saxpy(&mut self.ws.e_b1_selw, &self.ws.g_b1_selw, 1.0);
            add_bounded(&mut self.ws.e_b2_alpha, &self.ws.g_b2_alpha);
            add_bounded(&mut self.ws.e_b2_omega, &self.ws.g_b2_omega);
            add_bounded(&mut self.ws.e_b2_beta, &self.ws.g_b2_beta);
            add_bounded(&mut self.ws.e_b2_bdelta, &self.ws.g_b2_bdelta);
            add_bounded(&mut self.ws.e_b2_qfb, &self.ws.g_b2_qfb);
            fast_saxpy(&mut self.ws.e_b2_inw, &self.ws.g_b2_inw, 1.0);
            fast_saxpy(&mut self.ws.e_b2_selw, &self.ws.g_b2_selw, 1.0);
        }

        (act, total_get_copy(&self.ws.probs, act, 0.0))
    }

    /// Фиксирует награду шага: при `done=true` применяет терминальный `TD(lambda)`, иначе сохраняет `r_t`.
    pub fn observe_rl_reward(&mut self, reward: f32, _next_obs: &[f32], done: bool, train: bool) {
        let eff_train = train && (self.probe_eps_left == 0);
        if !eff_train || !self.has_pending_step {
            return;
        }
        let r = sanitize_bounded_f32(reward, 16.0);
        if done {
            let td_err = sanitize_bounded_f32(r - self.last_critic_val, 4.0);
            self.apply_rl_td_update(td_err);
            self.has_pending_step = false;
            self.has_pending_reward = false;
        } else {
            self.pending_reward = r;
            self.has_pending_reward = true;
        }
    }

    /// Переход к зрелому кандидату на замороженный пробный прогон, к свежему слоту или LRU-вытеснению (0 аллокаций).
    fn select_next_or_probe_slot(&mut self, avoid_slot: usize) {
        let nb = self.slots.len();
        if nb <= 1 {
            return;
        }
        let ep = self.total_episodes;
        for bi in 0..nb {
            if bi != avoid_slot {
                if let Some(s) = self.slots.get_mut(bi) {
                    if s.age_eps > 0 && ep >= s.cooldown_until {
                        s.ph.reset_cusum();
                        self.active_slot = bi;
                        self.probe_eps_left = 4;
                        self.probe_score_sum = 0.0;
                        self.probe_score_cnt = 0;
                        self.bank_switches = self.bank_switches.saturating_add(1);
                        return;
                    }
                }
            }
        }
        for bi in 0..nb {
            if bi != avoid_slot {
                if let Some(s) = self.slots.get_mut(bi) {
                    if s.age_eps == 0 && ep >= s.cooldown_until {
                        s.ph.reset_cusum();
                        s.cooldown_until = ep.saturating_add(20);
                        s.last_used_ep = ep;
                        self.active_slot = bi;
                        self.probe_eps_left = 0;
                        self.bank_switches = self.bank_switches.saturating_add(1);
                        return;
                    }
                }
            }
        }
        let mut lru_bi = total_rem_usize(avoid_slot + 1, nb, 0);
        let mut oldest = usize::MAX;
        for (bi, s) in self.slots.iter().enumerate() {
            if bi != avoid_slot && s.last_used_ep < oldest {
                oldest = s.last_used_ep;
                lru_bi = bi;
            }
        }
        self.active_slot = lru_bi;
        self.probe_eps_left = 0;
        self.bank_switches = self.bank_switches.saturating_add(1);
        if let Some(ns) = self.slots.get_mut(lru_bi) {
            ns.ph = BartlettPageHinkley::new(self.cfg.ph_delta, self.cfg.ph_threshold, self.cfg.ph_warmup);
            ns.age_eps = 0;
            ns.ret_ema = 0.5;
            ns.td_var_ema = 0.25;
            ns.cooldown_until = ep.saturating_add(20);
            ns.last_used_ep = ep;
            ns.actor_b.fill(0.0);
            ns.critic_w.fill(0.0);
            ns.critic_b = 0.0;
            ns.idbd_beta_actor = [0.0_f32; 4];
            ns.idbd_h_actor = [0.0_f32; 4];
            ns.idbd_beta_critic = [0.0_f32; 4];
            ns.idbd_h_critic = [0.0_f32; 4];
        }
    }

    /// Фиксирует конец эпизода, ведёт Bartlett Page-Hinkley и Probe-Before-Allocate банк (`A -> B -> A`).
    pub fn end_episode_score(&mut self, score_01: f32, train: bool) -> bool {
        self.total_episodes = self.total_episodes.saturating_add(1);
        let sc = sanitize_f32(score_01).clamp(0.0, 1.0);
        let use_bart = self.cfg.enable_bartlett_ph;
        let bank_sw = self.cfg.enable_bank_switching && self.slots.len() > 1;
        let active = self.active_slot.min(self.slots.len().saturating_sub(1));

        if train && bank_sw && self.probe_eps_left > 0 {
            self.probe_score_sum = sanitize_f32(self.probe_score_sum + sc);
            self.probe_score_cnt = self.probe_score_cnt.saturating_add(1);
            self.probe_eps_left = self.probe_eps_left.saturating_sub(1);

            if self.probe_eps_left == 0 {
                let mean_p = total_div_f32(self.probe_score_sum, self.probe_score_cnt.max(1) as f32, 0.0);
                let accept = if let Some(s) = self.slots.get_mut(active) {
                    let ok = mean_p >= 0.55 || mean_p >= 0.72 * s.ret_ema;
                    s.ph.reset_cusum();
                    s.cooldown_until = self.total_episodes.saturating_add(24);
                    if ok {
                        s.last_used_ep = self.total_episodes;
                        s.ret_ema = sanitize_f32(0.85 * s.ret_ema + 0.15 * mean_p);
                    }
                    ok
                } else {
                    false
                };
                if !accept {
                    let orig = self.probe_origin_slot;
                    self.select_next_or_probe_slot(orig);
                }
            }
            return false;
        }

        let mut drift = false;
        if let Some(slot) = self.slots.get_mut(active) {
            slot.age_eps = slot.age_eps.saturating_add(1);
            slot.last_used_ep = self.total_episodes;
            slot.ret_ema = sanitize_f32(0.90 * slot.ret_ema + 0.10 * sc);
            if train && bank_sw {
                let raw_drift = slot.ph.observe(sc, use_bart);
                if self.total_episodes >= slot.cooldown_until {
                    drift = raw_drift;
                } else {
                    slot.ph.reset_cusum();
                }
            }
        }

        if drift && bank_sw {
            if let Some(cur) = self.slots.get_mut(active) {
                cur.ph.reset_cusum();
                cur.cooldown_until = self.total_episodes.saturating_add(24);
            }
            self.probe_origin_slot = active;
            self.select_next_or_probe_slot(active);
        }
        drift
    }
}
