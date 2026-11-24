// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! 3 честных бейзлайна в том же бинарнике на чистом Rust (без PyTorch):
//! 1. Random / Oracle (границы).
//! 2. TabularPHBaseline (табличный Q-learning + Page-Hinkley).
//! 3. OnlineGruBaseline (GRU H=32 + A2C + Entropy + Gradient Clipping + BPTT + Adam).

use std::collections::HashMap;

use crate::belief::PageHinkleyDetector;
use crate::math::{
    extract_tokens, fast_dot_raw, fast_saxpy, fnv1a_bytes, rand_signed_f32, rand_unit_f32,
    splitmix64,
};

fn scrub_slice_seu(slice: &mut [f32], limit: f32, replacement: f32) -> u64 {
    let mut faults = 0u64;
    for v in slice.iter_mut() {
        if !v.is_finite() || v.abs() > limit {
            *v = replacement;
            faults = faults.saturating_add(1);
        }
    }
    faults
}

fn greedy_head_action(bias: &[f32], weight: &[f32], feat: &[f32], num_actions: usize) -> usize {
    let f_dim = feat.len();
    let mut logits = vec![0.0_f32; num_actions];
    let mut max_l = f32::NEG_INFINITY;
    for a in 0..num_actions {
        let off = a * f_dim;
        let s = bias[a] + fast_dot_raw(&weight[off..off + f_dim], feat);
        logits[a] = s;
        if s > max_l {
            max_l = s;
        }
    }
    let mut sum_e = 0.0_f32;
    let mut probs = vec![0.0_f32; num_actions];
    for a in 0..num_actions {
        let e = (logits[a] - max_l).exp();
        probs[a] = e;
        sum_e += e;
    }
    if !sum_e.is_finite() || sum_e <= 0.0 {
        return 0;
    }
    let inv = 1.0 / sum_e;
    let mut best = 0usize;
    let mut best_p = -1.0_f32;
    for (a, &p) in probs.iter().enumerate() {
        let pn = p * inv;
        if pn > best_p {
            best_p = pn;
            best = a;
        }
    }
    best
}

/// Uniform physical SRAM Weight-SEU injection across ALL live learnable weight slices
/// proportional to tensor size (identical Knuth-hash indexing to `MicroContinualCore`).
fn flip_weight_in_slices(slices: &mut [&mut [f32]], reg_idx: usize, bit: u32) {
    let total_len: usize = slices.iter().map(|s| s.len()).sum();
    if total_len == 0 {
        return;
    }
    let mask = 1u32 << (bit & 31);
    let mut rem = reg_idx.wrapping_mul(2_654_435_761usize) % total_len;
    for sl in slices.iter_mut() {
        let len = sl.len();
        if rem < len {
            let mut cur = sl[rem];
            if cur.is_finite() && (cur.abs() < 1.001 || cur.abs() >= 2.0) {
                cur = if cur < 0.0 { -1.25 } else { 1.25 };
            }
            sl[rem] = f32::from_bits(cur.to_bits() ^ mask);
            return;
        }
        rem -= len;
    }
}

pub struct TabularPHBaseline {
    pub num_actions: usize,
    pub banks: Vec<HashMap<u64, Vec<f32>>>,
    pub phs: Vec<PageHinkleyDetector>,
    pub active_bank: usize,
    pub trace: Vec<(u64, usize, f32)>,
    pub rng: u64,
    pub eps: f32,
    pub alpha: f32,
    pub gamma: f32,
}

impl TabularPHBaseline {
    pub fn new(num_actions: usize, max_banks: usize, seed: u64) -> Self {
        let mut banks = Vec::with_capacity(max_banks);
        let mut phs = Vec::with_capacity(max_banks);
        for _ in 0..max_banks {
            banks.push(HashMap::new());
            phs.push(PageHinkleyDetector::new(0.14, 1.55, 14));
        }
        Self {
            num_actions,
            banks,
            phs,
            active_bank: 0,
            trace: Vec::with_capacity(16),
            rng: splitmix64(seed ^ 0x7AB1_0001),
            eps: 0.15,
            alpha: 0.30,
            gamma: 0.90,
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.banks
            .iter()
            .map(|m| m.len() * (16 + self.num_actions * 4))
            .sum::<usize>()
            + self.phs.len() * 40
    }

    pub fn start_episode(&mut self) {
        self.trace.clear();
    }

    fn obs_key(obs: &str) -> u64 {
        fnv1a_bytes(obs.as_bytes(), 0x1234_5678_9ABC_DEF0)
    }

    pub fn decide(&mut self, obs: &str, train_mode: bool) -> usize {
        let k = Self::obs_key(obs);
        let q = self.banks[self.active_bank]
            .entry(k)
            .or_insert_with(|| vec![0.0_f32; self.num_actions]);
        let act = if train_mode && rand_unit_f32(&mut self.rng) < self.eps {
            self.rng = splitmix64(self.rng);
            (self.rng as usize) % self.num_actions
        } else {
            let mut best = 0usize;
            let mut best_v = -1e9_f32;
            for (i, &v) in q.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            best
        };
        self.trace.push((k, act, 0.0));
        act
    }

    pub fn observe_step_reward(&mut self, r: f32) {
        if let Some(last) = self.trace.last_mut() {
            last.2 = r;
        }
    }

    pub fn end_episode(&mut self, final_return: f32, train_mode: bool) {
        if !train_mode {
            return;
        }
        let has_dense = self.trace.iter().any(|x| x.2.abs() > 1e-6);
        let mut g = final_return;
        let mut running = 0.0_f32;
        for &(k, a, r_step) in self.trace.iter().rev() {
            let target = if has_dense {
                running = r_step + self.gamma * running;
                running
            } else {
                let cur = g;
                g *= self.gamma;
                cur
            };
            let q = self.banks[self.active_bank]
                .entry(k)
                .or_insert_with(|| vec![0.0_f32; self.num_actions]);
            q[a] += self.alpha * (target - q[a]);
        }
        let r_norm = ((final_return + 1.0) * 0.5).clamp(0.0, 1.0);
        if self.phs[self.active_bank].observe(r_norm) {
            self.phs[self.active_bank].reset_cusum();
            self.active_bank = (self.active_bank + 1) % self.banks.len();
        }
    }
}

#[derive(Clone)]
struct GruStepCache {
    x: Vec<f32>,
    h_prev: Vec<f32>,
    z: Vec<f32>,
    r: Vec<f32>,
    n: Vec<f32>,
    h: Vec<f32>,
    probs: Vec<f32>,
    v_pred: f32,
    action: usize,
    reward: f32,
}

#[derive(Clone)]
struct AdamParam {
    w: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl AdamParam {
    fn new(len: usize, scale: f32, rng: &mut u64) -> Self {
        let mut w = vec![0.0_f32; len];
        for cell in &mut w {
            *cell = rand_signed_f32(rng) * scale;
        }
        Self {
            w,
            m: vec![0.0_f32; len],
            v: vec![0.0_f32; len],
        }
    }

    fn bytes(&self) -> usize {
        (self.w.len() + self.m.len() + self.v.len()) * 4
    }

    fn step(&mut self, grad: &[f32], lr: f32, t: usize) {
        let b1 = 0.9_f32;
        let b2 = 0.999_f32;
        let eps = 1e-8_f32;
        let tp = t.clamp(1, 100_000) as i32;
        let inv_bc1 = 1.0 / (1.0 - b1.powi(tp));
        let inv_bc2 = 1.0 / (1.0 - b2.powi(tp));
        let one_m_b1 = 1.0 - b1;
        let one_m_b2 = 1.0 - b2;
        for i in 0..self.w.len() {
            let g = grad[i].clamp(-1.0, 1.0);
            let m_new = b1 * self.m[i] + one_m_b1 * g;
            let v_new = b2 * self.v[i] + one_m_b2 * g * g;
            self.m[i] = m_new;
            self.v[i] = v_new;
            let m_hat = m_new * inv_bc1;
            let v_hat = v_new * inv_bc2;
            self.w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    #[inline]
    fn bias_corr(t: usize) -> (f32, f32) {
        let b1 = 0.9_f32;
        let b2 = 0.999_f32;
        let tp = t.clamp(1, 100_000) as i32;
        (1.0 / (1.0 - b1.powi(tp)), 1.0 / (1.0 - b2.powi(tp)))
    }

    #[inline]
    fn step_row_scaled(
        &mut self,
        row: usize,
        cols: usize,
        row_scale: f32,
        col_vec: &[f32],
        lr: f32,
        inv_bc1: f32,
        inv_bc2: f32,
    ) {
        let off = row * cols;
        let w = &mut self.w[off..off + cols];
        let m = &mut self.m[off..off + cols];
        let v = &mut self.v[off..off + cols];
        for j in 0..cols {
            let g = (row_scale * col_vec[j]).clamp(-1.0, 1.0);
            let mn = 0.9_f32 * m[j] + 0.1_f32 * g;
            let vn = 0.999_f32 * v[j] + 0.001_f32 * g * g;
            m[j] = mn;
            v[j] = vn;
            w[j] -= lr * (mn * inv_bc1) / ((vn * inv_bc2).sqrt() + 1e-8_f32);
        }
    }

    #[inline]
    fn step_single(&mut self, idx: usize, g_raw: f32, lr: f32, inv_bc1: f32, inv_bc2: f32) {
        let g = g_raw.clamp(-1.0, 1.0);
        let mn = 0.9_f32 * self.m[idx] + 0.1_f32 * g;
        let vn = 0.999_f32 * self.v[idx] + 0.001_f32 * g * g;
        self.m[idx] = mn;
        self.v[idx] = vn;
        self.w[idx] -= lr * (mn * inv_bc1) / ((vn * inv_bc2).sqrt() + 1e-8_f32);
    }
}

#[derive(Clone)]
pub struct OnlineGruBaseline {
    pub x_dim: usize,
    pub h_dim: usize,
    pub num_actions: usize,
    wz: AdamParam,
    uz: AdamParam,
    bz: AdamParam,
    wr: AdamParam,
    ur: AdamParam,
    br: AdamParam,
    wn: AdamParam,
    un: AdamParam,
    bn: AdamParam,
    wo: AdamParam,
    bo: AdamParam,
    wv: AdamParam,
    bv: AdamParam,
    pub h_cur: Vec<f32>,
    cache: Vec<GruStepCache>,
    adam_step: usize,
    pub lr: f32,
    pub gamma: f32,
    pub rng: u64,
    pub had_nan: bool,
    pub ever_had_nan: bool,
    pub enable_input_clamp: bool,
    pub enable_state_weight_guard: bool,
    pub seu_fault_count: u64,
}

impl OnlineGruBaseline {
    pub fn new(x_dim: usize, h_dim: usize, num_actions: usize, seed: u64) -> Self {
        let mut rng = splitmix64(seed ^ 0x6200_BA5E_114E);
        let sx = 0.5 / (x_dim.max(1) as f32).sqrt();
        let sh = 0.5 / (h_dim.max(1) as f32).sqrt();
        Self {
            x_dim,
            h_dim,
            num_actions,
            wz: AdamParam::new(h_dim * x_dim, sx, &mut rng),
            uz: AdamParam::new(h_dim * h_dim, sh, &mut rng),
            bz: AdamParam::new(h_dim, 0.0, &mut rng),
            wr: AdamParam::new(h_dim * x_dim, sx, &mut rng),
            ur: AdamParam::new(h_dim * h_dim, sh, &mut rng),
            br: AdamParam::new(h_dim, 0.0, &mut rng),
            wn: AdamParam::new(h_dim * x_dim, sx, &mut rng),
            un: AdamParam::new(h_dim * h_dim, sh, &mut rng),
            bn: AdamParam::new(h_dim, 0.0, &mut rng),
            wo: AdamParam::new(num_actions * h_dim, sh, &mut rng),
            bo: AdamParam::new(num_actions, 0.0, &mut rng),
            wv: AdamParam::new(h_dim, sh, &mut rng),
            bv: AdamParam::new(1, 0.0, &mut rng),
            h_cur: vec![0.0_f32; h_dim],
            cache: Vec::with_capacity(16),
            adam_step: 0,
            lr: 0.006,
            gamma: 0.90,
            rng,
            had_nan: false,
            ever_had_nan: false,
            enable_input_clamp: false,
            enable_state_weight_guard: false,
            seu_fault_count: 0,
        }
    }

    pub fn new_supervised(x_dim: usize, h_dim: usize, num_actions: usize, seed: u64) -> Self {
        let mut rng = splitmix64(seed ^ 0x6200_BA5E_114E);
        let sx = 0.5 / (x_dim.max(1) as f32).sqrt();
        let sh = 0.5 / (h_dim.max(1) as f32).sqrt();
        Self {
            x_dim,
            h_dim,
            num_actions,
            wz: AdamParam::new(h_dim * x_dim, sx, &mut rng),
            uz: AdamParam::new(h_dim * h_dim, sh, &mut rng),
            bz: AdamParam::new(h_dim, 0.0, &mut rng),
            wr: AdamParam::new(h_dim * x_dim, sx, &mut rng),
            ur: AdamParam::new(h_dim * h_dim, sh, &mut rng),
            br: AdamParam::new(h_dim, 0.0, &mut rng),
            wn: AdamParam::new(h_dim * x_dim, sx, &mut rng),
            un: AdamParam::new(h_dim * h_dim, sh, &mut rng),
            bn: AdamParam::new(h_dim, 0.0, &mut rng),
            wo: AdamParam::new(num_actions * h_dim, sh, &mut rng),
            bo: AdamParam::new(num_actions, 0.0, &mut rng),
            wv: AdamParam { w: Vec::new(), m: Vec::new(), v: Vec::new() },
            bv: AdamParam { w: Vec::new(), m: Vec::new(), v: Vec::new() },
            h_cur: vec![0.0_f32; h_dim],
            cache: Vec::new(),
            adam_step: 0,
            lr: 0.006,
            gamma: 0.90,
            rng,
            had_nan: false,
            ever_had_nan: false,
            enable_input_clamp: false,
            enable_state_weight_guard: false,
            seu_fault_count: 0,
        }
    }

    pub fn scrub_state_and_weights(&mut self) -> u64 {
        let mut f = scrub_slice_seu(&mut self.h_cur, 24.0, 0.0);
        for p in [
            &mut self.wz, &mut self.uz, &mut self.bz,
            &mut self.wr, &mut self.ur, &mut self.br,
            &mut self.wn, &mut self.un, &mut self.bn,
            &mut self.wo, &mut self.bo, &mut self.wv, &mut self.bv,
        ] {
            f = f.saturating_add(scrub_slice_seu(&mut p.w, 16.1, 0.0));
            for m in p.m.iter_mut() {
                if !m.is_finite() || m.abs() > 100.0 {
                    *m = 0.0;
                }
            }
            for v in p.v.iter_mut() {
                if !v.is_finite() || *v < 0.0 || *v > 1e6 {
                    *v = 0.0;
                }
            }
        }
        self.seu_fault_count = self.seu_fault_count.saturating_add(f);
        f
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.wz.bytes()
            + self.uz.bytes()
            + self.bz.bytes()
            + self.wr.bytes()
            + self.ur.bytes()
            + self.br.bytes()
            + self.wn.bytes()
            + self.un.bytes()
            + self.bn.bytes()
            + self.wo.bytes()
            + self.bo.bytes()
            + self.wv.bytes()
            + self.bv.bytes()
            + self.h_cur.len() * 4
    }

    pub fn start_episode(&mut self) {
        self.h_cur.fill(0.0);
        self.cache.clear();
        self.had_nan = false;
    }

    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        if self.h_cur.is_empty() {
            return;
        }
        let idx = reg_idx % self.h_cur.len();
        self.h_cur[idx] = f32::from_bits(self.h_cur[idx].to_bits() ^ (1u32 << (bit & 31)));
    }

    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let mut slices: [&mut [f32]; 13] = [
            &mut self.wz.w,
            &mut self.uz.w,
            &mut self.bz.w,
            &mut self.wr.w,
            &mut self.ur.w,
            &mut self.br.w,
            &mut self.wn.w,
            &mut self.un.w,
            &mut self.bn.w,
            &mut self.wo.w,
            &mut self.bo.w,
            &mut self.wv.w,
            &mut self.bv.w,
        ];
        flip_weight_in_slices(&mut slices, reg_idx, bit);
    }

    fn encode_obs(&self, obs: &str, seu_nan: bool, seu_spikes: bool) -> Vec<f32> {
        let mut x = vec![0.0_f32; self.x_dim];
        for tok in extract_tokens(obs) {
            let h = fnv1a_bytes(tok.as_bytes(), 0x5151_2323);
            let idx = (h as usize) % self.x_dim;
            let sign = if ((h >> 16) & 1) == 0 { 1.0 } else { -1.0 };
            x[idx] += sign;
        }
        let nsq: f32 = x.iter().map(|v| v * v).sum();
        if nsq > 1e-8 {
            let inv = 1.0 / nsq.sqrt();
            for v in &mut x {
                *v *= inv;
            }
        }
        if seu_nan {
            x[0] = f32::NAN;
        }
        if seu_spikes {
            x[1] = 1e28;
        }
        x
    }

    #[inline]
    fn sigmoid(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

    pub fn decide(&mut self, obs: &str, train_mode: bool, seu_nan: bool, seu_spikes: bool) -> usize {
        if self.enable_state_weight_guard {
            let _ = self.scrub_state_and_weights();
        }
        let x = self.encode_obs(obs, seu_nan, seu_spikes);
        let h_prev = self.h_cur.clone();
        let hd = self.h_dim;
        let xd = self.x_dim;

        let mut z = vec![0.0_f32; hd];
        let mut r = vec![0.0_f32; hd];
        let mut n = vec![0.0_f32; hd];
        let mut h = vec![0.0_f32; hd];

        for i in 0..hd {
            let mut sz = self.bz.w[i];
            let mut sr = self.br.w[i];
            for j in 0..xd {
                sz += self.wz.w[i * xd + j] * x[j];
                sr += self.wr.w[i * xd + j] * x[j];
            }
            for j in 0..hd {
                sz += self.uz.w[i * hd + j] * h_prev[j];
                sr += self.ur.w[i * hd + j] * h_prev[j];
            }
            z[i] = Self::sigmoid(sz);
            r[i] = Self::sigmoid(sr);
        }
        for i in 0..hd {
            let mut sn = self.bn.w[i];
            for j in 0..xd {
                sn += self.wn.w[i * xd + j] * x[j];
            }
            for j in 0..hd {
                sn += self.un.w[i * hd + j] * (r[j] * h_prev[j]);
            }
            n[i] = sn.tanh();
            h[i] = (1.0 - z[i]) * h_prev[i] + z[i] * n[i];
        }

        let mut v_pred = self.bv.w[0];
        for j in 0..hd {
            v_pred += self.wv.w[j] * h[j];
        }
        if !v_pred.is_finite() || h.iter().any(|v| !v.is_finite()) {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
        }

        let mut logits = vec![0.0_f32; self.num_actions];
        let mut max_l = f32::NEG_INFINITY;
        for a in 0..self.num_actions {
            let mut s = self.bo.w[a];
            for j in 0..hd {
                s += self.wo.w[a * hd + j] * h[j];
            }
            if !s.is_finite() {
                if !self.had_nan {
                    self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                }
                self.had_nan = true;
                self.ever_had_nan = true;
            }
            logits[a] = s;
            if s > max_l {
                max_l = s;
            }
        }

        let mut probs = vec![0.0_f32; self.num_actions];
        let mut sum_e = 0.0_f32;
        for a in 0..self.num_actions {
            let e = (logits[a] - max_l).exp();
            probs[a] = e;
            sum_e += e;
        }
        if !sum_e.is_finite() || sum_e <= 0.0 {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
            probs.fill(1.0 / (self.num_actions as f32));
        } else {
            for p in &mut probs {
                *p /= sum_e;
            }
        }

        let action = if train_mode {
            let u = rand_unit_f32(&mut self.rng);
            let mut cdf = 0.0_f32;
            let mut picked = self.num_actions - 1;
            for (a, &p) in probs.iter().enumerate() {
                cdf += p;
                if u <= cdf {
                    picked = a;
                    break;
                }
            }
            picked
        } else {
            let mut best = 0usize;
            let mut best_p = -1.0_f32;
            for (a, &p) in probs.iter().enumerate() {
                if p > best_p {
                    best_p = p;
                    best = a;
                }
            }
            best
        };

        self.h_cur = h.clone();
        self.cache.push(GruStepCache {
            x,
            h_prev,
            z,
            r,
            n,
            h,
            probs,
            v_pred,
            action,
            reward: 0.0,
        });
        action
    }

    /// Числовой вход `&[f32]` для POMDP-бенчмарков (T-Maze, RepeatPrevious, Position-Only CartPole).
    pub fn decide_vec(
        &mut self,
        features: &[f32],
        train_mode: bool,
        seu_nan: bool,
        seu_spikes: bool,
    ) -> usize {
        let mut x = vec![0.0_f32; self.x_dim];
        for i in 0..self.x_dim.min(features.len()) {
            x[i] = features[i];
        }
        if seu_nan && !x.is_empty() {
            x[0] = f32::NAN;
        }
        if seu_spikes && x.len() > 1 {
            x[1] = 1e28;
        }
        if self.enable_input_clamp {
            for v in &mut x {
                *v = if v.is_finite() { v.clamp(-10.0, 10.0) } else { 0.0 };
            }
        }
        self.step_with_x(x, train_mode)
    }

    fn step_with_x(&mut self, x: Vec<f32>, train_mode: bool) -> usize {
        if self.enable_state_weight_guard {
            let _ = self.scrub_state_and_weights();
        }
        let h_prev = self.h_cur.clone();
        let hd = self.h_dim;
        let xd = self.x_dim;

        let mut z = vec![0.0_f32; hd];
        let mut r = vec![0.0_f32; hd];
        let mut n = vec![0.0_f32; hd];
        let mut h = vec![0.0_f32; hd];

        for i in 0..hd {
            let mut sz = self.bz.w[i];
            let mut sr = self.br.w[i];
            for j in 0..xd {
                sz += self.wz.w[i * xd + j] * x[j];
                sr += self.wr.w[i * xd + j] * x[j];
            }
            for j in 0..hd {
                sz += self.uz.w[i * hd + j] * h_prev[j];
                sr += self.ur.w[i * hd + j] * h_prev[j];
            }
            z[i] = Self::sigmoid(sz);
            r[i] = Self::sigmoid(sr);
        }
        for i in 0..hd {
            let mut sn = self.bn.w[i];
            for j in 0..xd {
                sn += self.wn.w[i * xd + j] * x[j];
            }
            for j in 0..hd {
                sn += self.un.w[i * hd + j] * (r[j] * h_prev[j]);
            }
            n[i] = sn.tanh();
            h[i] = (1.0 - z[i]) * h_prev[i] + z[i] * n[i];
        }

        let mut v_pred = self.bv.w[0];
        for j in 0..hd {
            v_pred += self.wv.w[j] * h[j];
        }
        if !v_pred.is_finite() || h.iter().any(|v| !v.is_finite()) {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
        }

        let mut logits = vec![0.0_f32; self.num_actions];
        let mut max_l = f32::NEG_INFINITY;
        for a in 0..self.num_actions {
            let mut s = self.bo.w[a];
            for j in 0..hd {
                s += self.wo.w[a * hd + j] * h[j];
            }
            if !s.is_finite() {
                if !self.had_nan {
                    self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                }
                self.had_nan = true;
                self.ever_had_nan = true;
            }
            logits[a] = s;
            if s > max_l {
                max_l = s;
            }
        }

        let mut probs = vec![0.0_f32; self.num_actions];
        let mut sum_e = 0.0_f32;
        for a in 0..self.num_actions {
            let e = (logits[a] - max_l).exp();
            probs[a] = e;
            sum_e += e;
        }
        if !sum_e.is_finite() || sum_e <= 0.0 {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
            probs.fill(1.0 / (self.num_actions as f32));
        } else {
            for p in &mut probs {
                *p /= sum_e;
            }
        }

        let action = if train_mode {
            let u = rand_unit_f32(&mut self.rng);
            let mut cdf = 0.0_f32;
            let mut picked = self.num_actions - 1;
            for (a, &p) in probs.iter().enumerate() {
                cdf += p;
                if u <= cdf {
                    picked = a;
                    break;
                }
            }
            picked
        } else {
            let mut best = 0usize;
            let mut best_p = -1.0_f32;
            for (a, &p) in probs.iter().enumerate() {
                if p > best_p {
                    best_p = p;
                    best = a;
                }
            }
            best
        };

        self.h_cur = h.clone();
        self.cache.push(GruStepCache {
            x,
            h_prev,
            z,
            r,
            n,
            h,
            probs,
            v_pred,
            action,
            reward: 0.0,
        });
        action
    }

    pub fn observe_step_reward(&mut self, r: f32) {
        if let Some(last) = self.cache.last_mut() {
            last.reward = r;
        }
    }

    /// Точный A2C + BPTT + Entropy Regularization + Adam.
    pub fn end_episode(&mut self, final_return: f32, train_mode: bool) {
        self.end_episode_truncated(final_return, false, train_mode);
    }

    pub fn end_episode_truncated(&mut self, final_return: f32, truncated: bool, train_mode: bool) {
        if !train_mode || self.cache.is_empty() || self.had_nan {
            return;
        }
        let t_len = self.cache.len();
        let has_dense = self.cache.iter().any(|c| c.reward.abs() > 1e-6);
        let mut returns = vec![0.0_f32; t_len];
        if has_dense {
            let mut run = if truncated { self.cache[t_len - 1].v_pred } else { 0.0_f32 };
            for t in (0..t_len).rev() {
                run = self.cache[t].reward + self.gamma * run;
                returns[t] = run;
            }
        } else {
            let mut g = final_return;
            for t in (0..t_len).rev() {
                returns[t] = g;
                g *= self.gamma;
            }
        }

        let hd = self.h_dim;
        let xd = self.x_dim;
        let na = self.num_actions;

        let mut gwz = vec![0.0_f32; hd * xd];
        let mut guz = vec![0.0_f32; hd * hd];
        let mut gbz = vec![0.0_f32; hd];
        let mut gwr = vec![0.0_f32; hd * xd];
        let mut gur = vec![0.0_f32; hd * hd];
        let mut gbr = vec![0.0_f32; hd];
        let mut gwn = vec![0.0_f32; hd * xd];
        let mut gun = vec![0.0_f32; hd * hd];
        let mut gbn = vec![0.0_f32; hd];
        let mut gwo = vec![0.0_f32; na * hd];
        let mut gbo = vec![0.0_f32; na];
        let mut gwv = vec![0.0_f32; hd];
        let mut gbv = vec![0.0_f32; 1];

        let mut dh_next = vec![0.0_f32; hd];
        let ent_beta = 0.015_f32;

        for t in (0..t_len).rev() {
            let c = &self.cache[t];
            let adv = (returns[t] - c.v_pred).clamp(-2.0, 2.0);

            // Critic loss gradient: 0.5 * (v - G)^2 -> dv = (v - G)
            let dv = 0.5 * (c.v_pred - returns[t]).clamp(-2.0, 2.0);
            gbv[0] += dv;
            let mut dh = dh_next.clone();
            for j in 0..hd {
                gwv[j] += dv * c.h[j];
                dh[j] += self.wv.w[j] * dv;
            }

            // Actor + entropy gradient
            let mut ent_mean_logp = 0.0_f32;
            for a in 0..na {
                let p = c.probs[a].max(1e-6);
                ent_mean_logp += p * p.ln();
            }
            for a in 0..na {
                let p = c.probs[a].max(1e-6);
                let d_pg = adv * (p - if a == c.action { 1.0 } else { 0.0 });
                let d_ent = ent_beta * p * (p.ln() - ent_mean_logp);
                let dlog = d_pg + d_ent;
                gbo[a] += dlog;
                for j in 0..hd {
                    gwo[a * hd + j] += dlog * c.h[j];
                    dh[j] += self.wo.w[a * hd + j] * dlog;
                }
            }

            let mut dn_pre = vec![0.0_f32; hd];
            let mut dz_pre = vec![0.0_f32; hd];
            let mut dh_prev = vec![0.0_f32; hd];
            let mut dr_h = vec![0.0_f32; hd];

            for i in 0..hd {
                let dn = dh[i] * c.z[i];
                dn_pre[i] = dn * (1.0 - c.n[i] * c.n[i]);
                let dz = dh[i] * (c.n[i] - c.h_prev[i]);
                dz_pre[i] = dz * c.z[i] * (1.0 - c.z[i]);
                dh_prev[i] += dh[i] * (1.0 - c.z[i]);

                gbn[i] += dn_pre[i];
                gbz[i] += dz_pre[i];
                for j in 0..xd {
                    gwn[i * xd + j] += dn_pre[i] * c.x[j];
                    gwz[i * xd + j] += dz_pre[i] * c.x[j];
                }
                for j in 0..hd {
                    gun[i * hd + j] += dn_pre[i] * (c.r[j] * c.h_prev[j]);
                    dr_h[j] += self.un.w[i * hd + j] * dn_pre[i];
                    guz[i * hd + j] += dz_pre[i] * c.h_prev[j];
                    dh_prev[j] += self.uz.w[i * hd + j] * dz_pre[i];
                }
            }

            for i in 0..hd {
                dh_prev[i] += dr_h[i] * c.r[i];
                let dr_pre = dr_h[i] * c.h_prev[i] * c.r[i] * (1.0 - c.r[i]);
                gbr[i] += dr_pre;
                for j in 0..xd {
                    gwr[i * xd + j] += dr_pre * c.x[j];
                }
                for j in 0..hd {
                    gur[i * hd + j] += dr_pre * c.h_prev[j];
                    dh_prev[j] += self.ur.w[i * hd + j] * dr_pre;
                }
            }
            dh_next = dh_prev;
        }

        self.adam_step += 1;
        let st = self.adam_step;
        let lr = self.lr;
        self.wz.step(&gwz, lr, st);
        self.uz.step(&guz, lr, st);
        self.bz.step(&gbz, lr, st);
        self.wr.step(&gwr, lr, st);
        self.ur.step(&gur, lr, st);
        self.br.step(&gbr, lr, st);
        self.wn.step(&gwn, lr, st);
        self.un.step(&gun, lr, st);
        self.bn.step(&gbn, lr, st);
        self.wo.step(&gwo, lr, st);
        self.bo.step(&gbo, lr, st);
        self.wv.step(&gwv, lr, st);
        self.bv.step(&gbv, lr, st);
    }
}

// =============================================================================
// 4. MFEC + Page-Hinkley (Model-Free Episodic Control, Blundell et al. 2016)
// =============================================================================

struct MfecEntry {
    key: Vec<f32>,
    q: Vec<f32>,
    visits: Vec<u32>,
    last_used: u64,
}

pub struct MfecPHBaseline {
    pub dim: usize,
    pub num_actions: usize,
    pub capacity: usize,
    pub k_nn: usize,
    pub coalesce_cos: f32,
    pub use_belief_ema: bool,
    banks: Vec<Vec<MfecEntry>>,
    phs: Vec<PageHinkleyDetector>,
    pub active_bank: usize,
    trace: Vec<(Vec<f32>, usize, f32)>,
    belief_state: Vec<f32>,
    pub tick: u64,
    pub rng: u64,
    pub eps: f32,
    pub gamma: f32,
}

impl MfecPHBaseline {
    pub fn new(dim: usize, num_actions: usize, capacity: usize, max_banks: usize, seed: u64) -> Self {
        let mut banks = Vec::with_capacity(max_banks);
        let mut phs = Vec::with_capacity(max_banks);
        for _ in 0..max_banks {
            banks.push(Vec::with_capacity(capacity));
            phs.push(PageHinkleyDetector::new(0.20, 3.0, 40));
        }
        Self {
            dim,
            num_actions,
            capacity,
            k_nn: 5,
            coalesce_cos: 0.94,
            use_belief_ema: false,
            banks,
            phs,
            active_bank: 0,
            trace: Vec::with_capacity(24),
            belief_state: vec![0.0_f32; dim],
            tick: 0,
            rng: splitmix64(seed ^ 0x4FEC_BA5E),
            eps: 0.14,
            gamma: 0.90,
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        let entries: usize = self
            .banks
            .iter()
            .map(|b| b.len() * (self.dim * 4 + self.num_actions * 8 + 16))
            .sum();
        entries + self.phs.len() * 40 + self.belief_state.len() * 4
    }

    pub fn start_episode(&mut self) {
        self.trace.clear();
        self.belief_state.fill(0.0);
    }

    fn encode_obs(&self, obs: &str) -> Vec<f32> {
        let mut v = vec![0.0_f32; self.dim];
        for tok in extract_tokens(obs) {
            // Match SpinorEncoder's numeric downweighting when belief mode is enabled
            let is_num = !tok.is_empty() && tok.bytes().all(|b| b.is_ascii_digit());
            let w = if self.use_belief_ema && is_num { 0.15_f32 } else { 1.0_f32 };
            let h1 = fnv1a_bytes(tok.as_bytes(), 0x3141_5926_5358_9793);
            let h2 = splitmix64(h1 ^ 0x2718_2818_2845_9045);
            for k in 0..4usize {
                let m = splitmix64(h2.wrapping_add((k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
                let idx = (m as usize) % self.dim;
                let sgn = if ((m >> 20) & 1) == 0 { w } else { -w };
                v[idx] += sgn;
            }
        }
        let nsq: f32 = v.iter().map(|x| x * x).sum();
        if nsq > 1e-8 {
            let inv = 1.0 / nsq.sqrt();
            for x in &mut v {
                *x *= inv;
            }
        }
        v
    }

    pub fn decide(&mut self, obs: &str, train_mode: bool) -> usize {
        self.tick = self.tick.saturating_add(1);
        let obs_v = self.encode_obs(obs);
        let key = if self.use_belief_ema {
            let is_first = self.trace.is_empty();
            if is_first {
                self.belief_state.copy_from_slice(&obs_v);
            } else {
                // Simple non-Cayley exponential moving average state memory (lambda = 0.65)
                let mut nsq = 0.0_f32;
                for i in 0..self.dim {
                    let nv = 0.65 * self.belief_state[i] + 0.76 * obs_v[i];
                    self.belief_state[i] = nv;
                    nsq += nv * nv;
                }
                if nsq > 1e-8 {
                    let inv = 1.0 / nsq.sqrt();
                    for v in &mut self.belief_state {
                        *v *= inv;
                    }
                }
            }
            self.belief_state.clone()
        } else {
            obs_v
        };
        let bank = &self.banks[self.active_bank];
        let mut q_est = vec![0.0_f32; self.num_actions];
        let mut w_sum = vec![0.0_f32; self.num_actions];

        if !bank.is_empty() {
            let mut sims: Vec<(f32, usize)> = bank
                .iter()
                .enumerate()
                .map(|(idx, e)| {
                    let dot: f32 = e.key.iter().zip(key.iter()).map(|(a, b)| a * b).sum();
                    (dot, idx)
                })
                .collect();
            sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for &(sim, idx) in sims.iter().take(self.k_nn) {
                if sim < 0.30 {
                    break;
                }
                let w = sim * sim;
                let e = &bank[idx];
                for a in 0..self.num_actions {
                    if e.visits[a] > 0 {
                        q_est[a] += w * e.q[a];
                        w_sum[a] += w;
                    }
                }
            }
        }

        let act = if train_mode && rand_unit_f32(&mut self.rng) < self.eps {
            self.rng = splitmix64(self.rng);
            (self.rng as usize) % self.num_actions
        } else {
            let mut best = 0usize;
            let mut best_v = -1e9_f32;
            for a in 0..self.num_actions {
                let v = if w_sum[a] > 1e-6 { q_est[a] / w_sum[a] } else { 0.0 };
                let jitter = if train_mode { rand_signed_f32(&mut self.rng) * 0.02 } else { 0.0 };
                if v + jitter > best_v {
                    best_v = v + jitter;
                    best = a;
                }
            }
            best
        };
        self.trace.push((key, act, 0.0));
        act
    }

    pub fn observe_step_reward(&mut self, r: f32) {
        if let Some(last) = self.trace.last_mut() {
            last.2 = r;
        }
    }

    pub fn end_episode_with_solved(&mut self, final_return: f32, solved: bool, train_mode: bool) {
        if !train_mode {
            return;
        }
        let has_dense = self.trace.iter().any(|x| x.2.abs() > 1e-6);
        let mut g = final_return;
        let mut running = 0.0_f32;
        let bank = &mut self.banks[self.active_bank];

        for (key, a, r_step) in self.trace.iter().rev() {
            let target = if has_dense {
                running = *r_step + self.gamma * running;
                running
            } else {
                let cur = g;
                g *= self.gamma;
                cur
            };
            let mut best_idx = None;
            let mut best_sim = self.coalesce_cos;
            for (i, e) in bank.iter().enumerate() {
                let dot: f32 = e.key.iter().zip(key.iter()).map(|(u, v)| u * v).sum();
                if dot > best_sim {
                    best_sim = dot;
                    best_idx = Some(i);
                }
            }
            if let Some(idx) = best_idx {
                let e = &mut bank[idx];
                e.last_used = self.tick;
                if e.visits[*a] == 0 {
                    e.q[*a] = target;
                } else {
                    e.q[*a] = 0.65 * e.q[*a].max(target) + 0.35 * target;
                }
                e.visits[*a] = e.visits[*a].saturating_add(1);
            } else {
                let mut e = MfecEntry {
                    key: key.clone(),
                    q: vec![0.0_f32; self.num_actions],
                    visits: vec![0u32; self.num_actions],
                    last_used: self.tick,
                };
                e.q[*a] = target;
                e.visits[*a] = 1;
                if bank.len() < self.capacity {
                    bank.push(e);
                } else if let Some((lru_i, _)) = bank.iter().enumerate().min_by_key(|(_, x)| x.last_used) {
                    bank[lru_i] = e;
                }
            }
        }
        let r_norm = if solved { 1.0_f32 } else { 0.0_f32 };
        if self.phs[self.active_bank].observe(r_norm) {
            self.phs[self.active_bank].reset_cusum();
            self.active_bank = (self.active_bank + 1) % self.banks.len();
        }
    }
}

// =============================================================================
// 5. RTRRL-LRU (AAAI 2025): Diagonal Complex Linear Recurrent Unit + Exact RTRL
// =============================================================================

#[derive(Clone)]
pub struct RtrrlLruBaseline {
    pub in_dim: usize,
    pub units: usize,       // M комплексных единиц (2M вещественных координат состояния)
    pub num_actions: usize,
    pub kappa: f32,
    // Комплексные параметры LRU: r_j = exp(-exp(nu_j)), lambda_j = r_j * exp(i * theta_j), gamma_j = sqrt(1 - r_j^2)
    pub nu: Vec<f32>,       // [M]
    pub theta: Vec<f32>,    // [M]
    pub b_re: Vec<f32>,     // [M * in_dim]
    pub b_im: Vec<f32>,     // [M * in_dim]
    // RMSProp-статистики для внутренних параметров LRU
    v_nu: Vec<f32>,
    v_theta: Vec<f32>,
    v_bre: Vec<f32>,
    v_bim: Vec<f32>,
    // Головки Actor-Critic на [Re(y), Im(y), x] размерности f_dim = 2*M + in_dim
    actor_w: Vec<f32>,      // [num_actions * f_dim]
    actor_b: Vec<f32>,      // [num_actions]
    critic_w: Vec<f32>,     // [f_dim]
    critic_b: f32,
    // Runtime-состояние и точные RTRL-чувствительности O(M * in_dim)
    pub h_re: Vec<f32>,
    pub h_im: Vec<f32>,
    y_re: Vec<f32>,
    y_im: Vec<f32>,
    rho: Vec<f32>,
    pub p_nu_re: Vec<f32>,
    pub p_nu_im: Vec<f32>,
    pub p_th_re: Vec<f32>,
    pub p_th_im: Vec<f32>,
    pub p_b_re: Vec<f32>,   // dh / d(b_re); dh / d(b_im) == i * p_b (точная комплексная линейность!)
    pub p_b_im: Vec<f32>,
    // Следы приемлемости TD(lambda)
    e_nu: Vec<f32>,
    e_th: Vec<f32>,
    e_bre: Vec<f32>,
    e_bim: Vec<f32>,
    e_actor_w: Vec<f32>,
    e_actor_b: Vec<f32>,
    e_critic_w: Vec<f32>,
    e_critic_b: f32,
    // Онлайн TD-состояние
    has_pending: bool,
    pending_v: f32,
    pending_r: f32,
    td_mean: f32,
    td_var: f32,
    // Предвыделенные буферы шага (0 аллокаций в куче на шаге!)
    x_buf: Vec<f32>,
    feat_buf: Vec<f32>,
    logits_buf: Vec<f32>,
    probs_buf: Vec<f32>,
    fb_re_buf: Vec<f32>,
    fb_im_buf: Vec<f32>,
    pub gamma: f32,
    pub trace_lambda: f32,
    pub actor_lr: f32,
    pub critic_lr: f32,
    pub rtrl_lr: f32,
    pub entropy_beta: f32,
    pub rng: u64,
    pub had_nan: bool,
    pub ever_had_nan: bool,
    pub enable_input_clamp: bool,
    pub enable_state_weight_guard: bool,
    pub seu_fault_count: u64,
}

impl RtrrlLruBaseline {
    pub fn new(in_dim: usize, units: usize, num_actions: usize, seed: u64) -> Self {
        Self::new_inner(in_dim, units, num_actions, seed, true)
    }

    /// Конструктор чистого рекуррентного слоя без RL-головок (0 байт мёртвой памяти для `DeepRtrrlLruBank`).
    pub fn new_backbone(in_dim: usize, units: usize, seed: u64) -> Self {
        Self::new_inner(in_dim, units, 0, seed, false)
    }

    fn new_inner(in_dim: usize, units: usize, num_actions: usize, seed: u64, with_heads: bool) -> Self {
        let mut rng = splitmix64(seed ^ 0x2721_1200_AA41);
        let f_dim = if with_heads { 2 * units + in_dim } else { 0 };
        let na = if with_heads { num_actions } else { 0 };
        let mut nu = vec![0.0_f32; units];
        let mut theta = vec![0.0_f32; units];
        let mut b_re = vec![0.0_f32; units * in_dim];
        let mut b_im = vec![0.0_f32; units * in_dim];
        let scale_b = 0.65 / (in_dim.max(1) as f32).sqrt();

        let r_min = 0.55_f32;
        let r_max = 0.99_f32;
        let r_min2 = r_min * r_min;
        let r_max2 = r_max * r_max;
        for j in 0..units {
            // Независимая инициализация на кольце (r_j, theta_j), как в Orvieto et al. (2023) / AAAI 2025 RTRRL
            let u_r = rand_unit_f32(&mut rng);
            let r_init = (r_min2 + u_r * (r_max2 - r_min2)).sqrt();
            nu[j] = (-r_init.ln()).ln();
            let u_th = rand_unit_f32(&mut rng);
            theta[j] = (u_th * 2.0 - 1.0) * std::f32::consts::PI;
            for k in 0..in_dim {
                b_re[j * in_dim + k] = rand_signed_f32(&mut rng) * scale_b;
                b_im[j * in_dim + k] = rand_signed_f32(&mut rng) * scale_b;
            }
        }
        let mut actor_w = if with_heads { vec![0.0_f32; na * f_dim] } else { Vec::new() };
        let mut critic_w = if with_heads { vec![0.0_f32; f_dim] } else { Vec::new() };
        for w in &mut actor_w {
            *w = rand_signed_f32(&mut rng) * 0.05;
        }
        for w in &mut critic_w {
            *w = rand_signed_f32(&mut rng) * 0.05;
        }
        Self {
            in_dim,
            units,
            num_actions: na,
            kappa: 0.5,
            nu,
            theta,
            b_re,
            b_im,
            v_nu: vec![0.04; units],
            v_theta: vec![0.04; units],
            v_bre: vec![0.04; units * in_dim],
            v_bim: vec![0.04; units * in_dim],
            actor_w,
            actor_b: if with_heads { vec![0.0; na] } else { Vec::new() },
            critic_w,
            critic_b: 0.0,
            h_re: vec![0.0; units],
            h_im: vec![0.0; units],
            y_re: vec![0.0; units],
            y_im: vec![0.0; units],
            rho: vec![0.5; units],
            p_nu_re: vec![0.0; units],
            p_nu_im: vec![0.0; units],
            p_th_re: vec![0.0; units],
            p_th_im: vec![0.0; units],
            p_b_re: vec![0.0; units * in_dim],
            p_b_im: vec![0.0; units * in_dim],
            e_nu: vec![0.0; units],
            e_th: vec![0.0; units],
            e_bre: vec![0.0; units * in_dim],
            e_bim: vec![0.0; units * in_dim],
            e_actor_w: if with_heads { vec![0.0; na * f_dim] } else { Vec::new() },
            e_actor_b: if with_heads { vec![0.0; na] } else { Vec::new() },
            e_critic_w: if with_heads { vec![0.0; f_dim] } else { Vec::new() },
            e_critic_b: 0.0,
            has_pending: false,
            pending_v: 0.0,
            pending_r: 0.0,
            td_mean: 0.0,
            td_var: 0.25,
            x_buf: if with_heads { vec![0.0; in_dim] } else { Vec::new() },
            feat_buf: if with_heads { vec![0.0; f_dim] } else { Vec::new() },
            logits_buf: if with_heads { vec![0.0; na] } else { Vec::new() },
            probs_buf: if with_heads { vec![0.0; na] } else { Vec::new() },
            fb_re_buf: if with_heads { vec![0.0; units] } else { Vec::new() },
            fb_im_buf: if with_heads { vec![0.0; units] } else { Vec::new() },
            gamma: 0.95,
            trace_lambda: 0.80,
            actor_lr: 0.045,
            critic_lr: 0.12,
            rtrl_lr: 0.006,
            entropy_beta: 0.008,
            rng,
            had_nan: false,
            ever_had_nan: false,
            enable_input_clamp: false,
            enable_state_weight_guard: false,
            seu_fault_count: 0,
        }
    }

    pub fn scrub_state_and_weights(&mut self) -> u64 {
        let mut f = scrub_slice_seu(&mut self.h_re, 24.0, 0.0);
        f = f.saturating_add(scrub_slice_seu(&mut self.h_im, 24.0, 0.0));
        for v in self.nu.iter_mut() {
            if !v.is_finite() || *v < -6.1 || *v > 6.1 {
                *v = -2.0;
                f = f.saturating_add(1);
            }
        }
        f = f.saturating_add(scrub_slice_seu(&mut self.theta, 6.5, 0.0));
        f = f.saturating_add(scrub_slice_seu(&mut self.b_re, 16.1, 0.0));
        f = f.saturating_add(scrub_slice_seu(&mut self.b_im, 16.1, 0.0));
        f = f.saturating_add(scrub_slice_seu(&mut self.actor_w, 16.1, 0.0));
        f = f.saturating_add(scrub_slice_seu(&mut self.actor_b, 16.1, 0.0));
        f = f.saturating_add(scrub_slice_seu(&mut self.critic_w, 16.1, 0.0));
        if !self.critic_b.is_finite() || self.critic_b.abs() > 26.0 {
            self.critic_b = 0.0;
            f = f.saturating_add(1);
        }
        self.seu_fault_count = self.seu_fault_count.saturating_add(f);
        f
    }

    pub fn total_seu_faults(&self) -> u64 {
        self.seu_fault_count
    }

    pub fn ever_had_nan(&self) -> bool {
        self.ever_had_nan
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        (self.nu.len()
            + self.theta.len()
            + self.b_re.len()
            + self.b_im.len()
            + self.v_nu.len()
            + self.v_theta.len()
            + self.v_bre.len()
            + self.v_bim.len()
            + self.actor_w.len()
            + self.actor_b.len()
            + self.critic_w.len()
            + self.h_re.len()
            + self.h_im.len()
            + self.y_re.len()
            + self.y_im.len()
            + self.rho.len()
            + self.p_nu_re.len()
            + self.p_nu_im.len()
            + self.p_th_re.len()
            + self.p_th_im.len()
            + self.p_b_re.len()
            + self.p_b_im.len()
            + self.e_nu.len()
            + self.e_th.len()
            + self.e_bre.len()
            + self.e_bim.len()
            + self.e_actor_w.len()
            + self.e_actor_b.len()
            + self.e_critic_w.len()
            + self.x_buf.len()
            + self.feat_buf.len()
            + self.logits_buf.len()
            + self.probs_buf.len()
            + self.fb_re_buf.len()
            + self.fb_im_buf.len())
            * 4
            + 64
    }

    pub fn start_episode(&mut self) {
        self.h_re.fill(0.0);
        self.h_im.fill(0.0);
        self.y_re.fill(0.0);
        self.y_im.fill(0.0);
        self.rho.fill(self.kappa);
        self.p_nu_re.fill(0.0);
        self.p_nu_im.fill(0.0);
        self.p_th_re.fill(0.0);
        self.p_th_im.fill(0.0);
        self.p_b_re.fill(0.0);
        self.p_b_im.fill(0.0);
        self.e_nu.fill(0.0);
        self.e_th.fill(0.0);
        self.e_bre.fill(0.0);
        self.e_bim.fill(0.0);
        self.e_actor_w.fill(0.0);
        self.e_actor_b.fill(0.0);
        self.e_critic_w.fill(0.0);
        self.e_critic_b = 0.0;
        self.has_pending = false;
        self.pending_v = 0.0;
        self.pending_r = 0.0;
        self.had_nan = false;
    }

    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        if self.units == 0 {
            return;
        }
        let tot = self.units * 2;
        let p = reg_idx % tot;
        let u = p / 2;
        let mask = 1u32 << (bit & 31);
        if (p & 1) == 0 {
            self.h_re[u] = f32::from_bits(self.h_re[u].to_bits() ^ mask);
        } else {
            self.h_im[u] = f32::from_bits(self.h_im[u].to_bits() ^ mask);
        }
    }

    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let mut slices: [&mut [f32]; 6] = [
            &mut self.nu,
            &mut self.theta,
            &mut self.b_re,
            &mut self.b_im,
            &mut self.actor_w,
            &mut self.critic_w,
        ];
        flip_weight_in_slices(&mut slices, reg_idx, bit);
    }

    /// Один прямой такт комплексного LRU + точное обновление чувствительностей RTRL (exp/sin/cos).
    pub fn step_lru_forward(&mut self, x: &[f32]) {
        if self.enable_state_weight_guard {
            let _ = self.scrub_state_and_weights();
        }
        let m = self.units;
        let d = self.in_dim;
        let x_len = x.len().min(d);
        let kap2 = self.kappa * self.kappa;

        for j in 0..m {
            let nu_c = self.nu[j].clamp(-5.2, 1.5);
            let exp_nu = nu_c.exp();
            let r = (-exp_nu).exp();
            let th = self.theta[j];
            let cth = th.cos();
            let sth = th.sin();
            let lam_r = r * cth;
            let lam_i = r * sth;
            let gam = (1.0 - r * r).max(1e-4).sqrt();
            // d(gam)/d(nu) = r^2 * exp(nu) / gam
            let dgam_dnu = (r * r * exp_nu) / gam;

            let hr_prev = self.h_re[j];
            let hi_prev = self.h_im[j];
            let lhr = lam_r * hr_prev - lam_i * hi_prev;
            let lhi = lam_r * hi_prev + lam_i * hr_prev;

            let mut ur = 0.0_f32;
            let mut ui = 0.0_f32;
            let off = j * d;
            for k in 0..x_len {
                let xk = x[k];
                if !xk.is_finite() {
                    if !self.had_nan {
                        self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                    }
                    self.had_nan = true;
                    self.ever_had_nan = true;
                }
                ur += self.b_re[off + k] * xk;
                ui += self.b_im[off + k] * xk;
            }

            let hr = lhr + gam * ur;
            let hi = lhi + gam * ui;
            if !hr.is_finite() || !hi.is_finite() {
                if !self.had_nan {
                    self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                }
                self.had_nan = true;
                self.ever_had_nan = true;
            }
            self.h_re[j] = hr;
            self.h_im[j] = hi;

            let rho = (hr * hr + hi * hi + kap2).sqrt();
            self.rho[j] = rho;
            self.y_re[j] = hr / rho;
            self.y_im[j] = hi / rho;
            if !rho.is_finite() || !self.y_re[j].is_finite() || !self.y_im[j].is_finite() {
                if !self.had_nan {
                    self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                }
                self.had_nan = true;
                self.ever_had_nan = true;
            }

            // 1. p_nu = lam * p_nu_prev - exp(nu) * (lam * h_prev) + dgam_dnu * u
            let pnr = self.p_nu_re[j];
            let pni = self.p_nu_im[j];
            self.p_nu_re[j] = (lam_r * pnr - lam_i * pni) - exp_nu * lhr + dgam_dnu * ur;
            self.p_nu_im[j] = (lam_r * pni + lam_i * pnr) - exp_nu * lhi + dgam_dnu * ui;

            // 2. p_th = lam * p_th_prev + i * (lam * h_prev)
            let ptr = self.p_th_re[j];
            let pti = self.p_th_im[j];
            self.p_th_re[j] = (lam_r * ptr - lam_i * pti) - lhi;
            self.p_th_im[j] = (lam_r * pti + lam_i * ptr) + lhr;

            // 3. p_b[j, k] = lam * p_b_prev[j, k] + gam * x_k (вещественная инъекция -> dh/db_im = i * p_b)
            for k in 0..x_len {
                let pbr = self.p_b_re[off + k];
                let pbi = self.p_b_im[off + k];
                self.p_b_re[off + k] = (lam_r * pbr - lam_i * pbi) + gam * x[k];
                self.p_b_im[off + k] = lam_r * pbi + lam_i * pbr;
            }
        }
    }

    fn fill_features(&mut self) {
        let sc = 1.0 / (self.units.max(1) as f32).sqrt();
        for j in 0..self.units {
            self.feat_buf[2 * j] = self.y_re[j] * sc;
            self.feat_buf[2 * j + 1] = self.y_im[j] * sc;
        }
        for k in 0..self.in_dim {
            self.feat_buf[2 * self.units + k] = self.x_buf[k] * 0.5;
        }
    }

    fn critic_val_from_buf(&self) -> f32 {
        self.critic_b + fast_dot_raw(&self.critic_w, &self.feat_buf)
    }

    fn apply_td_update(&mut self, td_err: f32) {
        if self.had_nan {
            return;
        }
        let raw_d = td_err.clamp(-6.0, 6.0);
        let crit_err = raw_d.clamp(-3.0, 3.0);
        let centered = raw_d - self.td_mean;
        self.td_mean = 0.99 * self.td_mean + 0.01 * raw_d;
        self.td_var = (0.98 * self.td_var + 0.02 * centered * centered).max(0.08);
        let adv = (centered / self.td_var.sqrt()).clamp(-2.5, 2.5);

        let f_dim = 2 * self.units + self.in_dim;
        for i in 0..f_dim {
            self.critic_w[i] += self.critic_lr * crit_err * self.e_critic_w[i];
        }
        let cn: f32 = self.critic_w.iter().map(|v| v * v).sum();
        if cn > 256.0 {
            let s = 16.0 / cn.sqrt();
            for v in &mut self.critic_w {
                *v *= s;
            }
        }
        self.critic_b = (self.critic_b + 0.25 * self.critic_lr * crit_err * self.e_critic_b).clamp(-25.0, 25.0);

        for a in 0..self.num_actions {
            let off = a * f_dim;
            let mut nsq = 0.0_f32;
            for i in 0..f_dim {
                let nv = self.actor_w[off + i] * 0.9999 + self.actor_lr * adv * self.e_actor_w[off + i];
                self.actor_w[off + i] = nv;
                nsq += nv * nv;
            }
            if nsq > 36.0 {
                let s = 6.0 / nsq.sqrt();
                for i in 0..f_dim {
                    self.actor_w[off + i] *= s;
                }
            }
            self.actor_b[a] = (self.actor_b[a] * 0.9999 + 0.25 * self.actor_lr * adv * self.e_actor_b[a]).clamp(-3.0, 3.0);
        }

        // RMSProp-обновление внутренних параметров LRU (nu, theta, B_re, B_im)
        let rlr = self.rtrl_lr;
        let d = self.in_dim;
        for j in 0..self.units {
            let gnu = adv * self.e_nu[j];
            self.v_nu[j] = 0.98 * self.v_nu[j] + 0.02 * gnu * gnu;
            let snu = (gnu / (self.v_nu[j] + 0.0025).sqrt()).clamp(-3.0, 3.0);
            self.nu[j] = (self.nu[j] + rlr * snu).clamp(-5.2, 1.5);

            let gth = adv * self.e_th[j];
            self.v_theta[j] = 0.98 * self.v_theta[j] + 0.02 * gth * gth;
            let sth = (gth / (self.v_theta[j] + 0.0025).sqrt()).clamp(-3.0, 3.0);
            self.theta[j] = (self.theta[j] + rlr * sth).clamp(-6.28, 6.28);

            let off = j * d;
            for k in 0..d {
                let gbr = adv * self.e_bre[off + k];
                self.v_bre[off + k] = 0.98 * self.v_bre[off + k] + 0.02 * gbr * gbr;
                let sbr = (gbr / (self.v_bre[off + k] + 0.0025).sqrt()).clamp(-3.0, 3.0);
                self.b_re[off + k] = (self.b_re[off + k] + rlr * sbr).clamp(-3.0, 3.0);

                let gbi = adv * self.e_bim[off + k];
                self.v_bim[off + k] = 0.98 * self.v_bim[off + k] + 0.02 * gbi * gbi;
                let sbi = (gbi / (self.v_bim[off + k] + 0.0025).sqrt()).clamp(-3.0, 3.0);
                self.b_im[off + k] = (self.b_im[off + k] + rlr * sbi).clamp(-3.0, 3.0);
            }
        }
    }

    pub fn decide_vec(&mut self, features: &[f32], train_mode: bool, seu_nan: bool, seu_spikes: bool) -> usize {
        if self.enable_state_weight_guard {
            let _ = self.scrub_state_and_weights();
        }
        self.x_buf.fill(0.0);
        for i in 0..self.in_dim.min(features.len()) {
            self.x_buf[i] = features[i];
        }
        if seu_nan && !self.x_buf.is_empty() {
            self.x_buf[0] = f32::NAN;
        }
        if seu_spikes && self.x_buf.len() > 1 {
            self.x_buf[1] = 1e28;
        }
        if self.enable_input_clamp {
            for v in &mut self.x_buf {
                *v = if v.is_finite() { v.clamp(-10.0, 10.0) } else { 0.0 };
            }
        }

        // Temporarily swap x_buf out to satisfy borrow checker with 0 heap allocations
        let x_local = std::mem::take(&mut self.x_buf);
        self.step_lru_forward(&x_local);
        self.x_buf = x_local;
        self.fill_features();
        let v_cur = self.critic_val_from_buf();
        if !v_cur.is_finite() {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
        }

        if train_mode && self.has_pending && !self.had_nan {
            let td_err = self.pending_r + self.gamma * v_cur - self.pending_v;
            self.apply_td_update(td_err);
        }

        let f_dim = self.feat_buf.len();
        let mut max_l = f32::NEG_INFINITY;
        for a in 0..self.num_actions {
            let off = a * f_dim;
            let s = self.actor_b[a] + fast_dot_raw(&self.actor_w[off..off + f_dim], &self.feat_buf);
            if !s.is_finite() {
                if !self.had_nan {
                    self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                }
                self.had_nan = true;
                self.ever_had_nan = true;
            }
            self.logits_buf[a] = s;
            if s > max_l {
                max_l = s;
            }
        }
        let mut sum_e = 0.0_f32;
        for a in 0..self.num_actions {
            let e = (self.logits_buf[a] - max_l).exp();
            self.probs_buf[a] = e;
            sum_e += e;
        }
        if !sum_e.is_finite() || sum_e <= 0.0 {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
            self.probs_buf.fill(1.0 / (self.num_actions as f32));
        } else {
            let inv = 1.0 / sum_e;
            for p in &mut self.probs_buf {
                *p *= inv;
            }
        }

        let act = if train_mode && !self.had_nan {
            let u = rand_unit_f32(&mut self.rng);
            let mut cdf = 0.0_f32;
            let mut picked = self.num_actions - 1;
            for (a, &p) in self.probs_buf.iter().enumerate() {
                cdf += p;
                if u <= cdf {
                    picked = a;
                    break;
                }
            }
            picked
        } else {
            let mut ba = 0usize;
            let mut bp = -1.0_f32;
            for (a, &p) in self.probs_buf.iter().enumerate() {
                if p > bp {
                    bp = p;
                    ba = a;
                }
            }
            ba
        };

        if train_mode && !self.had_nan {
            let decay = self.gamma * self.trace_lambda;
            let sc = 1.0 / (self.units.max(1) as f32).sqrt();

            for i in 0..f_dim {
                self.e_critic_w[i] = decay * self.e_critic_w[i] + self.feat_buf[i];
            }
            let ecn: f32 = self.e_critic_w.iter().map(|v| v * v).sum();
            if ecn > 16.0 {
                let s = 4.0 / ecn.sqrt();
                for v in &mut self.e_critic_w {
                    *v *= s;
                }
            }
            self.e_critic_b = (decay * self.e_critic_b + 1.0).clamp(-5.0, 5.0);

            self.fb_re_buf.fill(0.0);
            self.fb_im_buf.fill(0.0);
            for a in 0..self.num_actions {
                let score = (if a == act { 1.0 } else { 0.0 }) - self.probs_buf[a];
                let off = a * f_dim;
                let mut ean = 0.0_f32;
                for i in 0..f_dim {
                    let nv = decay * self.e_actor_w[off + i] + score * self.feat_buf[i];
                    self.e_actor_w[off + i] = nv;
                    ean += nv * nv;
                }
                if ean > 16.0 {
                    let s = 4.0 / ean.sqrt();
                    for i in 0..f_dim {
                        self.e_actor_w[off + i] *= s;
                    }
                }
                self.e_actor_b[a] = (decay * self.e_actor_b[a] + score).clamp(-5.0, 5.0);

                for j in 0..self.units {
                    self.fb_re_buf[j] += score * self.actor_w[off + 2 * j] * sc;
                    self.fb_im_buf[j] += score * self.actor_w[off + 2 * j + 1] * sc;
                }
            }
            for j in 0..self.units {
                self.fb_re_buf[j] += 0.15 * self.critic_w[2 * j] * sc;
                self.fb_im_buf[j] += 0.15 * self.critic_w[2 * j + 1] * sc;
            }

            // Точный якобиан Norm_kappa: f_h = (1/rho) * (fb - (fb . y) * y)
            let d = self.in_dim;
            for j in 0..self.units {
                let yr = self.y_re[j];
                let yi = self.y_im[j];
                let gr = self.fb_re_buf[j];
                let gi = self.fb_im_buf[j];
                let dot = gr * yr + gi * yi;
                let inv_rho = 1.0 / self.rho[j].max(self.kappa);
                let fr = (gr - dot * yr) * inv_rho;
                let fi = (gi - dot * yi) * inv_rho;

                let dnu = fr * self.p_nu_re[j] + fi * self.p_nu_im[j];
                self.e_nu[j] = (decay * self.e_nu[j] + dnu).clamp(-16.0, 16.0);
                let dth = fr * self.p_th_re[j] + fi * self.p_th_im[j];
                self.e_th[j] = (decay * self.e_th[j] + dth).clamp(-16.0, 16.0);

                let off = j * d;
                for k in 0..d {
                    let pbr = self.p_b_re[off + k];
                    let pbi = self.p_b_im[off + k];
                    // dh/db_re = (pbr, pbi), dh/db_im = (-pbi, pbr)
                    let dbr = fr * pbr + fi * pbi;
                    let dbi = -fr * pbi + fi * pbr;
                    self.e_bre[off + k] = (decay * self.e_bre[off + k] + dbr).clamp(-16.0, 16.0);
                    self.e_bim[off + k] = (decay * self.e_bim[off + k] + dbi).clamp(-16.0, 16.0);
                }
            }

            // Прямая регуляризация энтропии Шеннона (стандарт AAAI 2025)
            if self.entropy_beta > 0.0 {
                let mut sum_plogp = 0.0_f32;
                for &p in &self.probs_buf {
                    let pc = p.max(1e-6);
                    sum_plogp += pc * pc.ln();
                }
                for a in 0..self.num_actions {
                    let pc = self.probs_buf[a].max(1e-6);
                    let d_ent = self.entropy_beta * pc * (sum_plogp - pc.ln());
                    let step = self.actor_lr * d_ent;
                    let off = a * f_dim;
                    for i in 0..f_dim {
                        self.actor_w[off + i] += step * self.feat_buf[i];
                    }
                    self.actor_b[a] = (self.actor_b[a] + 0.25 * step).clamp(-3.0, 3.0);
                }
            }

            self.pending_v = self.critic_val_from_buf();
            self.pending_r = 0.0;
            self.has_pending = true;
        }
        act
    }

    pub fn observe_step_reward(&mut self, r: f32) {
        self.pending_r = r;
    }

    pub fn end_episode(&mut self, train_mode: bool) {
        self.end_episode_truncated(train_mode, false);
    }

    pub fn end_episode_truncated(&mut self, train_mode: bool, truncated: bool) {
        if train_mode && self.has_pending && !self.had_nan {
            let v_next = if truncated { self.pending_v } else { 0.0 };
            let td_err = self.pending_r + self.gamma * v_next - self.pending_v;
            self.apply_td_update(td_err);
            self.has_pending = false;
        }
    }

    pub(crate) fn step_lru_forward_readonly(
        &self,
        x: &[f32],
        h_re: &mut [f32],
        h_im: &mut [f32],
        y_re: &mut [f32],
        y_im: &mut [f32],
    ) {
        let m = self.units;
        let d = self.in_dim;
        let x_len = x.len().min(d);
        let kap2 = self.kappa * self.kappa;

        for j in 0..m {
            let nu_c = self.nu[j].clamp(-5.2, 1.5);
            let exp_nu = nu_c.exp();
            let r = (-exp_nu).exp();
            let th = self.theta[j];
            let (cth, sth) = (th.cos(), th.sin());
            let lam_r = r * cth;
            let lam_i = r * sth;
            let gam = (1.0 - r * r).max(1e-4).sqrt();

            let hr_prev = h_re[j];
            let hi_prev = h_im[j];
            let lhr = lam_r * hr_prev - lam_i * hi_prev;
            let lhi = lam_r * hi_prev + lam_i * hr_prev;

            let mut ur = 0.0_f32;
            let mut ui = 0.0_f32;
            let off = j * d;
            for k in 0..x_len {
                let xk = x[k];
                ur += self.b_re[off + k] * xk;
                ui += self.b_im[off + k] * xk;
            }
            let hr = lhr + gam * ur;
            let hi = lhi + gam * ui;
            h_re[j] = hr;
            h_im[j] = hi;
            let rho = (hr * hr + hi * hi + kap2).sqrt();
            y_re[j] = hr / rho;
            y_im[j] = hi / rho;
        }
    }

    pub fn predict_k_ahead_readonly(&self, window: &[&[f32]]) -> usize {
        let mut h_re = self.h_re.clone();
        let mut h_im = self.h_im.clone();
        let mut y_re = self.y_re.clone();
        let mut y_im = self.y_im.clone();
        let mut x_buf = self.x_buf.clone();

        for &features in window {
            x_buf.fill(0.0);
            for i in 0..self.in_dim.min(features.len()) {
                x_buf[i] = features[i];
            }
            if self.enable_input_clamp {
                for v in &mut x_buf {
                    *v = if v.is_finite() { v.clamp(-10.0, 10.0) } else { 0.0 };
                }
            }
            self.step_lru_forward_readonly(&x_buf, &mut h_re, &mut h_im, &mut y_re, &mut y_im);
        }

        let mut feat_buf = vec![0.0_f32; self.feat_buf.len()];
        let sc = 1.0 / (self.units.max(1) as f32).sqrt();
        for j in 0..self.units {
            feat_buf[2 * j] = y_re[j] * sc;
            feat_buf[2 * j + 1] = y_im[j] * sc;
        }
        for k in 0..self.in_dim {
            feat_buf[2 * self.units + k] = x_buf[k] * 0.5;
        }
        greedy_head_action(&self.actor_b, &self.actor_w, &feat_buf, self.num_actions)
    }
}

/// Контрольный бейзлайн: `RtrrlLruBaseline` + те же банки `PageHinkleyDetector` + входной clamp,
/// чтобы изолировать вклад роторов Кэли от вклада переключения режимов и санации входа (`v0.4.0`).
pub struct RtrrlLruPhBaseline {
    pub banks: Vec<RtrrlLruBaseline>,
    pub phs: Vec<PageHinkleyDetector>,
    pub age_eps: Vec<usize>,
    pub cooldowns: Vec<usize>,
    pub active_regime: usize,
    pub episode_counter: usize,
    pub probe_eps_left: usize,
    pub probe_return_sum: f32,
    pub probe_return_cnt: usize,
    pub total_switches: usize,
    pub probe_accept_threshold: f32,
    retired_had_nan: bool,
    retired_seu_faults: u64,
}

impl RtrrlLruPhBaseline {
    pub fn new(
        in_dim: usize,
        units: usize,
        num_actions: usize,
        max_regimes: usize,
        ph_delta: f32,
        ph_threshold: f32,
        ph_warmup: usize,
        seed: u64,
    ) -> Self {
        let n = max_regimes.clamp(2, 12);
        let mut banks = Vec::with_capacity(n);
        let mut phs = Vec::with_capacity(n);
        for b in 0..n {
            let mut lru = RtrrlLruBaseline::new(
                in_dim,
                units,
                num_actions,
                seed ^ (0x5048_0000_u64 + b as u64),
            );
            lru.enable_input_clamp = true;
            banks.push(lru);
            phs.push(PageHinkleyDetector::new(ph_delta, ph_threshold, ph_warmup));
        }
        Self {
            banks,
            phs,
            age_eps: vec![0; n],
            cooldowns: vec![0; n],
            active_regime: 0,
            episode_counter: 0,
            probe_eps_left: 0,
            probe_return_sum: 0.0,
            probe_return_cnt: 0,
            total_switches: 0,
            probe_accept_threshold: 0.38,
            retired_had_nan: false,
            retired_seu_faults: 0,
        }
    }

    pub fn set_probe_accept_threshold(&mut self, th: f32) {
        self.probe_accept_threshold = th.clamp(0.0, 1.0);
    }

    pub fn set_state_weight_guard(&mut self, on: bool) {
        for b in &mut self.banks {
            b.enable_state_weight_guard = on;
        }
    }

    pub fn total_seu_faults(&self) -> u64 {
        self.retired_seu_faults
            .saturating_add(self.banks.iter().map(|b| b.seu_fault_count).sum())
    }

    pub fn set_hyperparams(
        &mut self,
        gamma: f32,
        trace_lambda: f32,
        actor_lr: f32,
        critic_lr: f32,
        rtrl_lr: f32,
        entropy_beta: f32,
    ) {
        for b in &mut self.banks {
            b.gamma = gamma;
            b.trace_lambda = trace_lambda;
            b.actor_lr = actor_lr;
            b.critic_lr = critic_lr;
            b.rtrl_lr = rtrl_lr;
            b.entropy_beta = entropy_beta;
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.banks
            .iter()
            .map(|b| b.memory_footprint_bytes())
            .sum::<usize>()
            + self.phs.len() * 48
    }

    pub fn had_nan(&self) -> bool {
        self.retired_had_nan || self.banks.iter().any(|b| b.ever_had_nan)
    }

    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        self.banks[self.active_regime].inject_register_bit_flip(reg_idx, bit);
    }

    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        self.banks[self.active_regime].inject_weight_bit_flip(reg_idx, bit);
    }

    pub fn start_episode(&mut self) {
        self.banks[self.active_regime].start_episode();
    }

    pub fn decide_vec(
        &mut self,
        features: &[f32],
        train_mode: bool,
        seu_nan: bool,
        seu_spikes: bool,
    ) -> usize {
        let r = self.active_regime;
        let in_probe = self.probe_eps_left > 0;
        let suspect = self.phs[r].cusum > self.phs[r].threshold * 0.42 && self.phs[r].mu_bank >= 0.62;
        let eff_train = train_mode && !in_probe && !suspect;
        self.banks[r].decide_vec(features, eff_train, seu_nan, seu_spikes)
    }

    pub fn observe_step_reward(&mut self, r: f32) {
        self.banks[self.active_regime].observe_step_reward(r);
    }

    pub fn end_episode_with_score(&mut self, norm_ret: f32, truncated: bool, train_mode: bool) {
        let r = self.active_regime;
        self.episode_counter = self.episode_counter.saturating_add(1);

        if self.probe_eps_left > 0 {
            self.probe_return_sum += norm_ret;
            self.probe_return_cnt += 1;
            self.probe_eps_left -= 1;
            if self.probe_eps_left == 0 {
                let avg = self.probe_return_sum / (self.probe_return_cnt.max(1) as f32);
                if avg >= self.probe_accept_threshold {
                    self.phs[r].reset_cusum();
                } else {
                    self.cooldowns[r] = self.episode_counter.saturating_add(48);
                    self.switch_bank();
                }
            }
            return;
        }

        self.age_eps[r] = self.age_eps[r].saturating_add(1);
        let suspect = self.phs[r].cusum > self.phs[r].threshold * 0.42 && self.phs[r].mu_bank >= 0.62;
        if train_mode && !suspect {
            self.banks[r].end_episode_truncated(true, truncated);
        }

        if train_mode && self.phs[r].observe(norm_ret.clamp(0.0, 1.0)) {
            self.phs[r].reset_cusum();
            self.cooldowns[r] = self.episode_counter.saturating_add(48);
            self.switch_bank();
        }
    }

    fn switch_bank(&mut self) {
        let prev = self.active_regime;
        let n = self.banks.len();
        for idx in 0..n {
            if idx != prev && self.age_eps[idx] > 0 && self.episode_counter >= self.cooldowns[idx] {
                self.active_regime = idx;
                self.probe_eps_left = 6;
                self.probe_return_sum = 0.0;
                self.probe_return_cnt = 0;
                self.total_switches = self.total_switches.saturating_add(1);
                return;
            }
        }
        for idx in 0..n {
            if idx != prev && self.age_eps[idx] == 0 && self.episode_counter >= self.cooldowns[idx] {
                self.active_regime = idx;
                self.probe_eps_left = 0;
                self.total_switches = self.total_switches.saturating_add(1);
                return;
            }
        }
        let next_r = (prev + 1) % n;
        let units = self.banks[next_r].units;
        let in_dim = self.banks[next_r].in_dim;
        let na = self.banks[next_r].num_actions;
        let seed = 0x5048_1000_u64 + (self.episode_counter as u64);
        let mut fresh = RtrrlLruBaseline::new(in_dim, units, na, seed);
        fresh.enable_input_clamp = self.banks[prev].enable_input_clamp;
        fresh.enable_state_weight_guard = self.banks[prev].enable_state_weight_guard;
        fresh.gamma = self.banks[prev].gamma;
        fresh.trace_lambda = self.banks[prev].trace_lambda;
        fresh.actor_lr = self.banks[prev].actor_lr;
        fresh.critic_lr = self.banks[prev].critic_lr;
        fresh.rtrl_lr = self.banks[prev].rtrl_lr;
        fresh.entropy_beta = self.banks[prev].entropy_beta;
        self.retired_had_nan |= self.banks[next_r].ever_had_nan;
        self.retired_seu_faults = self
            .retired_seu_faults
            .saturating_add(self.banks[next_r].seu_fault_count);
        self.banks[next_r] = fresh;
        self.phs[next_r].reset_cusum();
        self.age_eps[next_r] = 0;
        self.cooldowns[next_r] = 0;
        self.active_regime = next_r;
        self.probe_eps_left = 0;
        self.total_switches = self.total_switches.saturating_add(1);
    }
}

// =============================================================================
// v0.4.0 Stronger Baselines:
//   1. PhRegimeCtrl (shared multi-bank PageHinkley controller)
//   2. OnlineSelectiveSsmPhBaseline (Mamba-style Selective SSM + RTRL + PH + Clamp)
//   3. DeepRtrrlLruPhBaseline (2-Layer Hierarchical Fast-Slow RTRRL-LRU + PH + Clamp)
//   4. OnlineGruPerStepPhBaseline (Per-Step Online TD(lambda) GRU + PH + Clamp)
// =============================================================================

#[derive(Clone)]
struct PhRegimeCtrl {
    phs: Vec<PageHinkleyDetector>,
    age_eps: Vec<usize>,
    cooldowns: Vec<usize>,
    pub active_regime: usize,
    episode_counter: usize,
    pub probe_eps_left: usize,
    probe_return_sum: f32,
    probe_return_cnt: usize,
    pub total_switches: usize,
    pub probe_accept_threshold: f32,
    win_hits: usize,
    win_n: usize,
}

impl PhRegimeCtrl {
    fn new(max_regimes: usize, ph_delta: f32, ph_threshold: f32, ph_warmup: usize) -> Self {
        let n = max_regimes.clamp(2, 12);
        Self {
            phs: (0..n)
                .map(|_| PageHinkleyDetector::new(ph_delta, ph_threshold, ph_warmup))
                .collect(),
            age_eps: vec![0; n],
            cooldowns: vec![0; n],
            active_regime: 0,
            episode_counter: 0,
            probe_eps_left: 0,
            probe_return_sum: 0.0,
            probe_return_cnt: 0,
            total_switches: 0,
            probe_accept_threshold: 0.38,
            win_hits: 0,
            win_n: 0,
        }
    }

    fn observe_stream_step(&mut self, hit: bool, train_mode: bool, win_len: usize) -> Option<usize> {
        self.win_hits += usize::from(hit);
        self.win_n += 1;
        if self.win_n >= win_len.max(1) {
            let acc = (self.win_hits as f32) / (self.win_n as f32);
            self.win_hits = 0;
            self.win_n = 0;
            let (_, rst) = self.end_episode_observe(acc, train_mode);
            return rst;
        }
        None
    }

    fn is_frozen_for_train(&self) -> bool {
        let r = self.active_regime;
        let suspect = self.phs[r].cusum > self.phs[r].threshold * 0.42 && self.phs[r].mu_bank >= 0.62;
        self.probe_eps_left > 0 || suspect
    }

    /// Возвращает `(should_td_update, reset_bank_idx)` — `Some(idx)` если нужен свежий банк `idx`.
    fn end_episode_observe(&mut self, norm_ret: f32, train_mode: bool) -> (bool, Option<usize>) {
        let r = self.active_regime;
        self.episode_counter = self.episode_counter.saturating_add(1);

        if self.probe_eps_left > 0 {
            self.probe_return_sum += norm_ret;
            self.probe_return_cnt += 1;
            self.probe_eps_left -= 1;
            if self.probe_eps_left == 0 {
                let avg = self.probe_return_sum / (self.probe_return_cnt.max(1) as f32);
                if avg >= self.probe_accept_threshold {
                    self.phs[r].reset_cusum();
                } else {
                    self.cooldowns[r] = self.episode_counter.saturating_add(48);
                    return (false, self.switch_bank());
                }
            }
            return (false, None);
        }

        self.age_eps[r] = self.age_eps[r].saturating_add(1);
        let suspect = self.phs[r].cusum > self.phs[r].threshold * 0.42 && self.phs[r].mu_bank >= 0.62;
        let do_update = train_mode && !suspect;
        let mut reset_idx = None;
        if train_mode && self.phs[r].observe(norm_ret.clamp(0.0, 1.0)) {
            self.phs[r].reset_cusum();
            self.cooldowns[r] = self.episode_counter.saturating_add(48);
            reset_idx = self.switch_bank();
        }
        (do_update, reset_idx)
    }

    fn switch_bank(&mut self) -> Option<usize> {
        let prev = self.active_regime;
        let n = self.phs.len();
        for idx in 0..n {
            if idx != prev && self.age_eps[idx] > 0 && self.episode_counter >= self.cooldowns[idx] {
                self.active_regime = idx;
                self.probe_eps_left = 6;
                self.probe_return_sum = 0.0;
                self.probe_return_cnt = 0;
                self.total_switches = self.total_switches.saturating_add(1);
                return None;
            }
        }
        for idx in 0..n {
            if idx != prev && self.age_eps[idx] == 0 && self.episode_counter >= self.cooldowns[idx] {
                self.active_regime = idx;
                self.probe_eps_left = 0;
                self.total_switches = self.total_switches.saturating_add(1);
                return None;
            }
        }
        let next_r = (prev + 1) % n;
        self.phs[next_r].reset_cusum();
        self.age_eps[next_r] = 0;
        self.cooldowns[next_r] = 0;
        self.active_regime = next_r;
        self.probe_eps_left = 0;
        self.total_switches = self.total_switches.saturating_add(1);
        Some(next_r)
    }
}

/// 1. Mamba-style Input-Selective Diagonal SSM с онлайн RTRL чувствительностями для `nu, omega, W_delta, W_B`.
#[derive(Clone)]
pub struct SelectiveSsmBank {
    pub in_dim: usize,
    pub units: usize,
    pub num_actions: usize,
    pub kappa: f32,
    pub nu: Vec<f32>,
    pub omega: Vec<f32>,
    pub w_delta: Vec<f32>, // [units * in_dim]
    pub b_delta: Vec<f32>, // [units]
    pub w_bre: Vec<f32>,   // [units * in_dim]
    pub w_bim: Vec<f32>,   // [units * in_dim]
    v_nu: Vec<f32>,
    v_om: Vec<f32>,
    v_wd: Vec<f32>,
    v_bd: Vec<f32>,
    v_wbr: Vec<f32>,
    v_wbi: Vec<f32>,
    pub h_re: Vec<f32>,
    pub h_im: Vec<f32>,
    rho: Vec<f32>,
    y_re: Vec<f32>,
    y_im: Vec<f32>,
    p_nu_re: Vec<f32>,
    p_nu_im: Vec<f32>,
    p_om_re: Vec<f32>,
    p_om_im: Vec<f32>,
    p_bd_re: Vec<f32>,
    p_bd_im: Vec<f32>,
    p_wd_re: Vec<f32>,
    p_wd_im: Vec<f32>,
    p_br_re: Vec<f32>,
    p_br_im: Vec<f32>,
    p_bi_re: Vec<f32>,
    p_bi_im: Vec<f32>,
    actor_w: Vec<f32>,
    actor_b: Vec<f32>,
    v_aw: Vec<f32>,
    critic_w: Vec<f32>,
    critic_b: f32,
    e_actor_w: Vec<f32>,
    e_actor_b: Vec<f32>,
    e_critic_w: Vec<f32>,
    e_critic_b: f32,
    e_nu: Vec<f32>,
    e_om: Vec<f32>,
    e_bd: Vec<f32>,
    e_wd: Vec<f32>,
    e_wbr: Vec<f32>,
    e_wbi: Vec<f32>,
    // Предвыделенные буферы шага (0 аллокаций в куче на шаге!)
    x_buf: Vec<f32>,
    feat_buf: Vec<f32>,
    logits_buf: Vec<f32>,
    probs_buf: Vec<f32>,
    fb_re_buf: Vec<f32>,
    fb_im_buf: Vec<f32>,
    pub gamma: f32,
    pub trace_lambda: f32,
    pub actor_lr: f32,
    pub critic_lr: f32,
    pub rtrl_lr: f32,
    pub entropy_beta: f32,
    pub enable_input_clamp: bool,
    pub enable_state_weight_guard: bool,
    pub had_nan: bool,
    pub ever_had_nan: bool,
    pub seu_fault_count: u64,
    pending_v: f32,
    pending_r: f32,
    has_pending: bool,
    td_mean: f32,
    td_var: f32,
    rng: u64,
}

impl SelectiveSsmBank {
    pub fn new(in_dim: usize, units: usize, num_actions: usize, seed: u64) -> Self {
        let mut rng = splitmix64(seed ^ 0x55AA_3141_5926);
        let u = units.max(2);
        let d = in_dim.max(1);
        let na = num_actions.max(1);
        let f_dim = 2 * u + d;
        let ud = u * d;

        let mut nu = Vec::with_capacity(u);
        let mut omega = Vec::with_capacity(u);
        let mut b_delta = Vec::with_capacity(u);
        for j in 0..u {
            let frac = j as f32 / u as f32;
            nu.push((-4.2 + 2.6 * frac).clamp(-5.2, 1.5));
            let sign = if (j & 1) == 0 { 1.0 } else { -1.0 };
            omega.push(sign * (0.25 + 2.8 * frac));
            b_delta.push(-0.6 + 0.8 * frac);
        }
        let init_mat = |n: usize, sc: f32, r: &mut u64| -> Vec<f32> {
            (0..n).map(|_| rand_signed_f32(r) * sc).collect()
        };
        let bs = 0.55 / (d as f32).sqrt();
        let actor_w = init_mat(na * f_dim, 0.05, &mut rng);
        let critic_w = init_mat(f_dim, 0.05, &mut rng);
        Self {
            in_dim: d,
            units: u,
            num_actions: na,
            kappa: 0.65,
            nu,
            omega,
            w_delta: init_mat(ud, bs, &mut rng),
            b_delta,
            w_bre: init_mat(ud, bs, &mut rng),
            w_bim: init_mat(ud, bs, &mut rng),
            v_nu: vec![0.04; u],
            v_om: vec![0.04; u],
            v_wd: vec![0.04; ud],
            v_bd: vec![0.04; u],
            v_wbr: vec![0.04; ud],
            v_wbi: vec![0.04; ud],
            h_re: vec![0.0; u],
            h_im: vec![0.0; u],
            rho: vec![0.65; u],
            y_re: vec![0.0; u],
            y_im: vec![0.0; u],
            p_nu_re: vec![0.0; u],
            p_nu_im: vec![0.0; u],
            p_om_re: vec![0.0; u],
            p_om_im: vec![0.0; u],
            p_bd_re: vec![0.0; u],
            p_bd_im: vec![0.0; u],
            p_wd_re: vec![0.0; ud],
            p_wd_im: vec![0.0; ud],
            p_br_re: vec![0.0; ud],
            p_br_im: vec![0.0; ud],
            p_bi_re: vec![0.0; ud],
            p_bi_im: vec![0.0; ud],
            actor_w,
            actor_b: vec![0.0; na],
            v_aw: vec![0.04; na * f_dim],
            critic_w,
            critic_b: 0.0,
            e_actor_w: vec![0.0; na * f_dim],
            e_actor_b: vec![0.0; na],
            e_critic_w: vec![0.0; f_dim],
            e_critic_b: 0.0,
            e_nu: vec![0.0; u],
            e_om: vec![0.0; u],
            e_bd: vec![0.0; u],
            e_wd: vec![0.0; ud],
            e_wbr: vec![0.0; ud],
            e_wbi: vec![0.0; ud],
            x_buf: vec![0.0; d],
            feat_buf: vec![0.0; f_dim],
            logits_buf: vec![0.0; na],
            probs_buf: vec![0.0; na],
            fb_re_buf: vec![0.0; u],
            fb_im_buf: vec![0.0; u],
            gamma: 0.95,
            trace_lambda: 0.80,
            actor_lr: 0.06,
            critic_lr: 0.14,
            rtrl_lr: 0.008,
            entropy_beta: 0.006,
            enable_input_clamp: true,
            enable_state_weight_guard: false,
            had_nan: false,
            ever_had_nan: false,
            seu_fault_count: 0,
            pending_v: 0.0,
            pending_r: 0.0,
            has_pending: false,
            td_mean: 0.0,
            td_var: 1.0,
            rng,
        }
    }

    pub fn scrub_state_and_weights(&mut self) -> u64 {
        let mut f = scrub_slice_seu(&mut self.h_re, 24.0, 0.0);
        f = f.saturating_add(scrub_slice_seu(&mut self.h_im, 24.0, 0.0));
        for v in self.nu.iter_mut() {
            if !v.is_finite() || *v < -6.1 || *v > 6.1 {
                *v = -2.0;
                f = f.saturating_add(1);
            }
        }
        f = f.saturating_add(scrub_slice_seu(&mut self.omega, 6.5, 0.0));
        f = f.saturating_add(scrub_slice_seu(&mut self.b_delta, 6.1, 0.0));
        f = f.saturating_add(scrub_slice_seu(&mut self.w_delta, 16.1, 0.0));
        f = f.saturating_add(scrub_slice_seu(&mut self.w_bre, 16.1, 0.0));
        f = f.saturating_add(scrub_slice_seu(&mut self.w_bim, 16.1, 0.0));
        f = f.saturating_add(scrub_slice_seu(&mut self.actor_w, 16.1, 0.0));
        f = f.saturating_add(scrub_slice_seu(&mut self.actor_b, 16.1, 0.0));
        f = f.saturating_add(scrub_slice_seu(&mut self.critic_w, 16.1, 0.0));
        if !self.critic_b.is_finite() || self.critic_b.abs() > 26.0 {
            self.critic_b = 0.0;
            f = f.saturating_add(1);
        }
        self.seu_fault_count = self.seu_fault_count.saturating_add(f);
        f
    }

    pub fn new_supervised(in_dim: usize, units: usize, num_actions: usize, seed: u64) -> Self {
        let mut s = Self::new(in_dim, units, num_actions.max(1), seed);
        s.critic_w = Vec::new();
        s.e_actor_w = Vec::new();
        s.e_actor_b = Vec::new();
        s.e_critic_w = Vec::new();
        s
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        (self.nu.len() * 22
            + self.w_delta.len() * 15
            + self.actor_w.len()
            + self.v_aw.len()
            + self.e_actor_w.len()
            + self.actor_b.len()
            + self.e_actor_b.len()
            + self.logits_buf.len()
            + self.probs_buf.len()
            + self.critic_w.len()
            + self.e_critic_w.len()
            + self.feat_buf.len()
            + self.x_buf.len())
            * 4
            + 128
    }

    pub fn start_episode(&mut self) {
        self.h_re.fill(0.0);
        self.h_im.fill(0.0);
        self.y_re.fill(0.0);
        self.y_im.fill(0.0);
        self.rho.fill(self.kappa);
        self.p_nu_re.fill(0.0);
        self.p_nu_im.fill(0.0);
        self.p_om_re.fill(0.0);
        self.p_om_im.fill(0.0);
        self.p_bd_re.fill(0.0);
        self.p_bd_im.fill(0.0);
        self.p_wd_re.fill(0.0);
        self.p_wd_im.fill(0.0);
        self.p_br_re.fill(0.0);
        self.p_br_im.fill(0.0);
        self.p_bi_re.fill(0.0);
        self.p_bi_im.fill(0.0);
        self.e_actor_w.fill(0.0);
        self.e_actor_b.fill(0.0);
        self.e_critic_w.fill(0.0);
        self.e_critic_b = 0.0;
        self.e_nu.fill(0.0);
        self.e_om.fill(0.0);
        self.e_bd.fill(0.0);
        self.e_wd.fill(0.0);
        self.e_wbr.fill(0.0);
        self.e_wbi.fill(0.0);
        self.has_pending = false;
        self.pending_r = 0.0;
        self.pending_v = 0.0;
        self.had_nan = false;
    }

    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let n = 2 * self.units;
        if n == 0 { return; }
        let idx = reg_idx % n;
        let u = idx / 2;
        let mask = 1u32 << (bit & 31);
        if (idx & 1) == 0 {
            self.h_re[u] = f32::from_bits(self.h_re[u].to_bits() ^ mask);
        } else {
            self.h_im[u] = f32::from_bits(self.h_im[u].to_bits() ^ mask);
        }
    }

    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let mut slices: [&mut [f32]; 8] = [
            &mut self.nu,
            &mut self.omega,
            &mut self.b_delta,
            &mut self.w_delta,
            &mut self.w_bre,
            &mut self.w_bim,
            &mut self.actor_w,
            &mut self.critic_w,
        ];
        flip_weight_in_slices(&mut slices, reg_idx, bit);
    }

    fn step_forward(&mut self, x: &[f32]) {
        if self.enable_state_weight_guard {
            let _ = self.scrub_state_and_weights();
        }
        let u = self.units;
        let d = self.in_dim;
        let xl = x.len().min(d);
        let kap2 = self.kappa * self.kappa;

        for j in 0..u {
            let off = j * d;
            let mut zd = self.b_delta[j];
            let mut br_in = 0.0_f32;
            let mut bi_in = 0.0_f32;
            for k in 0..xl {
                let xk = x[k];
                if !xk.is_finite() {
                    if !self.had_nan {
                        self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                    }
                    self.had_nan = true;
                    self.ever_had_nan = true;
                }
                zd += self.w_delta[off + k] * xk;
                br_in += self.w_bre[off + k] * xk;
                bi_in += self.w_bim[off + k] * xk;
            }
            let zc = zd.clamp(-10.0, 10.0);
            let exp_z = zc.exp();
            let softp = (1.0 + exp_z).ln();
            let delta = softp.clamp(0.02, 2.0);
            let sig_z = exp_z / (1.0 + exp_z);

            let sig_nu = self.nu[j].clamp(-5.2, 1.5).exp();
            let r = (-delta * sig_nu).exp();
            let th = delta * self.omega[j];
            let (cth, sth) = (th.cos(), th.sin());
            let (lam_r, lam_i) = (r * cth, r * sth);
            let gam = (1.0 - r * r).max(1e-4).sqrt();
            let g_dt = gam * delta;

            let hp_r = self.h_re[j];
            let hp_i = self.h_im[j];
            let lhr = lam_r * hp_r - lam_i * hp_i;
            let lhi = lam_r * hp_i + lam_i * hp_r;

            let hr = lhr + g_dt * br_in;
            let hi = lhi + g_dt * bi_in;
            if !hr.is_finite() || !hi.is_finite() {
                if !self.had_nan {
                    self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                }
                self.had_nan = true;
                self.ever_had_nan = true;
            }
            self.h_re[j] = hr;
            self.h_im[j] = hi;
            let rho = (hr * hr + hi * hi + kap2).sqrt();
            self.rho[j] = rho;
            self.y_re[j] = hr / rho;
            self.y_im[j] = hi / rho;

            // Чувствительности RTRL
            let dgam_dr = -r / gam;
            let dr_dnu = -delta * sig_nu * r;
            let dgdt_dnu = dgam_dr * dr_dnu * delta;
            let (pnr, pni) = (self.p_nu_re[j], self.p_nu_im[j]);
            self.p_nu_re[j] = (lam_r * pnr - lam_i * pni) - delta * sig_nu * lhr + dgdt_dnu * br_in;
            self.p_nu_im[j] = (lam_r * pni + lam_i * pnr) - delta * sig_nu * lhi + dgdt_dnu * bi_in;

            let (por, poi) = (self.p_om_re[j], self.p_om_im[j]);
            self.p_om_re[j] = (lam_r * por - lam_i * poi) - delta * lhi;
            self.p_om_im[j] = (lam_r * poi + lam_i * por) + delta * lhr;

            // Производные по delta -> b_delta и w_delta
            let om = self.omega[j];
            let dr_dd = -sig_nu * r;
            let dgdt_dd = gam + delta * dgam_dr * dr_dd;
            let inh_dr = (-sig_nu * lhr - om * lhi) + dgdt_dd * br_in;
            let inh_di = (-sig_nu * lhi + om * lhr) + dgdt_dd * bi_in;
            let inh_bz_r = inh_dr * sig_z;
            let inh_bz_i = inh_di * sig_z;

            let (pbdr, pbdi) = (self.p_bd_re[j], self.p_bd_im[j]);
            self.p_bd_re[j] = (lam_r * pbdr - lam_i * pbdi) + inh_bz_r;
            self.p_bd_im[j] = (lam_r * pbdi + lam_i * pbdr) + inh_bz_i;

            for k in 0..d {
                let xk = if k < xl { x[k] } else { 0.0 };
                let (pwdr, pwdi) = (self.p_wd_re[off + k], self.p_wd_im[off + k]);
                self.p_wd_re[off + k] = (lam_r * pwdr - lam_i * pwdi) + xk * inh_bz_r;
                self.p_wd_im[off + k] = (lam_r * pwdi + lam_i * pwdr) + xk * inh_bz_i;

                let (pbrr, pbri) = (self.p_br_re[off + k], self.p_br_im[off + k]);
                self.p_br_re[off + k] = (lam_r * pbrr - lam_i * pbri) + g_dt * xk;
                self.p_br_im[off + k] = lam_r * pbri + lam_i * pbrr;

                let (pbir, pbii) = (self.p_bi_re[off + k], self.p_bi_im[off + k]);
                self.p_bi_re[off + k] = (lam_r * pbir - lam_i * pbii) - 0.0 * g_dt * xk;
                self.p_bi_im[off + k] = (lam_r * pbii + lam_i * pbir) + g_dt * xk;
            }
        }
    }

    fn fill_features(&mut self) {
        let sc = 1.0 / (self.units.max(1) as f32).sqrt();
        for j in 0..self.units {
            self.feat_buf[2 * j] = self.y_re[j] * sc;
            self.feat_buf[2 * j + 1] = self.y_im[j] * sc;
        }
        for k in 0..self.in_dim {
            self.feat_buf[2 * self.units + k] = self.x_buf[k] * 0.5;
        }
    }

    fn forward_logits_into(&mut self) {
        let f_dim = self.feat_buf.len();
        for a in 0..self.num_actions {
            let off = a * f_dim;
            let s = self.actor_b[a] + fast_dot_raw(&self.actor_w[off..off + f_dim], &self.feat_buf);
            if !s.is_finite() {
                if !self.had_nan {
                    self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                }
                self.had_nan = true;
                self.ever_had_nan = true;
            }
            self.logits_buf[a] = s;
        }
    }

    fn apply_rms_rtrl(&mut self, adv: f32) {
        let rlr = self.rtrl_lr;
        let d = self.in_dim;
        for j in 0..self.units {
            let gnu = adv * self.e_nu[j];
            self.v_nu[j] = 0.98 * self.v_nu[j] + 0.02 * gnu * gnu;
            let snu = (gnu / (self.v_nu[j] + 0.0025).sqrt()).clamp(-3.0, 3.0);
            self.nu[j] = (self.nu[j] + rlr * snu).clamp(-5.2, 1.5);

            let gom = adv * self.e_om[j];
            self.v_om[j] = 0.98 * self.v_om[j] + 0.02 * gom * gom;
            let som = (gom / (self.v_om[j] + 0.0025).sqrt()).clamp(-3.0, 3.0);
            self.omega[j] = (self.omega[j] + rlr * som).clamp(-6.28, 6.28);

            let gbd = adv * self.e_bd[j];
            self.v_bd[j] = 0.98 * self.v_bd[j] + 0.02 * gbd * gbd;
            let sbd = (gbd / (self.v_bd[j] + 0.0025).sqrt()).clamp(-3.0, 3.0);
            self.b_delta[j] = (self.b_delta[j] + rlr * sbd).clamp(-4.0, 4.0);

            let off = j * d;
            for k in 0..d {
                let gwd = adv * self.e_wd[off + k];
                self.v_wd[off + k] = 0.98 * self.v_wd[off + k] + 0.02 * gwd * gwd;
                let swd = (gwd / (self.v_wd[off + k] + 0.0025).sqrt()).clamp(-3.0, 3.0);
                self.w_delta[off + k] = (self.w_delta[off + k] + rlr * swd).clamp(-3.0, 3.0);

                let gbr = adv * self.e_wbr[off + k];
                self.v_wbr[off + k] = 0.98 * self.v_wbr[off + k] + 0.02 * gbr * gbr;
                let sbr = (gbr / (self.v_wbr[off + k] + 0.0025).sqrt()).clamp(-3.0, 3.0);
                self.w_bre[off + k] = (self.w_bre[off + k] + rlr * sbr).clamp(-3.0, 3.0);

                let gbi = adv * self.e_wbi[off + k];
                self.v_wbi[off + k] = 0.98 * self.v_wbi[off + k] + 0.02 * gbi * gbi;
                let sbi = (gbi / (self.v_wbi[off + k] + 0.0025).sqrt()).clamp(-3.0, 3.0);
                self.w_bim[off + k] = (self.w_bim[off + k] + rlr * sbi).clamp(-3.0, 3.0);
            }
        }
    }

    pub fn step_and_learn_supervised(
        &mut self,
        raw_x: &[f32],
        target: Option<usize>,
        ro_lr: f32,
        rtrl_lr: f32,
    ) -> usize {
        self.x_buf.fill(0.0);
        for i in 0..self.in_dim.min(raw_x.len()) {
            let v = raw_x[i];
            self.x_buf[i] = if self.enable_input_clamp {
                if v.is_finite() { v.clamp(-10.0, 10.0) } else { 0.0 }
            } else {
                v
            };
        }
        let x_local = std::mem::take(&mut self.x_buf);
        self.step_forward(&x_local);
        self.x_buf = x_local;
        self.fill_features();
        self.forward_logits_into();
        let mut max_l = f32::NEG_INFINITY;
        let mut best_a = 0usize;
        for a in 0..self.num_actions {
            if self.logits_buf[a] > max_l {
                max_l = self.logits_buf[a];
                best_a = a;
            }
        }
        let mut sum_e = 0.0_f32;
        for a in 0..self.num_actions {
            let e = (self.logits_buf[a] - max_l).exp();
            self.probs_buf[a] = e;
            sum_e += e;
        }
        if !sum_e.is_finite() || sum_e <= 0.0 {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
            self.probs_buf.fill(1.0 / self.num_actions as f32);
        } else {
            let inv = 1.0 / sum_e;
            for p in &mut self.probs_buf { *p *= inv; }
        }

        if let Some(y) = target {
            if y < self.num_actions && !self.had_nan {
                let f_dim = self.feat_buf.len();
                let sc = 1.0 / (self.units.max(1) as f32).sqrt();
                self.fb_re_buf.fill(0.0);
                self.fb_im_buf.fill(0.0);
                for a in 0..self.num_actions {
                    let err = (if a == y { 1.0 } else { 0.0 }) - self.probs_buf[a];
                    let off = a * f_dim;
                    for i in 0..f_dim {
                        let g = err * self.feat_buf[i];
                        self.v_aw[off + i] = 0.98 * self.v_aw[off + i] + 0.02 * g * g;
                        let st = (g / (self.v_aw[off + i] + 0.0025).sqrt()).clamp(-3.0, 3.0);
                        self.actor_w[off + i] = (self.actor_w[off + i] * 0.99995 + ro_lr * st).clamp(-8.0, 8.0);
                    }
                    self.actor_b[a] = (self.actor_b[a] + 0.25 * ro_lr * err).clamp(-4.0, 4.0);
                    for j in 0..self.units {
                        self.fb_re_buf[j] += err * self.actor_w[off + 2 * j] * sc;
                        self.fb_im_buf[j] += err * self.actor_w[off + 2 * j + 1] * sc;
                    }
                }
                if rtrl_lr > 0.0 {
                    let d = self.in_dim;
                    for j in 0..self.units {
                        let (yr, yi) = (self.y_re[j], self.y_im[j]);
                        let (gr, gi) = (self.fb_re_buf[j], self.fb_im_buf[j]);
                        let dot = gr * yr + gi * yi;
                        let inv_rho = 1.0 / self.rho[j].max(self.kappa);
                        let fr = (gr - dot * yr) * inv_rho;
                        let fi = (gi - dot * yi) * inv_rho;
                        self.e_nu[j] = fr * self.p_nu_re[j] + fi * self.p_nu_im[j];
                        self.e_om[j] = fr * self.p_om_re[j] + fi * self.p_om_im[j];
                        self.e_bd[j] = fr * self.p_bd_re[j] + fi * self.p_bd_im[j];
                        let off = j * d;
                        for k in 0..d {
                            self.e_wd[off + k] = fr * self.p_wd_re[off + k] + fi * self.p_wd_im[off + k];
                            self.e_wbr[off + k] = fr * self.p_br_re[off + k] + fi * self.p_br_im[off + k];
                            self.e_wbi[off + k] = fr * self.p_bi_re[off + k] + fi * self.p_bi_im[off + k];
                        }
                    }
                    let prev_lr = self.rtrl_lr;
                    self.rtrl_lr = rtrl_lr;
                    self.apply_rms_rtrl(1.0);
                    self.rtrl_lr = prev_lr;
                }
            }
        }
        best_a
    }

    pub fn step_and_learn_regression(
        &mut self,
        raw_x: &[f32],
        target: Option<f32>,
        ro_lr: f32,
        rtrl_lr: f32,
    ) -> f32 {
        if self.enable_state_weight_guard {
            let _ = self.scrub_state_and_weights();
            self.had_nan = false;
        }
        self.x_buf.fill(0.0);
        for i in 0..self.in_dim.min(raw_x.len()) {
            let v = raw_x[i];
            if (!v.is_finite() || v.abs() > 1e5) && self.enable_state_weight_guard {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.x_buf[i] = if self.enable_input_clamp {
                if v.is_finite() && v.abs() <= 1e5 { v.clamp(-10.0, 10.0) } else { 0.0 }
            } else {
                v
            };
        }
        let x_local = std::mem::take(&mut self.x_buf);
        self.step_forward(&x_local);
        self.x_buf = x_local;
        self.fill_features();
        let f_dim = self.feat_buf.len();
        let pred = self.actor_b[0] + fast_dot_raw(&self.actor_w[0..f_dim], &self.feat_buf);
        if !pred.is_finite() {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
        }
        if let Some(y) = target {
            if !self.had_nan && y.is_finite() {
                let err = (y - pred).clamp(-2.5, 2.5);
                let sc = 1.0 / (self.units.max(1) as f32).sqrt();
                self.fb_re_buf.fill(0.0);
                self.fb_im_buf.fill(0.0);
                for j in 0..self.units {
                    self.fb_re_buf[j] = err * self.actor_w[2 * j] * sc;
                    self.fb_im_buf[j] = err * self.actor_w[2 * j + 1] * sc;
                }
                if ro_lr > 0.0 {
                    for i in 0..f_dim {
                        let g = err * self.feat_buf[i];
                        self.v_aw[i] = 0.98 * self.v_aw[i] + 0.02 * g * g;
                        let st = (g / (self.v_aw[i] + 0.0025).sqrt()).clamp(-3.0, 3.0);
                        self.actor_w[i] = (self.actor_w[i] * 0.99995 + ro_lr * st).clamp(-8.0, 8.0);
                    }
                    self.actor_b[0] = (self.actor_b[0] + 0.5 * ro_lr * err).clamp(-4.0, 4.0);
                }
                if rtrl_lr > 0.0 {
                    let d = self.in_dim;
                    for j in 0..self.units {
                        let (yr, yi) = (self.y_re[j], self.y_im[j]);
                        let (gr, gi) = (self.fb_re_buf[j], self.fb_im_buf[j]);
                        let dot = gr * yr + gi * yi;
                        let inv_rho = 1.0 / self.rho[j].max(self.kappa);
                        let fr = (gr - dot * yr) * inv_rho;
                        let fi = (gi - dot * yi) * inv_rho;
                        self.e_nu[j] = fr * self.p_nu_re[j] + fi * self.p_nu_im[j];
                        self.e_om[j] = fr * self.p_om_re[j] + fi * self.p_om_im[j];
                        self.e_bd[j] = fr * self.p_bd_re[j] + fi * self.p_bd_im[j];
                        let off = j * d;
                        for k in 0..d {
                            self.e_wd[off + k] = fr * self.p_wd_re[off + k] + fi * self.p_wd_im[off + k];
                            self.e_wbr[off + k] = fr * self.p_br_re[off + k] + fi * self.p_br_im[off + k];
                            self.e_wbi[off + k] = fr * self.p_bi_re[off + k] + fi * self.p_bi_im[off + k];
                        }
                    }
                    let prev_lr = self.rtrl_lr;
                    self.rtrl_lr = rtrl_lr;
                    self.apply_rms_rtrl(1.0);
                    self.rtrl_lr = prev_lr;
                }
            }
        }
        pred
    }

    pub fn decide_vec(&mut self, features: &[f32], train_mode: bool, seu_nan: bool, seu_spikes: bool) -> usize {
        if self.enable_state_weight_guard {
            let _ = self.scrub_state_and_weights();
        }
        self.x_buf.fill(0.0);
        for i in 0..self.in_dim.min(features.len()) { self.x_buf[i] = features[i]; }
        if seu_nan && !self.x_buf.is_empty() { self.x_buf[0] = f32::NAN; }
        if seu_spikes && self.x_buf.len() > 1 { self.x_buf[1] = 1e28; }
        if self.enable_input_clamp {
            for v in &mut self.x_buf { *v = if v.is_finite() { v.clamp(-10.0, 10.0) } else { 0.0 }; }
        }
        let x_local = std::mem::take(&mut self.x_buf);
        self.step_forward(&x_local);
        self.x_buf = x_local;
        self.fill_features();
        let f_dim = self.feat_buf.len();
        let v_cur = self.critic_b + fast_dot_raw(&self.critic_w, &self.feat_buf);
        if !v_cur.is_finite() {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
        }

        if train_mode && self.has_pending && !self.had_nan {
            let td = (self.pending_r + self.gamma * v_cur - self.pending_v).clamp(-6.0, 6.0);
            let crit_err = td.clamp(-3.0, 3.0);
            let cent = td - self.td_mean;
            self.td_mean = 0.99 * self.td_mean + 0.01 * td;
            self.td_var = (0.98 * self.td_var + 0.02 * cent * cent).max(0.08);
            let adv = (cent / self.td_var.sqrt()).clamp(-2.5, 2.5);
            for i in 0..f_dim {
                self.critic_w[i] = (self.critic_w[i] + self.critic_lr * crit_err * self.e_critic_w[i]).clamp(-16.0, 16.0);
            }
            self.critic_b = (self.critic_b + 0.25 * self.critic_lr * crit_err * self.e_critic_b).clamp(-25.0, 25.0);
            for a in 0..self.num_actions {
                let off = a * f_dim;
                for i in 0..f_dim {
                    self.actor_w[off + i] = (self.actor_w[off + i] * 0.9999 + self.actor_lr * adv * self.e_actor_w[off + i]).clamp(-6.0, 6.0);
                }
                self.actor_b[a] = (self.actor_b[a] * 0.9999 + 0.25 * self.actor_lr * adv * self.e_actor_b[a]).clamp(-3.0, 3.0);
            }
            self.apply_rms_rtrl(adv);
        }

        self.forward_logits_into();
        let max_l = self.logits_buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_e = 0.0_f32;
        for a in 0..self.num_actions {
            let e = (self.logits_buf[a] - max_l).exp();
            self.probs_buf[a] = e;
            sum_e += e;
        }
        if !sum_e.is_finite() || sum_e <= 0.0 {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
            self.probs_buf.fill(1.0 / self.num_actions as f32);
        } else {
            let inv = 1.0 / sum_e;
            for p in &mut self.probs_buf { *p *= inv; }
        }
        let act = if train_mode && !self.had_nan {
            let u = rand_unit_f32(&mut self.rng);
            let mut cdf = 0.0;
            let mut picked = self.num_actions - 1;
            for (a, &p) in self.probs_buf.iter().enumerate() { cdf += p; if u <= cdf { picked = a; break; } }
            picked
        } else {
            self.probs_buf.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map(|x| x.0).unwrap_or(0)
        };

        if train_mode && !self.had_nan {
            let decay = self.gamma * self.trace_lambda;
            let sc = 1.0 / (self.units.max(1) as f32).sqrt();
            for i in 0..f_dim {
                self.e_critic_w[i] = (decay * self.e_critic_w[i] + self.feat_buf[i]).clamp(-4.0, 4.0);
            }
            self.e_critic_b = (decay * self.e_critic_b + 1.0).clamp(-5.0, 5.0);
            self.fb_re_buf.fill(0.0);
            self.fb_im_buf.fill(0.0);
            for a in 0..self.num_actions {
                let score = (if a == act { 1.0 } else { 0.0 }) - self.probs_buf[a];
                let off = a * f_dim;
                for i in 0..f_dim {
                    self.e_actor_w[off + i] = (decay * self.e_actor_w[off + i] + score * self.feat_buf[i]).clamp(-4.0, 4.0);
                }
                self.e_actor_b[a] = (decay * self.e_actor_b[a] + score).clamp(-5.0, 5.0);
                for j in 0..self.units {
                    self.fb_re_buf[j] += score * self.actor_w[off + 2 * j] * sc;
                    self.fb_im_buf[j] += score * self.actor_w[off + 2 * j + 1] * sc;
                }
            }
            for j in 0..self.units {
                self.fb_re_buf[j] += 0.15 * self.critic_w[2 * j] * sc;
                self.fb_im_buf[j] += 0.15 * self.critic_w[2 * j + 1] * sc;
            }
            let d = self.in_dim;
            for j in 0..self.units {
                let (yr, yi) = (self.y_re[j], self.y_im[j]);
                let (gr, gi) = (self.fb_re_buf[j], self.fb_im_buf[j]);
                let dot = gr * yr + gi * yi;
                let inv_rho = 1.0 / self.rho[j].max(self.kappa);
                let fr = (gr - dot * yr) * inv_rho;
                let fi = (gi - dot * yi) * inv_rho;
                self.e_nu[j] = (decay * self.e_nu[j] + fr * self.p_nu_re[j] + fi * self.p_nu_im[j]).clamp(-16.0, 16.0);
                self.e_om[j] = (decay * self.e_om[j] + fr * self.p_om_re[j] + fi * self.p_om_im[j]).clamp(-16.0, 16.0);
                self.e_bd[j] = (decay * self.e_bd[j] + fr * self.p_bd_re[j] + fi * self.p_bd_im[j]).clamp(-16.0, 16.0);
                let off = j * d;
                for k in 0..d {
                    self.e_wd[off + k] = (decay * self.e_wd[off + k] + fr * self.p_wd_re[off + k] + fi * self.p_wd_im[off + k]).clamp(-16.0, 16.0);
                    self.e_wbr[off + k] = (decay * self.e_wbr[off + k] + fr * self.p_br_re[off + k] + fi * self.p_br_im[off + k]).clamp(-16.0, 16.0);
                    self.e_wbi[off + k] = (decay * self.e_wbi[off + k] + fr * self.p_bi_re[off + k] + fi * self.p_bi_im[off + k]).clamp(-16.0, 16.0);
                }
            }
            if self.entropy_beta > 0.0 {
                let sum_plogp: f32 = self.probs_buf.iter().map(|&p| { let pc = p.max(1e-6); pc * pc.ln() }).sum();
                for a in 0..self.num_actions {
                    let pc = self.probs_buf[a].max(1e-6);
                    let step = self.actor_lr * self.entropy_beta * pc * (sum_plogp - pc.ln());
                    let off = a * f_dim;
                    for i in 0..f_dim { self.actor_w[off + i] += step * self.feat_buf[i]; }
                    self.actor_b[a] = (self.actor_b[a] + 0.25 * step).clamp(-3.0, 3.0);
                }
            }
            self.pending_v = self.critic_b + fast_dot_raw(&self.critic_w, &self.feat_buf);
            self.pending_r = 0.0;
            self.has_pending = true;
        }
        act
    }

    pub fn observe_step_reward(&mut self, r: f32) { self.pending_r = r; }

    pub fn end_episode_truncated(&mut self, train_mode: bool, truncated: bool) {
        if train_mode && self.has_pending && !self.had_nan {
            let v_next = if truncated { self.pending_v } else { 0.0 };
            let td = (self.pending_r + self.gamma * v_next - self.pending_v).clamp(-6.0, 6.0);
            let crit_err = td.clamp(-3.0, 3.0);
            let cent = td - self.td_mean;
            self.td_mean = 0.99 * self.td_mean + 0.01 * td;
            self.td_var = (0.98 * self.td_var + 0.02 * cent * cent).max(0.08);
            let adv = (cent / self.td_var.sqrt()).clamp(-2.5, 2.5);
            for i in 0..self.critic_w.len() {
                self.critic_w[i] = (self.critic_w[i] + self.critic_lr * crit_err * self.e_critic_w[i]).clamp(-16.0, 16.0);
            }
            self.critic_b = (self.critic_b + 0.25 * self.critic_lr * crit_err * self.e_critic_b).clamp(-25.0, 25.0);
            let f_dim = self.critic_w.len();
            for a in 0..self.num_actions {
                let off = a * f_dim;
                for i in 0..f_dim {
                    self.actor_w[off + i] = (self.actor_w[off + i] * 0.9999 + self.actor_lr * adv * self.e_actor_w[off + i]).clamp(-6.0, 6.0);
                }
                self.actor_b[a] = (self.actor_b[a] * 0.9999 + 0.25 * self.actor_lr * adv * self.e_actor_b[a]).clamp(-3.0, 3.0);
            }
            self.apply_rms_rtrl(adv);
            self.has_pending = false;
        }
    }

    pub fn predict_k_ahead_readonly(&self, window: &[&[f32]]) -> usize {
        let mut h_re = self.h_re.clone();
        let mut h_im = self.h_im.clone();
        let mut y_re = self.y_re.clone();
        let mut y_im = self.y_im.clone();
        let mut x_buf = self.x_buf.clone();
        let m = self.units;
        let d = self.in_dim;
        let kap2 = self.kappa * self.kappa;

        for &raw_x in window {
            x_buf.fill(0.0);
            for i in 0..d.min(raw_x.len()) {
                let v = raw_x[i];
                x_buf[i] = if self.enable_input_clamp {
                    if v.is_finite() { v.clamp(-10.0, 10.0) } else { 0.0 }
                } else {
                    v
                };
            }
            for j in 0..m {
                let off = j * d;
                let mut zd = self.b_delta[j];
                let mut br_in = 0.0_f32;
                let mut bi_in = 0.0_f32;
                for k in 0..d {
                    let xk = x_buf[k];
                    zd += self.w_delta[off + k] * xk;
                    br_in += self.w_bre[off + k] * xk;
                    bi_in += self.w_bim[off + k] * xk;
                }
                let exp_z = zd.clamp(-10.0, 10.0).exp();
                let delta = (1.0 + exp_z).ln().clamp(0.02, 2.0);
                let sig_nu = self.nu[j].clamp(-5.2, 1.5).exp();
                let r = (-delta * sig_nu).exp();
                let th = delta * self.omega[j];
                let (cth, sth) = (th.cos(), th.sin());
                let (lam_r, lam_i) = (r * cth, r * sth);
                let gam = (1.0 - r * r).max(1e-4).sqrt();
                let g_dt = gam * delta;

                let hp_r = h_re[j];
                let hp_i = h_im[j];
                let hr = (lam_r * hp_r - lam_i * hp_i) + g_dt * br_in;
                let hi = (lam_r * hp_i + lam_i * hp_r) + g_dt * bi_in;
                h_re[j] = hr;
                h_im[j] = hi;
                let rho = (hr * hr + hi * hi + kap2).sqrt();
                y_re[j] = hr / rho;
                y_im[j] = hi / rho;
            }
        }

        let mut feat_buf = vec![0.0_f32; self.feat_buf.len()];
        let sc = 1.0 / (m.max(1) as f32).sqrt();
        for j in 0..m {
            feat_buf[2 * j] = y_re[j] * sc;
            feat_buf[2 * j + 1] = y_im[j] * sc;
        }
        for k in 0..d {
            feat_buf[2 * m + k] = x_buf[k] * 0.5;
        }
        greedy_head_action(&self.actor_b, &self.actor_w, &feat_buf, self.num_actions)
    }
}

#[derive(Clone)]
pub struct OnlineSelectiveSsmPhBaseline {
    pub banks: Vec<SelectiveSsmBank>,
    ctrl: PhRegimeCtrl,
    retired_had_nan: bool,
    retired_seu_faults: u64,
}

impl OnlineSelectiveSsmPhBaseline {
    pub fn new(
        in_dim: usize,
        units: usize,
        num_actions: usize,
        max_regimes: usize,
        ph_delta: f32,
        ph_threshold: f32,
        ph_warmup: usize,
        seed: u64,
    ) -> Self {
        let n = max_regimes.clamp(2, 12);
        let banks = (0..n)
            .map(|b| SelectiveSsmBank::new(in_dim, units, num_actions, seed ^ (0x55AA_0000 + b as u64)))
            .collect();
        Self {
            banks,
            ctrl: PhRegimeCtrl::new(n, ph_delta, ph_threshold, ph_warmup),
            retired_had_nan: false,
            retired_seu_faults: 0,
        }
    }

    pub fn set_hyperparams(&mut self, gamma: f32, trace_lambda: f32, actor_lr: f32, critic_lr: f32, rtrl_lr: f32, entropy_beta: f32) {
        for b in &mut self.banks {
            b.gamma = gamma; b.trace_lambda = trace_lambda; b.actor_lr = actor_lr;
            b.critic_lr = critic_lr; b.rtrl_lr = rtrl_lr; b.entropy_beta = entropy_beta;
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.banks.iter().map(|b| b.memory_footprint_bytes()).sum::<usize>() + self.ctrl.phs.len() * 48
    }

    pub fn had_nan(&self) -> bool { self.retired_had_nan || self.banks.iter().any(|b| b.ever_had_nan) }

    pub fn set_probe_accept_threshold(&mut self, th: f32) {
        self.ctrl.probe_accept_threshold = th.clamp(0.0, 1.0);
    }

    pub fn set_state_weight_guard(&mut self, on: bool) {
        for b in &mut self.banks { b.enable_state_weight_guard = on; }
    }

    pub fn total_seu_faults(&self) -> u64 {
        self.retired_seu_faults
            .saturating_add(self.banks.iter().map(|b| b.seu_fault_count).sum::<u64>())
    }

    pub fn active_regime(&self) -> usize { self.ctrl.active_regime }

    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        self.banks[self.ctrl.active_regime].inject_register_bit_flip(reg_idx, bit);
    }

    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        self.banks[self.ctrl.active_regime].inject_weight_bit_flip(reg_idx, bit);
    }

    pub fn start_episode(&mut self) { self.banks[self.ctrl.active_regime].start_episode(); }

    pub fn decide_vec(&mut self, features: &[f32], train_mode: bool, seu_nan: bool, seu_spikes: bool) -> usize {
        let eff = train_mode && !self.ctrl.is_frozen_for_train();
        self.banks[self.ctrl.active_regime].decide_vec(features, eff, seu_nan, seu_spikes)
    }

    pub fn observe_step_reward(&mut self, r: f32) { self.banks[self.ctrl.active_regime].observe_step_reward(r); }

    pub fn end_episode_with_score(&mut self, norm_ret: f32, truncated: bool, train_mode: bool) {
        let prev = self.ctrl.active_regime;
        let (do_td, reset_idx) = self.ctrl.end_episode_observe(norm_ret, train_mode);
        if do_td { self.banks[prev].end_episode_truncated(true, truncated); }
        if let Some(nr) = reset_idx {
            self.retired_had_nan |= self.banks[nr].ever_had_nan;
            self.retired_seu_faults = self.retired_seu_faults.saturating_add(self.banks[nr].seu_fault_count);
            let p = &self.banks[prev];
            let (in_d, u, na, ga, tl, al, cl, rl, eb, ic, swg) = (p.in_dim, p.units, p.num_actions, p.gamma, p.trace_lambda, p.actor_lr, p.critic_lr, p.rtrl_lr, p.entropy_beta, p.enable_input_clamp, p.enable_state_weight_guard);
            let mut fresh = SelectiveSsmBank::new(in_d, u, na, 0x55AA_1000 + self.ctrl.episode_counter as u64);
            fresh.gamma = ga; fresh.trace_lambda = tl; fresh.actor_lr = al; fresh.critic_lr = cl;
            fresh.rtrl_lr = rl; fresh.entropy_beta = eb; fresh.enable_input_clamp = ic;
            fresh.enable_state_weight_guard = swg;
            self.banks[nr] = fresh;
        }
    }

    pub fn step_supervised(&mut self, raw_x: &[f32], target: usize, ro_lr: f32, rtrl_lr: f32, train: bool, win_len: usize) -> usize {
        let prev = self.ctrl.active_regime;
        let allow_learn = train && !self.ctrl.is_frozen_for_train();
        let pred = self.banks[prev].step_and_learn_supervised(raw_x, if allow_learn { Some(target) } else { None }, ro_lr, rtrl_lr);
        if let Some(nr) = self.ctrl.observe_stream_step(pred == target, train, win_len) {
            self.retired_had_nan |= self.banks[nr].ever_had_nan;
            self.retired_seu_faults = self.retired_seu_faults.saturating_add(self.banks[nr].seu_fault_count);
            let p = &self.banks[prev];
            let (in_d, u, na, ga, tl, al, cl, rl, eb, ic, swg) = (p.in_dim, p.units, p.num_actions, p.gamma, p.trace_lambda, p.actor_lr, p.critic_lr, p.rtrl_lr, p.entropy_beta, p.enable_input_clamp, p.enable_state_weight_guard);
            let mut fresh = SelectiveSsmBank::new(in_d, u, na, 0x55AA_2000 + self.ctrl.episode_counter as u64);
            fresh.gamma = ga; fresh.trace_lambda = tl; fresh.actor_lr = al; fresh.critic_lr = cl;
            fresh.rtrl_lr = rl; fresh.entropy_beta = eb; fresh.enable_input_clamp = ic;
            fresh.enable_state_weight_guard = swg;
            self.banks[nr] = fresh;
        }
        pred
    }

    pub fn predict_k_ahead_readonly(&self, window: &[&[f32]]) -> usize {
        self.banks[self.ctrl.active_regime].predict_k_ahead_readonly(window)
    }
}

/// 2. Двухслойный иерархический Fast->Slow RTRRL-LRU (точная структурная копия Band 1 + Band 2 + PH).
#[derive(Clone)]
pub struct DeepRtrrlLruBank {
    pub fast: RtrrlLruBaseline,
    pub slow: RtrrlLruBaseline,
    pool_w: Vec<f32>, // [4 * 2 * fast.units]
    in_dim: usize,
    num_actions: usize,
    actor_w: Vec<f32>,
    actor_b: Vec<f32>,
    critic_w: Vec<f32>,
    critic_b: f32,
    e_actor_w: Vec<f32>,
    e_actor_b: Vec<f32>,
    e_critic_w: Vec<f32>,
    e_critic_b: f32,
    // Предвыделенные буферы шага (0 аллокаций в куче на шаге!)
    x_buf: Vec<f32>,
    slow_in_buf: Vec<f32>,
    feat_buf: Vec<f32>,
    logits_buf: Vec<f32>,
    probs_buf: Vec<f32>,
    fb1_re_buf: Vec<f32>,
    fb1_im_buf: Vec<f32>,
    fb2_re_buf: Vec<f32>,
    fb2_im_buf: Vec<f32>,
    pub gamma: f32,
    pub trace_lambda: f32,
    pub actor_lr: f32,
    pub critic_lr: f32,
    pub rtrl_lr: f32,
    pub entropy_beta: f32,
    pub enable_input_clamp: bool,
    pub enable_state_weight_guard: bool,
    pub had_nan: bool,
    pub ever_had_nan: bool,
    pub seu_fault_count: u64,
    pending_v: f32,
    pending_r: f32,
    has_pending: bool,
    td_mean: f32,
    td_var: f32,
    rng: u64,
}

impl DeepRtrrlLruBank {
    pub fn new(in_dim: usize, fast_units: usize, slow_units: usize, num_actions: usize, seed: u64) -> Self {
        let mut fast = RtrrlLruBaseline::new_backbone(in_dim, fast_units, seed ^ 0xD111_0001);
        fast.enable_input_clamp = true;
        let mut slow = RtrrlLruBaseline::new_backbone(in_dim + 4, slow_units, seed ^ 0xD111_0002);
        slow.enable_input_clamp = true;
        // Медленный диапазон Orvieto (r in [0.92, 0.99]) для огибающей Band 2
        for j in 0..slow.units {
            let frac = j as f32 / slow.units.max(1) as f32;
            slow.nu[j] = (-4.6 + 1.8 * frac).clamp(-5.2, 1.5);
            let sign = if (j & 1) == 0 { 1.0 } else { -1.0 };
            slow.theta[j] = sign * (0.04 + 0.55 * frac);
        }
        let mut rng = splitmix64(seed ^ 0xD111_0003);
        let fast_dim = 2 * fast.units;
        let pscale = 1.0 / (fast_dim.max(1) as f32).sqrt();
        let pool_w: Vec<f32> = (0..4 * fast_dim).map(|_| rand_signed_f32(&mut rng) * pscale).collect();
        let f_dim = 2 * fast.units + 2 * slow.units + in_dim;
        let actor_w: Vec<f32> = (0..num_actions * f_dim).map(|_| rand_signed_f32(&mut rng) * 0.05).collect();
        let critic_w: Vec<f32> = (0..f_dim).map(|_| rand_signed_f32(&mut rng) * 0.05).collect();
        let u1 = fast.units;
        let u2 = slow.units;
        Self {
            fast,
            slow,
            pool_w,
            in_dim,
            num_actions,
            actor_w,
            actor_b: vec![0.0; num_actions],
            critic_w,
            critic_b: 0.0,
            e_actor_w: vec![0.0; num_actions * f_dim],
            e_actor_b: vec![0.0; num_actions],
            e_critic_w: vec![0.0; f_dim],
            e_critic_b: 0.0,
            x_buf: vec![0.0; in_dim],
            slow_in_buf: vec![0.0; in_dim + 4],
            feat_buf: vec![0.0; f_dim],
            logits_buf: vec![0.0; num_actions],
            probs_buf: vec![0.0; num_actions],
            fb1_re_buf: vec![0.0; u1],
            fb1_im_buf: vec![0.0; u1],
            fb2_re_buf: vec![0.0; u2],
            fb2_im_buf: vec![0.0; u2],
            gamma: 0.95,
            trace_lambda: 0.80,
            actor_lr: 0.055,
            critic_lr: 0.14,
            rtrl_lr: 0.008,
            entropy_beta: 0.006,
            enable_input_clamp: true,
            enable_state_weight_guard: false,
            had_nan: false,
            ever_had_nan: false,
            seu_fault_count: 0,
            pending_v: 0.0,
            pending_r: 0.0,
            has_pending: false,
            td_mean: 0.0,
            td_var: 1.0,
            rng,
        }
    }

    pub fn scrub_state_and_weights(&mut self) -> u64 {
        // fast & slow сами учитывают свои сбои в fast.seu_fault_count / slow.seu_fault_count;
        // здесь суммируем в self.seu_fault_count ТОЛЬКО собственные массивы (без двойного счёта!).
        let f_sub = self
            .fast
            .scrub_state_and_weights()
            .saturating_add(self.slow.scrub_state_and_weights());
        let mut f_own = scrub_slice_seu(&mut self.pool_w, 16.1, 0.0);
        f_own = f_own.saturating_add(scrub_slice_seu(&mut self.actor_w, 16.1, 0.0));
        f_own = f_own.saturating_add(scrub_slice_seu(&mut self.actor_b, 16.1, 0.0));
        f_own = f_own.saturating_add(scrub_slice_seu(&mut self.critic_w, 16.1, 0.0));
        if !self.critic_b.is_finite() || self.critic_b.abs() > 26.0 {
            self.critic_b = 0.0;
            f_own = f_own.saturating_add(1);
        }
        self.seu_fault_count = self.seu_fault_count.saturating_add(f_own);
        f_own.saturating_add(f_sub)
    }

    pub fn set_state_weight_guard(&mut self, on: bool) {
        self.enable_state_weight_guard = on;
        self.fast.enable_state_weight_guard = on;
        self.slow.enable_state_weight_guard = on;
    }

    pub fn total_seu_faults(&self) -> u64 {
        self.seu_fault_count
            .saturating_add(self.fast.seu_fault_count)
            .saturating_add(self.slow.seu_fault_count)
    }

    pub fn is_ever_had_nan(&self) -> bool {
        self.ever_had_nan || self.fast.ever_had_nan || self.slow.ever_had_nan
    }

    pub fn new_supervised(
        in_dim: usize,
        fast_units: usize,
        slow_units: usize,
        num_actions: usize,
        seed: u64,
    ) -> Self {
        let mut s = Self::new(in_dim, fast_units, slow_units, num_actions.max(1), seed);
        s.critic_w = Vec::new();
        s.e_actor_w = Vec::new();
        s.e_actor_b = Vec::new();
        s.e_critic_w = Vec::new();
        s
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.fast.memory_footprint_bytes()
            + self.slow.memory_footprint_bytes()
            + (self.pool_w.len()
                + self.actor_w.len()
                + self.e_actor_w.len()
                + self.actor_b.len()
                + self.e_actor_b.len()
                + self.critic_w.len()
                + self.e_critic_w.len()
                + self.x_buf.len()
                + self.slow_in_buf.len()
                + self.feat_buf.len()
                + self.logits_buf.len()
                + self.probs_buf.len()
                + self.fb1_re_buf.len()
                + self.fb1_im_buf.len()
                + self.fb2_re_buf.len()
                + self.fb2_im_buf.len())
                * 4
    }

    pub fn start_episode(&mut self) {
        self.fast.start_episode();
        self.slow.start_episode();
        self.e_actor_w.fill(0.0);
        self.e_actor_b.fill(0.0);
        self.e_critic_w.fill(0.0);
        self.e_critic_b = 0.0;
        self.has_pending = false;
        self.pending_r = 0.0;
        self.pending_v = 0.0;
        self.had_nan = false;
    }

    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let nf = 2 * self.fast.units;
        let ns = 2 * self.slow.units;
        let tot = nf + ns;
        if tot == 0 { return; }
        let r = reg_idx % tot;
        if r < nf {
            self.fast.inject_register_bit_flip(r, bit);
        } else {
            self.slow.inject_register_bit_flip(r - nf, bit);
        }
    }

    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let mut slices: [&mut [f32]; 11] = [
            &mut self.fast.nu,
            &mut self.fast.theta,
            &mut self.fast.b_re,
            &mut self.fast.b_im,
            &mut self.slow.nu,
            &mut self.slow.theta,
            &mut self.slow.b_re,
            &mut self.slow.b_im,
            &mut self.pool_w,
            &mut self.actor_w,
            &mut self.critic_w,
        ];
        flip_weight_in_slices(&mut slices, reg_idx, bit);
    }

    fn fill_joint_features(&mut self) {
        let u1 = self.fast.units;
        let u2 = self.slow.units;
        let sc1 = 1.0 / (u1.max(1) as f32).sqrt();
        let sc2 = 1.0 / (u2.max(1) as f32).sqrt();
        for j in 0..u1 {
            self.feat_buf[2 * j] = self.fast.y_re[j] * sc1;
            self.feat_buf[2 * j + 1] = self.fast.y_im[j] * sc1;
        }
        let o2 = 2 * u1;
        for j in 0..u2 {
            self.feat_buf[o2 + 2 * j] = self.slow.y_re[j] * sc2;
            self.feat_buf[o2 + 2 * j + 1] = self.slow.y_im[j] * sc2;
        }
        let ox = o2 + 2 * u2;
        for k in 0..self.in_dim {
            self.feat_buf[ox + k] = self.x_buf[k] * 0.5;
        }
    }

    fn apply_joint_td(&mut self, td_err: f32) {
        if self.had_nan || self.fast.had_nan || self.slow.had_nan {
            self.had_nan = true;
            return;
        }
        let td = td_err.clamp(-6.0, 6.0);
        let crit_err = td.clamp(-3.0, 3.0);
        let cent = td - self.td_mean;
        self.td_mean = 0.99 * self.td_mean + 0.01 * td;
        self.td_var = (0.98 * self.td_var + 0.02 * cent * cent).max(0.08);
        let adv = (cent / self.td_var.sqrt()).clamp(-2.5, 2.5);
        let f_dim = self.critic_w.len();
        for i in 0..f_dim {
            self.critic_w[i] = (self.critic_w[i] + self.critic_lr * crit_err * self.e_critic_w[i]).clamp(-16.0, 16.0);
        }
        self.critic_b = (self.critic_b + 0.25 * self.critic_lr * crit_err * self.e_critic_b).clamp(-25.0, 25.0);
        for a in 0..self.num_actions {
            let off = a * f_dim;
            for i in 0..f_dim {
                self.actor_w[off + i] = (self.actor_w[off + i] * 0.9999 + self.actor_lr * adv * self.e_actor_w[off + i]).clamp(-6.0, 6.0);
            }
            self.actor_b[a] = (self.actor_b[a] * 0.9999 + 0.25 * self.actor_lr * adv * self.e_actor_b[a]).clamp(-3.0, 3.0);
        }
        // Обновляем внутренние RTRL веса обоих слоёв
        self.fast.rtrl_lr = self.rtrl_lr;
        self.slow.rtrl_lr = self.rtrl_lr * 0.65;
        let apply_layer_rms = |lru: &mut RtrrlLruBaseline| {
            let rlr = lru.rtrl_lr;
            let d = lru.in_dim;
            for j in 0..lru.units {
                let gnu = adv * lru.e_nu[j];
                lru.v_nu[j] = 0.98 * lru.v_nu[j] + 0.02 * gnu * gnu;
                lru.nu[j] = (lru.nu[j] + rlr * (gnu / (lru.v_nu[j] + 0.0025).sqrt()).clamp(-3.0, 3.0)).clamp(-5.2, 1.5);
                let gth = adv * lru.e_th[j];
                lru.v_theta[j] = 0.98 * lru.v_theta[j] + 0.02 * gth * gth;
                lru.theta[j] = (lru.theta[j] + rlr * (gth / (lru.v_theta[j] + 0.0025).sqrt()).clamp(-3.0, 3.0)).clamp(-6.28, 6.28);
                let off = j * d;
                for k in 0..d {
                    let gbr = adv * lru.e_bre[off + k];
                    lru.v_bre[off + k] = 0.98 * lru.v_bre[off + k] + 0.02 * gbr * gbr;
                    lru.b_re[off + k] = (lru.b_re[off + k] + rlr * (gbr / (lru.v_bre[off + k] + 0.0025).sqrt()).clamp(-3.0, 3.0)).clamp(-3.0, 3.0);
                    let gbi = adv * lru.e_bim[off + k];
                    lru.v_bim[off + k] = 0.98 * lru.v_bim[off + k] + 0.02 * gbi * gbi;
                    lru.b_im[off + k] = (lru.b_im[off + k] + rlr * (gbi / (lru.v_bim[off + k] + 0.0025).sqrt()).clamp(-3.0, 3.0)).clamp(-3.0, 3.0);
                }
            }
        };
        apply_layer_rms(&mut self.fast);
        apply_layer_rms(&mut self.slow);
    }

    pub fn decide_vec(&mut self, features: &[f32], train_mode: bool, seu_nan: bool, seu_spikes: bool) -> usize {
        if self.enable_state_weight_guard {
            let _ = self.scrub_state_and_weights();
        }
        self.x_buf.fill(0.0);
        for i in 0..self.in_dim.min(features.len()) { self.x_buf[i] = features[i]; }
        if seu_nan && !self.x_buf.is_empty() { self.x_buf[0] = f32::NAN; }
        if seu_spikes && self.x_buf.len() > 1 { self.x_buf[1] = 1e28; }
        if self.enable_input_clamp {
            for v in &mut self.x_buf { *v = if v.is_finite() { v.clamp(-10.0, 10.0) } else { 0.0 }; }
        }
        self.fast.step_lru_forward(&self.x_buf);
        let u1 = self.fast.units;
        let fast_dim = 2 * u1;
        for k in 0..self.in_dim { self.slow_in_buf[k] = self.x_buf[k]; }
        for c in 0..4usize {
            let mut acc = 0.0_f32;
            let row = c * fast_dim;
            for j in 0..u1 {
                acc += self.pool_w[row + 2 * j] * self.fast.y_re[j] + self.pool_w[row + 2 * j + 1] * self.fast.y_im[j];
            }
            self.slow_in_buf[self.in_dim + c] = acc;
        }
        self.slow.step_lru_forward(&self.slow_in_buf);
        if self.fast.had_nan || self.slow.had_nan {
            self.had_nan = true;
            self.ever_had_nan = true;
        }

        self.fill_joint_features();
        let f_dim = self.feat_buf.len();
        let v_cur = self.critic_b + fast_dot_raw(&self.critic_w, &self.feat_buf);
        if !v_cur.is_finite() {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
        }
        if train_mode && self.has_pending && !self.had_nan {
            self.apply_joint_td(self.pending_r + self.gamma * v_cur - self.pending_v);
        }

        let mut max_l = f32::NEG_INFINITY;
        for a in 0..self.num_actions {
            let off = a * f_dim;
            let s = self.actor_b[a] + fast_dot_raw(&self.actor_w[off..off + f_dim], &self.feat_buf);
            if !s.is_finite() {
                if !self.had_nan {
                    self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                }
                self.had_nan = true;
                self.ever_had_nan = true;
            }
            self.logits_buf[a] = s;
            if s > max_l { max_l = s; }
        }
        let mut sum_e = 0.0_f32;
        for a in 0..self.num_actions {
            let e = (self.logits_buf[a] - max_l).exp();
            self.probs_buf[a] = e;
            sum_e += e;
        }
        if !sum_e.is_finite() || sum_e <= 0.0 {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
            self.probs_buf.fill(1.0 / self.num_actions as f32);
        } else {
            let inv = 1.0 / sum_e;
            for p in &mut self.probs_buf { *p *= inv; }
        }
        let act = if train_mode && !self.had_nan {
            let u = rand_unit_f32(&mut self.rng);
            let mut cdf = 0.0;
            let mut picked = self.num_actions - 1;
            for (a, &p) in self.probs_buf.iter().enumerate() { cdf += p; if u <= cdf { picked = a; break; } }
            picked
        } else {
            self.probs_buf.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map(|x| x.0).unwrap_or(0)
        };

        if train_mode && !self.had_nan {
            let decay = self.gamma * self.trace_lambda;
            let u2 = self.slow.units;
            let sc1 = 1.0 / (u1.max(1) as f32).sqrt();
            let sc2 = 1.0 / (u2.max(1) as f32).sqrt();
            for i in 0..f_dim {
                self.e_critic_w[i] = (decay * self.e_critic_w[i] + self.feat_buf[i]).clamp(-4.0, 4.0);
            }
            self.e_critic_b = (decay * self.e_critic_b + 1.0).clamp(-5.0, 5.0);
            self.fb1_re_buf.fill(0.0);
            self.fb1_im_buf.fill(0.0);
            self.fb2_re_buf.fill(0.0);
            self.fb2_im_buf.fill(0.0);
            for a in 0..self.num_actions {
                let score = (if a == act { 1.0 } else { 0.0 }) - self.probs_buf[a];
                let off = a * f_dim;
                for i in 0..f_dim {
                    self.e_actor_w[off + i] = (decay * self.e_actor_w[off + i] + score * self.feat_buf[i]).clamp(-4.0, 4.0);
                }
                self.e_actor_b[a] = (decay * self.e_actor_b[a] + score).clamp(-5.0, 5.0);
                for j in 0..u1 {
                    self.fb1_re_buf[j] += score * self.actor_w[off + 2 * j] * sc1;
                    self.fb1_im_buf[j] += score * self.actor_w[off + 2 * j + 1] * sc1;
                }
                for j in 0..u2 {
                    self.fb2_re_buf[j] += score * self.actor_w[off + fast_dim + 2 * j] * sc2;
                    self.fb2_im_buf[j] += score * self.actor_w[off + fast_dim + 2 * j + 1] * sc2;
                }
            }
            for j in 0..u1 {
                self.fb1_re_buf[j] += 0.15 * self.critic_w[2 * j] * sc1;
                self.fb1_im_buf[j] += 0.15 * self.critic_w[2 * j + 1] * sc1;
            }
            for j in 0..u2 {
                self.fb2_re_buf[j] += 0.15 * self.critic_w[fast_dim + 2 * j] * sc2;
                self.fb2_im_buf[j] += 0.15 * self.critic_w[fast_dim + 2 * j + 1] * sc2;
            }
            let accum_layer = |lru: &mut RtrrlLruBaseline, fbr: &[f32], fbi: &[f32]| {
                let d = lru.in_dim;
                for j in 0..lru.units {
                    let (yr, yi) = (lru.y_re[j], lru.y_im[j]);
                    let (gr, gi) = (fbr[j], fbi[j]);
                    let dot = gr * yr + gi * yi;
                    let inv_rho = 1.0 / lru.rho[j].max(lru.kappa);
                    let fr = (gr - dot * yr) * inv_rho;
                    let fi = (gi - dot * yi) * inv_rho;
                    lru.e_nu[j] = (decay * lru.e_nu[j] + fr * lru.p_nu_re[j] + fi * lru.p_nu_im[j]).clamp(-16.0, 16.0);
                    lru.e_th[j] = (decay * lru.e_th[j] + fr * lru.p_th_re[j] + fi * lru.p_th_im[j]).clamp(-16.0, 16.0);
                    let off = j * d;
                    for k in 0..d {
                        let (pbr, pbi) = (lru.p_b_re[off + k], lru.p_b_im[off + k]);
                        lru.e_bre[off + k] = (decay * lru.e_bre[off + k] + fr * pbr + fi * pbi).clamp(-16.0, 16.0);
                        lru.e_bim[off + k] = (decay * lru.e_bim[off + k] - fr * pbi + fi * pbr).clamp(-16.0, 16.0);
                    }
                }
            };
            accum_layer(&mut self.fast, &self.fb1_re_buf, &self.fb1_im_buf);
            accum_layer(&mut self.slow, &self.fb2_re_buf, &self.fb2_im_buf);

            if self.entropy_beta > 0.0 {
                let sum_plogp: f32 = self.probs_buf.iter().map(|&p| { let pc = p.max(1e-6); pc * pc.ln() }).sum();
                for a in 0..self.num_actions {
                    let pc = self.probs_buf[a].max(1e-6);
                    let step = self.actor_lr * self.entropy_beta * pc * (sum_plogp - pc.ln());
                    let off = a * f_dim;
                    for i in 0..f_dim { self.actor_w[off + i] += step * self.feat_buf[i]; }
                    self.actor_b[a] = (self.actor_b[a] + 0.25 * step).clamp(-3.0, 3.0);
                }
            }
            self.pending_v = self.critic_b + fast_dot_raw(&self.critic_w, &self.feat_buf);
            self.pending_r = 0.0;
            self.has_pending = true;
        }
        act
    }

    pub fn observe_step_reward(&mut self, r: f32) { self.pending_r = r; }

    pub fn end_episode_truncated(&mut self, train_mode: bool, truncated: bool) {
        if train_mode && self.has_pending && !self.had_nan {
            let v_next = if truncated { self.pending_v } else { 0.0 };
            self.apply_joint_td(self.pending_r + self.gamma * v_next - self.pending_v);
            self.has_pending = false;
        }
    }

    pub fn step_and_learn_supervised(&mut self, raw_x: &[f32], target: Option<usize>, ro_lr: f32, rtrl_lr: f32) -> usize {
        if self.enable_state_weight_guard {
            let _ = self.scrub_state_and_weights();
        }
        self.x_buf.fill(0.0);
        for i in 0..self.in_dim.min(raw_x.len()) {
            let v = raw_x[i];
            self.x_buf[i] = if self.enable_input_clamp { if v.is_finite() { v.clamp(-10.0, 10.0) } else { 0.0 } } else { v };
        }
        self.fast.step_lru_forward(&self.x_buf);
        let u1 = self.fast.units;
        let fast_dim = 2 * u1;
        for k in 0..self.in_dim { self.slow_in_buf[k] = self.x_buf[k]; }
        for c in 0..4usize {
            let mut acc = 0.0_f32;
            let row = c * fast_dim;
            for j in 0..u1 { acc += self.pool_w[row + 2 * j] * self.fast.y_re[j] + self.pool_w[row + 2 * j + 1] * self.fast.y_im[j]; }
            self.slow_in_buf[self.in_dim + c] = acc;
        }
        self.slow.step_lru_forward(&self.slow_in_buf);
        if self.fast.had_nan || self.slow.had_nan {
            self.had_nan = true;
            self.ever_had_nan = true;
        }

        self.fill_joint_features();
        let f_dim = self.feat_buf.len();
        let mut max_l = f32::NEG_INFINITY;
        let mut best = 0usize;
        for a in 0..self.num_actions {
            let off = a * f_dim;
            let s = self.actor_b[a] + fast_dot_raw(&self.actor_w[off..off + f_dim], &self.feat_buf);
            if !s.is_finite() {
                if !self.had_nan {
                    self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                }
                self.had_nan = true;
                self.ever_had_nan = true;
            }
            self.logits_buf[a] = s;
            if s > max_l { max_l = s; best = a; }
        }
        let mut sum_e = 0.0_f32;
        for a in 0..self.num_actions { let e = (self.logits_buf[a] - max_l).exp(); self.probs_buf[a] = e; sum_e += e; }
        if !sum_e.is_finite() || sum_e <= 0.0 {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
            self.probs_buf.fill(1.0 / self.num_actions as f32);
        } else {
            let inv = 1.0 / sum_e;
            for p in &mut self.probs_buf { *p *= inv; }
        }

        if let Some(y) = target {
            if y < self.num_actions && !self.had_nan {
                let u2 = self.slow.units;
                let sc1 = 1.0 / (u1.max(1) as f32).sqrt();
                let sc2 = 1.0 / (u2.max(1) as f32).sqrt();
                self.fb1_re_buf.fill(0.0);
                self.fb1_im_buf.fill(0.0);
                self.fb2_re_buf.fill(0.0);
                self.fb2_im_buf.fill(0.0);
                for a in 0..self.num_actions {
                    let err = (if a == y { 1.0 } else { 0.0 }) - self.probs_buf[a];
                    let off = a * f_dim;
                    for i in 0..f_dim {
                        self.actor_w[off + i] = (self.actor_w[off + i] * 0.99995 + ro_lr * err * self.feat_buf[i]).clamp(-8.0, 8.0);
                    }
                    self.actor_b[a] = (self.actor_b[a] + 0.25 * ro_lr * err).clamp(-4.0, 4.0);
                    for j in 0..u1 { self.fb1_re_buf[j] += err * self.actor_w[off + 2 * j] * sc1; self.fb1_im_buf[j] += err * self.actor_w[off + 2 * j + 1] * sc1; }
                    for j in 0..u2 { self.fb2_re_buf[j] += err * self.actor_w[off + fast_dim + 2 * j] * sc2; self.fb2_im_buf[j] += err * self.actor_w[off + fast_dim + 2 * j + 1] * sc2; }
                }
                if rtrl_lr > 0.0 {
                    let upd = |lru: &mut RtrrlLruBaseline, fbr: &[f32], fbi: &[f32], rlr: f32| {
                        let d = lru.in_dim;
                        for j in 0..lru.units {
                            let (yr, yi) = (lru.y_re[j], lru.y_im[j]);
                            let (gr, gi) = (fbr[j], fbi[j]);
                            let dot = gr * yr + gi * yi;
                            let inv_rho = 1.0 / lru.rho[j].max(lru.kappa);
                            let fr = (gr - dot * yr) * inv_rho;
                            let fi = (gi - dot * yi) * inv_rho;
                            let gnu = fr * lru.p_nu_re[j] + fi * lru.p_nu_im[j];
                            lru.v_nu[j] = 0.98 * lru.v_nu[j] + 0.02 * gnu * gnu;
                            lru.nu[j] = (lru.nu[j] + rlr * (gnu / (lru.v_nu[j] + 0.0025).sqrt()).clamp(-3.0, 3.0)).clamp(-5.2, 1.5);
                            let gth = fr * lru.p_th_re[j] + fi * lru.p_th_im[j];
                            lru.v_theta[j] = 0.98 * lru.v_theta[j] + 0.02 * gth * gth;
                            lru.theta[j] = (lru.theta[j] + rlr * (gth / (lru.v_theta[j] + 0.0025).sqrt()).clamp(-3.0, 3.0)).clamp(-6.28, 6.28);
                            let off = j * d;
                            for k in 0..d {
                                let (pbr, pbi) = (lru.p_b_re[off + k], lru.p_b_im[off + k]);
                                let gbr = fr * pbr + fi * pbi;
                                let gbi = -fr * pbi + fi * pbr;
                                lru.v_bre[off + k] = 0.98 * lru.v_bre[off + k] + 0.02 * gbr * gbr;
                                lru.b_re[off + k] = (lru.b_re[off + k] + rlr * (gbr / (lru.v_bre[off + k] + 0.0025).sqrt()).clamp(-3.0, 3.0)).clamp(-3.0, 3.0);
                                lru.v_bim[off + k] = 0.98 * lru.v_bim[off + k] + 0.02 * gbi * gbi;
                                lru.b_im[off + k] = (lru.b_im[off + k] + rlr * (gbi / (lru.v_bim[off + k] + 0.0025).sqrt()).clamp(-3.0, 3.0)).clamp(-3.0, 3.0);
                            }
                        }
                    };
                    upd(&mut self.fast, &self.fb1_re_buf, &self.fb1_im_buf, rtrl_lr);
                    upd(&mut self.slow, &self.fb2_re_buf, &self.fb2_im_buf, rtrl_lr * 0.65);
                }
            }
        }
        best
    }

    pub fn step_and_learn_regression(
        &mut self,
        raw_x: &[f32],
        target: Option<f32>,
        ro_lr: f32,
        rtrl_lr: f32,
    ) -> f32 {
        if self.enable_state_weight_guard {
            let _ = self.scrub_state_and_weights();
            self.had_nan = false;
            self.fast.had_nan = false;
            self.slow.had_nan = false;
        }
        self.x_buf.fill(0.0);
        for i in 0..self.in_dim.min(raw_x.len()) {
            let v = raw_x[i];
            if (!v.is_finite() || v.abs() > 1e5) && self.enable_state_weight_guard {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.x_buf[i] = if self.enable_input_clamp {
                if v.is_finite() && v.abs() <= 1e5 { v.clamp(-10.0, 10.0) } else { 0.0 }
            } else {
                v
            };
        }
        self.fast.step_lru_forward(&self.x_buf);
        let u1 = self.fast.units;
        let fast_dim = 2 * u1;
        for k in 0..self.in_dim {
            self.slow_in_buf[k] = self.x_buf[k];
        }
        for c in 0..4usize {
            let mut acc = 0.0_f32;
            let row = c * fast_dim;
            for j in 0..u1 {
                acc += self.pool_w[row + 2 * j] * self.fast.y_re[j]
                    + self.pool_w[row + 2 * j + 1] * self.fast.y_im[j];
            }
            self.slow_in_buf[self.in_dim + c] = acc;
        }
        self.slow.step_lru_forward(&self.slow_in_buf);
        if self.fast.had_nan || self.slow.had_nan {
            self.had_nan = true;
            self.ever_had_nan = true;
        }

        self.fill_joint_features();
        let f_dim = self.feat_buf.len();
        let pred = self.actor_b[0] + fast_dot_raw(&self.actor_w[0..f_dim], &self.feat_buf);
        if !pred.is_finite() {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
        }

        if let Some(y) = target {
            if !self.had_nan && y.is_finite() {
                let u2 = self.slow.units;
                let sc1 = 1.0 / (u1.max(1) as f32).sqrt();
                let sc2 = 1.0 / (u2.max(1) as f32).sqrt();
                let err = (y - pred).clamp(-2.5, 2.5);
                self.fb1_re_buf.fill(0.0);
                self.fb1_im_buf.fill(0.0);
                self.fb2_re_buf.fill(0.0);
                self.fb2_im_buf.fill(0.0);
                for j in 0..u1 {
                    self.fb1_re_buf[j] = err * self.actor_w[2 * j] * sc1;
                    self.fb1_im_buf[j] = err * self.actor_w[2 * j + 1] * sc1;
                }
                for j in 0..u2 {
                    self.fb2_re_buf[j] = err * self.actor_w[fast_dim + 2 * j] * sc2;
                    self.fb2_im_buf[j] = err * self.actor_w[fast_dim + 2 * j + 1] * sc2;
                }
                if ro_lr > 0.0 {
                    for i in 0..f_dim {
                        self.actor_w[i] =
                            (self.actor_w[i] * 0.99995 + ro_lr * err * self.feat_buf[i]).clamp(-8.0, 8.0);
                    }
                    self.actor_b[0] = (self.actor_b[0] + 0.5 * ro_lr * err).clamp(-4.0, 4.0);
                }
                if rtrl_lr > 0.0 {
                    let upd = |lru: &mut RtrrlLruBaseline, fbr: &[f32], fbi: &[f32], rlr: f32| {
                        let d = lru.in_dim;
                        for j in 0..lru.units {
                            let (yr, yi) = (lru.y_re[j], lru.y_im[j]);
                            let (gr, gi) = (fbr[j], fbi[j]);
                            let dot = gr * yr + gi * yi;
                            let inv_rho = 1.0 / lru.rho[j].max(lru.kappa);
                            let fr = (gr - dot * yr) * inv_rho;
                            let fi = (gi - dot * yi) * inv_rho;
                            let gnu = fr * lru.p_nu_re[j] + fi * lru.p_nu_im[j];
                            lru.v_nu[j] = 0.98 * lru.v_nu[j] + 0.02 * gnu * gnu;
                            lru.nu[j] = (lru.nu[j]
                                + rlr * (gnu / (lru.v_nu[j] + 0.0025).sqrt()).clamp(-3.0, 3.0))
                                .clamp(-5.2, 1.5);
                            let gth = fr * lru.p_th_re[j] + fi * lru.p_th_im[j];
                            lru.v_theta[j] = 0.98 * lru.v_theta[j] + 0.02 * gth * gth;
                            lru.theta[j] = (lru.theta[j]
                                + rlr * (gth / (lru.v_theta[j] + 0.0025).sqrt()).clamp(-3.0, 3.0))
                                .clamp(-6.28, 6.28);
                            let off = j * d;
                            for k in 0..d {
                                let (pbr, pbi) = (lru.p_b_re[off + k], lru.p_b_im[off + k]);
                                let gbr = fr * pbr + fi * pbi;
                                let gbi = -fr * pbi + fi * pbr;
                                lru.v_bre[off + k] = 0.98 * lru.v_bre[off + k] + 0.02 * gbr * gbr;
                                lru.b_re[off + k] = (lru.b_re[off + k]
                                    + rlr
                                        * (gbr / (lru.v_bre[off + k] + 0.0025).sqrt())
                                            .clamp(-3.0, 3.0))
                                .clamp(-3.0, 3.0);
                                lru.v_bim[off + k] = 0.98 * lru.v_bim[off + k] + 0.02 * gbi * gbi;
                                lru.b_im[off + k] = (lru.b_im[off + k]
                                    + rlr
                                        * (gbi / (lru.v_bim[off + k] + 0.0025).sqrt())
                                            .clamp(-3.0, 3.0))
                                .clamp(-3.0, 3.0);
                            }
                        }
                    };
                    upd(&mut self.fast, &self.fb1_re_buf, &self.fb1_im_buf, rtrl_lr);
                    upd(&mut self.slow, &self.fb2_re_buf, &self.fb2_im_buf, rtrl_lr * 0.65);
                }
            }
        }
        pred
    }

    pub fn predict_k_ahead_readonly(&self, window: &[&[f32]]) -> usize {
        let mut f_h_re = self.fast.h_re.clone();
        let mut f_h_im = self.fast.h_im.clone();
        let mut f_y_re = self.fast.y_re.clone();
        let mut f_y_im = self.fast.y_im.clone();
        let mut s_h_re = self.slow.h_re.clone();
        let mut s_h_im = self.slow.h_im.clone();
        let mut s_y_re = self.slow.y_re.clone();
        let mut s_y_im = self.slow.y_im.clone();
        let mut x_buf = self.x_buf.clone();
        let mut slow_in_buf = vec![0.0_f32; self.slow_in_buf.len()];
        let u1 = self.fast.units;
        let u2 = self.slow.units;
        let fast_dim = 2 * u1;

        for &raw_x in window {
            x_buf.fill(0.0);
            for i in 0..self.in_dim.min(raw_x.len()) {
                let v = raw_x[i];
                x_buf[i] = if self.enable_input_clamp {
                    if v.is_finite() { v.clamp(-10.0, 10.0) } else { 0.0 }
                } else {
                    v
                };
            }
            self.fast
                .step_lru_forward_readonly(&x_buf, &mut f_h_re, &mut f_h_im, &mut f_y_re, &mut f_y_im);
            for k in 0..self.in_dim {
                slow_in_buf[k] = x_buf[k];
            }
            for c in 0..4usize {
                let mut acc = 0.0_f32;
                let row = c * fast_dim;
                for j in 0..u1 {
                    acc += self.pool_w[row + 2 * j] * f_y_re[j]
                        + self.pool_w[row + 2 * j + 1] * f_y_im[j];
                }
                slow_in_buf[self.in_dim + c] = acc;
            }
            self.slow.step_lru_forward_readonly(
                &slow_in_buf,
                &mut s_h_re,
                &mut s_h_im,
                &mut s_y_re,
                &mut s_y_im,
            );
        }

        let mut feat_buf = vec![0.0_f32; self.feat_buf.len()];
        let sc1 = 1.0 / (u1.max(1) as f32).sqrt();
        let sc2 = 1.0 / (u2.max(1) as f32).sqrt();
        for j in 0..u1 {
            feat_buf[2 * j] = f_y_re[j] * sc1;
            feat_buf[2 * j + 1] = f_y_im[j] * sc1;
        }
        let o2 = 2 * u1;
        for j in 0..u2 {
            feat_buf[o2 + 2 * j] = s_y_re[j] * sc2;
            feat_buf[o2 + 2 * j + 1] = s_y_im[j] * sc2;
        }
        let ox = o2 + 2 * u2;
        for k in 0..self.in_dim {
            feat_buf[ox + k] = x_buf[k] * 0.5;
        }
        greedy_head_action(&self.actor_b, &self.actor_w, &feat_buf, self.num_actions)
    }
}

#[derive(Clone)]
pub struct DeepRtrrlLruPhBaseline {
    pub banks: Vec<DeepRtrrlLruBank>,
    ctrl: PhRegimeCtrl,
    retired_had_nan: bool,
    retired_seu_faults: u64,
}

impl DeepRtrrlLruPhBaseline {
    pub fn new(
        in_dim: usize,
        fast_units: usize,
        slow_units: usize,
        num_actions: usize,
        max_regimes: usize,
        ph_delta: f32,
        ph_threshold: f32,
        ph_warmup: usize,
        seed: u64,
    ) -> Self {
        let n = max_regimes.clamp(2, 12);
        let banks = (0..n)
            .map(|b| DeepRtrrlLruBank::new(in_dim, fast_units, slow_units, num_actions, seed ^ (0xD111_0000 + b as u64)))
            .collect();
        Self {
            banks,
            ctrl: PhRegimeCtrl::new(n, ph_delta, ph_threshold, ph_warmup),
            retired_had_nan: false,
            retired_seu_faults: 0,
        }
    }

    pub fn set_hyperparams(&mut self, gamma: f32, trace_lambda: f32, actor_lr: f32, critic_lr: f32, rtrl_lr: f32, entropy_beta: f32) {
        for b in &mut self.banks {
            b.gamma = gamma; b.trace_lambda = trace_lambda; b.actor_lr = actor_lr;
            b.critic_lr = critic_lr; b.rtrl_lr = rtrl_lr; b.entropy_beta = entropy_beta;
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.banks.iter().map(|b| b.memory_footprint_bytes()).sum::<usize>() + self.ctrl.phs.len() * 48
    }

    pub fn had_nan(&self) -> bool { self.retired_had_nan || self.banks.iter().any(|b| b.is_ever_had_nan()) }

    pub fn set_probe_accept_threshold(&mut self, th: f32) {
        self.ctrl.probe_accept_threshold = th.clamp(0.0, 1.0);
    }

    pub fn set_state_weight_guard(&mut self, on: bool) {
        for b in &mut self.banks { b.set_state_weight_guard(on); }
    }

    pub fn total_seu_faults(&self) -> u64 {
        self.retired_seu_faults
            .saturating_add(self.banks.iter().map(|b| b.total_seu_faults()).sum::<u64>())
    }

    pub fn active_regime(&self) -> usize { self.ctrl.active_regime }

    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        self.banks[self.ctrl.active_regime].inject_register_bit_flip(reg_idx, bit);
    }

    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        self.banks[self.ctrl.active_regime].inject_weight_bit_flip(reg_idx, bit);
    }

    pub fn start_episode(&mut self) { self.banks[self.ctrl.active_regime].start_episode(); }

    pub fn decide_vec(&mut self, features: &[f32], train_mode: bool, seu_nan: bool, seu_spikes: bool) -> usize {
        let eff = train_mode && !self.ctrl.is_frozen_for_train();
        self.banks[self.ctrl.active_regime].decide_vec(features, eff, seu_nan, seu_spikes)
    }

    pub fn observe_step_reward(&mut self, r: f32) { self.banks[self.ctrl.active_regime].observe_step_reward(r); }

    pub fn end_episode_with_score(&mut self, norm_ret: f32, truncated: bool, train_mode: bool) {
        let prev = self.ctrl.active_regime;
        let (do_td, reset_idx) = self.ctrl.end_episode_observe(norm_ret, train_mode);
        if do_td { self.banks[prev].end_episode_truncated(true, truncated); }
        if let Some(nr) = reset_idx {
            self.retired_had_nan |= self.banks[nr].is_ever_had_nan();
            self.retired_seu_faults = self.retired_seu_faults.saturating_add(self.banks[nr].total_seu_faults());
            let p = &self.banks[prev];
            let (in_d, u1, u2, na, ga, tl, al, cl, rl, eb, ic, swg) = (p.in_dim, p.fast.units, p.slow.units, p.num_actions, p.gamma, p.trace_lambda, p.actor_lr, p.critic_lr, p.rtrl_lr, p.entropy_beta, p.enable_input_clamp, p.enable_state_weight_guard);
            let mut fresh = DeepRtrrlLruBank::new(in_d, u1, u2, na, 0xD111_1000 + self.ctrl.episode_counter as u64);
            fresh.gamma = ga; fresh.trace_lambda = tl; fresh.actor_lr = al; fresh.critic_lr = cl;
            fresh.rtrl_lr = rl; fresh.entropy_beta = eb; fresh.enable_input_clamp = ic;
            fresh.set_state_weight_guard(swg);
            self.banks[nr] = fresh;
        }
    }

    pub fn step_supervised(&mut self, raw_x: &[f32], target: usize, ro_lr: f32, rtrl_lr: f32, train: bool, win_len: usize) -> usize {
        let prev = self.ctrl.active_regime;
        let allow_learn = train && !self.ctrl.is_frozen_for_train();
        let pred = self.banks[prev].step_and_learn_supervised(raw_x, if allow_learn { Some(target) } else { None }, ro_lr, rtrl_lr);
        if let Some(nr) = self.ctrl.observe_stream_step(pred == target, train, win_len) {
            self.retired_had_nan |= self.banks[nr].is_ever_had_nan();
            self.retired_seu_faults = self.retired_seu_faults.saturating_add(self.banks[nr].total_seu_faults());
            let p = &self.banks[prev];
            let (in_d, u1, u2, na, ga, tl, al, cl, rl, eb, ic, swg) = (p.in_dim, p.fast.units, p.slow.units, p.num_actions, p.gamma, p.trace_lambda, p.actor_lr, p.critic_lr, p.rtrl_lr, p.entropy_beta, p.enable_input_clamp, p.enable_state_weight_guard);
            let mut fresh = DeepRtrrlLruBank::new(in_d, u1, u2, na, 0xD111_2000 + self.ctrl.episode_counter as u64);
            fresh.gamma = ga; fresh.trace_lambda = tl; fresh.actor_lr = al; fresh.critic_lr = cl;
            fresh.rtrl_lr = rl; fresh.entropy_beta = eb; fresh.enable_input_clamp = ic;
            fresh.set_state_weight_guard(swg);
            self.banks[nr] = fresh;
        }
        pred
    }

    pub fn predict_k_ahead_readonly(&self, window: &[&[f32]]) -> usize {
        self.banks[self.ctrl.active_regime].predict_k_ahead_readonly(window)
    }
}

/// 3. Пошаговый онлайн-GRU (zero-alloc AVX2+FMA; в supervised-режиме — без фиктивной RL-памяти и с опц. BPTT-кольцом) + PageHinkley + Clamp.
#[derive(Clone)]
pub struct GruPerStepBank {
    pub gru: OnlineGruBaseline,
    pub is_supervised: bool,
    pub bptt_len: usize,
    pub trace_lambda: f32,
    pub entropy_beta: f32,
    // RL eligibility traces (пустые в supervised-режиме -> 0 фиктивной памяти!)
    e_actor_w: Vec<f32>,
    e_actor_b: Vec<f32>,
    e_wv: Vec<f32>,
    e_bv: f32,
    e_wz: Vec<f32>,
    e_uz: Vec<f32>,
    e_bz: Vec<f32>,
    e_wr: Vec<f32>,
    e_ur: Vec<f32>,
    e_br: Vec<f32>,
    e_wn: Vec<f32>,
    e_un: Vec<f32>,
    e_bn: Vec<f32>,
    // Предвыделенные рабочие буферы (0 аллокаций в куче на шаге!)
    x_buf: Vec<f32>,
    hp_buf: Vec<f32>,
    z_buf: Vec<f32>,
    r_buf: Vec<f32>,
    n_buf: Vec<f32>,
    hn_buf: Vec<f32>,
    rhp_buf: Vec<f32>,
    logits_buf: Vec<f32>,
    probs_buf: Vec<f32>,
    dh_buf: Vec<f32>,
    dh_prev_buf: Vec<f32>,
    dr_buf: Vec<f32>,
    // Буферы градиентов (H*H выделяются только для RL или truncated BPTT > 1)
    g_wo: Vec<f32>,
    g_bo: Vec<f32>,
    g_wv: Vec<f32>,
    g_wz: Vec<f32>,
    g_uz: Vec<f32>,
    g_bz: Vec<f32>,
    g_wr: Vec<f32>,
    g_ur: Vec<f32>,
    g_br: Vec<f32>,
    g_wn: Vec<f32>,
    g_un: Vec<f32>,
    g_bn: Vec<f32>,
    // Предвыделенное кольцо для усечённого BPTT (пустое при bptt_len <= 1)
    ring_x: Vec<f32>,
    ring_hp: Vec<f32>,
    ring_z: Vec<f32>,
    ring_r: Vec<f32>,
    ring_n: Vec<f32>,
    ring_head: usize,
    ring_count: usize,
    pending_v: f32,
    pending_r: f32,
    has_pending: bool,
    td_mean: f32,
    td_var: f32,
    adam_t: usize,
}

impl GruPerStepBank {
    pub fn new(x_dim: usize, h_dim: usize, num_actions: usize, seed: u64) -> Self {
        let mut gru = OnlineGruBaseline::new(x_dim, h_dim, num_actions, seed);
        gru.enable_input_clamp = true;
        gru.lr = 0.0025;
        gru.gamma = 0.95;
        let hx = h_dim * x_dim;
        let hh = h_dim * h_dim;
        // Онлайн усечённый BPTT(1, 4) в RL-режиме (предвыделенное кольцо, 0 аллокаций на шаге),
        // оптимальный баланс между переносом кредита и свежестью активаций при пошаговом Adam.
        let blen = 4usize;
        Self {
            gru,
            is_supervised: false,
            bptt_len: blen,
            trace_lambda: 0.80,
            entropy_beta: 0.008,
            e_actor_w: vec![0.0; num_actions * h_dim],
            e_actor_b: vec![0.0; num_actions],
            e_wv: vec![0.0; h_dim],
            e_bv: 0.0,
            e_wz: vec![0.0; hx],
            e_uz: vec![0.0; hh],
            e_bz: vec![0.0; h_dim],
            e_wr: vec![0.0; hx],
            e_ur: vec![0.0; hh],
            e_br: vec![0.0; h_dim],
            e_wn: vec![0.0; hx],
            e_un: vec![0.0; hh],
            e_bn: vec![0.0; h_dim],
            x_buf: vec![0.0; x_dim],
            hp_buf: vec![0.0; h_dim],
            z_buf: vec![0.0; h_dim],
            r_buf: vec![0.0; h_dim],
            n_buf: vec![0.0; h_dim],
            hn_buf: vec![0.0; h_dim],
            rhp_buf: vec![0.0; h_dim],
            logits_buf: vec![0.0; num_actions],
            probs_buf: vec![0.0; num_actions],
            dh_buf: vec![0.0; h_dim],
            dh_prev_buf: vec![0.0; h_dim],
            dr_buf: vec![0.0; h_dim],
            g_wo: vec![0.0; num_actions * h_dim],
            g_bo: vec![0.0; num_actions],
            g_wv: vec![0.0; h_dim],
            g_wz: vec![0.0; hx],
            g_uz: vec![0.0; hh],
            g_bz: vec![0.0; h_dim],
            g_wr: vec![0.0; hx],
            g_ur: vec![0.0; hh],
            g_br: vec![0.0; h_dim],
            g_wn: vec![0.0; hx],
            g_un: vec![0.0; hh],
            g_bn: vec![0.0; h_dim],
            ring_x: vec![0.0; blen * x_dim],
            ring_hp: vec![0.0; blen * h_dim],
            ring_z: vec![0.0; blen * h_dim],
            ring_r: vec![0.0; blen * h_dim],
            ring_n: vec![0.0; blen * h_dim],
            ring_head: 0,
            ring_count: 0,
            pending_v: 0.0,
            pending_r: 0.0,
            has_pending: false,
            td_mean: 0.0,
            td_var: 1.0,
            adam_t: 0,
        }
    }

    /// Конструктор чисто для supervised-задач (0 фиктивных RL-следов в памяти, AVX2+FMA, zero heap allocs).
    /// При `bptt_len == 1` — пошаговое обновление без буфера матриц градиентов; при `bptt_len > 1` — усечённый BPTT по предвыделенному кольцу.
    pub fn new_supervised(
        x_dim: usize,
        h_dim: usize,
        num_actions: usize,
        bptt_len: usize,
        seed: u64,
    ) -> Self {
        let mut gru = OnlineGruBaseline::new_supervised(x_dim, h_dim, num_actions, seed);
        gru.enable_input_clamp = true;
        gru.lr = 0.005;
        gru.gamma = 0.95;
        let blen = bptt_len.max(1);
        let hx = h_dim * x_dim;
        let hh = h_dim * h_dim;
        let need_grad_mats = blen > 1;
        let ring_cap = if blen > 1 { blen } else { 0 };
        Self {
            gru,
            is_supervised: true,
            bptt_len: blen,
            trace_lambda: 0.80,
            entropy_beta: 0.0,
            e_actor_w: Vec::new(),
            e_actor_b: Vec::new(),
            e_wv: Vec::new(),
            e_bv: 0.0,
            e_wz: Vec::new(),
            e_uz: Vec::new(),
            e_bz: Vec::new(),
            e_wr: Vec::new(),
            e_ur: Vec::new(),
            e_br: Vec::new(),
            e_wn: Vec::new(),
            e_un: Vec::new(),
            e_bn: Vec::new(),
            x_buf: vec![0.0; x_dim],
            hp_buf: vec![0.0; h_dim],
            z_buf: vec![0.0; h_dim],
            r_buf: vec![0.0; h_dim],
            n_buf: vec![0.0; h_dim],
            hn_buf: vec![0.0; h_dim],
            rhp_buf: vec![0.0; h_dim],
            logits_buf: vec![0.0; num_actions],
            probs_buf: vec![0.0; num_actions],
            dh_buf: vec![0.0; h_dim],
            dh_prev_buf: if need_grad_mats { vec![0.0; h_dim] } else { Vec::new() },
            dr_buf: vec![0.0; h_dim],
            g_wo: if need_grad_mats { vec![0.0; num_actions * h_dim] } else { Vec::new() },
            g_bo: if need_grad_mats { vec![0.0; num_actions] } else { Vec::new() },
            g_wv: Vec::new(),
            g_wz: if need_grad_mats { vec![0.0; hx] } else { Vec::new() },
            g_uz: if need_grad_mats { vec![0.0; hh] } else { Vec::new() },
            g_bz: if need_grad_mats { vec![0.0; h_dim] } else { Vec::new() },
            g_wr: if need_grad_mats { vec![0.0; hx] } else { Vec::new() },
            g_ur: if need_grad_mats { vec![0.0; hh] } else { Vec::new() },
            g_br: if need_grad_mats { vec![0.0; h_dim] } else { Vec::new() },
            g_wn: if need_grad_mats { vec![0.0; hx] } else { Vec::new() },
            g_un: if need_grad_mats { vec![0.0; hh] } else { Vec::new() },
            g_bn: if need_grad_mats { vec![0.0; h_dim] } else { Vec::new() },
            ring_x: vec![0.0; ring_cap * x_dim],
            ring_hp: vec![0.0; ring_cap * h_dim],
            ring_z: vec![0.0; ring_cap * h_dim],
            ring_r: vec![0.0; ring_cap * h_dim],
            ring_n: vec![0.0; ring_cap * h_dim],
            ring_head: 0,
            ring_count: 0,
            pending_v: 0.0,
            pending_r: 0.0,
            has_pending: false,
            td_mean: 0.0,
            td_var: 1.0,
            adam_t: 0,
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.gru.memory_footprint_bytes()
            + (self.e_actor_w.len()
                + self.e_actor_b.len()
                + self.e_wv.len()
                + self.e_wz.len()
                + self.e_uz.len()
                + self.e_bz.len()
                + self.e_wr.len()
                + self.e_ur.len()
                + self.e_br.len()
                + self.e_wn.len()
                + self.e_un.len()
                + self.e_bn.len()
                + self.x_buf.len()
                + self.hp_buf.len()
                + self.z_buf.len()
                + self.r_buf.len()
                + self.n_buf.len()
                + self.hn_buf.len()
                + self.rhp_buf.len()
                + self.logits_buf.len()
                + self.probs_buf.len()
                + self.dh_buf.len()
                + self.dh_prev_buf.len()
                + self.dr_buf.len()
                + self.g_wo.len()
                + self.g_bo.len()
                + self.g_wv.len()
                + self.g_wz.len()
                + self.g_uz.len()
                + self.g_bz.len()
                + self.g_wr.len()
                + self.g_ur.len()
                + self.g_br.len()
                + self.g_wn.len()
                + self.g_un.len()
                + self.g_bn.len()
                + self.ring_x.len()
                + self.ring_hp.len()
                + self.ring_z.len()
                + self.ring_r.len()
                + self.ring_n.len())
                * 4
    }

    pub fn set_bptt_len(&mut self, bptt_len: usize) {
        let blen = bptt_len.max(1);
        self.bptt_len = blen;
        let ring_cap = if blen > 1 { blen } else { 0 };
        let xd = self.gru.x_dim;
        let hd = self.gru.h_dim;
        self.ring_x = vec![0.0; ring_cap * xd];
        self.ring_hp = vec![0.0; ring_cap * hd];
        self.ring_z = vec![0.0; ring_cap * hd];
        self.ring_r = vec![0.0; ring_cap * hd];
        self.ring_n = vec![0.0; ring_cap * hd];
        self.ring_head = 0;
        self.ring_count = 0;
    }

    pub fn start_episode(&mut self) {
        self.gru.start_episode();
        if !self.is_supervised {
            self.e_actor_w.fill(0.0);
            self.e_actor_b.fill(0.0);
            self.e_wv.fill(0.0);
            self.e_bv = 0.0;
            self.e_wz.fill(0.0);
            self.e_uz.fill(0.0);
            self.e_bz.fill(0.0);
            self.e_wr.fill(0.0);
            self.e_ur.fill(0.0);
            self.e_br.fill(0.0);
            self.e_wn.fill(0.0);
            self.e_un.fill(0.0);
            self.e_bn.fill(0.0);
        }
        self.ring_head = 0;
        self.ring_count = 0;
        self.has_pending = false;
        self.pending_r = 0.0;
        self.pending_v = 0.0;
    }

    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        self.gru.inject_register_bit_flip(reg_idx, bit);
    }

    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        self.gru.inject_weight_bit_flip(reg_idx, bit);
    }

    fn apply_step_td(&mut self, td_err: f32) {
        if self.gru.had_nan || self.is_supervised {
            return;
        }
        let td = td_err.clamp(-6.0, 6.0);
        let crit = td.clamp(-3.0, 3.0);
        let cent = td - self.td_mean;
        self.td_mean = 0.99 * self.td_mean + 0.01 * td;
        self.td_var = (0.98 * self.td_var + 0.02 * cent * cent).max(0.08);
        let adv = (cent / self.td_var.sqrt()).clamp(-2.5, 2.5);

        for i in 0..self.g_wo.len() {
            self.g_wo[i] = -adv * self.e_actor_w[i];
        }
        for i in 0..self.g_bo.len() {
            self.g_bo[i] = -adv * self.e_actor_b[i];
        }
        for i in 0..self.g_wv.len() {
            self.g_wv[i] = -crit * self.e_wv[i];
        }
        let g_bv = [-crit * self.e_bv];
        for i in 0..self.g_wz.len() {
            self.g_wz[i] = -adv * self.e_wz[i];
            self.g_wr[i] = -adv * self.e_wr[i];
            self.g_wn[i] = -adv * self.e_wn[i];
        }
        for i in 0..self.g_uz.len() {
            self.g_uz[i] = -adv * self.e_uz[i];
            self.g_ur[i] = -adv * self.e_ur[i];
            self.g_un[i] = -adv * self.e_un[i];
        }
        for i in 0..self.g_bz.len() {
            self.g_bz[i] = -adv * self.e_bz[i];
            self.g_br[i] = -adv * self.e_br[i];
            self.g_bn[i] = -adv * self.e_bn[i];
        }

        self.adam_t += 1;
        let t = self.adam_t;
        let lr = self.gru.lr;
        let head_lr = lr * 2.0;
        let crit_lr = lr * 3.0;
        self.gru.wo.step(&self.g_wo, head_lr, t);
        self.gru.bo.step(&self.g_bo, head_lr, t);
        self.gru.wv.step(&self.g_wv, crit_lr, t);
        self.gru.bv.step(&g_bv, crit_lr, t);
        self.gru.wz.step(&self.g_wz, lr, t);
        self.gru.uz.step(&self.g_uz, lr, t);
        self.gru.bz.step(&self.g_bz, lr, t);
        self.gru.wr.step(&self.g_wr, lr, t);
        self.gru.ur.step(&self.g_ur, lr, t);
        self.gru.br.step(&self.g_br, lr, t);
        self.gru.wn.step(&self.g_wn, lr, t);
        self.gru.un.step(&self.g_un, lr, t);
        self.gru.bn.step(&self.g_bn, lr, t);
    }

    #[inline]
    fn forward_core(&mut self) {
        let hd = self.gru.h_dim;
        let xd = self.gru.x_dim;
        self.hp_buf.copy_from_slice(&self.gru.h_cur);

        // AVX2+FMA без sanitize внутри dot (fast_dot_raw) -> полная скорость и честное
        // распространение NaN/Inf при выключенном SEU-guard в обоих режимах (RL и supervised).
        for i in 0..hd {
            let row_x = i * xd;
            let row_h = i * hd;
            let sz = self.gru.bz.w[i]
                + fast_dot_raw(&self.gru.wz.w[row_x..row_x + xd], &self.x_buf)
                + fast_dot_raw(&self.gru.uz.w[row_h..row_h + hd], &self.hp_buf);
            let sr = self.gru.br.w[i]
                + fast_dot_raw(&self.gru.wr.w[row_x..row_x + xd], &self.x_buf)
                + fast_dot_raw(&self.gru.ur.w[row_h..row_h + hd], &self.hp_buf);
            let zi = if sz.is_finite() { 1.0 / (1.0 + (-sz.clamp(-12.0, 12.0)).exp()) } else { sz };
            let ri = if sr.is_finite() { 1.0 / (1.0 + (-sr.clamp(-12.0, 12.0)).exp()) } else { sr };
            self.z_buf[i] = zi;
            self.r_buf[i] = ri;
            self.rhp_buf[i] = ri * self.hp_buf[i];
        }
        for i in 0..hd {
            let row_x = i * xd;
            let row_h = i * hd;
            let sn = self.gru.bn.w[i]
                + fast_dot_raw(&self.gru.wn.w[row_x..row_x + xd], &self.x_buf)
                + fast_dot_raw(&self.gru.un.w[row_h..row_h + hd], &self.rhp_buf);
            let ni = if sn.is_finite() { sn.tanh() } else { sn };
            self.n_buf[i] = ni;
            self.hn_buf[i] = (1.0 - self.z_buf[i]) * self.hp_buf[i] + self.z_buf[i] * ni;
        }
        if !self.hn_buf.iter().all(|v| v.is_finite()) {
            if !self.gru.had_nan {
                self.gru.seu_fault_count = self.gru.seu_fault_count.saturating_add(1);
            }
            self.gru.had_nan = true;
            self.gru.ever_had_nan = true;
        }
        self.gru.h_cur.copy_from_slice(&self.hn_buf);
    }

    pub fn decide_vec(
        &mut self,
        features: &[f32],
        train_mode: bool,
        seu_nan: bool,
        seu_spikes: bool,
    ) -> usize {
        if self.gru.enable_state_weight_guard {
            let _ = self.gru.scrub_state_and_weights();
        }
        let hd = self.gru.h_dim;
        let xd = self.gru.x_dim;
        let na = self.gru.num_actions;
        self.x_buf.fill(0.0);
        for i in 0..xd.min(features.len()) {
            self.x_buf[i] = features[i];
        }
        if seu_nan && !self.x_buf.is_empty() {
            self.x_buf[0] = f32::NAN;
        }
        if seu_spikes && self.x_buf.len() > 1 {
            self.x_buf[1] = 1e28;
        }
        if self.gru.enable_input_clamp {
            for v in &mut self.x_buf {
                *v = if v.is_finite() { v.clamp(-10.0, 10.0) } else { 0.0 };
            }
        }

        self.forward_core();

        let ring_cap = if xd > 0 { self.ring_x.len() / xd } else { 0 };
        let cap = if self.bptt_len > 1 && ring_cap > 0 {
            self.bptt_len.min(ring_cap)
        } else {
            0
        };
        if cap > 0 {
            let slot = self.ring_head;
            self.ring_x[slot * xd..(slot + 1) * xd].copy_from_slice(&self.x_buf);
            let h_off = slot * hd;
            self.ring_hp[h_off..h_off + hd].copy_from_slice(&self.hp_buf);
            self.ring_z[h_off..h_off + hd].copy_from_slice(&self.z_buf);
            self.ring_r[h_off..h_off + hd].copy_from_slice(&self.r_buf);
            self.ring_n[h_off..h_off + hd].copy_from_slice(&self.n_buf);
            self.ring_head = (slot + 1) % cap;
            if self.ring_count < cap {
                self.ring_count += 1;
            }
        }

        let v_cur = self.gru.bv.w[0] + fast_dot_raw(&self.gru.wv.w, &self.hn_buf);
        if !v_cur.is_finite() {
            if !self.gru.had_nan {
                self.gru.seu_fault_count = self.gru.seu_fault_count.saturating_add(1);
            }
            self.gru.had_nan = true;
            self.gru.ever_had_nan = true;
        }
        if train_mode && self.has_pending && !self.gru.had_nan {
            self.apply_step_td(self.pending_r + self.gru.gamma * v_cur - self.pending_v);
        }

        let mut max_l = f32::NEG_INFINITY;
        for a in 0..na {
            let off = a * hd;
            let s = self.gru.bo.w[a] + fast_dot_raw(&self.gru.wo.w[off..off + hd], &self.hn_buf);
            if !s.is_finite() {
                if !self.gru.had_nan {
                    self.gru.seu_fault_count = self.gru.seu_fault_count.saturating_add(1);
                }
                self.gru.had_nan = true;
                self.gru.ever_had_nan = true;
            }
            self.logits_buf[a] = s;
            if s > max_l {
                max_l = s;
            }
        }
        let mut sum_e = 0.0_f32;
        for a in 0..na {
            let e = (self.logits_buf[a] - max_l).exp();
            self.probs_buf[a] = e;
            sum_e += e;
        }
        if !sum_e.is_finite() || sum_e <= 0.0 {
            if !self.gru.had_nan {
                self.gru.seu_fault_count = self.gru.seu_fault_count.saturating_add(1);
            }
            self.gru.had_nan = true;
            self.gru.ever_had_nan = true;
            self.probs_buf.fill(1.0 / na as f32);
        } else {
            let inv_s = 1.0 / sum_e;
            for p in &mut self.probs_buf {
                *p *= inv_s;
            }
        }
        let act = if train_mode && !self.gru.had_nan {
            let u = rand_unit_f32(&mut self.gru.rng);
            let mut cdf = 0.0;
            let mut picked = na - 1;
            for (a, &p) in self.probs_buf.iter().enumerate() {
                cdf += p;
                if u <= cdf {
                    picked = a;
                    break;
                }
            }
            picked
        } else {
            self.probs_buf
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1))
                .map(|x| x.0)
                .unwrap_or(0)
        };

        if train_mode && !self.gru.had_nan && !self.is_supervised {
            let decay = self.gru.gamma * self.trace_lambda;
            for j in 0..hd {
                self.e_wv[j] = (decay * self.e_wv[j] + self.hn_buf[j]).clamp(-4.0, 4.0);
                self.dh_buf[j] = 0.15 * self.gru.wv.w[j];
            }
            self.e_bv = (decay * self.e_bv + 1.0).clamp(-5.0, 5.0);
            for a in 0..na {
                let sc = (if a == act { 1.0 } else { 0.0 }) - self.probs_buf[a];
                self.e_actor_b[a] = (decay * self.e_actor_b[a] + sc).clamp(-5.0, 5.0);
                let off = a * hd;
                for j in 0..hd {
                    self.e_actor_w[off + j] =
                        (decay * self.e_actor_w[off + j] + sc * self.hn_buf[j]).clamp(-4.0, 4.0);
                    self.dh_buf[j] += sc * self.gru.wo.w[off + j];
                }
            }
            if self.entropy_beta > 0.0 {
                let mut sum_plogp = 0.0_f32;
                for &p in &self.probs_buf {
                    let pc = p.max(1e-6);
                    sum_plogp += pc * pc.ln();
                }
                for a in 0..na {
                    let pc = self.probs_buf[a].max(1e-6);
                    let d_ent = self.entropy_beta * pc * (sum_plogp - pc.ln());
                    let step = self.gru.lr * 2.0 * d_ent;
                    self.gru.bo.w[a] = (self.gru.bo.w[a] + 0.25 * step).clamp(-3.0, 3.0);
                    let off = a * hd;
                    fast_saxpy(&mut self.gru.wo.w[off..off + hd], &self.hn_buf, step);
                }
            }

            // Затухание рекуррентных следов приемлемости (один раз за такт)
            for v in &mut self.e_bn { *v *= decay; }
            for v in &mut self.e_bz { *v *= decay; }
            for v in &mut self.e_br { *v *= decay; }
            for v in &mut self.e_wn { *v *= decay; }
            for v in &mut self.e_wz { *v *= decay; }
            for v in &mut self.e_wr { *v *= decay; }
            for v in &mut self.e_un { *v *= decay; }
            for v in &mut self.e_uz { *v *= decay; }
            for v in &mut self.e_ur { *v *= decay; }

            if cap > 1 && self.ring_count > 0 {
                // Истинный усечённый онлайн BPTT(1, T) сквозь рекуррентные вентили GRU в следы приемлемости
                let steps = self.ring_count;
                for back in 0..steps {
                    let slot = (self.ring_head + cap - 1 - back) % cap;
                    let rx = &self.ring_x[slot * xd..(slot + 1) * xd];
                    let h_off = slot * hd;
                    let rhp_s = &self.ring_hp[h_off..h_off + hd];
                    let rz = &self.ring_z[h_off..h_off + hd];
                    let rr = &self.ring_r[h_off..h_off + hd];
                    let rn = &self.ring_n[h_off..h_off + hd];

                    for j in 0..hd {
                        self.rhp_buf[j] = rr[j] * rhp_s[j];
                    }
                    self.dr_buf.fill(0.0);
                    self.dh_prev_buf.fill(0.0);
                    for i in 0..hd {
                        let dhi = self.dh_buf[i].clamp(-4.0, 4.0);
                        let dsn = dhi * rz[i] * (1.0 - rn[i] * rn[i]);
                        let dsz = dhi * (rn[i] - rhp_s[i]) * rz[i] * (1.0 - rz[i]);
                        self.dh_prev_buf[i] += dhi * (1.0 - rz[i]);
                        self.e_bn[i] += dsn;
                        self.e_bz[i] += dsz;
                        let row_x = i * xd;
                        let row_h = i * hd;
                        fast_saxpy(&mut self.e_wn[row_x..row_x + xd], rx, dsn);
                        fast_saxpy(&mut self.e_wz[row_x..row_x + xd], rx, dsz);
                        fast_saxpy(&mut self.e_un[row_h..row_h + hd], &self.rhp_buf, dsn);
                        fast_saxpy(&mut self.e_uz[row_h..row_h + hd], rhp_s, dsz);
                        fast_saxpy(&mut self.dr_buf, &self.gru.un.w[row_h..row_h + hd], dsn);
                        fast_saxpy(&mut self.dh_prev_buf, &self.gru.uz.w[row_h..row_h + hd], dsz);
                    }
                    for i in 0..hd {
                        self.dh_prev_buf[i] += self.dr_buf[i] * rr[i];
                        let dsr = self.dr_buf[i] * rhp_s[i] * rr[i] * (1.0 - rr[i]);
                        self.e_br[i] += dsr;
                        let row_x = i * xd;
                        let row_h = i * hd;
                        fast_saxpy(&mut self.e_wr[row_x..row_x + xd], rx, dsr);
                        fast_saxpy(&mut self.e_ur[row_h..row_h + hd], rhp_s, dsr);
                        fast_saxpy(&mut self.dh_prev_buf, &self.gru.ur.w[row_h..row_h + hd], dsr);
                    }
                    self.dh_buf.copy_from_slice(&self.dh_prev_buf);
                }
            } else {
                self.dr_buf.fill(0.0);
                for i in 0..hd {
                    let dhi = self.dh_buf[i].clamp(-4.0, 4.0);
                    let ni = self.n_buf[i];
                    let zi = self.z_buf[i];
                    let dsn = dhi * zi * (1.0 - ni * ni);
                    let dsz = dhi * (ni - self.hp_buf[i]) * zi * (1.0 - zi);
                    self.e_bn[i] += dsn;
                    self.e_bz[i] += dsz;
                    let row_x = i * xd;
                    let row_h = i * hd;
                    fast_saxpy(&mut self.e_wn[row_x..row_x + xd], &self.x_buf, dsn);
                    fast_saxpy(&mut self.e_wz[row_x..row_x + xd], &self.x_buf, dsz);
                    fast_saxpy(&mut self.e_un[row_h..row_h + hd], &self.rhp_buf, dsn);
                    fast_saxpy(&mut self.e_uz[row_h..row_h + hd], &self.hp_buf, dsz);
                    fast_saxpy(&mut self.dr_buf, &self.gru.un.w[row_h..row_h + hd], dsn);
                }
                for i in 0..hd {
                    let ri = self.r_buf[i];
                    let dsr = self.dr_buf[i] * self.hp_buf[i] * ri * (1.0 - ri);
                    self.e_br[i] += dsr;
                    let row_x = i * xd;
                    let row_h = i * hd;
                    fast_saxpy(&mut self.e_wr[row_x..row_x + xd], &self.x_buf, dsr);
                    fast_saxpy(&mut self.e_ur[row_h..row_h + hd], &self.hp_buf, dsr);
                }
            }
            for v in &mut self.e_bn { *v = v.clamp(-4.0, 4.0); }
            for v in &mut self.e_bz { *v = v.clamp(-4.0, 4.0); }
            for v in &mut self.e_br { *v = v.clamp(-4.0, 4.0); }
            for v in &mut self.e_wn { *v = v.clamp(-4.0, 4.0); }
            for v in &mut self.e_wz { *v = v.clamp(-4.0, 4.0); }
            for v in &mut self.e_wr { *v = v.clamp(-4.0, 4.0); }
            for v in &mut self.e_un { *v = v.clamp(-4.0, 4.0); }
            for v in &mut self.e_uz { *v = v.clamp(-4.0, 4.0); }
            for v in &mut self.e_ur { *v = v.clamp(-4.0, 4.0); }

            self.pending_v = self.gru.bv.w[0] + fast_dot_raw(&self.gru.wv.w, &self.hn_buf);
            self.pending_r = 0.0;
            self.has_pending = true;
        }
        act
    }

    pub fn observe_step_reward(&mut self, r: f32) {
        self.pending_r = r;
    }

    pub fn end_episode_truncated(&mut self, train_mode: bool, truncated: bool) {
        if train_mode && self.has_pending && !self.gru.had_nan {
            let v_next = if truncated { self.pending_v } else { 0.0 };
            self.apply_step_td(self.pending_r + self.gru.gamma * v_next - self.pending_v);
            self.has_pending = false;
        }
    }

    pub fn step_and_learn_supervised(&mut self, raw_x: &[f32], target: Option<usize>) -> usize {
        if self.gru.enable_state_weight_guard {
            let _ = self.gru.scrub_state_and_weights();
        }
        let hd = self.gru.h_dim;
        let xd = self.gru.x_dim;
        let na = self.gru.num_actions;

        self.x_buf.fill(0.0);
        for i in 0..xd.min(raw_x.len()) {
            let v = raw_x[i];
            self.x_buf[i] = if self.gru.enable_input_clamp {
                if v.is_finite() { v.clamp(-10.0, 10.0) } else { 0.0 }
            } else {
                v
            };
        }

        self.forward_core();

        let mut max_l = f32::NEG_INFINITY;
        let mut best = 0usize;
        for a in 0..na {
            let off = a * hd;
            let s = self.gru.bo.w[a] + fast_dot_raw(&self.gru.wo.w[off..off + hd], &self.hn_buf);
            if !s.is_finite() {
                if !self.gru.had_nan {
                    self.gru.seu_fault_count = self.gru.seu_fault_count.saturating_add(1);
                }
                self.gru.had_nan = true;
                self.gru.ever_had_nan = true;
            }
            self.logits_buf[a] = s;
            if s > max_l {
                max_l = s;
                best = a;
            }
        }
        let mut sum_e = 0.0_f32;
        for a in 0..na {
            let e = (self.logits_buf[a] - max_l).exp();
            self.probs_buf[a] = e;
            sum_e += e;
        }
        if !sum_e.is_finite() || sum_e <= 0.0 {
            if !self.gru.had_nan {
                self.gru.seu_fault_count = self.gru.seu_fault_count.saturating_add(1);
            }
            self.gru.had_nan = true;
            self.gru.ever_had_nan = true;
            self.probs_buf.fill(1.0 / na as f32);
        } else {
            let inv_s = 1.0 / sum_e;
            for p in &mut self.probs_buf {
                *p *= inv_s;
            }
        }

        let cap = if self.bptt_len > 1 { self.bptt_len } else { 0 };
        if cap > 0 {
            let slot = self.ring_head;
            self.ring_x[slot * xd..(slot + 1) * xd].copy_from_slice(&self.x_buf);
            let h_off = slot * hd;
            self.ring_hp[h_off..h_off + hd].copy_from_slice(&self.hp_buf);
            self.ring_z[h_off..h_off + hd].copy_from_slice(&self.z_buf);
            self.ring_r[h_off..h_off + hd].copy_from_slice(&self.r_buf);
            self.ring_n[h_off..h_off + hd].copy_from_slice(&self.n_buf);
            self.ring_head = (slot + 1) % cap;
            if self.ring_count < cap {
                self.ring_count += 1;
            }
        }

        if let Some(y) = target {
            if y < na && !self.gru.had_nan {
                self.adam_t += 1;
                let t = self.adam_t;
                let lr = self.gru.lr;

                if cap <= 1 || self.g_wz.is_empty() {
                    // Нулевые дополнительные матрицы градиентов + AVX2/автовекторизация (bptt_len = 1)
                    let (inv_bc1, inv_bc2) = AdamParam::bias_corr(t);
                    self.dh_buf.fill(0.0);
                    for a in 0..na {
                        let dl = self.probs_buf[a] - if a == y { 1.0 } else { 0.0 };
                        let off = a * hd;
                        fast_saxpy(&mut self.dh_buf, &self.gru.wo.w[off..off + hd], dl);
                        self.gru.bo.step_single(a, dl, lr, inv_bc1, inv_bc2);
                        self.gru.wo.step_row_scaled(a, hd, dl, &self.hn_buf, lr, inv_bc1, inv_bc2);
                    }
                    self.dr_buf.fill(0.0);
                    for i in 0..hd {
                        let dhi = self.dh_buf[i].clamp(-4.0, 4.0);
                        let ni = self.n_buf[i];
                        let zi = self.z_buf[i];
                        let dsn = dhi * zi * (1.0 - ni * ni);
                        let dsz = dhi * (ni - self.hp_buf[i]) * zi * (1.0 - zi);
                        let row_h = i * hd;
                        fast_saxpy(&mut self.dr_buf, &self.gru.un.w[row_h..row_h + hd], dsn);
                        self.gru.bn.step_single(i, dsn, lr, inv_bc1, inv_bc2);
                        self.gru.bz.step_single(i, dsz, lr, inv_bc1, inv_bc2);
                        self.gru.wn.step_row_scaled(i, xd, dsn, &self.x_buf, lr, inv_bc1, inv_bc2);
                        self.gru.wz.step_row_scaled(i, xd, dsz, &self.x_buf, lr, inv_bc1, inv_bc2);
                        self.gru.un.step_row_scaled(i, hd, dsn, &self.rhp_buf, lr, inv_bc1, inv_bc2);
                        self.gru.uz.step_row_scaled(i, hd, dsz, &self.hp_buf, lr, inv_bc1, inv_bc2);
                    }
                    for i in 0..hd {
                        let ri = self.r_buf[i];
                        let dsr = self.dr_buf[i] * self.hp_buf[i] * ri * (1.0 - ri);
                        self.gru.br.step_single(i, dsr, lr, inv_bc1, inv_bc2);
                        self.gru.wr.step_row_scaled(i, xd, dsr, &self.x_buf, lr, inv_bc1, inv_bc2);
                        self.gru.ur.step_row_scaled(i, hd, dsr, &self.hp_buf, lr, inv_bc1, inv_bc2);
                    }
                } else {
                    // Истинный усечённый онлайн BPTT(1, T) через предвыделенное кольцо без единой heap-аллокации
                    self.g_wo.fill(0.0);
                    self.g_bo.fill(0.0);
                    self.g_wz.fill(0.0);
                    self.g_uz.fill(0.0);
                    self.g_bz.fill(0.0);
                    self.g_wr.fill(0.0);
                    self.g_ur.fill(0.0);
                    self.g_br.fill(0.0);
                    self.g_wn.fill(0.0);
                    self.g_un.fill(0.0);
                    self.g_bn.fill(0.0);
                    self.dh_buf.fill(0.0);

                    for a in 0..na {
                        let dl = self.probs_buf[a] - if a == y { 1.0 } else { 0.0 };
                        self.g_bo[a] = dl;
                        let off = a * hd;
                        fast_saxpy(&mut self.g_wo[off..off + hd], &self.hn_buf, dl);
                        fast_saxpy(&mut self.dh_buf, &self.gru.wo.w[off..off + hd], dl);
                    }

                    let steps = self.ring_count;
                    for back in 0..steps {
                        let slot = (self.ring_head + cap - 1 - back) % cap;
                        let rx = &self.ring_x[slot * xd..(slot + 1) * xd];
                        let h_off = slot * hd;
                        let rhp_s = &self.ring_hp[h_off..h_off + hd];
                        let rz = &self.ring_z[h_off..h_off + hd];
                        let rr = &self.ring_r[h_off..h_off + hd];
                        let rn = &self.ring_n[h_off..h_off + hd];

                        for j in 0..hd {
                            self.rhp_buf[j] = rr[j] * rhp_s[j];
                        }
                        self.dr_buf.fill(0.0);
                        self.dh_prev_buf.fill(0.0);
                        for i in 0..hd {
                            let dhi = self.dh_buf[i].clamp(-4.0, 4.0);
                            let dsn = dhi * rz[i] * (1.0 - rn[i] * rn[i]);
                            let dsz = dhi * (rn[i] - rhp_s[i]) * rz[i] * (1.0 - rz[i]);
                            self.dh_prev_buf[i] += dhi * (1.0 - rz[i]);
                            self.g_bn[i] += dsn;
                            self.g_bz[i] += dsz;
                            let row_x = i * xd;
                            let row_h = i * hd;
                            fast_saxpy(&mut self.g_wn[row_x..row_x + xd], rx, dsn);
                            fast_saxpy(&mut self.g_wz[row_x..row_x + xd], rx, dsz);
                            fast_saxpy(&mut self.g_un[row_h..row_h + hd], &self.rhp_buf, dsn);
                            fast_saxpy(&mut self.g_uz[row_h..row_h + hd], rhp_s, dsz);
                            fast_saxpy(&mut self.dr_buf, &self.gru.un.w[row_h..row_h + hd], dsn);
                            fast_saxpy(&mut self.dh_prev_buf, &self.gru.uz.w[row_h..row_h + hd], dsz);
                        }
                        for i in 0..hd {
                            self.dh_prev_buf[i] += self.dr_buf[i] * rr[i];
                            let dsr = self.dr_buf[i] * rhp_s[i] * rr[i] * (1.0 - rr[i]);
                            self.g_br[i] += dsr;
                            let row_x = i * xd;
                            let row_h = i * hd;
                            fast_saxpy(&mut self.g_wr[row_x..row_x + xd], rx, dsr);
                            fast_saxpy(&mut self.g_ur[row_h..row_h + hd], rhp_s, dsr);
                            fast_saxpy(&mut self.dh_prev_buf, &self.gru.ur.w[row_h..row_h + hd], dsr);
                        }
                        self.dh_buf.copy_from_slice(&self.dh_prev_buf);
                    }

                    self.gru.wo.step(&self.g_wo, lr, t);
                    self.gru.bo.step(&self.g_bo, lr, t);
                    self.gru.wz.step(&self.g_wz, lr, t);
                    self.gru.uz.step(&self.g_uz, lr, t);
                    self.gru.bz.step(&self.g_bz, lr, t);
                    self.gru.wr.step(&self.g_wr, lr, t);
                    self.gru.ur.step(&self.g_ur, lr, t);
                    self.gru.br.step(&self.g_br, lr, t);
                    self.gru.wn.step(&self.g_wn, lr, t);
                    self.gru.un.step(&self.g_un, lr, t);
                    self.gru.bn.step(&self.g_bn, lr, t);
                }
            }
        }
        best
    }

    pub fn step_and_learn_regression(&mut self, raw_x: &[f32], target: Option<f32>) -> f32 {
        if self.gru.enable_state_weight_guard {
            let _ = self.gru.scrub_state_and_weights();
            self.gru.had_nan = false;
        }
        let hd = self.gru.h_dim;
        let xd = self.gru.x_dim;

        self.x_buf.fill(0.0);
        for i in 0..xd.min(raw_x.len()) {
            let v = raw_x[i];
            if (!v.is_finite() || v.abs() > 1e5) && self.gru.enable_state_weight_guard {
                self.gru.seu_fault_count = self.gru.seu_fault_count.saturating_add(1);
            }
            self.x_buf[i] = if self.gru.enable_input_clamp {
                if v.is_finite() && v.abs() <= 1e5 { v.clamp(-10.0, 10.0) } else { 0.0 }
            } else {
                v
            };
        }

        self.forward_core();

        let pred = self.gru.bo.w[0] + fast_dot_raw(&self.gru.wo.w[0..hd], &self.hn_buf);
        if !pred.is_finite() {
            if !self.gru.had_nan {
                self.gru.seu_fault_count = self.gru.seu_fault_count.saturating_add(1);
            }
            self.gru.had_nan = true;
            self.gru.ever_had_nan = true;
        }

        let cap = if self.bptt_len > 1 { self.bptt_len } else { 0 };
        if cap > 0 {
            let slot = self.ring_head;
            self.ring_x[slot * xd..(slot + 1) * xd].copy_from_slice(&self.x_buf);
            let h_off = slot * hd;
            self.ring_hp[h_off..h_off + hd].copy_from_slice(&self.hp_buf);
            self.ring_z[h_off..h_off + hd].copy_from_slice(&self.z_buf);
            self.ring_r[h_off..h_off + hd].copy_from_slice(&self.r_buf);
            self.ring_n[h_off..h_off + hd].copy_from_slice(&self.n_buf);
            self.ring_head = (slot + 1) % cap;
            if self.ring_count < cap {
                self.ring_count += 1;
            }
        }

        if let Some(y) = target {
            if !self.gru.had_nan && y.is_finite() {
                self.adam_t += 1;
                let t = self.adam_t;
                let lr = self.gru.lr;
                let dl = (pred - y).clamp(-2.5, 2.5);

                if cap <= 1 || self.g_wz.is_empty() {
                    let (inv_bc1, inv_bc2) = AdamParam::bias_corr(t);
                    self.dh_buf.fill(0.0);
                    fast_saxpy(&mut self.dh_buf, &self.gru.wo.w[0..hd], dl);
                    self.gru.bo.step_single(0, dl, lr, inv_bc1, inv_bc2);
                    self.gru.wo.step_row_scaled(0, hd, dl, &self.hn_buf, lr, inv_bc1, inv_bc2);
                    self.dr_buf.fill(0.0);
                    for i in 0..hd {
                        let dhi = self.dh_buf[i].clamp(-4.0, 4.0);
                        let ni = self.n_buf[i];
                        let zi = self.z_buf[i];
                        let dsn = dhi * zi * (1.0 - ni * ni);
                        let dsz = dhi * (ni - self.hp_buf[i]) * zi * (1.0 - zi);
                        let row_h = i * hd;
                        fast_saxpy(&mut self.dr_buf, &self.gru.un.w[row_h..row_h + hd], dsn);
                        self.gru.bn.step_single(i, dsn, lr, inv_bc1, inv_bc2);
                        self.gru.bz.step_single(i, dsz, lr, inv_bc1, inv_bc2);
                        self.gru.wn.step_row_scaled(i, xd, dsn, &self.x_buf, lr, inv_bc1, inv_bc2);
                        self.gru.wz.step_row_scaled(i, xd, dsz, &self.x_buf, lr, inv_bc1, inv_bc2);
                        self.gru.un.step_row_scaled(i, hd, dsn, &self.rhp_buf, lr, inv_bc1, inv_bc2);
                        self.gru.uz.step_row_scaled(i, hd, dsz, &self.hp_buf, lr, inv_bc1, inv_bc2);
                    }
                    for i in 0..hd {
                        let ri = self.r_buf[i];
                        let dsr = self.dr_buf[i] * self.hp_buf[i] * ri * (1.0 - ri);
                        self.gru.br.step_single(i, dsr, lr, inv_bc1, inv_bc2);
                        self.gru.wr.step_row_scaled(i, xd, dsr, &self.x_buf, lr, inv_bc1, inv_bc2);
                        self.gru.ur.step_row_scaled(i, hd, dsr, &self.hp_buf, lr, inv_bc1, inv_bc2);
                    }
                } else {
                    self.g_wo.fill(0.0);
                    self.g_bo.fill(0.0);
                    self.g_wz.fill(0.0);
                    self.g_uz.fill(0.0);
                    self.g_bz.fill(0.0);
                    self.g_wr.fill(0.0);
                    self.g_ur.fill(0.0);
                    self.g_br.fill(0.0);
                    self.g_wn.fill(0.0);
                    self.g_un.fill(0.0);
                    self.g_bn.fill(0.0);
                    self.dh_buf.fill(0.0);

                    self.g_bo[0] = dl;
                    fast_saxpy(&mut self.g_wo[0..hd], &self.hn_buf, dl);
                    fast_saxpy(&mut self.dh_buf, &self.gru.wo.w[0..hd], dl);

                    let steps = self.ring_count;
                    for back in 0..steps {
                        let slot = (self.ring_head + cap - 1 - back) % cap;
                        let rx = &self.ring_x[slot * xd..(slot + 1) * xd];
                        let h_off = slot * hd;
                        let rhp_s = &self.ring_hp[h_off..h_off + hd];
                        let rz = &self.ring_z[h_off..h_off + hd];
                        let rr = &self.ring_r[h_off..h_off + hd];
                        let rn = &self.ring_n[h_off..h_off + hd];

                        for j in 0..hd {
                            self.rhp_buf[j] = rr[j] * rhp_s[j];
                        }
                        self.dr_buf.fill(0.0);
                        self.dh_prev_buf.fill(0.0);
                        for i in 0..hd {
                            let dhi = self.dh_buf[i].clamp(-4.0, 4.0);
                            let dsn = dhi * rz[i] * (1.0 - rn[i] * rn[i]);
                            let dsz = dhi * (rn[i] - rhp_s[i]) * rz[i] * (1.0 - rz[i]);
                            self.dh_prev_buf[i] += dhi * (1.0 - rz[i]);
                            self.g_bn[i] += dsn;
                            self.g_bz[i] += dsz;
                            let row_x = i * xd;
                            let row_h = i * hd;
                            fast_saxpy(&mut self.g_wn[row_x..row_x + xd], rx, dsn);
                            fast_saxpy(&mut self.g_wz[row_x..row_x + xd], rx, dsz);
                            fast_saxpy(&mut self.g_un[row_h..row_h + hd], &self.rhp_buf, dsn);
                            fast_saxpy(&mut self.g_uz[row_h..row_h + hd], rhp_s, dsz);
                            fast_saxpy(&mut self.dr_buf, &self.gru.un.w[row_h..row_h + hd], dsn);
                            fast_saxpy(&mut self.dh_prev_buf, &self.gru.uz.w[row_h..row_h + hd], dsz);
                        }
                        for i in 0..hd {
                            self.dh_prev_buf[i] += self.dr_buf[i] * rr[i];
                            let dsr = self.dr_buf[i] * rhp_s[i] * rr[i] * (1.0 - rr[i]);
                            self.g_br[i] += dsr;
                            let row_x = i * xd;
                            let row_h = i * hd;
                            fast_saxpy(&mut self.g_wr[row_x..row_x + xd], rx, dsr);
                            fast_saxpy(&mut self.g_ur[row_h..row_h + hd], rhp_s, dsr);
                            fast_saxpy(&mut self.dh_prev_buf, &self.gru.ur.w[row_h..row_h + hd], dsr);
                        }
                        self.dh_buf.copy_from_slice(&self.dh_prev_buf);
                    }

                    self.gru.wo.step(&self.g_wo, lr, t);
                    self.gru.bo.step(&self.g_bo, lr, t);
                    self.gru.wz.step(&self.g_wz, lr, t);
                    self.gru.uz.step(&self.g_uz, lr, t);
                    self.gru.bz.step(&self.g_bz, lr, t);
                    self.gru.wr.step(&self.g_wr, lr, t);
                    self.gru.ur.step(&self.g_ur, lr, t);
                    self.gru.br.step(&self.g_br, lr, t);
                    self.gru.wn.step(&self.g_wn, lr, t);
                    self.gru.un.step(&self.g_un, lr, t);
                    self.gru.bn.step(&self.g_bn, lr, t);
                }
            }
        }
        pred
    }

    pub fn predict_k_ahead_readonly(&self, window: &[&[f32]]) -> usize {
        let hd = self.gru.h_dim;
        let xd = self.gru.x_dim;
        let na = self.gru.num_actions;
        let mut x_buf = self.x_buf.clone();
        let mut hn_buf = self.gru.h_cur.clone();
        let mut hp_buf = vec![0.0_f32; hd];
        let mut z_buf = vec![0.0_f32; hd];
        let mut r_buf = vec![0.0_f32; hd];
        let mut rhp_buf = vec![0.0_f32; hd];

        for &raw_x in window {
            x_buf.fill(0.0);
            for i in 0..xd.min(raw_x.len()) {
                let v = raw_x[i];
                x_buf[i] = if self.gru.enable_input_clamp {
                    if v.is_finite() { v.clamp(-10.0, 10.0) } else { 0.0 }
                } else {
                    v
                };
            }
            hp_buf.copy_from_slice(&hn_buf);
            for i in 0..hd {
                let row_x = i * xd;
                let row_h = i * hd;
                let sz = self.gru.bz.w[i]
                    + fast_dot_raw(&self.gru.wz.w[row_x..row_x + xd], &x_buf)
                    + fast_dot_raw(&self.gru.uz.w[row_h..row_h + hd], &hp_buf);
                let sr = self.gru.br.w[i]
                    + fast_dot_raw(&self.gru.wr.w[row_x..row_x + xd], &x_buf)
                    + fast_dot_raw(&self.gru.ur.w[row_h..row_h + hd], &hp_buf);
                let zi = if sz.is_finite() { 1.0 / (1.0 + (-sz.clamp(-12.0, 12.0)).exp()) } else { sz };
                let ri = if sr.is_finite() { 1.0 / (1.0 + (-sr.clamp(-12.0, 12.0)).exp()) } else { sr };
                z_buf[i] = zi;
                r_buf[i] = ri;
                rhp_buf[i] = ri * hp_buf[i];
            }
            for i in 0..hd {
                let row_x = i * xd;
                let row_h = i * hd;
                let sn = self.gru.bn.w[i]
                    + fast_dot_raw(&self.gru.wn.w[row_x..row_x + xd], &x_buf)
                    + fast_dot_raw(&self.gru.un.w[row_h..row_h + hd], &rhp_buf);
                let ni = if sn.is_finite() { sn.tanh() } else { sn };
                hn_buf[i] = (1.0 - z_buf[i]) * hp_buf[i] + z_buf[i] * ni;
            }
        }
        greedy_head_action(&self.gru.bo.w, &self.gru.wo.w, &hn_buf, na)
    }
}

#[derive(Clone)]
pub struct OnlineGruPerStepPhBaseline {
    pub banks: Vec<GruPerStepBank>,
    ctrl: PhRegimeCtrl,
    pub is_supervised: bool,
    pub bptt_len: usize,
    retired_had_nan: bool,
    retired_seu_faults: u64,
}

impl OnlineGruPerStepPhBaseline {
    pub fn new(
        x_dim: usize,
        h_dim: usize,
        num_actions: usize,
        max_regimes: usize,
        ph_delta: f32,
        ph_threshold: f32,
        ph_warmup: usize,
        seed: u64,
    ) -> Self {
        let n = max_regimes.clamp(2, 12);
        let banks = (0..n)
            .map(|b| GruPerStepBank::new(x_dim, h_dim, num_actions, seed ^ (0x6400_5000 + b as u64)))
            .collect();
        Self {
            banks,
            ctrl: PhRegimeCtrl::new(n, ph_delta, ph_threshold, ph_warmup),
            is_supervised: false,
            bptt_len: 4,
            retired_had_nan: false,
            retired_seu_faults: 0,
        }
    }

    pub fn new_supervised(
        x_dim: usize,
        h_dim: usize,
        num_actions: usize,
        max_regimes: usize,
        bptt_len: usize,
        ph_delta: f32,
        ph_threshold: f32,
        ph_warmup: usize,
        seed: u64,
    ) -> Self {
        let n = max_regimes.clamp(2, 12);
        let bl = bptt_len.max(1);
        let banks = (0..n)
            .map(|b| {
                GruPerStepBank::new_supervised(
                    x_dim,
                    h_dim,
                    num_actions,
                    bl,
                    seed ^ (0x6400_5000 + b as u64),
                )
            })
            .collect();
        Self {
            banks,
            ctrl: PhRegimeCtrl::new(n, ph_delta, ph_threshold, ph_warmup),
            is_supervised: true,
            bptt_len: bl,
            retired_had_nan: false,
            retired_seu_faults: 0,
        }
    }

    pub fn set_hyperparams(&mut self, gamma: f32, trace_lambda: f32, lr: f32) {
        for b in &mut self.banks {
            b.gru.gamma = gamma;
            b.trace_lambda = trace_lambda;
            b.gru.lr = lr;
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.banks.iter().map(|b| b.memory_footprint_bytes()).sum::<usize>() + self.ctrl.phs.len() * 48
    }

    pub fn had_nan(&self) -> bool {
        self.retired_had_nan || self.banks.iter().any(|b| b.gru.ever_had_nan)
    }

    pub fn set_probe_accept_threshold(&mut self, th: f32) {
        self.ctrl.probe_accept_threshold = th.clamp(0.0, 1.0);
    }

    pub fn set_state_weight_guard(&mut self, on: bool) {
        for b in &mut self.banks {
            b.gru.enable_state_weight_guard = on;
        }
    }

    pub fn total_seu_faults(&self) -> u64 {
        self.retired_seu_faults
            .saturating_add(self.banks.iter().map(|b| b.gru.seu_fault_count).sum::<u64>())
    }

    pub fn active_regime(&self) -> usize {
        self.ctrl.active_regime
    }

    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        self.banks[self.ctrl.active_regime].inject_register_bit_flip(reg_idx, bit);
    }

    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        self.banks[self.ctrl.active_regime].inject_weight_bit_flip(reg_idx, bit);
    }

    pub fn start_episode(&mut self) {
        self.banks[self.ctrl.active_regime].start_episode();
    }

    pub fn decide_vec(&mut self, features: &[f32], train_mode: bool, seu_nan: bool, seu_spikes: bool) -> usize {
        let eff = train_mode && !self.ctrl.is_frozen_for_train();
        self.banks[self.ctrl.active_regime].decide_vec(features, eff, seu_nan, seu_spikes)
    }

    pub fn observe_step_reward(&mut self, r: f32) {
        self.banks[self.ctrl.active_regime].observe_step_reward(r);
    }

    pub fn end_episode_with_score(&mut self, norm_ret: f32, truncated: bool, train_mode: bool) {
        let prev = self.ctrl.active_regime;
        let (do_td, reset_idx) = self.ctrl.end_episode_observe(norm_ret, train_mode);
        if do_td {
            self.banks[prev].end_episode_truncated(true, truncated);
        }
        if let Some(nr) = reset_idx {
            self.retired_had_nan |= self.banks[nr].gru.ever_had_nan;
            self.retired_seu_faults = self.retired_seu_faults.saturating_add(self.banks[nr].gru.seu_fault_count);
            let p = &self.banks[prev];
            let (xd, hd, na, ga, tl, lr, ic, swg) = (
                p.gru.x_dim,
                p.gru.h_dim,
                p.gru.num_actions,
                p.gru.gamma,
                p.trace_lambda,
                p.gru.lr,
                p.gru.enable_input_clamp,
                p.gru.enable_state_weight_guard,
            );
            let mut fresh = GruPerStepBank::new(xd, hd, na, 0x6400_6000 + self.ctrl.episode_counter as u64);
            fresh.gru.gamma = ga;
            fresh.trace_lambda = tl;
            fresh.gru.lr = lr;
            fresh.gru.enable_input_clamp = ic;
            fresh.gru.enable_state_weight_guard = swg;
            self.banks[nr] = fresh;
        }
    }

    pub fn step_supervised(&mut self, raw_x: &[f32], target: usize, train: bool, win_len: usize) -> usize {
        let prev = self.ctrl.active_regime;
        let allow_learn = train && !self.ctrl.is_frozen_for_train();
        let pred = self.banks[prev]
            .step_and_learn_supervised(raw_x, if allow_learn { Some(target) } else { None });
        if let Some(nr) = self.ctrl.observe_stream_step(pred == target, train, win_len) {
            self.retired_had_nan |= self.banks[nr].gru.ever_had_nan;
            self.retired_seu_faults = self.retired_seu_faults.saturating_add(self.banks[nr].gru.seu_fault_count);
            let p = &self.banks[prev];
            let (xd, hd, na, ga, tl, lr, ic, swg) = (
                p.gru.x_dim,
                p.gru.h_dim,
                p.gru.num_actions,
                p.gru.gamma,
                p.trace_lambda,
                p.gru.lr,
                p.gru.enable_input_clamp,
                p.gru.enable_state_weight_guard,
            );
            let mut fresh = if self.is_supervised {
                GruPerStepBank::new_supervised(
                    xd,
                    hd,
                    na,
                    self.bptt_len,
                    0x6400_7000 + self.ctrl.episode_counter as u64,
                )
            } else {
                GruPerStepBank::new(xd, hd, na, 0x6400_7000 + self.ctrl.episode_counter as u64)
            };
            fresh.gru.gamma = ga;
            fresh.trace_lambda = tl;
            fresh.gru.lr = lr;
            fresh.gru.enable_input_clamp = ic;
            fresh.gru.enable_state_weight_guard = swg;
            self.banks[nr] = fresh;
        }
        pred
    }

    pub fn predict_k_ahead_readonly(&self, window: &[&[f32]]) -> usize {
        self.banks[self.ctrl.active_regime].predict_k_ahead_readonly(window)
    }
}

#[cfg(test)]
mod gru_bptt_tests {
    use super::*;

    #[test]
    fn test_gru_per_step_bptt_exact_gradients() {
        let seq = [[0.7_f32, -0.3], [-0.4, 0.8], [0.2, 0.5], [-0.9, -0.2]];
        let target = 2usize;

        let loss_of = |mut bank: GruPerStepBank| -> f32 {
            for x in &seq {
                let _ = bank.step_and_learn_supervised(x, None);
            }
            -bank.probs_buf[target].max(1e-12).ln()
        };

        let mut g_bank = GruPerStepBank::new_supervised(2, 4, 3, 4, 31415);
        g_bank.gru.lr = 0.0; // compute gradients into g_* without changing weights
        for (idx, x) in seq.iter().enumerate() {
            let tgt = if idx + 1 == seq.len() { Some(target) } else { None };
            let _ = g_bank.step_and_learn_supervised(x, tgt);
        }

        let eps = 1e-3_f32;
        for i in 0..4usize {
            // Check wz, uz, wr, ur, wn, un
            for j in 0..2usize {
                let idx = i * 2 + j;
                let mut bp = GruPerStepBank::new_supervised(2, 4, 3, 4, 31415);
                let mut bm = GruPerStepBank::new_supervised(2, 4, 3, 4, 31415);
                bp.gru.wz.w[idx] += eps;
                bm.gru.wz.w[idx] -= eps;
                let fd = (loss_of(bp) - loss_of(bm)) / (2.0 * eps);
                assert!((fd - g_bank.g_wz[idx]).abs() < 2e-3, "wz[{idx}] fd={fd} ex={}", g_bank.g_wz[idx]);

                let mut bp = GruPerStepBank::new_supervised(2, 4, 3, 4, 31415);
                let mut bm = GruPerStepBank::new_supervised(2, 4, 3, 4, 31415);
                bp.gru.wn.w[idx] += eps;
                bm.gru.wn.w[idx] -= eps;
                let fd = (loss_of(bp) - loss_of(bm)) / (2.0 * eps);
                assert!((fd - g_bank.g_wn[idx]).abs() < 2e-3, "wn[{idx}] fd={fd} ex={}", g_bank.g_wn[idx]);

                let mut bp = GruPerStepBank::new_supervised(2, 4, 3, 4, 31415);
                let mut bm = GruPerStepBank::new_supervised(2, 4, 3, 4, 31415);
                bp.gru.wr.w[idx] += eps;
                bm.gru.wr.w[idx] -= eps;
                let fd = (loss_of(bp) - loss_of(bm)) / (2.0 * eps);
                assert!((fd - g_bank.g_wr[idx]).abs() < 2e-3, "wr[{idx}] fd={fd} ex={}", g_bank.g_wr[idx]);
            }
            for j in 0..4usize {
                let idx = i * 4 + j;
                let mut bp = GruPerStepBank::new_supervised(2, 4, 3, 4, 31415);
                let mut bm = GruPerStepBank::new_supervised(2, 4, 3, 4, 31415);
                bp.gru.uz.w[idx] += eps;
                bm.gru.uz.w[idx] -= eps;
                let fd = (loss_of(bp) - loss_of(bm)) / (2.0 * eps);
                assert!((fd - g_bank.g_uz[idx]).abs() < 2e-3, "uz[{idx}] fd={fd} ex={}", g_bank.g_uz[idx]);

                let mut bp = GruPerStepBank::new_supervised(2, 4, 3, 4, 31415);
                let mut bm = GruPerStepBank::new_supervised(2, 4, 3, 4, 31415);
                bp.gru.un.w[idx] += eps;
                bm.gru.un.w[idx] -= eps;
                let fd = (loss_of(bp) - loss_of(bm)) / (2.0 * eps);
                assert!((fd - g_bank.g_un[idx]).abs() < 2e-3, "un[{idx}] fd={fd} ex={}", g_bank.g_un[idx]);

                let mut bp = GruPerStepBank::new_supervised(2, 4, 3, 4, 31415);
                let mut bm = GruPerStepBank::new_supervised(2, 4, 3, 4, 31415);
                bp.gru.ur.w[idx] += eps;
                bm.gru.ur.w[idx] -= eps;
                let fd = (loss_of(bp) - loss_of(bm)) / (2.0 * eps);
                assert!((fd - g_bank.g_ur[idx]).abs() < 2e-3, "ur[{idx}] fd={fd} ex={}", g_bank.g_ur[idx]);
            }
        }
    }

    #[test]
    fn test_gru_per_step_bptt_regression_exact_gradients() {
        let seq = [[0.6_f32, -0.4], [-0.3, 0.8], [0.5, 0.2], [-0.7, -0.1]];
        let target = 0.72_f32;
        let loss_of = |mut bank: GruPerStepBank| -> f32 {
            bank.start_episode();
            let mut p = 0.0_f32;
            for x in &seq {
                p = bank.step_and_learn_regression(x, None);
            }
            0.5 * (p - target) * (p - target)
        };
        let mut g_bank = GruPerStepBank::new_supervised(2, 4, 1, 4, 27182);
        g_bank.gru.lr = 0.0;
        for (idx, x) in seq.iter().enumerate() {
            let tgt = if idx + 1 == seq.len() { Some(target) } else { None };
            let _ = g_bank.step_and_learn_regression(x, tgt);
        }
        let eps = 1e-3_f32;
        for i in 0..4usize {
            let mut bp = GruPerStepBank::new_supervised(2, 4, 1, 4, 27182);
            let mut bm = GruPerStepBank::new_supervised(2, 4, 1, 4, 27182);
            bp.gru.wo.w[i] += eps;
            bm.gru.wo.w[i] -= eps;
            let fd = (loss_of(bp) - loss_of(bm)) / (2.0 * eps);
            assert!((fd - g_bank.g_wo[i]).abs() < 2e-3, "reg wo[{i}] fd={fd} ex={}", g_bank.g_wo[i]);

            for j in 0..2usize {
                let idx = i * 2 + j;
                let mut bp = GruPerStepBank::new_supervised(2, 4, 1, 4, 27182);
                let mut bm = GruPerStepBank::new_supervised(2, 4, 1, 4, 27182);
                bp.gru.wz.w[idx] += eps;
                bm.gru.wz.w[idx] -= eps;
                let fd = (loss_of(bp) - loss_of(bm)) / (2.0 * eps);
                assert!((fd - g_bank.g_wz[idx]).abs() < 2e-3, "reg wz[{idx}] fd={fd} ex={}", g_bank.g_wz[idx]);

                let mut bp = GruPerStepBank::new_supervised(2, 4, 1, 4, 27182);
                let mut bm = GruPerStepBank::new_supervised(2, 4, 1, 4, 27182);
                bp.gru.wn.w[idx] += eps;
                bm.gru.wn.w[idx] -= eps;
                let fd = (loss_of(bp) - loss_of(bm)) / (2.0 * eps);
                assert!((fd - g_bank.g_wn[idx]).abs() < 2e-3, "reg wn[{idx}] fd={fd} ex={}", g_bank.g_wn[idx]);
            }
            for j in 0..4usize {
                let idx = i * 4 + j;
                let mut bp = GruPerStepBank::new_supervised(2, 4, 1, 4, 27182);
                let mut bm = GruPerStepBank::new_supervised(2, 4, 1, 4, 27182);
                bp.gru.un.w[idx] += eps;
                bm.gru.un.w[idx] -= eps;
                let fd = (loss_of(bp) - loss_of(bm)) / (2.0 * eps);
                assert!((fd - g_bank.g_un[idx]).abs() < 2e-3, "reg un[{idx}] fd={fd} ex={}", g_bank.g_un[idx]);
            }
        }
    }
}
