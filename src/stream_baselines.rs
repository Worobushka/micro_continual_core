// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Честные потоковые бейзлайны на чистом Rust (1-pass Test-then-Train):
//! 1. OnlineMlpAdam (2-слойный MLP H=32 + ReLU + Softmax Cross-Entropy + Adam, классический backprop)
//! 2. OnlineGaussianNaiveBayes (с экспоненциальным забыванием для дрейфа, стандарт MOA/River)
//! 3. TabularBinnedPH (дискретизация признаков + HashMap + Page-Hinkley)

use std::collections::HashMap;

use crate::belief::PageHinkleyDetector;
use crate::math::{fnv1a_bytes, rand_signed_f32, splitmix64};

// -----------------------------------------------------------------------------
// 1. Online MLP (H=32) + Adam (Backprop)
// -----------------------------------------------------------------------------

#[derive(Clone)]
struct AdamBuf {
    w: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl AdamBuf {
    fn new(len: usize, scale: f32, rng: &mut u64) -> Self {
        let mut w = vec![0.0_f32; len];
        for c in &mut w {
            *c = rand_signed_f32(rng) * scale;
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
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        for i in 0..self.w.len() {
            let g = grad[i].clamp(-2.0, 2.0);
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let mh = self.m[i] / bc1;
            let vh = self.v[i] / bc2;
            self.w[i] -= lr * mh / (vh.sqrt() + eps);
        }
    }
}

#[derive(Clone)]
pub struct OnlineMlpAdam {
    pub in_dim: usize,
    pub hidden: usize,
    pub num_classes: usize,
    w1: AdamBuf,
    b1: AdamBuf,
    w2: AdamBuf,
    b2: AdamBuf,
    feat_mean: Vec<f32>,
    feat_var: Vec<f32>,
    feat_count: u64,
    // Pre-allocated zero-alloc step scratch buffers
    xn_buf: Vec<f32>,
    h_buf: Vec<f32>,
    logits_buf: Vec<f32>,
    probs_buf: Vec<f32>,
    gw2_buf: Vec<f32>,
    gb2_buf: Vec<f32>,
    dh_buf: Vec<f32>,
    gw1_buf: Vec<f32>,
    gb1_buf: Vec<f32>,
    pub standardize: bool,
    pub lr: f32,
    t: usize,
    pub enable_input_clamp: bool,
    pub enable_state_weight_guard: bool,
    pub had_nan: bool,
    pub ever_had_nan: bool,
    pub seu_fault_count: u64,
}

impl OnlineMlpAdam {
    pub fn new(in_dim: usize, hidden: usize, num_classes: usize, seed: u64) -> Self {
        let mut rng = splitmix64(seed ^ 0xA1B2_C3D4);
        let s1 = (2.0 / (in_dim.max(1) as f32)).sqrt() * 0.5;
        let s2 = (2.0 / (hidden.max(1) as f32)).sqrt() * 0.5;
        Self {
            in_dim,
            hidden,
            num_classes,
            w1: AdamBuf::new(hidden * in_dim, s1, &mut rng),
            b1: AdamBuf::new(hidden, 0.0, &mut rng),
            w2: AdamBuf::new(num_classes * hidden, s2, &mut rng),
            b2: AdamBuf::new(num_classes, 0.0, &mut rng),
            feat_mean: vec![0.0_f32; in_dim],
            feat_var: vec![1.0_f32; in_dim],
            feat_count: 0,
            xn_buf: vec![0.0_f32; in_dim],
            h_buf: vec![0.0_f32; hidden],
            logits_buf: vec![0.0_f32; num_classes],
            probs_buf: vec![0.0_f32; num_classes],
            gw2_buf: vec![0.0_f32; num_classes * hidden],
            gb2_buf: vec![0.0_f32; num_classes],
            dh_buf: vec![0.0_f32; hidden],
            gw1_buf: vec![0.0_f32; hidden * in_dim],
            gb1_buf: vec![0.0_f32; hidden],
            standardize: true,
            lr: 0.006,
            t: 0,
            enable_input_clamp: false,
            enable_state_weight_guard: false,
            had_nan: false,
            ever_had_nan: false,
            seu_fault_count: 0,
        }
    }

    pub fn set_state_weight_guard(&mut self, on: bool) {
        self.enable_state_weight_guard = on;
    }

    pub fn total_seu_faults(&self) -> u64 {
        self.seu_fault_count
    }

    pub fn scrub_state_and_weights(&mut self) -> u64 {
        let mut f = 0u64;
        for buf in [&mut self.w1, &mut self.b1, &mut self.w2, &mut self.b2] {
            for v in buf.w.iter_mut() {
                if !v.is_finite() || v.abs() > 16.1 {
                    *v = 0.0;
                    f = f.saturating_add(1);
                }
            }
            for v in buf.m.iter_mut() {
                if !v.is_finite() || v.abs() > 100.0 {
                    *v = 0.0;
                }
            }
            for v in buf.v.iter_mut() {
                if !v.is_finite() || *v < 0.0 || *v > 1e6 {
                    *v = 0.0;
                }
            }
        }
        self.seu_fault_count = self.seu_fault_count.saturating_add(f);
        self.had_nan = false;
        f
    }

    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let mut slices: [&mut [f32]; 4] = [
            &mut self.w1.w,
            &mut self.b1.w,
            &mut self.w2.w,
            &mut self.b2.w,
        ];
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

    pub fn step_and_learn_regression(&mut self, x: &[f32], target: Option<f32>) -> f32 {
        if self.enable_state_weight_guard {
            let _ = self.scrub_state_and_weights();
        }
        self.xn_buf.fill(0.0);
        let n = self.in_dim.min(x.len());
        for j in 0..n {
            let raw = x[j];
            if !raw.is_finite() || raw.abs() > 1e5 {
                if self.enable_state_weight_guard {
                    self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                } else if !self.enable_input_clamp {
                    self.had_nan = true;
                    self.ever_had_nan = true;
                }
            }
            self.xn_buf[j] = if self.enable_input_clamp {
                if raw.is_finite() && raw.abs() <= 1e5 { raw.clamp(-10.0, 10.0) } else { 0.0 }
            } else {
                raw
            };
        }
        for i in 0..self.hidden {
            let mut s = self.b1.w[i];
            let off = i * self.in_dim;
            for j in 0..self.in_dim {
                s += self.w1.w[off + j] * self.xn_buf[j];
            }
            if !s.is_finite() {
                if !self.had_nan {
                    self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                }
                self.had_nan = true;
                self.ever_had_nan = true;
            }
            self.h_buf[i] = if s.is_finite() { s.max(0.0) } else { f32::NAN };
        }
        let mut pred = self.b2.w[0];
        for j in 0..self.hidden {
            pred += self.w2.w[j] * self.h_buf[j];
        }
        if !pred.is_finite() {
            if !self.had_nan {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            self.had_nan = true;
            self.ever_had_nan = true;
        }
        if let Some(y) = target {
            if !self.had_nan && y.is_finite() {
                let dl = (pred - y).clamp(-2.5, 2.5);
                self.gb2_buf[0] = dl;
                for j in 0..self.hidden {
                    self.gw2_buf[j] = dl * self.h_buf[j];
                    self.dh_buf[j] = self.w2.w[j] * dl;
                }
                for i in 0..self.hidden {
                    let dz = if self.h_buf[i] > 0.0 { self.dh_buf[i] } else { 0.0 };
                    self.gb1_buf[i] = dz;
                    let off = i * self.in_dim;
                    for j in 0..self.in_dim {
                        self.gw1_buf[off + j] = dz * self.xn_buf[j];
                    }
                }
                self.t += 1;
                let lr = self.lr;
                let st = self.t;
                self.w1.step(&self.gw1_buf, lr, st);
                self.b1.step(&self.gb1_buf, lr, st);
                self.w2.step(&self.gw2_buf, lr, st);
                self.b2.step(&self.gb2_buf, lr, st);
            }
        }
        pred
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.w1.bytes()
            + self.b1.bytes()
            + self.w2.bytes()
            + self.b2.bytes()
            + (self.feat_mean.len()
                + self.feat_var.len()
                + self.xn_buf.len()
                + self.h_buf.len()
                + self.logits_buf.len()
                + self.probs_buf.len()
                + self.gw2_buf.len()
                + self.gb2_buf.len()
                + self.dh_buf.len()
                + self.gw1_buf.len()
                + self.gb1_buf.len())
                * 4
    }

    fn normalize_input_into(&mut self, x: &[f32], update_stats: bool) {
        self.xn_buf.fill(0.0);
        let n = self.in_dim.min(x.len());
        if !self.standardize {
            for j in 0..n {
                self.xn_buf[j] = x[j];
            }
            return;
        }
        if update_stats {
            self.feat_count = self.feat_count.saturating_add(1);
            let alpha = if self.feat_count < 40 {
                (1.0 / (self.feat_count as f32)).clamp(0.025, 0.5)
            } else {
                0.008_f32
            };
            let keep = 1.0 - alpha;
            for j in 0..n {
                let raw = x[j];
                if !raw.is_finite() {
                    self.had_nan = true;
                }
                let mu = self.feat_mean[j];
                let diff = raw - mu;
                let new_mu = mu + alpha * diff;
                let new_var = (keep * self.feat_var[j] + alpha * diff * (raw - new_mu)).max(1e-3);
                self.feat_mean[j] = new_mu;
                self.feat_var[j] = new_var;
                let std = new_var.sqrt().max(0.05);
                self.xn_buf[j] = if self.feat_count <= 2 {
                    raw.clamp(-4.0, 4.0)
                } else {
                    ((raw - new_mu) / std).clamp(-4.0, 4.0)
                };
            }
        } else {
            for j in 0..n {
                let raw = x[j];
                if !raw.is_finite() {
                    self.had_nan = true;
                }
                let std = self.feat_var[j].sqrt().max(0.05);
                self.xn_buf[j] = if self.feat_count <= 2 {
                    raw
                } else {
                    (raw - self.feat_mean[j]) / std
                };
            }
        }
    }

    fn forward_into(&mut self) {
        for i in 0..self.hidden {
            let mut s = self.b1.w[i];
            let off = i * self.in_dim;
            for j in 0..self.in_dim {
                s += self.w1.w[off + j] * self.xn_buf[j];
            }
            if !s.is_finite() {
                self.had_nan = true;
            }
            self.h_buf[i] = s.max(0.0);
        }
        let mut max_l = f32::NEG_INFINITY;
        for c in 0..self.num_classes {
            let mut s = self.b2.w[c];
            let off = c * self.hidden;
            for j in 0..self.hidden {
                s += self.w2.w[off + j] * self.h_buf[j];
            }
            if !s.is_finite() {
                self.had_nan = true;
            }
            self.logits_buf[c] = s;
            if s > max_l {
                max_l = s;
            }
        }
        let mut sum = 0.0_f32;
        for c in 0..self.num_classes {
            let e = (self.logits_buf[c] - max_l).exp();
            self.probs_buf[c] = e;
            sum += e;
        }
        if !sum.is_finite() || sum <= 0.0 {
            self.had_nan = true;
            self.probs_buf.fill(1.0 / (self.num_classes as f32));
        } else {
            let inv = 1.0 / sum;
            for p in &mut self.probs_buf {
                *p *= inv;
            }
        }
    }

    pub fn predict_and_train(&mut self, x: &[f32], y: usize, train: bool) -> usize {
        self.normalize_input_into(x, train);
        self.forward_into();
        let mut best = 0usize;
        let mut best_p = -1.0_f32;
        for (c, &p) in self.probs_buf.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best = c;
            }
        }
        if !train || self.had_nan {
            return best;
        }

        self.dh_buf.fill(0.0);
        for c in 0..self.num_classes {
            let dl = self.probs_buf[c] - if c == y { 1.0 } else { 0.0 };
            self.gb2_buf[c] = dl;
            let off = c * self.hidden;
            for j in 0..self.hidden {
                self.gw2_buf[off + j] = dl * self.h_buf[j];
                self.dh_buf[j] += self.w2.w[off + j] * dl;
            }
        }
        for i in 0..self.hidden {
            let dz = if self.h_buf[i] > 0.0 { self.dh_buf[i] } else { 0.0 };
            self.gb1_buf[i] = dz;
            let off = i * self.in_dim;
            for j in 0..self.in_dim {
                self.gw1_buf[off + j] = dz * self.xn_buf[j];
            }
        }
        self.t += 1;
        let lr = self.lr;
        let st = self.t;
        self.w1.step(&self.gw1_buf, lr, st);
        self.b1.step(&self.gb1_buf, lr, st);
        self.w2.step(&self.gw2_buf, lr, st);
        self.b2.step(&self.gb2_buf, lr, st);
        best
    }
}

// -----------------------------------------------------------------------------
// 2. Online Gaussian Naive Bayes (с экспоненциальным забыванием, стандарт MOA/River)
// -----------------------------------------------------------------------------

pub struct OnlineGaussianNaiveBayes {
    pub in_dim: usize,
    pub num_classes: usize,
    pub decay: f32,
    class_w: Vec<f32>,
    means: Vec<f32>,
    vars: Vec<f32>,
    pub had_nan: bool,
}

impl OnlineGaussianNaiveBayes {
    pub fn new(in_dim: usize, num_classes: usize, decay: f32) -> Self {
        Self {
            in_dim,
            num_classes,
            decay,
            class_w: vec![1.0_f32; num_classes],
            means: vec![0.0_f32; num_classes * in_dim],
            vars: vec![1.0_f32; num_classes * in_dim],
            had_nan: false,
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        (self.class_w.len() + self.means.len() + self.vars.len()) * 4
    }

    pub fn predict_and_train(&mut self, x: &[f32], y: usize, train: bool) -> usize {
        let total_w: f32 = self.class_w.iter().sum::<f32>().max(1e-6);
        let mut best_c = 0usize;
        let mut best_ll = f32::NEG_INFINITY;
        for c in 0..self.num_classes {
            let mut ll = (self.class_w[c] / total_w).max(1e-6).ln();
            for j in 0..self.in_dim.min(x.len()) {
                let mu = self.means[c * self.in_dim + j];
                let var = self.vars[c * self.in_dim + j].max(1e-3);
                let diff = x[j] - mu;
                ll += -0.5 * (var.ln() + (diff * diff) / var);
            }
            if !ll.is_finite() {
                self.had_nan = true;
            }
            if ll > best_ll {
                best_ll = ll;
                best_c = c;
            }
        }
        if train && y < self.num_classes {
            let alpha = 1.0 - self.decay;
            for c in 0..self.num_classes {
                self.class_w[c] *= self.decay;
            }
            self.class_w[y] += 1.0;
            for j in 0..self.in_dim.min(x.len()) {
                let idx = y * self.in_dim + j;
                let mu = self.means[idx];
                let diff = x[j] - mu;
                let new_mu = mu + alpha * diff;
                let new_var = self.decay * self.vars[idx] + alpha * diff * (x[j] - new_mu);
                self.means[idx] = new_mu;
                self.vars[idx] = new_var.max(1e-3);
            }
        }
        best_c
    }
}

// -----------------------------------------------------------------------------
// 3. Tabular Binned + Page-Hinkley (классическая дискретизация в HashMap)
// -----------------------------------------------------------------------------

pub struct TabularBinnedPH {
    pub num_classes: usize,
    pub banks: Vec<HashMap<u64, Vec<f32>>>,
    pub phs: Vec<PageHinkleyDetector>,
    pub active: usize,
}

impl TabularBinnedPH {
    pub fn new(num_classes: usize, max_banks: usize) -> Self {
        let mut banks = Vec::with_capacity(max_banks);
        let mut phs = Vec::with_capacity(max_banks);
        for _ in 0..max_banks {
            banks.push(HashMap::new());
            phs.push(PageHinkleyDetector::new(0.10, 3.0, 40));
        }
        Self {
            num_classes,
            banks,
            phs,
            active: 0,
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.banks
            .iter()
            .map(|m| m.len() * (16 + self.num_classes * 4))
            .sum::<usize>()
            + self.phs.len() * 40
    }

    fn bin_key(x: &[f32]) -> u64 {
        let mut buf = Vec::with_capacity(x.len());
        for &v in x {
            let b = ((v * 2.0).round().clamp(-120.0, 120.0) as i8) as u8;
            buf.push(b);
        }
        fnv1a_bytes(&buf, 0xBEEF_1234)
    }

    pub fn predict_and_train(&mut self, x: &[f32], y: usize, train: bool) -> usize {
        let k = Self::bin_key(x);
        let counts = self.banks[self.active]
            .entry(k)
            .or_insert_with(|| vec![0.0_f32; self.num_classes]);
        let mut best = 0usize;
        let mut best_v = -1e9_f32;
        for (c, &v) in counts.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = c;
            }
        }
        if train && y < self.num_classes {
            counts[y] += 1.0;
            let ok = if best == y { 1.0 } else { 0.0 };
            if self.phs[self.active].observe(ok) {
                self.phs[self.active].reset_cusum();
                self.active = (self.active + 1) % self.banks.len();
            }
        }
        best
    }
}

// -----------------------------------------------------------------------------
// 4. No-Change (Persistent) Baseline (стандарт Bifet et al. 2013 для Kappa_temp)
// -----------------------------------------------------------------------------

pub struct NoChangePersistent {
    last_y: usize,
}

impl NoChangePersistent {
    pub fn new() -> Self {
        Self { last_y: 0 }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        std::mem::size_of::<usize>()
    }

    pub fn predict_and_train(&mut self, _x: &[f32], y: usize, train: bool) -> usize {
        let pred = self.last_y;
        if train {
            self.last_y = y;
        }
        pred
    }
}

// -----------------------------------------------------------------------------
// 5. Online LRU-RTRL Stream (рекуррентный комплексный диагональный LRU + RTRL)
// -----------------------------------------------------------------------------

pub struct OnlineLruStream {
    pub in_dim: usize,
    pub units: usize,
    pub num_classes: usize,
    pub kappa: f32,
    pub nu: Vec<f32>,
    pub theta: Vec<f32>,
    pub b_re: Vec<f32>,
    pub b_im: Vec<f32>,
    v_nu: Vec<f32>,
    v_theta: Vec<f32>,
    v_bre: Vec<f32>,
    v_bim: Vec<f32>,
    w: AdamBuf,
    b: AdamBuf,
    h_re: Vec<f32>,
    h_im: Vec<f32>,
    y_re: Vec<f32>,
    y_im: Vec<f32>,
    rho: Vec<f32>,
    p_nu_re: Vec<f32>,
    p_nu_im: Vec<f32>,
    p_th_re: Vec<f32>,
    p_th_im: Vec<f32>,
    p_b_re: Vec<f32>,
    p_b_im: Vec<f32>,
    feat_mean: Vec<f32>,
    feat_var: Vec<f32>,
    feat_count: u64,
    // Pre-allocated zero-alloc step scratch buffers
    xn_buf: Vec<f32>,
    feat_buf: Vec<f32>,
    logits_buf: Vec<f32>,
    probs_buf: Vec<f32>,
    gw_buf: Vec<f32>,
    gb_buf: Vec<f32>,
    fb_re_buf: Vec<f32>,
    fb_im_buf: Vec<f32>,
    pub standardize: bool,
    pub lr: f32,
    pub rtrl_lr: f32,
    t: usize,
    pub had_nan: bool,
    pub enable_input_clamp: bool,
}

impl OnlineLruStream {
    pub fn new(in_dim: usize, units: usize, num_classes: usize, seed: u64) -> Self {
        let mut rng = splitmix64(seed ^ 0x1205_72EA_AA41);
        let f_dim = 2 * units + in_dim;
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
            let r_u = crate::math::rand_unit_f32(&mut rng);
            let r_init = (r_min2 + r_u * (r_max2 - r_min2)).sqrt();
            nu[j] = (-r_init.ln()).ln();
            let th_u = crate::math::rand_unit_f32(&mut rng);
            theta[j] = (th_u * 2.0 - 1.0) * std::f32::consts::PI;
            for k in 0..in_dim {
                b_re[j * in_dim + k] = rand_signed_f32(&mut rng) * scale_b;
                b_im[j * in_dim + k] = rand_signed_f32(&mut rng) * scale_b;
            }
        }

        Self {
            in_dim,
            units,
            num_classes,
            kappa: 0.65,
            nu,
            theta,
            b_re,
            b_im,
            v_nu: vec![0.04; units],
            v_theta: vec![0.04; units],
            v_bre: vec![0.04; units * in_dim],
            v_bim: vec![0.04; units * in_dim],
            w: AdamBuf::new(num_classes * f_dim, 0.05, &mut rng),
            b: AdamBuf::new(num_classes, 0.0, &mut rng),
            h_re: vec![0.0; units],
            h_im: vec![0.0; units],
            y_re: vec![0.0; units],
            y_im: vec![0.0; units],
            rho: vec![0.65; units],
            p_nu_re: vec![0.0; units],
            p_nu_im: vec![0.0; units],
            p_th_re: vec![0.0; units],
            p_th_im: vec![0.0; units],
            p_b_re: vec![0.0; units * in_dim],
            p_b_im: vec![0.0; units * in_dim],
            feat_mean: vec![0.0_f32; in_dim],
            feat_var: vec![1.0_f32; in_dim],
            feat_count: 0,
            xn_buf: vec![0.0_f32; in_dim],
            feat_buf: vec![0.0_f32; f_dim],
            logits_buf: vec![0.0_f32; num_classes],
            probs_buf: vec![0.0_f32; num_classes],
            gw_buf: vec![0.0_f32; num_classes * f_dim],
            gb_buf: vec![0.0_f32; num_classes],
            fb_re_buf: vec![0.0_f32; units],
            fb_im_buf: vec![0.0_f32; units],
            standardize: true,
            lr: 0.01,
            rtrl_lr: 0.004,
            t: 0,
            had_nan: false,
            enable_input_clamp: false,
        }
    }

    pub fn reset_state(&mut self) {
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

    pub fn memory_footprint_bytes(&self) -> usize {
        (self.nu.len()
            + self.theta.len()
            + self.b_re.len()
            + self.b_im.len()
            + self.v_nu.len()
            + self.v_theta.len()
            + self.v_bre.len()
            + self.v_bim.len()
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
            + self.feat_mean.len()
            + self.feat_var.len()
            + self.xn_buf.len()
            + self.feat_buf.len()
            + self.logits_buf.len()
            + self.probs_buf.len()
            + self.gw_buf.len()
            + self.gb_buf.len()
            + self.fb_re_buf.len()
            + self.fb_im_buf.len())
            * 4
            + self.w.bytes()
            + self.b.bytes()
    }

    fn normalize_input_into(&mut self, x: &[f32], update_stats: bool) {
        self.xn_buf.fill(0.0);
        let n = self.in_dim.min(x.len());
        if !self.standardize {
            for j in 0..n {
                let mut v = x[j];
                if self.enable_input_clamp {
                    v = if v.is_finite() { v.clamp(-10.0, 10.0) } else { 0.0 };
                } else if !v.is_finite() {
                    self.had_nan = true;
                }
                self.xn_buf[j] = v;
            }
            return;
        }
        if update_stats {
            self.feat_count = self.feat_count.saturating_add(1);
            let alpha = if self.feat_count < 40 {
                (1.0 / (self.feat_count as f32)).clamp(0.025, 0.5)
            } else {
                0.008_f32
            };
            let keep = 1.0 - alpha;
            for j in 0..n {
                let mut raw = x[j];
                if self.enable_input_clamp {
                    raw = if raw.is_finite() { raw.clamp(-10.0, 10.0) } else { 0.0 };
                } else if !raw.is_finite() {
                    self.had_nan = true;
                }
                let mu = self.feat_mean[j];
                let diff = raw - mu;
                let new_mu = mu + alpha * diff;
                let new_var = (keep * self.feat_var[j] + alpha * diff * (raw - new_mu)).max(1e-3);
                self.feat_mean[j] = new_mu;
                self.feat_var[j] = new_var;
                let std = new_var.sqrt().max(0.05);
                self.xn_buf[j] = if self.feat_count <= 2 {
                    raw.clamp(-4.0, 4.0)
                } else {
                    ((raw - new_mu) / std).clamp(-4.0, 4.0)
                };
            }
        } else {
            for j in 0..n {
                let mut raw = x[j];
                if self.enable_input_clamp {
                    raw = if raw.is_finite() { raw.clamp(-10.0, 10.0) } else { 0.0 };
                } else if !raw.is_finite() {
                    self.had_nan = true;
                }
                let std = self.feat_var[j].sqrt().max(0.05);
                self.xn_buf[j] = if self.feat_count <= 2 {
                    raw
                } else {
                    (raw - self.feat_mean[j]) / std
                };
            }
        }
    }

    fn step_lru_forward_from_buf(&mut self) {
        let m = self.units;
        let d = self.in_dim;
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
            let dgam_dnu = (r * r * exp_nu) / gam;

            let hr_prev = self.h_re[j];
            let hi_prev = self.h_im[j];
            let lhr = lam_r * hr_prev - lam_i * hi_prev;
            let lhi = lam_r * hi_prev + lam_i * hr_prev;

            let mut ur = 0.0_f32;
            let mut ui = 0.0_f32;
            let off = j * d;
            for k in 0..d {
                let xk = self.xn_buf[k];
                if !xk.is_finite() {
                    self.had_nan = true;
                }
                ur += self.b_re[off + k] * xk;
                ui += self.b_im[off + k] * xk;
            }

            let hr = lhr + gam * ur;
            let hi = lhi + gam * ui;
            if !hr.is_finite() || !hi.is_finite() {
                self.had_nan = true;
            }
            self.h_re[j] = hr;
            self.h_im[j] = hi;

            let rho = (hr * hr + hi * hi + kap2).sqrt();
            self.rho[j] = rho;
            self.y_re[j] = hr / rho;
            self.y_im[j] = hi / rho;

            let pnr = self.p_nu_re[j];
            let pni = self.p_nu_im[j];
            self.p_nu_re[j] = (lam_r * pnr - lam_i * pni) - exp_nu * lhr + dgam_dnu * ur;
            self.p_nu_im[j] = (lam_r * pni + lam_i * pnr) - exp_nu * lhi + dgam_dnu * ui;

            let ptr = self.p_th_re[j];
            let pti = self.p_th_im[j];
            self.p_th_re[j] = (lam_r * ptr - lam_i * pti) - lhi;
            self.p_th_im[j] = (lam_r * pti + lam_i * ptr) + lhr;

            for k in 0..d {
                let pbr = self.p_b_re[off + k];
                let pbi = self.p_b_im[off + k];
                let xk = self.xn_buf[k];
                self.p_b_re[off + k] = (lam_r * pbr - lam_i * pbi) + gam * xk;
                self.p_b_im[off + k] = lam_r * pbi + lam_i * pbr;
            }
        }
    }

    pub fn predict_and_train(&mut self, x: &[f32], y: usize, train: bool) -> usize {
        self.normalize_input_into(x, train);
        self.step_lru_forward_from_buf();

        let f_dim = 2 * self.units + self.in_dim;
        let sc = 1.0 / (self.units.max(1) as f32).sqrt();
        for j in 0..self.units {
            self.feat_buf[2 * j] = self.y_re[j] * sc;
            self.feat_buf[2 * j + 1] = self.y_im[j] * sc;
        }
        for k in 0..self.in_dim {
            self.feat_buf[2 * self.units + k] = self.xn_buf[k] * 0.5;
        }

        let mut max_l = f32::NEG_INFINITY;
        for c in 0..self.num_classes {
            let off = c * f_dim;
            let s = self.b.w[c] + crate::math::fast_dot_raw(&self.w.w[off..off + f_dim], &self.feat_buf);
            if !s.is_finite() {
                self.had_nan = true;
            }
            self.logits_buf[c] = s;
            if s > max_l {
                max_l = s;
            }
        }

        let mut sum_e = 0.0_f32;
        for c in 0..self.num_classes {
            let e = (self.logits_buf[c] - max_l).exp();
            self.probs_buf[c] = e;
            sum_e += e;
        }
        if !sum_e.is_finite() || sum_e <= 0.0 {
            self.had_nan = true;
            self.probs_buf.fill(1.0 / (self.num_classes as f32));
        } else {
            let inv = 1.0 / sum_e;
            for p in &mut self.probs_buf {
                *p *= inv;
            }
        }

        let mut best = 0usize;
        let mut best_p = -1.0_f32;
        for (c, &p) in self.probs_buf.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best = c;
            }
        }

        if !train || self.had_nan || y >= self.num_classes {
            return best;
        }

        self.fb_re_buf.fill(0.0);
        self.fb_im_buf.fill(0.0);

        for c in 0..self.num_classes {
            let target = if c == y { 1.0_f32 } else { 0.0_f32 };
            let dl = self.probs_buf[c] - target;
            let err = target - self.probs_buf[c];
            self.gb_buf[c] = dl;
            let off = c * f_dim;
            for i in 0..f_dim {
                self.gw_buf[off + i] = dl * self.feat_buf[i];
            }
            for j in 0..self.units {
                self.fb_re_buf[j] += err * self.w.w[off + 2 * j] * sc;
                self.fb_im_buf[j] += err * self.w.w[off + 2 * j + 1] * sc;
            }
        }

        let rlr = self.rtrl_lr;
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

            let gnu = fr * self.p_nu_re[j] + fi * self.p_nu_im[j];
            self.v_nu[j] = 0.98 * self.v_nu[j] + 0.02 * gnu * gnu;
            let snu = (gnu / (self.v_nu[j] + 0.0025).sqrt()).clamp(-3.0, 3.0);
            self.nu[j] = (self.nu[j] + rlr * snu).clamp(-5.2, 1.5);

            let gth = fr * self.p_th_re[j] + fi * self.p_th_im[j];
            self.v_theta[j] = 0.98 * self.v_theta[j] + 0.02 * gth * gth;
            let sth = (gth / (self.v_theta[j] + 0.0025).sqrt()).clamp(-3.0, 3.0);
            self.theta[j] = (self.theta[j] + rlr * sth).clamp(-6.28, 6.28);

            let off = j * d;
            for k in 0..d {
                let pbr = self.p_b_re[off + k];
                let pbi = self.p_b_im[off + k];
                let gbr = fr * pbr + fi * pbi;
                let gbi = -fr * pbi + fi * pbr;

                self.v_bre[off + k] = 0.98 * self.v_bre[off + k] + 0.02 * gbr * gbr;
                let sbr = (gbr / (self.v_bre[off + k] + 0.0025).sqrt()).clamp(-3.0, 3.0);
                self.b_re[off + k] = (self.b_re[off + k] + rlr * sbr).clamp(-3.0, 3.0);

                self.v_bim[off + k] = 0.98 * self.v_bim[off + k] + 0.02 * gbi * gbi;
                let sbi = (gbi / (self.v_bim[off + k] + 0.0025).sqrt()).clamp(-3.0, 3.0);
                self.b_im[off + k] = (self.b_im[off + k] + rlr * sbi).clamp(-3.0, 3.0);
            }
        }

        self.t += 1;
        let lr = self.lr;
        let st = self.t;
        self.w.step(&self.gw_buf, lr, st);
        self.b.step(&self.gb_buf, lr, st);
        best
    }
}