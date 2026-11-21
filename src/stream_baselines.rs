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
    pub standardize: bool,
    pub lr: f32,
    t: usize,
    pub had_nan: bool,
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
            standardize: true,
            lr: 0.006,
            t: 0,
            had_nan: false,
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.w1.bytes()
            + self.b1.bytes()
            + self.w2.bytes()
            + self.b2.bytes()
            + (self.feat_mean.len() + self.feat_var.len()) * 4
    }

    fn normalize_input(&mut self, x: &[f32], update_stats: bool) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.in_dim];
        let n = self.in_dim.min(x.len());
        if !self.standardize {
            for j in 0..n {
                out[j] = x[j];
            }
            return out;
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
                out[j] = if self.feat_count <= 2 {
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
                out[j] = if self.feat_count <= 2 {
                    raw
                } else {
                    (raw - self.feat_mean[j]) / std
                };
            }
        }
        out
    }

    fn forward(&mut self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut h = vec![0.0_f32; self.hidden];
        for i in 0..self.hidden {
            let mut s = self.b1.w[i];
            for j in 0..self.in_dim.min(x.len()) {
                s += self.w1.w[i * self.in_dim + j] * x[j];
            }
            if !s.is_finite() {
                self.had_nan = true;
            }
            h[i] = s.max(0.0);
        }
        let mut logits = vec![0.0_f32; self.num_classes];
        let mut max_l = f32::NEG_INFINITY;
        for c in 0..self.num_classes {
            let mut s = self.b2.w[c];
            for j in 0..self.hidden {
                s += self.w2.w[c * self.hidden + j] * h[j];
            }
            if !s.is_finite() {
                self.had_nan = true;
            }
            logits[c] = s;
            if s > max_l {
                max_l = s;
            }
        }
        let mut probs = vec![0.0_f32; self.num_classes];
        let mut sum = 0.0_f32;
        for c in 0..self.num_classes {
            let e = (logits[c] - max_l).exp();
            probs[c] = e;
            sum += e;
        }
        if !sum.is_finite() || sum <= 0.0 {
            self.had_nan = true;
            probs.fill(1.0 / (self.num_classes as f32));
        } else {
            for p in &mut probs {
                *p /= sum;
            }
        }
        (h, probs)
    }

    pub fn predict_and_train(&mut self, x: &[f32], y: usize, train: bool) -> usize {
        let xn = self.normalize_input(x, train);
        let (h, probs) = self.forward(&xn);
        let mut best = 0usize;
        let mut best_p = -1.0_f32;
        for (c, &p) in probs.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best = c;
            }
        }
        if !train || self.had_nan {
            return best;
        }

        let mut gw2 = vec![0.0_f32; self.num_classes * self.hidden];
        let mut gb2 = vec![0.0_f32; self.num_classes];
        let mut dh = vec![0.0_f32; self.hidden];
        for c in 0..self.num_classes {
            let dl = probs[c] - if c == y { 1.0 } else { 0.0 };
            gb2[c] = dl;
            for j in 0..self.hidden {
                gw2[c * self.hidden + j] = dl * h[j];
                dh[j] += self.w2.w[c * self.hidden + j] * dl;
            }
        }
        let mut gw1 = vec![0.0_f32; self.hidden * self.in_dim];
        let mut gb1 = vec![0.0_f32; self.hidden];
        for i in 0..self.hidden {
            let dz = if h[i] > 0.0 { dh[i] } else { 0.0 };
            gb1[i] = dz;
            for j in 0..self.in_dim.min(xn.len()) {
                gw1[i * self.in_dim + j] = dz * xn[j];
            }
        }
        self.t += 1;
        let lr = self.lr;
        let st = self.t;
        self.w1.step(&gw1, lr, st);
        self.b1.step(&gb1, lr, st);
        self.w2.step(&gw2, lr, st);
        self.b2.step(&gb2, lr, st);
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
    pub standardize: bool,
    pub lr: f32,
    pub rtrl_lr: f32,
    t: usize,
    pub had_nan: bool,
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
            standardize: true,
            lr: 0.01,
            rtrl_lr: 0.004,
            t: 0,
            had_nan: false,
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
            + self.feat_var.len())
            * 4
            + self.w.bytes()
            + self.b.bytes()
    }

    fn normalize_input(&mut self, x: &[f32], update_stats: bool) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.in_dim];
        let n = self.in_dim.min(x.len());
        if !self.standardize {
            for j in 0..n {
                if !x[j].is_finite() {
                    self.had_nan = true;
                }
                out[j] = x[j];
            }
            return out;
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
                out[j] = if self.feat_count <= 2 {
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
                out[j] = if self.feat_count <= 2 {
                    raw
                } else {
                    (raw - self.feat_mean[j]) / std
                };
            }
        }
        out
    }

    fn step_lru_forward(&mut self, x: &[f32]) {
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

            for k in 0..x_len {
                let pbr = self.p_b_re[off + k];
                let pbi = self.p_b_im[off + k];
                self.p_b_re[off + k] = (lam_r * pbr - lam_i * pbi) + gam * x[k];
                self.p_b_im[off + k] = lam_r * pbi + lam_i * pbr;
            }
        }
    }

    pub fn predict_and_train(&mut self, x: &[f32], y: usize, train: bool) -> usize {
        let xn = self.normalize_input(x, train);
        self.step_lru_forward(&xn);

        let f_dim = 2 * self.units + self.in_dim;
        let mut feat = vec![0.0_f32; f_dim];
        let sc = 1.0 / (self.units.max(1) as f32).sqrt();
        for j in 0..self.units {
            feat[2 * j] = self.y_re[j] * sc;
            feat[2 * j + 1] = self.y_im[j] * sc;
        }
        for k in 0..self.in_dim.min(xn.len()) {
            feat[2 * self.units + k] = xn[k] * 0.5;
        }

        let mut logits = vec![0.0_f32; self.num_classes];
        let mut max_l = f32::NEG_INFINITY;
        for c in 0..self.num_classes {
            let mut s = self.b.w[c];
            let off = c * f_dim;
            for i in 0..f_dim {
                s += self.w.w[off + i] * feat[i];
            }
            if !s.is_finite() {
                self.had_nan = true;
            }
            logits[c] = s;
            if s > max_l {
                max_l = s;
            }
        }

        let mut probs = vec![0.0_f32; self.num_classes];
        let mut sum_e = 0.0_f32;
        for c in 0..self.num_classes {
            let e = (logits[c] - max_l).exp();
            probs[c] = e;
            sum_e += e;
        }
        if !sum_e.is_finite() || sum_e <= 0.0 {
            self.had_nan = true;
            probs.fill(1.0 / (self.num_classes as f32));
        } else {
            for p in &mut probs {
                *p /= sum_e;
            }
        }

        let mut best = 0usize;
        let mut best_p = -1.0_f32;
        for (c, &p) in probs.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best = c;
            }
        }

        if !train || self.had_nan || y >= self.num_classes {
            return best;
        }

        let mut gw = vec![0.0_f32; self.num_classes * f_dim];
        let mut gb = vec![0.0_f32; self.num_classes];
        let mut fb_re = vec![0.0_f32; self.units];
        let mut fb_im = vec![0.0_f32; self.units];

        for c in 0..self.num_classes {
            let target = if c == y { 1.0_f32 } else { 0.0_f32 };
            let dl = probs[c] - target;
            let err = target - probs[c];
            gb[c] = dl;
            let off = c * f_dim;
            for i in 0..f_dim {
                gw[off + i] = dl * feat[i];
            }
            for j in 0..self.units {
                fb_re[j] += err * self.w.w[off + 2 * j] * sc;
                fb_im[j] += err * self.w.w[off + 2 * j + 1] * sc;
            }
        }

        let rlr = self.rtrl_lr;
        let d = self.in_dim;
        for j in 0..self.units {
            let yr = self.y_re[j];
            let yi = self.y_im[j];
            let gr = fb_re[j];
            let gi = fb_im[j];
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
        self.w.step(&gw, lr, st);
        self.b.step(&gb, lr, st);
        best
    }
}