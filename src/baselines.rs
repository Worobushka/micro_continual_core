// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! 3 честных бейзлайна в том же бинарнике на чистом Rust (без PyTorch):
//! 1. Random / Oracle (границы).
//! 2. TabularPHBaseline (табличный Q-learning + Page-Hinkley).
//! 3. OnlineGruBaseline (GRU H=32 + A2C + Entropy + Gradient Clipping + BPTT + Adam).

use std::collections::HashMap;

use crate::belief::PageHinkleyDetector;
use crate::math::{extract_tokens, fnv1a_bytes, rand_signed_f32, rand_unit_f32, splitmix64};

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
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        for i in 0..self.w.len() {
            let g = grad[i].clamp(-1.0, 1.0);
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            self.w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

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
    h_cur: Vec<f32>,
    cache: Vec<GruStepCache>,
    adam_step: usize,
    pub lr: f32,
    pub gamma: f32,
    pub rng: u64,
    pub had_nan: bool,
}

impl OnlineGruBaseline {
    pub fn new(x_dim: usize, h_dim: usize, num_actions: usize, seed: u64) -> Self {
        let mut rng = splitmix64(seed ^ 0x6200_BA5E_114E);
        let sx = 0.5 / (x_dim as f32).sqrt();
        let sh = 0.5 / (h_dim as f32).sqrt();
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
        }
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
    }

    pub fn start_episode(&mut self) {
        self.h_cur.fill(0.0);
        self.cache.clear();
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

        let mut logits = vec![0.0_f32; self.num_actions];
        let mut max_l = f32::NEG_INFINITY;
        for a in 0..self.num_actions {
            let mut s = self.bo.w[a];
            for j in 0..hd {
                s += self.wo.w[a * hd + j] * h[j];
            }
            if !s.is_finite() {
                self.had_nan = true;
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
            self.had_nan = true;
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
        self.step_with_x(x, train_mode)
    }

    fn step_with_x(&mut self, x: Vec<f32>, train_mode: bool) -> usize {
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

        let mut logits = vec![0.0_f32; self.num_actions];
        let mut max_l = f32::NEG_INFINITY;
        for a in 0..self.num_actions {
            let mut s = self.bo.w[a];
            for j in 0..hd {
                s += self.wo.w[a * hd + j] * h[j];
            }
            if !s.is_finite() {
                self.had_nan = true;
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
            self.had_nan = true;
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
        if !train_mode || self.cache.is_empty() || self.had_nan {
            return;
        }
        let t_len = self.cache.len();
        let has_dense = self.cache.iter().any(|c| c.reward.abs() > 1e-6);
        let mut returns = vec![0.0_f32; t_len];
        if has_dense {
            let mut run = 0.0_f32;
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
                dh_prev[i] = dh[i] * (1.0 - c.z[i]);

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
    banks: Vec<Vec<MfecEntry>>,
    phs: Vec<PageHinkleyDetector>,
    pub active_bank: usize,
    trace: Vec<(Vec<f32>, usize, f32)>,
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
            banks,
            phs,
            active_bank: 0,
            trace: Vec::with_capacity(24),
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
        entries + self.phs.len() * 40
    }

    pub fn start_episode(&mut self) {
        self.trace.clear();
    }

    fn encode_obs(&self, obs: &str) -> Vec<f32> {
        let mut v = vec![0.0_f32; self.dim];
        for tok in extract_tokens(obs) {
            let h1 = fnv1a_bytes(tok.as_bytes(), 0x3141_5926_5358_9793);
            let h2 = splitmix64(h1 ^ 0x2718_2818_2845_9045);
            for k in 0..4usize {
                let m = splitmix64(h2.wrapping_add((k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
                let idx = (m as usize) % self.dim;
                let sgn = if ((m >> 20) & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
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
        let key = self.encode_obs(obs);
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
            let mut best_sim = 0.94_f32;
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
    pub gamma: f32,
    pub trace_lambda: f32,
    pub actor_lr: f32,
    pub critic_lr: f32,
    pub rtrl_lr: f32,
    pub entropy_beta: f32,
    pub rng: u64,
    pub had_nan: bool,
}

impl RtrrlLruBaseline {
    pub fn new(in_dim: usize, units: usize, num_actions: usize, seed: u64) -> Self {
        let mut rng = splitmix64(seed ^ 0x2721_1200_AA41);
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
        let mut actor_w = vec![0.0_f32; num_actions * f_dim];
        let mut critic_w = vec![0.0_f32; f_dim];
        for w in &mut actor_w {
            *w = rand_signed_f32(&mut rng) * 0.05;
        }
        for w in &mut critic_w {
            *w = rand_signed_f32(&mut rng) * 0.05;
        }
        Self {
            in_dim,
            units,
            num_actions,
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
            actor_b: vec![0.0; num_actions],
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
            e_actor_w: vec![0.0; num_actions * f_dim],
            e_actor_b: vec![0.0; num_actions],
            e_critic_w: vec![0.0; f_dim],
            e_critic_b: 0.0,
            has_pending: false,
            pending_v: 0.0,
            pending_r: 0.0,
            td_mean: 0.0,
            td_var: 0.25,
            gamma: 0.95,
            trace_lambda: 0.80,
            actor_lr: 0.045,
            critic_lr: 0.12,
            rtrl_lr: 0.006,
            entropy_beta: 0.008,
            rng,
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
            + self.e_critic_w.len())
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
    }

    /// Один прямой такт комплексного LRU + точное обновление чувствительностей RTRL (exp/sin/cos).
    pub fn step_lru_forward(&mut self, x: &[f32]) {
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

    fn make_features(&self, x: &[f32]) -> Vec<f32> {
        let f_dim = 2 * self.units + self.in_dim;
        let mut f = vec![0.0_f32; f_dim];
        let sc = 1.0 / (self.units.max(1) as f32).sqrt();
        for j in 0..self.units {
            f[2 * j] = self.y_re[j] * sc;
            f[2 * j + 1] = self.y_im[j] * sc;
        }
        for k in 0..self.in_dim.min(x.len()) {
            f[2 * self.units + k] = x[k] * 0.5;
        }
        f
    }

    fn critic_val(&self, feat: &[f32]) -> f32 {
        let mut v = self.critic_b;
        for (w, &x) in self.critic_w.iter().zip(feat.iter()) {
            v += w * x;
        }
        v
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
        let mut x = vec![0.0_f32; self.in_dim];
        for i in 0..self.in_dim.min(features.len()) {
            x[i] = features[i];
        }
        if seu_nan && !x.is_empty() {
            x[0] = f32::NAN;
        }
        if seu_spikes && x.len() > 1 {
            x[1] = 1e28;
        }

        self.step_lru_forward(&x);
        let feat = self.make_features(&x);
        let v_cur = self.critic_val(&feat);
        if !v_cur.is_finite() {
            self.had_nan = true;
        }

        if train_mode && self.has_pending && !self.had_nan {
            let td_err = self.pending_r + self.gamma * v_cur - self.pending_v;
            self.apply_td_update(td_err);
        }

        let f_dim = feat.len();
        let mut logits = vec![0.0_f32; self.num_actions];
        let mut max_l = f32::NEG_INFINITY;
        for a in 0..self.num_actions {
            let off = a * f_dim;
            let mut s = self.actor_b[a];
            for i in 0..f_dim {
                s += self.actor_w[off + i] * feat[i];
            }
            if !s.is_finite() {
                self.had_nan = true;
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
            self.had_nan = true;
            probs.fill(1.0 / (self.num_actions as f32));
        } else {
            for p in &mut probs {
                *p /= sum_e;
            }
        }

        let act = if train_mode && !self.had_nan {
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
            let mut ba = 0usize;
            let mut bp = -1.0_f32;
            for (a, &p) in probs.iter().enumerate() {
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
                self.e_critic_w[i] = decay * self.e_critic_w[i] + feat[i];
            }
            let ecn: f32 = self.e_critic_w.iter().map(|v| v * v).sum();
            if ecn > 16.0 {
                let s = 4.0 / ecn.sqrt();
                for v in &mut self.e_critic_w {
                    *v *= s;
                }
            }
            self.e_critic_b = (decay * self.e_critic_b + 1.0).clamp(-5.0, 5.0);

            let mut fb_re = vec![0.0_f32; self.units];
            let mut fb_im = vec![0.0_f32; self.units];
            for a in 0..self.num_actions {
                let score = (if a == act { 1.0 } else { 0.0 }) - probs[a];
                let off = a * f_dim;
                let mut ean = 0.0_f32;
                for i in 0..f_dim {
                    let nv = decay * self.e_actor_w[off + i] + score * feat[i];
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
                    fb_re[j] += score * self.actor_w[off + 2 * j] * sc;
                    fb_im[j] += score * self.actor_w[off + 2 * j + 1] * sc;
                }
            }
            for j in 0..self.units {
                fb_re[j] += 0.15 * self.critic_w[2 * j] * sc;
                fb_im[j] += 0.15 * self.critic_w[2 * j + 1] * sc;
            }

            // Точный якобиан Norm_kappa: f_h = (1/rho) * (fb - (fb . y) * y)
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
                for &p in &probs {
                    let pc = p.max(1e-6);
                    sum_plogp += pc * pc.ln();
                }
                for a in 0..self.num_actions {
                    let pc = probs[a].max(1e-6);
                    let d_ent = self.entropy_beta * pc * (sum_plogp - pc.ln());
                    let step = self.actor_lr * d_ent;
                    let off = a * f_dim;
                    for i in 0..f_dim {
                        self.actor_w[off + i] += step * feat[i];
                    }
                    self.actor_b[a] = (self.actor_b[a] + 0.25 * step).clamp(-3.0, 3.0);
                }
            }

            self.pending_v = self.critic_val(&feat);
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
}