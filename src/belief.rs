// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! 3 фундаментальных механизма без backprop (100% #[axiom_zero]):
//! 1. CayleyBeliefState с unsupervised novelty-гейтированием (защита от затирания на глитчах).
//! 2. EligibilityTraceRing: 3-факторные следы приемлемости для dense, sparse и Actor-Critic.
//! 3. PageHinkleyDetector: статистический CUSUM-детектор смены режима по возврату R_e.

use axiom_zero::{
    axiom_zero, sanitize_f32, total_div_f32, total_div_usize, total_get_copy, total_set,
};

use crate::math::{
    cayley_rotate_sectors, fast_dot, fast_saxpy, rand_signed_f32, sanitize_and_normalize_unit,
    splitmix64, su2_lie_bracket,
};

#[derive(Clone)]
pub struct CayleyBeliefState {
    pub dim: usize,
    pub half_dim: usize,
    pub lambda: f32,
    pub gamma: f32,
    pub rotor_gauge: Vec<f32>,
    pub state: Vec<f32>,
    pub obs_ema: Vec<f32>,
    pub obs_count: u64,
    pub rot_buf: Vec<f32>,
    pub comm_buf: Vec<f32>,
    pub initialized: bool,
    pub novelty_gate: f32,
}

#[axiom_zero]
impl CayleyBeliefState {
    pub fn new(
        dim: usize,
        half_dim: usize,
        lambda: f32,
        gamma: f32,
        rotor_angle_norm: f32,
        seed: u64,
    ) -> Self {
        if dim == 0 {
            return Self {
                dim: 0,
                half_dim: 0,
                lambda: sanitize_f32(lambda).clamp(0.2, 0.98),
                gamma: sanitize_f32(gamma).clamp(0.0, 1.5),
                rotor_gauge: Vec::new(),
                state: Vec::new(),
                obs_ema: Vec::new(),
                obs_count: 0,
                rot_buf: Vec::new(),
                comm_buf: Vec::new(),
                initialized: false,
                novelty_gate: 0.28,
            };
        }
        let d = dim.max(3);
        let mut rotor_gauge = vec![0.0_f32; d];
        let mut rng = splitmix64(seed ^ 0xCA71_E7B0_0001_55AA);
        let target_norm = sanitize_f32(rotor_angle_norm).clamp(0.03, 0.65);
        let num_sec = total_div_usize(d, 3, 1).max(1);
        let mut sec = 0usize;
        while sec + 3 <= d {
            let sec_idx = total_div_usize(sec, 3, 0);
            let frac = total_div_f32(sec_idx as f32, num_sec as f32, 0.5);
            let sec_norm = sanitize_f32(target_norm * (0.25 + 1.65 * frac)).clamp(0.03, 1.45);
            let x = rand_signed_f32(&mut rng);
            let y = rand_signed_f32(&mut rng);
            let z = rand_signed_f32(&mut rng);
            let n = (x * x + y * y + z * z).sqrt().max(1e-6);
            let sc = total_div_f32(sec_norm, n, sec_norm);
            let _ = total_set(&mut rotor_gauge, sec, sanitize_f32(x * sc));
            let _ = total_set(&mut rotor_gauge, sec + 1, sanitize_f32(y * sc));
            let _ = total_set(&mut rotor_gauge, sec + 2, sanitize_f32(z * sc));
            sec += 3;
        }
        Self {
            dim: d,
            half_dim: half_dim.min(d),
            lambda: sanitize_f32(lambda).clamp(0.2, 0.98),
            gamma: sanitize_f32(gamma).clamp(0.0, 1.5),
            rotor_gauge,
            state: vec![0.0_f32; d],
            obs_ema: vec![0.0_f32; d],
            obs_count: 0,
            rot_buf: vec![0.0_f32; d],
            comm_buf: vec![0.0_f32; d],
            initialized: false,
            novelty_gate: 0.28,
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        (self.rotor_gauge.len()
            + self.state.len()
            + self.obs_ema.len()
            + self.rot_buf.len()
            + self.comm_buf.len())
        .saturating_mul(4)
    }

    #[inline]
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.initialized = false;
    }

    pub fn measure_and_update_novelty(&mut self, obs_vec: &[f32]) -> f32 {
        self.obs_count = self.obs_count.saturating_add(1);
        let ema_norm = fast_dot(&self.obs_ema, &self.obs_ema).sqrt();
        let familiarity = if ema_norm > 1e-4 && self.obs_count > 8 {
            total_div_f32(fast_dot(obs_vec, &self.obs_ema), ema_norm, 0.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let alpha = if self.obs_count < 32 { 0.08_f32 } else { 0.025_f32 };
        let keep = 1.0 - alpha;
        for (e, &x) in self.obs_ema.iter_mut().zip(obs_vec.iter()) {
            *e = sanitize_f32(*e * keep + alpha * x);
        }
        sanitize_f32(1.0 - familiarity).clamp(0.0, 1.0)
    }

    pub fn update(&mut self, obs_vec: &[f32]) -> f32 {
        let d = self.dim;
        if d == 0 || obs_vec.is_empty() {
            return 0.0;
        }
        let novelty = self.measure_and_update_novelty(obs_vec);

        if !self.initialized {
            let n = d.min(obs_vec.len());
            self.state.fill(0.0);
            for i in 0..n {
                let _ = total_set(&mut self.state, i, total_get_copy(obs_vec, i, 0.0));
            }
            sanitize_and_normalize_unit(&mut self.state, 8.0);
            self.initialized = true;
            return novelty;
        }

        if self.novelty_gate > 0.0 && novelty < self.novelty_gate {
            return novelty;
        }
        let eff_nov = if self.novelty_gate <= 0.0 { 1.0_f32 } else { novelty };
        // При novelty_gate <= 0 сохраняем точную унитарную норму λ² + w_x² = 1 -> эффективное затухание == λ!
        let w_x = if self.novelty_gate <= 0.0 {
            sanitize_f32((1.0 - self.lambda * self.lambda).max(0.04).sqrt()).clamp(0.18, 0.85)
        } else {
            sanitize_f32(eff_nov * eff_nov * 0.72).clamp(0.15, 0.85)
        };
        cayley_rotate_sectors(&self.rotor_gauge, &self.state, &mut self.rot_buf);
        su2_lie_bracket(&self.state, obs_vec, &mut self.comm_buf);
        sanitize_and_normalize_unit(&mut self.comm_buf, 8.0);

        self.state.fill(0.0);
        fast_saxpy(&mut self.state, &self.rot_buf, self.lambda);
        fast_saxpy(&mut self.state, obs_vec, w_x);
        if self.gamma > 0.0 {
            fast_saxpy(&mut self.state, &self.comm_buf, self.gamma * eff_nov);
        }
        sanitize_and_normalize_unit(&mut self.state, 8.0);
        novelty
    }
}

#[derive(Clone)]
pub struct EligibilityTraceRing {
    pub dim: usize,
    pub capacity: usize,
    pub num_actions: usize,
    pub len: usize,
    pub beliefs: Vec<f32>,
    pub actions: Vec<usize>,
    pub rewards: Vec<f32>,
    pub novelties: Vec<f32>,
    pub next_values: Vec<f32>,
    pub values: Vec<f32>,
    pub action_probs: Vec<f32>,
    pub action_psi: Vec<f32>,
}

#[axiom_zero]
impl EligibilityTraceRing {
    pub fn new(dim: usize, capacity: usize) -> Self {
        Self::with_actions(dim, capacity, 0)
    }

    pub fn with_actions(dim: usize, capacity: usize, num_actions: usize) -> Self {
        let na = num_actions.min(32);
        if capacity == 0 || dim == 0 {
            return Self {
                dim: 0,
                capacity: 0,
                num_actions: na,
                len: 0,
                beliefs: Vec::new(),
                actions: Vec::new(),
                rewards: Vec::new(),
                novelties: Vec::new(),
                next_values: Vec::new(),
                values: Vec::new(),
                action_probs: Vec::new(),
                action_psi: Vec::new(),
            };
        }
        let d = dim.max(3);
        let cap = capacity.clamp(4, 256);
        Self {
            dim: d,
            capacity: cap,
            num_actions: na,
            len: 0,
            beliefs: vec![0.0_f32; cap.saturating_mul(d)],
            actions: vec![0usize; cap],
            rewards: vec![0.0_f32; cap],
            novelties: vec![0.0_f32; cap],
            next_values: vec![0.0_f32; cap],
            values: if na > 0 { vec![0.0_f32; cap] } else { Vec::new() },
            action_probs: if na > 0 { vec![0.0_f32; cap.saturating_mul(na)] } else { Vec::new() },
            action_psi: if na > 0 { vec![1.0_f32; cap.saturating_mul(na)] } else { Vec::new() },
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.beliefs.len().saturating_mul(4)
            + self.actions.len().saturating_mul(std::mem::size_of::<usize>())
            + (self.rewards.len()
                + self.novelties.len()
                + self.next_values.len()
                + self.values.len()
                + self.action_probs.len()
                + self.action_psi.len())
            .saturating_mul(4)
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn push(&mut self, belief: &[f32], action: usize, novelty: f32) {
        self.push_full(belief, action, novelty, 0.0, &[], &[]);
    }

    pub fn push_full(
        &mut self,
        belief: &[f32],
        action: usize,
        novelty: f32,
        value: f32,
        probs: &[f32],
        psis: &[f32],
    ) {
        if self.capacity == 0 {
            return;
        }
        let idx = if self.len < self.capacity {
            let i = self.len;
            self.len += 1;
            i
        } else {
            let d = self.dim;
            let na = self.num_actions;
            for k in 1..self.capacity {
                let dst_off = (k - 1).saturating_mul(d);
                let src_off = k.saturating_mul(d);
                for j in 0..d {
                    let v = total_get_copy(&self.beliefs, src_off + j, 0.0);
                    let _ = total_set(&mut self.beliefs, dst_off + j, v);
                }
                let a = total_get_copy(&self.actions, k, 0);
                let r = total_get_copy(&self.rewards, k, 0.0);
                let nov = total_get_copy(&self.novelties, k, 0.0);
                let nv = total_get_copy(&self.next_values, k, 0.0);
                let _ = total_set(&mut self.actions, k - 1, a);
                let _ = total_set(&mut self.rewards, k - 1, r);
                let _ = total_set(&mut self.novelties, k - 1, nov);
                let _ = total_set(&mut self.next_values, k - 1, nv);
                if na > 0 {
                    let vp = total_get_copy(&self.values, k, 0.0);
                    let _ = total_set(&mut self.values, k - 1, vp);
                    let pdst = (k - 1).saturating_mul(na);
                    let psrc = k.saturating_mul(na);
                    for j in 0..na {
                        let pv = total_get_copy(&self.action_probs, psrc + j, 0.0);
                        let _ = total_set(&mut self.action_probs, pdst + j, pv);
                        let sv = total_get_copy(&self.action_psi, psrc + j, 1.0);
                        let _ = total_set(&mut self.action_psi, pdst + j, sv);
                    }
                }
            }
            self.capacity - 1
        };

        let off = idx.saturating_mul(self.dim);
        if let Some(dst) = self.beliefs.get_mut(off..off + self.dim) {
            for (d_c, &b_c) in dst.iter_mut().zip(belief.iter()) {
                *d_c = sanitize_f32(b_c);
            }
        }
        let _ = total_set(&mut self.actions, idx, action);
        let _ = total_set(&mut self.rewards, idx, 0.0);
        let _ = total_set(&mut self.novelties, idx, sanitize_f32(novelty));
        let _ = total_set(&mut self.next_values, idx, 0.0);
        if self.num_actions > 0 {
            let _ = total_set(&mut self.values, idx, sanitize_f32(value));
            let poff = idx.saturating_mul(self.num_actions);
            for j in 0..self.num_actions {
                let pv = total_get_copy(probs, j, 0.0);
                let _ = total_set(&mut self.action_probs, poff + j, sanitize_f32(pv));
                let sv = total_get_copy(psis, j, 1.0);
                let _ = total_set(&mut self.action_psi, poff + j, sanitize_f32(sv));
            }
        }
    }

    pub fn set_last_reward(&mut self, reward: f32) {
        if self.len > 0 {
            let _ = total_set(&mut self.rewards, self.len - 1, sanitize_f32(reward));
        }
    }

    pub fn set_prev_next_value(&mut self, v_next: f32) {
        if self.len > 0 {
            let _ = total_set(&mut self.next_values, self.len - 1, sanitize_f32(v_next));
        }
    }

    #[inline]
    pub fn belief_at(&self, step: usize) -> &[f32] {
        let off = step.saturating_mul(self.dim);
        self.beliefs.get(off..off + self.dim).unwrap_or(&[])
    }

    #[inline]
    pub fn probs_at(&self, step: usize) -> &[f32] {
        let off = step.saturating_mul(self.num_actions);
        self.action_probs.get(off..off + self.num_actions).unwrap_or(&[])
    }

    #[inline]
    pub fn psi_at(&self, step: usize) -> &[f32] {
        let off = step.saturating_mul(self.num_actions);
        self.action_psi.get(off..off + self.num_actions).unwrap_or(&[])
    }
}

#[derive(Clone, Debug)]
pub struct PageHinkleyDetector {
    pub mu_bank: f32,
    pub cusum: f32,
    pub delta: f32,
    pub threshold: f32,
    pub min_warmup: usize,
    pub episodes_seen: usize,
    pub ema_alpha: f32,
}

#[axiom_zero]
impl PageHinkleyDetector {
    pub fn new(delta: f32, threshold: f32, min_warmup: usize) -> Self {
        Self {
            mu_bank: 0.0,
            cusum: 0.0,
            delta: sanitize_f32(delta).clamp(0.02, 0.45),
            threshold: sanitize_f32(threshold).clamp(0.4, 8.0),
            min_warmup: min_warmup.max(6),
            episodes_seen: 0,
            ema_alpha: 0.06,
        }
    }

    pub fn reset_cusum(&mut self) {
        self.cusum = 0.0;
    }

    pub fn observe(&mut self, r_norm: f32) -> bool {
        let r = sanitize_f32(r_norm).clamp(0.0, 1.0);
        self.episodes_seen = self.episodes_seen.saturating_add(1);

        if self.episodes_seen == 1 {
            self.mu_bank = r;
            self.cusum = 0.0;
            return false;
        }

        if self.episodes_seen <= self.min_warmup {
            let inv_n = total_div_f32(1.0, self.episodes_seen as f32, 0.1);
            self.mu_bank = sanitize_f32(self.mu_bank + inv_n * (r - self.mu_bank));
            self.cusum = 0.0;
            return false;
        }

        if self.mu_bank < 0.48 {
            self.mu_bank = sanitize_f32(self.mu_bank + 0.08 * (r - self.mu_bank));
            self.cusum = 0.0;
            return false;
        }

        let drop_sig = sanitize_f32(self.mu_bank - r - self.delta);
        self.cusum = sanitize_f32(self.cusum + drop_sig).max(0.0);

        if self.cusum < self.threshold * 0.35 {
            self.mu_bank = sanitize_f32(self.mu_bank + self.ema_alpha * (r - self.mu_bank));
        }

        self.cusum >= self.threshold
    }
}