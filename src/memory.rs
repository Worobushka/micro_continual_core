// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Непрерывная векторная кернел-память `KerrKernelBank` (W1 Kerr p=4 + i8-ключи):
//! - Хранит `q_vals`, `q_max` (детектор алиасинга скрытых состояний!) и `q_visits`.
//! - Один линейный проход по i8-ключам считает и `d_full`, и семантическое `d_sem`.

use axiom_zero::{
    axiom_zero, sanitize_bounded_f32, sanitize_f32, total_div_f32, total_get_copy,
    total_rem_usize, total_set,
};

use crate::math::{fast_dot, kerr_competition_into, sanitize_and_normalize_unit, sqrt_newton_f32};

#[derive(Clone)]
pub struct KerrKernelBank {
    pub dim: usize,
    pub half_dim: usize,
    pub num_actions: usize,
    pub capacity: usize,
    pub active_count: usize,
    pub keys_i8: Vec<i8>,
    pub key_scales: Vec<f32>,
    pub key_sem_norms: Vec<f32>,
    pub q_vals: Vec<f32>,   // [capacity * num_actions]
    pub q_max: Vec<f32>,    // [capacity * num_actions] оптимистический максимум (детектор алиасинга)
    pub q_visits: Vec<f32>, // [capacity * num_actions]
    pub key_masses: Vec<f32>,
    pub coalesce_cos: f32,
    pub overwrite_cursor: usize,
}

#[axiom_zero]
impl KerrKernelBank {
    pub fn new(dim: usize, half_dim: usize, num_actions: usize, capacity: usize, coalesce_cos: f32) -> Self {
        let na = num_actions.clamp(2, 64);
        if capacity == 0 || dim == 0 {
            return Self {
                dim: 0,
                half_dim: 0,
                num_actions: na,
                capacity: 0,
                active_count: 0,
                keys_i8: Vec::new(),
                key_scales: Vec::new(),
                key_sem_norms: Vec::new(),
                q_vals: Vec::new(),
                q_max: Vec::new(),
                q_visits: Vec::new(),
                key_masses: Vec::new(),
                coalesce_cos: sanitize_f32(coalesce_cos).clamp(0.55, 0.96),
                overwrite_cursor: 0,
            };
        }
        let d = dim.max(3);
        let hd = half_dim.min(d).max(3);
        let cap = capacity.clamp(16, 256);
        Self {
            dim: d,
            half_dim: hd,
            num_actions: na,
            capacity: cap,
            active_count: 0,
            keys_i8: vec![0i8; cap.saturating_mul(d)],
            key_scales: vec![0.0_f32; cap],
            key_sem_norms: vec![1.0_f32; cap],
            q_vals: vec![0.0_f32; cap.saturating_mul(na)],
            q_max: vec![-2.0_f32; cap.saturating_mul(na)],
            q_visits: vec![0.0_f32; cap.saturating_mul(na)],
            key_masses: vec![0.0_f32; cap],
            coalesce_cos: sanitize_f32(coalesce_cos).clamp(0.55, 0.96),
            overwrite_cursor: 0,
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.keys_i8.len()
            + (self.key_scales.len()
                + self.key_sem_norms.len()
                + self.q_vals.len()
                + self.q_max.len()
                + self.q_visits.len()
                + self.key_masses.len())
            .saturating_mul(4)
    }

    pub fn clear(&mut self) {
        self.keys_i8.fill(0);
        self.key_scales.fill(0.0);
        self.key_sem_norms.fill(1.0);
        self.q_vals.fill(0.0);
        self.q_max.fill(-2.0);
        self.q_visits.fill(0.0);
        self.key_masses.fill(0.0);
        self.active_count = 0;
        self.overwrite_cursor = 0;
    }

    fn quant_write_key(src: &[f32], half_dim: usize, dst: &mut [i8]) -> (f32, f32) {
        let mut ma = 0.0_f32;
        for &v in src {
            let av = sanitize_f32(v).abs();
            if av > ma {
                ma = av;
            }
        }
        if ma <= 1e-8 {
            return (0.0, 1.0);
        }
        let sc = total_div_f32(ma, 127.0, 0.0);
        let inv = total_div_f32(127.0, ma, 0.0);
        for (d, &v) in dst.iter_mut().zip(src.iter()) {
            *d = (sanitize_f32(v) * inv).round().clamp(-127.0, 127.0) as i8;
        }
        let sem_sl = src.get(0..half_dim).unwrap_or(&[]);
        let sem_norm = sqrt_newton_f32(fast_dot(sem_sl, sem_sl)).max(1e-4);
        (sc, sem_norm)
    }

    #[inline]
    fn dot_key_dual(&self, slot: usize, q: &[f32], q_sem_norm: f32) -> (f32, f32) {
        let sc = total_get_copy(&self.key_scales, slot, 0.0);
        if sc == 0.0 {
            return (0.0, 0.0);
        }
        let off = slot.saturating_mul(self.dim);
        let Some(sl) = self.keys_i8.get(off..off + self.dim) else {
            return (0.0, 0.0);
        };
        let hd = self.half_dim;
        let mut s_sem = 0.0_f32;
        let mut s_rest = 0.0_f32;
        if let (Some(k_lo), Some(q_lo)) = (sl.get(0..hd), q.get(0..hd)) {
            for (&kv, &qv) in k_lo.iter().zip(q_lo.iter()) {
                s_sem += (kv as f32) * qv;
            }
        }
        if let (Some(k_hi), Some(q_hi)) = (sl.get(hd..self.dim), q.get(hd..self.dim)) {
            for (&kv, &qv) in k_hi.iter().zip(q_hi.iter()) {
                s_rest += (kv as f32) * qv;
            }
        }
        let d_full = sanitize_f32((s_sem + s_rest) * sc);
        let k_sem_n = total_get_copy(&self.key_sem_norms, slot, 1.0);
        let d_sem = total_div_f32(s_sem * sc, k_sem_n * q_sem_norm, 0.0);
        (d_full, sanitize_f32(d_sem))
    }

    fn decode_key(&self, slot: usize, out: &mut [f32]) {
        out.fill(0.0);
        let sc = total_get_copy(&self.key_scales, slot, 0.0);
        if sc == 0.0 {
            return;
        }
        let off = slot.saturating_mul(self.dim);
        if let Some(sl) = self.keys_i8.get(off..off + self.dim) {
            for (d, &kv) in out.iter_mut().zip(sl.iter()) {
                *d = (kv as f32) * sc;
            }
        }
    }

    /// Чтение непрерывных оценок `out_q[a]` и посещений `out_v[a]` по W1 Kerr-ядру.
    pub fn query_into(
        &self,
        query: &[f32],
        kerr_p: usize,
        tau: f32,
        use_semantic_fallback: bool,
        sims_buf: &mut [f32],
        weights_buf: &mut [f32],
        out_q: &mut [f32],
        out_v: &mut [f32],
    ) -> f32 {
        out_q.fill(0.0);
        out_v.fill(0.0);
        let n = self.active_count.min(sims_buf.len()).min(weights_buf.len());
        if n == 0 {
            return 0.0;
        }
        let q_sem = query.get(0..self.half_dim).unwrap_or(&[]);
        let q_sem_norm = sqrt_newton_f32(fast_dot(q_sem, q_sem)).max(1e-4);

        let mut max_sim = 0.0_f32;
        let mut max_full = 0.0_f32;
        for sl in 0..n {
            let (d_full, d_sem) = self.dot_key_dual(sl, query, q_sem_norm);
            if d_full > max_full {
                max_full = d_full;
            }
            let sim = if use_semantic_fallback {
                d_full.max(0.98 * d_sem)
            } else {
                d_full
            };
            if sim > max_sim {
                max_sim = sim;
            }
            let _ = total_set(sims_buf, sl, sim);
        }

        let (eff_tau, eff_eps) = if use_semantic_fallback && max_full < 0.65 {
            (tau.min(0.22), 1e-4_f32)
        } else {
            (tau, 1e-3_f32)
        };

        let den = kerr_competition_into(
            sims_buf.get(0..n).unwrap_or(&[]),
            eff_tau,
            kerr_p,
            eff_eps,
            weights_buf.get_mut(0..n).unwrap_or(&mut []),
        );
        if den <= eff_eps * 1.05 {
            return max_sim.clamp(0.0, 1.0);
        }

        let na = self.num_actions;
        for sl in 0..n {
            let w = total_get_copy(weights_buf, sl, 0.0);
            if w > 1e-5 {
                let off = sl.saturating_mul(na);
                for a in 0..na {
                    let qm = total_get_copy(&self.q_vals, off + a, 0.0);
                    let vm = total_get_copy(&self.q_visits, off + a, 0.0);
                    let _ = total_set(out_q, a, sanitize_f32(total_get_copy(out_q, a, 0.0) + w * qm));
                    let _ = total_set(out_v, a, sanitize_f32(total_get_copy(out_v, a, 0.0) + w * vm));
                }
            }
        }
        max_sim.clamp(0.0, 1.0)
    }

    /// Чтение реактивного `obs_kernel` с детекцией алиасинга (по `q_max`):
    /// - Если действие иногда срабатывало в этом наблюдении (`q_max > 0.08`), но в среднем `< 0.08`,
    ///   оно зависит от скрытого состояния -> отдаём `0.0` (делегируем в `belief_kernel`!).
    /// - Если действие стабильно положительно (`q_mean > 0.08`) или НИ РАЗУ не сработало (`q_max <= 0.05` и `v >= 2`),
    ///   обобщаем его на все скрытые состояния сразу!
    pub fn query_obs_gated_into(
        &self,
        query: &[f32],
        kerr_p: usize,
        tau: f32,
        use_semantic_fallback: bool,
        sims_buf: &mut [f32],
        weights_buf: &mut [f32],
        out_q: &mut [f32],
        out_v: &mut [f32],
    ) -> f32 {
        out_q.fill(0.0);
        out_v.fill(0.0);
        let n = self.active_count.min(sims_buf.len()).min(weights_buf.len());
        if n == 0 {
            return 0.0;
        }
        let q_sem = query.get(0..self.half_dim).unwrap_or(&[]);
        let q_sem_norm = sqrt_newton_f32(fast_dot(q_sem, q_sem)).max(1e-4);

        let mut max_sim = 0.0_f32;
        for sl in 0..n {
            let (d_full, d_sem) = self.dot_key_dual(sl, query, q_sem_norm);
            let sim = if use_semantic_fallback {
                d_full.max(0.92 * d_sem)
            } else {
                d_full
            };
            if sim > max_sim {
                max_sim = sim;
            }
            let _ = total_set(sims_buf, sl, sim);
        }

        let den = kerr_competition_into(
            sims_buf.get(0..n).unwrap_or(&[]),
            tau,
            kerr_p,
            1e-4,
            weights_buf.get_mut(0..n).unwrap_or(&mut []),
        );
        if den <= 1.05e-4 {
            return max_sim.clamp(0.0, 1.0);
        }

        let na = self.num_actions;
        let mut max_acc = [-2.0_f32; 64];
        for sl in 0..n {
            let w = total_get_copy(weights_buf, sl, 0.0);
            if w > 1e-5 {
                let off = sl.saturating_mul(na);
                for a in 0..na {
                    let qm = total_get_copy(&self.q_vals, off + a, 0.0);
                    let qmx = total_get_copy(&self.q_max, off + a, -2.0);
                    let vm = total_get_copy(&self.q_visits, off + a, 0.0);
                    let _ = total_set(out_q, a, sanitize_f32(total_get_copy(out_q, a, 0.0) + w * qm));
                    let _ = total_set(out_v, a, sanitize_f32(total_get_copy(out_v, a, 0.0) + w * vm));
                    let cur_mx = total_get_copy(&max_acc, a, -2.0);
                    if w > 0.15 && qmx > cur_mx {
                        let _ = total_set(&mut max_acc, a, qmx);
                    }
                }
            }
        }

        for a in 0..na {
            let qm = total_get_copy(out_q, a, 0.0);
            let qmx = total_get_copy(&max_acc, a, -2.0);
            let vm = total_get_copy(out_v, a, 0.0);
            if qmx > 0.08 && qm < 0.08 {
                // Алиасированное действие (успех зависит от скрытого состояния!) -> не подавлять!
                let _ = total_set(out_q, a, 0.0);
                let _ = total_set(out_v, a, 0.0);
            } else if qm < 0.0 && vm < 1.5 {
                // Ещё мало наблюдений для глобального запрета
                let _ = total_set(out_q, a, 0.0);
            }
        }
        max_sim.clamp(0.0, 1.0)
    }

    pub fn update_action_value(
        &mut self,
        key: &[f32],
        action: usize,
        target_q: f32,
        step_weight: f32,
        tmp_key: &mut [f32],
    ) {
        if self.capacity == 0 || action >= self.num_actions {
            return;
        }
        let tgt = sanitize_bounded_f32(target_q, 3.0);
        let sw = sanitize_f32(step_weight).clamp(0.05, 1.0);
        let q_sem = key.get(0..self.half_dim).unwrap_or(&[]);
        let q_sem_norm = sqrt_newton_f32(fast_dot(q_sem, q_sem)).max(1e-4);

        let mut best_slot = 0usize;
        let mut best_sim = -1.0_f32;
        for sl in 0..self.active_count {
            let (d_full, _) = self.dot_key_dual(sl, key, q_sem_norm);
            if d_full > best_sim {
                best_sim = d_full;
                best_slot = sl;
            }
        }

        let na = self.num_actions;
        if self.active_count > 0 && best_sim >= self.coalesce_cos {
            let off = best_slot.saturating_mul(na);
            let old_q = total_get_copy(&self.q_vals, off + action, 0.0);
            let old_mx = total_get_copy(&self.q_max, off + action, -2.0);
            let old_v = total_get_copy(&self.q_visits, off + action, 0.0);
            let alpha = total_div_f32(0.80 * sw, 1.0 + 0.25 * old_v, 0.15).clamp(0.15, 0.85);
            let new_q = sanitize_bounded_f32(old_q + alpha * (tgt - old_q), 2.5);
            let new_mx = if old_mx < -1.5 {
                tgt
            } else {
                sanitize_bounded_f32((old_mx * 0.985).max(tgt), 2.5)
            };
            let _ = total_set(&mut self.q_vals, off + action, new_q);
            let _ = total_set(&mut self.q_max, off + action, new_mx);
            let _ = total_set(&mut self.q_visits, off + action, (old_v + sw).min(30.0));

            let km = total_get_copy(&self.key_masses, best_slot, 1.0);
            let blend = total_div_f32(0.35, km + 1.0, 0.05).clamp(0.02, 0.25);
            self.decode_key(best_slot, tmp_key);
            for (dst, &k_c) in tmp_key.iter_mut().zip(key.iter()) {
                *dst = sanitize_f32(*dst * (1.0 - blend) + k_c * blend);
            }
            sanitize_and_normalize_unit(tmp_key, 8.0);
            let koff = best_slot.saturating_mul(self.dim);
            if let Some(dk) = self.keys_i8.get_mut(koff..koff + self.dim) {
                let (sc, sn) = Self::quant_write_key(tmp_key, self.half_dim, dk);
                let _ = total_set(&mut self.key_scales, best_slot, sc);
                let _ = total_set(&mut self.key_sem_norms, best_slot, sn);
            }
            let _ = total_set(&mut self.key_masses, best_slot, (km + 1.0).min(64.0));
            return;
        }

        let target = if self.active_count < self.capacity {
            let idx = self.active_count;
            self.active_count += 1;
            idx
        } else {
            // Выселяем самый малозначимый (одноразовый) прототип, защищая зрелые состояния!
            let mut min_idx = total_rem_usize(self.overwrite_cursor, self.capacity, 0);
            let mut min_mass = total_get_copy(&self.key_masses, min_idx, 1e9);
            for sl in 0..self.capacity {
                let m = total_get_copy(&self.key_masses, sl, 0.0);
                if m < min_mass {
                    min_mass = m;
                    min_idx = sl;
                }
            }
            self.overwrite_cursor = min_idx.wrapping_add(1);
            min_idx
        };

        let koff = target.saturating_mul(self.dim);
        if let Some(dk) = self.keys_i8.get_mut(koff..koff + self.dim) {
            let (sc, sn) = Self::quant_write_key(key, self.half_dim, dk);
            let _ = total_set(&mut self.key_scales, target, sc);
            let _ = total_set(&mut self.key_sem_norms, target, sn);
        }
        let qoff = target.saturating_mul(na);
        for a in 0..na {
            let _ = total_set(&mut self.q_vals, qoff + a, 0.0);
            let _ = total_set(&mut self.q_max, qoff + a, -2.0);
            let _ = total_set(&mut self.q_visits, qoff + a, 0.0);
        }
        let _ = total_set(&mut self.q_vals, qoff + action, sanitize_bounded_f32(tgt * 0.80, 2.5));
        let _ = total_set(&mut self.q_max, qoff + action, tgt);
        let _ = total_set(&mut self.q_visits, qoff + action, sw);
        let _ = total_set(&mut self.key_masses, target, 1.0);
    }
}