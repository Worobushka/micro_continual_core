// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Точный блочно-диагональный онлайн-RTRL на рациональных 3D роторах Кэли SO(3)^S:
//! - 0% BPTT, 0% exp/sin/cos/sigmoid в рекуррентности, 100% #[axiom_zero].
//! - Линейная динамика внутри:
//!     u_{t,s} = lambda_s * R(omega_s * e_s) u_{t-1,s} + d_s * sum_j w_{s,j} x_{t,j}
//!   где lambda_s = 0.995 / (1 + alpha_s^2) in (0, 0.995] (строгая BIBO/SEU-устойчивость!).
//! - Точный рациональный якобиан Norm_kappa(u) = u / sqrt(||u||^2 + kappa^2) на чтении.
//! - Поддерживает как сопряжённую мультимасштабную сетку, так и развязанную сетку (частота x Q)
//!   для Fast- и Slow-слоёв волнового разложения внутри единого `MicroContinualCore`.

use axiom_zero::{
    axiom_zero, sanitize_bounded_f32, sanitize_f32, total_div_f32, total_div_unit_shifted_f32,
    total_div_usize, total_get_copy, total_rem_usize, total_set,
};

use crate::algebra::{
    hh5_apply_pre, hh5_deriv_axis, mat5_mul_vec5, ortho_dir_3d_next, ortho_dir_4d_next,
    rmsprop_momentum_step, Vec5,
};
use crate::math::{
    fast_dot, fast_saxpy, fast_scale, inv_sqrt_newton_f32, poly_repulsion_grad_coeff,
    rand_signed_f32, rand_unit_f32, rational_pos_and_deriv, rational_probs_and_psi_into,
    rational_probs_p4_into, splitmix64, sqrt_newton_f32,
};

/// Выучиваемые параметры пластичной полосы (хранятся внутри каждого `RegimeBank`, ~1.0–1.5 KB).
#[derive(Clone)]
pub struct PlasticCayleyParams {
    pub sectors: usize,
    pub max_in: usize,
    pub alpha: Vec<f32>,   // [sectors] -> lambda_s = 0.995 / (1 + alpha^2)
    pub omega: Vec<f32>,   // [sectors] -> R(omega * e_s)
    pub axis: Vec<f32>,    // [sectors * 3] единичные оси вращения e_s
    pub in_dir: Vec<f32>,  // [sectors * 3] единичные направления ввода d_s (перпендикулярны e_s!)
    pub in_w: Vec<f32>,    // [sectors * max_in] входные веса w_{s,j}
    // Алгебраический поэлементный RMS-масштаб (v <- 0.98 v + 0.02 g^2)
    pub v_alpha: Vec<f32>, // [sectors]
    pub v_omega: Vec<f32>, // [sectors]
    pub v_w: Vec<f32>,     // [sectors * max_in]
}

#[axiom_zero]
impl PlasticCayleyParams {
    pub fn new(sectors: usize, max_in: usize, seed: u64) -> Self {
        Self::new_with_grid(sectors, max_in, seed, 0)
    }

    /// Инициализация полосы `PlasticCayleyParams` с выбором частотно-добротной сетки:
    /// - `band_mode == 0`: сопряжённый мультимасштабный спектр (совместимость со Спринтом 2);
    /// - `band_mode == 1`: развязанная сетка `(частота x добротность Q)` — быстрые/полноспектральные волны;
    /// - `band_mode == 2`: развязанная высокодобротная сетка медленных огибающих (Band 2).
    /// При `sectors == 0` не выделяет память (`0` байт).
    pub fn new_with_grid(sectors: usize, max_in: usize, seed: u64, band_mode: u8) -> Self {
        let s_cnt = sectors.min(512);
        let m_in = max_in.clamp(1, 256);
        let mut p = Self {
            sectors: s_cnt,
            max_in: m_in,
            alpha: vec![0.4_f32; s_cnt],
            omega: vec![0.3_f32; s_cnt],
            axis: vec![0.0_f32; s_cnt.saturating_mul(3)],
            in_dir: vec![0.0_f32; s_cnt.saturating_mul(3)],
            in_w: vec![0.0_f32; s_cnt.saturating_mul(m_in)],
            v_alpha: vec![0.04_f32; s_cnt],
            v_omega: vec![0.04_f32; s_cnt],
            v_w: vec![0.04_f32; s_cnt.saturating_mul(m_in)],
        };
        p.reset_with_grid(seed, band_mode);
        p
    }

    /// Переинициализация параметров полосы на месте без аллокаций на куче.
    pub fn reset_with_grid(&mut self, seed: u64, band_mode: u8) {
        let s_cnt = self.sectors;
        let m_in = self.max_in;
        self.v_alpha.fill(0.04);
        self.v_omega.fill(0.04);
        self.v_w.fill(0.04);

        let mut rng = splitmix64(seed ^ 0x2721_CA71_E700_0001);

        for s in 0..s_cnt {
            let frac = total_div_f32(s as f32, s_cnt.max(1) as f32, 0.5);
            let sign_w = if (s & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
            let q_band = total_rem_usize(s, 3, 0) as f32;
            let (a0, w0) = match band_mode {
                1 => (
                    sanitize_f32(0.18 + 0.24 * q_band),
                    sanitize_f32(sign_w * (0.32 + 2.15 * frac)),
                ),
                2 => (
                    sanitize_f32(0.10 + 0.14 * q_band),
                    sanitize_f32(sign_w * (0.03 + 0.62 * frac)),
                ),
                _ => (
                    sanitize_f32(0.14 + 1.40 * frac * frac),
                    sanitize_f32(sign_w * (0.14 + 0.86 * frac)),
                ),
            };
            let _ = total_set(&mut self.alpha, s, a0);
            let _ = total_set(&mut self.omega, s, w0);

            // Единичная ось e_s
            let ax = rand_signed_f32(&mut rng);
            let ay = rand_signed_f32(&mut rng);
            let az = rand_signed_f32(&mut rng);
            let inv_an = inv_sqrt_newton_f32((ax * ax + ay * ay + az * az).max(1e-10));
            let ex = sanitize_f32(ax * inv_an);
            let ey = sanitize_f32(ay * inv_an);
            let ez = sanitize_f32(az * inv_an);

            // Единичный вектор ввода d_s, ортогональный оси e_s (чтобы ротор реально вращал вход!)
            let rx = rand_signed_f32(&mut rng);
            let ry = rand_signed_f32(&mut rng);
            let rz = rand_signed_f32(&mut rng);
            let dot_re = rx * ex + ry * ey + rz * ez;
            let mut dx = rx - dot_re * ex;
            let mut dy = ry - dot_re * ey;
            let mut dz = rz - dot_re * ez;
            let mut dn2 = dx * dx + dy * dy + dz * dz;
            if dn2 < 1e-8 {
                dx = -ey;
                dy = ex;
                dz = 0.0;
                dn2 = (dx * dx + dy * dy).max(1e-10);
            }
            let inv_dn = inv_sqrt_newton_f32(dn2);
            let off = s.saturating_mul(3);
            let _ = total_set(&mut self.axis, off, sanitize_f32(ex));
            let _ = total_set(&mut self.axis, off + 1, sanitize_f32(ey));
            let _ = total_set(&mut self.axis, off + 2, sanitize_f32(ez));
            let _ = total_set(&mut self.in_dir, off, sanitize_f32(dx * inv_dn));
            let _ = total_set(&mut self.in_dir, off + 1, sanitize_f32(dy * inv_dn));
            let _ = total_set(&mut self.in_dir, off + 2, sanitize_f32(dz * inv_dn));

            let woff = s.saturating_mul(m_in);
            for j in 0..m_in {
                let wv = sanitize_f32(rand_signed_f32(&mut rng) * 0.65);
                let _ = total_set(&mut self.in_w, woff + j, wv);
            }
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        (self.alpha.len()
            + self.omega.len()
            + self.axis.len()
            + self.in_dir.len()
            + self.in_w.len()
            + self.v_alpha.len()
            + self.v_omega.len()
            + self.v_w.len())
        .saturating_mul(4)
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.sectors.saturating_mul(3)
    }

    /// Сканирует и восстанавливает параметры весов `(alpha, omega, in_w)` при сбоях IEEE-754.
    pub fn scrub_weight_seu(&mut self) -> u64 {
        let mut faults = 0u64;
        for s in 0..self.sectors {
            let a = total_get_copy(&self.alpha, s, 0.4);
            if !a.is_finite() || a < 0.05 || a > 3.5 {
                faults = faults.saturating_add(1);
                let _ = total_set(&mut self.alpha, s, 0.35);
            }
            let w = total_get_copy(&self.omega, s, 0.3);
            if !w.is_finite() || w.abs() > 3.6 {
                faults = faults.saturating_add(1);
                let _ = total_set(&mut self.omega, s, 0.30);
            }
        }
        for i in 0..self.in_w.len() {
            let wi = total_get_copy(&self.in_w, i, 0.0);
            if !wi.is_finite() || wi.abs() > 3.1 {
                faults = faults.saturating_add(1);
                let _ = total_set(&mut self.in_w, i, 0.0);
            }
        }
        faults
    }

    /// Аппаратная инъекция битового переворота IEEE-754 в регистры весов `(alpha, omega, in_w)`.
    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let mask = 1u32 << (bit & 31);
        let sel = total_rem_usize(reg_idx, 3, 0);
        let target: &mut [f32] = match sel {
            0 => &mut self.alpha,
            1 => &mut self.omega,
            _ => &mut self.in_w,
        };
        if target.is_empty() {
            return;
        }
        let idx = total_rem_usize(reg_idx, target.len(), 0);
        let mut cur = total_get_copy(target, idx, 1.25);
        if cur.is_finite() && (cur.abs() < 0.25 || cur.abs() >= 1.9) {
            cur = if cur < 0.0 { -1.25 } else { 1.25 };
        }
        let _ = total_set(target, idx, f32::from_bits(cur.to_bits() ^ mask));
    }
}

/// Вспомогательный 3D рациональный поворот Кэли вокруг единичной оси e:
/// R(w * e) v = v + (2w / (1+w^2)) (e x v) + (2w^2 / (1+w^2)) (e x (e x v))
#[axiom_zero]
#[inline]
fn cayley_rot3(ex: f32, ey: f32, ez: f32, w: f32, vx: f32, vy: f32, vz: f32) -> (f32, f32, f32) {
    let qx = ey * vz - ez * vy;
    let qy = ez * vx - ex * vz;
    let qz = ex * vy - ey * vx;

    let rx = ey * qz - ez * qy;
    let ry = ez * qx - ex * qz;
    let rz = ex * qy - ey * qx;

    let w2 = w * w;
    let c1 = total_div_unit_shifted_f32(2.0 * w, w2);
    let c2 = total_div_unit_shifted_f32(2.0 * w2, w2);

    (
        sanitize_f32(vx + c1 * qx + c2 * rx),
        sanitize_f32(vy + c1 * qy + c2 * ry),
        sanitize_f32(vz + c1 * qz + c2 * rz),
    )
}

/// Runtime-состояние и точные векторы чувствительности RTRL (общий scratch ядра, ~2–3 KB).
#[derive(Clone)]
pub struct PlasticCayleyState {
    pub sectors: usize,
    pub max_in: usize,
    pub kappa: f32,
    pub u: Vec<f32>,       // [sectors * 3] линейное внутреннее состояние
    pub b: Vec<f32>,       // [sectors * 3] нормализованное чтение Norm_kappa(u)
    pub rho: Vec<f32>,     // [sectors] sqrt(||u_s||^2 + kappa^2)
    pub p_alpha: Vec<f32>, // [sectors * 3] du_s / d(alpha_s)
    pub p_omega: Vec<f32>, // [sectors * 3] du_s / d(omega_s)
    pub p_w: Vec<f32>,     // [sectors * max_in * 3] du_s / d(w_{s,j})
    pub e_alpha: Vec<f32>, // [sectors] 3-факторный след приемлемости для alpha_s
    pub e_omega: Vec<f32>, // [sectors] 3-факторный след приемлемости для omega_s
    pub e_w: Vec<f32>,     // [sectors * max_in] след приемлемости для w_{s,j}
    pub seu_fault_count: u64,
}

#[axiom_zero]
impl PlasticCayleyState {
    pub fn new(sectors: usize, max_in: usize, kappa: f32) -> Self {
        let s_cnt = sectors.min(512);
        let m_in = max_in.clamp(1, 256);
        let d = s_cnt.saturating_mul(3);
        let sm = s_cnt.saturating_mul(m_in);
        Self {
            sectors: s_cnt,
            max_in: m_in,
            kappa: sanitize_f32(kappa).clamp(0.1, 2.0),
            u: vec![0.0_f32; d],
            b: vec![0.0_f32; d],
            rho: vec![kappa.clamp(0.1, 2.0); s_cnt],
            p_alpha: vec![0.0_f32; d],
            p_omega: vec![0.0_f32; d],
            p_w: vec![0.0_f32; sm.saturating_mul(3)],
            e_alpha: vec![0.0_f32; s_cnt],
            e_omega: vec![0.0_f32; s_cnt],
            e_w: vec![0.0_f32; sm],
            seu_fault_count: 0,
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        (self.u.len()
            + self.b.len()
            + self.rho.len()
            + self.p_alpha.len()
            + self.p_omega.len()
            + self.p_w.len()
            + self.e_alpha.len()
            + self.e_omega.len()
            + self.e_w.len())
        .saturating_mul(4)
    }

    pub fn reset(&mut self) {
        self.u.fill(0.0);
        self.b.fill(0.0);
        self.rho.fill(self.kappa);
        self.p_alpha.fill(0.0);
        self.p_omega.fill(0.0);
        self.p_w.fill(0.0);
        self.e_alpha.fill(0.0);
        self.e_omega.fill(0.0);
        self.e_w.fill(0.0);
    }

    /// Аппаратная/битовая инъекция SEU прямо в IEEE-754 регистр состояния `u[reg_idx]`.
    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        if self.u.is_empty() {
            return;
        }
        let idx = total_rem_usize(reg_idx, self.u.len(), 0);
        let cur = total_get_copy(&self.u, idx, 0.0);
        let flipped = f32::from_bits(cur.to_bits() ^ (1u32 << (bit & 31)));
        let _ = total_set(&mut self.u, idx, flipped);
    }

    /// Чисто чтение-ориентированный такт состояния `(u_buf, b_buf)` без мутации `&self` и без вычисления
    /// матриц чувствительности RTRL (для каузального $K$-шагового прогноза без утечки).
    pub fn step_state_readonly(
        params: &PlasticCayleyParams,
        kappa: f32,
        max_in: usize,
        raw_x: &[f32],
        u_buf: &mut [f32],
        b_buf: &mut [f32],
    ) {
        let s_cnt = params
            .sectors
            .min(total_div_usize(u_buf.len(), 3, 0))
            .min(total_div_usize(b_buf.len(), 3, 0));
        let m_in = max_in.min(params.max_in);
        let x_len = raw_x.len().min(m_in);
        let kap2 = kappa * kappa;

        for s in 0..s_cnt {
            let off = s.saturating_mul(3);
            let a = total_get_copy(&params.alpha, s, 0.4);
            let w = total_get_copy(&params.omega, s, 0.3);

            let a2 = a * a;
            let inv_a = total_div_unit_shifted_f32(1.0, a2);
            let lam = sanitize_f32(0.995 * inv_a);

            let w2 = w * w;
            let inv_w = total_div_unit_shifted_f32(1.0, w2);

            let ex = total_get_copy(&params.axis, off, 1.0);
            let ey = total_get_copy(&params.axis, off + 1, 0.0);
            let ez = total_get_copy(&params.axis, off + 2, 0.0);

            let dx = total_get_copy(&params.in_dir, off, 0.0);
            let dy = total_get_copy(&params.in_dir, off + 1, 1.0);
            let dz = total_get_copy(&params.in_dir, off + 2, 0.0);

            let ux = sanitize_bounded_f32(total_get_copy(u_buf, off, 0.0), 24.0);
            let uy = sanitize_bounded_f32(total_get_copy(u_buf, off + 1, 0.0), 24.0);
            let uz = sanitize_bounded_f32(total_get_copy(u_buf, off + 2, 0.0), 24.0);

            let qx = ey * uz - ez * uy;
            let qy = ez * ux - ex * uz;
            let qz = ex * uy - ey * ux;

            let rx = ey * qz - ez * qy;
            let ry = ez * qx - ex * qz;
            let rz = ex * qy - ey * qx;

            let c1 = sanitize_f32(2.0 * w * inv_w);
            let c2 = sanitize_f32(2.0 * w2 * inv_w);
            let rux = sanitize_f32(ux + c1 * qx + c2 * rx);
            let ruy = sanitize_f32(uy + c1 * qy + c2 * ry);
            let ruz = sanitize_f32(uz + c1 * qz + c2 * rz);

            let woff = s.saturating_mul(m_in);
            let mut drive = 0.0_f32;
            for j in 0..x_len {
                let xj = sanitize_bounded_f32(total_get_copy(raw_x, j, 0.0), 8.0);
                let wj = total_get_copy(&params.in_w, woff + j, 0.0);
                drive = sanitize_f32(drive + wj * xj);
            }

            let nux = sanitize_bounded_f32(lam * rux + drive * dx, 24.0);
            let nuy = sanitize_bounded_f32(lam * ruy + drive * dy, 24.0);
            let nuz = sanitize_bounded_f32(lam * ruz + drive * dz, 24.0);
            let _ = total_set(u_buf, off, nux);
            let _ = total_set(u_buf, off + 1, nuy);
            let _ = total_set(u_buf, off + 2, nuz);

            let rho = sanitize_f32(sqrt_newton_f32(nux * nux + nuy * nuy + nuz * nuz + kap2)).max(kappa);
            let inv_rho = total_div_f32(1.0, rho, 1.0);
            let _ = total_set(b_buf, off, sanitize_f32(nux * inv_rho));
            let _ = total_set(b_buf, off + 1, sanitize_f32(nuy * inv_rho));
            let _ = total_set(b_buf, off + 2, sanitize_f32(nuz * inv_rho));
        }
    }

    /// Один прямой такт линейного ротора Кэли + точное обновление всех чувствительностей RTRL.
    pub fn step_forward(&mut self, params: &PlasticCayleyParams, raw_x: &[f32]) {
        let s_cnt = self.sectors.min(params.sectors);
        let m_in = self.max_in.min(params.max_in);
        let x_len = raw_x.len().min(m_in);
        let kap2 = self.kappa * self.kappa;

        for j in 0..x_len {
            let rx = total_get_copy(raw_x, j, 0.0);
            if !rx.is_finite() || rx.abs() > 8.0 {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
        }

        for s in 0..s_cnt {
            let off = s.saturating_mul(3);
            let a = total_get_copy(&params.alpha, s, 0.4);
            let w = total_get_copy(&params.omega, s, 0.3);

            let a2 = a * a;
            let inv_a = total_div_unit_shifted_f32(1.0, a2);
            let lam = sanitize_f32(0.995 * inv_a);
            let dlam_da = sanitize_f32(-1.99 * a * inv_a * inv_a);

            let w2 = w * w;
            let inv_w = total_div_unit_shifted_f32(1.0, w2);
            let inv_w2 = sanitize_f32(inv_w * inv_w);
            let dc1_dw = sanitize_f32(2.0 * (1.0 - w2) * inv_w2);
            let dc2_dw = sanitize_f32(4.0 * w * inv_w2);

            let ex = total_get_copy(&params.axis, off, 1.0);
            let ey = total_get_copy(&params.axis, off + 1, 0.0);
            let ez = total_get_copy(&params.axis, off + 2, 0.0);

            let dx = total_get_copy(&params.in_dir, off, 0.0);
            let dy = total_get_copy(&params.in_dir, off + 1, 1.0);
            let dz = total_get_copy(&params.in_dir, off + 2, 0.0);

            let raw_ux = total_get_copy(&self.u, off, 0.0);
            let raw_uy = total_get_copy(&self.u, off + 1, 0.0);
            let raw_uz = total_get_copy(&self.u, off + 2, 0.0);
            if !raw_ux.is_finite()
                || raw_ux.abs() > 24.0
                || !raw_uy.is_finite()
                || raw_uy.abs() > 24.0
                || !raw_uz.is_finite()
                || raw_uz.abs() > 24.0
            {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            let ux = sanitize_bounded_f32(raw_ux, 24.0);
            let uy = sanitize_bounded_f32(raw_uy, 24.0);
            let uz = sanitize_bounded_f32(raw_uz, 24.0);

            let qx = ey * uz - ez * uy;
            let qy = ez * ux - ex * uz;
            let qz = ex * uy - ey * ux;

            let rx = ey * qz - ez * qy;
            let ry = ez * qx - ex * qz;
            let rz = ex * qy - ey * qx;

            let c1 = sanitize_f32(2.0 * w * inv_w);
            let c2 = sanitize_f32(2.0 * w2 * inv_w);
            let rux = sanitize_f32(ux + c1 * qx + c2 * rx);
            let ruy = sanitize_f32(uy + c1 * qy + c2 * ry);
            let ruz = sanitize_f32(uz + c1 * qz + c2 * rz);

            let woff = s.saturating_mul(m_in);
            let mut drive = 0.0_f32;
            for j in 0..x_len {
                let xj = sanitize_bounded_f32(total_get_copy(raw_x, j, 0.0), 8.0);
                let wj = total_get_copy(&params.in_w, woff + j, 0.0);
                drive = sanitize_f32(drive + wj * xj);
            }

            let nux = sanitize_bounded_f32(lam * rux + drive * dx, 24.0);
            let nuy = sanitize_bounded_f32(lam * ruy + drive * dy, 24.0);
            let nuz = sanitize_bounded_f32(lam * ruz + drive * dz, 24.0);
            let _ = total_set(&mut self.u, off, nux);
            let _ = total_set(&mut self.u, off + 1, nuy);
            let _ = total_set(&mut self.u, off + 2, nuz);

            let rho = sanitize_f32(sqrt_newton_f32(nux * nux + nuy * nuy + nuz * nuz + kap2)).max(self.kappa);
            let inv_rho = total_div_f32(1.0, rho, 1.0);
            let _ = total_set(&mut self.rho, s, rho);
            let _ = total_set(&mut self.b, off, sanitize_f32(nux * inv_rho));
            let _ = total_set(&mut self.b, off + 1, sanitize_f32(nuy * inv_rho));
            let _ = total_set(&mut self.b, off + 2, sanitize_f32(nuz * inv_rho));

            let pax = total_get_copy(&self.p_alpha, off, 0.0);
            let pay = total_get_copy(&self.p_alpha, off + 1, 0.0);
            let paz = total_get_copy(&self.p_alpha, off + 2, 0.0);
            let (rpax, rpay, rpaz) = cayley_rot3(ex, ey, ez, w, pax, pay, paz);
            let _ = total_set(&mut self.p_alpha, off, sanitize_bounded_f32(lam * rpax + dlam_da * rux, 32.0));
            let _ = total_set(&mut self.p_alpha, off + 1, sanitize_bounded_f32(lam * rpay + dlam_da * ruy, 32.0));
            let _ = total_set(&mut self.p_alpha, off + 2, sanitize_bounded_f32(lam * rpaz + dlam_da * ruz, 32.0));

            let pwx = total_get_copy(&self.p_omega, off, 0.0);
            let pwy = total_get_copy(&self.p_omega, off + 1, 0.0);
            let pwz = total_get_copy(&self.p_omega, off + 2, 0.0);
            let (rpwx, rpwy, rpwz) = cayley_rot3(ex, ey, ez, w, pwx, pwy, pwz);
            let drx = sanitize_f32(dc1_dw * qx + dc2_dw * rx);
            let dry = sanitize_f32(dc1_dw * qy + dc2_dw * ry);
            let drz = sanitize_f32(dc1_dw * qz + dc2_dw * rz);
            let _ = total_set(&mut self.p_omega, off, sanitize_bounded_f32(lam * (rpwx + drx), 32.0));
            let _ = total_set(&mut self.p_omega, off + 1, sanitize_bounded_f32(lam * (rpwy + dry), 32.0));
            let _ = total_set(&mut self.p_omega, off + 2, sanitize_bounded_f32(lam * (rpwz + drz), 32.0));

            for j in 0..x_len {
                let xj = sanitize_bounded_f32(total_get_copy(raw_x, j, 0.0), 8.0);
                let poff = (woff + j).saturating_mul(3);
                let pjx = total_get_copy(&self.p_w, poff, 0.0);
                let pjy = total_get_copy(&self.p_w, poff + 1, 0.0);
                let pjz = total_get_copy(&self.p_w, poff + 2, 0.0);
                let (rpjx, rpjy, rpjz) = cayley_rot3(ex, ey, ez, w, pjx, pjy, pjz);
                let _ = total_set(&mut self.p_w, poff, sanitize_bounded_f32(lam * rpjx + xj * dx, 32.0));
                let _ = total_set(&mut self.p_w, poff + 1, sanitize_bounded_f32(lam * rpjy + xj * dy, 32.0));
                let _ = total_set(&mut self.p_w, poff + 2, sanitize_bounded_f32(lam * rpjz + xj * dz, 32.0));
            }
        }
    }

    /// Накапливает 3-факторные следы приемлемости для `(alpha_s, omega_s, w_{s,j})`
    /// через точный якобиан `Norm_kappa`.
    pub fn accumulate_traces(&mut self, grad_b: &[f32], decay: f32, active_in: usize) {
        let s_cnt = self.sectors;
        let m_in = self.max_in;
        let x_len = active_in.min(m_in);
        let dec = sanitize_f32(decay).clamp(0.0, 0.99);

        for s in 0..s_cnt {
            let off = s.saturating_mul(3);
            let gx = total_get_copy(grad_b, off, 0.0);
            let gy = total_get_copy(grad_b, off + 1, 0.0);
            let gz = total_get_copy(grad_b, off + 2, 0.0);

            let bx = total_get_copy(&self.b, off, 0.0);
            let by = total_get_copy(&self.b, off + 1, 0.0);
            let bz = total_get_copy(&self.b, off + 2, 0.0);

            let gb = sanitize_f32(gx * bx + gy * by + gz * bz);
            let rho = total_get_copy(&self.rho, s, self.kappa).max(self.kappa);
            let inv_rho = total_div_f32(1.0, rho, 1.0);
            let fbx = sanitize_f32((gx - gb * bx) * inv_rho);
            let fby = sanitize_f32((gy - gb * by) * inv_rho);
            let fbz = sanitize_f32((gz - gb * bz) * inv_rho);

            let da = fbx * total_get_copy(&self.p_alpha, off, 0.0)
                + fby * total_get_copy(&self.p_alpha, off + 1, 0.0)
                + fbz * total_get_copy(&self.p_alpha, off + 2, 0.0);
            let old_ea = total_get_copy(&self.e_alpha, s, 0.0);
            let _ = total_set(&mut self.e_alpha, s, sanitize_bounded_f32(dec * old_ea + da, 16.0));

            let dw = fbx * total_get_copy(&self.p_omega, off, 0.0)
                + fby * total_get_copy(&self.p_omega, off + 1, 0.0)
                + fbz * total_get_copy(&self.p_omega, off + 2, 0.0);
            let old_ew = total_get_copy(&self.e_omega, s, 0.0);
            let _ = total_set(&mut self.e_omega, s, sanitize_bounded_f32(dec * old_ew + dw, 16.0));

            let woff = s.saturating_mul(m_in);
            for j in 0..x_len {
                let poff = (woff + j).saturating_mul(3);
                let dj = fbx * total_get_copy(&self.p_w, poff, 0.0)
                    + fby * total_get_copy(&self.p_w, poff + 1, 0.0)
                    + fbz * total_get_copy(&self.p_w, poff + 2, 0.0);
                let old_ej = total_get_copy(&self.e_w, woff + j, 0.0);
                let _ = total_set(&mut self.e_w, woff + j, sanitize_bounded_f32(dec * old_ej + dj, 16.0));
            }
        }
    }

    /// 3-факторное онлайн-обновление `(alpha_s, omega_s, w_{s,j})`
    /// с поэлементным алгебраическим RMS-масштабом по градиенту: v <- 0.98 v + 0.02 g^2.
    pub fn apply_td_update(
        &self,
        params: &mut PlasticCayleyParams,
        td_err: f32,
        lr: f32,
        active_in: usize,
    ) {
        let s_cnt = self.sectors.min(params.sectors);
        let m_in = self.max_in.min(params.max_in);
        let x_len = active_in.min(m_in);
        let td = sanitize_bounded_f32(td_err, 2.5);
        if td == 0.0 || lr <= 0.0 {
            return;
        }

        for s in 0..s_cnt {
            let old_a = total_get_copy(&params.alpha, s, 0.4);
            let ga = sanitize_f32(td * total_get_copy(&self.e_alpha, s, 0.0));
            let old_va = total_get_copy(&params.v_alpha, s, 0.04);
            let va = sanitize_f32(0.98 * old_va + 0.02 * ga * ga).max(0.0);
            let _ = total_set(&mut params.v_alpha, s, va);
            let step_a = sanitize_bounded_f32(sanitize_f32(ga * inv_sqrt_newton_f32(va + 0.0025)), 3.0);
            let new_a = sanitize_f32(old_a + lr * step_a).clamp(0.08, 3.0);
            let _ = total_set(&mut params.alpha, s, new_a);

            let old_w = total_get_copy(&params.omega, s, 0.3);
            let gw = sanitize_f32(td * total_get_copy(&self.e_omega, s, 0.0));
            let old_vw = total_get_copy(&params.v_omega, s, 0.04);
            let vw = sanitize_f32(0.98 * old_vw + 0.02 * gw * gw).max(0.0);
            let _ = total_set(&mut params.v_omega, s, vw);
            let step_w = sanitize_bounded_f32(sanitize_f32(gw * inv_sqrt_newton_f32(vw + 0.0025)), 3.0);
            let new_w = sanitize_f32(old_w + lr * step_w).clamp(-3.5, 3.5);
            let _ = total_set(&mut params.omega, s, new_w);

            let woff = s.saturating_mul(m_in);
            for j in 0..x_len {
                let old_wj = total_get_copy(&params.in_w, woff + j, 0.0);
                let gj = sanitize_f32(td * total_get_copy(&self.e_w, woff + j, 0.0));
                let old_vj = total_get_copy(&params.v_w, woff + j, 0.04);
                let vj = sanitize_f32(0.98 * old_vj + 0.02 * gj * gj).max(0.0);
                let _ = total_set(&mut params.v_w, woff + j, vj);
                let step_j = sanitize_bounded_f32(sanitize_f32(gj * inv_sqrt_newton_f32(vj + 0.0025)), 3.0);
                let new_wj = sanitize_f32(old_wj + lr * step_j).clamp(-3.0, 3.0);
                let _ = total_set(&mut params.in_w, woff + j, new_wj);
            }
        }
    }
}

/// Автономная 2-слойная обёртка Fast-Slow Cayley-RTRL (< 15 KB RAM, 100% #[axiom_zero], 0% трансцендентных).
/// Полноценный вариант с банками `PageHinkley` и опциональной замороженной полосой также встроен
/// напрямую в `MicroContinualCore` (`rtrl_slow_sectors > 0`).
#[derive(Clone)]
pub struct HierarchicalCayleyRtrl {
    pub in_dim: usize,
    pub num_actions: usize,
    pub fast_params: PlasticCayleyParams,
    pub fast_state: PlasticCayleyState,
    pub slow_params: PlasticCayleyParams,
    pub slow_state: PlasticCayleyState,
    pub slow_in_buf: Vec<f32>,
    pub pool_w: Vec<f32>,
    pub actor_w: Vec<f32>,
    pub actor_b: Vec<f32>,
    pub critic_w: Vec<f32>,
    pub critic_b: f32,
    feat_buf: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<f32>,
    psis: Vec<f32>,
    e_actor_w: Vec<f32>,
    e_actor_b: Vec<f32>,
    e_critic_w: Vec<f32>,
    e_critic_b: f32,
    fb_fast: Vec<f32>,
    fb_slow: Vec<f32>,
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
}

#[axiom_zero]
impl HierarchicalCayleyRtrl {
    pub fn new(
        in_dim: usize,
        fast_sectors: usize,
        slow_sectors: usize,
        num_actions: usize,
        seed: u64,
    ) -> Self {
        let d_in = in_dim.clamp(1, 256);
        let na = num_actions.clamp(2, 64);
        let s1 = fast_sectors.clamp(2, 512);
        let s2 = slow_sectors.clamp(2, 512);
        let slow_in = (d_in + 4).min(260);

        let fast_params = PlasticCayleyParams::new_with_grid(s1, d_in, seed ^ 0xFA57_0001, 1);
        let slow_params = PlasticCayleyParams::new_with_grid(s2, slow_in, seed ^ 0x510B_0002, 2);

        let fast_state = PlasticCayleyState::new(s1, d_in, 0.65);
        let slow_state = PlasticCayleyState::new(s2, slow_in, 0.65);

        let fdim1 = fast_params.dim();
        let fdim2 = slow_params.dim();
        let feat_dim = fdim1 + fdim2 + d_in;

        let mut rng = splitmix64(seed ^ 0x41E2_CA71_E700);
        let mut pool_w = vec![0.0_f32; 4usize.saturating_mul(fdim1)];
        let p_scale = inv_sqrt_newton_f32(fdim1.max(1) as f32);
        for v in pool_w.iter_mut() {
            *v = sanitize_f32(rand_signed_f32(&mut rng) * p_scale);
        }

        let mut actor_w = vec![0.0_f32; na.saturating_mul(feat_dim)];
        let mut critic_w = vec![0.0_f32; feat_dim];
        for v in actor_w.iter_mut() {
            *v = sanitize_f32(rand_signed_f32(&mut rng) * 0.05);
        }
        for v in critic_w.iter_mut() {
            *v = sanitize_f32(rand_signed_f32(&mut rng) * 0.05);
        }

        Self {
            in_dim: d_in,
            num_actions: na,
            fast_params,
            fast_state,
            slow_params,
            slow_state,
            slow_in_buf: vec![0.0_f32; slow_in],
            pool_w,
            actor_w,
            actor_b: vec![0.0_f32; na],
            critic_w,
            critic_b: 0.0,
            feat_buf: vec![0.0_f32; feat_dim],
            logits: vec![0.0_f32; na],
            probs: vec![0.0_f32; na],
            psis: vec![1.0_f32; na],
            e_actor_w: vec![0.0_f32; na.saturating_mul(feat_dim)],
            e_actor_b: vec![0.0_f32; na],
            e_critic_w: vec![0.0_f32; feat_dim],
            e_critic_b: 0.0,
            fb_fast: vec![0.0_f32; fdim1],
            fb_slow: vec![0.0_f32; fdim2],
            has_pending: false,
            pending_v: 0.0,
            pending_r: 0.0,
            td_mean: 0.0,
            td_var: 0.25,
            gamma: 0.0,
            trace_lambda: 0.0,
            actor_lr: 0.09,
            critic_lr: 0.16,
            rtrl_lr: 0.010,
            entropy_beta: 0.004,
            rng,
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.fast_params.memory_footprint_bytes()
            + self.fast_state.memory_footprint_bytes()
            + self.slow_params.memory_footprint_bytes()
            + self.slow_state.memory_footprint_bytes()
            + (self.slow_in_buf.len()
                + self.pool_w.len()
                + self.actor_w.len()
                + self.actor_b.len()
                + self.critic_w.len()
                + self.feat_buf.len()
                + self.logits.len()
                + self.probs.len()
                + self.psis.len()
                + self.e_actor_w.len()
                + self.e_actor_b.len()
                + self.e_critic_w.len()
                + self.fb_fast.len()
                + self.fb_slow.len())
            .saturating_mul(4)
            + 96
    }

    pub fn start_episode(&mut self) {
        self.fast_state.reset();
        self.slow_state.reset();
        self.e_actor_w.fill(0.0);
        self.e_actor_b.fill(0.0);
        self.e_critic_w.fill(0.0);
        self.e_critic_b = 0.0;
        self.has_pending = false;
        self.pending_v = 0.0;
        self.pending_r = 0.0;
    }

    fn apply_td_update(&mut self, td_err: f32) {
        let raw_d = sanitize_bounded_f32(td_err, 6.0);
        let crit_err = sanitize_bounded_f32(raw_d, 3.0);
        let centered = sanitize_f32(raw_d - self.td_mean);
        self.td_mean = sanitize_f32(0.99 * self.td_mean + 0.01 * raw_d);
        self.td_var = sanitize_f32(0.98 * self.td_var + 0.02 * centered * centered).max(0.08);
        let adv = sanitize_f32(centered * inv_sqrt_newton_f32(self.td_var)).clamp(-2.5, 2.5);

        let fdim = self.feat_buf.len();
        fast_saxpy(&mut self.critic_w, &self.e_critic_w, self.critic_lr * crit_err);
        let cn = fast_dot(&self.critic_w, &self.critic_w);
        if cn > 256.0 {
            fast_scale(&mut self.critic_w, sanitize_f32(16.0 * inv_sqrt_newton_f32(cn)));
        }
        self.critic_b = sanitize_bounded_f32(
            self.critic_b + 0.25 * self.critic_lr * crit_err * self.e_critic_b,
            25.0,
        );

        for a in 0..self.num_actions {
            let off = a.saturating_mul(fdim);
            if let (Some(aw), Some(ew)) = (
                self.actor_w.get_mut(off..off + fdim),
                self.e_actor_w.get(off..off + fdim),
            ) {
                fast_scale(aw, 0.9999);
                fast_saxpy(aw, ew, self.actor_lr * adv);
                let an = fast_dot(aw, aw);
                if an > 36.0 {
                    fast_scale(aw, sanitize_f32(6.0 * inv_sqrt_newton_f32(an)));
                }
            }
            let ob = total_get_copy(&self.actor_b, a, 0.0);
            let eb = total_get_copy(&self.e_actor_b, a, 0.0);
            let _ = total_set(
                &mut self.actor_b,
                a,
                sanitize_bounded_f32(ob * 0.9999 + 0.25 * self.actor_lr * adv * eb, 3.0),
            );
        }

        if self.rtrl_lr > 0.0 {
            self.fast_state
                .apply_td_update(&mut self.fast_params, adv, self.rtrl_lr, self.in_dim);
            self.slow_state.apply_td_update(
                &mut self.slow_params,
                adv,
                self.rtrl_lr,
                self.slow_in_buf.len(),
            );
        }
    }

    pub fn decide_vec(&mut self, features: &[f32], train_mode: bool, seu_nan: bool, seu_spikes: bool) -> usize {
        let d_in = self.in_dim;
        for i in 0..d_in {
            let mut v = sanitize_bounded_f32(total_get_copy(features, i, 0.0), 8.0);
            if seu_nan && i == 0 {
                v = f32::NAN;
            }
            if seu_spikes && i == 1 {
                v = 1e28_f32;
            }
            let _ = total_set(&mut self.slow_in_buf, i, sanitize_bounded_f32(v, 8.0));
        }
        {
            let x_slice = self.slow_in_buf.get(0..d_in).unwrap_or(&[]);
            self.fast_state.step_forward(&self.fast_params, x_slice);
        }
        let fdim1 = self.fast_state.b.len();
        let slow_len = self.slow_in_buf.len();
        for c in 0..4usize {
            if d_in + c < slow_len {
                let poff = c.saturating_mul(fdim1);
                let pw = self.pool_w.get(poff..poff + fdim1).unwrap_or(&[]);
                let proj = sanitize_bounded_f32(fast_dot(pw, &self.fast_state.b), 4.0);
                let _ = total_set(&mut self.slow_in_buf, d_in + c, proj);
            }
        }
        self.slow_state.step_forward(&self.slow_params, &self.slow_in_buf);

        let fdim2 = self.slow_state.b.len();
        let sc1 = inv_sqrt_newton_f32(self.fast_state.sectors.max(1) as f32);
        let sc2 = inv_sqrt_newton_f32(self.slow_state.sectors.max(1) as f32);
        for i in 0..fdim1 {
            let _ = total_set(&mut self.feat_buf, i, sanitize_f32(total_get_copy(&self.fast_state.b, i, 0.0) * sc1));
        }
        for i in 0..fdim2 {
            let _ = total_set(&mut self.feat_buf, fdim1 + i, sanitize_f32(total_get_copy(&self.slow_state.b, i, 0.0) * sc2));
        }
        for i in 0..d_in {
            let _ = total_set(
                &mut self.feat_buf,
                fdim1 + fdim2 + i,
                sanitize_f32(total_get_copy(&self.slow_in_buf, i, 0.0) * 0.5),
            );
        }

        let v_cur = sanitize_f32(self.critic_b + fast_dot(&self.critic_w, &self.feat_buf));
        if train_mode && self.has_pending {
            let td_err = sanitize_f32(self.pending_r + self.gamma * v_cur - self.pending_v);
            self.apply_td_update(td_err);
        }

        let na = self.num_actions;
        let fdim = self.feat_buf.len();
        for a in 0..na {
            let off = a.saturating_mul(fdim);
            let aw = self.actor_w.get(off..off + fdim).unwrap_or(&[]);
            let lg = sanitize_f32(fast_dot(aw, &self.feat_buf) + total_get_copy(&self.actor_b, a, 0.0));
            let _ = total_set(&mut self.logits, a, lg);
        }
        rational_probs_and_psi_into(&self.logits, &mut self.probs, &mut self.psis);

        let best_a = if train_mode {
            let u = rand_unit_f32(&mut self.rng);
            let mut cdf = 0.0_f32;
            let mut picked = na - 1;
            for a in 0..na {
                cdf = sanitize_f32(cdf + total_get_copy(&self.probs, a, 0.0));
                if u <= cdf {
                    picked = a;
                    break;
                }
            }
            picked
        } else {
            let mut ba = 0usize;
            let mut bp = -1.0_f32;
            for a in 0..na {
                let p = total_get_copy(&self.probs, a, 0.0);
                if p > bp {
                    bp = p;
                    ba = a;
                }
            }
            ba
        };

        if train_mode {
            let decay = sanitize_f32(self.gamma * self.trace_lambda).clamp(0.0, 0.98);
            fast_scale(&mut self.e_critic_w, decay);
            fast_saxpy(&mut self.e_critic_w, &self.feat_buf, 1.0);
            let ecn = fast_dot(&self.e_critic_w, &self.e_critic_w);
            if ecn > 16.0 {
                fast_scale(&mut self.e_critic_w, sanitize_f32(4.0 * inv_sqrt_newton_f32(ecn)));
            }
            self.e_critic_b = sanitize_bounded_f32(decay * self.e_critic_b + 1.0, 5.0);

            let mut sum_s = 0.0_f32;
            let mut sum_p2 = 0.0_f32;
            for a in 0..na {
                let p = total_get_copy(&self.probs, a, 0.0).max(0.0);
                let psi = total_get_copy(&self.psis, a, 1.0);
                let ind = if a == best_a { 1.0_f32 } else { 0.0_f32 };
                sum_s = sanitize_f32(sum_s + psi * (ind - p));
                sum_p2 = sanitize_f32(sum_p2 + p * p);
            }
            let mean_s = total_div_f32(sum_s, na.max(1) as f32, 0.0);
            let mut sum_h = 0.0_f32;
            for a in 0..na {
                let p = total_get_copy(&self.probs, a, 0.0).max(0.0);
                let psi = total_get_copy(&self.psis, a, 1.0);
                sum_h = sanitize_f32(sum_h + p * psi * (sum_p2 - p));
            }
            let mean_h = total_div_f32(sum_h, na.max(1) as f32, 0.0);

            self.fb_fast.fill(0.0);
            self.fb_slow.fill(0.0);
            for a in 0..na {
                let p = total_get_copy(&self.probs, a, 0.0).max(0.0);
                let psi = total_get_copy(&self.psis, a, 1.0);
                let ind = if a == best_a { 1.0_f32 } else { 0.0_f32 };
                let score_a = sanitize_f32(psi * (ind - p) - mean_s);

                let off = a.saturating_mul(fdim);
                if let Some(ew) = self.e_actor_w.get_mut(off..off + fdim) {
                    fast_scale(ew, decay);
                    fast_saxpy(ew, &self.feat_buf, score_a);
                    let ean = fast_dot(ew, ew);
                    if ean > 16.0 {
                        fast_scale(ew, sanitize_f32(4.0 * inv_sqrt_newton_f32(ean)));
                    }
                }
                let ob = total_get_copy(&self.e_actor_b, a, 0.0);
                let _ = total_set(
                    &mut self.e_actor_b,
                    a,
                    sanitize_bounded_f32(decay * ob + score_a, 5.0),
                );
                if let Some(aw_f) = self.actor_w.get(off..off + fdim1) {
                    fast_saxpy(&mut self.fb_fast, aw_f, score_a * sc1);
                }
                if let Some(aw_s) = self.actor_w.get(off + fdim1..off + fdim1 + fdim2) {
                    fast_saxpy(&mut self.fb_slow, aw_s, score_a * sc2);
                }
            }
            if let Some(cw_f) = self.critic_w.get(0..fdim1) {
                fast_saxpy(&mut self.fb_fast, cw_f, 0.15 * sc1);
            }
            if let Some(cw_s) = self.critic_w.get(fdim1..fdim1 + fdim2) {
                fast_saxpy(&mut self.fb_slow, cw_s, 0.15 * sc2);
            }
            self.fast_state.accumulate_traces(&self.fb_fast, decay, d_in);
            self.slow_state.accumulate_traces(&self.fb_slow, decay, slow_len);

            if self.entropy_beta > 0.0 {
                for a in 0..na {
                    let p = total_get_copy(&self.probs, a, 0.0).max(0.0);
                    let psi = total_get_copy(&self.psis, a, 1.0);
                    let d_ent = sanitize_f32(self.entropy_beta * (p * psi * (sum_p2 - p) - mean_h));
                    let step = sanitize_f32(self.actor_lr * d_ent);
                    let off = a.saturating_mul(fdim);
                    if let Some(aw) = self.actor_w.get_mut(off..off + fdim) {
                        fast_saxpy(aw, &self.feat_buf, step);
                    }
                    let ob = total_get_copy(&self.actor_b, a, 0.0);
                    let _ = total_set(&mut self.actor_b, a, sanitize_bounded_f32(ob + 0.25 * step, 3.0));
                }
            }

            self.pending_v = sanitize_f32(self.critic_b + fast_dot(&self.critic_w, &self.feat_buf));
            self.pending_r = 0.0;
            self.has_pending = true;
        }
        best_a
    }

    #[inline]
    pub fn observe_step_reward(&mut self, r: f32) {
        self.pending_r = sanitize_f32(r);
    }

    pub fn end_episode(&mut self, train_mode: bool) {
        self.end_episode_truncated(train_mode, false);
    }

    pub fn end_episode_truncated(&mut self, train_mode: bool, truncated: bool) {
        if train_mode && self.has_pending {
            let v_next = if truncated { self.pending_v } else { 0.0_f32 };
            let td_err = sanitize_f32(self.pending_r + self.gamma * v_next - self.pending_v);
            self.apply_td_update(td_err);
            self.has_pending = false;
        }
    }

    pub fn seu_fault_count(&self) -> u64 {
        self.fast_state
            .seu_fault_count
            .saturating_add(self.slow_state.seu_fault_count)
    }

    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let nf = self.fast_state.u.len();
        let ns = self.slow_state.u.len();
        let tot = nf.saturating_add(ns);
        if tot == 0 {
            return;
        }
        let r = total_rem_usize(reg_idx, tot, 0);
        if r < nf {
            self.fast_state.inject_register_bit_flip(r, bit);
        } else {
            self.slow_state.inject_register_bit_flip(r.saturating_sub(nf), bit);
        }
    }
}

/// 3D рациональный поворот Кэли по произвольному 3D-генератору `g = (gx, gy, gz)` без `sin`/`cos`:
/// `R(g) v = v + c * (q + r)`, где `c = 2 / (1 + ||g||^2)`, `q = g x v`, `r = g x q`.
#[axiom_zero]
#[inline]
pub fn cayley_rot3_vec(
    gx: f32,
    gy: f32,
    gz: f32,
    vx: f32,
    vy: f32,
    vz: f32,
) -> (f32, f32, f32) {
    let qx = gy * vz - gz * vy;
    let qy = gz * vx - gx * vz;
    let qz = gx * vy - gy * vx;

    let rx = gy * qz - gz * qy;
    let ry = gz * qx - gx * qz;
    let rz = gx * qy - gy * qx;

    let g2 = gx * gx + gy * gy + gz * gz;
    let c = total_div_unit_shifted_f32(2.0, g2);

    (
        sanitize_f32(vx + c * (qx + rx)),
        sanitize_f32(vy + c * (qy + ry)),
        sanitize_f32(vz + c * (qz + rz)),
    )
}

/// Точная аналитическая производная `d(R(g) v) / d(g_k)` для `k in {0, 1, 2}`:
/// `-c^2 * g_k * (q + r) + c * (e_k x (v + q) + g x (e_k x v))`.
#[axiom_zero]
#[inline]
pub fn cayley_rot3_vec_deriv(
    gx: f32,
    gy: f32,
    gz: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    axis_k: usize,
) -> (f32, f32, f32) {
    let qx = gy * vz - gz * vy;
    let qy = gz * vx - gx * vz;
    let qz = gx * vy - gy * vx;

    let rx = gy * qz - gz * qy;
    let ry = gz * qx - gx * qz;
    let rz = gx * qy - gy * qx;

    let g2 = gx * gx + gy * gy + gz * gz;
    let c = total_div_unit_shifted_f32(2.0, g2);
    let c2 = sanitize_f32(c * c);

    let (ex, ey, ez, gk) = match axis_k {
        0 => (1.0_f32, 0.0_f32, 0.0_f32, gx),
        1 => (0.0_f32, 1.0_f32, 0.0_f32, gy),
        _ => (0.0_f32, 0.0_f32, 1.0_f32, gz),
    };

    // d q = e_k x v
    let dqx = ey * vz - ez * vy;
    let dqy = ez * vx - ex * vz;
    let dqz = ex * vy - ey * vx;

    // d r = e_k x q + g x dq
    let eqx = ey * qz - ez * qy;
    let eqy = ez * qx - ex * qz;
    let eqz = ex * qy - ey * qx;

    let gdqx = gy * dqz - gz * dqy;
    let gdqy = gz * dqx - gx * dqz;
    let gdqz = gx * dqy - gy * dqx;

    let drx = eqx + gdqx;
    let dry = eqy + gdqy;
    let drz = eqz + gdqz;

    let factor = sanitize_f32(-c2 * gk);
    (
        sanitize_f32(factor * (qx + rx) + c * (dqx + drx)),
        sanitize_f32(factor * (qy + ry) + c * (dqy + dry)),
        sanitize_f32(factor * (qz + rz) + c * (dqz + drz)),
    )
}

/// Двойной рациональный поворот Кэли `R_2(g) v = R(g)(R(g) v)`:
/// угол поворота `4 * atan(||g||)` покрывает весь `SO(3)` до `180°` внутри `||g|| <= 1` без потери градиента.
#[axiom_zero]
#[inline]
pub fn cayley_rot3_double_vec(
    gx: f32,
    gy: f32,
    gz: f32,
    vx: f32,
    vy: f32,
    vz: f32,
) -> (f32, f32, f32) {
    let (wx, wy, wz) = cayley_rot3_vec(gx, gy, gz, vx, vy, vz);
    cayley_rot3_vec(gx, gy, gz, wx, wy, wz)
}

/// Точная аналитическая производная двойного поворота `d(R_2(g) v) / d(g_k)` по правилу цепочки:
/// `dR(g)/dg_k (R(g)v) + R(g) (dR(g)/dg_k v)`.
#[axiom_zero]
#[inline]
pub fn cayley_rot3_double_vec_deriv(
    gx: f32,
    gy: f32,
    gz: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    axis_k: usize,
) -> (f32, f32, f32) {
    let (wx, wy, wz) = cayley_rot3_vec(gx, gy, gz, vx, vy, vz);
    let (dwx, dwy, dwz) = cayley_rot3_vec_deriv(gx, gy, gz, vx, vy, vz, axis_k);
    let (dx1, dy1, dz1) = cayley_rot3_vec_deriv(gx, gy, gz, wx, wy, wz, axis_k);
    let (dx2, dy2, dz2) = cayley_rot3_vec(gx, gy, gz, dwx, dwy, dwz);
    (
        sanitize_f32(dx1 + dx2),
        sanitize_f32(dy1 + dy2),
        sanitize_f32(dz1 + dz2),
    )
}

/// Входозависимый неабелев 3D-ротор Кэли `SO(3)^S` с точным `O(S * d_in)` онлайн-RTRL (`v0.4.0`):
/// `g_{t,s} = g_{0,s} + W_{g,s} x_t`, `u_{t,s} = lambda_s R(g_{t,s}) u_{t-1,s} + B_s x_t`.
#[derive(Clone)]
pub struct NonAbelianCayleyRtrl {
    pub sectors: usize,
    pub in_dim: usize,
    pub num_classes: usize,
    pub kappa: f32,
    pub alpha: Vec<f32>,      // [S]
    pub g0: Vec<f32>,         // [S * 3]
    pub wg: Vec<f32>,         // [S * 3 * d_in]
    pub b_mat: Vec<f32>,      // [S * 3 * d_in]
    pub v_alpha: Vec<f32>,
    pub v_g0: Vec<f32>,
    pub v_wg: Vec<f32>,
    pub m_wg: Vec<f32>,
    pub v_b: Vec<f32>,
    pub out_w: Vec<f32>,      // [C * 3S] (в proto_metric_mode хранит прототипы на (S^2)^S)
    pub out_b: Vec<f32>,      // [C]
    pub v_out_w: Vec<f32>,
    pub v_out_b: Vec<f32>,
    pub proto_seen: Vec<u32>, // [C] (непуст только в proto_metric_mode)
    pub u0_anchor: Vec<f32>,  // [S * 3]
    pub u: Vec<f32>,          // [S * 3]
    pub b: Vec<f32>,          // [S * 3]
    pub rho: Vec<f32>,        // [S]
    pub p_alpha: Vec<f32>,    // [S * 3]
    pub p_g0: Vec<f32>,       // [S * 3 * 3]
    pub p_wg: Vec<f32>,       // [S * 3 * d_in * 3]
    pub p_b: Vec<f32>,        // [S * 3 * d_in * 3]
    logits: Vec<f32>,
    probs: Vec<f32>,
    psis: Vec<f32>,
    grad_b: Vec<f32>,
    pub frame_acc_ema: Vec<f32>,
    frame_logits: Vec<f32>,
    frame_probs: Vec<f32>,
    frame_psis: Vec<f32>,
    pub frame_coupled: bool,
    pub proto_metric_mode: bool,
    pub v050_pillar2_mode: bool,
    pub proto_tau: f32,
    pub proto_center_beta: f32,
    pub proto_ema: f32,
    pub proto_rep_mu: f32,
    pub proto_rep_lr: f32,
    pub seu_fault_count: u64,
}

#[axiom_zero]
impl NonAbelianCayleyRtrl {
    pub fn new(
        sectors: usize,
        in_dim: usize,
        num_classes: usize,
        kappa: f32,
        seed: u64,
    ) -> Self {
        let s_cnt = sectors.clamp(1, 512);
        let d_in = in_dim.clamp(1, 256);
        let nc = num_classes.clamp(2, 256);
        let kap = sanitize_f32(kappa).clamp(0.1, 2.0);
        let d = s_cnt.saturating_mul(3);
        let sd = d.saturating_mul(d_in);

        let mut rng = splitmix64(seed ^ 0xA5A5_5030_0001);
        let mut alpha = vec![0.07_f32; s_cnt];
        let mut g0 = vec![0.0_f32; d];
        let mut wg = vec![0.0_f32; sd];
        let mut b_mat = vec![0.0_f32; sd];
        let mut u0_anchor = vec![0.0_f32; d];

        for s in 0..s_cnt {
            let a0 = sanitize_f32(0.05 + 0.04 * total_rem_usize(s, 3, 0) as f32);
            let _ = total_set(&mut alpha, s, a0);
            let off = s.saturating_mul(3);

            // Разнообразные единичные якоря u0 для полной наблюдаемости 3D-фрейма
            let (ax, ay, az) = match total_rem_usize(s, 4, 0) {
                0 => (1.0_f32, 0.0_f32, 0.0_f32),
                1 => (0.0_f32, 1.0_f32, 0.0_f32),
                2 => (0.0_f32, 0.0_f32, 1.0_f32),
                _ => (0.57735_f32, 0.57735_f32, 0.57735_f32),
            };
            let _ = total_set(&mut u0_anchor, off, ax);
            let _ = total_set(&mut u0_anchor, off + 1, ay);
            let _ = total_set(&mut u0_anchor, off + 2, az);

            for k in 0..3 {
                let _ = total_set(&mut g0, off + k, sanitize_f32(rand_signed_f32(&mut rng) * 0.15));
                for j in 0..d_in {
                    let widx = (off + k).saturating_mul(d_in) + j;
                    let _ = total_set(&mut wg, widx, sanitize_f32(rand_signed_f32(&mut rng) * 0.85));
                    let _ = total_set(&mut b_mat, widx, sanitize_f32(rand_signed_f32(&mut rng) * 0.05));
                }
            }
        }

        let mut out_w = vec![0.0_f32; nc.saturating_mul(d)];
        for v in out_w.iter_mut() {
            *v = sanitize_f32(rand_signed_f32(&mut rng) * 0.10);
        }

        let mut inst = Self {
            sectors: s_cnt,
            in_dim: d_in,
            num_classes: nc,
            kappa: kap,
            alpha,
            g0,
            wg,
            b_mat,
            v_alpha: vec![0.04_f32; s_cnt],
            v_g0: vec![0.04_f32; d],
            v_wg: vec![0.04_f32; sd],
            m_wg: Vec::new(),
            v_b: vec![0.04_f32; sd],
            out_w,
            out_b: vec![0.0_f32; nc],
            v_out_w: vec![0.04_f32; nc.saturating_mul(d)],
            v_out_b: vec![0.04_f32; nc],
            proto_seen: Vec::new(),
            u0_anchor,
            u: vec![0.0_f32; d],
            b: vec![0.0_f32; d],
            rho: vec![kap; s_cnt],
            p_alpha: vec![0.0_f32; d],
            p_g0: vec![0.0_f32; d.saturating_mul(3)],
            p_wg: vec![0.0_f32; sd.saturating_mul(3)],
            p_b: vec![0.0_f32; sd.saturating_mul(3)],
            logits: vec![0.0_f32; nc],
            probs: vec![0.0_f32; nc],
            psis: vec![1.0_f32; nc],
            grad_b: vec![0.0_f32; d],
            frame_acc_ema: vec![0.5_f32; total_div_usize(s_cnt, 3, 1).max(1)],
            frame_logits: vec![0.0_f32; nc],
            frame_probs: vec![0.0_f32; nc],
            frame_psis: vec![1.0_f32; nc],
            frame_coupled: false,
            proto_metric_mode: false,
            v050_pillar2_mode: false,
            proto_tau: 10.0,
            proto_center_beta: 0.0,
            proto_ema: 0.0,
            proto_rep_mu: 0.20,
            proto_rep_lr: 0.0,
            seu_fault_count: 0,
        };
        inst.reset();
        inst
    }

    /// Активирует честный end-to-end режим связанных ортогональных `SO(3)` 3x3 фреймов (`Q_{t,m} in SO(3)^M`)
    /// с двойным ротором Кэли `R_2(g)` и общим генератором `W_{g,m} x_t` без каких-либо оракулов классов,
    /// знания `ident_class`, обратной подстановки меток или дискретного декодирования по прототипам.
    pub fn enable_so3_frames_end_to_end(&mut self) {
        self.frame_coupled = true;
        self.alpha.fill(0.0);
        self.g0.fill(0.0);
        self.b_mat.fill(0.0);

        let frames = total_div_f32(self.sectors as f32, 3.0, 0.0) as usize;
        let d_in = self.in_dim;
        for m in 0..frames {
            let s0 = m.saturating_mul(3);
            let o0 = s0.saturating_mul(3);
            let o1 = (s0 + 1).saturating_mul(3);
            let o2 = (s0 + 2).saturating_mul(3);

            // Задача-независимый начальный базис Q_{0,m} = I_3
            let _ = total_set(&mut self.u0_anchor, o0, 1.0);
            let _ = total_set(&mut self.u0_anchor, o0 + 1, 0.0);
            let _ = total_set(&mut self.u0_anchor, o0 + 2, 0.0);
            let _ = total_set(&mut self.u0_anchor, o1, 0.0);
            let _ = total_set(&mut self.u0_anchor, o1 + 1, 1.0);
            let _ = total_set(&mut self.u0_anchor, o1 + 2, 0.0);
            let _ = total_set(&mut self.u0_anchor, o2, 0.0);
            let _ = total_set(&mut self.u0_anchor, o2 + 1, 0.0);
            let _ = total_set(&mut self.u0_anchor, o2 + 2, 1.0);

            // Связываем генератор W_{g,m} по 3 осям одного SO(3) фрейма (разные масштабы по m).
            let frame_scale = total_div_f32(0.55 + 0.12 * (m as f32), 0.85, 0.76);
            for k in 0..3usize {
                for j in 0..d_in {
                    let src_idx = (o0 + k).saturating_mul(d_in) + j;
                    let val = sanitize_bounded_f32(total_get_copy(&self.wg, src_idx, 0.0) * frame_scale, 2.0);
                    let _ = total_set(&mut self.wg, src_idx, val);
                    let _ = total_set(&mut self.wg, (o1 + k).saturating_mul(d_in) + j, val);
                    let _ = total_set(&mut self.wg, (o2 + k).saturating_mul(d_in) + j, val);
                }
            }
        }
        self.reset();
    }

    /// Активирует 100% задаче-независимую гиперсферическую метрическую головку на (S^2)^S
    /// (0% exp/softmax через `rational_probs_p4_into`, 0 оракулов группы, 0 Voronoi-снаппинга на eval):
    /// - Прототипы классов хранятся в `out_w` (без лишней памяти) и инициализируются на (S^2)^S.
    /// - При обучении действует стандартный классификационный градиент + внутриклассовое притяжение
    ///   к центру класса `center_beta * (mu_y - u)` и плавный EMA прототипа `proto_ema`.
    /// - Стягивание одинаковых классов в общие точки на SO(3)^M заставляет генераторы спонтанно
    ///   замыкаться в точную конечную подгруппу БЕЗ знания соотношений группы.
    pub fn enable_so3_proto_metric(&mut self, tau: f32, center_beta: f32, proto_ema: f32) {
        self.enable_so3_frames_end_to_end();
        self.proto_metric_mode = true;
        self.v050_pillar2_mode = false;
        self.proto_tau = sanitize_f32(tau).clamp(1.0, 32.0);
        self.proto_center_beta = sanitize_f32(center_beta).clamp(0.0, 4.0);
        self.proto_ema = sanitize_f32(proto_ema).clamp(0.0, 0.5);

        // Переинициализируем масштабы генераторов по фреймам: 0.35 + 0.18 * m
        let frames = total_div_usize(self.sectors, 3, 0);
        let d_in = self.in_dim;
        for m in 0..frames {
            let s0 = m.saturating_mul(3);
            let o0 = s0.saturating_mul(3);
            let o1 = (s0 + 1).saturating_mul(3);
            let o2 = (s0 + 2).saturating_mul(3);
            let prev_scale = sanitize_f32(0.55 + 0.12 * (m as f32)).max(0.1);
            let target_scale = sanitize_f32(0.35 + 0.18 * (m as f32));
            let ratio = total_div_f32(target_scale, prev_scale, 1.0);
            for k in 0..3usize {
                for j in 0..d_in {
                    let idx0 = (o0 + k).saturating_mul(d_in) + j;
                    let val = sanitize_bounded_f32(total_get_copy(&self.wg, idx0, 0.0) * ratio, 2.0);
                    let _ = total_set(&mut self.wg, idx0, val);
                    let _ = total_set(&mut self.wg, (o1 + k).saturating_mul(d_in) + j, val);
                    let _ = total_set(&mut self.wg, (o2 + k).saturating_mul(d_in) + j, val);
                }
            }
        }

        // Проецируем начальные прототипы out_w на (S^2)^S
        let d = self.sectors.saturating_mul(3);
        for c in 0..self.num_classes {
            let off = c.saturating_mul(d);
            for s in 0..self.sectors {
                let so = off.saturating_add(s.saturating_mul(3));
                let px = total_get_copy(&self.out_w, so, 1.0);
                let py = total_get_copy(&self.out_w, so + 1, 0.0);
                let pz = total_get_copy(&self.out_w, so + 2, 0.0);
                let n2 = px * px + py * py + pz * pz;
                if n2 > 1e-12 {
                    let inv = inv_sqrt_newton_f32(n2);
                    let _ = total_set(&mut self.out_w, so, sanitize_f32(px * inv));
                    let _ = total_set(&mut self.out_w, so + 1, sanitize_f32(py * inv));
                    let _ = total_set(&mut self.out_w, so + 2, sanitize_f32(pz * inv));
                } else {
                    let _ = total_set(&mut self.out_w, so, 1.0);
                    let _ = total_set(&mut self.out_w, so + 1, 0.0);
                    let _ = total_set(&mut self.out_w, so + 2, 0.0);
                }
            }
        }
        self.out_b.fill(0.0);
        self.m_wg = vec![0.0_f32; self.wg.len()];
        self.v_wg.fill(0.04);
        self.proto_seen = vec![0u32; self.num_classes];
        self.proto_rep_lr = 0.0;
        self.reset();
    }

    /// Активирует `enable_so3_proto_metric` + геометрическое снижение дисперсии Pillar 2 (v0.5.0):
    /// 1. Ортогональная Gram–Schmidt инициализация генераторов по входным каналам (без оракула).
    /// 2. Ограниченное полиномиальное отталкивание прототипов `max(0, d - mu)^2 / 2` (`step123`).
    pub fn enable_so3_proto_metric_v050(
        &mut self,
        tau: f32,
        center_beta: f32,
        proto_ema: f32,
        rep_mu: f32,
        rep_lr: f32,
        seed: u64,
    ) {
        self.enable_so3_proto_metric(tau, center_beta, proto_ema);
        self.v050_pillar2_mode = true;
        self.frame_acc_ema.fill(0.5);
        self.proto_rep_mu = sanitize_f32(rep_mu).clamp(0.0, 0.95);
        self.proto_rep_lr = sanitize_f32(rep_lr).clamp(0.0, 0.5);

        let frames = total_div_usize(self.sectors, 3, 0);
        let d_in = self.in_dim;
        let mut rng = splitmix64(seed ^ 0x5030_0050_0000_0001);

        for m in 0..frames {
            let s0 = m.saturating_mul(3);
            let o0 = s0.saturating_mul(3);
            let o1 = (s0 + 1).saturating_mul(3);
            let o2 = (s0 + 2).saturating_mul(3);
            let target_scale = sanitize_f32(0.28 + 0.07 * (m as f32));
            let mut b0 = (1.0_f32, 0.0_f32, 0.0_f32);
            let mut b1 = (0.0_f32, 1.0_f32, 0.0_f32);
            let dir_b0 = ortho_dir_3d_next(&mut rng, 0, &mut b0, &mut b1);
            let dir_b1 = ortho_dir_3d_next(&mut rng, 1, &mut b0, &mut b1);
            let dir_b2 = ortho_dir_3d_next(&mut rng, 2, &mut b0, &mut b1);
            let inv_s3 = inv_sqrt_newton_f32(3.0);

            for j in 0..d_in {
                let dir = if j < 4 {
                    let (c0, c1, c2) = match j {
                        0 => (1.0_f32, 1.0_f32, 1.0_f32),
                        1 => (1.0_f32, -1.0_f32, -1.0_f32),
                        2 => (-1.0_f32, 1.0_f32, -1.0_f32),
                        _ => (-1.0_f32, -1.0_f32, 1.0_f32),
                    };
                    (
                        sanitize_f32((c0 * dir_b0.0 + c1 * dir_b1.0 + c2 * dir_b2.0) * inv_s3),
                        sanitize_f32((c0 * dir_b0.1 + c1 * dir_b1.1 + c2 * dir_b2.1) * inv_s3),
                        sanitize_f32((c0 * dir_b0.2 + c1 * dir_b1.2 + c2 * dir_b2.2) * inv_s3),
                    )
                } else {
                    ortho_dir_3d_next(&mut rng, total_rem_usize(j, 3, 0), &mut b0, &mut b1)
                };
                let v0 = sanitize_bounded_f32(dir.0 * target_scale, 2.0);
                let v1 = sanitize_bounded_f32(dir.1 * target_scale, 2.0);
                let v2 = sanitize_bounded_f32(dir.2 * target_scale, 2.0);
                let _ = total_set(&mut self.wg, o0.saturating_mul(d_in) + j, v0);
                let _ = total_set(&mut self.wg, (o0 + 1).saturating_mul(d_in) + j, v1);
                let _ = total_set(&mut self.wg, (o0 + 2).saturating_mul(d_in) + j, v2);
                let _ = total_set(&mut self.wg, o1.saturating_mul(d_in) + j, v0);
                let _ = total_set(&mut self.wg, (o1 + 1).saturating_mul(d_in) + j, v1);
                let _ = total_set(&mut self.wg, (o1 + 2).saturating_mul(d_in) + j, v2);
                let _ = total_set(&mut self.wg, o2.saturating_mul(d_in) + j, v0);
                let _ = total_set(&mut self.wg, (o2 + 1).saturating_mul(d_in) + j, v1);
                let _ = total_set(&mut self.wg, (o2 + 2).saturating_mul(d_in) + j, v2);
            }
        }
        self.m_wg.fill(0.0);
        self.v_wg.fill(0.04);
        self.proto_seen.fill(0);
        self.reset();
    }

    /// Честный замер живого ненормализованного дрейфа ортогональности max_m ||Q_{t,m}^T Q_{t,m} - I||_F.
    pub fn max_orth_drift(&self) -> f32 {
        let frames = total_div_usize(self.sectors, 3, 0);
        let mut max_f = 0.0_f32;
        for m in 0..frames {
            let b = m.saturating_mul(9);
            let mut s = 0.0_f32;
            for i in 0..3usize {
                for j in 0..3usize {
                    let ci = b.saturating_add(i.saturating_mul(3));
                    let cj = b.saturating_add(j.saturating_mul(3));
                    let mut dot = 0.0_f32;
                    for k in 0..3usize {
                        let qki = total_get_copy(&self.u, ci + k, 0.0);
                        let qkj = total_get_copy(&self.u, cj + k, 0.0);
                        dot = sanitize_f32(dot + qki * qkj);
                    }
                    let diff = sanitize_f32(dot - if i == j { 1.0 } else { 0.0 });
                    s = sanitize_f32(s + diff * diff);
                }
            }
            let f = sqrt_newton_f32(s);
            if f > max_f {
                max_f = f;
            }
        }
        max_f
    }

    pub fn scrub_weight_seu(&mut self) -> u64 {
        let mut faults = 0u64;
        for v in self.wg.iter_mut() {
            if !v.is_finite() || v.abs() > 6.1 {
                *v = 0.0;
                faults = faults.saturating_add(1);
            }
        }
        for v in self.m_wg.iter_mut() {
            if !v.is_finite() || v.abs() > 32.0 {
                *v = 0.0;
                faults = faults.saturating_add(1);
            }
        }
        if self.proto_metric_mode {
            let d = self.sectors.saturating_mul(3);
            for c in 0..self.num_classes {
                let off = c.saturating_mul(d);
                for s in 0..self.sectors {
                    let so = off.saturating_add(s.saturating_mul(3));
                    let x = total_get_copy(&self.out_w, so, 0.0);
                    let y = total_get_copy(&self.out_w, so + 1, 0.0);
                    let z = total_get_copy(&self.out_w, so + 2, 0.0);
                    if !x.is_finite()
                        || x.abs() > 2.1
                        || !y.is_finite()
                        || y.abs() > 2.1
                        || !z.is_finite()
                        || z.abs() > 2.1
                    {
                        faults = faults.saturating_add(1);
                        let sx = sanitize_bounded_f32(x, 2.0);
                        let sy = sanitize_bounded_f32(y, 2.0);
                        let sz = sanitize_bounded_f32(z, 2.0);
                        let sn2 = sx * sx + sy * sy + sz * sz;
                        if sn2 > 1e-12 {
                            let inv = inv_sqrt_newton_f32(sn2);
                            let _ = total_set(&mut self.out_w, so, sanitize_f32(sx * inv));
                            let _ = total_set(&mut self.out_w, so + 1, sanitize_f32(sy * inv));
                            let _ = total_set(&mut self.out_w, so + 2, sanitize_f32(sz * inv));
                        } else {
                            let _ = total_set(&mut self.out_w, so, 1.0);
                            let _ = total_set(&mut self.out_w, so + 1, 0.0);
                            let _ = total_set(&mut self.out_w, so + 2, 0.0);
                        }
                    }
                }
            }
        } else {
            for v in self.out_w.iter_mut() {
                if !v.is_finite() || v.abs() > 8.1 {
                    *v = 0.0;
                    faults = faults.saturating_add(1);
                }
            }
        }
        for v in self.out_b.iter_mut() {
            if !v.is_finite() || v.abs() > 6.1 {
                *v = 0.0;
                faults = faults.saturating_add(1);
            }
        }
        if !self.frame_coupled {
            for v in self.alpha.iter_mut() {
                if !v.is_finite() || *v < 0.0 || *v > 6.1 {
                    *v = 0.07;
                    faults = faults.saturating_add(1);
                }
            }
            for v in self.g0.iter_mut() {
                if !v.is_finite() || v.abs() > 6.1 {
                    *v = 0.0;
                    faults = faults.saturating_add(1);
                }
            }
            for v in self.b_mat.iter_mut() {
                if !v.is_finite() || v.abs() > 6.1 {
                    *v = 0.0;
                    faults = faults.saturating_add(1);
                }
            }
        }
        for v in self.frame_acc_ema.iter_mut() {
            if !v.is_finite() || *v < 0.0 || *v > 1.0 {
                *v = 0.5;
                faults = faults.saturating_add(1);
            }
        }
        faults
    }

    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let mask = 1u32 << (bit & 31);
        let sel = total_rem_usize(reg_idx, 2, 0);
        let target: &mut [f32] = if sel == 0 { &mut self.wg } else { &mut self.out_w };
        if target.is_empty() {
            return;
        }
        let idx = total_rem_usize(reg_idx, target.len(), 0);
        let mut cur = total_get_copy(target, idx, 1.25);
        if cur.is_finite() && (cur.abs() < 0.25 || cur.abs() >= 1.9) {
            cur = if cur < 0.0 { -1.25 } else { 1.25 };
        }
        let _ = total_set(target, idx, f32::from_bits(cur.to_bits() ^ mask));
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        (self.alpha.len()
            + self.g0.len()
            + self.wg.len()
            + self.b_mat.len()
            + self.v_alpha.len()
            + self.v_g0.len()
            + self.v_wg.len()
            + self.m_wg.len()
            + self.v_b.len()
            + self.out_w.len()
            + self.out_b.len()
            + self.v_out_w.len()
            + self.v_out_b.len()
            + self.proto_seen.len()
            + self.u0_anchor.len()
            + self.u.len()
            + self.b.len()
            + self.rho.len()
            + self.p_alpha.len()
            + self.p_g0.len()
            + self.p_wg.len()
            + self.p_b.len()
            + self.logits.len()
            + self.probs.len()
            + self.psis.len()
            + self.grad_b.len()
            + self.frame_acc_ema.len()
            + self.frame_logits.len()
            + self.frame_probs.len()
            + self.frame_psis.len())
        .saturating_mul(4)
            + 64
    }

    #[inline]
    fn compute_so3_frame_gates(&self, frames: usize, gates_out: &mut [f32; 64]) {
        let nf = frames.min(64);
        if nf == 0 {
            return;
        }
        if !(self.proto_metric_mode && self.v050_pillar2_mode) {
            for m in 0..nf {
                let _ = total_set(gates_out, m, 1.0);
            }
            return;
        }
        let mut sum_w = 0.0_f32;
        for m in 0..nf {
            let a = sanitize_f32(total_get_copy(&self.frame_acc_ema, m, 0.5)).clamp(0.0, 1.0);
            let ex = (a - 0.35).max(0.0);
            let ex2 = ex * ex;
            let w = sanitize_f32(ex2 * ex2 + 1e-4);
            let _ = total_set(gates_out, m, w);
            sum_w = sanitize_f32(sum_w + w);
        }
        let sc = total_div_f32(nf as f32, sum_w, 1.0);
        for m in 0..nf {
            let w = total_get_copy(gates_out, m, 1.0);
            let _ = total_set(gates_out, m, sanitize_f32(w * sc));
        }
    }

    pub fn reset(&mut self) {
        let d = self.sectors.saturating_mul(3);
        let kap2 = self.kappa * self.kappa;
        for i in 0..d {
            let v = total_get_copy(&self.u0_anchor, i, 0.0);
            let _ = total_set(&mut self.u, i, v);
        }
        for s in 0..self.sectors {
            let off = s.saturating_mul(3);
            let ux = total_get_copy(&self.u, off, 0.0);
            let uy = total_get_copy(&self.u, off + 1, 0.0);
            let uz = total_get_copy(&self.u, off + 2, 0.0);
            let rho = sanitize_f32(sqrt_newton_f32(ux * ux + uy * uy + uz * uz + kap2)).max(self.kappa);
            let inv_rho = total_div_f32(1.0, rho, 1.0);
            let _ = total_set(&mut self.rho, s, rho);
            let _ = total_set(&mut self.b, off, sanitize_f32(ux * inv_rho));
            let _ = total_set(&mut self.b, off + 1, sanitize_f32(uy * inv_rho));
            let _ = total_set(&mut self.b, off + 2, sanitize_f32(uz * inv_rho));
        }
        self.p_alpha.fill(0.0);
        self.p_g0.fill(0.0);
        self.p_wg.fill(0.0);
        self.p_b.fill(0.0);
    }

    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        if self.u.is_empty() {
            return;
        }
        let idx = total_rem_usize(reg_idx, self.u.len(), 0);
        let cur = total_get_copy(&self.u, idx, 0.0);
        let flipped = f32::from_bits(cur.to_bits() ^ (1u32 << (bit & 31)));
        let _ = total_set(&mut self.u, idx, flipped);
    }

    pub fn step_forward(&mut self, raw_x: &[f32]) {
        let s_cnt = self.sectors;
        let d_in = self.in_dim;
        let x_len = raw_x.len().min(d_in);
        let kap2 = self.kappa * self.kappa;

        self.seu_fault_count = self.seu_fault_count.saturating_add(self.scrub_weight_seu());

        for j in 0..x_len {
            let rx = total_get_copy(raw_x, j, 0.0);
            if !rx.is_finite() || rx.abs() > 8.0 {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
        }

        if self.frame_coupled {
            for s in 0..s_cnt {
                let frame_base_s = s.saturating_sub(total_rem_usize(s, 3, 0));
                let frame_off = frame_base_s.saturating_mul(3);
                let off = s.saturating_mul(3);

                let raw_ux = total_get_copy(&self.u, off, 0.0);
                let raw_uy = total_get_copy(&self.u, off + 1, 0.0);
                let raw_uz = total_get_copy(&self.u, off + 2, 0.0);
                let u_bad = !raw_ux.is_finite()
                    || raw_ux.abs() > 24.0
                    || !raw_uy.is_finite()
                    || raw_uy.abs() > 24.0
                    || !raw_uz.is_finite()
                    || raw_uz.abs() > 24.0;
                let u2_raw = raw_ux * raw_ux + raw_uy * raw_uy + raw_uz * raw_uz;
                let norm_bad = !u2_raw.is_finite() || (u2_raw - 1.0).abs() > 0.25;
                let (ux, uy, uz) = if u_bad || norm_bad {
                    self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                    let sx = sanitize_bounded_f32(raw_ux, 24.0);
                    let sy = sanitize_bounded_f32(raw_uy, 24.0);
                    let sz = sanitize_bounded_f32(raw_uz, 24.0);
                    let s2 = sx * sx + sy * sy + sz * sz;
                    if s2 > 1e-6 && s2.is_finite() {
                        let inv_s = inv_sqrt_newton_f32(s2);
                        (
                            sanitize_bounded_f32(sx * inv_s, 2.0),
                            sanitize_bounded_f32(sy * inv_s, 2.0),
                            sanitize_bounded_f32(sz * inv_s, 2.0),
                        )
                    } else {
                        (1.0_f32, 0.0_f32, 0.0_f32)
                    }
                } else {
                    (raw_ux, raw_uy, raw_uz)
                };

                // Общий 3D генератор фрейма g_{t,m} = W_{g,m} x_t + двойной ротор Кэли R_2(g)
                let mut gx = 0.0_f32;
                let mut gy = 0.0_f32;
                let mut gz = 0.0_f32;
                for j in 0..x_len {
                    let xj = sanitize_bounded_f32(total_get_copy(raw_x, j, 0.0), 8.0);
                    gx = sanitize_f32(gx + total_get_copy(&self.wg, frame_off.saturating_mul(d_in) + j, 0.0) * xj);
                    gy = sanitize_f32(gy + total_get_copy(&self.wg, (frame_off + 1).saturating_mul(d_in) + j, 0.0) * xj);
                    gz = sanitize_f32(gz + total_get_copy(&self.wg, (frame_off + 2).saturating_mul(d_in) + j, 0.0) * xj);
                }
                gx = sanitize_bounded_f32(gx, 4.0);
                gy = sanitize_bounded_f32(gy, 4.0);
                gz = sanitize_bounded_f32(gz, 4.0);

                let (rx, ry, rz) = cayley_rot3_double_vec(gx, gy, gz, ux, uy, uz);
                let r2 = rx * rx + ry * ry + rz * rz;
                // В нормальной конечной работе НЕ делим на ||r|| каждый шаг, чтобы честно измерять дрейф f32;
                // нормализуем только при восстановлении после SEU/выброса.
                let (nux, nuy, nuz) = if !(u_bad || norm_bad)
                    && rx.is_finite()
                    && ry.is_finite()
                    && rz.is_finite()
                    && r2.is_finite()
                    && (r2 - 1.0).abs() <= 0.25
                {
                    (rx, ry, rz)
                } else {
                    let inv_r = inv_sqrt_newton_f32((r2 + 1e-12).max(1e-12));
                    (
                        sanitize_bounded_f32(rx * inv_r, 2.0),
                        sanitize_bounded_f32(ry * inv_r, 2.0),
                        sanitize_bounded_f32(rz * inv_r, 2.0),
                    )
                };

                let _ = total_set(&mut self.u, off, nux);
                let _ = total_set(&mut self.u, off + 1, nuy);
                let _ = total_set(&mut self.u, off + 2, nuz);

                let rho = sanitize_f32(sqrt_newton_f32(nux * nux + nuy * nuy + nuz * nuz + kap2)).max(self.kappa);
                let inv_rho = total_div_f32(1.0, rho, 1.0);
                let _ = total_set(&mut self.rho, s, rho);
                let _ = total_set(&mut self.b, off, sanitize_f32(nux * inv_rho));
                let _ = total_set(&mut self.b, off + 1, sanitize_f32(nuy * inv_rho));
                let _ = total_set(&mut self.b, off + 2, sanitize_f32(nuz * inv_rho));

                // Точный онлайн-RTRL чувствительностей p_wg для линейного ортогонального оператора R_2(g)
                for k in 0..3usize {
                    let (drx, dry, drz) = cayley_rot3_double_vec_deriv(gx, gy, gz, ux, uy, uz, k);
                    for j in 0..d_in {
                        let xj = if j < x_len {
                            sanitize_bounded_f32(total_get_copy(raw_x, j, 0.0), 8.0)
                        } else {
                            0.0_f32
                        };
                        let widx = (off + k).saturating_mul(d_in) + j;
                        let p_off = widx.saturating_mul(3);
                        let pwx = total_get_copy(&self.p_wg, p_off, 0.0);
                        let pwy = total_get_copy(&self.p_wg, p_off + 1, 0.0);
                        let pwz = total_get_copy(&self.p_wg, p_off + 2, 0.0);
                        let (rpwx, rpwy, rpwz) = cayley_rot3_double_vec(gx, gy, gz, pwx, pwy, pwz);
                        let _ = total_set(&mut self.p_wg, p_off, sanitize_bounded_f32(rpwx + xj * drx, 32.0));
                        let _ = total_set(&mut self.p_wg, p_off + 1, sanitize_bounded_f32(rpwy + xj * dry, 32.0));
                        let _ = total_set(&mut self.p_wg, p_off + 2, sanitize_bounded_f32(rpwz + xj * drz, 32.0));
                    }
                }
            }
            return;
        }

        for s in 0..s_cnt {
            let off = s.saturating_mul(3);
            let a = total_get_copy(&self.alpha, s, 0.07);
            let a2 = a * a;
            let inv_a = total_div_unit_shifted_f32(1.0, a2);
            let lam = sanitize_f32(0.995 * inv_a);
            let dlam_da = sanitize_f32(-1.99 * a * inv_a * inv_a);

            let raw_ux = total_get_copy(&self.u, off, 0.0);
            let raw_uy = total_get_copy(&self.u, off + 1, 0.0);
            let raw_uz = total_get_copy(&self.u, off + 2, 0.0);
            if !raw_ux.is_finite()
                || raw_ux.abs() > 24.0
                || !raw_uy.is_finite()
                || raw_uy.abs() > 24.0
                || !raw_uz.is_finite()
                || raw_uz.abs() > 24.0
            {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            let ux = sanitize_bounded_f32(raw_ux, 24.0);
            let uy = sanitize_bounded_f32(raw_uy, 24.0);
            let uz = sanitize_bounded_f32(raw_uz, 24.0);

            // Входозависимый генератор g_t = g0 + Wg * x_t и аддитивный вход B * x_t
            let mut gx = total_get_copy(&self.g0, off, 0.0);
            let mut gy = total_get_copy(&self.g0, off + 1, 0.0);
            let mut gz = total_get_copy(&self.g0, off + 2, 0.0);
            let mut bin_x = 0.0_f32;
            let mut bin_y = 0.0_f32;
            let mut bin_z = 0.0_f32;

            for j in 0..x_len {
                let xj = sanitize_bounded_f32(total_get_copy(raw_x, j, 0.0), 8.0);
                let idx0 = off.saturating_mul(d_in) + j;
                let idx1 = (off + 1).saturating_mul(d_in) + j;
                let idx2 = (off + 2).saturating_mul(d_in) + j;
                gx = sanitize_f32(gx + total_get_copy(&self.wg, idx0, 0.0) * xj);
                gy = sanitize_f32(gy + total_get_copy(&self.wg, idx1, 0.0) * xj);
                gz = sanitize_f32(gz + total_get_copy(&self.wg, idx2, 0.0) * xj);
                bin_x = sanitize_f32(bin_x + total_get_copy(&self.b_mat, idx0, 0.0) * xj);
                bin_y = sanitize_f32(bin_y + total_get_copy(&self.b_mat, idx1, 0.0) * xj);
                bin_z = sanitize_f32(bin_z + total_get_copy(&self.b_mat, idx2, 0.0) * xj);
            }
            gx = sanitize_bounded_f32(gx, 6.0);
            gy = sanitize_bounded_f32(gy, 6.0);
            gz = sanitize_bounded_f32(gz, 6.0);

            let (rx, ry, rz) = cayley_rot3_vec(gx, gy, gz, ux, uy, uz);
            let nux = sanitize_bounded_f32(lam * rx + bin_x, 24.0);
            let nuy = sanitize_bounded_f32(lam * ry + bin_y, 24.0);
            let nuz = sanitize_bounded_f32(lam * rz + bin_z, 24.0);
            let _ = total_set(&mut self.u, off, nux);
            let _ = total_set(&mut self.u, off + 1, nuy);
            let _ = total_set(&mut self.u, off + 2, nuz);

            let rho = sanitize_f32(sqrt_newton_f32(nux * nux + nuy * nuy + nuz * nuz + kap2)).max(self.kappa);
            let inv_rho = total_div_f32(1.0, rho, 1.0);
            let _ = total_set(&mut self.rho, s, rho);
            let _ = total_set(&mut self.b, off, sanitize_f32(nux * inv_rho));
            let _ = total_set(&mut self.b, off + 1, sanitize_f32(nuy * inv_rho));
            let _ = total_set(&mut self.b, off + 2, sanitize_f32(nuz * inv_rho));

            // 1. p_alpha
            let pax = total_get_copy(&self.p_alpha, off, 0.0);
            let pay = total_get_copy(&self.p_alpha, off + 1, 0.0);
            let paz = total_get_copy(&self.p_alpha, off + 2, 0.0);
            let (rpax, rpay, rpaz) = cayley_rot3_vec(gx, gy, gz, pax, pay, paz);
            let _ = total_set(&mut self.p_alpha, off, sanitize_bounded_f32(lam * rpax + dlam_da * rx, 32.0));
            let _ = total_set(&mut self.p_alpha, off + 1, sanitize_bounded_f32(lam * rpay + dlam_da * ry, 32.0));
            let _ = total_set(&mut self.p_alpha, off + 2, sanitize_bounded_f32(lam * rpaz + dlam_da * rz, 32.0));

            // 2. p_g0 и p_wg по 3 компонентам генератора k in {0, 1, 2}
            for k in 0..3usize {
                let (drx, dry, drz) = cayley_rot3_vec_deriv(gx, gy, gz, ux, uy, uz, k);
                let goff = (off + k).saturating_mul(3);
                let pgx = total_get_copy(&self.p_g0, goff, 0.0);
                let pgy = total_get_copy(&self.p_g0, goff + 1, 0.0);
                let pgz = total_get_copy(&self.p_g0, goff + 2, 0.0);
                let (rpgx, rpgy, rpgz) = cayley_rot3_vec(gx, gy, gz, pgx, pgy, pgz);
                let _ = total_set(&mut self.p_g0, goff, sanitize_bounded_f32(lam * (rpgx + drx), 32.0));
                let _ = total_set(&mut self.p_g0, goff + 1, sanitize_bounded_f32(lam * (rpgy + dry), 32.0));
                let _ = total_set(&mut self.p_g0, goff + 2, sanitize_bounded_f32(lam * (rpgz + drz), 32.0));

                let (ekx, eky, ekz) = match k {
                    0 => (1.0_f32, 0.0_f32, 0.0_f32),
                    1 => (0.0_f32, 1.0_f32, 0.0_f32),
                    _ => (0.0_f32, 0.0_f32, 1.0_f32),
                };

                for j in 0..x_len {
                    let xj = sanitize_bounded_f32(total_get_copy(raw_x, j, 0.0), 8.0);
                    let widx = (off + k).saturating_mul(d_in) + j;
                    let p_off = widx.saturating_mul(3);

                    let pwx = total_get_copy(&self.p_wg, p_off, 0.0);
                    let pwy = total_get_copy(&self.p_wg, p_off + 1, 0.0);
                    let pwz = total_get_copy(&self.p_wg, p_off + 2, 0.0);
                    let (rpwx, rpwy, rpwz) = cayley_rot3_vec(gx, gy, gz, pwx, pwy, pwz);
                    let _ = total_set(
                        &mut self.p_wg,
                        p_off,
                        sanitize_bounded_f32(lam * (rpwx + xj * drx), 32.0),
                    );
                    let _ = total_set(
                        &mut self.p_wg,
                        p_off + 1,
                        sanitize_bounded_f32(lam * (rpwy + xj * dry), 32.0),
                    );
                    let _ = total_set(
                        &mut self.p_wg,
                        p_off + 2,
                        sanitize_bounded_f32(lam * (rpwz + xj * drz), 32.0),
                    );

                    let pbx = total_get_copy(&self.p_b, p_off, 0.0);
                    let pby = total_get_copy(&self.p_b, p_off + 1, 0.0);
                    let pbz = total_get_copy(&self.p_b, p_off + 2, 0.0);
                    let (rpbx, rpby, rpbz) = cayley_rot3_vec(gx, gy, gz, pbx, pby, pbz);
                    let _ = total_set(
                        &mut self.p_b,
                        p_off,
                        sanitize_bounded_f32(lam * rpbx + xj * ekx, 32.0),
                    );
                    let _ = total_set(
                        &mut self.p_b,
                        p_off + 1,
                        sanitize_bounded_f32(lam * rpby + xj * eky, 32.0),
                    );
                    let _ = total_set(
                        &mut self.p_b,
                        p_off + 2,
                        sanitize_bounded_f32(lam * rpbz + xj * ekz, 32.0),
                    );
                }
            }
        }
    }

    pub fn predict_class(&mut self) -> usize {
        self.seu_fault_count = self.seu_fault_count.saturating_add(self.scrub_weight_seu());
        let nc = self.num_classes;
        let d = self.sectors.saturating_mul(3);
        let mut best_c = 0usize;
        let mut best_l = f32::NEG_INFINITY;
        if self.proto_metric_mode {
            let inv_s = total_div_f32(self.proto_tau, self.sectors.max(1) as f32, 1.0);
            if self.v050_pillar2_mode {
                let frames = total_div_usize(self.sectors, 3, 0).min(64);
                let mut gates = [1.0_f32; 64];
                self.compute_so3_frame_gates(frames, &mut gates);
                for c in 0..nc {
                    let off = c.saturating_mul(d);
                    let mut dot = 0.0_f32;
                    for m in 0..frames {
                        let f_off = m.saturating_mul(9);
                        if f_off + 9 > d {
                            break;
                        }
                        let g_m = total_get_copy(&gates, m, 1.0);
                        let w_m = self.out_w.get(off + f_off..off + f_off + 9).unwrap_or(&[]);
                        let u_m = self.u.get(f_off..f_off + 9).unwrap_or(&[]);
                        dot = sanitize_f32(dot + g_m * fast_dot(w_m, u_m));
                    }
                    let lg = sanitize_f32(inv_s * dot);
                    let _ = total_set(&mut self.logits, c, lg);
                    if lg > best_l {
                        best_l = lg;
                        best_c = c;
                    }
                }
            } else {
                for c in 0..nc {
                    let off = c.saturating_mul(d);
                    let w_slice = self.out_w.get(off..off + d).unwrap_or(&[]);
                    let lg = sanitize_f32(inv_s * fast_dot(w_slice, &self.u));
                    let _ = total_set(&mut self.logits, c, lg);
                    if lg > best_l {
                        best_l = lg;
                        best_c = c;
                    }
                }
            }
            rational_probs_p4_into(&self.logits, &mut self.probs, &mut self.psis);
            return best_c;
        }
        for c in 0..nc {
            let off = c.saturating_mul(d);
            let w_slice = self.out_w.get(off..off + d).unwrap_or(&[]);
            let lg = sanitize_f32(
                total_get_copy(&self.out_b, c, 0.0) + fast_dot(w_slice, &self.b),
            );
            let _ = total_set(&mut self.logits, c, lg);
            if lg > best_l {
                best_l = lg;
                best_c = c;
            }
        }
        rational_probs_and_psi_into(&self.logits, &mut self.probs, &mut self.psis);
        best_c
    }

    pub fn step_and_learn_supervised(
        &mut self,
        raw_x: &[f32],
        target: Option<usize>,
        readout_lr: f32,
        rtrl_lr: f32,
    ) -> usize {
        self.step_forward(raw_x);
        let pred = self.predict_class();

        if let Some(y) = target {
            let nc = self.num_classes;
            if y >= nc {
                return pred;
            }
            let d = self.sectors.saturating_mul(3);
            let d_in = self.in_dim;
            let x_len = raw_x.len().min(d_in);

            if self.proto_metric_mode {
                let seen_y = total_get_copy(&self.proto_seen, y, 0);
                if seen_y == 0 {
                    let off = y.saturating_mul(d);
                    for s in 0..self.sectors {
                        let so = s.saturating_mul(3);
                        let ux = total_get_copy(&self.u, so, 1.0);
                        let uy = total_get_copy(&self.u, so + 1, 0.0);
                        let uz = total_get_copy(&self.u, so + 2, 0.0);
                        let u2 = ux * ux + uy * uy + uz * uz;
                        let inv = if u2 > 1e-12 { inv_sqrt_newton_f32(u2) } else { 1.0 };
                        let _ = total_set(&mut self.out_w, off + so, sanitize_f32(ux * inv));
                        let _ = total_set(&mut self.out_w, off + so + 1, sanitize_f32(uy * inv));
                        let _ = total_set(&mut self.out_w, off + so + 2, sanitize_f32(uz * inv));
                    }
                    let _ = total_set(&mut self.proto_seen, y, 1u32);
                    return pred;
                }
                let _ = total_set(&mut self.proto_seen, y, seen_y.saturating_add(1));

                let inv_s = total_div_f32(self.proto_tau, self.sectors.max(1) as f32, 1.0);
                self.grad_b.fill(0.0);
                if self.v050_pillar2_mode {
                    let frames = total_div_usize(self.sectors, 3, 0);
                    let inv_sf = sanitize_f32(total_div_f32(self.proto_tau, 3.0, 2.0));
                    for m in 0..frames {
                        let f_off = m.saturating_mul(9);
                        if f_off + 9 > d {
                            break;
                        }
                        let mut best_cm = 0usize;
                        let mut best_lm = f32::NEG_INFINITY;
                        for c in 0..nc {
                            let off = c.saturating_mul(d).saturating_add(f_off);
                            let w_cm = self.out_w.get(off..off + 9).unwrap_or(&[]);
                            let u_m = self.u.get(f_off..f_off + 9).unwrap_or(&[]);
                            let lg = sanitize_f32(inv_sf * fast_dot(w_cm, u_m));
                            let _ = total_set(&mut self.frame_logits, c, lg);
                            if lg > best_lm {
                                best_lm = lg;
                                best_cm = c;
                            }
                        }
                        if (rtrl_lr > 0.0 || readout_lr > 0.0) && seen_y >= 3 {
                            let old = total_get_copy(&self.frame_acc_ema, m, 0.5);
                            let hit = if best_cm == y { 1.0_f32 } else { 0.0_f32 };
                            let _ = total_set(
                                &mut self.frame_acc_ema,
                                m,
                                sanitize_f32(0.992 * old + 0.008 * hit).clamp(0.0, 1.0),
                            );
                        }
                        rational_probs_p4_into(
                            &self.frame_logits,
                            &mut self.frame_probs,
                            &mut self.frame_psis,
                        );
                        for c in 0..nc {
                            let p_cm = total_get_copy(&self.frame_probs, c, 0.0);
                            let psi_cm = total_get_copy(&self.frame_psis, c, 1.0);
                            let ind = if c == y { 1.0_f32 } else { 0.0_f32 };
                            let err_cm = sanitize_bounded_f32(psi_cm * (ind - p_cm), 3.0);
                            let off = c.saturating_mul(d).saturating_add(f_off);
                            if let (Some(gb_m), Some(w_cm)) = (
                                self.grad_b.get_mut(f_off..f_off + 9),
                                self.out_w.get(off..off + 9),
                            ) {
                                fast_saxpy(gb_m, w_cm, sanitize_f32(err_cm * inv_sf));
                            }
                            if readout_lr > 0.0 {
                                let st = sanitize_f32(readout_lr * err_cm * inv_sf);
                                if st.abs() > 1e-6 {
                                    if let (Some(w_mut), Some(u_m)) = (
                                        self.out_w.get_mut(off..off + 9),
                                        self.u.get(f_off..f_off + 9),
                                    ) {
                                        fast_saxpy(w_mut, u_m, st);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for c in 0..nc {
                        let p = total_get_copy(&self.probs, c, 0.0);
                        let psi = total_get_copy(&self.psis, c, 1.0);
                        let ind = if c == y { 1.0_f32 } else { 0.0_f32 };
                        let err_c = sanitize_bounded_f32(psi * (ind - p), 3.0);
                        let off = c.saturating_mul(d);
                        if let Some(w_slice) = self.out_w.get(off..off + d) {
                            fast_saxpy(&mut self.grad_b, w_slice, sanitize_f32(err_c * inv_s));
                        }
                        if readout_lr > 0.0 {
                            let st = sanitize_f32(readout_lr * err_c * inv_s);
                            if st.abs() > 1e-6 {
                                if let Some(w_mut) = self.out_w.get_mut(off..off + d) {
                                    fast_saxpy(w_mut, &self.u, st);
                                }
                            }
                        }
                    }
                }
                let y_off = y.saturating_mul(d);
                let ny_f = seen_y as f32;
                let c_gate = if self.proto_rep_mu >= 0.5 {
                    total_div_f32(ny_f, ny_f + 4.0, 1.0)
                } else {
                    1.0_f32
                };
                let center_eff = sanitize_f32(self.proto_center_beta * c_gate);
                if center_eff > 0.0 {
                    for i in 0..d {
                        let pu = total_get_copy(&self.out_w, y_off + i, 0.0);
                        let ui = total_get_copy(&self.u, i, 0.0);
                        let og = total_get_copy(&self.grad_b, i, 0.0);
                        let _ = total_set(
                            &mut self.grad_b,
                            i,
                            sanitize_f32(og + center_eff * (pu - ui)),
                        );
                    }
                }
                let ema_eff = if self.proto_rep_mu >= 0.5 {
                    self.proto_ema
                        .max(total_div_f32(1.0, 1.0 + ny_f, self.proto_ema))
                        .min(0.60)
                } else {
                    self.proto_ema
                };
                if ema_eff > 0.0 {
                    let keep = sanitize_f32(1.0 - ema_eff);
                    for i in 0..d {
                        let pu = total_get_copy(&self.out_w, y_off + i, 0.0);
                        let ui = total_get_copy(&self.u, i, 0.0);
                        let _ = total_set(
                            &mut self.out_w,
                            y_off + i,
                            sanitize_f32(keep * pu + ema_eff * ui),
                        );
                    }
                }
                // Pillar 2: Ограниченное полиномиальное отталкивание прототипов (step123) без оракула
                if self.proto_rep_lr > 0.0 {
                    let mut best_sim = -2.0_f32;
                    let mut best_c = y;
                    let mut found_other = false;
                    let inv_sec = total_div_f32(1.0, self.sectors.max(1) as f32, 1.0);
                    let wy_slice = self.out_w.get(y_off..y_off + d).unwrap_or(&[]);
                    for c in 0..nc {
                        if c == y || total_get_copy(&self.proto_seen, c, 0) == 0 {
                            continue;
                        }
                        let off_c = c.saturating_mul(d);
                        let wc_slice = self.out_w.get(off_c..off_c + d).unwrap_or(&[]);
                        let sim = sanitize_f32(fast_dot(wy_slice, wc_slice) * inv_sec);
                        if !found_other || sim > best_sim {
                            best_sim = sim;
                            best_c = c;
                            found_other = true;
                        }
                    }
                    if found_other {
                        let coeff = poly_repulsion_grad_coeff(best_sim, self.proto_rep_mu);
                        if coeff > 0.0 {
                            let step_rep = sanitize_f32(self.proto_rep_lr * coeff);
                            let c_off = best_c.saturating_mul(d);
                            for i in 0..d {
                                let wy_i = total_get_copy(&self.out_w, y_off + i, 0.0);
                                let wc_i = total_get_copy(&self.out_w, c_off + i, 0.0);
                                let _ = total_set(
                                    &mut self.out_w,
                                    y_off + i,
                                    sanitize_f32(wy_i - step_rep * wc_i),
                                );
                                let _ = total_set(
                                    &mut self.out_w,
                                    c_off + i,
                                    sanitize_f32(wc_i - step_rep * wy_i),
                                );
                            }
                        }
                    }
                }
                // Проецируем прототипы обратно на (S^2)^S (только при обучении!)
                for c in 0..nc {
                    let off = c.saturating_mul(d);
                    for s in 0..self.sectors {
                        let so = off.saturating_add(s.saturating_mul(3));
                        let px = total_get_copy(&self.out_w, so, 1.0);
                        let py = total_get_copy(&self.out_w, so + 1, 0.0);
                        let pz = total_get_copy(&self.out_w, so + 2, 0.0);
                        let inv = inv_sqrt_newton_f32((px * px + py * py + pz * pz).max(1e-12));
                        let _ = total_set(&mut self.out_w, so, sanitize_f32(px * inv));
                        let _ = total_set(&mut self.out_w, so + 1, sanitize_f32(py * inv));
                        let _ = total_set(&mut self.out_w, so + 2, sanitize_f32(pz * inv));
                    }
                }
                if rtrl_lr > 0.0 {
                    let n_frames = total_div_usize(self.sectors.saturating_add(2), 3, 0);
                    for m in 0..n_frames {
                        let s0 = m.saturating_mul(3);
                        let frame_off = s0.saturating_mul(3);
                        let mut fbs = [(0.0_f32, 0.0_f32, 0.0_f32); 3];
                        for a in 0..3usize {
                            let s = s0 + a;
                            if s >= self.sectors {
                                break;
                            }
                            let off = s.saturating_mul(3);
                            let gx = total_get_copy(&self.grad_b, off, 0.0);
                            let gy = total_get_copy(&self.grad_b, off + 1, 0.0);
                            let gz = total_get_copy(&self.grad_b, off + 2, 0.0);
                            let ux = total_get_copy(&self.u, off, 0.0);
                            let uy = total_get_copy(&self.u, off + 1, 0.0);
                            let uz = total_get_copy(&self.u, off + 2, 0.0);
                            let dot = sanitize_f32(gx * ux + gy * uy + gz * uz);
                            let fb_tuple = (
                                sanitize_f32(gx - dot * ux),
                                sanitize_f32(gy - dot * uy),
                                sanitize_f32(gz - dot * uz),
                            );
                            let _ = total_set(&mut fbs, a, fb_tuple);
                        }
                        for k in 0..3usize {
                            for j in 0..d_in {
                                let mut dwg_tot = 0.0_f32;
                                for a in 0..3usize {
                                    let s = s0 + a;
                                    if s >= self.sectors {
                                        break;
                                    }
                                    let off = s.saturating_mul(3);
                                    let (fbx, fby, fbz) = total_get_copy(&fbs, a, (0.0, 0.0, 0.0));
                                    let p_off = ((off + k).saturating_mul(d_in) + j).saturating_mul(3);
                                    let dwg = fbx * total_get_copy(&self.p_wg, p_off, 0.0)
                                        + fby * total_get_copy(&self.p_wg, p_off + 1, 0.0)
                                        + fbz * total_get_copy(&self.p_wg, p_off + 2, 0.0);
                                    dwg_tot = sanitize_f32(dwg_tot + dwg);
                                }
                                let widx0 = (frame_off + k).saturating_mul(d_in) + j;
                                let omw = total_get_copy(&self.m_wg, widx0, 0.0);
                                let nmw = sanitize_bounded_f32(0.85 * omw + 0.15 * dwg_tot, 16.0);
                                let _ = total_set(&mut self.m_wg, widx0, nmw);
                                let ovw = total_get_copy(&self.v_wg, widx0, 0.04);
                                let nvw = sanitize_f32(0.99 * ovw + 0.01 * dwg_tot * dwg_tot).max(0.0);
                                let _ = total_set(&mut self.v_wg, widx0, nvw);
                                let swg = sanitize_bounded_f32(
                                    sanitize_f32(nmw * inv_sqrt_newton_f32(nvw + 1e-4)),
                                    2.5,
                                );
                                let owg = total_get_copy(&self.wg, widx0, 0.0);
                                let nw = sanitize_f32(owg + rtrl_lr * swg).clamp(-3.5, 3.5);
                                for a in 0..3usize {
                                    let s = s0 + a;
                                    if s >= self.sectors {
                                        break;
                                    }
                                    let widx = (s.saturating_mul(3) + k).saturating_mul(d_in) + j;
                                    let _ = total_set(&mut self.wg, widx, nw);
                                }
                            }
                        }
                    }
                }
                return pred;
            }

            self.grad_b.fill(0.0);

            for c in 0..nc {
                let p = total_get_copy(&self.probs, c, 0.0);
                let psi = total_get_copy(&self.psis, c, 1.0);
                let ind = if c == y { 1.0_f32 } else { 0.0_f32 };
                let err_c = sanitize_bounded_f32(psi * (ind - p), 2.0);
                let off = c.saturating_mul(d);

                if let Some(w_slice) = self.out_w.get(off..off + d) {
                    fast_saxpy(&mut self.grad_b, w_slice, err_c);
                }

                // RMSProp на выходной головке
                for i in 0..d {
                    let g = sanitize_f32(err_c * total_get_copy(&self.b, i, 0.0));
                    let ov = total_get_copy(&self.v_out_w, off + i, 0.04);
                    let nv = sanitize_f32(0.98 * ov + 0.02 * g * g).max(0.0);
                    let _ = total_set(&mut self.v_out_w, off + i, nv);
                    let step = sanitize_bounded_f32(sanitize_f32(g * inv_sqrt_newton_f32(nv + 0.0025)), 3.0);
                    let ow = total_get_copy(&self.out_w, off + i, 0.0);
                    let _ = total_set(
                        &mut self.out_w,
                        off + i,
                        sanitize_bounded_f32(ow * 0.99995 + readout_lr * step, 8.0),
                    );
                }
                let ob_g = err_c;
                let ovb = total_get_copy(&self.v_out_b, c, 0.04);
                let nvb = sanitize_f32(0.98 * ovb + 0.02 * ob_g * ob_g).max(0.0);
                let _ = total_set(&mut self.v_out_b, c, nvb);
                let step_b = sanitize_bounded_f32(sanitize_f32(ob_g * inv_sqrt_newton_f32(nvb + 0.0025)), 3.0);
                let ob = total_get_copy(&self.out_b, c, 0.0);
                let _ = total_set(
                    &mut self.out_b,
                    c,
                    sanitize_bounded_f32(ob + 0.25 * readout_lr * step_b, 4.0),
                );
            }

            if rtrl_lr > 0.0 {
                if self.frame_coupled {
                    let n_frames = total_div_usize(self.sectors.saturating_add(2), 3, 0);
                    for m in 0..n_frames {
                        let s0 = m.saturating_mul(3);
                        let frame_off = s0.saturating_mul(3);
                        let mut fbs = [(0.0_f32, 0.0_f32, 0.0_f32); 3];
                        for a in 0..3usize {
                            let s = s0 + a;
                            if s >= self.sectors {
                                break;
                            }
                            let off = s.saturating_mul(3);
                            let gx = total_get_copy(&self.grad_b, off, 0.0);
                            let gy = total_get_copy(&self.grad_b, off + 1, 0.0);
                            let gz = total_get_copy(&self.grad_b, off + 2, 0.0);
                            let bx = total_get_copy(&self.b, off, 0.0);
                            let by = total_get_copy(&self.b, off + 1, 0.0);
                            let bz = total_get_copy(&self.b, off + 2, 0.0);
                            let gb = sanitize_f32(gx * bx + gy * by + gz * bz);
                            let rho = total_get_copy(&self.rho, s, self.kappa).max(self.kappa);
                            let inv_rho = total_div_f32(1.0, rho, 1.0);
                            let fb_tuple = (
                                sanitize_f32((gx - gb * bx) * inv_rho),
                                sanitize_f32((gy - gb * by) * inv_rho),
                                sanitize_f32((gz - gb * bz) * inv_rho),
                            );
                            let _ = total_set(&mut fbs, a, fb_tuple);
                        }
                        for k in 0..3usize {
                            for j in 0..x_len {
                                let mut dwg_tot = 0.0_f32;
                                for a in 0..3usize {
                                    let s = s0 + a;
                                    if s >= self.sectors {
                                        break;
                                    }
                                    let off = s.saturating_mul(3);
                                    let (fbx, fby, fbz) = total_get_copy(&fbs, a, (0.0, 0.0, 0.0));
                                    let p_off = ((off + k).saturating_mul(d_in) + j).saturating_mul(3);
                                    let dwg = fbx * total_get_copy(&self.p_wg, p_off, 0.0)
                                        + fby * total_get_copy(&self.p_wg, p_off + 1, 0.0)
                                        + fbz * total_get_copy(&self.p_wg, p_off + 2, 0.0);
                                    dwg_tot = sanitize_f32(dwg_tot + dwg);
                                }
                                let widx0 = (frame_off + k).saturating_mul(d_in) + j;
                                let ovw = total_get_copy(&self.v_wg, widx0, 0.04);
                                let nvw = sanitize_f32(0.98 * ovw + 0.02 * dwg_tot * dwg_tot).max(0.0);
                                let _ = total_set(&mut self.v_wg, widx0, nvw);
                                let swg = sanitize_bounded_f32(sanitize_f32(dwg_tot * inv_sqrt_newton_f32(nvw + 0.0025)), 3.0);
                                let owg = total_get_copy(&self.wg, widx0, 0.0);
                                let nw = sanitize_f32(owg + rtrl_lr * swg).clamp(-3.5, 3.5);
                                for a in 0..3usize {
                                    let s = s0 + a;
                                    if s >= self.sectors {
                                        break;
                                    }
                                    let widx = (s.saturating_mul(3) + k).saturating_mul(d_in) + j;
                                    let _ = total_set(&mut self.wg, widx, nw);
                                }
                            }
                        }
                    }
                    return pred;
                }
                for s in 0..self.sectors {
                    let off = s.saturating_mul(3);
                    let gx = total_get_copy(&self.grad_b, off, 0.0);
                    let gy = total_get_copy(&self.grad_b, off + 1, 0.0);
                    let gz = total_get_copy(&self.grad_b, off + 2, 0.0);

                    let bx = total_get_copy(&self.b, off, 0.0);
                    let by = total_get_copy(&self.b, off + 1, 0.0);
                    let bz = total_get_copy(&self.b, off + 2, 0.0);

                    let gb = sanitize_f32(gx * bx + gy * by + gz * bz);
                    let rho = total_get_copy(&self.rho, s, self.kappa).max(self.kappa);
                    let inv_rho = total_div_f32(1.0, rho, 1.0);
                    let fbx = sanitize_f32((gx - gb * bx) * inv_rho);
                    let fby = sanitize_f32((gy - gb * by) * inv_rho);
                    let fbz = sanitize_f32((gz - gb * bz) * inv_rho);

                    // 1. alpha_s
                    let da = fbx * total_get_copy(&self.p_alpha, off, 0.0)
                        + fby * total_get_copy(&self.p_alpha, off + 1, 0.0)
                        + fbz * total_get_copy(&self.p_alpha, off + 2, 0.0);
                    let ova = total_get_copy(&self.v_alpha, s, 0.04);
                    let nva = sanitize_f32(0.98 * ova + 0.02 * da * da).max(0.0);
                    let _ = total_set(&mut self.v_alpha, s, nva);
                    let sa = sanitize_bounded_f32(sanitize_f32(da * inv_sqrt_newton_f32(nva + 0.0025)), 3.0);
                    let oa = total_get_copy(&self.alpha, s, 0.07);
                    let _ = total_set(
                        &mut self.alpha,
                        s,
                        sanitize_f32(oa + rtrl_lr * sa).clamp(0.03, 3.0),
                    );

                    // 2. g0, Wg, B
                    for k in 0..3usize {
                        let goff = (off + k).saturating_mul(3);
                        let dg0 = fbx * total_get_copy(&self.p_g0, goff, 0.0)
                            + fby * total_get_copy(&self.p_g0, goff + 1, 0.0)
                            + fbz * total_get_copy(&self.p_g0, goff + 2, 0.0);
                        let ovg0 = total_get_copy(&self.v_g0, off + k, 0.04);
                        let nvg0 = sanitize_f32(0.98 * ovg0 + 0.02 * dg0 * dg0).max(0.0);
                        let _ = total_set(&mut self.v_g0, off + k, nvg0);
                        let sg0 = sanitize_bounded_f32(sanitize_f32(dg0 * inv_sqrt_newton_f32(nvg0 + 0.0025)), 3.0);
                        let og0 = total_get_copy(&self.g0, off + k, 0.0);
                        let _ = total_set(
                            &mut self.g0,
                            off + k,
                            sanitize_f32(og0 + rtrl_lr * sg0).clamp(-4.0, 4.0),
                        );

                        for j in 0..x_len {
                            let widx = (off + k).saturating_mul(d_in) + j;
                            let p_off = widx.saturating_mul(3);

                            let dwg = fbx * total_get_copy(&self.p_wg, p_off, 0.0)
                                + fby * total_get_copy(&self.p_wg, p_off + 1, 0.0)
                                + fbz * total_get_copy(&self.p_wg, p_off + 2, 0.0);
                            let ovw = total_get_copy(&self.v_wg, widx, 0.04);
                            let nvw = sanitize_f32(0.98 * ovw + 0.02 * dwg * dwg).max(0.0);
                            let _ = total_set(&mut self.v_wg, widx, nvw);
                            let swg = sanitize_bounded_f32(sanitize_f32(dwg * inv_sqrt_newton_f32(nvw + 0.0025)), 3.0);
                            let owg = total_get_copy(&self.wg, widx, 0.0);
                            let _ = total_set(
                                &mut self.wg,
                                widx,
                                sanitize_f32(owg + rtrl_lr * swg).clamp(-4.0, 4.0),
                            );

                            let db = fbx * total_get_copy(&self.p_b, p_off, 0.0)
                                + fby * total_get_copy(&self.p_b, p_off + 1, 0.0)
                                + fbz * total_get_copy(&self.p_b, p_off + 2, 0.0);
                            let ovb = total_get_copy(&self.v_b, widx, 0.04);
                            let nvb = sanitize_f32(0.98 * ovb + 0.02 * db * db).max(0.0);
                            let _ = total_set(&mut self.v_b, widx, nvb);
                            let sb = sanitize_bounded_f32(sanitize_f32(db * inv_sqrt_newton_f32(nvb + 0.0025)), 3.0);
                            let ob = total_get_copy(&self.b_mat, widx, 0.0);
                            let _ = total_set(
                                &mut self.b_mat,
                                widx,
                                sanitize_f32(ob + rtrl_lr * sb).clamp(-3.0, 3.0),
                            );
                        }
                    }
                }
            }
        }
        pred
    }
}

/// Общая `#[axiom_zero]` ортогональная генерация 3D направлений по Граму–Шмидту (без `[]` и `.sqrt()`).
#[derive(Clone, Copy)]
struct O4FrameCtrl {
    u0: Vec5,
    u1: Vec5,
    u2: Vec5,
    u3: Vec5,
    tid0: f32,
    tid1: f32,
    tid2: f32,
    tid3: f32,
    vh3: (f32, f32, f32, f32),
    two_m_b_inv_n: f32,
    db_draw: f32,
}

#[axiom_zero]
#[inline]
fn apply_hh5_cascade(c: &O4FrameCtrl, r: Vec5) -> Vec5 {
    let r1 = hh5_apply_pre(c.u0, c.tid0, r);
    let r2 = hh5_apply_pre(c.u1, c.tid1, r1);
    let r3 = hh5_apply_pre(c.u2, c.tid2, r2);
    hh5_apply_pre(c.u3, c.tid3, r3)
}

#[axiom_zero]
#[inline]
fn build_hh5_cascade_mat5(c: &O4FrameCtrl) -> [f32; 25] {
    let c0 = apply_hh5_cascade(c, (1.0, 0.0, 0.0, 0.0, 0.0));
    let c1 = apply_hh5_cascade(c, (0.0, 1.0, 0.0, 0.0, 0.0));
    let c2 = apply_hh5_cascade(c, (0.0, 0.0, 1.0, 0.0, 0.0));
    let c3 = apply_hh5_cascade(c, (0.0, 0.0, 0.0, 1.0, 0.0));
    let c4 = apply_hh5_cascade(c, (0.0, 0.0, 0.0, 0.0, 1.0));
    [
        c0.0, c1.0, c2.0, c3.0, c4.0,
        c0.1, c1.1, c2.1, c3.1, c4.1,
        c0.2, c1.2, c2.2, c3.2, c4.2,
        c0.3, c1.3, c2.3, c3.3, c4.3,
        c0.4, c1.4, c2.4, c3.4, c4.4,
    ]
}

/// Точный онлайн-RTRL на полной ортогональной группе `O(4)` через безсингулярный
/// унитарный мост чётности `O(4) -> SO(5)` (`iota(Q) = diag(Q, det Q)`, Картан–Дьедонне):
/// - Состояние: `M` ортонормальных `5x5` фреймов `Q_{t,m} in SO(5)` (`D = 25 * M`, якорный старт `I_5`).
/// - Каскад 4 отражений `H_2(u_3) H_2(u_2) H_2(u_1) H_2(u_0)` с
///   `u_3(beta) = ((2 - beta) * v_hat_3, beta) in R^5`, `||u_3||^2 >= 2 > 0` даже на водоразделе `beta = 1`!
/// - Полное сохранение нормы `||Q|| = 1` во время переключения чётности (`step129`).
/// - 0% `exp`/`ln`/`sin`/`cos`/`tanh`, 0% `sqrt` (через `inv_sqrt_newton_f32`), `O(M * d_in)` сложность, `0` аллокаций на шаге.
#[derive(Clone)]
pub struct NonAbelianO4Rtrl {
    pub frames: usize,
    pub sectors: usize, // 5 * frames (столбцы 5x5 фреймов)
    pub in_dim: usize,
    pub num_classes: usize,
    pub d_state: usize, // 25 * frames
    // Параметры каскада O(4) -> SO(5): wv [M * 4 * 4 * d_in], w_beta [M * d_in]
    pub wv: Vec<f32>,
    pub w_beta: Vec<f32>,
    // Оптимизаторы: RMSProp+Momentum для wv и покадрового w_beta
    m_wv: Vec<f32>,
    v_wv: Vec<f32>,
    m_wbeta: Vec<f32>,
    v_wbeta: Vec<f32>,
    // Головка чтения / прототипы на (S^4)^(5M)
    pub out_w: Vec<f32>, // [C * 25M]
    pub out_b: Vec<f32>, // [C]
    v_out_w: Vec<f32>,
    v_out_b: Vec<f32>,
    pub proto_seen: Vec<u32>,
    proto_dirty: Vec<u8>,
    // Ортонормированное состояние, якорь Q_0 = I_5 и сферическое чтение b
    pub u0_anchor: Vec<f32>, // [25M]
    pub u: Vec<f32>,         // [25M]
    pub b: Vec<f32>,         // [25M]
    pub inv_rho: Vec<f32>,   // [5M]
    // Точные RTRL чувствительности O(M * d_in)
    pub p_wv: Vec<f32>,    // [5M * 4 * 4 * d_in * 5]
    pub p_wbeta: Vec<f32>, // [5M * d_in * 5]
    // Предаллоцированные буферы
    logits: Vec<f32>,
    probs: Vec<f32>,
    psis: Vec<f32>,
    grad_u: Vec<f32>,
    pub frame_acc_ema: Vec<f32>,
    frame_logits: Vec<f32>,
    frame_probs: Vec<f32>,
    frame_psis: Vec<f32>,
    // Гиперсферические прототипы и отложенный регуляризатор Мины №2
    pub proto_metric_mode: bool,
    pub proto_tau: f32,
    pub proto_center_beta: f32,
    pub proto_ema: f32,
    pub proto_rep_mu: f32,
    pub proto_rep_lr: f32,
    pub parity_reg_weight: f32,
    pub parity_warmup_steps: u64,
    pub parity_ramp_steps: u64,
    pub train_learn_steps: u64,
    pub seu_fault_count: u64,
}

#[axiom_zero]
impl NonAbelianO4Rtrl {
    pub fn new(frames: usize, in_dim: usize, num_classes: usize, seed: u64) -> Self {
        let m_cnt = frames.clamp(1, 64);
        let s_cnt = m_cnt.saturating_mul(5);
        let d_state = m_cnt.saturating_mul(25);
        let d_in = in_dim.clamp(1, 256);
        let nc = num_classes.clamp(2, 1024);

        let wv_len = m_cnt.saturating_mul(16).saturating_mul(d_in);
        let wb_len = m_cnt.saturating_mul(d_in);

        let mut rng = splitmix64(seed ^ 0x0404_F4A3_0001);
        let mut wv = vec![0.0_f32; wv_len];
        let mut w_beta = vec![1.00_f32; wb_len];
        let mut u0_anchor = vec![0.0_f32; d_state];

        for m in 0..m_cnt {
            // 1. Начальный базис Q_{0,m} = I_5
            let fm_off = m.saturating_mul(25);
            for a in 0..5usize {
                let col_off = fm_off.saturating_add(a.saturating_mul(5));
                let _ = total_set(&mut u0_anchor, col_off.saturating_add(a), 1.0);
            }

            // 2. Pillar 2: Ортогональная 4D Gram–Schmidt инициализация для всех 4 отражений k in 0..4
            let sc_v = sanitize_f32(0.80 + 0.10 * (m as f32));
            for k in 0..4usize {
                let mut v_b0 = (1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
                let mut v_b1 = (0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32);
                let mut v_b2 = (0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32);
                for j in 0..d_in {
                    let j4 = total_rem_usize(j, 4, 0);
                    let v4 = ortho_dir_4d_next(&mut rng, j4, &mut v_b0, &mut v_b1, &mut v_b2);
                    let base_vk = m.saturating_mul(4).saturating_add(k).saturating_mul(4);
                    let _ = total_set(
                        &mut wv,
                        base_vk.saturating_mul(d_in) + j,
                        sanitize_bounded_f32(v4.0 * sc_v, 2.0),
                    );
                    let _ = total_set(
                        &mut wv,
                        (base_vk + 1).saturating_mul(d_in) + j,
                        sanitize_bounded_f32(v4.1 * sc_v, 2.0),
                    );
                    let _ = total_set(
                        &mut wv,
                        (base_vk + 2).saturating_mul(d_in) + j,
                        sanitize_bounded_f32(v4.2 * sc_v, 2.0),
                    );
                    let _ = total_set(
                        &mut wv,
                        (base_vk + 3).saturating_mul(d_in) + j,
                        sanitize_bounded_f32(v4.3 * sc_v, 2.0),
                    );
                }
            }
        }
        for v in w_beta.iter_mut() {
            *v = 1.00_f32;
        }

        let out_scale =
            sanitize_f32(0.15 * inv_sqrt_newton_f32((d_state.max(1)) as f32)).clamp(0.02, 0.25);
        let mut out_w = vec![0.0_f32; nc.saturating_mul(d_state)];
        for v in out_w.iter_mut() {
            *v = sanitize_f32(rand_signed_f32(&mut rng) * out_scale);
        }

        let mut inst = Self {
            frames: m_cnt,
            sectors: s_cnt,
            in_dim: d_in,
            num_classes: nc,
            d_state,
            wv,
            w_beta,
            m_wv: vec![0.0_f32; wv_len],
            v_wv: vec![0.04_f32; wv_len],
            m_wbeta: vec![0.0_f32; wb_len],
            v_wbeta: vec![0.04_f32; wb_len],
            out_w,
            out_b: vec![0.0_f32; nc],
            v_out_w: vec![0.04_f32; nc.saturating_mul(d_state)],
            v_out_b: vec![0.04_f32; nc],
            proto_seen: Vec::new(),
            proto_dirty: Vec::new(),
            u0_anchor,
            u: vec![0.0_f32; d_state],
            b: vec![0.0_f32; d_state],
            inv_rho: vec![1.0_f32; s_cnt],
            p_wv: vec![0.0_f32; s_cnt.saturating_mul(16).saturating_mul(d_in).saturating_mul(5)],
            p_wbeta: vec![0.0_f32; s_cnt.saturating_mul(d_in).saturating_mul(5)],
            logits: vec![0.0_f32; nc],
            probs: vec![0.0_f32; nc],
            psis: vec![1.0_f32; nc],
            grad_u: vec![0.0_f32; d_state],
            frame_acc_ema: vec![0.5_f32; m_cnt],
            frame_logits: vec![0.0_f32; nc],
            frame_probs: vec![0.0_f32; nc],
            frame_psis: vec![1.0_f32; nc],
            proto_metric_mode: false,
            proto_tau: 12.0,
            proto_center_beta: 1.4,
            proto_ema: 0.08,
            proto_rep_mu: 0.55,
            proto_rep_lr: 0.0,
            parity_reg_weight: 0.04,
            parity_warmup_steps: 32000,
            parity_ramp_steps: 12000,
            train_learn_steps: 0,
            seu_fault_count: 0,
        };
        inst.reset();
        inst
    }

    /// Активирует гиперсферическую метрическую головку на `(S^4)^(5M)` без оракулов, `exp` и `sqrt`:
    /// - `rational_probs_p4_into` (`0% exp`);
    /// - Pillar 2 полиномиальное отталкивание прототипов (`step123`);
    /// - Отложенный прогрев регуляризатора чётности `beta(2-beta)`.
    pub fn enable_o4_proto_metric(
        &mut self,
        tau: f32,
        center_beta: f32,
        proto_ema: f32,
        rep_mu: f32,
        rep_lr: f32,
    ) {
        self.proto_metric_mode = true;
        self.proto_tau = sanitize_f32(tau).clamp(1.0, 32.0);
        self.proto_center_beta = sanitize_f32(center_beta).clamp(0.0, 4.0);
        self.proto_ema = sanitize_f32(proto_ema).clamp(0.0, 0.5);
        self.proto_rep_mu = sanitize_f32(rep_mu).clamp(0.0, 0.95);
        self.proto_rep_lr = sanitize_f32(rep_lr).clamp(0.0, 0.5);
        self.parity_reg_weight = 0.04;
        self.parity_warmup_steps = 32000;
        self.parity_ramp_steps = 12000;
        self.train_learn_steps = 0;

        let d = self.d_state;
        for c in 0..self.num_classes {
            let off = c.saturating_mul(d);
            for s in 0..self.sectors {
                let so = off.saturating_add(s.saturating_mul(5));
                let p0 = total_get_copy(&self.out_w, so, 1.0);
                let p1 = total_get_copy(&self.out_w, so + 1, 0.0);
                let p2 = total_get_copy(&self.out_w, so + 2, 0.0);
                let p3 = total_get_copy(&self.out_w, so + 3, 0.0);
                let p4 = total_get_copy(&self.out_w, so + 4, 0.0);
                let n2 = p0 * p0 + p1 * p1 + p2 * p2 + p3 * p3 + p4 * p4;
                if n2 > 1e-12 && n2.is_finite() {
                    let inv = inv_sqrt_newton_f32(n2);
                    let _ = total_set(&mut self.out_w, so, sanitize_f32(p0 * inv));
                    let _ = total_set(&mut self.out_w, so + 1, sanitize_f32(p1 * inv));
                    let _ = total_set(&mut self.out_w, so + 2, sanitize_f32(p2 * inv));
                    let _ = total_set(&mut self.out_w, so + 3, sanitize_f32(p3 * inv));
                    let _ = total_set(&mut self.out_w, so + 4, sanitize_f32(p4 * inv));
                } else {
                    let _ = total_set(&mut self.out_w, so, 1.0);
                    let _ = total_set(&mut self.out_w, so + 1, 0.0);
                    let _ = total_set(&mut self.out_w, so + 2, 0.0);
                    let _ = total_set(&mut self.out_w, so + 3, 0.0);
                    let _ = total_set(&mut self.out_w, so + 4, 0.0);
                }
            }
        }
        self.out_b.fill(0.0);
        // В прототипном режиме v_out_w/v_out_b не используются (-35.2 KB RAM при C=120, M=3, бит-идентично!)
        self.v_out_w = Vec::new();
        self.v_out_b = Vec::new();
        self.proto_seen = vec![0u32; self.num_classes];
        self.proto_dirty = vec![0u8; self.num_classes];
        self.m_wv.fill(0.0);
        self.v_wv.fill(0.04);
        self.m_wbeta.fill(0.0);
        self.v_wbeta.fill(0.04);
        self.frame_acc_ema.fill(0.5);
        self.reset();
    }

    pub fn set_parity_schedule(&mut self, weight: f32, warmup_steps: u64, ramp_steps: u64) {
        self.parity_reg_weight = sanitize_f32(weight).clamp(0.0, 0.5);
        self.parity_warmup_steps = warmup_steps;
        self.parity_ramp_steps = ramp_steps.max(1);
    }

    #[inline]
    fn compute_frame_gates(&self, gates_out: &mut [f32; 64]) {
        let nf = self.frames.min(64);
        if nf == 0 {
            return;
        }
        if !self.proto_metric_mode {
            for m in 0..nf {
                let _ = total_set(gates_out, m, 1.0);
            }
            return;
        }
        let mut sum_w = 0.0_f32;
        for m in 0..nf {
            let a = sanitize_f32(total_get_copy(&self.frame_acc_ema, m, 0.5)).clamp(0.0, 1.0);
            let ex = (a - 0.35).max(0.0);
            let ex2 = ex * ex;
            let w = sanitize_f32(ex2 * ex2 + 1e-4);
            let _ = total_set(gates_out, m, w);
            sum_w = sanitize_f32(sum_w + w);
        }
        let sc = total_div_f32(nf as f32, sum_w, 1.0);
        for m in 0..nf {
            let w = total_get_copy(gates_out, m, 1.0);
            let _ = total_set(gates_out, m, sanitize_f32(w * sc));
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        (self.wv.len()
            + self.w_beta.len()
            + self.m_wv.len()
            + self.v_wv.len()
            + self.m_wbeta.len()
            + self.v_wbeta.len()
            + self.out_w.len()
            + self.out_b.len()
            + self.v_out_w.len()
            + self.v_out_b.len()
            + self.proto_seen.len()
            + self.u0_anchor.len()
            + self.u.len()
            + self.b.len()
            + self.inv_rho.len()
            + self.p_wv.len()
            + self.p_wbeta.len()
            + self.logits.len()
            + self.probs.len()
            + self.psis.len()
            + self.grad_u.len()
            + self.frame_acc_ema.len()
            + self.frame_logits.len()
            + self.frame_probs.len()
            + self.frame_psis.len())
        .saturating_mul(4)
            + self.proto_dirty.len()
            + 128
    }

    pub fn reset(&mut self) {
        let d = self.d_state;
        for i in 0..d {
            let v = total_get_copy(&self.u0_anchor, i, 0.0);
            let _ = total_set(&mut self.u, i, v);
        }
        for s in 0..self.sectors {
            let off = s.saturating_mul(5);
            let u0 = total_get_copy(&self.u, off, 0.0);
            let u1 = total_get_copy(&self.u, off + 1, 0.0);
            let u2 = total_get_copy(&self.u, off + 2, 0.0);
            let u3 = total_get_copy(&self.u, off + 3, 0.0);
            let u4 = total_get_copy(&self.u, off + 4, 0.0);
            let b_s2 = sanitize_f32(u0 * u0 + u1 * u1 + u2 * u2 + u3 * u3 + u4 * u4 + 1e-6);
            let inv_r = inv_sqrt_newton_f32(b_s2);
            let _ = total_set(&mut self.inv_rho, s, inv_r);
            let _ = total_set(&mut self.b, off, sanitize_f32(u0 * inv_r));
            let _ = total_set(&mut self.b, off + 1, sanitize_f32(u1 * inv_r));
            let _ = total_set(&mut self.b, off + 2, sanitize_f32(u2 * inv_r));
            let _ = total_set(&mut self.b, off + 3, sanitize_f32(u3 * inv_r));
            let _ = total_set(&mut self.b, off + 4, sanitize_f32(u4 * inv_r));
        }
        self.p_wv.fill(0.0);
        self.p_wbeta.fill(0.0);
    }

    #[inline]
    fn current_parity_lambda(&self) -> f32 {
        if self.parity_reg_weight <= 0.0 || self.train_learn_steps < self.parity_warmup_steps {
            return 0.0;
        }
        let elapsed = self.train_learn_steps.saturating_sub(self.parity_warmup_steps);
        let ramp = self.parity_ramp_steps.max(1);
        if elapsed >= ramp {
            self.parity_reg_weight
        } else {
            sanitize_f32(
                self.parity_reg_weight * total_div_f32(elapsed as f32, ramp as f32, 0.0),
            )
        }
    }

    /// Честный живой ненормализованный дрейф ортонормированности `max_m ||Q_{t,m}^T Q_{t,m} - I_5||_F` без `.sqrt()`.
    pub fn max_orth_drift_4d(&self) -> f32 {
        let mut max_f = 0.0_f32;
        for m in 0..self.frames {
            let b = m.saturating_mul(25);
            let mut s = 0.0_f32;
            for i in 0..5usize {
                for j in 0..5usize {
                    let ci = b.saturating_add(i.saturating_mul(5));
                    let cj = b.saturating_add(j.saturating_mul(5));
                    let mut dot = 0.0_f32;
                    for k in 0..5usize {
                        let qki = total_get_copy(&self.u, ci + k, 0.0);
                        let qkj = total_get_copy(&self.u, cj + k, 0.0);
                        dot = sanitize_f32(dot + qki * qkj);
                    }
                    let diff = sanitize_f32(dot - if i == j { 1.0 } else { 0.0 });
                    s = sanitize_f32(s + diff * diff);
                }
            }
            let f = if s > 1e-30 && s.is_finite() {
                sanitize_f32(s * inv_sqrt_newton_f32(s))
            } else {
                0.0
            };
            if f > max_f {
                max_f = f;
            }
        }
        max_f
    }

    /// Максимальный сферический дрейф по всем столбцам `max_s | ||u_s||_2 - 1 |` без `.sqrt()`.
    pub fn max_sphere_drift_4d(&self) -> f32 {
        let mut max_d = 0.0_f32;
        for s in 0..self.sectors {
            let off = s.saturating_mul(5);
            let u0 = total_get_copy(&self.u, off, 0.0);
            let u1 = total_get_copy(&self.u, off + 1, 0.0);
            let u2 = total_get_copy(&self.u, off + 2, 0.0);
            let u3 = total_get_copy(&self.u, off + 3, 0.0);
            let u4 = total_get_copy(&self.u, off + 4, 0.0);
            let s2 = sanitize_f32(u0 * u0 + u1 * u1 + u2 * u2 + u3 * u3 + u4 * u4);
            let n = if s2 > 1e-30 && s2.is_finite() {
                sanitize_f32(s2 * inv_sqrt_newton_f32(s2))
            } else {
                0.0
            };
            let d = (n - 1.0).abs();
            if d > max_d {
                max_d = d;
            }
        }
        max_d
    }

    #[inline]
    fn mat5_idx(p_mat: &[f32], r: usize, c: usize) -> f32 {
        total_get_copy(p_mat, r.saturating_mul(5).saturating_add(c), 0.0)
    }

    #[inline]
    fn mat5_off_idx(buf: &[f32], base: usize, r: usize, c: usize) -> f32 {
        total_get_copy(buf, base.saturating_add(r.saturating_mul(5)).saturating_add(c), 0.0)
    }

    /// Базисно-инвариантное выделение собственной 1D линии чётности `p_m in S^4` фрейма `m` без оракула:
    /// строим `P_m = mean_j S_j^2`, `Sbar_m = mean_j S_j` и коммутаторный лапласиан
    /// `C_m = mean_{i,j} (G_i G_j - G_j G_i)^T (G_i G_j - G_j G_i)`, подавляющий неприводимые
    /// неабелевы 5D-представления (напр. `V_5 x sgn`) через `A_m = I_5 - 0.25 C_m`.
    /// Возвращает `(p_m, rq_m)`, где `rq_m = mean_j <p_m, G_j p_m>^2 in [0, 1]` — точный
    /// критерий Коши–Шварца `1D`-инвариантности (`1.0 iff G_j p_m = +-p_m` для всех `j`).
    pub fn intrinsic_frame_parity_axis(&self, m: usize) -> (Vec5, f32) {
        if m >= self.frames || self.in_dim == 0 {
            return ((0.0, 0.0, 0.0, 0.0, 1.0), 0.0);
        }
        let d_in = self.in_dim.min(256);
        let ng_comm = d_in.min(12);
        let inv_din = total_div_f32(1.0, d_in as f32, 1.0);
        let mut p_mat = [0.0_f32; 25];
        let mut sbar_mat = [0.0_f32; 25];
        let mut g_mats = [0.0_f32; 300];
        let mut x_oh = [0.0_f32; 256];

        for j in 0..d_in {
            let _ = total_set(&mut x_oh, j, 1.0);
            let ctrl = self.frame_controls(m, x_oh.get(0..d_in).unwrap_or(&[]));
            let _ = total_set(&mut x_oh, j, 0.0);

            let mut g_mat = [0.0_f32; 25];
            for col in 0..5usize {
                let e = (
                    if col == 0 { 1.0 } else { 0.0 },
                    if col == 1 { 1.0 } else { 0.0 },
                    if col == 2 { 1.0 } else { 0.0 },
                    if col == 3 { 1.0 } else { 0.0 },
                    if col == 4 { 1.0 } else { 0.0 },
                );
                let out = apply_hh5_cascade(&ctrl, e);
                let _ = total_set(&mut g_mat, col, out.0);
                let _ = total_set(&mut g_mat, 5 + col, out.1);
                let _ = total_set(&mut g_mat, 10 + col, out.2);
                let _ = total_set(&mut g_mat, 15 + col, out.3);
                let _ = total_set(&mut g_mat, 20 + col, out.4);
            }
            if j < ng_comm {
                let base_j = j.saturating_mul(25);
                for idx in 0..25usize {
                    let _ = total_set(&mut g_mats, base_j.saturating_add(idx), total_get_copy(&g_mat, idx, 0.0));
                }
            }
            let mut s_mat = [0.0_f32; 25];
            for r in 0..5usize {
                for c in 0..5usize {
                    let s_rc = sanitize_f32(
                        0.5 * (Self::mat5_idx(&g_mat, r, c) + Self::mat5_idx(&g_mat, c, r)),
                    );
                    let idx = r.saturating_mul(5).saturating_add(c);
                    let _ = total_set(&mut s_mat, idx, s_rc);
                    let cur_sb = total_get_copy(&sbar_mat, idx, 0.0);
                    let _ = total_set(&mut sbar_mat, idx, sanitize_f32(cur_sb + inv_din * s_rc));
                }
            }
            for r in 0..5usize {
                for c in 0..5usize {
                    let mut sum = 0.0_f32;
                    for k in 0..5usize {
                        sum = sanitize_f32(
                            sum + Self::mat5_idx(&s_mat, r, k) * Self::mat5_idx(&s_mat, k, c),
                        );
                    }
                    let idx = r.saturating_mul(5).saturating_add(c);
                    let cur = total_get_copy(&p_mat, idx, 0.0);
                    let _ = total_set(&mut p_mat, idx, sanitize_f32(cur + inv_din * sum));
                }
            }
        }

        // Коммутаторный лапласиан C_m = (1 / ng^2) sum_{i,j} (G_i G_j - G_j G_i)^T (G_i G_j - G_j G_i)
        let mut c_mat = [0.0_f32; 25];
        if ng_comm > 1 {
            let inv_ng2 = total_div_f32(1.0, (ng_comm.saturating_mul(ng_comm)) as f32, 1.0);
            let mut comm_ij = [0.0_f32; 25];
            for i in 0..ng_comm {
                let bi = i.saturating_mul(25);
                for j in (i + 1)..ng_comm {
                    let bj = j.saturating_mul(25);
                    for r in 0..5usize {
                        for c in 0..5usize {
                            let mut ab = 0.0_f32;
                            let mut ba = 0.0_f32;
                            for k in 0..5usize {
                                ab = sanitize_f32(
                                    ab + Self::mat5_off_idx(&g_mats, bi, r, k)
                                        * Self::mat5_off_idx(&g_mats, bj, k, c),
                                );
                                ba = sanitize_f32(
                                    ba + Self::mat5_off_idx(&g_mats, bj, r, k)
                                        * Self::mat5_off_idx(&g_mats, bi, k, c),
                                );
                            }
                            let _ = total_set(
                                &mut comm_ij,
                                r.saturating_mul(5).saturating_add(c),
                                sanitize_f32(ab - ba),
                            );
                        }
                    }
                    for r in 0..5usize {
                        for c in 0..5usize {
                            let mut dtd = 0.0_f32;
                            for k in 0..5usize {
                                dtd = sanitize_f32(
                                    dtd + Self::mat5_idx(&comm_ij, k, r) * Self::mat5_idx(&comm_ij, k, c),
                                );
                            }
                            let idx = r.saturating_mul(5).saturating_add(c);
                            let cur = total_get_copy(&c_mat, idx, 0.0);
                            let _ = total_set(&mut c_mat, idx, sanitize_f32(cur + 2.0 * inv_ng2 * dtd));
                        }
                    }
                }
            }
        }
        let mut a_mat = [0.0_f32; 25];
        for r in 0..5usize {
            for c in 0..5usize {
                let idx = r.saturating_mul(5).saturating_add(c);
                let eye = if r == c { 1.0_f32 } else { 0.0_f32 };
                let _ = total_set(
                    &mut a_mat,
                    idx,
                    sanitize_f32(eye - 0.25 * total_get_copy(&c_mat, idx, 0.0)),
                );
            }
        }

        // M_m = A_m P_m (I_5 - 0.25 Sbar_m^2) P_m A_m: строго симметрична и PSD
        let mut k_mat = [0.0_f32; 25];
        for r in 0..5usize {
            for c in 0..5usize {
                let mut sb2 = 0.0_f32;
                for k in 0..5usize {
                    sb2 = sanitize_f32(
                        sb2 + Self::mat5_idx(&sbar_mat, r, k) * Self::mat5_idx(&sbar_mat, k, c),
                    );
                }
                let eye = if r == c { 1.0_f32 } else { 0.0_f32 };
                let _ = total_set(
                    &mut k_mat,
                    r.saturating_mul(5).saturating_add(c),
                    sanitize_f32(eye - 0.25 * sb2),
                );
            }
        }
        let mat5_mul = |left: &[f32], right: &[f32], out: &mut [f32; 25]| {
            for r in 0..5usize {
                for c in 0..5usize {
                    let mut sum = 0.0_f32;
                    for k in 0..5usize {
                        sum = sanitize_f32(
                            sum + Self::mat5_idx(left, r, k) * Self::mat5_idx(right, k, c),
                        );
                    }
                    let _ = total_set(out, r.saturating_mul(5).saturating_add(c), sum);
                }
            }
        };
        let mut tmp_ap = [0.0_f32; 25];
        let mut tmp_pk = [0.0_f32; 25];
        let mut tmp_apk = [0.0_f32; 25];
        let mut m_mat = [0.0_f32; 25];
        mat5_mul(&a_mat, &p_mat, &mut tmp_ap);
        mat5_mul(&tmp_ap, &k_mat, &mut tmp_apk);
        mat5_mul(&p_mat, &a_mat, &mut tmp_pk);
        mat5_mul(&tmp_apk, &tmp_pk, &mut m_mat);

        let mul_mat = |mat: &[f32], v: Vec5| -> Vec5 {
            (
                sanitize_f32(
                    Self::mat5_idx(mat, 0, 0) * v.0
                        + Self::mat5_idx(mat, 0, 1) * v.1
                        + Self::mat5_idx(mat, 0, 2) * v.2
                        + Self::mat5_idx(mat, 0, 3) * v.3
                        + Self::mat5_idx(mat, 0, 4) * v.4,
                ),
                sanitize_f32(
                    Self::mat5_idx(mat, 1, 0) * v.0
                        + Self::mat5_idx(mat, 1, 1) * v.1
                        + Self::mat5_idx(mat, 1, 2) * v.2
                        + Self::mat5_idx(mat, 1, 3) * v.3
                        + Self::mat5_idx(mat, 1, 4) * v.4,
                ),
                sanitize_f32(
                    Self::mat5_idx(mat, 2, 0) * v.0
                        + Self::mat5_idx(mat, 2, 1) * v.1
                        + Self::mat5_idx(mat, 2, 2) * v.2
                        + Self::mat5_idx(mat, 2, 3) * v.3
                        + Self::mat5_idx(mat, 2, 4) * v.4,
                ),
                sanitize_f32(
                    Self::mat5_idx(mat, 3, 0) * v.0
                        + Self::mat5_idx(mat, 3, 1) * v.1
                        + Self::mat5_idx(mat, 3, 2) * v.2
                        + Self::mat5_idx(mat, 3, 3) * v.3
                        + Self::mat5_idx(mat, 3, 4) * v.4,
                ),
                sanitize_f32(
                    Self::mat5_idx(mat, 4, 0) * v.0
                        + Self::mat5_idx(mat, 4, 1) * v.1
                        + Self::mat5_idx(mat, 4, 2) * v.2
                        + Self::mat5_idx(mat, 4, 3) * v.3
                        + Self::mat5_idx(mat, 4, 4) * v.4,
                ),
            )
        };

        let mut best_v: Vec5 = (0.0, 0.0, 0.0, 0.0, 1.0);
        let mut best_score = -1.0_f32;
        for seed_id in 0..8usize {
            let raw_seed: Vec5 = match seed_id {
                0 => (1.0, 0.0, 0.0, 0.0, 0.0),
                1 => (0.0, 1.0, 0.0, 0.0, 0.0),
                2 => (0.0, 0.0, 1.0, 0.0, 0.0),
                3 => (0.0, 0.0, 0.0, 1.0, 0.0),
                4 => (0.0, 0.0, 0.0, 0.0, 1.0),
                5 => (1.0, 1.0, 1.0, 1.0, 1.0),
                6 => (1.0, -1.0, 1.0, -1.0, 0.0),
                _ => (0.0, 1.0, -1.0, 1.0, -1.0),
            };
            let sn2 = raw_seed.0 * raw_seed.0
                + raw_seed.1 * raw_seed.1
                + raw_seed.2 * raw_seed.2
                + raw_seed.3 * raw_seed.3
                + raw_seed.4 * raw_seed.4;
            let inv_s = inv_sqrt_newton_f32(sn2.max(1e-12));
            let mut v: Vec5 = (
                raw_seed.0 * inv_s,
                raw_seed.1 * inv_s,
                raw_seed.2 * inv_s,
                raw_seed.3 * inv_s,
                raw_seed.4 * inv_s,
            );
            for _ in 0..40usize {
                let pv = mul_mat(&m_mat, v);
                let n2 = sanitize_f32(
                    pv.0 * pv.0 + pv.1 * pv.1 + pv.2 * pv.2 + pv.3 * pv.3 + pv.4 * pv.4,
                );
                if n2 > 1e-12 && n2.is_finite() {
                    let inv = inv_sqrt_newton_f32(n2);
                    v = (
                        sanitize_f32(pv.0 * inv),
                        sanitize_f32(pv.1 * inv),
                        sanitize_f32(pv.2 * inv),
                        sanitize_f32(pv.3 * inv),
                        sanitize_f32(pv.4 * inv),
                    );
                }
            }
            let mv = mul_mat(&m_mat, v);
            let score = sanitize_f32(v.0 * mv.0 + v.1 * mv.1 + v.2 * mv.2 + v.3 * mv.3 + v.4 * mv.4);
            if score > best_score {
                best_score = score;
                best_v = v;
            }
        }

        // Точный критерий Коши–Шварца 1D-инвариантности: mean_j <best_v, G_j best_v>^2 in [0, 1]
        let mut rq_sum = 0.0_f32;
        for j in 0..d_in {
            let _ = total_set(&mut x_oh, j, 1.0);
            let ctrl = self.frame_controls(m, x_oh.get(0..d_in).unwrap_or(&[]));
            let _ = total_set(&mut x_oh, j, 0.0);
            let g_bv = apply_hh5_cascade(&ctrl, best_v);
            let dot = sanitize_f32(
                best_v.0 * g_bv.0
                    + best_v.1 * g_bv.1
                    + best_v.2 * g_bv.2
                    + best_v.3 * g_bv.3
                    + best_v.4 * g_bv.4,
            );
            rq_sum = sanitize_f32(rq_sum + dot * dot);
        }
        let best_rq = sanitize_f32(inv_din * rq_sum).clamp(0.0, 1.0);
        (best_v, best_rq)
    }

    /// Возвращает коэффициент Рэлея `p_m^T P_m p_m in [0, 1]` линии чётности фрейма `m`.
    pub fn frame_parity_line_rq(&self, m: usize) -> f32 {
        self.intrinsic_frame_parity_axis(m).1
    }

    /// Записывает базисно-инвариантную собственную чётность `beta_eff = clamp(1 - <p_m, G_{m,ch} p_m>, 0, 2)`
    /// для входного канала `ch` во всех фреймах (`~2.0` для нечётных `det=-1`, `~0.0` для чётных `det=+1`).
    pub fn channel_betas_into(&self, ch: usize, out: &mut [f32]) {
        if ch >= self.in_dim || ch >= 256 {
            out.fill(0.0);
            return;
        }
        let mut x_oh = [0.0_f32; 256];
        let _ = total_set(&mut x_oh, ch, 1.0);
        let x_sl = x_oh.get(0..self.in_dim).unwrap_or(&[]);
        let lim = self.frames.min(out.len());
        for m in 0..lim {
            let (p_m, _rq) = self.intrinsic_frame_parity_axis(m);
            let ctrl = self.frame_controls(m, x_sl);
            let g_pm = apply_hh5_cascade(&ctrl, p_m);
            let lam_ch = sanitize_f32(
                p_m.0 * g_pm.0 + p_m.1 * g_pm.1 + p_m.2 * g_pm.2 + p_m.3 * g_pm.3 + p_m.4 * g_pm.4,
            );
            let _ = total_set(out, m, sanitize_f32(1.0 - lam_ch).clamp(0.0, 2.0));
        }
    }

    pub fn scrub_weight_seu(&mut self) -> u64 {
        let mut faults = 0u64;
        let scrub_arr = |arr: &mut [f32], bound: f32, reset_val: f32, faults_acc: &mut u64| {
            for v in arr.iter_mut() {
                if !v.is_finite() || v.abs() > bound {
                    *v = reset_val;
                    *faults_acc = faults_acc.saturating_add(1);
                }
            }
        };
        scrub_arr(&mut self.wv, 6.1, 0.0, &mut faults);
        scrub_arr(&mut self.w_beta, 6.1, 1.0, &mut faults);
        scrub_arr(&mut self.m_wv, 32.0, 0.0, &mut faults);
        scrub_arr(&mut self.m_wbeta, 32.0, 0.0, &mut faults);
        scrub_arr(&mut self.v_wbeta, 1e6, 0.04, &mut faults);
        for v in self.frame_acc_ema.iter_mut() {
            if !v.is_finite() || *v < 0.0 || *v > 1.0 {
                *v = 0.5;
                faults = faults.saturating_add(1);
            }
        }

        if self.proto_metric_mode {
            let d = self.d_state;
            for c in 0..self.num_classes {
                let off = c.saturating_mul(d);
                for s in 0..self.sectors {
                    let so = off.saturating_add(s.saturating_mul(5));
                    let w0 = total_get_copy(&self.out_w, so, 0.0);
                    let w1 = total_get_copy(&self.out_w, so + 1, 0.0);
                    let w2 = total_get_copy(&self.out_w, so + 2, 0.0);
                    let w3 = total_get_copy(&self.out_w, so + 3, 0.0);
                    let w4 = total_get_copy(&self.out_w, so + 4, 0.0);
                    if !w0.is_finite()
                        || w0.abs() > 2.1
                        || !w1.is_finite()
                        || w1.abs() > 2.1
                        || !w2.is_finite()
                        || w2.abs() > 2.1
                        || !w3.is_finite()
                        || w3.abs() > 2.1
                        || !w4.is_finite()
                        || w4.abs() > 2.1
                    {
                        faults = faults.saturating_add(1);
                        let s0 = sanitize_bounded_f32(w0, 2.0);
                        let s1 = sanitize_bounded_f32(w1, 2.0);
                        let s2 = sanitize_bounded_f32(w2, 2.0);
                        let s3 = sanitize_bounded_f32(w3, 2.0);
                        let s4 = sanitize_bounded_f32(w4, 2.0);
                        let n2 = s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4;
                        if n2 > 1e-10 && n2.is_finite() {
                            let inv = inv_sqrt_newton_f32(n2);
                            let _ = total_set(&mut self.out_w, so, sanitize_f32(s0 * inv));
                            let _ = total_set(&mut self.out_w, so + 1, sanitize_f32(s1 * inv));
                            let _ = total_set(&mut self.out_w, so + 2, sanitize_f32(s2 * inv));
                            let _ = total_set(&mut self.out_w, so + 3, sanitize_f32(s3 * inv));
                            let _ = total_set(&mut self.out_w, so + 4, sanitize_f32(s4 * inv));
                        } else {
                            let _ = total_set(&mut self.out_w, so, 1.0);
                            let _ = total_set(&mut self.out_w, so + 1, 0.0);
                            let _ = total_set(&mut self.out_w, so + 2, 0.0);
                            let _ = total_set(&mut self.out_w, so + 3, 0.0);
                            let _ = total_set(&mut self.out_w, so + 4, 0.0);
                        }
                    }
                }
            }
        } else {
            scrub_arr(&mut self.out_w, 8.1, 0.0, &mut faults);
        }
        scrub_arr(&mut self.out_b, 6.1, 0.0, &mut faults);
        faults
    }

    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        if self.u.is_empty() {
            return;
        }
        let idx = total_rem_usize(reg_idx, self.u.len(), 0);
        let mut cur = total_get_copy(&self.u, idx, 1.25);
        if cur.is_finite() && (cur.abs() < 0.25 || cur.abs() >= 1.9) {
            cur = if cur < 0.0 { -1.25 } else { 1.25 };
        }
        let flipped = f32::from_bits(cur.to_bits() ^ (1u32 << (bit & 31)));
        let _ = total_set(&mut self.u, idx, flipped);
    }

    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let mask = 1u32 << (bit & 31);
        let sel = total_rem_usize(reg_idx, 3, 0);
        let target: &mut [f32] = match sel {
            0 => &mut self.wv,
            1 => &mut self.w_beta,
            _ => &mut self.out_w,
        };
        if target.is_empty() {
            return;
        }
        let idx = total_rem_usize(reg_idx, target.len(), 0);
        let mut cur = total_get_copy(target, idx, 1.25);
        if cur.is_finite() && (cur.abs() < 0.25 || cur.abs() >= 1.9) {
            cur = if cur < 0.0 { -1.25 } else { 1.25 };
        }
        let _ = total_set(target, idx, f32::from_bits(cur.to_bits() ^ mask));
    }

    #[inline]
    fn frame_controls(&self, m: usize, raw_x: &[f32]) -> O4FrameCtrl {
        let d_in = self.in_dim;
        let x_len = raw_x.len().min(d_in);

        let mut v0 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        let mut v1 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        let mut v2 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        let mut v3 = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        let mut raw_beta = 0.0_f32;

        let b0 = m.saturating_mul(16);
        let b1 = b0.saturating_add(4);
        let b2 = b0.saturating_add(8);
        let b3 = b0.saturating_add(12);

        let wb_base = m.saturating_mul(d_in);
        for j in 0..x_len {
            let xj = sanitize_bounded_f32(total_get_copy(raw_x, j, 0.0), 8.0);
            raw_beta = sanitize_f32(raw_beta + total_get_copy(&self.w_beta, wb_base + j, 1.0) * xj);

            v0.0 = sanitize_f32(v0.0 + total_get_copy(&self.wv, b0.saturating_mul(d_in) + j, 0.0) * xj);
            v0.1 = sanitize_f32(v0.1 + total_get_copy(&self.wv, (b0 + 1).saturating_mul(d_in) + j, 0.0) * xj);
            v0.2 = sanitize_f32(v0.2 + total_get_copy(&self.wv, (b0 + 2).saturating_mul(d_in) + j, 0.0) * xj);
            v0.3 = sanitize_f32(v0.3 + total_get_copy(&self.wv, (b0 + 3).saturating_mul(d_in) + j, 0.0) * xj);

            v1.0 = sanitize_f32(v1.0 + total_get_copy(&self.wv, b1.saturating_mul(d_in) + j, 0.0) * xj);
            v1.1 = sanitize_f32(v1.1 + total_get_copy(&self.wv, (b1 + 1).saturating_mul(d_in) + j, 0.0) * xj);
            v1.2 = sanitize_f32(v1.2 + total_get_copy(&self.wv, (b1 + 2).saturating_mul(d_in) + j, 0.0) * xj);
            v1.3 = sanitize_f32(v1.3 + total_get_copy(&self.wv, (b1 + 3).saturating_mul(d_in) + j, 0.0) * xj);

            v2.0 = sanitize_f32(v2.0 + total_get_copy(&self.wv, b2.saturating_mul(d_in) + j, 0.0) * xj);
            v2.1 = sanitize_f32(v2.1 + total_get_copy(&self.wv, (b2 + 1).saturating_mul(d_in) + j, 0.0) * xj);
            v2.2 = sanitize_f32(v2.2 + total_get_copy(&self.wv, (b2 + 2).saturating_mul(d_in) + j, 0.0) * xj);
            v2.3 = sanitize_f32(v2.3 + total_get_copy(&self.wv, (b2 + 3).saturating_mul(d_in) + j, 0.0) * xj);

            v3.0 = sanitize_f32(v3.0 + total_get_copy(&self.wv, b3.saturating_mul(d_in) + j, 0.0) * xj);
            v3.1 = sanitize_f32(v3.1 + total_get_copy(&self.wv, (b3 + 1).saturating_mul(d_in) + j, 0.0) * xj);
            v3.2 = sanitize_f32(v3.2 + total_get_copy(&self.wv, (b3 + 2).saturating_mul(d_in) + j, 0.0) * xj);
            v3.3 = sanitize_f32(v3.3 + total_get_copy(&self.wv, (b3 + 3).saturating_mul(d_in) + j, 0.0) * xj);
        }

        let beta = sanitize_f32(raw_beta).clamp(0.0, 2.0);
        let db_draw = if raw_beta > 0.0 && raw_beta < 2.0 {
            1.0_f32
        } else {
            0.25_f32
        };
        let n3_2 = sanitize_f32(v3.0 * v3.0 + v3.1 * v3.1 + v3.2 * v3.2 + v3.3 * v3.3 + 1e-8);
        let inv_n3 = inv_sqrt_newton_f32(n3_2);
        let vh3 = (
            sanitize_f32(v3.0 * inv_n3),
            sanitize_f32(v3.1 * inv_n3),
            sanitize_f32(v3.2 * inv_n3),
            sanitize_f32(v3.3 * inv_n3),
        );
        let two_m_b = sanitize_f32(2.0 - beta);
        let u0 = (v0.0, v0.1, v0.2, v0.3, 0.0_f32);
        let u1 = (v1.0, v1.1, v1.2, v1.3, 0.0_f32);
        let u2 = (v2.0, v2.1, v2.2, v2.3, 0.0_f32);
        let u3 = (
            sanitize_f32(two_m_b * vh3.0),
            sanitize_f32(two_m_b * vh3.1),
            sanitize_f32(two_m_b * vh3.2),
            sanitize_f32(two_m_b * vh3.3),
            beta,
        );

        let tid_of = |u: Vec5| -> f32 {
            let d = sanitize_f32(u.0 * u.0 + u.1 * u.1 + u.2 * u.2 + u.3 * u.3 + u.4 * u.4 + 1e-12);
            total_div_f32(2.0, d, 0.0)
        };
        O4FrameCtrl {
            u0,
            u1,
            u2,
            u3,
            tid0: tid_of(u0),
            tid1: tid_of(u1),
            tid2: tid_of(u2),
            tid3: tid_of(u3),
            vh3,
            two_m_b_inv_n: sanitize_f32(two_m_b * inv_n3),
            db_draw,
        }
    }

    pub fn step_state_readonly(&self, raw_x: &[f32], u_buf: &mut [f32]) {
        for m in 0..self.frames {
            let ctrl = self.frame_controls(m, raw_x);
            let q_mat = build_hh5_cascade_mat5(&ctrl);
            let fm_off = m.saturating_mul(25);
            for a in 0..5usize {
                let off = fm_off.saturating_add(a.saturating_mul(5));
                let u_col = (
                    sanitize_bounded_f32(total_get_copy(u_buf, off, 0.0), 24.0),
                    sanitize_bounded_f32(total_get_copy(u_buf, off + 1, 0.0), 24.0),
                    sanitize_bounded_f32(total_get_copy(u_buf, off + 2, 0.0), 24.0),
                    sanitize_bounded_f32(total_get_copy(u_buf, off + 3, 0.0), 24.0),
                    sanitize_bounded_f32(total_get_copy(u_buf, off + 4, 0.0), 24.0),
                );
                let res = mat5_mul_vec5(&q_mat, u_col);
                let _ = total_set(u_buf, off, res.0);
                let _ = total_set(u_buf, off + 1, res.1);
                let _ = total_set(u_buf, off + 2, res.2);
                let _ = total_set(u_buf, off + 3, res.3);
                let _ = total_set(u_buf, off + 4, res.4);
            }
        }
    }

    /// Экспорт обученных гиперсферических прототипов `(S^4)^(5M)` в компактный формат `Q1.14` (`i16`, 2x меньше RAM, Lean 4 `step132`).
    pub fn export_proto_q14(&self) -> Vec<i16> {
        let mut out = vec![0i16; self.out_w.len()];
        for (i, &w) in self.out_w.iter().enumerate() {
            let q = (sanitize_f32(w).clamp(-1.0, 1.0) * 16384.0_f32)
                .round()
                .clamp(-16384.0, 16384.0) as i16;
            let _ = total_set(&mut out, i, q);
        }
        out
    }

    /// Read-only инференс `k` шагов вперёд по `Q1.14` (`i16`) прототипам (0 аллокаций, Lean 4 `step132`).
    pub fn predict_k_ahead_q14_readonly(&self, window: &[&[f32]], proto_q14: &[i16]) -> usize {
        let d = self.d_state.min(1600);
        let mut u_ro = [0.0_f32; 1600];
        for i in 0..d {
            let _ = total_set(&mut u_ro, i, total_get_copy(&self.u, i, 0.0));
        }
        let u_slice = u_ro.get_mut(0..d).unwrap_or(&mut []);
        for &raw_x in window {
            self.step_state_readonly(raw_x, u_slice);
        }
        for s in 0..self.sectors {
            let off = s.saturating_mul(5);
            if off + 4 >= d {
                break;
            }
            let u0 = total_get_copy(u_slice, off, 0.0);
            let u1 = total_get_copy(u_slice, off + 1, 0.0);
            let u2 = total_get_copy(u_slice, off + 2, 0.0);
            let u3 = total_get_copy(u_slice, off + 3, 0.0);
            let u4 = total_get_copy(u_slice, off + 4, 0.0);
            let b_s2 = sanitize_f32(u0 * u0 + u1 * u1 + u2 * u2 + u3 * u3 + u4 * u4 + 1e-6);
            let inv_r = inv_sqrt_newton_f32(b_s2);
            let _ = total_set(u_slice, off, sanitize_f32(u0 * inv_r));
            let _ = total_set(u_slice, off + 1, sanitize_f32(u1 * inv_r));
            let _ = total_set(u_slice, off + 2, sanitize_f32(u2 * inv_r));
            let _ = total_set(u_slice, off + 3, sanitize_f32(u3 * inv_r));
            let _ = total_set(u_slice, off + 4, sanitize_f32(u4 * inv_r));
        }
        let nc = self.num_classes;
        let inv_q14 = 6.103515625e-5_f32;
        let inv_s = total_div_f32(self.proto_tau, self.sectors.max(1) as f32, 1.0) * inv_q14;
        let mut gates = [1.0_f32; 64];
        self.compute_frame_gates(&mut gates);
        let nf = self.frames.min(64);
        let mut best_c = 0usize;
        let mut best_l = f32::NEG_INFINITY;
        for c in 0..nc {
            let off = c.saturating_mul(self.d_state);
            let mut dot = 0.0_f32;
            for m in 0..nf {
                let f_off = m.saturating_mul(25);
                if f_off + 25 > d {
                    break;
                }
                let g_m = total_get_copy(&gates, m, 1.0);
                let mut fdot = 0.0_f32;
                for k in 0..25usize {
                    let wq = total_get_copy(proto_q14, off + f_off + k, 0) as f32;
                    let uk = total_get_copy(u_slice, f_off + k, 0.0);
                    fdot += wq * uk;
                }
                dot = sanitize_f32(dot + g_m * fdot);
            }
            let lg = sanitize_f32(inv_s * dot);
            if lg > best_l {
                best_l = lg;
                best_c = c;
            }
        }
        best_c
    }

    /// Строго read-only предсказание после окна входных шагов без мутации `&self` и без аллокаций.
    pub fn predict_k_ahead_readonly(&self, window: &[&[f32]]) -> usize {
        let d = self.d_state.min(1600);
        let mut u_ro = [0.0_f32; 1600];
        for i in 0..d {
            let _ = total_set(&mut u_ro, i, total_get_copy(&self.u, i, 0.0));
        }
        let u_slice = u_ro.get_mut(0..d).unwrap_or(&mut []);
        for &raw_x in window {
            self.step_state_readonly(raw_x, u_slice);
        }
        for s in 0..self.sectors {
            let off = s.saturating_mul(5);
            if off + 4 >= d {
                break;
            }
            let u0 = total_get_copy(u_slice, off, 0.0);
            let u1 = total_get_copy(u_slice, off + 1, 0.0);
            let u2 = total_get_copy(u_slice, off + 2, 0.0);
            let u3 = total_get_copy(u_slice, off + 3, 0.0);
            let u4 = total_get_copy(u_slice, off + 4, 0.0);
            let b_s2 = sanitize_f32(u0 * u0 + u1 * u1 + u2 * u2 + u3 * u3 + u4 * u4 + 1e-6);
            let inv_r = inv_sqrt_newton_f32(b_s2);
            let _ = total_set(u_slice, off, sanitize_f32(u0 * inv_r));
            let _ = total_set(u_slice, off + 1, sanitize_f32(u1 * inv_r));
            let _ = total_set(u_slice, off + 2, sanitize_f32(u2 * inv_r));
            let _ = total_set(u_slice, off + 3, sanitize_f32(u3 * inv_r));
            let _ = total_set(u_slice, off + 4, sanitize_f32(u4 * inv_r));
        }
        let nc = self.num_classes;
        let inv_s = if self.proto_metric_mode {
            total_div_f32(self.proto_tau, self.sectors.max(1) as f32, 1.0)
        } else {
            1.0_f32
        };
        let mut gates = [1.0_f32; 64];
        if self.proto_metric_mode {
            self.compute_frame_gates(&mut gates);
        }
        let nf = self.frames.min(64);
        let mut best_c = 0usize;
        let mut best_l = f32::NEG_INFINITY;
        for c in 0..nc {
            let off = c.saturating_mul(self.d_state);
            let lg = if self.proto_metric_mode {
                let mut dot = 0.0_f32;
                for m in 0..nf {
                    let f_off = m.saturating_mul(25);
                    if f_off + 25 > d {
                        break;
                    }
                    let g_m = total_get_copy(&gates, m, 1.0);
                    let w_m = self.out_w.get(off + f_off..off + f_off + 25).unwrap_or(&[]);
                    let u_m = u_slice.get(f_off..f_off + 25).unwrap_or(&[]);
                    dot = sanitize_f32(dot + g_m * fast_dot(w_m, u_m));
                }
                sanitize_f32(inv_s * dot)
            } else {
                let w_slice = self.out_w.get(off..off + d).unwrap_or(&[]);
                let dot = fast_dot(w_slice, u_slice);
                sanitize_f32(total_get_copy(&self.out_b, c, 0.0) + dot)
            };
            if lg > best_l {
                best_l = lg;
                best_c = c;
            }
        }
        best_c
    }

    pub fn step_forward(&mut self, raw_x: &[f32]) {
        let d_in = self.in_dim;
        let x_len = raw_x.len().min(d_in);

        self.seu_fault_count = self.seu_fault_count.saturating_add(self.scrub_weight_seu());

        for j in 0..x_len {
            let rx = total_get_copy(raw_x, j, 0.0);
            if !rx.is_finite() || rx.abs() > 8.0 {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
        }

        for m in 0..self.frames {
            let ctrl = self.frame_controls(m, raw_x);
            let q_mat = build_hh5_cascade_mat5(&ctrl);
            let fm_off = m.saturating_mul(25);
            let s0 = m.saturating_mul(5);

            for a in 0..5usize {
                let s = s0.saturating_add(a);
                let off = fm_off.saturating_add(a.saturating_mul(5));

                let raw0 = total_get_copy(&self.u, off, 0.0);
                let raw1 = total_get_copy(&self.u, off + 1, 0.0);
                let raw2 = total_get_copy(&self.u, off + 2, 0.0);
                let raw3 = total_get_copy(&self.u, off + 3, 0.0);
                let raw4 = total_get_copy(&self.u, off + 4, 0.0);
                let u_bad = !raw0.is_finite()
                    || raw0.abs() > 24.0
                    || !raw1.is_finite()
                    || raw1.abs() > 24.0
                    || !raw2.is_finite()
                    || raw2.abs() > 24.0
                    || !raw3.is_finite()
                    || raw3.abs() > 24.0
                    || !raw4.is_finite()
                    || raw4.abs() > 24.0;
                let u2_raw = raw0 * raw0 + raw1 * raw1 + raw2 * raw2 + raw3 * raw3 + raw4 * raw4;
                let norm_bad = !u2_raw.is_finite() || (u2_raw - 1.0).abs() > 0.25;
                let q0 = if u_bad || norm_bad {
                    self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                    let sx = sanitize_bounded_f32(raw0, 24.0);
                    let sy = sanitize_bounded_f32(raw1, 24.0);
                    let sz = sanitize_bounded_f32(raw2, 24.0);
                    let sw = sanitize_bounded_f32(raw3, 24.0);
                    let sv = sanitize_bounded_f32(raw4, 24.0);
                    let s2 = sx * sx + sy * sy + sz * sz + sw * sw + sv * sv;
                    if s2 > 1e-8 && s2.is_finite() {
                        let inv_s = inv_sqrt_newton_f32(s2);
                        (
                            sanitize_bounded_f32(sx * inv_s, 2.0),
                            sanitize_bounded_f32(sy * inv_s, 2.0),
                            sanitize_bounded_f32(sz * inv_s, 2.0),
                            sanitize_bounded_f32(sw * inv_s, 2.0),
                            sanitize_bounded_f32(sv * inv_s, 2.0),
                        )
                    } else {
                        (
                            if a == 0 { 1.0 } else { 0.0 },
                            if a == 1 { 1.0 } else { 0.0 },
                            if a == 2 { 1.0 } else { 0.0 },
                            if a == 3 { 1.0 } else { 0.0 },
                            if a == 4 { 1.0 } else { 0.0 },
                        )
                    }
                } else {
                    (raw0, raw1, raw2, raw3, raw4)
                };

                let q1 = hh5_apply_pre(ctrl.u0, ctrl.tid0, q0);
                let q2 = hh5_apply_pre(ctrl.u1, ctrl.tid1, q1);
                let q3 = hh5_apply_pre(ctrl.u2, ctrl.tid2, q2);
                let res = hh5_apply_pre(ctrl.u3, ctrl.tid3, q3);
                let r2 = res.0 * res.0 + res.1 * res.1 + res.2 * res.2 + res.3 * res.3 + res.4 * res.4;

                let new_u = if !(u_bad || norm_bad)
                    && res.0.is_finite()
                    && res.1.is_finite()
                    && res.2.is_finite()
                    && res.3.is_finite()
                    && res.4.is_finite()
                    && r2.is_finite()
                    && (r2 - 1.0).abs() <= 0.25
                {
                    res
                } else {
                    let inv_r = inv_sqrt_newton_f32((r2 + 1e-12).max(1e-12));
                    (
                        sanitize_bounded_f32(res.0 * inv_r, 2.0),
                        sanitize_bounded_f32(res.1 * inv_r, 2.0),
                        sanitize_bounded_f32(res.2 * inv_r, 2.0),
                        sanitize_bounded_f32(res.3 * inv_r, 2.0),
                        sanitize_bounded_f32(res.4 * inv_r, 2.0),
                    )
                };

                let _ = total_set(&mut self.u, off, new_u.0);
                let _ = total_set(&mut self.u, off + 1, new_u.1);
                let _ = total_set(&mut self.u, off + 2, new_u.2);
                let _ = total_set(&mut self.u, off + 3, new_u.3);
                let _ = total_set(&mut self.u, off + 4, new_u.4);

                let b_s2 = sanitize_f32(
                    new_u.0 * new_u.0
                        + new_u.1 * new_u.1
                        + new_u.2 * new_u.2
                        + new_u.3 * new_u.3
                        + new_u.4 * new_u.4
                        + 1e-6,
                );
                let inv_r = inv_sqrt_newton_f32(b_s2);
                let _ = total_set(&mut self.inv_rho, s, inv_r);
                let _ = total_set(&mut self.b, off, sanitize_f32(new_u.0 * inv_r));
                let _ = total_set(&mut self.b, off + 1, sanitize_f32(new_u.1 * inv_r));
                let _ = total_set(&mut self.b, off + 2, sanitize_f32(new_u.2 * inv_r));
                let _ = total_set(&mut self.b, off + 3, sanitize_f32(new_u.3 * inv_r));
                let _ = total_set(&mut self.b, off + 4, sanitize_f32(new_u.4 * inv_r));

                let mut drv = [(0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32); 16];
                for c in 0..4usize {
                    let d0 = hh5_apply_pre(
                        ctrl.u3,
                        ctrl.tid3,
                        hh5_apply_pre(
                            ctrl.u2,
                            ctrl.tid2,
                            hh5_apply_pre(ctrl.u1, ctrl.tid1, hh5_deriv_axis(ctrl.u0, ctrl.tid0, q0, c)),
                        ),
                    );
                    let _ = total_set(&mut drv, c, d0);

                    let d1 = hh5_apply_pre(
                        ctrl.u3,
                        ctrl.tid3,
                        hh5_apply_pre(ctrl.u2, ctrl.tid2, hh5_deriv_axis(ctrl.u1, ctrl.tid1, q1, c)),
                    );
                    let _ = total_set(&mut drv, 4 + c, d1);

                    let d2 = hh5_apply_pre(ctrl.u3, ctrl.tid3, hh5_deriv_axis(ctrl.u2, ctrl.tid2, q2, c));
                    let _ = total_set(&mut drv, 8 + c, d2);
                }

                let g3_0 = hh5_deriv_axis(ctrl.u3, ctrl.tid3, q3, 0);
                let g3_1 = hh5_deriv_axis(ctrl.u3, ctrl.tid3, q3, 1);
                let g3_2 = hh5_deriv_axis(ctrl.u3, ctrl.tid3, q3, 2);
                let g3_3 = hh5_deriv_axis(ctrl.u3, ctrl.tid3, q3, 3);
                let g3_4 = hh5_deriv_axis(ctrl.u3, ctrl.tid3, q3, 4);
                let dot_vg: Vec5 = (
                    sanitize_f32(ctrl.vh3.0 * g3_0.0 + ctrl.vh3.1 * g3_1.0 + ctrl.vh3.2 * g3_2.0 + ctrl.vh3.3 * g3_3.0),
                    sanitize_f32(ctrl.vh3.0 * g3_0.1 + ctrl.vh3.1 * g3_1.1 + ctrl.vh3.2 * g3_2.1 + ctrl.vh3.3 * g3_3.1),
                    sanitize_f32(ctrl.vh3.0 * g3_0.2 + ctrl.vh3.1 * g3_1.2 + ctrl.vh3.2 * g3_2.2 + ctrl.vh3.3 * g3_3.2),
                    sanitize_f32(ctrl.vh3.0 * g3_0.3 + ctrl.vh3.1 * g3_1.3 + ctrl.vh3.2 * g3_2.3 + ctrl.vh3.3 * g3_3.3),
                    sanitize_f32(ctrl.vh3.0 * g3_0.4 + ctrl.vh3.1 * g3_1.4 + ctrl.vh3.2 * g3_2.4 + ctrl.vh3.3 * g3_3.4),
                );
                for c in 0..4usize {
                    let (gc, vhc) = match c {
                        0 => (g3_0, ctrl.vh3.0),
                        1 => (g3_1, ctrl.vh3.1),
                        2 => (g3_2, ctrl.vh3.2),
                        _ => (g3_3, ctrl.vh3.3),
                    };
                    let d3c: Vec5 = (
                        sanitize_f32(ctrl.two_m_b_inv_n * (gc.0 - vhc * dot_vg.0)),
                        sanitize_f32(ctrl.two_m_b_inv_n * (gc.1 - vhc * dot_vg.1)),
                        sanitize_f32(ctrl.two_m_b_inv_n * (gc.2 - vhc * dot_vg.2)),
                        sanitize_f32(ctrl.two_m_b_inv_n * (gc.3 - vhc * dot_vg.3)),
                        sanitize_f32(ctrl.two_m_b_inv_n * (gc.4 - vhc * dot_vg.4)),
                    );
                    let _ = total_set(&mut drv, 12 + c, d3c);
                }
                let dr_beta: Vec5 = (
                    sanitize_f32(ctrl.db_draw * (g3_4.0 - dot_vg.0)),
                    sanitize_f32(ctrl.db_draw * (g3_4.1 - dot_vg.1)),
                    sanitize_f32(ctrl.db_draw * (g3_4.2 - dot_vg.2)),
                    sanitize_f32(ctrl.db_draw * (g3_4.3 - dot_vg.3)),
                    sanitize_f32(ctrl.db_draw * (g3_4.4 - dot_vg.4)),
                );

                for j in 0..d_in {
                    let xj = if j < x_len {
                        sanitize_bounded_f32(total_get_copy(raw_x, j, 0.0), 8.0)
                    } else {
                        0.0_f32
                    };

                    for kc in 0..16usize {
                        let pv_off = ((s.saturating_mul(16) + kc).saturating_mul(d_in) + j).saturating_mul(5);
                        let pv: Vec5 = (
                            total_get_copy(&self.p_wv, pv_off, 0.0),
                            total_get_copy(&self.p_wv, pv_off + 1, 0.0),
                            total_get_copy(&self.p_wv, pv_off + 2, 0.0),
                            total_get_copy(&self.p_wv, pv_off + 3, 0.0),
                            total_get_copy(&self.p_wv, pv_off + 4, 0.0),
                        );
                        let rpv = mat5_mul_vec5(&q_mat, pv);
                        let dv = total_get_copy(&drv, kc, (0.0, 0.0, 0.0, 0.0, 0.0));
                        let _ = total_set(&mut self.p_wv, pv_off, sanitize_bounded_f32(rpv.0 + xj * dv.0, 32.0));
                        let _ = total_set(&mut self.p_wv, pv_off + 1, sanitize_bounded_f32(rpv.1 + xj * dv.1, 32.0));
                        let _ = total_set(&mut self.p_wv, pv_off + 2, sanitize_bounded_f32(rpv.2 + xj * dv.2, 32.0));
                        let _ = total_set(&mut self.p_wv, pv_off + 3, sanitize_bounded_f32(rpv.3 + xj * dv.3, 32.0));
                        let _ = total_set(&mut self.p_wv, pv_off + 4, sanitize_bounded_f32(rpv.4 + xj * dv.4, 32.0));
                    }

                    let pb_off = (s.saturating_mul(d_in) + j).saturating_mul(5);
                    let pb: Vec5 = (
                        total_get_copy(&self.p_wbeta, pb_off, 0.0),
                        total_get_copy(&self.p_wbeta, pb_off + 1, 0.0),
                        total_get_copy(&self.p_wbeta, pb_off + 2, 0.0),
                        total_get_copy(&self.p_wbeta, pb_off + 3, 0.0),
                        total_get_copy(&self.p_wbeta, pb_off + 4, 0.0),
                    );
                    let rpb = mat5_mul_vec5(&q_mat, pb);
                    let _ = total_set(&mut self.p_wbeta, pb_off, sanitize_bounded_f32(rpb.0 + xj * dr_beta.0, 32.0));
                    let _ = total_set(&mut self.p_wbeta, pb_off + 1, sanitize_bounded_f32(rpb.1 + xj * dr_beta.1, 32.0));
                    let _ = total_set(&mut self.p_wbeta, pb_off + 2, sanitize_bounded_f32(rpb.2 + xj * dr_beta.2, 32.0));
                    let _ = total_set(&mut self.p_wbeta, pb_off + 3, sanitize_bounded_f32(rpb.3 + xj * dr_beta.3, 32.0));
                    let _ = total_set(&mut self.p_wbeta, pb_off + 4, sanitize_bounded_f32(rpb.4 + xj * dr_beta.4, 32.0));
                }
            }
        }
    }

    pub fn predict_class(&mut self) -> usize {
        self.seu_fault_count = self.seu_fault_count.saturating_add(self.scrub_weight_seu());
        let nc = self.num_classes;
        let d = self.d_state;
        let mut best_c = 0usize;
        let mut best_l = f32::NEG_INFINITY;
        if self.proto_metric_mode {
            let inv_s = total_div_f32(self.proto_tau, self.sectors.max(1) as f32, 1.0);
            let mut gates = [1.0_f32; 64];
            self.compute_frame_gates(&mut gates);
            let nf = self.frames.min(64);
            for c in 0..nc {
                let off = c.saturating_mul(d);
                let mut dot = 0.0_f32;
                for m in 0..nf {
                    let f_off = m.saturating_mul(25);
                    if f_off + 25 > d {
                        break;
                    }
                    let g_m = total_get_copy(&gates, m, 1.0);
                    let w_m = self.out_w.get(off + f_off..off + f_off + 25).unwrap_or(&[]);
                    let u_m = self.b.get(f_off..f_off + 25).unwrap_or(&[]);
                    dot = sanitize_f32(dot + g_m * fast_dot(w_m, u_m));
                }
                let lg = sanitize_f32(inv_s * dot);
                let _ = total_set(&mut self.logits, c, lg);
                if lg > best_l {
                    best_l = lg;
                    best_c = c;
                }
            }
            rational_probs_p4_into(&self.logits, &mut self.probs, &mut self.psis);
            return best_c;
        }
        for c in 0..nc {
            let off = c.saturating_mul(d);
            let w_slice = self.out_w.get(off..off + d).unwrap_or(&[]);
            let lg = sanitize_f32(total_get_copy(&self.out_b, c, 0.0) + fast_dot(w_slice, &self.b));
            let _ = total_set(&mut self.logits, c, lg);
            if lg > best_l {
                best_l = lg;
                best_c = c;
            }
        }
        rational_probs_and_psi_into(&self.logits, &mut self.probs, &mut self.psis);
        best_c
    }

    fn apply_rtrl_updates(&mut self, raw_x: &[f32], rtrl_lr: f32) {
        let d_in = self.in_dim;
        let x_len = raw_x.len().min(d_in);
        let lam_par = self.current_parity_lambda();

        for m in 0..self.frames {
            let s0 = m.saturating_mul(5);
            let wb_base = m.saturating_mul(d_in);
            let mut raw_beta = 0.0_f32;
            for j in 0..x_len {
                let xj = sanitize_bounded_f32(total_get_copy(raw_x, j, 0.0), 8.0);
                raw_beta = sanitize_f32(raw_beta + total_get_copy(&self.w_beta, wb_base + j, 1.0) * xj);
            }
            let beta_cur = sanitize_f32(raw_beta).clamp(0.0, 2.0);

            for kc in 0..16usize {
                for j in 0..d_in {
                    let mut dv = 0.0_f32;
                    for a in 0..5usize {
                        let s = s0.saturating_add(a);
                        let off = s.saturating_mul(5);
                        let g0 = total_get_copy(&self.grad_u, off, 0.0);
                        let g1 = total_get_copy(&self.grad_u, off + 1, 0.0);
                        let g2 = total_get_copy(&self.grad_u, off + 2, 0.0);
                        let g3 = total_get_copy(&self.grad_u, off + 3, 0.0);
                        let g4 = total_get_copy(&self.grad_u, off + 4, 0.0);
                        let pv_off = ((s.saturating_mul(16) + kc).saturating_mul(d_in) + j).saturating_mul(5);
                        dv = sanitize_f32(
                            dv + g0 * total_get_copy(&self.p_wv, pv_off, 0.0)
                                + g1 * total_get_copy(&self.p_wv, pv_off + 1, 0.0)
                                + g2 * total_get_copy(&self.p_wv, pv_off + 2, 0.0)
                                + g3 * total_get_copy(&self.p_wv, pv_off + 3, 0.0)
                                + g4 * total_get_copy(&self.p_wv, pv_off + 4, 0.0),
                        );
                    }
                    let vidx = (m.saturating_mul(16) + kc).saturating_mul(d_in) + j;
                    rmsprop_momentum_step(
                        dv,
                        &mut self.m_wv,
                        &mut self.v_wv,
                        &mut self.wv,
                        vidx,
                        rtrl_lr,
                        -3.5,
                        3.5,
                    );
                }
            }

            let wb_lr = sanitize_f32(0.12 * rtrl_lr);
            for j in 0..d_in {
                let mut db = 0.0_f32;
                for a in 0..5usize {
                    let s = s0.saturating_add(a);
                    let off = s.saturating_mul(5);
                    let g0 = total_get_copy(&self.grad_u, off, 0.0);
                    let g1 = total_get_copy(&self.grad_u, off + 1, 0.0);
                    let g2 = total_get_copy(&self.grad_u, off + 2, 0.0);
                    let g3 = total_get_copy(&self.grad_u, off + 3, 0.0);
                    let g4 = total_get_copy(&self.grad_u, off + 4, 0.0);
                    let pb_off = (s.saturating_mul(d_in) + j).saturating_mul(5);
                    db = sanitize_f32(
                        db + g0 * total_get_copy(&self.p_wbeta, pb_off, 0.0)
                            + g1 * total_get_copy(&self.p_wbeta, pb_off + 1, 0.0)
                            + g2 * total_get_copy(&self.p_wbeta, pb_off + 2, 0.0)
                            + g3 * total_get_copy(&self.p_wbeta, pb_off + 3, 0.0)
                            + g4 * total_get_copy(&self.p_wbeta, pb_off + 4, 0.0),
                    );
                }
                if lam_par > 0.0 && j < x_len && beta_cur > 0.0 && beta_cur < 2.0 {
                    let xj = sanitize_bounded_f32(total_get_copy(raw_x, j, 0.0), 8.0);
                    db = sanitize_f32(db + 2.0 * lam_par * (beta_cur - 1.0) * xj);
                }
                rmsprop_momentum_step(
                    db,
                    &mut self.m_wbeta,
                    &mut self.v_wbeta,
                    &mut self.w_beta,
                    wb_base.saturating_add(j),
                    wb_lr,
                    0.0,
                    2.0,
                );
            }
        }
    }

    #[inline]
    fn project_grad_b_to_grad_u(&mut self) {
        for s in 0..self.sectors {
            let off = s.saturating_mul(5);
            let inv_r = total_get_copy(&self.inv_rho, s, 1.0);
            let b0 = total_get_copy(&self.b, off, 0.0);
            let b1 = total_get_copy(&self.b, off + 1, 0.0);
            let b2 = total_get_copy(&self.b, off + 2, 0.0);
            let b3 = total_get_copy(&self.b, off + 3, 0.0);
            let b4 = total_get_copy(&self.b, off + 4, 0.0);
            let g0 = total_get_copy(&self.grad_u, off, 0.0);
            let g1 = total_get_copy(&self.grad_u, off + 1, 0.0);
            let g2 = total_get_copy(&self.grad_u, off + 2, 0.0);
            let g3 = total_get_copy(&self.grad_u, off + 3, 0.0);
            let g4 = total_get_copy(&self.grad_u, off + 4, 0.0);
            let dot_gb = sanitize_f32(g0 * b0 + g1 * b1 + g2 * b2 + g3 * b3 + g4 * b4);
            let _ = total_set(&mut self.grad_u, off, sanitize_f32(inv_r * (g0 - dot_gb * b0)));
            let _ = total_set(&mut self.grad_u, off + 1, sanitize_f32(inv_r * (g1 - dot_gb * b1)));
            let _ = total_set(&mut self.grad_u, off + 2, sanitize_f32(inv_r * (g2 - dot_gb * b2)));
            let _ = total_set(&mut self.grad_u, off + 3, sanitize_f32(inv_r * (g3 - dot_gb * b3)));
            let _ = total_set(&mut self.grad_u, off + 4, sanitize_f32(inv_r * (g4 - dot_gb * b4)));
        }
    }

    pub fn step_and_learn_supervised(
        &mut self,
        raw_x: &[f32],
        target: Option<usize>,
        readout_lr: f32,
        rtrl_lr: f32,
    ) -> usize {
        self.step_forward(raw_x);
        let pred = self.predict_class();

        if let Some(y) = target {
            let nc = self.num_classes;
            if y >= nc {
                return pred;
            }
            let d = self.d_state;
            if rtrl_lr > 0.0 || readout_lr > 0.0 {
                self.train_learn_steps = self.train_learn_steps.saturating_add(1);
            }

            if self.proto_metric_mode {
                let seen_y = total_get_copy(&self.proto_seen, y, 0);
                if seen_y == 0 {
                    let off = y.saturating_mul(d);
                    for s in 0..self.sectors {
                        let so = s.saturating_mul(5);
                        let b0 = total_get_copy(&self.b, so, 1.0);
                        let b1 = total_get_copy(&self.b, so + 1, 0.0);
                        let b2 = total_get_copy(&self.b, so + 2, 0.0);
                        let b3 = total_get_copy(&self.b, so + 3, 0.0);
                        let b4 = total_get_copy(&self.b, so + 4, 0.0);
                        let n2 = b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3 + b4 * b4;
                        if n2 > 1e-12 && n2.is_finite() {
                            let inv = inv_sqrt_newton_f32(n2);
                            let _ = total_set(&mut self.out_w, off + so, sanitize_f32(b0 * inv));
                            let _ = total_set(&mut self.out_w, off + so + 1, sanitize_f32(b1 * inv));
                            let _ = total_set(&mut self.out_w, off + so + 2, sanitize_f32(b2 * inv));
                            let _ = total_set(&mut self.out_w, off + so + 3, sanitize_f32(b3 * inv));
                            let _ = total_set(&mut self.out_w, off + so + 4, sanitize_f32(b4 * inv));
                        } else {
                            let _ = total_set(&mut self.out_w, off + so, 1.0);
                            let _ = total_set(&mut self.out_w, off + so + 1, 0.0);
                            let _ = total_set(&mut self.out_w, off + so + 2, 0.0);
                            let _ = total_set(&mut self.out_w, off + so + 3, 0.0);
                            let _ = total_set(&mut self.out_w, off + so + 4, 0.0);
                        }
                    }
                    let _ = total_set(&mut self.proto_seen, y, 1u32);
                    return pred;
                }
                let _ = total_set(&mut self.proto_seen, y, seen_y.saturating_add(1));

                let inv_sf = sanitize_f32(total_div_f32(self.proto_tau, 5.0, 2.0));
                self.grad_u.fill(0.0);
                for m in 0..self.frames {
                    let f_off = m.saturating_mul(25);
                    if f_off + 25 > d {
                        break;
                    }
                    let mut best_cm = 0usize;
                    let mut best_lm = f32::NEG_INFINITY;
                    for c in 0..nc {
                        let off = c.saturating_mul(d).saturating_add(f_off);
                        let w_cm = self.out_w.get(off..off + 25).unwrap_or(&[]);
                        let b_m = self.b.get(f_off..f_off + 25).unwrap_or(&[]);
                        let lg = sanitize_f32(inv_sf * fast_dot(w_cm, b_m));
                        let _ = total_set(&mut self.frame_logits, c, lg);
                        if lg > best_lm {
                            best_lm = lg;
                            best_cm = c;
                        }
                    }
                    if (rtrl_lr > 0.0 || readout_lr > 0.0) && seen_y >= 3 {
                        let old = total_get_copy(&self.frame_acc_ema, m, 0.5);
                        let hit = if best_cm == y { 1.0_f32 } else { 0.0_f32 };
                        let _ = total_set(
                            &mut self.frame_acc_ema,
                            m,
                            sanitize_f32(0.992 * old + 0.008 * hit).clamp(0.0, 1.0),
                        );
                    }
                    rational_probs_p4_into(
                        &self.frame_logits,
                        &mut self.frame_probs,
                        &mut self.frame_psis,
                    );
                    for c in 0..nc {
                        let p_cm = total_get_copy(&self.frame_probs, c, 0.0);
                        let psi_cm = total_get_copy(&self.frame_psis, c, 1.0);
                        let ind = if c == y { 1.0_f32 } else { 0.0_f32 };
                        let err_cm = sanitize_bounded_f32(psi_cm * (ind - p_cm), 3.0);
                        let off = c.saturating_mul(d).saturating_add(f_off);
                        if let (Some(gu_m), Some(w_cm)) = (
                            self.grad_u.get_mut(f_off..f_off + 25),
                            self.out_w.get(off..off + 25),
                        ) {
                            fast_saxpy(gu_m, w_cm, sanitize_f32(err_cm * inv_sf));
                        }
                        if readout_lr > 0.0 {
                            let st = sanitize_f32(readout_lr * err_cm * inv_sf);
                            if st.abs() > 1e-6 {
                                if let (Some(w_mut), Some(b_m)) = (
                                    self.out_w.get_mut(off..off + 25),
                                    self.b.get(f_off..f_off + 25),
                                ) {
                                    fast_saxpy(w_mut, b_m, st);
                                    let _ = total_set(&mut self.proto_dirty, c, 1u8);
                                }
                            }
                        }
                    }
                }
                let y_off = y.saturating_mul(d);
                let ny_f = seen_y as f32;
                let c_gate = total_div_f32(ny_f, ny_f + 4.0, 1.0);
                let center_eff = sanitize_f32(self.proto_center_beta * c_gate);
                if center_eff > 0.0 {
                    for i in 0..d {
                        let pu = total_get_copy(&self.out_w, y_off + i, 0.0);
                        let bi = total_get_copy(&self.b, i, 0.0);
                        let og = total_get_copy(&self.grad_u, i, 0.0);
                        let _ = total_set(
                            &mut self.grad_u,
                            i,
                            sanitize_f32(og + center_eff * (pu - bi)),
                        );
                    }
                }
                let ema_eff = self
                    .proto_ema
                    .max(total_div_f32(0.8, 1.0 + ny_f, self.proto_ema))
                    .min(0.40);
                if ema_eff > 0.0 {
                    let keep = sanitize_f32(1.0 - ema_eff);
                    for i in 0..d {
                        let pu = total_get_copy(&self.out_w, y_off + i, 0.0);
                        let bi = total_get_copy(&self.b, i, 0.0);
                        let _ = total_set(
                            &mut self.out_w,
                            y_off + i,
                            sanitize_f32(keep * pu + ema_eff * bi),
                        );
                    }
                    let _ = total_set(&mut self.proto_dirty, y, 1u8);
                }
                if self.proto_rep_lr > 0.0 {
                    let mut best_sim = -2.0_f32;
                    let mut best_c = y;
                    let mut found_other = false;
                    let inv_sec = total_div_f32(1.0, self.sectors.max(1) as f32, 1.0);
                    let wy_slice = self.out_w.get(y_off..y_off + d).unwrap_or(&[]);
                    for c in 0..nc {
                        if c == y || total_get_copy(&self.proto_seen, c, 0) == 0 {
                            continue;
                        }
                        let off_c = c.saturating_mul(d);
                        let wc_slice = self.out_w.get(off_c..off_c + d).unwrap_or(&[]);
                        let sim = sanitize_f32(fast_dot(wy_slice, wc_slice) * inv_sec);
                        if !found_other || sim > best_sim {
                            best_sim = sim;
                            best_c = c;
                            found_other = true;
                        }
                    }
                    if found_other {
                        let coeff = poly_repulsion_grad_coeff(best_sim, self.proto_rep_mu);
                        if coeff > 0.0 {
                            let step_rep = sanitize_f32(self.proto_rep_lr * coeff);
                            let c_off = best_c.saturating_mul(d);
                            for i in 0..d {
                                let wy_i = total_get_copy(&self.out_w, y_off + i, 0.0);
                                let wc_i = total_get_copy(&self.out_w, c_off + i, 0.0);
                                let _ = total_set(&mut self.out_w, y_off + i, sanitize_f32(wy_i - step_rep * wc_i));
                                let _ = total_set(&mut self.out_w, c_off + i, sanitize_f32(wc_i - step_rep * wy_i));
                            }
                            let _ = total_set(&mut self.proto_dirty, y, 1u8);
                            let _ = total_set(&mut self.proto_dirty, best_c, 1u8);
                        }
                    }
                }
                // Ленивая проекция на (S^4)^(5M): репроецируем ТОЛЬКО реально изменённые классы!
                for c in 0..nc {
                    if total_get_copy(&self.proto_dirty, c, 0) == 0 {
                        continue;
                    }
                    let _ = total_set(&mut self.proto_dirty, c, 0u8);
                    let off = c.saturating_mul(d);
                    for s in 0..self.sectors {
                        let so = off.saturating_add(s.saturating_mul(5));
                        let p0 = total_get_copy(&self.out_w, so, 1.0);
                        let p1 = total_get_copy(&self.out_w, so + 1, 0.0);
                        let p2 = total_get_copy(&self.out_w, so + 2, 0.0);
                        let p3 = total_get_copy(&self.out_w, so + 3, 0.0);
                        let p4 = total_get_copy(&self.out_w, so + 4, 0.0);
                        let n2 = (p0 * p0 + p1 * p1 + p2 * p2 + p3 * p3 + p4 * p4).max(1e-12);
                        let inv = inv_sqrt_newton_f32(n2);
                        let _ = total_set(&mut self.out_w, so, sanitize_f32(p0 * inv));
                        let _ = total_set(&mut self.out_w, so + 1, sanitize_f32(p1 * inv));
                        let _ = total_set(&mut self.out_w, so + 2, sanitize_f32(p2 * inv));
                        let _ = total_set(&mut self.out_w, so + 3, sanitize_f32(p3 * inv));
                        let _ = total_set(&mut self.out_w, so + 4, sanitize_f32(p4 * inv));
                    }
                }
                if rtrl_lr > 0.0 {
                    self.project_grad_b_to_grad_u();
                    self.apply_rtrl_updates(raw_x, rtrl_lr);
                }
                return pred;
            }

            // Обычный линейный режим чтения
            self.grad_u.fill(0.0);
            for c in 0..nc {
                let p = total_get_copy(&self.probs, c, 0.0);
                let psi = total_get_copy(&self.psis, c, 1.0);
                let ind = if c == y { 1.0_f32 } else { 0.0_f32 };
                let err_c = sanitize_bounded_f32(psi * (ind - p), 2.0);
                let off = c.saturating_mul(d);

                if let Some(w_slice) = self.out_w.get(off..off + d) {
                    fast_saxpy(&mut self.grad_u, w_slice, err_c);
                }
                if readout_lr > 0.0 {
                    for i in 0..d {
                        let g = sanitize_f32(err_c * total_get_copy(&self.b, i, 0.0));
                        let ov = total_get_copy(&self.v_out_w, off + i, 0.04);
                        let nv = sanitize_f32(0.98 * ov + 0.02 * g * g).max(0.0);
                        let _ = total_set(&mut self.v_out_w, off + i, nv);
                        let st = sanitize_bounded_f32(g * inv_sqrt_newton_f32(nv + 0.0025), 3.0);
                        let ow = total_get_copy(&self.out_w, off + i, 0.0);
                        let _ = total_set(
                            &mut self.out_w,
                            off + i,
                            sanitize_bounded_f32(ow * 0.99995 + readout_lr * st, 8.0),
                        );
                    }
                    let ovb = total_get_copy(&self.v_out_b, c, 0.04);
                    let nvb = sanitize_f32(0.98 * ovb + 0.02 * err_c * err_c).max(0.0);
                    let _ = total_set(&mut self.v_out_b, c, nvb);
                    let sb = sanitize_bounded_f32(err_c * inv_sqrt_newton_f32(nvb + 0.0025), 3.0);
                    let ob = total_get_copy(&self.out_b, c, 0.0);
                    let _ = total_set(
                        &mut self.out_b,
                        c,
                        sanitize_bounded_f32(ob + 0.25 * readout_lr * sb, 4.0),
                    );
                }
            }
            if rtrl_lr > 0.0 {
                self.project_grad_b_to_grad_u();
                self.apply_rtrl_updates(raw_x, rtrl_lr);
            }
        }
        pred
    }
}

/// Чисто алгебраический 2D рациональный ротор Кэли (`selective = false`: LTI) и
/// алгебраический селективный ротор Кэли (`selective = true`: входозависимый шаг `Delta_t(x_t)`),
/// 100% `#[axiom_zero]`, строго 0% трансцендентных функций (`0% exp/ln/sin/cos/tanh`):
/// - Вентиль шага времени `Delta_t(x_t) = clamp(pos(b_delta + w_delta^T x_t), 0.05, 2.0)` (Lean 4 `step112`).
/// - Строгое BIBO-сжатие `r_t = 0.998 / (1 + Delta_t * alpha^2) in (0, 0.998)` (Lean 4 `step115`).
/// - Ортогональный 2D ротор Кэли `(c, s) = ((1 - w^2)/(1 + w^2), 2w/(1 + w^2))`, `w = Delta_t * omega`
///   с точной производной `d(c, s)/dw = (2 / (1 + w^2)) * (-s, c)` (Lean 4 `step107`, `step114`, `step115`).
/// - Точный онлайн `O(U * d_in)` RTRL, полная Bit-30 State/Weight SEU-защита, поддержка supervised и TD(lambda) RL.
#[derive(Clone)]
pub struct AlgebraicSelectiveCayley {
    pub in_dim: usize,
    pub units: usize,
    pub num_classes: usize,
    pub kappa: f32,
    pub selective: bool,
    pub standardize: bool,
    // Рекуррентные параметры
    pub alpha: Vec<f32>,
    pub omega: Vec<f32>,
    pub b_delta: Vec<f32>,
    pub w_delta: Vec<f32>,
    pub w_bre: Vec<f32>,
    pub w_bim: Vec<f32>,
    // Алгебраические RMSProp-аккумуляторы
    v_alpha: Vec<f32>,
    v_omega: Vec<f32>,
    v_bd: Vec<f32>,
    v_wd: Vec<f32>,
    v_wbr: Vec<f32>,
    v_wbi: Vec<f32>,
    // Головка чтения / Actor
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
    v_ow: Vec<f32>,
    v_ob: Vec<f32>,
    // Состояние и точные RTRL-чувствительности
    pub h_re: Vec<f32>,
    pub h_im: Vec<f32>,
    pub y_re: Vec<f32>,
    pub y_im: Vec<f32>,
    pub rho: Vec<f32>,
    pub p_a_re: Vec<f32>,
    pub p_a_im: Vec<f32>,
    pub p_w_re: Vec<f32>,
    pub p_w_im: Vec<f32>,
    pub p_bd_re: Vec<f32>,
    pub p_bd_im: Vec<f32>,
    pub p_wd_re: Vec<f32>,
    pub p_wd_im: Vec<f32>,
    pub p_br_re: Vec<f32>,
    pub p_br_im: Vec<f32>,
    pub p_bi_re: Vec<f32>,
    pub p_bi_im: Vec<f32>,
    // Буферы без аллокаций
    feat: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<f32>,
    psis: Vec<f32>,
    fb_re: Vec<f32>,
    fb_im: Vec<f32>,
    // Каузальная Welford-стандартизация
    wf_n: u64,
    wf_mean: Vec<f32>,
    wf_m2: Vec<f32>,
    x_buf: Vec<f32>,
    // Опциональные буферы онлайн TD(lambda) Actor-Critic RL (пусты в supervised-режиме!)
    rl_enabled: bool,
    critic_w: Vec<f32>,
    critic_b: f32,
    e_aw: Vec<f32>,
    e_ab: Vec<f32>,
    e_cw: Vec<f32>,
    e_cb: f32,
    e_a: Vec<f32>,
    e_w: Vec<f32>,
    e_bd: Vec<f32>,
    e_wd: Vec<f32>,
    e_br: Vec<f32>,
    e_bi: Vec<f32>,
    pending_v: f32,
    pending_r: f32,
    has_pending: bool,
    td_mean: f32,
    td_var: f32,
    rng: u64,
    pub seu_fault_count: u64,
}

#[axiom_zero]
impl AlgebraicSelectiveCayley {
    pub fn new(
        in_dim: usize,
        units: usize,
        num_classes: usize,
        selective: bool,
        seed: u64,
    ) -> Self {
        let mut rng = splitmix64(seed ^ 0xCA71_5E1E);
        let u = units.clamp(2, 256);
        let d = in_dim.clamp(1, 128);
        let nc = num_classes.clamp(1, 128);
        let ud = u.saturating_mul(d);
        let f_dim = u.saturating_mul(2).saturating_add(d);

        let mut alpha = vec![0.2_f32; u];
        let mut omega = vec![0.2_f32; u];
        let mut b_delta = vec![0.0_f32; u];
        for j in 0..u {
            let frac = total_div_f32(j as f32, u as f32, 0.5);
            let _ = total_set(&mut alpha, j, sanitize_f32(0.06 + 0.55 * frac));
            let sgn = if (j & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
            let _ = total_set(&mut omega, j, sanitize_f32(sgn * (0.14 + 3.2 * frac * frac)));
            let _ = total_set(&mut b_delta, j, sanitize_f32(-0.6 + 0.8 * frac));
        }
        let bs = sanitize_f32(0.55 * inv_sqrt_newton_f32(d as f32));
        let mut w_delta = if selective { vec![0.0_f32; ud] } else { Vec::new() };
        for v in w_delta.iter_mut() {
            *v = sanitize_f32(rand_signed_f32(&mut rng) * bs);
        }
        let mut w_bre = vec![0.0_f32; ud];
        let mut w_bim = vec![0.0_f32; ud];
        for v in w_bre.iter_mut() {
            *v = sanitize_f32(rand_signed_f32(&mut rng) * bs);
        }
        for v in w_bim.iter_mut() {
            *v = sanitize_f32(rand_signed_f32(&mut rng) * bs);
        }
        let mut out_w = vec![0.0_f32; nc.saturating_mul(f_dim)];
        for v in out_w.iter_mut() {
            *v = sanitize_f32(rand_signed_f32(&mut rng) * 0.05);
        }

        Self {
            in_dim: d,
            units: u,
            num_classes: nc,
            kappa: 0.65,
            selective,
            standardize: false,
            alpha,
            omega,
            b_delta: if selective { b_delta } else { Vec::new() },
            w_delta,
            w_bre,
            w_bim,
            v_alpha: vec![0.04_f32; u],
            v_omega: vec![0.04_f32; u],
            v_bd: if selective { vec![0.04_f32; u] } else { Vec::new() },
            v_wd: if selective { vec![0.04_f32; ud] } else { Vec::new() },
            v_wbr: vec![0.04_f32; ud],
            v_wbi: vec![0.04_f32; ud],
            out_w,
            out_b: vec![0.0_f32; nc],
            v_ow: vec![0.04_f32; nc.saturating_mul(f_dim)],
            v_ob: vec![0.04_f32; nc],
            h_re: vec![0.0_f32; u],
            h_im: vec![0.0_f32; u],
            y_re: vec![0.0_f32; u],
            y_im: vec![0.0_f32; u],
            rho: vec![0.65_f32; u],
            p_a_re: vec![0.0_f32; u],
            p_a_im: vec![0.0_f32; u],
            p_w_re: vec![0.0_f32; u],
            p_w_im: vec![0.0_f32; u],
            p_bd_re: if selective { vec![0.0_f32; u] } else { Vec::new() },
            p_bd_im: if selective { vec![0.0_f32; u] } else { Vec::new() },
            p_wd_re: if selective { vec![0.0_f32; ud] } else { Vec::new() },
            p_wd_im: if selective { vec![0.0_f32; ud] } else { Vec::new() },
            p_br_re: vec![0.0_f32; ud],
            p_br_im: vec![0.0_f32; ud],
            p_bi_re: vec![0.0_f32; ud],
            p_bi_im: vec![0.0_f32; ud],
            feat: vec![0.0_f32; f_dim],
            logits: vec![0.0_f32; nc],
            probs: vec![0.0_f32; nc],
            psis: vec![1.0_f32; nc],
            fb_re: vec![0.0_f32; u],
            fb_im: vec![0.0_f32; u],
            wf_n: 0,
            wf_mean: vec![0.0_f32; d],
            wf_m2: vec![0.0_f32; d],
            x_buf: vec![0.0_f32; d],
            rl_enabled: false,
            critic_w: Vec::new(),
            critic_b: 0.0,
            e_aw: Vec::new(),
            e_ab: Vec::new(),
            e_cw: Vec::new(),
            e_cb: 0.0,
            e_a: Vec::new(),
            e_w: Vec::new(),
            e_bd: Vec::new(),
            e_wd: Vec::new(),
            e_br: Vec::new(),
            e_bi: Vec::new(),
            pending_v: 0.0,
            pending_r: 0.0,
            has_pending: false,
            td_mean: 0.0,
            td_var: 1.0,
            rng,
            seu_fault_count: 0,
        }
    }

    pub fn with_standardize(mut self, on: bool) -> Self {
        self.standardize = on;
        self
    }

    /// Активирует онлайн TD(lambda) Actor-Critic RL (аллоцирует Critic и следы приемлемости один раз).
    pub fn enable_rl_mode(&mut self) {
        if self.rl_enabled {
            return;
        }
        let u = self.units;
        let ud = u.saturating_mul(self.in_dim);
        let f_dim = self.feat.len();
        self.critic_w = vec![0.0_f32; f_dim];
        for v in self.critic_w.iter_mut() {
            *v = sanitize_f32(rand_signed_f32(&mut self.rng) * 0.05);
        }
        self.critic_b = 0.0;
        self.e_aw = vec![0.0_f32; self.num_classes.saturating_mul(f_dim)];
        self.e_ab = vec![0.0_f32; self.num_classes];
        self.e_cw = vec![0.0_f32; f_dim];
        self.e_cb = 0.0;
        self.e_a = vec![0.0_f32; u];
        self.e_w = vec![0.0_f32; u];
        self.e_bd = if self.selective { vec![0.0_f32; u] } else { Vec::new() };
        self.e_wd = if self.selective { vec![0.0_f32; ud] } else { Vec::new() };
        self.e_br = vec![0.0_f32; ud];
        self.e_bi = vec![0.0_f32; ud];
        self.rl_enabled = true;
        self.reset();
    }

    pub fn reset(&mut self) {
        self.h_re.fill(0.0);
        self.h_im.fill(0.0);
        self.y_re.fill(0.0);
        self.y_im.fill(0.0);
        self.rho.fill(self.kappa);
        self.p_a_re.fill(0.0);
        self.p_a_im.fill(0.0);
        self.p_w_re.fill(0.0);
        self.p_w_im.fill(0.0);
        self.p_bd_re.fill(0.0);
        self.p_bd_im.fill(0.0);
        self.p_wd_re.fill(0.0);
        self.p_wd_im.fill(0.0);
        self.p_br_re.fill(0.0);
        self.p_br_im.fill(0.0);
        self.p_bi_re.fill(0.0);
        self.p_bi_im.fill(0.0);
        if self.rl_enabled {
            self.e_aw.fill(0.0);
            self.e_ab.fill(0.0);
            self.e_cw.fill(0.0);
            self.e_cb = 0.0;
            self.e_a.fill(0.0);
            self.e_w.fill(0.0);
            self.e_bd.fill(0.0);
            self.e_wd.fill(0.0);
            self.e_br.fill(0.0);
            self.e_bi.fill(0.0);
            self.pending_v = 0.0;
            self.pending_r = 0.0;
            self.has_pending = false;
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        (self.alpha.len()
            + self.omega.len()
            + self.b_delta.len()
            + self.w_delta.len()
            + self.w_bre.len()
            + self.w_bim.len()
            + self.v_alpha.len()
            + self.v_omega.len()
            + self.v_bd.len()
            + self.v_wd.len()
            + self.v_wbr.len()
            + self.v_wbi.len()
            + self.out_w.len()
            + self.out_b.len()
            + self.v_ow.len()
            + self.v_ob.len()
            + self.h_re.len()
            + self.h_im.len()
            + self.y_re.len()
            + self.y_im.len()
            + self.rho.len()
            + self.p_a_re.len()
            + self.p_a_im.len()
            + self.p_w_re.len()
            + self.p_w_im.len()
            + self.p_bd_re.len()
            + self.p_bd_im.len()
            + self.p_wd_re.len()
            + self.p_wd_im.len()
            + self.p_br_re.len()
            + self.p_br_im.len()
            + self.p_bi_re.len()
            + self.p_bi_im.len()
            + self.feat.len()
            + self.logits.len()
            + self.probs.len()
            + self.psis.len()
            + self.fb_re.len()
            + self.fb_im.len()
            + self.wf_mean.len()
            + self.wf_m2.len()
            + self.x_buf.len()
            + self.critic_w.len()
            + self.e_aw.len()
            + self.e_ab.len()
            + self.e_cw.len()
            + self.e_a.len()
            + self.e_w.len()
            + self.e_bd.len()
            + self.e_wd.len()
            + self.e_br.len()
            + self.e_bi.len())
        .saturating_mul(4)
            + 64
    }

    pub fn total_seu_faults(&self) -> u64 {
        self.seu_fault_count
    }

    pub fn scrub_weight_seu(&mut self) -> u64 {
        let mut faults = 0u64;
        for v in self.alpha.iter_mut() {
            if !v.is_finite() || *v < 0.01 || *v > 3.0 {
                *v = 0.25;
                faults = faults.saturating_add(1);
            }
        }
        for v in self.omega.iter_mut() {
            if !v.is_finite() || v.abs() > 6.5 {
                *v = 0.30;
                faults = faults.saturating_add(1);
            }
        }
        for v in self.b_delta.iter_mut() {
            if !v.is_finite() || v.abs() > 4.5 {
                *v = 0.0;
                faults = faults.saturating_add(1);
            }
        }
        for v in self.w_delta.iter_mut() {
            if !v.is_finite() || v.abs() > 3.5 {
                *v = 0.0;
                faults = faults.saturating_add(1);
            }
        }
        for v in self.w_bre.iter_mut() {
            if !v.is_finite() || v.abs() > 3.5 {
                *v = 0.0;
                faults = faults.saturating_add(1);
            }
        }
        for v in self.w_bim.iter_mut() {
            if !v.is_finite() || v.abs() > 3.5 {
                *v = 0.0;
                faults = faults.saturating_add(1);
            }
        }
        for v in self.out_w.iter_mut() {
            if !v.is_finite() || v.abs() > 8.5 {
                *v = 0.0;
                faults = faults.saturating_add(1);
            }
        }
        for v in self.out_b.iter_mut() {
            if !v.is_finite() || v.abs() > 4.5 {
                *v = 0.0;
                faults = faults.saturating_add(1);
            }
        }
        for v in self.critic_w.iter_mut() {
            if !v.is_finite() || v.abs() > 16.5 {
                *v = 0.0;
                faults = faults.saturating_add(1);
            }
        }
        if !self.critic_b.is_finite() || self.critic_b.abs() > 26.0 {
            self.critic_b = 0.0;
            faults = faults.saturating_add(1);
        }
        faults
    }

    pub fn inject_register_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let u = self.units;
        if u == 0 {
            return;
        }
        let mask = 1u32 << (bit & 31);
        let idx = total_rem_usize(reg_idx, u.saturating_mul(2), 0);
        if idx < u {
            let cur = total_get_copy(&self.h_re, idx, 0.0);
            let _ = total_set(&mut self.h_re, idx, f32::from_bits(cur.to_bits() ^ mask));
        } else {
            let k = idx.saturating_sub(u);
            let cur = total_get_copy(&self.h_im, k, 0.0);
            let _ = total_set(&mut self.h_im, k, f32::from_bits(cur.to_bits() ^ mask));
        }
    }

    pub fn inject_weight_bit_flip(&mut self, reg_idx: usize, bit: u32) {
        let mask = 1u32 << (bit & 31);
        let mut slices: [&mut [f32]; 7] = [
            &mut self.alpha,
            &mut self.omega,
            &mut self.b_delta,
            &mut self.w_delta,
            &mut self.w_bre,
            &mut self.w_bim,
            &mut self.out_w,
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

    pub fn step_forward(&mut self, raw_x: &[f32]) {
        self.seu_fault_count = self.seu_fault_count.saturating_add(self.scrub_weight_seu());
        let u = self.units;
        let d = self.in_dim;
        let xl = raw_x.len().min(d);

        self.x_buf.fill(0.0);
        let mut input_bad = false;
        for k in 0..xl {
            let rv = total_get_copy(raw_x, k, 0.0);
            if !rv.is_finite() || rv.abs() > 1e4 {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
                input_bad = true;
            }
        }
        if self.standardize {
            if !input_bad {
                self.wf_n = self.wf_n.saturating_add(1);
                let nf = self.wf_n as f32;
                for k in 0..xl {
                    let v = sanitize_bounded_f32(total_get_copy(raw_x, k, 0.0), 1e4);
                    let mu0 = total_get_copy(&self.wf_mean, k, 0.0);
                    let d1 = sanitize_f32(v - mu0);
                    let mu1 = sanitize_f32(mu0 + total_div_f32(d1, nf, 0.0));
                    let d2 = sanitize_f32(v - mu1);
                    let m2 = sanitize_f32(total_get_copy(&self.wf_m2, k, 0.0) + d1 * d2).max(0.0);
                    let _ = total_set(&mut self.wf_mean, k, mu1);
                    let _ = total_set(&mut self.wf_m2, k, m2);
                    let std = if self.wf_n >= 4 {
                        sqrt_newton_f32(total_div_f32(m2, (nf - 1.0).max(1.0), 1.0)).max(0.05)
                    } else {
                        1.0_f32
                    };
                    let _ = total_set(&mut self.x_buf, k, total_div_f32(v - mu1, std, 0.0).clamp(-4.0, 4.0));
                }
            } else {
                let nf = self.wf_n.max(1) as f32;
                for k in 0..xl {
                    let v = sanitize_bounded_f32(total_get_copy(raw_x, k, 0.0), 1e4);
                    let mu = total_get_copy(&self.wf_mean, k, 0.0);
                    let m2 = total_get_copy(&self.wf_m2, k, 0.0).max(0.0);
                    let std = if self.wf_n >= 4 {
                        sqrt_newton_f32(total_div_f32(m2, (nf - 1.0).max(1.0), 1.0)).max(0.05)
                    } else {
                        1.0_f32
                    };
                    let _ = total_set(&mut self.x_buf, k, total_div_f32(v - mu, std, 0.0).clamp(-4.0, 4.0));
                }
            }
        } else {
            for k in 0..xl {
                let v = sanitize_bounded_f32(total_get_copy(raw_x, k, 0.0), 10.0);
                let _ = total_set(&mut self.x_buf, k, v);
            }
        }

        let kap2 = sanitize_f32(self.kappa * self.kappa);
        for j in 0..u {
            let off = j.saturating_mul(d);
            let mut zd = if self.selective {
                total_get_copy(&self.b_delta, j, 0.0)
            } else {
                0.0_f32
            };
            let mut br_in = 0.0_f32;
            let mut bi_in = 0.0_f32;
            for k in 0..d {
                let xk = total_get_copy(&self.x_buf, k, 0.0);
                if self.selective {
                    zd = sanitize_f32(zd + total_get_copy(&self.w_delta, off + k, 0.0) * xk);
                }
                br_in = sanitize_f32(br_in + total_get_copy(&self.w_bre, off + k, 0.0) * xk);
                bi_in = sanitize_f32(bi_in + total_get_copy(&self.w_bim, off + k, 0.0) * xk);
            }
            let (delta, dd_dz) = if self.selective {
                rational_pos_and_deriv(zd)
            } else {
                (1.0_f32, 0.0_f32)
            };

            let a = total_get_copy(&self.alpha, j, 0.25);
            let a2 = sanitize_f32(a * a);
            let inv_da = total_div_unit_shifted_f32(1.0, sanitize_f32(delta * a2));
            let r = sanitize_f32(0.998 * inv_da);
            let dr_da = sanitize_f32(-2.0 * r * delta * a * inv_da);
            let dr_dd = sanitize_f32(-r * a2 * inv_da);

            let om = total_get_copy(&self.omega, j, 0.3);
            let w = sanitize_f32(delta * om);
            let w2 = sanitize_f32(w * w);
            let inv_w = total_div_unit_shifted_f32(1.0, w2);
            let cth = sanitize_f32((1.0 - w2) * inv_w);
            let sth = sanitize_f32(2.0 * w * inv_w);
            let dphi_dw = sanitize_f32(2.0 * inv_w);
            let lam_r = sanitize_f32(r * cth);
            let lam_i = sanitize_f32(r * sth);

            let gam = sanitize_f32(sqrt_newton_f32((1.0 - r * r).max(1e-4)));
            let dgam_dr = total_div_f32(-r, gam, 0.0);
            let g_dt = sanitize_f32(gam * delta);
            let dgdt_da = sanitize_f32(delta * dgam_dr * dr_da);
            let dgdt_dd = sanitize_f32(gam + delta * dgam_dr * dr_dd);

            let raw_hr = total_get_copy(&self.h_re, j, 0.0);
            let raw_hi = total_get_copy(&self.h_im, j, 0.0);
            if !raw_hr.is_finite() || raw_hr.abs() > 24.0 || !raw_hi.is_finite() || raw_hi.abs() > 24.0 {
                self.seu_fault_count = self.seu_fault_count.saturating_add(1);
            }
            let hp_r = sanitize_bounded_f32(raw_hr, 24.0);
            let hp_i = sanitize_bounded_f32(raw_hi, 24.0);

            let lhr = sanitize_f32(lam_r * hp_r - lam_i * hp_i);
            let lhi = sanitize_f32(lam_r * hp_i + lam_i * hp_r);
            let hr = sanitize_bounded_f32(lhr + g_dt * br_in, 24.0);
            let hi = sanitize_bounded_f32(lhi + g_dt * bi_in, 24.0);
            let _ = total_set(&mut self.h_re, j, hr);
            let _ = total_set(&mut self.h_im, j, hi);
            let rho = sanitize_f32(sqrt_newton_f32(hr * hr + hi * hi + kap2)).max(self.kappa);
            let inv_rho = total_div_f32(1.0, rho, 1.0);
            let _ = total_set(&mut self.rho, j, rho);
            let _ = total_set(&mut self.y_re, j, sanitize_f32(hr * inv_rho));
            let _ = total_set(&mut self.y_im, j, sanitize_f32(hi * inv_rho));

            let frac_a = total_div_f32(dr_da, r.max(1e-6), 0.0);
            let par = sanitize_bounded_f32(total_get_copy(&self.p_a_re, j, 0.0), 32.0);
            let pai = sanitize_bounded_f32(total_get_copy(&self.p_a_im, j, 0.0), 32.0);
            let _ = total_set(
                &mut self.p_a_re,
                j,
                sanitize_bounded_f32((lam_r * par - lam_i * pai) + frac_a * lhr + dgdt_da * br_in, 32.0),
            );
            let _ = total_set(
                &mut self.p_a_im,
                j,
                sanitize_bounded_f32((lam_r * pai + lam_i * par) + frac_a * lhi + dgdt_da * bi_in, 32.0),
            );

            let dph_dom = sanitize_f32(delta * dphi_dw);
            let pwr = sanitize_bounded_f32(total_get_copy(&self.p_w_re, j, 0.0), 32.0);
            let pwi = sanitize_bounded_f32(total_get_copy(&self.p_w_im, j, 0.0), 32.0);
            let _ = total_set(
                &mut self.p_w_re,
                j,
                sanitize_bounded_f32((lam_r * pwr - lam_i * pwi) - dph_dom * lhi, 32.0),
            );
            let _ = total_set(
                &mut self.p_w_im,
                j,
                sanitize_bounded_f32((lam_r * pwi + lam_i * pwr) + dph_dom * lhr, 32.0),
            );

            let mut inh_zr = 0.0_f32;
            let mut inh_zi = 0.0_f32;
            if self.selective {
                let frac_d = total_div_f32(dr_dd, r.max(1e-6), 0.0);
                let dph_dd = sanitize_f32(om * dphi_dw);
                inh_zr = sanitize_f32(((frac_d * lhr - dph_dd * lhi) + dgdt_dd * br_in) * dd_dz);
                inh_zi = sanitize_f32(((frac_d * lhi + dph_dd * lhr) + dgdt_dd * bi_in) * dd_dz);
                let pbdr = sanitize_bounded_f32(total_get_copy(&self.p_bd_re, j, 0.0), 32.0);
                let pbdi = sanitize_bounded_f32(total_get_copy(&self.p_bd_im, j, 0.0), 32.0);
                let _ = total_set(
                    &mut self.p_bd_re,
                    j,
                    sanitize_bounded_f32((lam_r * pbdr - lam_i * pbdi) + inh_zr, 32.0),
                );
                let _ = total_set(
                    &mut self.p_bd_im,
                    j,
                    sanitize_bounded_f32((lam_r * pbdi + lam_i * pbdr) + inh_zi, 32.0),
                );
            }

            for k in 0..d {
                let xk = total_get_copy(&self.x_buf, k, 0.0);
                if self.selective {
                    let pwdr = sanitize_bounded_f32(total_get_copy(&self.p_wd_re, off + k, 0.0), 32.0);
                    let pwdi = sanitize_bounded_f32(total_get_copy(&self.p_wd_im, off + k, 0.0), 32.0);
                    let _ = total_set(
                        &mut self.p_wd_re,
                        off + k,
                        sanitize_bounded_f32((lam_r * pwdr - lam_i * pwdi) + xk * inh_zr, 32.0),
                    );
                    let _ = total_set(
                        &mut self.p_wd_im,
                        off + k,
                        sanitize_bounded_f32((lam_r * pwdi + lam_i * pwdr) + xk * inh_zi, 32.0),
                    );
                }
                let pbrr = sanitize_bounded_f32(total_get_copy(&self.p_br_re, off + k, 0.0), 32.0);
                let pbri = sanitize_bounded_f32(total_get_copy(&self.p_br_im, off + k, 0.0), 32.0);
                let _ = total_set(
                    &mut self.p_br_re,
                    off + k,
                    sanitize_bounded_f32((lam_r * pbrr - lam_i * pbri) + g_dt * xk, 32.0),
                );
                let _ = total_set(
                    &mut self.p_br_im,
                    off + k,
                    sanitize_bounded_f32(lam_r * pbri + lam_i * pbrr, 32.0),
                );

                let pbir = sanitize_bounded_f32(total_get_copy(&self.p_bi_re, off + k, 0.0), 32.0);
                let pbii = sanitize_bounded_f32(total_get_copy(&self.p_bi_im, off + k, 0.0), 32.0);
                let _ = total_set(
                    &mut self.p_bi_re,
                    off + k,
                    sanitize_bounded_f32(lam_r * pbir - lam_i * pbii, 32.0),
                );
                let _ = total_set(
                    &mut self.p_bi_im,
                    off + k,
                    sanitize_bounded_f32((lam_r * pbii + lam_i * pbir) + g_dt * xk, 32.0),
                );
            }
        }
    }

    fn fill_features(&mut self) {
        let u = self.units;
        let d = self.in_dim;
        let sc = inv_sqrt_newton_f32(u as f32);
        for j in 0..u {
            let _ = total_set(&mut self.feat, j.saturating_mul(2), sanitize_f32(total_get_copy(&self.y_re, j, 0.0) * sc));
            let _ = total_set(&mut self.feat, j.saturating_mul(2) + 1, sanitize_f32(total_get_copy(&self.y_im, j, 0.0) * sc));
        }
        let base = u.saturating_mul(2);
        for k in 0..d {
            let _ = total_set(&mut self.feat, base + k, sanitize_f32(total_get_copy(&self.x_buf, k, 0.0) * 0.5));
        }
    }

    pub fn step_and_learn_supervised(
        &mut self,
        raw_x: &[f32],
        target: Option<usize>,
        readout_lr: f32,
        rtrl_lr: f32,
    ) -> usize {
        self.step_forward(raw_x);
        self.fill_features();
        let u = self.units;
        let d = self.in_dim;
        let f_dim = self.feat.len();
        let sc = inv_sqrt_newton_f32(u as f32);

        let mut best_c = 0usize;
        let mut best_l = f32::NEG_INFINITY;
        for c in 0..self.num_classes {
            let off = c.saturating_mul(f_dim);
            let w_sl = self.out_w.get(off..off + f_dim).unwrap_or(&[]);
            let lg = sanitize_f32(total_get_copy(&self.out_b, c, 0.0) + fast_dot(w_sl, &self.feat));
            let _ = total_set(&mut self.logits, c, lg);
            if lg > best_l {
                best_l = lg;
                best_c = c;
            }
        }
        rational_probs_p4_into(&self.logits, &mut self.probs, &mut self.psis);

        if let Some(y) = target {
            if y >= self.num_classes {
                return best_c;
            }
            self.fb_re.fill(0.0);
            self.fb_im.fill(0.0);
            for c in 0..self.num_classes {
                let p = total_get_copy(&self.probs, c, 0.0);
                let psi = total_get_copy(&self.psis, c, 1.0);
                let ind = if c == y { 1.0_f32 } else { 0.0_f32 };
                let err = sanitize_bounded_f32(psi * (ind - p), 3.0);
                let off = c.saturating_mul(f_dim);
                for j in 0..u {
                    let wr = total_get_copy(&self.out_w, off + j.saturating_mul(2), 0.0);
                    let wi = total_get_copy(&self.out_w, off + j.saturating_mul(2) + 1, 0.0);
                    let ofr = total_get_copy(&self.fb_re, j, 0.0);
                    let ofi = total_get_copy(&self.fb_im, j, 0.0);
                    let _ = total_set(&mut self.fb_re, j, sanitize_f32(ofr + err * wr * sc));
                    let _ = total_set(&mut self.fb_im, j, sanitize_f32(ofi + err * wi * sc));
                }
                if readout_lr > 0.0 {
                    for i in 0..f_dim {
                        let g = sanitize_f32(err * total_get_copy(&self.feat, i, 0.0));
                        let ov = total_get_copy(&self.v_ow, off + i, 0.04);
                        let nv = sanitize_f32(0.98 * ov + 0.02 * g * g).max(0.0);
                        let _ = total_set(&mut self.v_ow, off + i, nv);
                        let st = sanitize_bounded_f32(sanitize_f32(g * inv_sqrt_newton_f32(nv + 0.0025)), 3.0);
                        let ow = total_get_copy(&self.out_w, off + i, 0.0);
                        let _ = total_set(
                            &mut self.out_w,
                            off + i,
                            sanitize_f32(ow * 0.99995 + readout_lr * st).clamp(-8.0, 8.0),
                        );
                    }
                    let ovb = total_get_copy(&self.v_ob, c, 0.04);
                    let nvb = sanitize_f32(0.98 * ovb + 0.02 * err * err).max(0.0);
                    let _ = total_set(&mut self.v_ob, c, nvb);
                    let stb = sanitize_bounded_f32(sanitize_f32(err * inv_sqrt_newton_f32(nvb + 0.0025)), 3.0);
                    let ob = total_get_copy(&self.out_b, c, 0.0);
                    let _ = total_set(&mut self.out_b, c, sanitize_f32(ob + 0.25 * readout_lr * stb).clamp(-4.0, 4.0));
                }
            }

            if rtrl_lr > 0.0 {
                for j in 0..u {
                    let yr = total_get_copy(&self.y_re, j, 0.0);
                    let yi = total_get_copy(&self.y_im, j, 0.0);
                    let gr = total_get_copy(&self.fb_re, j, 0.0);
                    let gi = total_get_copy(&self.fb_im, j, 0.0);
                    let dot = sanitize_f32(gr * yr + gi * yi);
                    let inv_rho = total_div_f32(1.0, total_get_copy(&self.rho, j, self.kappa).max(self.kappa), 1.0);
                    let fr = sanitize_f32((gr - dot * yr) * inv_rho);
                    let fi = sanitize_f32((gi - dot * yi) * inv_rho);

                    let ga = sanitize_f32(fr * total_get_copy(&self.p_a_re, j, 0.0) + fi * total_get_copy(&self.p_a_im, j, 0.0));
                    let nva = sanitize_f32(0.98 * total_get_copy(&self.v_alpha, j, 0.04) + 0.02 * ga * ga).max(0.0);
                    let _ = total_set(&mut self.v_alpha, j, nva);
                    let sa = sanitize_bounded_f32(sanitize_f32(ga * inv_sqrt_newton_f32(nva + 0.0025)), 3.0);
                    let oa = total_get_copy(&self.alpha, j, 0.25);
                    let _ = total_set(&mut self.alpha, j, sanitize_f32(oa + rtrl_lr * sa).clamp(0.02, 2.5));

                    let gw = sanitize_f32(fr * total_get_copy(&self.p_w_re, j, 0.0) + fi * total_get_copy(&self.p_w_im, j, 0.0));
                    let nvw = sanitize_f32(0.98 * total_get_copy(&self.v_omega, j, 0.04) + 0.02 * gw * gw).max(0.0);
                    let _ = total_set(&mut self.v_omega, j, nvw);
                    let sw = sanitize_bounded_f32(sanitize_f32(gw * inv_sqrt_newton_f32(nvw + 0.0025)), 3.0);
                    let ow = total_get_copy(&self.omega, j, 0.3);
                    let _ = total_set(&mut self.omega, j, sanitize_f32(ow + rtrl_lr * sw).clamp(-6.0, 6.0));

                    if self.selective {
                        let gbd = sanitize_f32(fr * total_get_copy(&self.p_bd_re, j, 0.0) + fi * total_get_copy(&self.p_bd_im, j, 0.0));
                        let nvbd = sanitize_f32(0.98 * total_get_copy(&self.v_bd, j, 0.04) + 0.02 * gbd * gbd).max(0.0);
                        let _ = total_set(&mut self.v_bd, j, nvbd);
                        let sbd = sanitize_bounded_f32(sanitize_f32(gbd * inv_sqrt_newton_f32(nvbd + 0.0025)), 3.0);
                        let obd = total_get_copy(&self.b_delta, j, 0.0);
                        let _ = total_set(&mut self.b_delta, j, sanitize_f32(obd + rtrl_lr * sbd).clamp(-4.0, 4.0));
                    }

                    let off = j.saturating_mul(d);
                    for k in 0..d {
                        if self.selective {
                            let gwd = sanitize_f32(fr * total_get_copy(&self.p_wd_re, off + k, 0.0) + fi * total_get_copy(&self.p_wd_im, off + k, 0.0));
                            let nvwd = sanitize_f32(0.98 * total_get_copy(&self.v_wd, off + k, 0.04) + 0.02 * gwd * gwd).max(0.0);
                            let _ = total_set(&mut self.v_wd, off + k, nvwd);
                            let swd = sanitize_bounded_f32(sanitize_f32(gwd * inv_sqrt_newton_f32(nvwd + 0.0025)), 3.0);
                            let owd = total_get_copy(&self.w_delta, off + k, 0.0);
                            let _ = total_set(&mut self.w_delta, off + k, sanitize_f32(owd + rtrl_lr * swd).clamp(-3.0, 3.0));
                        }
                        let gbr = sanitize_f32(fr * total_get_copy(&self.p_br_re, off + k, 0.0) + fi * total_get_copy(&self.p_br_im, off + k, 0.0));
                        let nvbr = sanitize_f32(0.98 * total_get_copy(&self.v_wbr, off + k, 0.04) + 0.02 * gbr * gbr).max(0.0);
                        let _ = total_set(&mut self.v_wbr, off + k, nvbr);
                        let sbr = sanitize_bounded_f32(sanitize_f32(gbr * inv_sqrt_newton_f32(nvbr + 0.0025)), 3.0);
                        let obr = total_get_copy(&self.w_bre, off + k, 0.0);
                        let _ = total_set(&mut self.w_bre, off + k, sanitize_f32(obr + rtrl_lr * sbr).clamp(-3.0, 3.0));

                        let gbi = sanitize_f32(fr * total_get_copy(&self.p_bi_re, off + k, 0.0) + fi * total_get_copy(&self.p_bi_im, off + k, 0.0));
                        let nvbi = sanitize_f32(0.98 * total_get_copy(&self.v_wbi, off + k, 0.04) + 0.02 * gbi * gbi).max(0.0);
                        let _ = total_set(&mut self.v_wbi, off + k, nvbi);
                        let sbi = sanitize_bounded_f32(sanitize_f32(gbi * inv_sqrt_newton_f32(nvbi + 0.0025)), 3.0);
                        let obi = total_get_copy(&self.w_bim, off + k, 0.0);
                        let _ = total_set(&mut self.w_bim, off + k, sanitize_f32(obi + rtrl_lr * sbi).clamp(-3.0, 3.0));
                    }
                }
            }
        }
        best_c
    }

    /// Непрерывная онлайн-регрессия (100% #[axiom_zero], 0% exp, 0 heap allocs):
    /// MSE `0.5 * (pred - target)^2` с точным алгебраическим 2D Cayley-RTRL и встроенной SEU-защитой.
    pub fn step_and_learn_regression(
        &mut self,
        raw_x: &[f32],
        target: Option<f32>,
        readout_lr: f32,
        rtrl_lr: f32,
    ) -> f32 {
        self.step_forward(raw_x);
        self.fill_features();
        let u = self.units;
        let d = self.in_dim;
        let f_dim = self.feat.len();
        let sc = inv_sqrt_newton_f32(u as f32);

        let w_sl = self.out_w.get(0..f_dim).unwrap_or(&[]);
        let pred = sanitize_f32(total_get_copy(&self.out_b, 0, 0.0) + fast_dot(w_sl, &self.feat));

        if let Some(y_raw) = target {
            let y = sanitize_bounded_f32(y_raw, 16.0);
            let err = sanitize_bounded_f32(sanitize_f32(y - pred), 2.5);
            self.fb_re.fill(0.0);
            self.fb_im.fill(0.0);
            for j in 0..u {
                let wr = total_get_copy(&self.out_w, j.saturating_mul(2), 0.0);
                let wi = total_get_copy(&self.out_w, j.saturating_mul(2) + 1, 0.0);
                let _ = total_set(&mut self.fb_re, j, sanitize_f32(err * wr * sc));
                let _ = total_set(&mut self.fb_im, j, sanitize_f32(err * wi * sc));
            }
            if readout_lr > 0.0 {
                for i in 0..f_dim {
                    let g = sanitize_f32(err * total_get_copy(&self.feat, i, 0.0));
                    let ov = total_get_copy(&self.v_ow, i, 0.04);
                    let nv = sanitize_f32(0.98 * ov + 0.02 * g * g).max(0.0);
                    let _ = total_set(&mut self.v_ow, i, nv);
                    let st = sanitize_bounded_f32(sanitize_f32(g * inv_sqrt_newton_f32(nv + 0.0025)), 3.0);
                    let ow = total_get_copy(&self.out_w, i, 0.0);
                    let _ = total_set(
                        &mut self.out_w,
                        i,
                        sanitize_f32(ow * 0.99995 + readout_lr * st).clamp(-8.0, 8.0),
                    );
                }
                let ovb = total_get_copy(&self.v_ob, 0, 0.04);
                let nvb = sanitize_f32(0.98 * ovb + 0.02 * err * err).max(0.0);
                let _ = total_set(&mut self.v_ob, 0, nvb);
                let stb = sanitize_bounded_f32(sanitize_f32(err * inv_sqrt_newton_f32(nvb + 0.0025)), 3.0);
                let ob = total_get_copy(&self.out_b, 0, 0.0);
                let _ = total_set(
                    &mut self.out_b,
                    0,
                    sanitize_f32(ob + 0.5 * readout_lr * stb).clamp(-4.0, 4.0),
                );
            }
            if rtrl_lr > 0.0 {
                for j in 0..u {
                    let yr = total_get_copy(&self.y_re, j, 0.0);
                    let yi = total_get_copy(&self.y_im, j, 0.0);
                    let gr = total_get_copy(&self.fb_re, j, 0.0);
                    let gi = total_get_copy(&self.fb_im, j, 0.0);
                    let dot = sanitize_f32(gr * yr + gi * yi);
                    let inv_rho = total_div_f32(1.0, total_get_copy(&self.rho, j, self.kappa).max(self.kappa), 1.0);
                    let fr = sanitize_f32((gr - dot * yr) * inv_rho);
                    let fi = sanitize_f32((gi - dot * yi) * inv_rho);

                    let ga = sanitize_f32(fr * total_get_copy(&self.p_a_re, j, 0.0) + fi * total_get_copy(&self.p_a_im, j, 0.0));
                    let nva = sanitize_f32(0.98 * total_get_copy(&self.v_alpha, j, 0.04) + 0.02 * ga * ga).max(0.0);
                    let _ = total_set(&mut self.v_alpha, j, nva);
                    let sa = sanitize_bounded_f32(sanitize_f32(ga * inv_sqrt_newton_f32(nva + 0.0025)), 3.0);
                    let oa = total_get_copy(&self.alpha, j, 0.25);
                    let _ = total_set(&mut self.alpha, j, sanitize_f32(oa + rtrl_lr * sa).clamp(0.02, 2.5));

                    let gw = sanitize_f32(fr * total_get_copy(&self.p_w_re, j, 0.0) + fi * total_get_copy(&self.p_w_im, j, 0.0));
                    let nvw = sanitize_f32(0.98 * total_get_copy(&self.v_omega, j, 0.04) + 0.02 * gw * gw).max(0.0);
                    let _ = total_set(&mut self.v_omega, j, nvw);
                    let sw = sanitize_bounded_f32(sanitize_f32(gw * inv_sqrt_newton_f32(nvw + 0.0025)), 3.0);
                    let ow = total_get_copy(&self.omega, j, 0.3);
                    let _ = total_set(&mut self.omega, j, sanitize_f32(ow + rtrl_lr * sw).clamp(-6.0, 6.0));

                    if self.selective {
                        let gbd = sanitize_f32(fr * total_get_copy(&self.p_bd_re, j, 0.0) + fi * total_get_copy(&self.p_bd_im, j, 0.0));
                        let nvbd = sanitize_f32(0.98 * total_get_copy(&self.v_bd, j, 0.04) + 0.02 * gbd * gbd).max(0.0);
                        let _ = total_set(&mut self.v_bd, j, nvbd);
                        let sbd = sanitize_bounded_f32(sanitize_f32(gbd * inv_sqrt_newton_f32(nvbd + 0.0025)), 3.0);
                        let obd = total_get_copy(&self.b_delta, j, 0.0);
                        let _ = total_set(&mut self.b_delta, j, sanitize_f32(obd + rtrl_lr * sbd).clamp(-4.0, 4.0));
                    }

                    let off = j.saturating_mul(d);
                    for k in 0..d {
                        if self.selective {
                            let gwd = sanitize_f32(fr * total_get_copy(&self.p_wd_re, off + k, 0.0) + fi * total_get_copy(&self.p_wd_im, off + k, 0.0));
                            let nvwd = sanitize_f32(0.98 * total_get_copy(&self.v_wd, off + k, 0.04) + 0.02 * gwd * gwd).max(0.0);
                            let _ = total_set(&mut self.v_wd, off + k, nvwd);
                            let swd = sanitize_bounded_f32(sanitize_f32(gwd * inv_sqrt_newton_f32(nvwd + 0.0025)), 3.0);
                            let owd = total_get_copy(&self.w_delta, off + k, 0.0);
                            let _ = total_set(&mut self.w_delta, off + k, sanitize_f32(owd + rtrl_lr * swd).clamp(-3.0, 3.0));
                        }
                        let gbr = sanitize_f32(fr * total_get_copy(&self.p_br_re, off + k, 0.0) + fi * total_get_copy(&self.p_br_im, off + k, 0.0));
                        let nvbr = sanitize_f32(0.98 * total_get_copy(&self.v_wbr, off + k, 0.04) + 0.02 * gbr * gbr).max(0.0);
                        let _ = total_set(&mut self.v_wbr, off + k, nvbr);
                        let sbr = sanitize_bounded_f32(sanitize_f32(gbr * inv_sqrt_newton_f32(nvbr + 0.0025)), 3.0);
                        let obr = total_get_copy(&self.w_bre, off + k, 0.0);
                        let _ = total_set(&mut self.w_bre, off + k, sanitize_f32(obr + rtrl_lr * sbr).clamp(-3.0, 3.0));

                        let gbi = sanitize_f32(fr * total_get_copy(&self.p_bi_re, off + k, 0.0) + fi * total_get_copy(&self.p_bi_im, off + k, 0.0));
                        let nvbi = sanitize_f32(0.98 * total_get_copy(&self.v_wbi, off + k, 0.04) + 0.02 * gbi * gbi).max(0.0);
                        let _ = total_set(&mut self.v_wbi, off + k, nvbi);
                        let sbi = sanitize_bounded_f32(sanitize_f32(gbi * inv_sqrt_newton_f32(nvbi + 0.0025)), 3.0);
                        let obi = total_get_copy(&self.w_bim, off + k, 0.0);
                        let _ = total_set(&mut self.w_bim, off + k, sanitize_f32(obi + rtrl_lr * sbi).clamp(-3.0, 3.0));
                    }
                }
            }
        }
        pred
    }

    /// Каузальный шаг вперёд для онлайн-регрессии (возвращает прогноз y_{t+1} до прихода факта).
    pub fn decide_regression(&mut self, raw_x: &[f32]) -> f32 {
        self.step_forward(raw_x);
        self.fill_features();
        let f_dim = self.feat.len();
        let w_sl = self.out_w.get(0..f_dim).unwrap_or(&[]);
        sanitize_f32(total_get_copy(&self.out_b, 0, 0.0) + fast_dot(w_sl, &self.feat))
    }

    /// Каузальное онлайн-обновление по факту y_t, пришедшему на следующем тике (использует состояние t-1).
    pub fn learn_regression(&mut self, target: f32, readout_lr: f32, rtrl_lr: f32) {
        let u = self.units;
        let d = self.in_dim;
        let f_dim = self.feat.len();
        let sc = inv_sqrt_newton_f32(u as f32);
        let w_sl = self.out_w.get(0..f_dim).unwrap_or(&[]);
        let pred = sanitize_f32(total_get_copy(&self.out_b, 0, 0.0) + fast_dot(w_sl, &self.feat));
        let y = sanitize_bounded_f32(target, 16.0);
        let err = sanitize_bounded_f32(sanitize_f32(y - pred), 2.5);

        self.fb_re.fill(0.0);
        self.fb_im.fill(0.0);
        for j in 0..u {
            let wr = total_get_copy(&self.out_w, j.saturating_mul(2), 0.0);
            let wi = total_get_copy(&self.out_w, j.saturating_mul(2) + 1, 0.0);
            let _ = total_set(&mut self.fb_re, j, sanitize_f32(err * wr * sc));
            let _ = total_set(&mut self.fb_im, j, sanitize_f32(err * wi * sc));
        }
        if readout_lr > 0.0 {
            for i in 0..f_dim {
                let g = sanitize_f32(err * total_get_copy(&self.feat, i, 0.0));
                let ov = total_get_copy(&self.v_ow, i, 0.04);
                let nv = sanitize_f32(0.98 * ov + 0.02 * g * g).max(0.0);
                let _ = total_set(&mut self.v_ow, i, nv);
                let st = sanitize_bounded_f32(sanitize_f32(g * inv_sqrt_newton_f32(nv + 0.0025)), 3.0);
                let ow = total_get_copy(&self.out_w, i, 0.0);
                let _ = total_set(
                    &mut self.out_w,
                    i,
                    sanitize_f32(ow * 0.99995 + readout_lr * st).clamp(-8.0, 8.0),
                );
            }
            let ovb = total_get_copy(&self.v_ob, 0, 0.04);
            let nvb = sanitize_f32(0.98 * ovb + 0.02 * err * err).max(0.0);
            let _ = total_set(&mut self.v_ob, 0, nvb);
            let stb = sanitize_bounded_f32(sanitize_f32(err * inv_sqrt_newton_f32(nvb + 0.0025)), 3.0);
            let ob = total_get_copy(&self.out_b, 0, 0.0);
            let _ = total_set(
                &mut self.out_b,
                0,
                sanitize_f32(ob + 0.5 * readout_lr * stb).clamp(-4.0, 4.0),
            );
        }
        if rtrl_lr > 0.0 {
            for j in 0..u {
                let yr = total_get_copy(&self.y_re, j, 0.0);
                let yi = total_get_copy(&self.y_im, j, 0.0);
                let gr = total_get_copy(&self.fb_re, j, 0.0);
                let gi = total_get_copy(&self.fb_im, j, 0.0);
                let dot = sanitize_f32(gr * yr + gi * yi);
                let inv_rho =
                    total_div_f32(1.0, total_get_copy(&self.rho, j, self.kappa).max(self.kappa), 1.0);
                let fr = sanitize_f32((gr - dot * yr) * inv_rho);
                let fi = sanitize_f32((gi - dot * yi) * inv_rho);

                let ga = sanitize_f32(
                    fr * total_get_copy(&self.p_a_re, j, 0.0)
                        + fi * total_get_copy(&self.p_a_im, j, 0.0),
                );
                let nva = sanitize_f32(0.98 * total_get_copy(&self.v_alpha, j, 0.04) + 0.02 * ga * ga).max(0.0);
                let _ = total_set(&mut self.v_alpha, j, nva);
                let sa = sanitize_bounded_f32(sanitize_f32(ga * inv_sqrt_newton_f32(nva + 0.0025)), 3.0);
                let oa = total_get_copy(&self.alpha, j, 0.25);
                let _ = total_set(&mut self.alpha, j, sanitize_f32(oa + rtrl_lr * sa).clamp(0.02, 2.5));

                let gw = sanitize_f32(
                    fr * total_get_copy(&self.p_w_re, j, 0.0)
                        + fi * total_get_copy(&self.p_w_im, j, 0.0),
                );
                let nvw = sanitize_f32(0.98 * total_get_copy(&self.v_omega, j, 0.04) + 0.02 * gw * gw).max(0.0);
                let _ = total_set(&mut self.v_omega, j, nvw);
                let sw = sanitize_bounded_f32(sanitize_f32(gw * inv_sqrt_newton_f32(nvw + 0.0025)), 3.0);
                let ow = total_get_copy(&self.omega, j, 0.3);
                let _ = total_set(&mut self.omega, j, sanitize_f32(ow + rtrl_lr * sw).clamp(-6.0, 6.0));

                if self.selective {
                    let gbd = sanitize_f32(
                        fr * total_get_copy(&self.p_bd_re, j, 0.0)
                            + fi * total_get_copy(&self.p_bd_im, j, 0.0),
                    );
                    let nvbd = sanitize_f32(0.98 * total_get_copy(&self.v_bd, j, 0.04) + 0.02 * gbd * gbd).max(0.0);
                    let _ = total_set(&mut self.v_bd, j, nvbd);
                    let sbd = sanitize_bounded_f32(sanitize_f32(gbd * inv_sqrt_newton_f32(nvbd + 0.0025)), 3.0);
                    let obd = total_get_copy(&self.b_delta, j, 0.0);
                    let _ = total_set(&mut self.b_delta, j, sanitize_f32(obd + rtrl_lr * sbd).clamp(-4.0, 4.0));
                }

                let off = j.saturating_mul(d);
                for k in 0..d {
                    if self.selective {
                        let gwd = sanitize_f32(
                            fr * total_get_copy(&self.p_wd_re, off + k, 0.0)
                                + fi * total_get_copy(&self.p_wd_im, off + k, 0.0),
                        );
                        let nvwd = sanitize_f32(0.98 * total_get_copy(&self.v_wd, off + k, 0.04) + 0.02 * gwd * gwd).max(0.0);
                        let _ = total_set(&mut self.v_wd, off + k, nvwd);
                        let swd = sanitize_bounded_f32(sanitize_f32(gwd * inv_sqrt_newton_f32(nvwd + 0.0025)), 3.0);
                        let owd = total_get_copy(&self.w_delta, off + k, 0.0);
                        let _ = total_set(&mut self.w_delta, off + k, sanitize_f32(owd + rtrl_lr * swd).clamp(-3.0, 3.0));
                    }
                    let gbr = sanitize_f32(
                        fr * total_get_copy(&self.p_br_re, off + k, 0.0)
                            + fi * total_get_copy(&self.p_br_im, off + k, 0.0),
                    );
                    let nvbr = sanitize_f32(0.98 * total_get_copy(&self.v_wbr, off + k, 0.04) + 0.02 * gbr * gbr).max(0.0);
                    let _ = total_set(&mut self.v_wbr, off + k, nvbr);
                    let sbr = sanitize_bounded_f32(sanitize_f32(gbr * inv_sqrt_newton_f32(nvbr + 0.0025)), 3.0);
                    let obr = total_get_copy(&self.w_bre, off + k, 0.0);
                    let _ = total_set(&mut self.w_bre, off + k, sanitize_f32(obr + rtrl_lr * sbr).clamp(-3.0, 3.0));

                    let gbi = sanitize_f32(
                        fr * total_get_copy(&self.p_bi_re, off + k, 0.0)
                            + fi * total_get_copy(&self.p_bi_im, off + k, 0.0),
                    );
                    let nvbi = sanitize_f32(0.98 * total_get_copy(&self.v_wbi, off + k, 0.04) + 0.02 * gbi * gbi).max(0.0);
                    let _ = total_set(&mut self.v_wbi, off + k, nvbi);
                    let sbi = sanitize_bounded_f32(sanitize_f32(gbi * inv_sqrt_newton_f32(nvbi + 0.0025)), 3.0);
                    let obi = total_get_copy(&self.w_bim, off + k, 0.0);
                    let _ = total_set(&mut self.w_bim, off + k, sanitize_f32(obi + rtrl_lr * sbi).clamp(-3.0, 3.0));
                }
            }
        }
    }

    fn apply_td_rl(&mut self, td_raw: f32) {
        let td = sanitize_bounded_f32(td_raw, 6.0);
        let crit = sanitize_bounded_f32(td, 3.0);
        let cent = sanitize_f32(td - self.td_mean);
        self.td_mean = sanitize_f32(0.99 * self.td_mean + 0.01 * td);
        self.td_var = sanitize_f32(0.98 * self.td_var + 0.02 * cent * cent).max(0.08);
        let adv = sanitize_f32(cent * inv_sqrt_newton_f32(self.td_var)).clamp(-2.5, 2.5);
        let f = self.feat.len();
        for i in 0..f {
            let cw = total_get_copy(&self.critic_w, i, 0.0);
            let ecw = total_get_copy(&self.e_cw, i, 0.0);
            let _ = total_set(&mut self.critic_w, i, sanitize_f32(cw + 0.14 * crit * ecw).clamp(-16.0, 16.0));
        }
        self.critic_b = sanitize_f32(self.critic_b + 0.035 * crit * self.e_cb).clamp(-25.0, 25.0);
        for a in 0..self.num_classes {
            let off = a.saturating_mul(f);
            for i in 0..f {
                let ow = total_get_copy(&self.out_w, off + i, 0.0);
                let eaw = total_get_copy(&self.e_aw, off + i, 0.0);
                let _ = total_set(&mut self.out_w, off + i, sanitize_f32(ow * 0.9999 + 0.055 * adv * eaw).clamp(-6.0, 6.0));
            }
            let ob = total_get_copy(&self.out_b, a, 0.0);
            let eab = total_get_copy(&self.e_ab, a, 0.0);
            let _ = total_set(&mut self.out_b, a, sanitize_f32(ob * 0.9999 + 0.014 * adv * eab).clamp(-3.0, 3.0));
        }
        let rlr = 0.010_f32;
        let d = self.in_dim;
        for j in 0..self.units {
            let ga = sanitize_f32(adv * total_get_copy(&self.e_a, j, 0.0));
            let nva = sanitize_f32(0.98 * total_get_copy(&self.v_alpha, j, 0.04) + 0.02 * ga * ga).max(0.0);
            let _ = total_set(&mut self.v_alpha, j, nva);
            let sa = sanitize_bounded_f32(sanitize_f32(ga * inv_sqrt_newton_f32(nva + 0.0025)), 3.0);
            let oa = total_get_copy(&self.alpha, j, 0.25);
            let _ = total_set(&mut self.alpha, j, sanitize_f32(oa + rlr * sa).clamp(0.02, 2.5));

            let gw = sanitize_f32(adv * total_get_copy(&self.e_w, j, 0.0));
            let nvw = sanitize_f32(0.98 * total_get_copy(&self.v_omega, j, 0.04) + 0.02 * gw * gw).max(0.0);
            let _ = total_set(&mut self.v_omega, j, nvw);
            let sw = sanitize_bounded_f32(sanitize_f32(gw * inv_sqrt_newton_f32(nvw + 0.0025)), 3.0);
            let ow = total_get_copy(&self.omega, j, 0.3);
            let _ = total_set(&mut self.omega, j, sanitize_f32(ow + rlr * sw).clamp(-6.0, 6.0));

            if self.selective {
                let gbd = sanitize_f32(adv * total_get_copy(&self.e_bd, j, 0.0));
                let nvbd = sanitize_f32(0.98 * total_get_copy(&self.v_bd, j, 0.04) + 0.02 * gbd * gbd).max(0.0);
                let _ = total_set(&mut self.v_bd, j, nvbd);
                let sbd = sanitize_bounded_f32(sanitize_f32(gbd * inv_sqrt_newton_f32(nvbd + 0.0025)), 3.0);
                let obd = total_get_copy(&self.b_delta, j, 0.0);
                let _ = total_set(&mut self.b_delta, j, sanitize_f32(obd + rlr * sbd).clamp(-4.0, 4.0));
            }
            let off = j.saturating_mul(d);
            for k in 0..d {
                if self.selective {
                    let gwd = sanitize_f32(adv * total_get_copy(&self.e_wd, off + k, 0.0));
                    let nvwd = sanitize_f32(0.98 * total_get_copy(&self.v_wd, off + k, 0.04) + 0.02 * gwd * gwd).max(0.0);
                    let _ = total_set(&mut self.v_wd, off + k, nvwd);
                    let swd = sanitize_bounded_f32(sanitize_f32(gwd * inv_sqrt_newton_f32(nvwd + 0.0025)), 3.0);
                    let owd = total_get_copy(&self.w_delta, off + k, 0.0);
                    let _ = total_set(&mut self.w_delta, off + k, sanitize_f32(owd + rlr * swd).clamp(-3.0, 3.0));
                }
                let gbr = sanitize_f32(adv * total_get_copy(&self.e_br, off + k, 0.0));
                let nvbr = sanitize_f32(0.98 * total_get_copy(&self.v_wbr, off + k, 0.04) + 0.02 * gbr * gbr).max(0.0);
                let _ = total_set(&mut self.v_wbr, off + k, nvbr);
                let sbr = sanitize_bounded_f32(sanitize_f32(gbr * inv_sqrt_newton_f32(nvbr + 0.0025)), 3.0);
                let obr = total_get_copy(&self.w_bre, off + k, 0.0);
                let _ = total_set(&mut self.w_bre, off + k, sanitize_f32(obr + rlr * sbr).clamp(-3.0, 3.0));

                let gbi = sanitize_f32(adv * total_get_copy(&self.e_bi, off + k, 0.0));
                let nvbi = sanitize_f32(0.98 * total_get_copy(&self.v_wbi, off + k, 0.04) + 0.02 * gbi * gbi).max(0.0);
                let _ = total_set(&mut self.v_wbi, off + k, nvbi);
                let sbi = sanitize_bounded_f32(sanitize_f32(gbi * inv_sqrt_newton_f32(nvbi + 0.0025)), 3.0);
                let obi = total_get_copy(&self.w_bim, off + k, 0.0);
                let _ = total_set(&mut self.w_bim, off + k, sanitize_f32(obi + rlr * sbi).clamp(-3.0, 3.0));
            }
        }
    }

    pub fn decide_rl(&mut self, raw_x: &[f32], train: bool) -> usize {
        if !self.rl_enabled {
            self.enable_rl_mode();
        }
        self.step_forward(raw_x);
        self.fill_features();
        let u = self.units;
        let d = self.in_dim;
        let f = self.feat.len();
        let sc = inv_sqrt_newton_f32(u as f32);

        let v_cur = sanitize_f32(self.critic_b + fast_dot(&self.critic_w, &self.feat));
        if train && self.has_pending {
            self.apply_td_rl(sanitize_f32(self.pending_r + 0.96 * v_cur - self.pending_v));
        }

        let mut best_a = 0usize;
        let mut best_l = f32::NEG_INFINITY;
        for a in 0..self.num_classes {
            let off = a.saturating_mul(f);
            let w_sl = self.out_w.get(off..off + f).unwrap_or(&[]);
            let lg = sanitize_f32(total_get_copy(&self.out_b, a, 0.0) + fast_dot(w_sl, &self.feat));
            let _ = total_set(&mut self.logits, a, lg);
            if lg > best_l {
                best_l = lg;
                best_a = a;
            }
        }
        rational_probs_p4_into(&self.logits, &mut self.probs, &mut self.psis);

        let act = if train {
            let rv = rand_unit_f32(&mut self.rng);
            let mut cdf = 0.0_f32;
            let mut picked = self.num_classes.saturating_sub(1);
            for a in 0..self.num_classes {
                cdf = sanitize_f32(cdf + total_get_copy(&self.probs, a, 0.0));
                if rv <= cdf {
                    picked = a;
                    break;
                }
            }
            picked
        } else {
            best_a
        };

        if train {
            let dec = 0.96_f32 * 0.85_f32;
            for i in 0..f {
                let ecw = total_get_copy(&self.e_cw, i, 0.0);
                let _ = total_set(&mut self.e_cw, i, sanitize_f32(dec * ecw + total_get_copy(&self.feat, i, 0.0)).clamp(-4.0, 4.0));
            }
            self.e_cb = sanitize_f32(dec * self.e_cb + 1.0).clamp(-5.0, 5.0);
            self.fb_re.fill(0.0);
            self.fb_im.fill(0.0);
            for a in 0..self.num_classes {
                let s = sanitize_f32(
                    total_get_copy(&self.psis, a, 1.0)
                        * ((if a == act { 1.0_f32 } else { 0.0_f32 }) - total_get_copy(&self.probs, a, 0.0)),
                );
                let off = a.saturating_mul(f);
                for i in 0..f {
                    let eaw = total_get_copy(&self.e_aw, off + i, 0.0);
                    let _ = total_set(&mut self.e_aw, off + i, sanitize_f32(dec * eaw + s * total_get_copy(&self.feat, i, 0.0)).clamp(-4.0, 4.0));
                }
                let eab = total_get_copy(&self.e_ab, a, 0.0);
                let _ = total_set(&mut self.e_ab, a, sanitize_f32(dec * eab + s).clamp(-5.0, 5.0));
                for j in 0..u {
                    let wr = total_get_copy(&self.out_w, off + j.saturating_mul(2), 0.0);
                    let wi = total_get_copy(&self.out_w, off + j.saturating_mul(2) + 1, 0.0);
                    let ofr = total_get_copy(&self.fb_re, j, 0.0);
                    let ofi = total_get_copy(&self.fb_im, j, 0.0);
                    let _ = total_set(&mut self.fb_re, j, sanitize_f32(ofr + s * wr * sc));
                    let _ = total_set(&mut self.fb_im, j, sanitize_f32(ofi + s * wi * sc));
                }
            }
            for j in 0..u {
                let cr = total_get_copy(&self.critic_w, j.saturating_mul(2), 0.0);
                let ci = total_get_copy(&self.critic_w, j.saturating_mul(2) + 1, 0.0);
                let gr = sanitize_f32(total_get_copy(&self.fb_re, j, 0.0) + 0.15 * cr * sc);
                let gi = sanitize_f32(total_get_copy(&self.fb_im, j, 0.0) + 0.15 * ci * sc);
                let yr = total_get_copy(&self.y_re, j, 0.0);
                let yi = total_get_copy(&self.y_im, j, 0.0);
                let dot = sanitize_f32(gr * yr + gi * yi);
                let inv_r = total_div_f32(1.0, total_get_copy(&self.rho, j, self.kappa).max(self.kappa), 1.0);
                let fr = sanitize_f32((gr - dot * yr) * inv_r);
                let fi = sanitize_f32((gi - dot * yi) * inv_r);

                let nea = sanitize_f32(dec * total_get_copy(&self.e_a, j, 0.0) + fr * total_get_copy(&self.p_a_re, j, 0.0) + fi * total_get_copy(&self.p_a_im, j, 0.0)).clamp(-16.0, 16.0);
                let _ = total_set(&mut self.e_a, j, nea);
                let new_w = sanitize_f32(dec * total_get_copy(&self.e_w, j, 0.0) + fr * total_get_copy(&self.p_w_re, j, 0.0) + fi * total_get_copy(&self.p_w_im, j, 0.0)).clamp(-16.0, 16.0);
                let _ = total_set(&mut self.e_w, j, new_w);
                if self.selective {
                    let nebd = sanitize_f32(dec * total_get_copy(&self.e_bd, j, 0.0) + fr * total_get_copy(&self.p_bd_re, j, 0.0) + fi * total_get_copy(&self.p_bd_im, j, 0.0)).clamp(-16.0, 16.0);
                    let _ = total_set(&mut self.e_bd, j, nebd);
                }
                let off = j.saturating_mul(d);
                for k in 0..d {
                    if self.selective {
                        let newd = sanitize_f32(dec * total_get_copy(&self.e_wd, off + k, 0.0) + fr * total_get_copy(&self.p_wd_re, off + k, 0.0) + fi * total_get_copy(&self.p_wd_im, off + k, 0.0)).clamp(-16.0, 16.0);
                        let _ = total_set(&mut self.e_wd, off + k, newd);
                    }
                    let nebr = sanitize_f32(dec * total_get_copy(&self.e_br, off + k, 0.0) + fr * total_get_copy(&self.p_br_re, off + k, 0.0) + fi * total_get_copy(&self.p_br_im, off + k, 0.0)).clamp(-16.0, 16.0);
                    let _ = total_set(&mut self.e_br, off + k, nebr);
                    let nebi = sanitize_f32(dec * total_get_copy(&self.e_bi, off + k, 0.0) + fr * total_get_copy(&self.p_bi_re, off + k, 0.0) + fi * total_get_copy(&self.p_bi_im, off + k, 0.0)).clamp(-16.0, 16.0);
                    let _ = total_set(&mut self.e_bi, off + k, nebi);
                }
            }
            self.pending_v = sanitize_f32(self.critic_b + fast_dot(&self.critic_w, &self.feat));
            self.pending_r = 0.0;
            self.has_pending = true;
        }
        act
    }

    #[inline]
    pub fn observe_step_reward_rl(&mut self, r: f32) {
        self.pending_r = sanitize_f32(r);
    }

    pub fn end_episode_rl(&mut self, final_r: f32, train: bool) {
        if !train {
            return;
        }
        self.pending_r = sanitize_f32(final_r);
        if self.has_pending {
            let td = sanitize_f32(self.pending_r - self.pending_v);
            self.apply_td_rl(td);
            self.has_pending = false;
        }
    }
}
