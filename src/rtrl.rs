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
    total_get_copy, total_rem_usize, total_set,
};

use crate::math::{
    fast_dot, fast_saxpy, fast_scale, rand_signed_f32, rand_unit_f32, rational_probs_and_psi_into,
    splitmix64,
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
        let s_cnt = sectors.min(32);
        let m_in = max_in.clamp(1, 16);
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
            let an = (ax * ax + ay * ay + az * az).sqrt().max(1e-5);
            let ex = total_div_f32(ax, an, 1.0);
            let ey = total_div_f32(ay, an, 0.0);
            let ez = total_div_f32(az, an, 0.0);

            // Единичный вектор ввода d_s, ортогональный оси e_s (чтобы ротор реально вращал вход!)
            let rx = rand_signed_f32(&mut rng);
            let ry = rand_signed_f32(&mut rng);
            let rz = rand_signed_f32(&mut rng);
            let dot_re = rx * ex + ry * ey + rz * ez;
            let mut dx = rx - dot_re * ex;
            let mut dy = ry - dot_re * ey;
            let mut dz = rz - dot_re * ez;
            let mut dn = (dx * dx + dy * dy + dz * dz).sqrt();
            if dn < 1e-4 {
                dx = -ey;
                dy = ex;
                dz = 0.0;
                dn = (dx * dx + dy * dy).sqrt().max(1e-5);
            }
            let off = s.saturating_mul(3);
            let _ = total_set(&mut self.axis, off, sanitize_f32(ex));
            let _ = total_set(&mut self.axis, off + 1, sanitize_f32(ey));
            let _ = total_set(&mut self.axis, off + 2, sanitize_f32(ez));
            let _ = total_set(&mut self.in_dir, off, total_div_f32(dx, dn, 1.0));
            let _ = total_set(&mut self.in_dir, off + 1, total_div_f32(dy, dn, 0.0));
            let _ = total_set(&mut self.in_dir, off + 2, total_div_f32(dz, dn, 0.0));

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
}

#[axiom_zero]
impl PlasticCayleyState {
    pub fn new(sectors: usize, max_in: usize, kappa: f32) -> Self {
        let s_cnt = sectors.min(32);
        let m_in = max_in.clamp(1, 16);
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

    /// Один прямой такт линейного ротора Кэли + точное обновление всех чувствительностей RTRL.
    pub fn step_forward(&mut self, params: &PlasticCayleyParams, raw_x: &[f32]) {
        let s_cnt = self.sectors.min(params.sectors);
        let m_in = self.max_in.min(params.max_in);
        let x_len = raw_x.len().min(m_in);
        let kap2 = self.kappa * self.kappa;

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

            let ux = total_get_copy(&self.u, off, 0.0);
            let uy = total_get_copy(&self.u, off + 1, 0.0);
            let uz = total_get_copy(&self.u, off + 2, 0.0);

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

            let rho = sanitize_f32((nux * nux + nuy * nuy + nuz * nuz + kap2).sqrt()).max(self.kappa);
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
            let step_a = sanitize_bounded_f32(total_div_f32(ga, (va + 0.0025).sqrt(), 0.0), 3.0);
            let new_a = sanitize_f32(old_a + lr * step_a).clamp(0.08, 3.0);
            let _ = total_set(&mut params.alpha, s, new_a);

            let old_w = total_get_copy(&params.omega, s, 0.3);
            let gw = sanitize_f32(td * total_get_copy(&self.e_omega, s, 0.0));
            let old_vw = total_get_copy(&params.v_omega, s, 0.04);
            let vw = sanitize_f32(0.98 * old_vw + 0.02 * gw * gw).max(0.0);
            let _ = total_set(&mut params.v_omega, s, vw);
            let step_w = sanitize_bounded_f32(total_div_f32(gw, (vw + 0.0025).sqrt(), 0.0), 3.0);
            let new_w = sanitize_f32(old_w + lr * step_w).clamp(-3.5, 3.5);
            let _ = total_set(&mut params.omega, s, new_w);

            let woff = s.saturating_mul(m_in);
            for j in 0..x_len {
                let old_wj = total_get_copy(&params.in_w, woff + j, 0.0);
                let gj = sanitize_f32(td * total_get_copy(&self.e_w, woff + j, 0.0));
                let old_vj = total_get_copy(&params.v_w, woff + j, 0.04);
                let vj = sanitize_f32(0.98 * old_vj + 0.02 * gj * gj).max(0.0);
                let _ = total_set(&mut params.v_w, woff + j, vj);
                let step_j = sanitize_bounded_f32(total_div_f32(gj, (vj + 0.0025).sqrt(), 0.0), 3.0);
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
        let d_in = in_dim.clamp(1, 12);
        let na = num_actions.clamp(2, 32);
        let s1 = fast_sectors.clamp(4, 24);
        let s2 = slow_sectors.clamp(4, 24);
        let slow_in = (d_in + 4).min(16);

        let fast_params = PlasticCayleyParams::new_with_grid(s1, d_in, seed ^ 0xFA57_0001, 1);
        let slow_params = PlasticCayleyParams::new_with_grid(s2, slow_in, seed ^ 0x510B_0002, 2);

        let fast_state = PlasticCayleyState::new(s1, d_in, 0.65);
        let slow_state = PlasticCayleyState::new(s2, slow_in, 0.65);

        let fdim1 = fast_params.dim();
        let fdim2 = slow_params.dim();
        let feat_dim = fdim1 + fdim2 + d_in;

        let mut rng = splitmix64(seed ^ 0x41E2_CA71_E700);
        let mut pool_w = vec![0.0_f32; 4usize.saturating_mul(fdim1)];
        let p_scale = total_div_f32(1.0, (fdim1.max(1) as f32).sqrt(), 0.2);
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
        let adv = total_div_f32(centered, self.td_var.sqrt(), 0.0).clamp(-2.5, 2.5);

        let fdim = self.feat_buf.len();
        fast_saxpy(&mut self.critic_w, &self.e_critic_w, self.critic_lr * crit_err);
        let cn = fast_dot(&self.critic_w, &self.critic_w);
        if cn > 256.0 {
            fast_scale(&mut self.critic_w, total_div_f32(16.0, cn.sqrt(), 1.0));
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
                    fast_scale(aw, total_div_f32(6.0, an.sqrt(), 1.0));
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
        let sc1 = total_div_f32(1.0, (self.fast_state.sectors.max(1) as f32).sqrt(), 0.25);
        let sc2 = total_div_f32(1.0, (self.slow_state.sectors.max(1) as f32).sqrt(), 0.25);
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
                fast_scale(&mut self.e_critic_w, total_div_f32(4.0, ecn.sqrt(), 1.0));
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
                        fast_scale(ew, total_div_f32(4.0, ean.sqrt(), 1.0));
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
}
