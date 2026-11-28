// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Общие `#[axiom_zero]` алгебраические и геометрические примитивы (DRY-ядро v0.5.0):
//! - 100% `#[axiom_zero]`, 0% трансцендентных функций, 0% `.sqrt()`, 0 heap-аллокаций.
//! - Содержит единые реализации:
//!   * 3D/4D/5D ортогональных операторов (`norm3`, `rot3_apply`, `rot3_step_and_deriv`,
//!     `hh5_apply_pre`, `hh5_deriv_axis`, `ortho_dir_3d_next`, `ortho_dir_4d_next`);
//!   * `fill_p4_logprob_grads` (точный градиент лог-вероятности рационального p4 softmax);
//!   * `rmsnorm_band_into` / `rmsnorm_band_backward` (2/3-шаговый Ньютон RMSNorm, без `.sqrt()`);
//!   * `rmsprop_step` / `rmsprop_momentum_step` / `nlms_step` / `scrub_slice_seu`.

use axiom_zero::{
    axiom_zero, sanitize_bounded_f32, sanitize_f32, total_div_f32, total_div_unit_shifted_f32,
    total_get_copy, total_set,
};

use crate::math::{inv_sqrt_newton_2step_f32, inv_sqrt_newton_f32, rand_signed_f32};

pub type Vec5 = (f32, f32, f32, f32, f32);

/// Выбор между 2-шаговым (`< 1.3e-4`) и 3-шаговым (`< 1e-7`) Ньютоном `1 / sqrt(a)`.
#[axiom_zero]
#[inline(always)]
pub fn inv_sqrt_select(a: f32, exact: bool) -> f32 {
    if exact {
        inv_sqrt_newton_f32(a)
    } else {
        inv_sqrt_newton_2step_f32(a)
    }
}

/// Нормализация 3-вектора через точный 3-шаговый Ньютон (0% `.sqrt()`, fallback `[1,0,0]`).
#[axiom_zero]
#[inline(always)]
pub fn norm3(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let nsq = sanitize_f32(x * x + y * y + z * z);
    if nsq <= 1e-12 {
        return (1.0, 0.0, 0.0);
    }
    let inv = inv_sqrt_newton_f32(nsq);
    (sanitize_f32(x * inv), sanitize_f32(y * inv), sanitize_f32(z * inv))
}

/// Точный ротор Кэли-Родрига `R(theta * e) u` при заранее вычисленных `c1 = 2g/d, c2 = 2g^2/d`.
#[axiom_zero]
#[inline(always)]
pub fn rot3_apply(e: (f32, f32, f32), c1: f32, c2: f32, u: (f32, f32, f32)) -> (f32, f32, f32) {
    let q1_0 = e.1 * u.2 - e.2 * u.1;
    let q1_1 = e.2 * u.0 - e.0 * u.2;
    let q1_2 = e.0 * u.1 - e.1 * u.0;
    let q2_0 = e.1 * q1_2 - e.2 * q1_1;
    let q2_1 = e.2 * q1_0 - e.0 * q1_2;
    let q2_2 = e.0 * q1_1 - e.1 * q1_0;
    (
        sanitize_f32(u.0 + c1 * q1_0 + c2 * q2_0),
        sanitize_f32(u.1 + c1 * q1_1 + c2 * q2_1),
        sanitize_f32(u.2 + c1 * q1_2 + c2 * q2_2),
    )
}

/// Вычисляет `(R u, d(R u)/dtheta, c1, c2)` для 3D ротора Кэли-Родрига (0% sin/cos).
#[axiom_zero]
#[inline(always)]
pub fn rot3_step_and_deriv(
    e: (f32, f32, f32),
    theta: f32,
    u: (f32, f32, f32),
) -> ((f32, f32, f32), (f32, f32, f32), f32, f32) {
    let g = 0.5 * sanitize_bounded_f32(theta, 16.0);
    let g2 = g * g;
    let inv_d = total_div_unit_shifted_f32(1.0, g2);
    let inv_d2 = inv_d * inv_d;
    let c1 = 2.0 * g * inv_d;
    let c2 = 2.0 * g2 * inv_d;

    let q1_0 = e.1 * u.2 - e.2 * u.1;
    let q1_1 = e.2 * u.0 - e.0 * u.2;
    let q1_2 = e.0 * u.1 - e.1 * u.0;
    let q2_0 = e.1 * q1_2 - e.2 * q1_1;
    let q2_1 = e.2 * q1_0 - e.0 * q1_2;
    let q2_2 = e.0 * q1_1 - e.1 * q1_0;

    let r = (
        sanitize_f32(u.0 + c1 * q1_0 + c2 * q2_0),
        sanitize_f32(u.1 + c1 * q1_1 + c2 * q2_1),
        sanitize_f32(u.2 + c1 * q1_2 + c2 * q2_2),
    );
    let d_c1 = (1.0 - g2) * inv_d2;
    let d_c2 = 2.0 * g * inv_d2;
    let dr = (
        sanitize_f32(d_c1 * q1_0 + d_c2 * q2_0),
        sanitize_f32(d_c1 * q1_1 + d_c2 * q2_1),
        sanitize_f32(d_c1 * q1_2 + d_c2 * q2_2),
    );
    (r, dr, c1, c2)
}

/// Ортогональная генерация 3D направлений по Граму–Шмидту (0% `.sqrt()`).
#[axiom_zero]
#[inline]
pub fn ortho_dir_3d_next(
    rng: &mut u64,
    j3: usize,
    b0: &mut (f32, f32, f32),
    b1: &mut (f32, f32, f32),
) -> (f32, f32, f32) {
    if j3 == 0 {
        let mut vx = rand_signed_f32(rng);
        let mut vy = rand_signed_f32(rng);
        let mut vz = rand_signed_f32(rng);
        let n2 = vx * vx + vy * vy + vz * vz;
        if n2 < 1e-6 || !n2.is_finite() {
            vx = 1.0;
            vy = 0.0;
            vz = 0.0;
        } else {
            let inv = inv_sqrt_newton_f32(n2);
            vx *= inv;
            vy *= inv;
            vz *= inv;
        }
        *b0 = (vx, vy, vz);
        *b0
    } else if j3 == 1 {
        let mut vx = rand_signed_f32(rng);
        let mut vy = rand_signed_f32(rng);
        let mut vz = rand_signed_f32(rng);
        let d0 = vx * b0.0 + vy * b0.1 + vz * b0.2;
        vx -= d0 * b0.0;
        vy -= d0 * b0.1;
        vz -= d0 * b0.2;
        let mut n2 = vx * vx + vy * vy + vz * vz;
        if n2 < 1e-4 || !n2.is_finite() {
            let cand = if b0.0.abs() < 0.8 {
                (1.0_f32, 0.0_f32, 0.0_f32)
            } else {
                (0.0_f32, 1.0_f32, 0.0_f32)
            };
            let dd = cand.0 * b0.0 + cand.1 * b0.1 + cand.2 * b0.2;
            vx = cand.0 - dd * b0.0;
            vy = cand.1 - dd * b0.1;
            vz = cand.2 - dd * b0.2;
            n2 = vx * vx + vy * vy + vz * vz;
        }
        let inv = inv_sqrt_newton_f32(n2.max(1e-12));
        vx *= inv;
        vy *= inv;
        vz *= inv;
        *b1 = (vx, vy, vz);
        *b1
    } else {
        let sgn = if rand_signed_f32(rng) >= 0.0 { 1.0_f32 } else { -1.0_f32 };
        let cx = sgn * (b0.1 * b1.2 - b0.2 * b1.1);
        let cy = sgn * (b0.2 * b1.0 - b0.0 * b1.2);
        let cz = sgn * (b0.0 * b1.1 - b0.1 * b1.0);
        let n2 = cx * cx + cy * cy + cz * cz;
        let inv = inv_sqrt_newton_f32(n2.max(1e-12));
        (cx * inv, cy * inv, cz * inv)
    }
}

/// Ортогональная генерация 4D направлений по Граму–Шмидту (0% `.sqrt()`).
#[axiom_zero]
#[inline]
pub fn ortho_dir_4d_next(
    rng: &mut u64,
    j4: usize,
    b0: &mut (f32, f32, f32, f32),
    b1: &mut (f32, f32, f32, f32),
    b2: &mut (f32, f32, f32, f32),
) -> (f32, f32, f32, f32) {
    let mut v = (
        rand_signed_f32(rng),
        rand_signed_f32(rng),
        rand_signed_f32(rng),
        rand_signed_f32(rng),
    );
    let proj_sub = |mut x: (f32, f32, f32, f32), p: (f32, f32, f32, f32)| -> (f32, f32, f32, f32) {
        let d = x.0 * p.0 + x.1 * p.1 + x.2 * p.2 + x.3 * p.3;
        x.0 -= d * p.0;
        x.1 -= d * p.1;
        x.2 -= d * p.2;
        x.3 -= d * p.3;
        x
    };
    if j4 >= 1 {
        v = proj_sub(v, *b0);
    }
    if j4 >= 2 {
        v = proj_sub(v, *b1);
    }
    if j4 >= 3 {
        v = proj_sub(v, *b2);
    }
    let mut n2 = v.0 * v.0 + v.1 * v.1 + v.2 * v.2 + v.3 * v.3;
    if n2 < 1e-4 || !n2.is_finite() {
        let mut best_c = (1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        let mut best_n2 = -1.0_f32;
        for t in 0..4usize {
            let mut c = match t {
                0 => (1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32),
                1 => (0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32),
                2 => (0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32),
                _ => (0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32),
            };
            if j4 >= 1 {
                c = proj_sub(c, *b0);
            }
            if j4 >= 2 {
                c = proj_sub(c, *b1);
            }
            if j4 >= 3 {
                c = proj_sub(c, *b2);
            }
            let cn2 = c.0 * c.0 + c.1 * c.1 + c.2 * c.2 + c.3 * c.3;
            if cn2 > best_n2 {
                best_n2 = cn2;
                best_c = c;
            }
        }
        v = best_c;
        n2 = best_n2.max(1e-12);
    }
    let inv = inv_sqrt_newton_f32(n2);
    let out = (v.0 * inv, v.1 * inv, v.2 * inv, v.3 * inv);
    if j4 == 0 {
        *b0 = out;
    } else if j4 == 1 {
        *b1 = out;
    } else if j4 == 2 {
        *b2 = out;
    }
    out
}

/// Единичное 5D-отражение Хаусхолдера `H_2(u) r = r - two_inv_d * (u^T r) * u`.
#[axiom_zero]
#[inline(always)]
pub fn hh5_apply_pre(u: Vec5, two_inv_d: f32, r: Vec5) -> Vec5 {
    let ur = sanitize_f32(u.0 * r.0 + u.1 * r.1 + u.2 * r.2 + u.3 * r.3 + u.4 * r.4);
    let sc = sanitize_f32(two_inv_d * ur);
    (
        sanitize_f32(r.0 - sc * u.0),
        sanitize_f32(r.1 - sc * u.1),
        sanitize_f32(r.2 - sc * u.2),
        sanitize_f32(r.3 - sc * u.3),
        sanitize_f32(r.4 - sc * u.4),
    )
}

/// Точная производная `d(H_2(u) r) / d(u_axis)` по координате `axis in 0..5`.
#[axiom_zero]
#[inline(always)]
pub fn hh5_deriv_axis(u: Vec5, two_inv_d: f32, r: Vec5, axis: usize) -> Vec5 {
    let ur = sanitize_f32(u.0 * r.0 + u.1 * r.1 + u.2 * r.2 + u.3 * r.3 + u.4 * r.4);
    let (uc, rc) = match axis {
        0 => (u.0, r.0),
        1 => (u.1, r.1),
        2 => (u.2, r.2),
        3 => (u.3, r.3),
        _ => (u.4, r.4),
    };
    let coeff_u = sanitize_f32(rc - two_inv_d * ur * uc);
    let pref = sanitize_f32(-two_inv_d);
    (
        sanitize_f32(pref * (coeff_u * u.0 + if axis == 0 { ur } else { 0.0 })),
        sanitize_f32(pref * (coeff_u * u.1 + if axis == 1 { ur } else { 0.0 })),
        sanitize_f32(pref * (coeff_u * u.2 + if axis == 2 { ur } else { 0.0 })),
        sanitize_f32(pref * (coeff_u * u.3 + if axis == 3 { ur } else { 0.0 })),
        sanitize_f32(pref * (coeff_u * u.4 + if axis == 4 { ur } else { 0.0 })),
    )
}

/// Применение предвычисленной `5x5` матрицы (строковый порядок) к `Vec5` без bounds-checks (деструктуризация `[f32; 25]`).
#[axiom_zero]
#[inline(always)]
pub fn mat5_mul_vec5(m: &[f32; 25], r: Vec5) -> Vec5 {
    let [
        m00, m01, m02, m03, m04,
        m10, m11, m12, m13, m14,
        m20, m21, m22, m23, m24,
        m30, m31, m32, m33, m34,
        m40, m41, m42, m43, m44,
    ] = *m;
    (
        sanitize_f32(m00 * r.0 + m01 * r.1 + m02 * r.2 + m03 * r.3 + m04 * r.4),
        sanitize_f32(m10 * r.0 + m11 * r.1 + m12 * r.2 + m13 * r.3 + m14 * r.4),
        sanitize_f32(m20 * r.0 + m21 * r.1 + m22 * r.2 + m23 * r.3 + m24 * r.4),
        sanitize_f32(m30 * r.0 + m31 * r.1 + m32 * r.2 + m33 * r.3 + m34 * r.4),
        sanitize_f32(m40 * r.0 + m41 * r.1 + m42 * r.2 + m43 * r.3 + m44 * r.4),
    )
}

/// Точный градиент `d(log p_target)/d(logits)` для рационального `p4` softmax (учитывает сдвиг на `max_l`).
#[axiom_zero]
pub fn fill_p4_logprob_grads(logits: &[f32], probs: &[f32], psis: &[f32], target: usize, out: &mut [f32]) {
    let na = logits.len().min(probs.len()).min(psis.len()).min(out.len());
    if na == 0 {
        return;
    }
    let mut win = 0usize;
    let mut max_l = -1e30_f32;
    for c in 0..na {
        let l = sanitize_bounded_f32(total_get_copy(logits, c, 0.0), 64.0);
        if c == 0 || l > max_l {
            max_l = l;
            win = c;
        }
    }
    let mut sum_other = 0.0_f32;
    for c in 0..na {
        if c == win {
            continue;
        }
        let psi = total_get_copy(psis, c, 0.0);
        let pr = total_get_copy(probs, c, 0.0);
        let tgt = if c == target { 1.0_f32 } else { 0.0_f32 };
        let ec = sanitize_bounded_f32(psi * (tgt - pr), 4.0);
        let _ = total_set(out, c, ec);
        sum_other = sanitize_f32(sum_other + ec);
    }
    let _ = total_set(out, win, sanitize_bounded_f32(-sum_other, 4.0));
}

/// Полосовой RMSNorm на 2/3-шаговом Ньютоне (0% `.sqrt()`).
#[axiom_zero]
#[inline(always)]
pub fn rmsnorm_band_into(
    src: &[f32],
    dst: &mut [f32],
    dst_off: usize,
    eps_rms: f32,
    exact: bool,
) -> f32 {
    let n = src.len();
    if n == 0 {
        return 1.0;
    }
    let inv_n = total_div_f32(1.0, n as f32, 1.0);
    let mut sum_sq = 0.0_f32;
    for &v in src.iter() {
        let s = sanitize_bounded_f32(v, 1e6);
        sum_sq = sanitize_f32(sum_sq + s * s);
    }
    let a = sanitize_f32(sum_sq * inv_n + eps_rms).max(1e-8);
    let inv_rms = inv_sqrt_select(a, exact);
    if let Some(dst_sl) = dst.get_mut(dst_off..dst_off + n) {
        for (d, &v) in dst_sl.iter_mut().zip(src.iter()) {
            *d = sanitize_f32(v * inv_rms);
        }
    }
    inv_rms
}

/// Точный обратный проход через полосовой RMSNorm.
#[axiom_zero]
#[inline(always)]
pub fn rmsnorm_band_backward(
    grad_normed: &[f32],
    normed: &[f32],
    inv_rms: f32,
    grad_raw_out: &mut [f32],
) {
    let n = normed.len().min(grad_raw_out.len()).min(grad_normed.len());
    if n == 0 {
        return;
    }
    let inv_n = total_div_f32(1.0, n as f32, 1.0);
    let mut dot = 0.0_f32;
    for (&g, &h) in grad_normed.iter().take(n).zip(normed.iter().take(n)) {
        dot = sanitize_f32(dot + g * h);
    }
    let coeff = sanitize_f32(dot * inv_n);
    for (out, (&g, &h)) in grad_raw_out
        .iter_mut()
        .take(n)
        .zip(grad_normed.iter().take(n).zip(normed.iter().take(n)))
    {
        *out = sanitize_f32(inv_rms * (g - coeff * h));
    }
}

/// Коэрцивная очистка SEU (NaN/Inf/выброс за `[-lim, +lim]` -> `rep`).
#[axiom_zero]
#[inline(always)]
pub fn scrub_slice_seu(s: &mut [f32], lim: f32, rep: f32) -> u64 {
    let mut faults = 0u64;
    for v in s.iter_mut() {
        if !v.is_finite() || v.abs() > lim {
            *v = rep;
            faults = faults.saturating_add(1);
        }
    }
    faults
}

/// Демпфированный RMSProp-шаг без `.sqrt()` (`damp = 0.04`).
#[axiom_zero]
#[inline(always)]
pub fn rmsprop_step(param: &mut f32, v_acc: &mut f32, grad: f32, lr: f32, wd: f32, lim: f32, exact: bool) {
    let g = sanitize_bounded_f32(grad, 16.0);
    let nv = sanitize_f32(0.98 * *v_acc + 0.02 * g * g).clamp(1e-8, 1e6);
    *v_acc = nv;
    let inv_sd = inv_sqrt_select(nv + 0.04, exact);
    let step = sanitize_bounded_f32(g * inv_sd, 3.0);
    let decayed = if wd > 0.0 { *param * (1.0 - wd) } else { *param };
    *param = sanitize_f32(decayed + lr * step).clamp(-lim, lim);
}

/// Демпфированный RMSProp + Momentum шаг без `.sqrt()`.
#[axiom_zero]
#[inline(always)]
pub fn rmsprop_momentum_step(
    grad: f32,
    m_arr: &mut [f32],
    v_arr: &mut [f32],
    w_arr: &mut [f32],
    idx: usize,
    lr: f32,
    w_min: f32,
    w_max: f32,
) {
    let om = total_get_copy(m_arr, idx, 0.0);
    let nm = sanitize_bounded_f32(0.85 * om + 0.15 * grad, 16.0);
    let _ = total_set(m_arr, idx, nm);
    let ov = total_get_copy(v_arr, idx, 0.04);
    let nv = sanitize_f32(0.99 * ov + 0.01 * grad * grad).max(0.0);
    let _ = total_set(v_arr, idx, nv);
    let st = sanitize_bounded_f32(nm * inv_sqrt_newton_f32(nv + 1e-4), 2.5);
    let ow = total_get_copy(w_arr, idx, 0.0);
    let _ = total_set(w_arr, idx, sanitize_f32(ow + lr * st).clamp(w_min, w_max));
}

/// Нормированный шаг NLMS.
#[axiom_zero]
#[inline(always)]
pub fn nlms_step(param: &mut f32, grad: f32, eff_lr: f32, wd: f32, lim: f32) {
    let g = sanitize_bounded_f32(grad, 16.0);
    let decayed = if wd > 0.0 { *param * (1.0 - wd) } else { *param };
    *param = sanitize_f32(decayed + eff_lr * g).clamp(-lim, lim);
}

/// Рациональная алгебраическая сатурация `sigma_rat(z) = z / sqrt(1 + z^2)` и точная производная `(1 + z^2)^(-3/2)`
/// (0% `tanh`, 0% `.sqrt()`, строго нечётная, C^infinity).
#[axiom_zero]
#[inline(always)]
pub fn rational_sat_and_deriv(z: f32, exact: bool) -> (f32, f32) {
    let zc = sanitize_bounded_f32(z, 32.0);
    let r = inv_sqrt_select(sanitize_f32(1.0 + zc * zc), exact);
    let s = sanitize_f32(zc * r);
    let ds = sanitize_f32(r * r * r);
    (s, ds)
}

/// Чисто рациональная сатурация `sigma_abs(z) = z / (1 + |z|)` и производная `1 / (1 + |z|)^2` (0% `tanh`, 0% `.sqrt()`).
#[axiom_zero]
#[inline(always)]
pub fn rational_abs_sat_and_deriv(z: f32) -> (f32, f32) {
    let zc = sanitize_bounded_f32(z, 32.0);
    let inv = total_div_unit_shifted_f32(1.0, zc.abs());
    (sanitize_f32(zc * inv), sanitize_f32(inv * inv))
}

/// Симплектический антисимметричный `J`-XOR для пары 2D векторов `u = (u0, u1)`, `v = (v0, v1)`:
/// `Omega_XOR(u, v) = u ⊙ sigma_rat(J v) - v ⊙ sigma_rat(J u)`, где `J(q, p) = (-p, q)`.
/// Гарантирует точную антисимметрию `Omega(u, v) = -Omega(v, u)` и `Omega(u, u) = 0` (`step135`).
#[axiom_zero]
#[inline(always)]
pub fn symplectic_j_xor_2d(u: (f32, f32), v: (f32, f32), exact: bool) -> (f32, f32) {
    let (su0, _) = rational_sat_and_deriv(u.0, exact);
    let (su1, _) = rational_sat_and_deriv(u.1, exact);
    let (sv0, _) = rational_sat_and_deriv(v.0, exact);
    let (sv1, _) = rational_sat_and_deriv(v.1, exact);
    // J(v) = (-v1, v0), sigma odd => u0 * (-sv1) - v0 * (-su1) = v0 * su1 - u0 * sv1
    let o0 = sanitize_bounded_f32(v.0 * su1 - u.0 * sv1, 64.0);
    let o1 = sanitize_bounded_f32(u.1 * sv0 - v.1 * su0, 64.0);
    (o0, o1)
}

/// Точный обратный проход через `symplectic_j_xor_2d`: возвращает `(grad_u, grad_v)`.
#[axiom_zero]
#[inline(always)]
pub fn symplectic_j_xor_2d_backward(
    u: (f32, f32),
    v: (f32, f32),
    g_out: (f32, f32),
    exact: bool,
) -> ((f32, f32), (f32, f32)) {
    let (su0, du0) = rational_sat_and_deriv(u.0, exact);
    let (su1, du1) = rational_sat_and_deriv(u.1, exact);
    let (sv0, dv0) = rational_sat_and_deriv(v.0, exact);
    let (sv1, dv1) = rational_sat_and_deriv(v.1, exact);
    let g0 = sanitize_bounded_f32(g_out.0, 64.0);
    let g1 = sanitize_bounded_f32(g_out.1, 64.0);
    let gu0 = sanitize_f32(-g0 * sv1 - g1 * v.1 * du0);
    let gu1 = sanitize_f32(g0 * v.0 * du1 + g1 * sv0);
    let gv0 = sanitize_f32(g0 * su1 + g1 * u.1 * dv0);
    let gv1 = sanitize_f32(-g0 * u.0 * dv1 - g1 * su0);
    ((gu0, gu1), (gv0, gv1))
}

/// Антисимметричное 3D векторное произведение `u × v` для кросс-взаимодействия слотов `Band 2` (`step135`).
#[axiom_zero]
#[inline(always)]
pub fn cross_3d_pair(u: (f32, f32, f32), v: (f32, f32, f32)) -> (f32, f32, f32) {
    (
        sanitize_bounded_f32(u.1 * v.2 - u.2 * v.1, 64.0),
        sanitize_bounded_f32(u.2 * v.0 - u.0 * v.2, 64.0),
        sanitize_bounded_f32(u.0 * v.1 - u.1 * v.0, 64.0),
    )
}

/// Точный обратный проход через `cross_3d_pair`: `g_u = v × g`, `g_v = g × u`.
#[axiom_zero]
#[inline(always)]
pub fn cross_3d_pair_backward(
    u: (f32, f32, f32),
    v: (f32, f32, f32),
    g: (f32, f32, f32),
) -> ((f32, f32, f32), (f32, f32, f32)) {
    let gu = (
        sanitize_f32(v.1 * g.2 - v.2 * g.1),
        sanitize_f32(v.2 * g.0 - v.0 * g.2),
        sanitize_f32(v.0 * g.1 - v.1 * g.0),
    );
    let gv = (
        sanitize_f32(g.1 * u.2 - g.2 * u.1),
        sanitize_f32(g.2 * u.0 - g.0 * u.2),
        sanitize_f32(g.0 * u.1 - g.1 * u.0),
    );
    (gu, gv)
}

/// Ортогональный 2D ротор Кэли `R_2(t) u in SO(2)` для банковского адаптера (`step136`, 0% sin/cos).
#[axiom_zero]
#[inline(always)]
pub fn rot2_cayley_apply(t: f32, u: (f32, f32)) -> (f32, f32) {
    let tc = sanitize_bounded_f32(t, 4.5);
    let t2 = tc * tc;
    let inv_d = total_div_unit_shifted_f32(1.0, t2);
    let c = (1.0 - t2) * inv_d;
    let s = (2.0 * tc) * inv_d;
    (sanitize_f32(c * u.0 - s * u.1), sanitize_f32(s * u.0 + c * u.1))
}

/// Точный обратный проход через 2D ротор Кэли `R_2(t) u`: возвращает `(grad_u, grad_t)`.
#[axiom_zero]
#[inline(always)]
pub fn rot2_cayley_backward(t: f32, u: (f32, f32), grad_r: (f32, f32)) -> ((f32, f32), f32) {
    let tc = sanitize_bounded_f32(t, 4.5);
    let t2 = tc * tc;
    let inv_d = total_div_unit_shifted_f32(1.0, t2);
    let c = (1.0 - t2) * inv_d;
    let s = (2.0 * tc) * inv_d;
    let r0 = sanitize_f32(c * u.0 - s * u.1);
    let r1 = sanitize_f32(s * u.0 + c * u.1);
    let gu0 = sanitize_f32(c * grad_r.0 + s * grad_r.1);
    let gu1 = sanitize_f32(-s * grad_r.0 + c * grad_r.1);
    let gt = sanitize_f32(2.0 * inv_d * (-grad_r.0 * r1 + grad_r.1 * r0));
    ((gu0, gu1), gt)
}

/// Ортогональный 3D ротор Кэли `R_3(g) u in SO(3)` от свободного генератора `g in R^3` (`step136`, 0% sin/cos/sqrt):
/// `R_3(g) u = ((1 - ||g||^2) u + 2 (g × u) + 2 <g, u> g) / (1 + ||g||^2)`.
#[axiom_zero]
#[inline(always)]
pub fn rot3_cayley_vec_apply(g: (f32, f32, f32), u: (f32, f32, f32)) -> (f32, f32, f32) {
    let gx = sanitize_bounded_f32(g.0, 4.5);
    let gy = sanitize_bounded_f32(g.1, 4.5);
    let gz = sanitize_bounded_f32(g.2, 4.5);
    let g2 = gx * gx + gy * gy + gz * gz;
    let inv_d = total_div_unit_shifted_f32(1.0, g2);
    let om = 1.0 - g2;
    let dot_gu = gx * u.0 + gy * u.1 + gz * u.2;
    let cx = gy * u.2 - gz * u.1;
    let cy = gz * u.0 - gx * u.2;
    let cz = gx * u.1 - gy * u.0;
    (
        sanitize_f32((om * u.0 + 2.0 * cx + 2.0 * dot_gu * gx) * inv_d),
        sanitize_f32((om * u.1 + 2.0 * cy + 2.0 * dot_gu * gy) * inv_d),
        sanitize_f32((om * u.2 + 2.0 * cz + 2.0 * dot_gu * gz) * inv_d),
    )
}

/// Точный обратный проход через 3D ротор Кэли `R_3(g) u`: возвращает `(grad_u, grad_g)`.
#[axiom_zero]
#[inline(always)]
pub fn rot3_cayley_vec_backward(
    g: (f32, f32, f32),
    u: (f32, f32, f32),
    grad_r: (f32, f32, f32),
) -> ((f32, f32, f32), (f32, f32, f32)) {
    let gx = sanitize_bounded_f32(g.0, 4.5);
    let gy = sanitize_bounded_f32(g.1, 4.5);
    let gz = sanitize_bounded_f32(g.2, 4.5);
    let g2 = gx * gx + gy * gy + gz * gz;
    let inv_d = total_div_unit_shifted_f32(1.0, g2);
    let om = 1.0 - g2;
    let dot_gu = gx * u.0 + gy * u.1 + gz * u.2;
    let cx = gy * u.2 - gz * u.1;
    let cy = gz * u.0 - gx * u.2;
    let cz = gx * u.1 - gy * u.0;
    let r = (
        sanitize_f32((om * u.0 + 2.0 * cx + 2.0 * dot_gu * gx) * inv_d),
        sanitize_f32((om * u.1 + 2.0 * cy + 2.0 * dot_gu * gy) * inv_d),
        sanitize_f32((om * u.2 + 2.0 * cz + 2.0 * dot_gu * gz) * inv_d),
    );
    // grad_u = R_3(g)^T grad_r = R_3(-g) grad_r
    let dot_gp = gx * grad_r.0 + gy * grad_r.1 + gz * grad_r.2;
    let cgpx = gy * grad_r.2 - gz * grad_r.1;
    let cgpy = gz * grad_r.0 - gx * grad_r.2;
    let cgpz = gx * grad_r.1 - gy * grad_r.0;
    let gu = (
        sanitize_f32((om * grad_r.0 - 2.0 * cgpx + 2.0 * dot_gp * gx) * inv_d),
        sanitize_f32((om * grad_r.1 - 2.0 * cgpy + 2.0 * dot_gp * gy) * inv_d),
        sanitize_f32((om * grad_r.2 - 2.0 * cgpz + 2.0 * dot_gp * gz) * inv_d),
    );
    // grad_g = (2 / (1 + ||g||^2)) * (u × p + <g,u> p + <p,g> u - <p, u + r> g)
    let upx = u.1 * grad_r.2 - u.2 * grad_r.1;
    let upy = u.2 * grad_r.0 - u.0 * grad_r.2;
    let upz = u.0 * grad_r.1 - u.1 * grad_r.0;
    let p_ur = grad_r.0 * (u.0 + r.0) + grad_r.1 * (u.1 + r.1) + grad_r.2 * (u.2 + r.2);
    let two_inv = 2.0 * inv_d;
    let gg = (
        sanitize_f32(two_inv * (upx + dot_gu * grad_r.0 + dot_gp * u.0 - p_ur * gx)),
        sanitize_f32(two_inv * (upy + dot_gu * grad_r.1 + dot_gp * u.1 - p_ur * gy)),
        sanitize_f32(two_inv * (upz + dot_gu * grad_r.2 + dot_gp * u.2 - p_ur * gz)),
    );
    (gu, gg)
}

/// Рациональный 2D Финслер-Рандерс «причинный диод» (`v0.5.1 P3`, `step137`, 0% `.sqrt()` / `exp` / `tanh`):
/// `beta = v / sqrt(1 + ||v||^2)` (`||beta|| < 1`),
/// `tau(delta, v) = sqrt(||delta||^2 + mu0^2) + <beta, delta> > 0`.
/// Возвращает `(tau, d_tau/d_delta, d_tau/d_v)`.
#[axiom_zero]
#[inline(always)]
pub fn finsler_randers_diode_2d(
    v: (f32, f32),
    delta: (f32, f32),
    mu0: f32,
    exact: bool,
) -> (f32, (f32, f32), (f32, f32)) {
    let v0 = sanitize_bounded_f32(v.0, 12.0);
    let v1 = sanitize_bounded_f32(v.1, 12.0);
    let d0 = sanitize_bounded_f32(delta.0, 16.0);
    let d1 = sanitize_bounded_f32(delta.1, 16.0);
    let mu = sanitize_f32(mu0).clamp(1e-3, 4.0);

    let rv = inv_sqrt_select(sanitize_f32(1.0 + v0 * v0 + v1 * v1), exact);
    let b0 = sanitize_f32(v0 * rv);
    let b1 = sanitize_f32(v1 * rv);

    let a = sanitize_f32(d0 * d0 + d1 * d1 + mu * mu);
    let rd = inv_sqrt_select(a, exact);
    let rho = sanitize_f32(a * rd);
    let bd = sanitize_f32(b0 * d0 + b1 * d1);
    let tau = sanitize_f32(rho + bd).max(1e-6);

    let dt_dd = (sanitize_f32(d0 * rd + b0), sanitize_f32(d1 * rd + b1));
    let dt_dv = (
        sanitize_f32(rv * (d0 - bd * b0)),
        sanitize_f32(rv * (d1 - bd * b1)),
    );
    (tau, dt_dd, dt_dv)
}

/// Рациональный 3D Финслер-Рандерс «причинный диод» (`v0.5.1 P3`, `step137`, 0% `.sqrt()` / `exp` / `tanh`):
/// возвращает `(tau, d_tau/d_delta, d_tau/d_v)`.
#[axiom_zero]
#[inline(always)]
pub fn finsler_randers_diode_3d(
    v: (f32, f32, f32),
    delta: (f32, f32, f32),
    mu0: f32,
    exact: bool,
) -> (f32, (f32, f32, f32), (f32, f32, f32)) {
    let v0 = sanitize_bounded_f32(v.0, 12.0);
    let v1 = sanitize_bounded_f32(v.1, 12.0);
    let v2 = sanitize_bounded_f32(v.2, 12.0);
    let d0 = sanitize_bounded_f32(delta.0, 16.0);
    let d1 = sanitize_bounded_f32(delta.1, 16.0);
    let d2 = sanitize_bounded_f32(delta.2, 16.0);
    let mu = sanitize_f32(mu0).clamp(1e-3, 4.0);

    let rv = inv_sqrt_select(sanitize_f32(1.0 + v0 * v0 + v1 * v1 + v2 * v2), exact);
    let b0 = sanitize_f32(v0 * rv);
    let b1 = sanitize_f32(v1 * rv);
    let b2 = sanitize_f32(v2 * rv);

    let a = sanitize_f32(d0 * d0 + d1 * d1 + d2 * d2 + mu * mu);
    let rd = inv_sqrt_select(a, exact);
    let rho = sanitize_f32(a * rd);
    let bd = sanitize_f32(b0 * d0 + b1 * d1 + b2 * d2);
    let tau = sanitize_f32(rho + bd).max(1e-6);

    let dt_dd = (
        sanitize_f32(d0 * rd + b0),
        sanitize_f32(d1 * rd + b1),
        sanitize_f32(d2 * rd + b2),
    );
    let dt_dv = (
        sanitize_f32(rv * (d0 - bd * b0)),
        sanitize_f32(rv * (d1 - bd * b1)),
        sanitize_f32(rv * (d2 - bd * b2)),
    );
    (tau, dt_dd, dt_dv)
}

/// Нормированный направленный контраст 2D диода Финслера–Рандерса (`v0.5.1 P3`, `step137`):
/// `c(v, delta) = (tau(+delta) - tau(-delta)) / (tau(+delta) + tau(-delta)) = <beta(v), delta> / sqrt(||delta||^2 + mu0^2) in (-1, 1)`.
/// Строго равен 0 при `v = 0` (нулевой симметричный дрейф при старте), строго антисимметричен `c(v, -delta) = -c(v, delta)`.
/// Возвращает `(c, dc/ddelta, dc/dv)`.
#[axiom_zero]
#[inline(always)]
pub fn finsler_randers_contrast_2d(
    v: (f32, f32),
    delta: (f32, f32),
    mu0: f32,
    exact: bool,
) -> (f32, (f32, f32), (f32, f32)) {
    let v0 = sanitize_bounded_f32(v.0, 12.0);
    let v1 = sanitize_bounded_f32(v.1, 12.0);
    let d0 = sanitize_bounded_f32(delta.0, 16.0);
    let d1 = sanitize_bounded_f32(delta.1, 16.0);
    let mu = sanitize_f32(mu0).clamp(1e-3, 4.0);

    let rv = inv_sqrt_select(sanitize_f32(1.0 + v0 * v0 + v1 * v1), exact);
    let b0 = sanitize_f32(v0 * rv);
    let b1 = sanitize_f32(v1 * rv);

    let a = sanitize_f32(d0 * d0 + d1 * d1 + mu * mu);
    let rd = inv_sqrt_select(a, exact);
    let rd2 = sanitize_f32(rd * rd);
    let bd = sanitize_f32(b0 * d0 + b1 * d1);
    let c = sanitize_f32(bd * rd);

    let dc_dd = (
        sanitize_f32(b0 * rd - c * rd2 * d0),
        sanitize_f32(b1 * rd - c * rd2 * d1),
    );
    let rdv = sanitize_f32(rd * rv);
    let dc_dv = (
        sanitize_f32(rdv * (d0 - bd * b0)),
        sanitize_f32(rdv * (d1 - bd * b1)),
    );
    (c, dc_dd, dc_dv)
}

/// Нормированный направленный контраст 3D диода Финслера–Рандерса (`v0.5.1 P3`, `step137`):
/// `c(v, delta) = (tau(+delta) - tau(-delta)) / (tau(+delta) + tau(-delta)) in (-1, 1)`.
/// Возвращает `(c, dc/ddelta, dc/dv)`.
#[axiom_zero]
#[inline(always)]
pub fn finsler_randers_contrast_3d(
    v: (f32, f32, f32),
    delta: (f32, f32, f32),
    mu0: f32,
    exact: bool,
) -> (f32, (f32, f32, f32), (f32, f32, f32)) {
    let v0 = sanitize_bounded_f32(v.0, 12.0);
    let v1 = sanitize_bounded_f32(v.1, 12.0);
    let v2 = sanitize_bounded_f32(v.2, 12.0);
    let d0 = sanitize_bounded_f32(delta.0, 16.0);
    let d1 = sanitize_bounded_f32(delta.1, 16.0);
    let d2 = sanitize_bounded_f32(delta.2, 16.0);
    let mu = sanitize_f32(mu0).clamp(1e-3, 4.0);

    let rv = inv_sqrt_select(sanitize_f32(1.0 + v0 * v0 + v1 * v1 + v2 * v2), exact);
    let b0 = sanitize_f32(v0 * rv);
    let b1 = sanitize_f32(v1 * rv);
    let b2 = sanitize_f32(v2 * rv);

    let a = sanitize_f32(d0 * d0 + d1 * d1 + d2 * d2 + mu * mu);
    let rd = inv_sqrt_select(a, exact);
    let rd2 = sanitize_f32(rd * rd);
    let bd = sanitize_f32(b0 * d0 + b1 * d1 + b2 * d2);
    let c = sanitize_f32(bd * rd);

    let dc_dd = (
        sanitize_f32(b0 * rd - c * rd2 * d0),
        sanitize_f32(b1 * rd - c * rd2 * d1),
        sanitize_f32(b2 * rd - c * rd2 * d2),
    );
    let rdv = sanitize_f32(rd * rv);
    let dc_dv = (
        sanitize_f32(rdv * (d0 - bd * b0)),
        sanitize_f32(rdv * (d1 - bd * b1)),
        sanitize_f32(rdv * (d2 - bd * b2)),
    );
    (c, dc_dd, dc_dv)
}

/// Рациональный гейт Ляпунова для защиты общего хребта `slots[0]` (`v0.5.1 P4`, `step138`):
/// `g_Lyap = 1 / (1 + kappa * max(0, e^2 / (sigma^2 + eps) - tau)) in (0, 1]`.
/// 0% `exp/ln/tanh`, 0% `.sqrt()`, строго положителен и монотонно убывает по всплеску ошибки.
#[axiom_zero]
#[inline(always)]
pub fn lyapunov_backbone_gate(err_sq: f32, sigma_sq: f32, kappa: f32, tau: f32, eps: f32) -> f32 {
    let e2 = sanitize_bounded_f32(err_sq, 1e6).max(0.0);
    let s2 = sanitize_bounded_f32(sigma_sq, 1e6).max(0.0);
    let ep = sanitize_f32(eps).clamp(1e-5, 4.0);
    let ratio = total_div_f32(e2, sanitize_f32(s2 + ep), 0.0);
    let excess = sanitize_f32(ratio - sanitize_f32(tau)).max(0.0);
    total_div_unit_shifted_f32(1.0, sanitize_f32(sanitize_f32(kappa).max(0.0) * excess))
}

/// Рациональный ограниченный баланс каналов Либеля–Кёрнера (`v0.5.1 P4`, `step138`, 0% `exp`/`ln`):
/// `w(s) = clamp(1.25 / (1 + s^2), w_min, w_max)` с неотрицательным барьером `B(s) = 0.5 * s^2 >= 0`.
/// Выполняет шаг градиентного спуска по `s` на `w_raw(s) * L + 0.5 * lam_bar * s^2` и возвращает `w(s)`.
#[axiom_zero]
#[inline(always)]
pub fn rational_lk_balance_step(
    s: &mut f32,
    loss_proxy: f32,
    lr_s: f32,
    lam_bar: f32,
    w_min: f32,
    w_max: f32,
) -> f32 {
    let sc = sanitize_f32(*s).clamp(0.15, 3.0);
    let s2 = sanitize_f32(sc * sc);
    let inv_d = total_div_unit_shifted_f32(1.0, s2);
    let w = sanitize_f32(1.25 * inv_d).clamp(w_min, w_max);
    let l = sanitize_bounded_f32(loss_proxy, 16.0).max(0.0);
    // d/ds [ 1.25 * L / (1 + s^2) + 0.5 * lam_bar * s^2 ] = s * (lam_bar - 2.5 * L / (1 + s^2)^2)
    let grad_s = sanitize_bounded_f32(
        sc * (sanitize_f32(lam_bar) - 2.5 * l * inv_d * inv_d),
        4.0,
    );
    *s = sanitize_f32(sc - sanitize_f32(lr_s) * grad_s).clamp(0.15, 3.0);
    w
}
