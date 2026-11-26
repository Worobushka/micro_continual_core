// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Низкоуровневая алгебра микроядра:
//! - AVX2+FMA runtime dispatch (без аллокаций) + безопасные #[axiom_zero] обёртки.
//! - Ортогональный 3D ротор Кэли SO(3)^S и антисимметричный коммутатор su(2)^S.
//! - Алгебраическая Kerr-конкуренция W1 без exp/softmax.
//! - Универсальный энкодер: &[f32] -> непрерывные логарифмические спиноры su(2)
//!   и лёгкий &str энкодер (1024 слота в i8 = 96 KB + структурные секторы).

use std::collections::HashMap;

use axiom_zero::{
    axiom_zero, sanitize_bounded_f32, sanitize_f32, total_div_f32, total_div_unit_shifted_f32,
    total_div_usize, total_get_copy, total_rem_usize, total_set,
};

#[cfg(all(target_arch = "x86_64", not(feature = "strict_deterministic")))]
mod simd_x86 {
    use core::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn hsum256(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_castps256_ps128(v);
        let sum4 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum4);
        let sum2 = _mm_add_ps(sum4, shuf);
        let shuf2 = _mm_movehl_ps(sum2, sum2);
        _mm_cvtss_f32(_mm_add_ss(sum2, shuf2))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = if a.len() < b.len() { a.len() } else { b.len() };
            let ap = a.as_ptr();
            let bp = b.as_ptr();
            let mut s0 = _mm256_setzero_ps();
            let mut s1 = _mm256_setzero_ps();
            let mut s2 = _mm256_setzero_ps();
            let mut s3 = _mm256_setzero_ps();

            let mut i = 0usize;
            while i + 32 <= len {
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i + 8)), _mm256_loadu_ps(bp.add(i + 8)), s1);
                s2 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i + 16)), _mm256_loadu_ps(bp.add(i + 16)), s2);
                s3 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i + 24)), _mm256_loadu_ps(bp.add(i + 24)), s3);
                i += 32;
            }
            while i + 8 <= len {
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), s0);
                i += 8;
            }
            let comb = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
            let mut total = hsum256(comb);
            while i < len {
                total += *ap.add(i) * *bp.add(i);
                i += 1;
            }
            total
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn saxpy_avx2_fma(dst: &mut [f32], src: &[f32], scale: f32) {
        unsafe {
            let len = if dst.len() < src.len() { dst.len() } else { src.len() };
            let dp = dst.as_mut_ptr();
            let sp = src.as_ptr();
            let vs = _mm256_set1_ps(scale);
            let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFu32 as i32));
            let max_fin = _mm256_set1_ps(f32::MAX);
            let mut i = 0usize;
            while i + 16 <= len {
                let d0 = _mm256_loadu_ps(dp.add(i));
                let s0 = _mm256_loadu_ps(sp.add(i));
                let d1 = _mm256_loadu_ps(dp.add(i + 8));
                let s1 = _mm256_loadu_ps(sp.add(i + 8));
                let r0 = _mm256_fmadd_ps(s0, vs, d0);
                let r1 = _mm256_fmadd_ps(s1, vs, d1);
                let f0 = _mm256_cmp_ps::<_CMP_LE_OQ>(_mm256_and_ps(r0, abs_mask), max_fin);
                let f1 = _mm256_cmp_ps::<_CMP_LE_OQ>(_mm256_and_ps(r1, abs_mask), max_fin);
                _mm256_storeu_ps(dp.add(i), _mm256_and_ps(r0, f0));
                _mm256_storeu_ps(dp.add(i + 8), _mm256_and_ps(r1, f1));
                i += 16;
            }
            while i + 8 <= len {
                let d0 = _mm256_loadu_ps(dp.add(i));
                let s0 = _mm256_loadu_ps(sp.add(i));
                let r0 = _mm256_fmadd_ps(s0, vs, d0);
                let f0 = _mm256_cmp_ps::<_CMP_LE_OQ>(_mm256_and_ps(r0, abs_mask), max_fin);
                _mm256_storeu_ps(dp.add(i), _mm256_and_ps(r0, f0));
                i += 8;
            }
            while i < len {
                let v = *dp.add(i) + scale * *sp.add(i);
                *dp.add(i) = if v.is_finite() { v } else { 0.0 };
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn scale_avx2(dst: &mut [f32], scale: f32) {
        unsafe {
            let len = dst.len();
            let dp = dst.as_mut_ptr();
            let vs = _mm256_set1_ps(scale);
            let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFu32 as i32));
            let max_fin = _mm256_set1_ps(f32::MAX);
            let mut i = 0usize;
            while i + 8 <= len {
                let r0 = _mm256_mul_ps(_mm256_loadu_ps(dp.add(i)), vs);
                let f0 = _mm256_cmp_ps::<_CMP_LE_OQ>(_mm256_and_ps(r0, abs_mask), max_fin);
                _mm256_storeu_ps(dp.add(i), _mm256_and_ps(r0, f0));
                i += 8;
            }
            while i < len {
                let v = *dp.add(i) * scale;
                *dp.add(i) = if v.is_finite() { v } else { 0.0 };
                i += 1;
            }
        }
    }
}

#[axiom_zero]
#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[axiom_zero]
#[inline]
pub fn rand_unit_f32(state: &mut u64) -> f32 {
    *state = splitmix64(*state);
    let u = ((*state >> 40) & 0xFF_FFFF) as f32;
    total_div_f32(u, 16_777_216.0, 0.5)
}

#[axiom_zero]
#[inline]
pub fn rand_signed_f32(state: &mut u64) -> f32 {
    rand_unit_f32(state) * 2.0 - 1.0
}

#[axiom_zero]
#[inline]
pub fn fnv1a_bytes(bytes: &[u8], seed: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

#[axiom_zero]
#[inline]
pub fn fast_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", not(feature = "strict_deterministic")))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let v = unsafe { simd_x86::dot_avx2_fma(a, b) };
            return sanitize_f32(v);
        }
    }
    let mut sum = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sanitize_f32(sum)
}

/// Raw IEEE-754 AVX2+FMA dot product WITHOUT `sanitize_f32` (for unbiased baseline evaluation:
/// preserves full SIMD throughput while faithfully propagating `NaN`/`Inf` when SEU guards are OFF).
#[inline]
pub fn fast_dot_raw(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", not(feature = "strict_deterministic")))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_avx2_fma(a, b) };
        }
    }
    let mut sum = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

#[axiom_zero]
#[inline]
pub fn fast_saxpy(dst: &mut [f32], src: &[f32], scale: f32) {
    let s = sanitize_f32(scale);
    if s == 0.0 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "strict_deterministic")))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { simd_x86::saxpy_avx2_fma(dst, src, s) };
            return;
        }
    }
    for (d, &x) in dst.iter_mut().zip(src.iter()) {
        *d = sanitize_f32(*d + s * x);
    }
}

#[axiom_zero]
#[inline]
pub fn fast_scale(dst: &mut [f32], scale: f32) {
    let s = sanitize_f32(scale);
    #[cfg(all(target_arch = "x86_64", not(feature = "strict_deterministic")))]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { simd_x86::scale_avx2(dst, s) };
            return;
        }
    }
    for v in dst.iter_mut() {
        *v = sanitize_f32(*v * s);
    }
}

/// Коэрцивная SEU-защита и проекция на компактную единичную сферу S^{D-1}.
#[axiom_zero]
#[inline]
pub fn sanitize_and_normalize_unit(v: &mut [f32], max_amp: f32) {
    let cap = if max_amp.is_finite() && max_amp > 1.0 { max_amp } else { 8.0 };
    let mut nsq = 0.0_f32;
    for cell in v.iter_mut() {
        let s = sanitize_bounded_f32(*cell, cap);
        *cell = s;
        nsq += s * s;
    }
    if nsq > 1e-10 && nsq.is_finite() {
        let inv = inv_sqrt_newton_f32(nsq);
        fast_scale(v, inv);
    } else if let Some(first) = v.first_mut() {
        *first = 1.0;
    }
}

/// Строгий ортогональный 3D ротор Кэли G(a) in SO(3)^S: ||G(a) x|| == ||x||.
#[axiom_zero]
pub fn cayley_rotate_sectors(gauge: &[f32], x: &[f32], out: &mut [f32]) {
    let n = gauge.len().min(x.len()).min(out.len());
    let mut i = 0usize;
    while i + 3 <= n {
        let gx = total_get_copy(gauge, i, 0.0);
        let gy = total_get_copy(gauge, i + 1, 0.0);
        let gz = total_get_copy(gauge, i + 2, 0.0);

        let sx = total_get_copy(x, i, 0.0);
        let sy = total_get_copy(x, i + 1, 0.0);
        let sz = total_get_copy(x, i + 2, 0.0);

        let r = gx * gx + gy * gy + gz * gz;
        let omr = 1.0 - r;
        let dot_gx = gx * sx + gy * sy + gz * sz;
        let cx = gy * sz - gz * sy;
        let cy = gz * sx - gx * sz;
        let cz = gx * sy - gy * sx;

        let rx = total_div_unit_shifted_f32(omr * sx - 2.0 * cx + 2.0 * dot_gx * gx, r);
        let ry = total_div_unit_shifted_f32(omr * sy - 2.0 * cy + 2.0 * dot_gx * gy, r);
        let rz = total_div_unit_shifted_f32(omr * sz - 2.0 * cz + 2.0 * dot_gx * gz, r);

        let _ = total_set(out, i, sanitize_f32(rx));
        let _ = total_set(out, i + 1, sanitize_f32(ry));
        let _ = total_set(out, i + 2, sanitize_f32(rz));
        i += 3;
    }
}

/// Посекторный антисимметричный коммутатор su(2)^S: out_s = a_s x b_s.
#[axiom_zero]
pub fn su2_lie_bracket(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    let mut i = 0usize;
    while i + 3 <= n {
        let ax = total_get_copy(a, i, 0.0);
        let ay = total_get_copy(a, i + 1, 0.0);
        let az = total_get_copy(a, i + 2, 0.0);

        let bx = total_get_copy(b, i, 0.0);
        let by = total_get_copy(b, i + 1, 0.0);
        let bz = total_get_copy(b, i + 2, 0.0);

        let _ = total_set(out, i, sanitize_f32(ay * bz - az * by));
        let _ = total_set(out, i + 1, sanitize_f32(az * bx - ax * bz));
        let _ = total_set(out, i + 2, sanitize_f32(ax * by - ay * bx));
        i += 3;
    }
}

#[axiom_zero]
#[inline]
pub fn pow_nat_f32(base: f32, p: usize) -> f32 {
    let b = sanitize_f32(base);
    let mut acc = 1.0_f32;
    for _ in 0..p {
        acc = sanitize_f32(acc * b);
    }
    acc
}

/// Теорема W1: алгебраическая Kerr-конкуренция без exp/softmax (запись в буфер без аллокаций).
#[axiom_zero]
pub fn kerr_competition_into(scores: &[f32], tau: f32, p: usize, eps: f32, out: &mut [f32]) -> f32 {
    let safe_eps = if eps.is_finite() && eps > 1e-8 { eps } else { 1e-4 };
    let n = scores.len().min(out.len());
    let mut den = safe_eps;
    for i in 0..n {
        let s = sanitize_f32(total_get_copy(scores, i, 0.0));
        let g = (s - sanitize_f32(tau)).max(0.0);
        let w = if g > 0.0 { pow_nat_f32(g, p) } else { 0.0 };
        let _ = total_set(out, i, w);
        den = sanitize_f32(den + w);
    }
    for i in 0..n {
        let w = total_get_copy(out, i, 0.0);
        let _ = total_set(out, i, total_div_f32(w, den, 0.0));
    }
    den
}

/// Точная производная лог-веса рациональной политики без exp/ln:
/// g(z) = 1 + z + 0.5*z^2 (z >= 0) | 1 / (1 - z + 0.5*z^2) (z < 0)
/// psi(z) = g'(z) / g(z) = (1 + |z|) / (1 + |z| + 0.5*z^2) in (0, 1].
#[axiom_zero]
#[inline]
pub fn rational_psi(logit: f32) -> f32 {
    let z = sanitize_bounded_f32(logit, 16.0);
    let az = z.abs();
    let num = sanitize_f32(1.0 + az);
    let shift = sanitize_f32(az + 0.5 * az * az);
    total_div_unit_shifted_f32(num, shift)
}

/// Чисто рациональная сдвигово-инвариантная политика (0% exp, 0% softmax, 100% #[axiom_zero]):
///   z_a = l_a - mean(l)
///   g(z) = 1 + z + 0.5*z^2           при z >= 0
///   g(z) = 1 / (1 - z + 0.5*z^2)     при z < 0
///   p_a  = g(z_a) / sum_k g(z_k)
///   psi_a = g'(z_a) / g(z_a)
/// Знаменатель 1 - z + 0.5*z^2 = 0.5*(1 + (z - 1)^2) >= 0.5 > 0 доказан в Lean 4 (0 аксиом).
#[axiom_zero]
pub fn rational_probs_and_psi_into(logits: &[f32], out_probs: &mut [f32], out_psi: &mut [f32]) {
    let n = logits.len().min(out_probs.len());
    if n == 0 {
        return;
    }
    let mut mean_l = 0.0_f32;
    for i in 0..n {
        mean_l = sanitize_f32(mean_l + sanitize_bounded_f32(total_get_copy(logits, i, 0.0), 32.0));
    }
    mean_l = total_div_f32(mean_l, n as f32, 0.0);

    let mut sum_g = 0.0_f32;
    for i in 0..n {
        let raw = sanitize_bounded_f32(total_get_copy(logits, i, 0.0), 32.0);
        let z = sanitize_bounded_f32(raw - mean_l, 16.0);
        let g = if z >= 0.0 {
            sanitize_f32(1.0 + z + 0.5 * z * z)
        } else {
            total_div_unit_shifted_f32(1.0, sanitize_f32(-z + 0.5 * z * z))
        };
        let _ = total_set(out_probs, i, g);
        if i < out_psi.len() {
            let _ = total_set(out_psi, i, rational_psi(z));
        }
        sum_g = sanitize_f32(sum_g + g);
    }
    let inv_n = total_div_f32(1.0, n as f32, 0.5);
    for i in 0..n {
        let g = total_get_copy(out_probs, i, 0.0);
        let p = if sum_g > 1e-8 && sum_g.is_finite() {
            total_div_f32(g, sum_g, inv_n)
        } else {
            inv_n
        };
        let _ = total_set(out_probs, i, p);
    }
}

/// Чисто рациональная сдвигово-инвариантная политика без возврата `psi` (0% exp, 0% softmax).
#[axiom_zero]
pub fn rational_probs_into(logits: &[f32], out_probs: &mut [f32]) {
    rational_probs_and_psi_into(logits, out_probs, &mut []);
}

/// Совместимый псевдоним (делегирует в чисто рациональную политику `rational_probs_into`, 0% exp).
#[axiom_zero]
#[inline]
pub fn softmax_into(logits: &[f32], out_probs: &mut [f32]) {
    rational_probs_into(logits, out_probs);
}

/// Рациональный Softmax порядка p = 4 с резким хвостом O(1/z^8) и точной лог-производной (0% exp/ln):
///   z_i = clamp(l_i - max_k l_k, -20, 0)
///   q(z) = 1 - 0.5*z + 0.125*z^2 = 0.125*(4 + (z - 2)^2) >= 0.5 > 0 (доказано в Lean 4: step112b, 0 аксиом)
///   w(z) = 1 / q(z)^4,  p_i = w(z_i) / sum_k w(z_k),  psi(z) = w'(z)/w(z) = (2 - z) / q(z).
/// Устраняет размывание градиента на задачах с большим числом классов (например, 60 классов A_5).
#[axiom_zero]
pub fn rational_probs_p4_into(logits: &[f32], out_probs: &mut [f32], out_psi: &mut [f32]) {
    let n = logits.len().min(out_probs.len());
    if n == 0 {
        return;
    }
    let mut max_l = -1e30_f32;
    let mut has_fin = false;
    for i in 0..n {
        let raw = total_get_copy(logits, i, 0.0);
        if raw.is_finite() {
            let l = sanitize_bounded_f32(raw, 64.0);
            if !has_fin || l > max_l {
                max_l = l;
                has_fin = true;
            }
        }
    }
    let inv_n = total_div_f32(1.0, n as f32, 0.5);
    if !has_fin {
        for i in 0..n {
            let _ = total_set(out_probs, i, inv_n);
            if i < out_psi.len() {
                let _ = total_set(out_psi, i, 1.0);
            }
        }
        return;
    }
    let mut sum_w = 0.0_f32;
    for i in 0..n {
        let l = sanitize_bounded_f32(total_get_copy(logits, i, 0.0), 64.0);
        let z = sanitize_f32(l - max_l).clamp(-20.0, 0.0);
        let q = sanitize_f32(1.0 - 0.5 * z + 0.125 * z * z).max(0.5);
        let q2 = sanitize_f32(q * q);
        let w = total_div_f32(1.0, sanitize_f32(q2 * q2), 0.0);
        let _ = total_set(out_probs, i, w);
        if i < out_psi.len() {
            let psi = total_div_f32(sanitize_f32(2.0 - z), q, 0.0);
            let _ = total_set(out_psi, i, psi);
        }
        sum_w = sanitize_f32(sum_w + w);
    }
    for i in 0..n {
        let w = total_get_copy(out_probs, i, 0.0);
        let p = if sum_w > 1e-12 && sum_w.is_finite() {
            total_div_f32(w, sum_w, inv_n)
        } else {
            inv_n
        };
        let _ = total_set(out_probs, i, p);
    }
}

/// Гладкий C^1 строго положительный рациональный вентиль шага времени Delta(z) и его производная (0% exp/ln):
///   pos(z) = 1 + z                 при z >= 0
///   pos(z) = 1 / (1 - z + 0.5*z^2) при z < 0  (знаменатель >= 0.5 > 0 доказан в Lean 4: step112)
/// Результат ограничен отрезком [0.05, 2.0] с точной кусочной производной.
#[axiom_zero]
#[inline]
pub fn rational_pos_and_deriv(z: f32) -> (f32, f32) {
    let zc = sanitize_bounded_f32(z, 10.0);
    if zc >= 0.0 {
        let v = sanitize_f32(1.0 + zc);
        if v >= 2.0 {
            (2.0, 0.0)
        } else {
            (v, 1.0)
        }
    } else {
        let shift = sanitize_f32(-zc + 0.5 * zc * zc);
        let q = sanitize_f32(1.0 + shift);
        let v = total_div_unit_shifted_f32(1.0, shift);
        if v <= 0.05 {
            (0.05, 0.0)
        } else {
            let dv = total_div_f32(sanitize_f32(1.0 - zc), sanitize_f32(q * q), 0.0);
            (v, dv)
        }
    }
}

/// C^1-гладкая периодическая рациональная окружность Кэли (0% sin, 0% cos):
/// (s2, c2) с точным тождеством s2^2 + c2^2 == 1, s2(0) == 0, c2(0) == 1.
#[axiom_zero]
#[inline]
pub fn rational_circle_wave(angle: f32) -> (f32, f32) {
    let t = sanitize_bounded_f32(angle, 1e4) * 0.25;
    let u = sanitize_f32(t - 2.0 * (t * 0.5).round());
    let u2 = sanitize_f32(u * u);
    let s1 = total_div_unit_shifted_f32(2.0 * u, u2);
    let c1 = total_div_unit_shifted_f32(1.0 - u2, u2);
    let s2 = sanitize_f32(2.0 * s1 * c1);
    let c2 = sanitize_f32(c1 * c1 - s1 * s1);
    (s2, c2)
}

/// Алгебраическое сжатие без tanh: x / sqrt(1 + x^2) in (-1, 1) (0% `.sqrt()`).
#[axiom_zero]
#[inline]
pub fn algebraic_squash(x: f32) -> f32 {
    let sx = sanitize_bounded_f32(x, 1e4);
    sanitize_f32(sx * inv_sqrt_newton_f32(1.0 + sx * sx))
}

/// Алгебраическая монотонная лог-подобная компрессия без ln и без `.sqrt()`:
/// sign(x) * 2(sqrt(1+|x|)-1)/(1+0.42*(sqrt(1+|x|)-1)).
#[axiom_zero]
#[inline]
pub fn algebraic_signed_log(x: f32) -> f32 {
    let sx = sanitize_bounded_f32(x, 1e6);
    let dm = sanitize_f32(sqrt_newton_f32(1.0 + sx.abs()) - 1.0);
    let mag = total_div_unit_shifted_f32(2.0 * dm, sanitize_f32(0.42 * dm));
    if sx < 0.0 { -mag } else { mag }
}

/// Безопасный 2-шаговый `1 / sqrt(a)` Ньютона без `.sqrt()` и без делений (`step122` + `step129`):
/// экспоненциальный сдвиг пополам (с множителем `181/256` для нечётной экспоненты) и линейный seed `(5 - m)/4`
/// гарантируют `a * y0^2 in [0.9997, 1.1575] subset [0.5, 2.0]` и строгую положительность `y > 0`
/// при любых скачках мощности (в т.ч. на 10+ порядков, закрытие Мины №1). Относительная точность `< 1.3e-4`.
#[axiom_zero]
#[inline]
pub fn inv_sqrt_newton_2step_f32(a: f32) -> f32 {
    if !a.is_finite() || a <= 1e-30 {
        return 0.0;
    }
    let ac = a.min(1e30_f32);
    let bits = ac.to_bits();
    let exp_b = ((bits >> 23) & 0xFF) as i32;
    let e = exp_b - 127;
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    let k = e >> 1;
    let r = e & 1;
    let pow2_neg_k = f32::from_bits((((127 - k) as u32) & 0xFF) << 23);
    let odd_scale = if r != 0 { 0.70703125_f32 } else { 1.0_f32 };
    let y0 = 0.25_f32 * (5.0_f32 - m) * odd_scale * pow2_neg_k;
    let y1 = 0.5_f32 * y0 * (3.0_f32 - ac * y0 * y0);
    let y2 = 0.5_f32 * y1 * (3.0_f32 - ac * y1 * y1);
    sanitize_f32(y2).max(0.0)
}

/// Полная `f32`-точность `1 / sqrt(a)` (`< 1e-7`) без `.sqrt()` и делений:
/// экспоненциально-защищённый seed (`step129`) + 3 итерации Ньютона.
#[axiom_zero]
#[inline]
pub fn inv_sqrt_newton_f32(a: f32) -> f32 {
    if !a.is_finite() || a <= 1e-30 {
        return 0.0;
    }
    let ac = a.min(1e30_f32);
    let y2 = inv_sqrt_newton_2step_f32(ac);
    let y3 = 0.5_f32 * y2 * (3.0_f32 - ac * y2 * y2);
    sanitize_f32(y3).max(0.0)
}

/// Полная `f32`-точность `sqrt(a)` (`< 1e-7`) без `.sqrt()` и делений (`step130`):
/// `sqrt(a) = a * inv_sqrt_newton_f32(a)`.
#[axiom_zero]
#[inline]
pub fn sqrt_newton_f32(a: f32) -> f32 {
    if !a.is_finite() || a <= 1e-30 {
        return 0.0;
    }
    let ac = a.min(1e30_f32);
    sanitize_f32(ac * inv_sqrt_newton_f32(ac)).max(0.0)
}

/// Гейтированный Ньютон с кэшем `y_prev`: при выходе за скобку `a * y_prev^2 in [0.5, 2.5]`
/// мгновенно сбрасывает оценку на безопасный битовый сдвиг экспоненты.
#[axiom_zero]
#[inline]
pub fn inv_sqrt_newton_guarded_f32(a: f32, y_prev: f32) -> f32 {
    if !a.is_finite() || a <= 1e-30 {
        return 0.0;
    }
    let ac = a.min(1e30_f32);
    if y_prev.is_finite() && y_prev > 0.0 {
        let ay2 = ac * y_prev * y_prev;
        if ay2 >= 0.5 && ay2 <= 2.5 {
            let y1 = 0.5_f32 * y_prev * (3.0_f32 - ac * y_prev * y_prev);
            let y2 = 0.5_f32 * y1 * (3.0_f32 - ac * y1 * y1);
            if y2.is_finite() && y2 > 0.0 {
                return y2;
            }
        }
    }
    inv_sqrt_newton_f32(ac)
}

/// Умножение кватернионов `(w, x, y, z)` (`step117`).
#[axiom_zero]
#[inline]
pub fn quat_mul_4d(
    p: (f32, f32, f32, f32),
    q: (f32, f32, f32, f32),
) -> (f32, f32, f32, f32) {
    let (pw, px, py, pz) = p;
    let (qw, qx, qy, qz) = q;
    (
        sanitize_f32(pw * qw - px * qx - py * qy - pz * qz),
        sanitize_f32(pw * qx + px * qw + py * qz - pz * qy),
        sanitize_f32(pw * qy - px * qz + py * qw + pz * qx),
        sanitize_f32(pw * qz + px * qy - py * qx + pz * qw),
    )
}

/// Стереографический единичный кватернион `q(g) in S^3` без `sin`/`cos`/`sqrt` (`step117`):
/// `q(g) = ((1 - ||g||^2), 2 g) / (1 + ||g||^2)`.
#[axiom_zero]
#[inline]
pub fn q_stereo_unit_4d(gx: f32, gy: f32, gz: f32) -> (f32, f32, f32, f32) {
    let s = sanitize_f32(gx * gx + gy * gy + gz * gz);
    let inv_d = total_div_unit_shifted_f32(1.0, s);
    (
        sanitize_f32((1.0 - s) * inv_d),
        sanitize_f32(2.0 * gx * inv_d),
        sanitize_f32(2.0 * gy * inv_d),
        sanitize_f32(2.0 * gz * inv_d),
    )
}

/// Точная производная `dq(g) / d(g_axis)` стереографического кватерниона (`step121`).
#[axiom_zero]
#[inline]
pub fn q_stereo_unit_4d_deriv(
    gx: f32,
    gy: f32,
    gz: f32,
    axis: usize,
) -> (f32, f32, f32, f32) {
    let s = sanitize_f32(gx * gx + gy * gy + gz * gz);
    let inv_d = total_div_unit_shifted_f32(1.0, s);
    let two_inv = sanitize_f32(2.0 * inv_d);
    let g_ax = match axis {
        0 => gx,
        1 => gy,
        _ => gz,
    };
    let q_ax = sanitize_f32(two_inv * g_ax);
    let dw = sanitize_f32(-two_inv * q_ax);
    let dx = sanitize_f32(two_inv * ((if axis == 0 { 1.0 } else { 0.0 }) - gx * q_ax));
    let dy = sanitize_f32(two_inv * ((if axis == 1 { 1.0 } else { 0.0 }) - gy * q_ax));
    let dz = sanitize_f32(two_inv * ((if axis == 2 { 1.0 } else { 0.0 }) - gz * q_ax));
    (dw, dx, dy, dz)
}

/// Двойной лево-правый 4D ротор Кэли `q_L(g_L) * u * q_R(g_R) in SO(4)` (`step117`).
#[axiom_zero]
#[inline]
pub fn so4_double_rotor_4d(
    gl: (f32, f32, f32),
    gr: (f32, f32, f32),
    u: (f32, f32, f32, f32),
) -> (f32, f32, f32, f32) {
    let ql = q_stereo_unit_4d(gl.0, gl.1, gl.2);
    let qr = q_stereo_unit_4d(gr.0, gr.1, gr.2);
    quat_mul_4d(ql, quat_mul_4d(u, qr))
}

/// Гейтированное 4D отражение Хаусхолдера `H_beta(v) r = r - beta * (v^T r / (||v||^2 + eps)) * v` (`step118`).
#[axiom_zero]
#[inline]
pub fn householder4_gated(
    v: (f32, f32, f32, f32),
    beta: f32,
    eps: f32,
    r: (f32, f32, f32, f32),
) -> (f32, f32, f32, f32) {
    let b = sanitize_f32(beta).clamp(0.0, 2.0);
    let ep = sanitize_f32(eps).max(1e-12);
    let dv = sanitize_f32(v.0 * v.0 + v.1 * v.1 + v.2 * v.2 + v.3 * v.3 + ep);
    let inv_dv = total_div_f32(1.0, dv, 0.0);
    let vr = sanitize_f32(v.0 * r.0 + v.1 * r.1 + v.2 * r.2 + v.3 * r.3);
    let sc = sanitize_f32(b * vr * inv_dv);
    (
        sanitize_f32(r.0 - sc * v.0),
        sanitize_f32(r.1 - sc * v.1),
        sanitize_f32(r.2 - sc * v.2),
        sanitize_f32(r.3 - sc * v.3),
    )
}

/// Точная производная `d(H_beta(v) r) / d(v_c)` (`step127`).
#[axiom_zero]
#[inline]
pub fn householder4_gated_deriv_v(
    v: (f32, f32, f32, f32),
    beta: f32,
    eps: f32,
    r: (f32, f32, f32, f32),
    c_axis: usize,
) -> (f32, f32, f32, f32) {
    let b = sanitize_f32(beta).clamp(0.0, 2.0);
    if b <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let ep = sanitize_f32(eps).max(1e-12);
    let dv = sanitize_f32(v.0 * v.0 + v.1 * v.1 + v.2 * v.2 + v.3 * v.3 + ep);
    let inv_dv = total_div_f32(1.0, dv, 0.0);
    let vr = sanitize_f32(v.0 * r.0 + v.1 * r.1 + v.2 * r.2 + v.3 * r.3);
    let (vc, rc) = match c_axis {
        0 => (v.0, r.0),
        1 => (v.1, r.1),
        2 => (v.2, r.2),
        _ => (v.3, r.3),
    };
    let coeff_v = sanitize_f32(rc - 2.0 * vr * inv_dv * vc);
    let pref = sanitize_f32(-b * inv_dv);
    (
        sanitize_f32(pref * (coeff_v * v.0 + if c_axis == 0 { vr } else { 0.0 })),
        sanitize_f32(pref * (coeff_v * v.1 + if c_axis == 1 { vr } else { 0.0 })),
        sanitize_f32(pref * (coeff_v * v.2 + if c_axis == 2 { vr } else { 0.0 })),
        sanitize_f32(pref * (coeff_v * v.3 + if c_axis == 3 { vr } else { 0.0 })),
    )
}

/// Точный детерминант `4x4` матрицы по её 4 столбцам (0 делений, 0 `sqrt`).
#[axiom_zero]
#[inline]
pub fn det4x4_cols(
    c0: (f32, f32, f32, f32),
    c1: (f32, f32, f32, f32),
    c2: (f32, f32, f32, f32),
    c3: (f32, f32, f32, f32),
) -> f32 {
    let s0 = c0.0 * c1.1 - c1.0 * c0.1;
    let s1 = c0.0 * c2.1 - c2.0 * c0.1;
    let s2 = c0.0 * c3.1 - c3.0 * c0.1;
    let s3 = c1.0 * c2.1 - c2.0 * c1.1;
    let s4 = c1.0 * c3.1 - c3.0 * c1.1;
    let s5 = c2.0 * c3.1 - c3.0 * c2.1;

    let d5 = c2.2 * c3.3 - c3.2 * c2.3;
    let d4 = c1.2 * c3.3 - c3.2 * c1.3;
    let d3 = c1.2 * c2.3 - c2.2 * c1.3;
    let d2 = c0.2 * c3.3 - c3.2 * c0.3;
    let d1 = c0.2 * c2.3 - c2.2 * c0.3;
    let d0 = c0.2 * c1.3 - c1.2 * c0.3;

    sanitize_f32(s0 * d5 - s1 * d4 + s2 * d3 + s3 * d2 - s4 * d1 + s5 * d0)
}

/// Коэффициент ограниченного полиномиального отталкивания прототипов `max(0, dot - mu)` (`step123`).
#[axiom_zero]
#[inline]
pub fn poly_repulsion_grad_coeff(dot_sim: f32, mu: f32) -> f32 {
    let d = sanitize_f32(dot_sim).clamp(-1.0, 1.0);
    let m = sanitize_f32(mu).clamp(0.0, 0.99);
    if d > m {
        sanitize_f32(d - m)
    } else {
        0.0
    }
}

/// Разбор числового токена из байтов без аллокаций на куче.
#[axiom_zero]
pub fn parse_numeric_bytes(bytes: &[u8]) -> (bool, f32) {
    if bytes.is_empty() {
        return (false, 0.0);
    }
    let mut idx = 0usize;
    let mut sign = 1.0_f32;
    let b0 = total_get_copy(bytes, 0, 0);
    if b0 == b'-' {
        sign = -1.0;
        idx += 1;
    } else if b0 == b'+' {
        idx += 1;
    }
    let mut int_part = 0.0_f32;
    let mut frac_part = 0.0_f32;
    let mut frac_div = 1.0_f32;
    let mut in_frac = false;
    let mut digits = 0usize;

    while idx < bytes.len() {
        let b = total_get_copy(bytes, idx, 0);
        if b.is_ascii_digit() {
            let d = (b.saturating_sub(b'0')) as f32;
            if !in_frac {
                int_part = sanitize_f32(int_part * 10.0 + d);
            } else {
                frac_div = sanitize_f32(frac_div * 10.0);
                frac_part = sanitize_f32(frac_part + total_div_f32(d, frac_div, 0.0));
            }
            digits += 1;
            idx += 1;
        } else if (b == b'.' || b == b',') && !in_frac {
            in_frac = true;
            idx += 1;
        } else {
            return (false, 0.0);
        }
    }
    if digits == 0 {
        return (false, 0.0);
    }
    let v = sanitize_f32(sign * (int_part + frac_part));
    (v.is_finite(), v)
}

/// Разбор числового токена для непрерывного спинорного кодирования.
#[axiom_zero]
pub fn parse_numeric_token(token: &str) -> (bool, f32) {
    parse_numeric_bytes(token.as_bytes())
}

/// Извлекает чистые токены (без какого-либо test-specific удаления цифр!).
#[axiom_zero]
pub fn extract_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Компактный двухподпучковый спинорный энкодер (< 180 KB в замороженном i8 виде):
/// - Принимает универсально как `&[f32]` (через многомасштабные su(2) спиноры), так и `&str`.
/// - Поддерживает self-supervised Cloze-адаптацию для переноса синонимов (UnseenHoldout).
#[derive(Clone)]
pub struct SpinorEncoder {
    pub half_sectors: usize,
    pub total_sectors: usize,
    pub half_dim: usize,
    pub dim: usize,
    pub num_slots: usize,
    pub slot_mask: usize,
    pub sem_f32: Vec<f32>,
    pub sem_i8: Vec<i8>,
    pub sem_scales: Vec<f32>,
    pub is_frozen_i8: bool,
    pub domain_adapters: HashMap<usize, Vec<f32>>,
    pub mass_counts: Vec<u32>,
    pub mass_mask: usize,
    pub total_docs: u64,
}

#[axiom_zero]
impl SpinorEncoder {
    /// По умолчанию: `half_sectors = 32` -> `dim = 192` (`96D` sem + `96D` struct),
    /// `slots_pow2 = 10` (`1024` слота -> `96 KB` в i8), `mass_table = 4096` (`16 KB`).
    /// При `half_sectors == 0` или `slots_pow2 == 0` таблицы НЕ выделяются (`0` байт!).
    pub fn new(half_sectors: usize, slots_pow2: usize) -> Self {
        if half_sectors == 0 {
            return Self {
                half_sectors: 0,
                total_sectors: 0,
                half_dim: 0,
                dim: 0,
                num_slots: 0,
                slot_mask: 0,
                sem_f32: Vec::new(),
                sem_i8: Vec::new(),
                sem_scales: Vec::new(),
                is_frozen_i8: true,
                domain_adapters: HashMap::new(),
                mass_counts: Vec::new(),
                mass_mask: 0,
                total_docs: 0,
            };
        }
        let hs = half_sectors.clamp(8, 64);
        let ts = hs.saturating_mul(2);
        let hd = hs.saturating_mul(3);
        let d = ts.saturating_mul(3);

        if slots_pow2 == 0 {
            return Self {
                half_sectors: hs,
                total_sectors: ts,
                half_dim: hd,
                dim: d,
                num_slots: 0,
                slot_mask: 0,
                sem_f32: Vec::new(),
                sem_i8: Vec::new(),
                sem_scales: Vec::new(),
                is_frozen_i8: true,
                domain_adapters: HashMap::new(),
                mass_counts: Vec::new(),
                mass_mask: 0,
                total_docs: 0,
            };
        }

        let pow2 = slots_pow2.clamp(8, 12);
        let num_slots = 1usize << pow2;
        let slot_mask = num_slots - 1;
        let mass_len = 4096usize;

        let mut sem_f32 = vec![0.0_f32; num_slots.saturating_mul(hd)];
        for s in 0..num_slots {
            let off = s.saturating_mul(hd);
            let mut rng = splitmix64((s as u64).wrapping_add(1) ^ 0xA55A_3C3C_1122_3344);
            if let Some(sl) = sem_f32.get_mut(off..off + hd) {
                for v in sl.iter_mut() {
                    *v = rand_signed_f32(&mut rng);
                }
                sanitize_and_normalize_unit(sl, 8.0);
            }
        }

        Self {
            half_sectors: hs,
            total_sectors: ts,
            half_dim: hd,
            dim: d,
            num_slots,
            slot_mask,
            sem_f32,
            sem_i8: Vec::new(),
            sem_scales: Vec::new(),
            is_frozen_i8: false,
            domain_adapters: HashMap::new(),
            mass_counts: vec![0u32; mass_len],
            mass_mask: mass_len - 1,
            total_docs: 0,
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        let ad_bytes: usize = self
            .domain_adapters
            .values()
            .map(|v| v.len().saturating_mul(4) + 16)
            .sum();
        self.sem_f32.len().saturating_mul(4)
            + self.sem_i8.len()
            + (self.sem_scales.len() + self.mass_counts.len()).saturating_mul(4)
            + ad_bytes
    }

    pub fn freeze_to_i8(&mut self) {
        if self.is_frozen_i8 || self.sem_f32.is_empty() {
            return;
        }
        let hd = self.half_dim;
        let mut q = vec![0i8; self.num_slots.saturating_mul(hd)];
        let mut scales = vec![0.0_f32; self.num_slots];
        for s in 0..self.num_slots {
            let off = s.saturating_mul(hd);
            let Some(src) = self.sem_f32.get(off..off + hd) else {
                continue;
            };
            let mut ma = 0.0_f32;
            for &v in src {
                let av = sanitize_f32(v).abs();
                if av > ma {
                    ma = av;
                }
            }
            if ma > 1e-8 {
                let sc = total_div_f32(ma, 127.0, 0.0);
                let inv = total_div_f32(127.0, ma, 0.0);
                let _ = total_set(&mut scales, s, sc);
                if let Some(dst) = q.get_mut(off..off + hd) {
                    for (d, &v) in dst.iter_mut().zip(src.iter()) {
                        *d = (sanitize_f32(v) * inv).round().clamp(-127.0, 127.0) as i8;
                    }
                }
            }
        }
        self.sem_i8 = q;
        self.sem_scales = scales;
        self.sem_f32 = Vec::new();
        self.sem_f32.shrink_to_fit();
        self.is_frozen_i8 = true;
    }

    #[inline]
    pub fn token_slot_and_mass_bytes(&self, bytes: &[u8]) -> (usize, f32) {
        if self.num_slots == 0 {
            return (0, 0.14);
        }
        let h = fnv1a_bytes(bytes, 0x1111_2222_3333_4444);
        let slot = (h as usize) & self.slot_mask;
        let bucket = (h as usize) & self.mass_mask;
        let freq = total_get_copy(&self.mass_counts, bucket, 0);
        // Принципиальная защита от одноразового шума (без вырезания цифр!):
        // если слово ни разу не встречалось в корпусе, оно не получает гипертрофированный IDF.
        if freq == 0 && self.total_docs > 0 {
            return (slot, 0.14);
        }
        let docs = (self.total_docs.max(1) as f32) + 1.0;
        let ratio = total_div_f32(docs, (freq as f32) + 1.0, 1.0);
        let mass = sanitize_f32(algebraic_signed_log(ratio)).clamp(0.15, 6.0);
        (slot, mass)
    }

    #[inline]
    pub fn token_slot_and_mass(&self, tok: &str) -> (usize, f32) {
        self.token_slot_and_mass_bytes(tok.as_bytes())
    }

    #[inline]
    pub fn write_sem_slot_into(&self, slot: usize, dst: &mut [f32]) {
        if self.num_slots == 0 || dst.is_empty() {
            return;
        }
        let hd = self.half_dim;
        let off = slot.saturating_mul(hd);
        if self.is_frozen_i8 {
            let sc = total_get_copy(&self.sem_scales, slot, 0.0);
            if let Some(src) = self.sem_i8.get(off..off + hd) {
                for (d, &q) in dst.iter_mut().zip(src.iter()) {
                    *d = (q as f32) * sc;
                }
            }
            if let Some(ad) = self.domain_adapters.get(&slot) {
                fast_saxpy(dst, ad, 1.0);
            }
        } else if let Some(src) = self.sem_f32.get(off..off + hd) {
            for (d, &v) in dst.iter_mut().zip(src.iter()) {
                *d = v;
            }
        }
        sanitize_and_normalize_unit(dst, 8.0);
    }

    #[inline]
    fn add_struct_hash_into(&self, hash: u64, weight: f32, dst_half: &mut [f32]) {
        let w = sanitize_f32(weight);
        if w == 0.0 {
            return;
        }
        for k in 0..8usize {
            let m1 = splitmix64(hash.wrapping_add((k as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)));
            let m2 = splitmix64(m1 ^ 0xdead_beef_cafe_babe);
            let sec = total_rem_usize(m1 as usize, self.half_sectors, 0);
            let b = sec.saturating_mul(3);
            let qx = total_div_f32(((m1 >> 16) & 0xFFFF) as f32, 32767.5, 1.0) - 1.0;
            let qy = total_div_f32(((m1 >> 32) & 0xFFFF) as f32, 32767.5, 1.0) - 1.0;
            let qz = total_div_f32(((m2 >> 16) & 0xFFFF) as f32, 32767.5, 1.0) - 1.0;
            let inv = sanitize_f32(w * inv_sqrt_newton_f32((qx * qx + qy * qy + qz * qz).max(1e-12)));
            let _ = total_set(dst_half, b, total_get_copy(dst_half, b, 0.0) + qx * inv);
            let _ = total_set(dst_half, b + 1, total_get_copy(dst_half, b + 1, 0.0) + qy * inv);
            let _ = total_set(dst_half, b + 2, total_get_copy(dst_half, b + 2, 0.0) + qz * inv);
        }
    }

    /// Гладкое чисто алгебраическое многомасштабное su(2) кодирование скаляра (0% exp/ln/sin/cos/tanh).
    fn inject_scalar_spinor(&self, feat_idx: usize, val: f32, weight: f32, dst: &mut [f32]) {
        let x = sanitize_bounded_f32(val, 1e6);
        let signed_log = algebraic_signed_log(x);
        let smooth_z = sanitize_f32(algebraic_squash(x * 0.25) * 0.65);
        let xy_scale = sqrt_newton_f32((1.0_f32 - smooth_z * smooth_z).max(0.0));
        let freqs: [f32; 6] = [0.35, 0.85, 1.65, 3.2, 6.0, 11.0];

        let mut k = 0usize;
        while k < freqs.len() {
            let f = total_get_copy(&freqs, k, 1.0);
            let amp = total_div_f32(weight, 1.0 + 0.30 * (k as f32), weight);
            let sec = total_rem_usize(
                feat_idx.saturating_mul(7).saturating_add(k.saturating_mul(5)),
                self.total_sectors,
                0,
            );
            let b = sec.saturating_mul(3);
            let phase = sanitize_f32(f * signed_log + (feat_idx as f32) * 0.618034);
            let (sw, cw) = rational_circle_wave(phase);
            let qx = cw * xy_scale;
            let qy = sw * xy_scale;
            let _ = total_set(dst, b, total_get_copy(dst, b, 0.0) + amp * qx);
            let _ = total_set(dst, b + 1, total_get_copy(dst, b + 1, 0.0) + amp * qy);
            let _ = total_set(dst, b + 2, total_get_copy(dst, b + 2, 0.0) + amp * smooth_z);
            k += 1;
        }
    }

    /// Универсальный безколлизионный вход `&[f32] -> S^{D-1}` с нулевым DC-смещением в 0
    /// и СТРОГО 0% трансцендентных функций (0 exp, 0 ln, 0 sin, 0 cos, 0 tanh):
    /// - Каждый входной признак `i` получает собственные непересекающиеся 3D сектора с разными
    ///   частотами ротора Кэли (для точного декодирования K=2..8 в RepeatPrevious,
    ///   10D гиперплоскостей и точного знака угла theta в CartPole).
    pub fn encode_vec_into(&self, features: &[f32], out: &mut [f32]) {
        out.fill(0.0);
        let n_feat = features.len();
        if n_feat == 0 || self.total_sectors == 0 || out.is_empty() {
            sanitize_and_normalize_unit(out, 8.0);
            return;
        }
        // Сколько независимых 3D секторов выделяем каждому признаку (без коллизий до 16 признаков)
        let per_feat = total_div_usize(self.total_sectors, n_feat.max(1), 1).clamp(1, 6);
        let base_w = inv_sqrt_newton_f32(n_feat as f32);
        let freqs: [f32; 6] = [0.6, 1.3, 2.4, 4.2, 7.0, 11.5];

        for (i, &raw) in features.iter().enumerate() {
            let x = sanitize_bounded_f32(raw, 1e6);
            let lin = algebraic_squash(x * 0.5);
            let slog = algebraic_signed_log(x);

            for m in 0..per_feat {
                // Разносим сектора одного признака по всему спектру частот ротора Кэли!
                let sec = total_rem_usize(
                    m.saturating_mul(n_feat).saturating_add(i),
                    self.total_sectors,
                    0,
                );
                let b = sec.saturating_mul(3);
                let f = total_get_copy(&freqs, m, 1.0);
                let amp = total_div_f32(base_w, 1.0 + 0.65 * (m as f32), base_w);
                let (sw, cw) = rational_circle_wave(f * slog);
                let sx = sanitize_f32(lin);
                let sy = sanitize_f32(sw * 0.85);
                // cw(0) - 1 == 0 в нуле -> строго нулевое DC-смещение при x = 0!
                let sz = sanitize_f32((cw - 1.0) * 0.65);

                let _ = total_set(out, b, total_get_copy(out, b, 0.0) + amp * sx);
                let _ = total_set(out, b + 1, total_get_copy(out, b + 1, 0.0) + amp * sy);
                let _ = total_set(out, b + 2, total_get_copy(out, b + 2, 0.0) + amp * sz);
            }
        }

        // Стереографический якорь (0.50): сохраняет масштаб малых сигналов (CartPole theta, шум коридора T-Maze)
        // и даёт естественное смещение гиперплоскости w·x + b = 0 на сфере S^{D-1}.
        if let Some(last) = out.last_mut() {
            *last = sanitize_f32(*last + 0.50);
        }
        sanitize_and_normalize_unit(out, 8.0);
    }

    #[inline]
    fn encode_token_bytes_into(
        &self,
        tk: &[u8],
        out_full: &mut [f32],
        out_slots: &mut Vec<(usize, f32)>,
        tmp_sem: &mut [f32],
    ) {
        let hd = self.half_dim;
        let (is_num, num_val) = parse_numeric_bytes(tk);
        if is_num {
            // Числа кодируются непрерывно — без стриппинга цифр!
            self.inject_scalar_spinor(0, num_val, 0.20, out_full);
            return;
        }
        let (slot, mass) = self.token_slot_and_mass_bytes(tk);
        self.write_sem_slot_into(slot, tmp_sem);
        if let Some(sem_dst) = out_full.get_mut(0..hd) {
            fast_saxpy(sem_dst, tmp_sem, mass);
        }
        let h_id = fnv1a_bytes(tk, 0x517c_c1b7_2722_0a95);
        if let Some(st_dst) = out_full.get_mut(hd..self.dim) {
            self.add_struct_hash_into(h_id, mass, st_dst);
        }
        if out_slots.len() < 16 {
            out_slots.push((slot, mass));
        }
    }

    /// Кодирует текстовое наблюдение в полный `[dim]`-вектор без аллокаций на куче
    /// и извлекает слоты пакетов (для zero-shot переноса синонимов).
    pub fn encode_text_into(
        &self,
        text: &str,
        out_full: &mut [f32],
        out_slots: &mut Vec<(usize, f32)>,
        tmp_sem: &mut [f32],
    ) {
        out_full.fill(0.0);
        out_slots.clear();
        if self.dim == 0 || out_full.is_empty() {
            return;
        }
        let hd = self.half_dim;
        let mut tok_buf = [0u8; 64];
        let mut tok_len = 0usize;
        let mut tok_count = 0usize;
        let mut ch_buf = [0u8; 4];

        for ch in text.chars() {
            if ch.is_alphanumeric() {
                for lc in ch.to_lowercase() {
                    let s = lc.encode_utf8(&mut ch_buf);
                    let bs = s.as_bytes();
                    if tok_len + bs.len() <= 64 {
                        for &b in bs {
                            let _ = total_set(&mut tok_buf, tok_len, b);
                            tok_len += 1;
                        }
                    }
                }
            } else if tok_len > 0 {
                let tk = tok_buf.get(0..tok_len).unwrap_or(&[]);
                self.encode_token_bytes_into(tk, out_full, out_slots, tmp_sem);
                tok_count += 1;
                tok_len = 0;
            }
        }
        if tok_len > 0 {
            let tk = tok_buf.get(0..tok_len).unwrap_or(&[]);
            self.encode_token_bytes_into(tk, out_full, out_slots, tmp_sem);
            tok_count += 1;
        }
        if tok_count == 0 {
            sanitize_and_normalize_unit(out_full, 8.0);
            return;
        }
        if let Some(sem_dst) = out_full.get_mut(0..hd) {
            sanitize_and_normalize_unit(sem_dst, 8.0);
            fast_scale(sem_dst, 0.78);
        }
        if let Some(st_dst) = out_full.get_mut(hd..self.dim) {
            sanitize_and_normalize_unit(st_dst, 8.0);
            fast_scale(st_dst, 0.625);
        }
        sanitize_and_normalize_unit(out_full, 8.0);
    }

/// Самообучающаяся Cloze-адаптация на зашумлённых и несбалансированных (3:1) логах:
    /// - статистическая фильтрация глобального фона (df > 20% документов);
    /// - полнопредложенческое притяжение к доменным якорям + двустороннее связывание;
    /// - Bogoliubov-отталкивание по информативным словам;
    /// - Vacuum Gauge Removal в конце каждой эпохи (изотропия сферы).
    pub fn adapt_unlabeled_cloze(&mut self, docs: &[String], epochs: usize) {
        if docs.is_empty() || self.num_slots == 0 {
            return;
        }
        let hd = self.half_dim;
        let mut parsed: Vec<Vec<(usize, f32)>> = Vec::with_capacity(docs.len());
        let mut doc_sets: Vec<Vec<usize>> = Vec::with_capacity(docs.len());
        let mut all_slots: Vec<(usize, usize)> = Vec::new();
        let mut local_freq: HashMap<usize, u32> = HashMap::new();
        let mut doc_freq: HashMap<usize, u32> = HashMap::new();

        for doc in docs {
            let toks = extract_tokens(doc);
            if toks.len() < 3 {
                continue;
            }
            self.total_docs = self.total_docs.saturating_add(1);
            let d_idx = parsed.len();
            let mut row = Vec::with_capacity(toks.len());
            let mut uniq = Vec::with_capacity(toks.len());
            for tk in &toks {
                let (is_num, _) = parse_numeric_token(tk);
                if is_num {
                    continue;
                }
                let h = fnv1a_bytes(tk.as_bytes(), 0x1111_2222_3333_4444);
                let bucket = (h as usize) & self.mass_mask;
                let prev = total_get_copy(&self.mass_counts, bucket, 0);
                let _ = total_set(&mut self.mass_counts, bucket, prev.saturating_add(1));

                let sl = (h as usize) & self.slot_mask;
                *local_freq.entry(sl).or_insert(0) += 1;
                row.push((sl, 1.0_f32));
                if !uniq.contains(&sl) {
                    uniq.push(sl);
                    *doc_freq.entry(sl).or_insert(0) += 1;
                }
                all_slots.push((d_idx, sl));
            }
            if row.len() >= 2 {
                parsed.push(row);
                doc_sets.push(uniq);
            }
        }
        if parsed.is_empty() || all_slots.is_empty() {
            return;
        }

        // Статистический порог глобального фона: слово в > 20% документов = общий шум/шаблон
        let max_anchor_df = total_div_usize(parsed.len(), 5, 3).max(3) as u32;
        let docs_f = (self.total_docs.max(1) as f32) + 1.0;

        // Оставляем в doc_sets только информативные (не-фоновые) слова для честного Bogoliubov
        let mut info_doc_sets: Vec<Vec<usize>> = Vec::with_capacity(doc_sets.len());
        for set in &doc_sets {
            let mut info = Vec::new();
            for &sl in set {
                let df = doc_freq.get(&sl).copied().unwrap_or(1);
                if df <= max_anchor_df {
                    info.push(sl);
                }
            }
            info_doc_sets.push(info);
        }

        for row in &mut parsed {
            for (sl, ms) in row.iter_mut() {
                let df = doc_freq.get(sl).copied().unwrap_or(1).max(1);
                if df > max_anchor_df {
                    *ms = 0.0;
                } else {
                    let ratio = total_div_f32(docs_f, (df as f32) + 1.0, 1.0);
                    *ms = sanitize_f32(algebraic_signed_log(ratio)).clamp(0.5, 5.0);
                }
            }
        }

        let ep_n = epochs.clamp(1, 90);
        let radius = 8usize;
        let mut rng = 0x5EED_C102_E001_9988u64;
        let mut tv = vec![0.0_f32; hd];
        let mut ctx = vec![0.0_f32; hd];
        let mut cur_v = vec![0.0_f32; hd];
        let mut neg_v = vec![0.0_f32; hd];
        let mut active_slots: Vec<usize> = doc_freq
            .iter()
            .filter(|&(_, &df)| df <= max_anchor_df)
            .map(|(&sl, _)| sl)
            .collect();
        active_slots.sort_unstable();
        let mut mean_vec = vec![0.0_f32; hd];

        for ep in 0..ep_n {
            let ep_sc = total_div_f32(1.0, 1.0 + 0.02 * (ep as f32), 1.0);
            for (d_idx, row) in parsed.iter().enumerate() {
                let n = row.len();
                let cur_info = info_doc_sets.get(d_idx);
                for (pos, &(c_slot, c_mass)) in row.iter().enumerate() {
                    if c_mass <= 0.0 {
                        continue;
                    }
                    ctx.fill(0.0);
                    let start = pos.saturating_sub(radius);
                    let end = pos.saturating_add(radius + 1).min(n);
                    let mut nb_cnt = 0usize;
                    for nb in start..end {
                        if nb == pos {
                            continue;
                        }
                        if let Some(&(ns, nm)) = row.get(nb) {
                            if ns != c_slot && nm > 0.0 {
                                self.write_sem_slot_into(ns, &mut tv);
                                let dist = if pos > nb { pos - nb } else { nb - pos };
                                let ndf = doc_freq.get(&ns).copied().unwrap_or(1);
                                // Доменные слова-мосты (встречаются >= 2 раз, но не глобальный шум)
                                let bridge_boost = if ndf >= 2 { 1.8_f32 } else { 0.8_f32 };
                                let sal = total_div_f32(
                                    nm * bridge_boost,
                                    1.0 + 0.10 * ((dist - 1) as f32),
                                    nm,
                                );
                                fast_saxpy(&mut ctx, &tv, sal * sal);
                                nb_cnt += 1;
                            }
                        }
                    }
                    if nb_cnt == 0 {
                        continue;
                    }
                    sanitize_and_normalize_unit(&mut ctx, 8.0);
                    self.write_sem_slot_into(c_slot, &mut cur_v);
                    let dot_pos = fast_dot(&cur_v, &ctx);

                    let cfreq = local_freq.get(&c_slot).copied().unwrap_or(1).max(1) as f32;
                    let freq_damp = inv_sqrt_newton_f32(cfreq).clamp(0.25, 1.0);

                    if dot_pos < 0.94 {
                        let step = sanitize_f32((0.96 - dot_pos).clamp(0.05, 1.0) * 0.45 * freq_damp * ep_sc);
                        self.apply_sem_update(c_slot, &ctx, step);

                        let nb_step = sanitize_f32(step * 0.22);
                        for nb in start..end {
                            if nb != pos {
                                if let Some(&(ns, nm)) = row.get(nb) {
                                    if ns != c_slot && nm > 0.0 {
                                        let nf = local_freq.get(&ns).copied().unwrap_or(1).max(1) as f32;
                                        let adj = sanitize_f32(nb_step * inv_sqrt_newton_f32(nf));
                                        self.apply_sem_update(ns, &cur_v, adj);
                                    }
                                }
                            }
                        }
                    }

                    // Bogoliubov-отталкивание от других доменов (по информативным словам, без фоновых рамок!)
                    for _ in 0..5 {
                        rng = splitmix64(rng);
                        let ri = total_rem_usize((rng >> 16) as usize, all_slots.len(), 0);
                        let (nd, nslot) = all_slots.get(ri).copied().unwrap_or((d_idx, c_slot));
                        if nd == d_idx || nslot == c_slot {
                            continue;
                        }
                        if doc_freq.get(&nslot).copied().unwrap_or(1) > max_anchor_df {
                            continue;
                        }
                        if let (Some(cs), Some(ns_set)) = (cur_info, info_doc_sets.get(nd)) {
                            let mut shared = 0usize;
                            for &x in ns_set {
                                if cs.contains(&x) {
                                    shared += 1;
                                }
                            }
                            if shared >= 2 {
                                continue;
                            }
                        }
                        self.write_sem_slot_into(nslot, &mut neg_v);
                        let dot_neg = fast_dot(&neg_v, &ctx);
                        if dot_neg > 0.05 {
                            let viol = sanitize_f32((dot_neg - 0.05).clamp(0.02, 0.95));
                            self.apply_sem_update(c_slot, &neg_v, -0.22 * viol * freq_damp * ep_sc);
                        }
                    }
                }
            }

            // Vacuum Gauge Removal: вычитаем глобальное среднее всех активных слотов (изотропия сферы!)
            if active_slots.len() >= 8 {
                mean_vec.fill(0.0);
                for &sl in &active_slots {
                    self.write_sem_slot_into(sl, &mut tv);
                    fast_saxpy(&mut mean_vec, &tv, 1.0);
                }
                let mn = sqrt_newton_f32(fast_dot(&mean_vec, &mean_vec));
                if mn > 1e-4 {
                    sanitize_and_normalize_unit(&mut mean_vec, 8.0);
                    for &sl in &active_slots {
                        self.write_sem_slot_into(sl, &mut tv);
                        let proj = fast_dot(&tv, &mean_vec);
                        if proj.abs() > 0.02 {
                            self.apply_sem_update(sl, &mean_vec, -0.85 * proj);
                        }
                    }
                }
            }
        }
    }

    fn apply_sem_update(&mut self, slot: usize, dir: &[f32], alpha: f32) {
        let hd = self.half_dim;
        if self.is_frozen_i8 {
            let ad = self.domain_adapters.entry(slot).or_insert_with(|| vec![0.0_f32; hd]);
            fast_saxpy(ad, dir, alpha);
            let n2 = fast_dot(ad, ad);
            if n2 > 9.0 {
                fast_scale(ad, sanitize_f32(3.0 * inv_sqrt_newton_f32(n2)));
            }
        } else {
            let off = slot.saturating_mul(hd);
            if let Some(sl) = self.sem_f32.get_mut(off..off + hd) {
                fast_saxpy(sl, dir, alpha);
                sanitize_and_normalize_unit(sl, 8.0);
            }
        }
    }
}
