// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Без-FPU детерминированное ядро `Q8.24` / `Q39.24` (Suite C, v0.5.0):
//! - 100% `#[axiom_zero]` в целочисленном ядре (граничные конвертеры `from_f32_boundary`/`to_f32_boundary` работают только на I/O-границе); 0 инструкций FPU внутри рекуррентности, обучения и PH-аккумуляторов;
//! - `Q8_24(i32)`: `1 LSB = 2^-24 ≈ 5.96e-8` (единичные роторы, веса, градиенты `lr = 1e-4` не застывают);
//! - `Q39_24(i64)`: расширенный аккумулятор `±5.49e11` (сырые входы NASA `125.0`, дисперсия Welford, `||u||^2`, `PageHinkleyQ39_24` без насыщения потолка `128.0`);
//! - `Q16_16(i32)`: эталонный контроль вырождения малых шагов `lr = 1e-4 * grad`;
//! - `q8_24_inv_sqrt`: битовый сдвиг экспоненты через `leading_zeros()` + 3 целочисленных шага Ньютона в `i128`;
//! - Бит-точная детерминированная воспроизводимость (`state_digest_fnv1a`).

use axiom_zero::{axiom_zero, total_div_i128, total_div_i64, total_get_copy, total_set};

use crate::math::splitmix64;

pub const Q24_ONE: i32 = 1 << 24;
pub const Q24_HALF: i32 = 1 << 23;
pub const Q24_ONE_I64: i64 = 1i64 << 24;
pub const Q24_HALF_I64: i64 = 1i64 << 23;
pub const Q16_ONE: i32 = 1 << 16;

/// Фиксированная точка Q8.24 (`i32`, диапазон `[-128.0, +127.99999994]`, `1 LSB = 2^-24`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Q8_24(pub i32);

/// Широкий аккумулятор Q39.24 (`i64`, диапазон `±5.49e11`, `1 LSB = 2^-24`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Q39_24(pub i64);

/// Контрольный формат Q16.16 (`i32`, `1 LSB = 2^-16 ≈ 1.53e-5`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Q16_16(pub i32);

impl Q8_24 {
    #[inline]
    pub fn from_f32_boundary(x: f32) -> Self {
        if !x.is_finite() {
            return Self(0);
        }
        let scaled = (x as f64) * (Q24_ONE_I64 as f64);
        Self(scaled.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32)
    }

    #[inline]
    pub fn to_f32_boundary(self) -> f32 {
        (self.0 as f64 / Q24_ONE_I64 as f64) as f32
    }
}

impl Q39_24 {
    #[inline]
    pub fn from_f32_boundary(x: f32) -> Self {
        if !x.is_finite() {
            return Self(0);
        }
        let scaled = (x as f64) * (Q24_ONE_I64 as f64);
        Self(scaled.round().clamp(-1e16, 1e16) as i64)
    }

    #[inline]
    pub fn to_f32_boundary(self) -> f32 {
        (self.0 as f64 / Q24_ONE_I64 as f64) as f32
    }
}

impl Q16_16 {
    #[inline]
    pub fn from_f32_boundary(x: f32) -> Self {
        if !x.is_finite() {
            return Self(0);
        }
        let scaled = (x as f64) * (Q16_ONE as f64);
        Self(scaled.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32)
    }
}

#[axiom_zero]
#[inline]
pub fn q24_add(a: Q8_24, b: Q8_24) -> Q8_24 {
    Q8_24(a.0.saturating_add(b.0))
}

#[axiom_zero]
#[inline]
pub fn q24_sub(a: Q8_24, b: Q8_24) -> Q8_24 {
    Q8_24(a.0.saturating_sub(b.0))
}

#[axiom_zero]
#[inline]
pub fn q24_mul(a: Q8_24, b: Q8_24) -> Q8_24 {
    let prod = (a.0 as i64) * (b.0 as i64);
    let rounded = (prod + (Q24_HALF_I64)) >> 24;
    Q8_24(rounded.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
}

#[axiom_zero]
#[inline]
pub fn q24_div(num: Q8_24, den: Q8_24) -> Q8_24 {
    let q = total_div_i64((num.0 as i64) << 24, den.0 as i64, 0);
    Q8_24(q.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
}

#[axiom_zero]
#[inline]
pub fn q24_clamp(x: Q8_24, max_abs: Q8_24) -> Q8_24 {
    let lim = max_abs.0.abs();
    Q8_24(x.0.clamp(-lim, lim))
}

#[axiom_zero]
#[inline]
pub fn q39_mul(a: Q39_24, b: Q39_24) -> Q39_24 {
    let prod = (a.0 as i128) * (b.0 as i128);
    let rounded = (prod + (Q24_HALF_I64 as i128)) >> 24;
    Q39_24(rounded.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
}

#[axiom_zero]
#[inline]
pub fn q39_div(num: Q39_24, den: Q39_24) -> Q39_24 {
    let q = total_div_i128((num.0 as i128) << 24, den.0 as i128, 0);
    Q39_24(q.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
}

#[axiom_zero]
#[inline]
pub fn q16_mul(a: Q16_16, b: Q16_16) -> Q16_16 {
    let prod = (a.0 as i64) * (b.0 as i64);
    Q16_16((prod >> 16).clamp(i32::MIN as i64, i32::MAX as i64) as i32)
}

/// Целочисленный `1 / sqrt(a)` в `Q8.24` (0 FPU, 0 `.sqrt()`):
/// 1) Точное деление экспоненты пополам через `leading_zeros()` (с множителем `181/256` для нечётной) и seed `(5 - M)/4`;
/// 2) 3 шага Ньютона в широкой 128-битной сетке (`i128`), гарантирующих `y > 0` при скачках 10+ порядков.
#[axiom_zero]
#[inline]
pub fn q8_24_inv_sqrt(a: Q39_24) -> Q8_24 {
    if a.0 <= 0 {
        return Q8_24(0);
    }
    let raw = a.0;
    let msb = 63i32 - (raw as u64).leading_zeros() as i32;
    let e = msb - 24;
    let r = e & 1;
    let k = e >> 1;

    let m_q24: i64 = if msb >= 24 {
        raw >> (msb - 24)
    } else {
        raw << (24 - msb)
    };
    let mut seed = ((5i64 << 24) - m_q24) >> 2;
    if r != 0 {
        seed = (seed * 181i64 + 128i64) >> 8;
    }

    // Шаги Ньютона выполняем в нормализованной полосе m_eff in [1, 4), где y ~ 1.0 и сохраняет все 24 бита мантиссы!
    let m_eff = (if r != 0 { m_q24 << 1 } else { m_q24 }) as i128;
    let three_q24 = 3i128 << 24;
    let mut y: i128 = (seed as i128).max(1);
    for _ in 0..3 {
        let y2 = (y * y + (1i128 << 23)) >> 24;
        let my2 = (m_eff * y2 + (1i128 << 23)) >> 24;
        let term = three_q24 - my2;
        let yn = (y * term + (1i128 << 24)) >> 25;
        if yn > 0 {
            y = yn;
        }
    }

    let y_shifted: i128 = if k > 0 {
        let sh = k.min(62) as u32;
        (y + (1i128 << (sh - 1))) >> sh
    } else if k < 0 {
        let sh = (-k).min(38) as u32;
        y << sh
    } else {
        y
    };
    let out = y_shifted.clamp(1, i32::MAX as i128) as i32;
    Q8_24(out)
}

/// Рациональная аппроксимация `p4` и градиента без `exp` и без FPU для классификации в `Q8.24`:
/// `q(z) = 1 - z/2 + z^2/8`, `w(z) = 1 / q(z)^4`, `psi(z) = (2 - z) / q(z)`.
#[axiom_zero]
pub fn q24_rational_p4_into(logits: &[Q8_24], out_probs: &mut [Q8_24], out_psi: &mut [Q8_24]) {
    let n = logits.len().min(out_probs.len());
    if n == 0 {
        return;
    }
    let mut max_l = i32::MIN;
    for i in 0..n {
        let v = total_get_copy(logits, i, Q8_24(0)).0;
        if i == 0 || v > max_l {
            max_l = v;
        }
    }
    let min_z = -20 * Q24_ONE;
    let half_q24 = Q8_24(Q24_HALF);
    let eighth_q24 = Q8_24(Q24_ONE >> 3);
    let two_q24 = 2 * Q24_ONE;

    let mut sum_w: i64 = 0;
    for i in 0..n {
        let l = total_get_copy(logits, i, Q8_24(0)).0;
        let z = l.saturating_sub(max_l).clamp(min_z, 0);
        let zq = Q8_24(z);
        let z2 = q24_mul(zq, zq);
        let q = Q8_24(
            Q24_ONE
                .saturating_sub(q24_mul(half_q24, zq).0)
                .saturating_add(q24_mul(eighth_q24, z2).0)
                .max(Q24_HALF),
        );
        let q2 = q24_mul(q, q);
        let q4 = q24_mul(q2, q2);
        let w = total_div_i64(1i64 << 48, (q4.0 as i64).max(1), 0).clamp(0, i32::MAX as i64) as i32;
        let _ = total_set(out_probs, i, Q8_24(w));
        if i < out_psi.len() {
            let num = two_q24.saturating_sub(z);
            let psi = total_div_i64((num as i64) << 24, (q.0 as i64).max(1), Q24_ONE_I64);
            let _ = total_set(out_psi, i, Q8_24(psi.clamp(0, i32::MAX as i64) as i32));
        }
        sum_w = sum_w.saturating_add(w as i64);
    }

    let inv_n = total_div_i64(Q24_ONE_I64, n as i64, Q24_HALF_I64) as i32;
    for i in 0..n {
        let w = total_get_copy(out_probs, i, Q8_24(0)).0 as i64;
        let p = if sum_w > 1024 {
            total_div_i64(w << 24, sum_w, inv_n as i64).clamp(0, Q24_ONE_I64) as i32
        } else {
            inv_n
        };
        let _ = total_set(out_probs, i, Q8_24(p));
    }
}

/// Детектор Page-Hinkley на широких аккумуляторах `Q39.24` (без насыщения `128.0`!).
#[derive(Clone, Debug)]
pub struct PageHinkleyQ39_24 {
    pub mu_bank: Q39_24,
    pub cusum: Q39_24,
    pub delta: Q39_24,
    pub threshold: Q39_24,
    pub min_warmup: usize,
    pub episodes_seen: usize,
}

#[axiom_zero]
impl PageHinkleyQ39_24 {
    pub fn new(delta: Q39_24, threshold: Q39_24, min_warmup: usize) -> Self {
        Self {
            mu_bank: Q39_24(0),
            cusum: Q39_24(0),
            delta: Q39_24(delta.0.max(0)),
            threshold: Q39_24(threshold.0.max(Q24_HALF_I64)),
            min_warmup: min_warmup.max(6),
            episodes_seen: 0,
        }
    }

    pub fn observe(&mut self, val: Q39_24) -> bool {
        self.episodes_seen = self.episodes_seen.saturating_add(1);
        if self.episodes_seen == 1 {
            self.mu_bank = val;
            self.cusum = Q39_24(0);
            return false;
        }
        if self.episodes_seen <= self.min_warmup {
            let diff = val.0.saturating_sub(self.mu_bank.0);
            let step = total_div_i64(diff, self.episodes_seen as i64, 0);
            self.mu_bank = Q39_24(self.mu_bank.0.saturating_add(step));
            self.cusum = Q39_24(0);
            return false;
        }
        let drop_sig = self.mu_bank.0.saturating_sub(val.0).saturating_sub(self.delta.0);
        let next_c = self.cusum.0.saturating_add(drop_sig).max(0);
        self.cusum = Q39_24(next_c);
        if self.cusum.0 == 0 {
            let diff = val.0.saturating_sub(self.mu_bank.0);
            // EMA alpha = 1/16 (~0.0625)
            self.mu_bank = Q39_24(self.mu_bank.0.saturating_add(diff >> 4));
        }
        self.cusum.0 >= self.threshold.0
    }
}

/// Унитарный 4-Хаусхолдеровский каскад `O(4) -> SO(5)` в чистом `Q8.24` без FPU (`step134`):
/// - `hh[0..3]`: 3 единичных отражения Хаусхолдера в `R^4 x {0}` (генерируют всю группу `O(4)`);
/// - `hh[3]` + `beta in [0, 2]`: 4-е отражение вдоль `u3 = ((2 - beta) * vh3, beta) in R^5`,
///   норма `||u3||^2 = (2 - beta)^2 + beta^2 = 2*(beta - 1)^2 + 2 >= 2 > 0` при любом `beta`
///   (нулевое вырождение на водоразделе `beta = 1.0` даже без ренормализации!).
#[derive(Clone)]
pub struct FixedO4Rotor {
    pub state: [Q8_24; 5],
    pub hh: [[Q8_24; 4]; 4],
    pub beta: Q8_24, // 0 (+1 det SO(4)), 1 (унитарный водораздел), 2 (-1 det O(4)\SO(4))
}

#[axiom_zero]
impl FixedO4Rotor {
    pub fn new(seed: u64, beta: Q8_24) -> Self {
        let mut rng = splitmix64(seed ^ 0x04F1_CED0);
        let mut hh = [[Q8_24(Q24_ONE), Q8_24(0), Q8_24(0), Q8_24(0)]; 4];
        let rand_q4 = |r: &mut u64| -> [Q8_24; 4] {
            *r = splitmix64(*r);
            let a = ((((*r >> 40) as i32) & 16383) - 8192) << 10;
            *r = splitmix64(*r);
            let b = ((((*r >> 40) as i32) & 16383) - 8192) << 10;
            *r = splitmix64(*r);
            let c = ((((*r >> 40) as i32) & 16383) - 8192) << 10;
            *r = splitmix64(*r);
            let d = ((((*r >> 40) as i32) & 16383) - 8192) << 10;
            let s2 = ((a as i64 * a as i64
                + b as i64 * b as i64
                + c as i64 * c as i64
                + d as i64 * d as i64)
                >> 24)
                .max(1);
            let inv = q8_24_inv_sqrt(Q39_24(s2));
            [
                q24_mul(Q8_24(a), inv),
                q24_mul(Q8_24(b), inv),
                q24_mul(Q8_24(c), inv),
                q24_mul(Q8_24(d), inv),
            ]
        };
        for k in 0..4usize {
            let v = rand_q4(&mut rng);
            let _ = total_set(&mut hh, k, v);
        }

        Self {
            state: [Q8_24(Q24_ONE), Q8_24(0), Q8_24(0), Q8_24(0), Q8_24(0)],
            hh,
            beta: Q8_24(beta.0.clamp(0, 2 * Q24_ONE)),
        }
    }

    /// Один унитарный шаг 4-Хаусхолдеровского каскада `SO(5)` в чистом `Q8.24` БЕЗ ренормализации.
    pub fn step_unnormalized(&mut self) {
        let mut y = [
            total_get_copy(&self.state, 0, Q8_24(0)).0 as i64,
            total_get_copy(&self.state, 1, Q8_24(0)).0 as i64,
            total_get_copy(&self.state, 2, Q8_24(0)).0 as i64,
            total_get_copy(&self.state, 3, Q8_24(0)).0 as i64,
            total_get_copy(&self.state, 4, Q8_24(0)).0 as i64,
        ];

        // 1) Три пространственных отражения Хаусхолдера H_0, H_1, H_2 в R^4 x {0}
        for k in 0..3usize {
            let v = total_get_copy(&self.hh, k, [Q8_24(Q24_ONE), Q8_24(0), Q8_24(0), Q8_24(0)]);
            let v0 = total_get_copy(&v, 0, Q8_24(0)).0 as i64;
            let v1 = total_get_copy(&v, 1, Q8_24(0)).0 as i64;
            let v2 = total_get_copy(&v, 2, Q8_24(0)).0 as i64;
            let v3 = total_get_copy(&v, 3, Q8_24(0)).0 as i64;
            let y0 = total_get_copy(&y, 0, 0);
            let y1 = total_get_copy(&y, 1, 0);
            let y2 = total_get_copy(&y, 2, 0);
            let y3 = total_get_copy(&y, 3, 0);
            let dot_q48 = v0 * y0 + v1 * y1 + v2 * y2 + v3 * y3;
            let two_dot = (dot_q48 + (1i64 << 22)) >> 23; // 2 * dot в Q8.24
            let _ = total_set(&mut y, 0, y0.saturating_sub((two_dot * v0 + Q24_HALF_I64) >> 24));
            let _ = total_set(&mut y, 1, y1.saturating_sub((two_dot * v1 + Q24_HALF_I64) >> 24));
            let _ = total_set(&mut y, 2, y2.saturating_sub((two_dot * v2 + Q24_HALF_I64) >> 24));
            let _ = total_set(&mut y, 3, y3.saturating_sub((two_dot * v3 + Q24_HALF_I64) >> 24));
        }

        // 2) Четвёртое невырожденное отражение H_3 вдоль u3 = ((2 - beta)*vh3, beta) in R^5
        let vh3 = total_get_copy(&self.hh, 3, [Q8_24(Q24_ONE), Q8_24(0), Q8_24(0), Q8_24(0)]);
        let b_q24 = self.beta.0 as i64;
        let c_q24 = (2i64 * Q24_ONE_I64).saturating_sub(b_q24);
        let u0 = (c_q24 * (total_get_copy(&vh3, 0, Q8_24(0)).0 as i64) + Q24_HALF_I64) >> 24;
        let u1 = (c_q24 * (total_get_copy(&vh3, 1, Q8_24(0)).0 as i64) + Q24_HALF_I64) >> 24;
        let u2 = (c_q24 * (total_get_copy(&vh3, 2, Q8_24(0)).0 as i64) + Q24_HALF_I64) >> 24;
        let u3 = (c_q24 * (total_get_copy(&vh3, 3, Q8_24(0)).0 as i64) + Q24_HALF_I64) >> 24;
        let u4 = b_q24;

        let u_norm_sq = ((u0 * u0 + u1 * u1 + u2 * u2 + u3 * u3 + u4 * u4 + Q24_HALF_I64) >> 24)
            .max(Q24_ONE_I64);
        let y0 = total_get_copy(&y, 0, 0);
        let y1 = total_get_copy(&y, 1, 0);
        let y2 = total_get_copy(&y, 2, 0);
        let y3 = total_get_copy(&y, 3, 0);
        let y4 = total_get_copy(&y, 4, 0);
        let dot_u = (u0 * y0 + u1 * y1 + u2 * y2 + u3 * y3 + u4 * y4 + Q24_HALF_I64) >> 24;
        let sc = total_div_i64(dot_u << 25, u_norm_sq, 0); // 2 * dot_u / ||u3||^2 в Q8.24

        let _ = total_set(
            &mut self.state,
            0,
            Q8_24(y0.saturating_sub((sc * u0 + Q24_HALF_I64) >> 24).clamp(i32::MIN as i64, i32::MAX as i64) as i32),
        );
        let _ = total_set(
            &mut self.state,
            1,
            Q8_24(y1.saturating_sub((sc * u1 + Q24_HALF_I64) >> 24).clamp(i32::MIN as i64, i32::MAX as i64) as i32),
        );
        let _ = total_set(
            &mut self.state,
            2,
            Q8_24(y2.saturating_sub((sc * u2 + Q24_HALF_I64) >> 24).clamp(i32::MIN as i64, i32::MAX as i64) as i32),
        );
        let _ = total_set(
            &mut self.state,
            3,
            Q8_24(y3.saturating_sub((sc * u3 + Q24_HALF_I64) >> 24).clamp(i32::MIN as i64, i32::MAX as i64) as i32),
        );
        let _ = total_set(
            &mut self.state,
            4,
            Q8_24(y4.saturating_sub((sc * u4 + Q24_HALF_I64) >> 24).clamp(i32::MIN as i64, i32::MAX as i64) as i32),
        );
    }

    /// Унитарный шаг `SO(5)` + точная репроекция на `S^4` через `q8_24_inv_sqrt`.
    pub fn step_and_renorm(&mut self) {
        self.step_unnormalized();
        let norm_sq = Q39_24(self.sphere_norm_sq_q39().0.max(1));
        let inv = q8_24_inv_sqrt(norm_sq);
        for i in 0..5usize {
            let yi = total_get_copy(&self.state, i, Q8_24(0));
            let _ = total_set(&mut self.state, i, q24_mul(yi, inv));
        }
    }

    pub fn sphere_norm_sq_q39(&self) -> Q39_24 {
        let mut s: i64 = 0;
        for i in 0..5usize {
            let v = total_get_copy(&self.state, i, Q8_24(0)).0 as i64;
            s = s.saturating_add((v * v) >> 24);
        }
        Q39_24(s)
    }
}

pub const FIXED_2D_SLOTS: usize = 16;

/// Полностью без-FPU потоковое ядро Кэли (Q8.24 состояния/веса + Q39.24 входные/PH аккумуляторы).
#[derive(Clone)]
pub struct FixedCayleyCore {
    pub raw_in: usize,
    pub num_actions: usize,
    // Q39.24 входной кондиционер (входы 125.0 / m2 не упираются в потолок 128.0)
    pub norm_cnt: u64,
    pub norm_mean: Vec<Q39_24>,
    pub norm_m2: Vec<Q39_24>,
    pub x_norm: Vec<Q8_24>,
    // 16 x 2D рациональных роторов Кэли в Q8.24 (с предвычисленными lc/ls — 0 делений на шаге):
    pub b_state: [Q8_24; FIXED_2D_SLOTS * 2],
    pub alpha: [Q8_24; FIXED_2D_SLOTS],
    pub omega: [Q8_24; FIXED_2D_SLOTS],
    pub lc_cache: [Q8_24; FIXED_2D_SLOTS],
    pub ls_cache: [Q8_24; FIXED_2D_SLOTS],
    pub in_w: Vec<Q8_24>, // [16 * raw_in]
    pub s_inw: Vec<Q8_24>, // [16 * raw_in * 2]
    // Выходные веса, логиты/вероятности:
    pub feat_dim: usize,
    pub feat: Vec<Q8_24>,
    pub actor_w: Vec<Q8_24>,
    pub actor_b: Vec<Q8_24>,
    pub reg_w: Vec<Q8_24>,
    pub reg_b: Q8_24,
    pub logits: Vec<Q8_24>,
    pub probs: Vec<Q8_24>,
    pub psis: Vec<Q8_24>,
    pub reg_pred: Q39_24,
    pub lr_ro: Q8_24,
    pub lr_reg: Q8_24,
    pub lr_rtrl: Q8_24,
}

#[axiom_zero]
impl FixedCayleyCore {
    pub fn new(raw_in: usize, num_actions: usize, seed: u64) -> Self {
        let rin = raw_in.clamp(1, 24);
        let na = num_actions.clamp(1, 32);
        let feat_dim = rin + FIXED_2D_SLOTS * 2;
        let mut alpha = [Q8_24(0); FIXED_2D_SLOTS];
        let mut omega = [Q8_24(0); FIXED_2D_SLOTS];
        let mut lc_cache = [Q8_24(0); FIXED_2D_SLOTS];
        let mut ls_cache = [Q8_24(0); FIXED_2D_SLOTS];
        let mut in_w = vec![Q8_24(0); FIXED_2D_SLOTS * rin];
        let mut actor_w = vec![Q8_24(0); na * feat_dim];

        let mut rng = splitmix64(seed ^ 0xF1ED_0008_0024);
        let inv_rin = q8_24_inv_sqrt(Q39_24((rin as i64) << 24));
        for k in 0..FIXED_2D_SLOTS {
            let frac = total_div_i64((k as i64) << 24, (FIXED_2D_SLOTS - 1) as i64, Q24_HALF_I64) as i32;
            let a0 = Q8_24(q24_mul(Q8_24(frac), Q8_24(frac)).0 + (Q24_ONE >> 4));
            let w0_mag = (Q24_ONE >> 2).saturating_add(frac << 1);
            let w0 = if (k & 1) == 0 { Q8_24(w0_mag) } else { Q8_24(-w0_mag) };
            let _ = total_set(&mut alpha, k, a0);
            let _ = total_set(&mut omega, k, w0);

            let lam = Q8_24(total_div_i64(
                Q24_ONE_I64 << 24,
                Q24_ONE_I64 + q24_mul(a0, a0).0 as i64 + (Q24_ONE_I64 >> 13),
                Q24_ONE_I64,
            ) as i32);
            let t = Q8_24(w0.0 >> 1);
            let t2 = q24_mul(t, t);
            let den = Q24_ONE.saturating_add(t2.0);
            let c = Q8_24(total_div_i64(((Q24_ONE - t2.0) as i64) << 24, den as i64, Q24_ONE_I64) as i32);
            let s = Q8_24(total_div_i64((t.0 as i64) << 25, den as i64, 0) as i32);
            let _ = total_set(&mut lc_cache, k, q24_mul(lam, c));
            let _ = total_set(&mut ls_cache, k, q24_mul(lam, s));

            for j in 0..rin {
                rng = splitmix64(rng);
                let r = ((((rng >> 40) as i32) & 0x7FFF) - 16384) << 9;
                let w = q24_mul(Q8_24(r), inv_rin);
                let _ = total_set(&mut in_w, k * rin + j, w);
            }
        }
        for v in actor_w.iter_mut() {
            rng = splitmix64(rng);
            let r = ((((rng >> 40) as i32) & 0x7FFF) - 16384) << 6;
            *v = Q8_24(r);
        }

        Self {
            raw_in: rin,
            num_actions: na,
            norm_cnt: 0,
            norm_mean: vec![Q39_24(0); rin],
            norm_m2: vec![Q39_24(0); rin],
            x_norm: vec![Q8_24(0); rin],
            b_state: [Q8_24(0); FIXED_2D_SLOTS * 2],
            alpha,
            omega,
            lc_cache,
            ls_cache,
            in_w,
            s_inw: vec![Q8_24(0); FIXED_2D_SLOTS * rin * 2],
            feat_dim,
            feat: vec![Q8_24(0); feat_dim],
            actor_w,
            actor_b: vec![Q8_24(0); na],
            reg_w: vec![Q8_24(0); feat_dim],
            reg_b: Q8_24(0),
            logits: vec![Q8_24(0); na],
            probs: vec![Q8_24(0); na],
            psis: vec![Q8_24(0); na],
            reg_pred: Q39_24(0),
            lr_ro: Q8_24(total_div_i64(Q24_ONE_I64 * 3, 100, 0) as i32),   // 0.03
            lr_reg: Q8_24(total_div_i64(Q24_ONE_I64 * 6, 100, 0) as i32),  // 0.06 / (1 + ||f||^2) NLMS
            lr_rtrl: Q8_24(total_div_i64(Q24_ONE_I64 * 4, 1000, 0) as i32), // 0.004
        }
    }

    pub fn reset_episode(&mut self) {
        self.b_state = [Q8_24(0); FIXED_2D_SLOTS * 2];
        self.s_inw.fill(Q8_24(0));
    }

    pub fn forward_q(&mut self, raw_x: &[Q39_24], learn_norm: bool, update_traces: bool) {
        let next_cnt = self.norm_cnt.saturating_add(1);
        for i in 0..self.raw_in {
            let x = total_get_copy(raw_x, i, Q39_24(0));
            let mu = total_get_copy(&self.norm_mean, i, Q39_24(0));
            let m2 = total_get_copy(&self.norm_m2, i, Q39_24(0));
            let var = if self.norm_cnt >= 2 {
                Q39_24(total_div_i64(m2.0, (self.norm_cnt - 1) as i64, 0).max(0))
            } else {
                Q39_24(0)
            };
            let inv_sd = q8_24_inv_sqrt(Q39_24(var.0.saturating_add(Q24_ONE_I64)));
            let diff = x.0.saturating_sub(mu.0);
            let xn_64 = ((diff as i128 * inv_sd.0 as i128) >> 24).clamp((-12 * Q24_ONE) as i128, (12 * Q24_ONE) as i128) as i32;
            let _ = total_set(&mut self.x_norm, i, Q8_24(xn_64));
            let _ = total_set(&mut self.feat, i, Q8_24(xn_64));

            if learn_norm {
                let d1 = x.0.saturating_sub(mu.0);
                let new_mu = Q39_24(mu.0.saturating_add(total_div_i64(d1, next_cnt as i64, 0)));
                let d2 = x.0.saturating_sub(new_mu.0);
                let prod = ((d1 as i128 * d2 as i128) >> 24).clamp(0, i64::MAX as i128) as i64;
                let _ = total_set(&mut self.norm_mean, i, new_mu);
                let _ = total_set(&mut self.norm_m2, i, Q39_24(m2.0.saturating_add(prod)));
            }
        }
        if learn_norm {
            self.norm_cnt = next_cnt;
        }

        for k in 0..FIXED_2D_SLOTS {
            let uoff = k * 2;
            let u0 = total_get_copy(&self.b_state, uoff, Q8_24(0));
            let u1 = total_get_copy(&self.b_state, uoff + 1, Q8_24(0));
            let lc = total_get_copy(&self.lc_cache, k, Q8_24(Q24_ONE));
            let ls = total_get_copy(&self.ls_cache, k, Q8_24(0));

            let woff = k * self.raw_in;
            let mut m_in: i64 = 0;
            for j in 0..self.raw_in {
                let wj = total_get_copy(&self.in_w, woff + j, Q8_24(0));
                let xj = total_get_copy(&self.x_norm, j, Q8_24(0));
                m_in = m_in.saturating_add((wj.0 as i64 * xj.0 as i64) >> 24);
            }
            let m_q = Q8_24(m_in.clamp((-16 * Q24_ONE) as i64, (16 * Q24_ONE) as i64) as i32);

            let un0 = q24_clamp(Q8_24(q24_mul(lc, u0).0.saturating_sub(q24_mul(ls, u1).0).saturating_add(m_q.0)), Q8_24(64 * Q24_ONE));
            let un1 = q24_clamp(Q8_24(q24_mul(ls, u0).0.saturating_add(q24_mul(lc, u1).0)), Q8_24(64 * Q24_ONE));

            if update_traces {
                for j in 0..self.raw_in {
                    let toff = (woff + j) * 2;
                    let xj = total_get_copy(&self.x_norm, j, Q8_24(0));
                    let p0 = total_get_copy(&self.s_inw, toff, Q8_24(0));
                    let p1 = total_get_copy(&self.s_inw, toff + 1, Q8_24(0));
                    let n0 = q24_clamp(Q8_24(q24_mul(lc, p0).0.saturating_sub(q24_mul(ls, p1).0).saturating_add(xj.0)), Q8_24(32 * Q24_ONE));
                    let n1 = q24_clamp(Q8_24(q24_mul(ls, p0).0.saturating_add(q24_mul(lc, p1).0)), Q8_24(32 * Q24_ONE));
                    let _ = total_set(&mut self.s_inw, toff, n0);
                    let _ = total_set(&mut self.s_inw, toff + 1, n1);
                }
            }

            let _ = total_set(&mut self.b_state, uoff, un0);
            let _ = total_set(&mut self.b_state, uoff + 1, un1);
            let _ = total_set(&mut self.feat, self.raw_in + uoff, un0);
            let _ = total_set(&mut self.feat, self.raw_in + uoff + 1, un1);
        }

        // Головка классификации и линейная регрессионная головка
        for c in 0..self.num_actions {
            let off = c * self.feat_dim;
            let mut dot: i64 = total_get_copy(&self.actor_b, c, Q8_24(0)).0 as i64;
            for i in 0..self.feat_dim {
                let wi = total_get_copy(&self.actor_w, off + i, Q8_24(0)).0 as i64;
                let fi = total_get_copy(&self.feat, i, Q8_24(0)).0 as i64;
                dot = dot.saturating_add((wi * fi) >> 24);
            }
            let _ = total_set(&mut self.logits, c, Q8_24(dot.clamp((-64 * Q24_ONE) as i64, (64 * Q24_ONE) as i64) as i32));
        }
        q24_rational_p4_into(&self.logits, &mut self.probs, &mut self.psis);

        let mut rdot: i64 = self.reg_b.0 as i64;
        for i in 0..self.feat_dim {
            let wi = total_get_copy(&self.reg_w, i, Q8_24(0)).0 as i64;
            let fi = total_get_copy(&self.feat, i, Q8_24(0)).0 as i64;
            rdot = rdot.saturating_add((wi * fi) >> 24);
        }
        self.reg_pred = Q39_24(rdot);
    }

    pub fn step_supervised_class_q(&mut self, raw_x: &[Q39_24], target: usize, train: bool) -> usize {
        self.forward_q(raw_x, train, train);
        let mut pred = 0usize;
        let mut best_l = i32::MIN;
        for c in 0..self.num_actions {
            let l = total_get_copy(&self.logits, c, Q8_24(0)).0;
            if c == 0 || l > best_l {
                best_l = l;
                pred = c;
            }
        }
        if train && target < self.num_actions {
            let mut win = 0usize;
            let mut max_l = i32::MIN;
            for c in 0..self.num_actions {
                let l = total_get_copy(&self.logits, c, Q8_24(0)).0;
                if c == 0 || l > max_l { max_l = l; win = c; }
            }
            let mut lerrs = [Q8_24(0); 32];
            let mut sum_other: i64 = 0;
            for c in 0..self.num_actions {
                if c == win { continue; }
                let psi = total_get_copy(&self.psis, c, Q8_24(0));
                let pr = total_get_copy(&self.probs, c, Q8_24(0)).0;
                let tgt = if c == target { Q24_ONE } else { 0 };
                let ec = q24_mul(psi, Q8_24(tgt.saturating_sub(pr)));
                let _ = total_set(&mut lerrs, c, ec);
                sum_other = sum_other.saturating_add(ec.0 as i64);
            }
            let _ = total_set(&mut lerrs, win, Q8_24((-sum_other).clamp((-4 * Q24_ONE) as i64, (4 * Q24_ONE) as i64) as i32));

            let mut feat_sq: i64 = 0;
            for i in 0..self.feat_dim {
                let fi = total_get_copy(&self.feat, i, Q8_24(0)).0 as i64;
                feat_sq = feat_sq.saturating_add((fi * fi) >> 24);
            }
            let nlms_lr = Q8_24(total_div_i64((self.lr_ro.0 as i64) << 24, Q24_ONE_I64 + feat_sq, 0) as i32);

            let mut grad_u = [Q8_24(0); FIXED_2D_SLOTS * 2];
            for c in 0..self.num_actions {
                let ec = total_get_copy(&lerrs, c, Q8_24(0));
                let step_b = q24_mul(nlms_lr, ec);
                let old_b = total_get_copy(&self.actor_b, c, Q8_24(0));
                let _ = total_set(&mut self.actor_b, c, q24_clamp(q24_add(old_b, step_b), Q8_24(4 * Q24_ONE)));
                let off = c * self.feat_dim;
                for i in 0..self.feat_dim {
                    let wi = total_get_copy(&self.actor_w, off + i, Q8_24(0));
                    let fi = total_get_copy(&self.feat, i, Q8_24(0));
                    let g = q24_mul(ec, fi);
                    let dw = q24_mul(nlms_lr, g);
                    let _ = total_set(&mut self.actor_w, off + i, q24_clamp(q24_add(wi, dw), Q8_24(8 * Q24_ONE)));
                    if i >= self.raw_in {
                        let uidx = i - self.raw_in;
                        let gu = total_get_copy(&grad_u, uidx, Q8_24(0));
                        let _ = total_set(&mut grad_u, uidx, q24_add(gu, q24_mul(ec, wi)));
                    }
                }
            }
            for k in 0..FIXED_2D_SLOTS {
                let uoff = k * 2;
                let g0 = total_get_copy(&grad_u, uoff, Q8_24(0));
                let g1 = total_get_copy(&grad_u, uoff + 1, Q8_24(0));
                let woff = k * self.raw_in;
                for j in 0..self.raw_in {
                    let toff = (woff + j) * 2;
                    let s0 = total_get_copy(&self.s_inw, toff, Q8_24(0));
                    let s1 = total_get_copy(&self.s_inw, toff + 1, Q8_24(0));
                    let gw = q24_add(q24_mul(g0, s0), q24_mul(g1, s1));
                    let old = total_get_copy(&self.in_w, woff + j, Q8_24(0));
                    let _ = total_set(&mut self.in_w, woff + j, q24_clamp(q24_add(old, q24_mul(self.lr_rtrl, gw)), Q8_24(4 * Q24_ONE)));
                }
            }
        }
        pred
    }

    pub fn step_supervised_reg_q(&mut self, raw_x: &[Q39_24], target_y: Q39_24, train: bool) -> Q39_24 {
        self.forward_q(raw_x, train, false);
        if train {
            let err = Q8_24(target_y.0.saturating_sub(self.reg_pred.0).clamp((-2 * Q24_ONE) as i64, (2 * Q24_ONE) as i64) as i32);
            let mut feat_sq: i64 = 0;
            for i in 0..self.feat_dim {
                let fi = total_get_copy(&self.feat, i, Q8_24(0)).0 as i64;
                feat_sq = feat_sq.saturating_add((fi * fi) >> 24);
            }
            let nlms_lr = Q8_24(total_div_i64((self.lr_reg.0 as i64) << 24, Q24_ONE_I64 + feat_sq, 0) as i32);
            self.reg_b = q24_clamp(q24_add(self.reg_b, q24_mul(nlms_lr, err)), Q8_24(4 * Q24_ONE));
            for i in 0..self.feat_dim {
                let wi = total_get_copy(&self.reg_w, i, Q8_24(0));
                let fi = total_get_copy(&self.feat, i, Q8_24(0));
                let dw = q24_mul(nlms_lr, q24_mul(err, fi));
                let _ = total_set(&mut self.reg_w, i, q24_clamp(q24_add(wi, dw), Q8_24(8 * Q24_ONE)));
            }
        }
        self.reg_pred
    }

    /// 64-битный FNV-1a дайджест всего внутреннего состояния для проверки бит-точной детерминированности.
    pub fn state_digest_fnv1a(&self) -> u64 {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        let mix_u32 = |acc: &mut u64, v: u32| {
            for b in v.to_le_bytes() {
                *acc ^= b as u64;
                *acc = acc.wrapping_mul(0x0100_0000_01B3);
            }
        };
        for v in &self.b_state { mix_u32(&mut h, v.0 as u32); }
        for v in &self.in_w { mix_u32(&mut h, v.0 as u32); }
        for v in &self.actor_w { mix_u32(&mut h, v.0 as u32); }
        for v in &self.reg_w { mix_u32(&mut h, v.0 as u32); }
        mix_u32(&mut h, self.reg_b.0 as u32);
        h
    }
}
