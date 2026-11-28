// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Адаптивный ортогональный RTRL-контур `AdaptiveOrthogonalRtrl` (`O(d) ↪ SO(d+1)`, Suite D, v0.5.0):
//! - Эволюция полного многоколончатого ортонормированного репера (`ADAPTIVE_MAX_COLS = 5`: 4 пространственных
//!   зонда `e_0..e_3` + 1 зонд чётности `e_6`), устраняющая коллизии стабилизатора `SO(d-1)` при `d >= 3`;
//! - Вложение в фиксированное объемлющее пространство `SO(D_MAX + 1)` = `SO(7)` (`D_MAX = 6`):
//!   активные оси `0..active_dim`, фиксированная ось чётности `PARITY_AXIS = 6`;
//! - Изометрический апгрейд `d -> d + 1` (`upgrade_dimension_isometric`): < 1e-6 изменения вероятностей / < 1e-5 логитов
//!   при нулевом забывании (новые столбцы стартуют с нулевыми прототипами и `±1e-9` посевом новых плоскостей);
//! - Строго унитарный поворот чётности `R_v(p)^2` в `(v, PARITY_AXIS)` с замедленным `lr_par = 0.35 * lr_rot`,
//!   демпфированным RMSProp (`damp = 0.04`), сферическим отталкиванием ближайшего соседа (`mu = 0.55`) и
//!   2-масштабным блоковым детектором плато без оракулов (`fast_ema` vs `slow_ema` + 5-sigma Z^2).

use axiom_zero::{
    axiom_zero, total_div_f32, total_div_unit_shifted_f32, total_get_copy, total_rem_usize,
    total_set,
};

use crate::algebra::fill_p4_logprob_grads;
use crate::math::{inv_sqrt_newton_2step_f32, rational_probs_p4_into, splitmix64};

pub const ADAPTIVE_D_MAX: usize = 6;
pub const ADAPTIVE_MAX_FRAMES: usize = 8;
pub const ADAPTIVE_STATE_DIM: usize = 7; // 6 пространственных осей + 1 фиксированная ось чётности
pub const ADAPTIVE_PARITY_AXIS: usize = 6;
pub const ADAPTIVE_MAX_COLS: usize = 5; // 4 пространственных столбца + 1 столбец чётности
pub const ADAPTIVE_PARITY_COL: usize = 4;
pub const ADAPTIVE_MAX_PLANES: usize = 15; // 6 * 5 / 2
pub const ADAPTIVE_PARAMS_PER_CH: usize = ADAPTIVE_MAX_PLANES + ADAPTIVE_D_MAX + 1; // 15 planes + 6 v + 1 p_rot = 22
pub const ADAPTIVE_COL_LOGIT_SCALE: f32 = 0.25; // Фиксированный масштаб: добавление нулевого столбца даёт ровно 0 смещения

#[inline(always)]
pub const fn planes_for_dim(d: usize) -> usize {
    if d < 2 {
        0
    } else {
        let clamped = if d > ADAPTIVE_D_MAX { ADAPTIVE_D_MAX } else { d };
        (clamped * (clamped - 1)) >> 1
    }
}

#[inline(always)]
pub const fn is_col_active(col: usize, active_dim: usize) -> bool {
    let sp = if active_dim < 4 { active_dim } else { 4 };
    col < sp || col == ADAPTIVE_PARITY_COL
}

/// Канонический индекс `(p, q)` (сначала внешнее `q = 1..D_MAX`, затем `p = 0..q`),
/// чтобы при апгрейде `d -> d + 1` индексы старых плоскостей не сдвигались!
#[inline(always)]
pub const fn plane_pair_at(k: usize) -> (usize, usize) {
    match k {
        0 => (0, 1),   // d >= 2: 1 плоскость
        1 => (0, 2),   // d >= 3: +2 (итого 3)
        2 => (1, 2),
        3 => (0, 3),   // d >= 4: +3 (итого 6)
        4 => (1, 3),
        5 => (2, 3),
        6 => (0, 4),   // d >= 5: +4 (итого 10)
        7 => (1, 4),
        8 => (2, 4),
        9 => (3, 4),
        10 => (0, 5),  // d = 6: +5 (итого 15)
        11 => (1, 5),
        12 => (2, 5),
        13 => (3, 5),
        _ => (4, 5),
    }
}

#[derive(Clone, Debug)]
pub struct AdaptiveOrthogonalRtrl {
    pub frames: usize,
    pub num_inputs: usize,
    pub num_classes: usize,
    pub active_dim: usize,
    pub max_dim: usize,

    // [frame][col]
    pub state: Vec<[[f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_COLS]>,
    pub prev_state: Vec<[[f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_COLS]>,

    // [frame * num_inputs + g]
    pub planes: Vec<[f32; ADAPTIVE_MAX_PLANES]>,
    pub h_vec: Vec<[f32; ADAPTIVE_D_MAX]>,
    pub betas: Vec<f32>, // параметр R(p)^2: 0.0 -> +I (+1 det), 0.4142 -> водораздел, 1.0 -> -I_v (-1 det)

    // Точные онлайн RTRL-чувствительности: [((frame * ADAPTIVE_MAX_COLS + col) * num_inputs + g) * ADAPTIVE_PARAMS_PER_CH + p_idx]
    pub sens: Vec<[f32; ADAPTIVE_STATE_DIM]>,
    pub m_theta: Vec<f32>,   // Моментум первого порядка (P5, аналог NonAbelianO4Rtrl)
    pub rms_theta: Vec<f32>, // [frame * num_inputs * ADAPTIVE_PARAMS_PER_CH]

    // Многоколончатая сферическая метрика прототипов: [(frame * num_classes + c) * ADAPTIVE_MAX_COLS + col]
    pub proto: Vec<[f32; ADAPTIVE_STATE_DIM]>,
    pub proto_visits: Vec<f32>, // [frame * num_classes]
    pub col_visit_offset: [f32; ADAPTIVE_MAX_COLS], // Смещение счётчика посещений для свежеоткрытого столбца
    pub frame_acc_ema: Vec<f32>,

    pub proto_scale: f32,
    pub proto_center_beta: f32, // Задача-независимое притяжение к центру прототипа (сжатие ошибки соотношений)
    pub lambda_par: f32,
    pub comm_w: f32,
    pub solo_w: f32,
    pub proto_rep_mu: f32,
    pub proto_rep_lr: f32,

    // Рабочие буферы (0 аллокаций на горячем пути)
    pub frame_logits: Vec<f32>,
    pub frame_probs: Vec<f32>,
    pub frame_psi: Vec<f32>,
    pub lerrs: Vec<f32>,
    pub comb_logits: Vec<f32>,

    // 2-масштабная блоковая эскалация без оракулов (`fast_ema`/`slow_ema` + 5-sigma Z^2 + утилизация оболочки)
    pub class_counts: Vec<f32>,
    pub total_class_obs: f32,
    pub seen_classes: usize,
    pub fast_ema: f32,
    pub slow_ema: f32,
    pub ultra_slow_ema: f32,
    pub prev_block_acc: f32,
    pub block_hits: usize,
    pub block_total: usize,
    pub eps_in_dim: usize,
    pub total_eps: usize,
    pub win_hits: usize,
    pub win_total: usize,
    pub prev_active_dim: usize,
    pub prev_active_planes: usize,
    pub plateau_streak: usize,
    pub upgrade_seed: u64,
}

#[axiom_zero]
#[inline]
fn parity_rot_params(p: f32) -> (f32, f32, f32) {
    let p2 = p * p;
    let d1 = total_div_unit_shifted_f32(1.0, p2);
    let c1 = (1.0 - p2) * d1;
    let s1 = (2.0 * p) * d1;
    let c_p = c1 * c1 - s1 * s1;
    let s_p = 2.0 * c1 * s1;
    (c_p, s_p, d1)
}

#[axiom_zero]
fn vec_norm_active(v: &[f32; ADAPTIVE_STATE_DIM], active_dim: usize) -> f32 {
    let mut s = 0.0_f32;
    for i in 0..ADAPTIVE_D_MAX {
        if i < active_dim {
            let x = total_get_copy(v, i, 0.0);
            s += x * x;
        }
    }
    let xp = total_get_copy(v, ADAPTIVE_PARITY_AXIS, 0.0);
    s += xp * xp;
    s
}

#[axiom_zero]
fn normalize_state_active_nonzero(v: &mut [f32; ADAPTIVE_STATE_DIM], active_dim: usize) {
    for i in active_dim..ADAPTIVE_D_MAX {
        let _ = total_set(v, i, 0.0);
    }
    let s2 = vec_norm_active(v, active_dim);
    if s2 <= 1e-18 {
        return;
    }
    let inv = inv_sqrt_newton_2step_f32(s2);
    for i in 0..ADAPTIVE_D_MAX {
        if i < active_dim {
            let cur = total_get_copy(v, i, 0.0);
            let _ = total_set(v, i, cur * inv);
        }
    }
    let cp = total_get_copy(v, ADAPTIVE_PARITY_AXIS, 0.0);
    let _ = total_set(v, ADAPTIVE_PARITY_AXIS, cp * inv);
}

#[axiom_zero]
fn normalized_hv_active(hv: &[f32; ADAPTIVE_D_MAX], active_dim: usize) -> [f32; ADAPTIVE_D_MAX] {
    let mut out = [0.0_f32; ADAPTIVE_D_MAX];
    let mut s2 = 0.0_f32;
    for i in 0..ADAPTIVE_D_MAX {
        if i < active_dim {
            let x = total_get_copy(hv, i, 0.0);
            s2 += x * x;
            let _ = total_set(&mut out, i, x);
        }
    }
    if s2 > 1e-18 {
        let inv = inv_sqrt_newton_2step_f32(s2);
        for i in 0..ADAPTIVE_D_MAX {
            if i < active_dim {
                let x = total_get_copy(&out, i, 0.0);
                let _ = total_set(&mut out, i, x * inv);
            }
        }
    } else {
        let _ = total_set(&mut out, 0, 1.0);
    }
    out
}

/// Предвычисленный ортогональный оператор шага `Q_{g,m} = P_{g,m} * G_{n_pl-1} * ... * G_0 in SO(7)`
/// и кэш тригонометрических коэффициентов Кэли `(cs, ss, dcs, dss)` (Lean 4: `step133`).
/// Устраняет ~6 585 повторных делений `1 / (1 + t^2)` на фрейм за шаг без потери точности.
#[derive(Clone, Copy)]
struct GivensParityOp {
    q_mat: [[f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_STATE_DIM], // q_mat[row][col]
    cs: [f32; ADAPTIVE_MAX_PLANES],
    ss: [f32; ADAPTIVE_MAX_PLANES],
    dcs: [f32; ADAPTIVE_MAX_PLANES],
    dss: [f32; ADAPTIVE_MAX_PLANES],
}

#[axiom_zero]
impl GivensParityOp {
    /// Вычисляет только кэш тригонометрических коэффициентов Кэли `(cs, ss, dcs, dss)` (0% `q_mat`).
    /// Оптимально для read-only инференса и `step_forward` (<= 5 столбцов на фрейм).
    #[inline]
    fn build_trig(planes: &[f32; ADAPTIVE_MAX_PLANES], n_pl: usize, need_derivs: bool) -> Self {
        let mut cs = [1.0_f32; ADAPTIVE_MAX_PLANES];
        let mut ss = [0.0_f32; ADAPTIVE_MAX_PLANES];
        let mut dcs = [0.0_f32; ADAPTIVE_MAX_PLANES];
        let mut dss = [2.0_f32; ADAPTIVE_MAX_PLANES];
        for k in 0..ADAPTIVE_MAX_PLANES {
            if k >= n_pl {
                break;
            }
            let t = total_get_copy(planes, k, 0.0);
            let t2 = t * t;
            let den = total_div_unit_shifted_f32(1.0, t2);
            let _ = total_set(&mut cs, k, (1.0 - t2) * den);
            let _ = total_set(&mut ss, k, (2.0 * t) * den);
            if need_derivs {
                let den2 = den * den;
                let _ = total_set(&mut dcs, k, -4.0 * t * den2);
                let _ = total_set(&mut dss, k, 2.0 * (1.0 - t2) * den2);
            }
        }
        Self {
            q_mat: [[0.0_f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_STATE_DIM],
            cs,
            ss,
            dcs,
            dss,
        }
    }

    /// Достраивает предвычисленную матрицу `Q_{g,m} in SO(7)` для массовой транспортировки RTRL-трасс (`step133`).
    #[inline]
    fn build_q_mat(
        &mut self,
        n_pl: usize,
        vh: &[f32; ADAPTIVE_D_MAX],
        c_p: f32,
        s_p: f32,
        active_dim: usize,
    ) {
        let mut q_mat = [[0.0_f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_STATE_DIM];
        for j in 0..ADAPTIVE_STATE_DIM {
            if j >= active_dim && j != ADAPTIVE_PARITY_AXIS {
                continue;
            }
            let mut col = [0.0_f32; ADAPTIVE_STATE_DIM];
            let _ = total_set(&mut col, j, 1.0);
            col = self.apply_givens_cached_from(&col, 0, n_pl);
            let z = apply_unitary_parity(&col, vh, c_p, s_p, active_dim);
            let [z0, z1, z2, z3, z4, z5, zp] = z;
            if let Some(r0) = q_mat.get_mut(0) { let _ = total_set(r0, j, z0); }
            if let Some(r1) = q_mat.get_mut(1) { let _ = total_set(r1, j, z1); }
            if active_dim > 2 {
                if let Some(r2) = q_mat.get_mut(2) { let _ = total_set(r2, j, z2); }
            }
            if active_dim > 3 {
                if let Some(r3) = q_mat.get_mut(3) { let _ = total_set(r3, j, z3); }
            }
            if active_dim > 4 {
                if let Some(r4) = q_mat.get_mut(4) { let _ = total_set(r4, j, z4); }
            }
            if active_dim > 5 {
                if let Some(r5) = q_mat.get_mut(5) { let _ = total_set(r5, j, z5); }
            }
            if let Some(rp) = q_mat.get_mut(ADAPTIVE_PARITY_AXIS) { let _ = total_set(rp, j, zp); }
        }
        self.q_mat = q_mat;
    }

    /// Быстрое бесцикловое умножение `Q_{g,m} * x` через деструктуризацию `[f32; 7]` (0 bounds-checks).
    #[inline(always)]
    fn apply_q(&self, x: &[f32; ADAPTIVE_STATE_DIM], active_dim: usize) -> [f32; ADAPTIVE_STATE_DIM] {
        let [x0, x1, x2, x3, x4, x5, xp] = *x;
        let [r0, r1, r2, r3, r4, r5, rp] = self.q_mat;
        let dot7 = |r: [f32; ADAPTIVE_STATE_DIM]| -> f32 {
            let [a0, a1, a2, a3, a4, a5, ap] = r;
            a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + ap * xp
        };
        [
            dot7(r0),
            dot7(r1),
            if active_dim > 2 { dot7(r2) } else { 0.0 },
            if active_dim > 3 { dot7(r3) } else { 0.0 },
            if active_dim > 4 { dot7(r4) } else { 0.0 },
            if active_dim > 5 { dot7(r5) } else { 0.0 },
            dot7(rp),
        ]
    }

    #[inline(always)]
    fn apply_givens_cached_from(
        &self,
        x: &[f32; ADAPTIVE_STATE_DIM],
        start_k: usize,
        n_pl: usize,
    ) -> [f32; ADAPTIVE_STATE_DIM] {
        let mut y = *x;
        for k in start_k..ADAPTIVE_MAX_PLANES {
            if k >= n_pl {
                break;
            }
            let (p, q) = plane_pair_at(k);
            let c = total_get_copy(&self.cs, k, 1.0);
            let s = total_get_copy(&self.ss, k, 0.0);
            let yp = total_get_copy(&y, p, 0.0);
            let yq = total_get_copy(&y, q, 0.0);
            let _ = total_set(&mut y, p, c * yp - s * yq);
            let _ = total_set(&mut y, q, s * yp + c * yq);
        }
        y
    }
}

#[axiom_zero]
fn apply_unitary_parity(
    y: &[f32; ADAPTIVE_STATE_DIM],
    vh: &[f32; ADAPTIVE_D_MAX],
    c_p: f32,
    s_p: f32,
    active_dim: usize,
) -> [f32; ADAPTIVE_STATE_DIM] {
    let mut y_v = 0.0_f32;
    for i in 0..ADAPTIVE_D_MAX {
        if i < active_dim {
            let vi = total_get_copy(vh, i, 0.0);
            let yi = total_get_copy(y, i, 0.0);
            y_v += vi * yi;
        }
    }
    let y_p = total_get_copy(y, ADAPTIVE_PARITY_AXIS, 0.0);
    let z_v = c_p * y_v - s_p * y_p;
    let z_p = s_p * y_v + c_p * y_p;
    let delta_v = z_v - y_v;

    let mut z = *y;
    for i in 0..ADAPTIVE_D_MAX {
        if i < active_dim {
            let yi = total_get_copy(y, i, 0.0);
            let vi = total_get_copy(vh, i, 0.0);
            let _ = total_set(&mut z, i, yi + delta_v * vi);
        } else {
            let _ = total_set(&mut z, i, 0.0);
        }
    }
    let _ = total_set(&mut z, ADAPTIVE_PARITY_AXIS, z_p);
    z
}

#[axiom_zero]
fn init_basis_columns(active_dim: usize) -> [[f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_COLS] {
    let mut cols = [[0.0_f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_COLS];
    let sp = if active_dim < 4 { active_dim } else { 4 };
    for c in 0..sp {
        let mut v = [0.0_f32; ADAPTIVE_STATE_DIM];
        let _ = total_set(&mut v, c, 1.0);
        let _ = total_set(&mut cols, c, v);
    }
    let mut vp = [0.0_f32; ADAPTIVE_STATE_DIM];
    let _ = total_set(&mut vp, ADAPTIVE_PARITY_AXIS, 1.0);
    let _ = total_set(&mut cols, ADAPTIVE_PARITY_COL, vp);
    cols
}

#[axiom_zero]
#[inline(always)]
fn dot_active(a: &[f32; ADAPTIVE_STATE_DIM], b: &[f32; ADAPTIVE_STATE_DIM], active_dim: usize) -> f32 {
    let [a0, a1, a2, a3, a4, a5, ap] = *a;
    let [b0, b1, b2, b3, b4, b5, bp] = *b;
    let mut dot = a0 * b0 + a1 * b1 + ap * bp;
    if active_dim > 2 {
        dot += a2 * b2;
    }
    if active_dim > 3 {
        dot += a3 * b3;
    }
    if active_dim > 4 {
        dot += a4 * b4;
    }
    if active_dim > 5 {
        dot += a5 * b5;
    }
    dot
}

#[axiom_zero]
impl AdaptiveOrthogonalRtrl {
    pub fn new(frames: usize, num_inputs: usize, num_classes: usize, seed: u64) -> Self {
        let f = frames.clamp(1, 8);
        let ni = num_inputs.clamp(1, 32);
        let nc = num_classes.clamp(2, 2048);
        let d0 = 2usize;

        let mut rng = splitmix64(seed ^ 0xA0AD_0007_0005);
        let mut planes = Vec::with_capacity(f * ni);
        let mut h_vec = Vec::with_capacity(f * ni);
        let mut betas = Vec::with_capacity(f * ni);

        for fm in 0..f {
            for _g in 0..ni {
                let mut pl = [0.0_f32; ADAPTIVE_MAX_PLANES];
                rng = splitmix64(rng);
                let t01 = (((rng >> 40) & 0xFFFF) as f32 * 1.5258789e-5_f32 - 0.5) * (1.2 + 0.3 * fm as f32);
                let _ = total_set(&mut pl, 0, t01);
                planes.push(pl);

                let mut hv = [0.0_f32; ADAPTIVE_D_MAX];
                for i in 0..d0 {
                    rng = splitmix64(rng);
                    let r = ((rng >> 40) & 0xFFFF) as f32 * 3.0517578e-5_f32 - 1.0;
                    let _ = total_set(&mut hv, i, r);
                }
                hv = normalized_hv_active(&hv, d0);
                h_vec.push(hv);

                // Точный симметричный водораздел p = sqrt(2) - 1 ≈ 0.41421356 (0 случайного сдвига)
                betas.push(0.41421356_f32);
            }
        }

        // Начальные прототипы: активны столбцы 0, 1 и PARITY_COL (4) в подпространстве (0, 1, PARITY_AXIS=6)
        let mut proto = vec![[0.0_f32; ADAPTIVE_STATE_DIM]; f * nc * ADAPTIVE_MAX_COLS];
        for fm in 0..f {
            for c in 0..nc {
                let base = (fm * nc + c) * ADAPTIVE_MAX_COLS;
                for col in 0..ADAPTIVE_MAX_COLS {
                    if !is_col_active(col, d0) {
                        continue;
                    }
                    let mut p = [0.0_f32; ADAPTIVE_STATE_DIM];
                    rng = splitmix64(rng);
                    let a = ((rng >> 40) & 0xFFFF) as f32 * 3.0517578e-5_f32 - 1.0;
                    rng = splitmix64(rng);
                    let b = ((rng >> 40) & 0xFFFF) as f32 * 3.0517578e-5_f32 - 1.0;
                    rng = splitmix64(rng);
                    let z = ((rng >> 40) & 0xFFFF) as f32 * 3.0517578e-5_f32 - 1.0;
                    let _ = total_set(&mut p, 0, a);
                    let _ = total_set(&mut p, 1, b);
                    let _ = total_set(&mut p, ADAPTIVE_PARITY_AXIS, z);
                    normalize_state_active_nonzero(&mut p, d0);
                    let _ = total_set(&mut proto, base + col, p);
                }
            }
        }

        let init_cols = init_basis_columns(d0);

        Self {
            frames: f,
            num_inputs: ni,
            num_classes: nc,
            active_dim: d0,
            max_dim: ADAPTIVE_D_MAX,
            state: vec![init_cols; f],
            prev_state: vec![init_cols; f],
            planes,
            h_vec,
            betas,
            sens: vec![[0.0_f32; ADAPTIVE_STATE_DIM]; f * ADAPTIVE_MAX_COLS * ni * ADAPTIVE_PARAMS_PER_CH],
            m_theta: vec![0.0_f32; f * ni * ADAPTIVE_PARAMS_PER_CH],
            rms_theta: vec![0.04_f32; f * ni * ADAPTIVE_PARAMS_PER_CH],
            proto,
            proto_visits: vec![0.0_f32; f * nc],
            col_visit_offset: [0.0_f32; ADAPTIVE_MAX_COLS],
            frame_acc_ema: vec![0.05_f32; f],
            proto_scale: 12.0,
            proto_center_beta: 1.2,
            lambda_par: 0.8,
            comm_w: 0.08,
            solo_w: 0.55,
            proto_rep_mu: 0.55,
            proto_rep_lr: 0.06,
            frame_logits: vec![0.0_f32; nc],
            frame_probs: vec![0.0_f32; nc],
            frame_psi: vec![0.0_f32; nc],
            lerrs: vec![0.0_f32; nc],
            comb_logits: vec![0.0_f32; nc],
            class_counts: vec![0.0_f32; nc],
            total_class_obs: 0.0,
            seen_classes: 0,
            fast_ema: 0.05,
            slow_ema: 0.05,
            ultra_slow_ema: 0.05,
            prev_block_acc: 0.0,
            block_hits: 0,
            block_total: 0,
            eps_in_dim: 0,
            total_eps: 0,
            win_hits: 0,
            win_total: 0,
            prev_active_dim: 0,
            prev_active_planes: 0,
            plateau_streak: 0,
            upgrade_seed: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        core::mem::size_of::<Self>()
            + self.state.len() * core::mem::size_of::<[[f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_COLS]>()
            + self.prev_state.len() * core::mem::size_of::<[[f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_COLS]>()
            + self.planes.len() * core::mem::size_of::<[f32; ADAPTIVE_MAX_PLANES]>()
            + self.h_vec.len() * core::mem::size_of::<[f32; ADAPTIVE_D_MAX]>()
            + self.betas.len() * 4
            + self.sens.len() * core::mem::size_of::<[f32; ADAPTIVE_STATE_DIM]>()
            + self.m_theta.len() * 4
            + self.rms_theta.len() * 4
            + self.proto.len() * core::mem::size_of::<[f32; ADAPTIVE_STATE_DIM]>()
            + self.proto_visits.len() * 4
            + self.frame_acc_ema.len() * 4
            + (self.frame_logits.len()
                + self.frame_probs.len()
                + self.frame_psi.len()
                + self.lerrs.len()
                + self.comb_logits.len()
                + self.class_counts.len())
                * 4
    }

    pub fn reset(&mut self) {
        let init_cols = init_basis_columns(self.active_dim);
        for s in self.state.iter_mut() {
            *s = init_cols;
        }
        for s in self.prev_state.iter_mut() {
            *s = init_cols;
        }
        self.sens.fill([0.0_f32; ADAPTIVE_STATE_DIM]);
    }

    fn channel_index_from_input(&self, x: &[f32]) -> usize {
        let mut best_g = 0usize;
        let mut best_v = -1.0_f32;
        for g in 0..self.num_inputs {
            let v = total_get_copy(x, g, 0.0);
            if g == 0 || v > best_v {
                best_v = v;
                best_g = g;
            }
        }
        best_g
    }

    pub fn step_forward(&mut self, x: &[f32]) {
        let g = self.channel_index_from_input(x);
        let d = self.active_dim;
        let n_pl = planes_for_dim(d);
        let zero_cols = [[0.0_f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_COLS];
        for m in 0..self.frames {
            let ch = m * self.num_inputs + g;
            let u_prev_cols = total_get_copy(&self.state, m, zero_cols);
            let _ = total_set(&mut self.prev_state, m, u_prev_cols);
            let pl = total_get_copy(&self.planes, ch, [0.0_f32; ADAPTIVE_MAX_PLANES]);
            let hv_raw = total_get_copy(&self.h_vec, ch, [0.0_f32; ADAPTIVE_D_MAX]);
            let vh = normalized_hv_active(&hv_raw, d);
            let p_rot = total_get_copy(&self.betas, ch, 0.41421356);
            let (c_p, s_p, _) = parity_rot_params(p_rot);
            let op = GivensParityOp::build_trig(&pl, n_pl, false);

            let mut next_cols = zero_cols;
            for col in 0..ADAPTIVE_MAX_COLS {
                if !is_col_active(col, d) {
                    continue;
                }
                let u_prev = total_get_copy(&u_prev_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                let y = op.apply_givens_cached_from(&u_prev, 0, n_pl);
                let mut z = apply_unitary_parity(&y, &vh, c_p, s_p, d);
                normalize_state_active_nonzero(&mut z, d);
                let _ = total_set(&mut next_cols, col, z);
            }
            let _ = total_set(&mut self.state, m, next_cols);
        }
    }

    pub fn compute_combined_logits_into(&self, out_logits: &mut [f32]) {
        out_logits.fill(0.0);
        let d = self.active_dim;
        let zero_cols = [[0.0_f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_COLS];
        let eff_scale = self.proto_scale * ADAPTIVE_COL_LOGIT_SCALE;
        for m in 0..self.frames {
            let u_cols = total_get_copy(&self.state, m, zero_cols);
            let fa = total_get_copy(&self.frame_acc_ema, m, 0.05);
            let wm = 0.25 + fa * fa;
            let base_m = m * self.num_classes;
            for c in 0..self.num_classes {
                let p_off = (base_m + c) * ADAPTIVE_MAX_COLS;
                let mut sum_dot = 0.0_f32;
                for col in 0..ADAPTIVE_MAX_COLS {
                    if !is_col_active(col, d) {
                        continue;
                    }
                    let u = total_get_copy(&u_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                    let pr = total_get_copy(&self.proto, p_off + col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                    sum_dot += dot_active(&u, &pr, d);
                }
                let prev = total_get_copy(out_logits, c, 0.0);
                let _ = total_set(out_logits, c, prev + wm * eff_scale * sum_dot);
            }
        }
    }

    pub fn predict_k_ahead_readonly(&self, inputs: &[&[f32]]) -> usize {
        let d = self.active_dim;
        let n_pl = planes_for_dim(d);
        let init_cols = init_basis_columns(d);
        let zero_cols = [[0.0_f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_COLS];
        let n_frames = self.frames.min(ADAPTIVE_MAX_FRAMES);
        let mut cur = [init_cols; ADAPTIVE_MAX_FRAMES];

        for &x_slice in inputs {
            let g = self.channel_index_from_input(x_slice);
            for m in 0..n_frames {
                let ch = m * self.num_inputs + g;
                let u_prev_cols = total_get_copy(&cur, m, zero_cols);
                let pl = total_get_copy(&self.planes, ch, [0.0_f32; ADAPTIVE_MAX_PLANES]);
                let hv_raw = total_get_copy(&self.h_vec, ch, [0.0_f32; ADAPTIVE_D_MAX]);
                let vh = normalized_hv_active(&hv_raw, d);
                let p_rot = total_get_copy(&self.betas, ch, 0.41421356);
                let (c_p, s_p, _) = parity_rot_params(p_rot);
                let op = GivensParityOp::build_trig(&pl, n_pl, false);

                let mut next_cols = zero_cols;
                for col in 0..ADAPTIVE_MAX_COLS {
                    if !is_col_active(col, d) {
                        continue;
                    }
                    let u_prev = total_get_copy(&u_prev_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                    let y = op.apply_givens_cached_from(&u_prev, 0, n_pl);
                    let mut z = apply_unitary_parity(&y, &vh, c_p, s_p, d);
                    normalize_state_active_nonzero(&mut z, d);
                    let _ = total_set(&mut next_cols, col, z);
                }
                let _ = total_set(&mut cur, m, next_cols);
            }
        }

        let eff_scale = self.proto_scale * ADAPTIVE_COL_LOGIT_SCALE;
        let mut best_c = 0usize;
        let mut best_l = f32::NEG_INFINITY;
        for c in 0..self.num_classes {
            let mut sum_l = 0.0_f32;
            for m in 0..n_frames {
                let u_cols = total_get_copy(&cur, m, zero_cols);
                let fa = total_get_copy(&self.frame_acc_ema, m, 0.05);
                let ex = (fa - 0.30).max(0.0);
                let ex2 = ex * ex;
                let wm = 0.02 + ex2 * ex2;
                let p_off = (m * self.num_classes + c) * ADAPTIVE_MAX_COLS;
                let mut sum_dot = 0.0_f32;
                for col in 0..ADAPTIVE_MAX_COLS {
                    if !is_col_active(col, d) {
                        continue;
                    }
                    let u = total_get_copy(&u_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                    let pr = total_get_copy(&self.proto, p_off + col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                    sum_dot += dot_active(&u, &pr, d);
                }
                sum_l += wm * eff_scale * sum_dot;
            }
            if c == 0 || sum_l > best_l {
                best_l = sum_l;
                best_c = c;
            }
        }
        best_c
    }

    pub fn predict_current_readonly(&self) -> usize {
        let d = self.active_dim;
        let zero_cols = [[0.0_f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_COLS];
        let eff_scale = self.proto_scale * ADAPTIVE_COL_LOGIT_SCALE;
        let mut best_c = 0usize;
        let mut best_l = f32::NEG_INFINITY;
        for c in 0..self.num_classes {
            let mut sum_l = 0.0_f32;
            for m in 0..self.frames {
                let u_cols = total_get_copy(&self.state, m, zero_cols);
                let fa = total_get_copy(&self.frame_acc_ema, m, 0.05);
                let wm = 0.25 + fa * fa;
                let p_off = (m * self.num_classes + c) * ADAPTIVE_MAX_COLS;
                let mut sum_dot = 0.0_f32;
                for col in 0..ADAPTIVE_MAX_COLS {
                    if !is_col_active(col, d) {
                        continue;
                    }
                    let u = total_get_copy(&u_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                    let pr = total_get_copy(&self.proto, p_off + col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                    sum_dot += dot_active(&u, &pr, d);
                }
                sum_l += wm * eff_scale * sum_dot;
            }
            if c == 0 || sum_l > best_l {
                best_l = sum_l;
                best_c = c;
            }
        }
        best_c
    }

    /// Изометрический апгрейд `d -> d + 1` в `SO(7)`:
    /// оставляет старые столбцы и их прототипы нетронутыми (`u[old_d]=0`, `proto[old_d]=0`),
    /// новый столбец `old_d` получает `e_{old_d}`, а его прототипы остаются `[0; 7]` (ровно 0 смещения выхода!),
    /// новые плоскости `n_old..n_new` открываются микро-посевом `±1e-9` для выхода из седла (< 1e-6 дрейфа P).
    pub fn upgrade_dimension_isometric(&mut self) -> bool {
        if self.active_dim >= self.max_dim {
            return false;
        }
        let old_d = self.active_dim;
        let new_d = old_d + 1;
        let n_old = planes_for_dim(old_d);
        let n_new = planes_for_dim(new_d);
        let zero_cols = [[0.0_f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_COLS];

        // Смещение счётчика посещений для плавного размораживания прототипов в новом измерении (0 скачка логитов!)
        {
            let mut sum_vis = 0.0_f32;
            let mut cnt_vis = 0.0_f32;
            for idx in 0..self.proto_visits.len() {
                let v = total_get_copy(&self.proto_visits, idx, 0.0);
                if v > 0.0 {
                    sum_vis += v;
                    cnt_vis += 1.0;
                }
            }
            let mean_vis = if cnt_vis > 0.0 { total_div_f32(sum_vis, cnt_vis, 0.0) } else { 0.0 };
            for col in 0..ADAPTIVE_MAX_COLS {
                let _ = total_set(&mut self.col_visit_offset, col, mean_vis);
            }
        }

        for m in 0..self.frames {
            for g in 0..self.num_inputs {
                let ch = m * self.num_inputs + g;
                let mut pl = total_get_copy(&self.planes, ch, [0.0_f32; ADAPTIVE_MAX_PLANES]);
                for k in n_old..n_new {
                    self.upgrade_seed = splitmix64(self.upgrade_seed);
                    let sign = if (self.upgrade_seed & 1) == 0 { 1.0e-9_f32 } else { -1.0e-9_f32 };
                    let _ = total_set(&mut pl, k, sign);
                }
                let _ = total_set(&mut self.planes, ch, pl);

                let mut hv = total_get_copy(&self.h_vec, ch, [0.0_f32; ADAPTIVE_D_MAX]);
                self.upgrade_seed = splitmix64(self.upgrade_seed);
                let vsign = if (self.upgrade_seed & 1) == 0 { 1.0e-9_f32 } else { -1.0e-9_f32 };
                let _ = total_set(&mut hv, old_d, vsign);
                let _ = total_set(&mut self.h_vec, ch, hv);

                // P5: Новые плоскости стартуют со свежим демпфированным RMSProp = 0.04 и обнулённым моментумом
                let rms_base = ch * ADAPTIVE_PARAMS_PER_CH;
                for k in n_old..n_new {
                    let _ = total_set(&mut self.rms_theta, rms_base + k, 0.04);
                    let _ = total_set(&mut self.m_theta, rms_base + k, 0.0);
                }
                let _ = total_set(&mut self.rms_theta, rms_base + ADAPTIVE_MAX_PLANES + old_d, 0.04);
                let _ = total_set(&mut self.m_theta, rms_base + ADAPTIVE_MAX_PLANES + old_d, 0.0);
            }
            if old_d < 4 {
                let mut cols = total_get_copy(&self.state, m, zero_cols);
                let mut e_new = [0.0_f32; ADAPTIVE_STATE_DIM];
                let _ = total_set(&mut e_new, old_d, 1.0);
                let _ = total_set(&mut cols, old_d, e_new);
                let _ = total_set(&mut self.state, m, cols);
                let _ = total_set(&mut self.prev_state, m, cols);
            }
        }

        // Обнуляем чувствительности для всех активных столбцов по новым параметрам и для нового столбца полностью
        for m in 0..self.frames {
            for col in 0..ADAPTIVE_MAX_COLS {
                for g in 0..self.num_inputs {
                    let base = ((m * ADAPTIVE_MAX_COLS + col) * self.num_inputs + g) * ADAPTIVE_PARAMS_PER_CH;
                    if old_d < 4 && col == old_d {
                        for p in 0..ADAPTIVE_PARAMS_PER_CH {
                            let _ = total_set(&mut self.sens, base + p, [0.0_f32; ADAPTIVE_STATE_DIM]);
                        }
                    } else {
                        for k in n_old..n_new {
                            let _ = total_set(&mut self.sens, base + k, [0.0_f32; ADAPTIVE_STATE_DIM]);
                        }
                        let _ = total_set(
                            &mut self.sens,
                            base + ADAPTIVE_MAX_PLANES + old_d,
                            [0.0_f32; ADAPTIVE_STATE_DIM],
                        );
                    }
                }
            }
        }

        self.prev_active_dim = old_d;
        self.prev_active_planes = n_old;
        self.plateau_streak = 0;
        self.active_dim = new_d;
        self.eps_in_dim = 0;
        self.win_hits = 0;
        self.win_total = 0;
        self.prev_block_acc = self.slow_ema;
        self.block_hits = 0;
        self.block_total = 0;
        self.ultra_slow_ema = self.slow_ema;
        self.fast_ema = self.slow_ema;
        true
    }

    /// Проверяет, исчерпана ли геометрическая ёмкость вновь открытой оболочки плоскостей текущего `active_dim` (P5).
    /// Возвращает `(active_ratio, mean_shell_energy)` по плоскостям `planes_for_dim(d-1)..planes_for_dim(d)`.
    pub fn current_shell_utilization(&self) -> (f32, f32) {
        let d = self.active_dim;
        if d <= 2 {
            return (1.0, 1.0);
        }
        let n_prev = planes_for_dim(d - 1);
        let n_cur = planes_for_dim(d);
        let mut active_cnt = 0usize;
        let mut total_cnt = 0usize;
        let mut sum_sq = 0.0_f32;
        for ch in 0..self.planes.len() {
            let pl = total_get_copy(&self.planes, ch, [0.0_f32; ADAPTIVE_MAX_PLANES]);
            for k in n_prev..n_cur {
                let t = total_get_copy(&pl, k, 0.0);
                if t.abs() > 0.06 {
                    active_cnt = active_cnt.saturating_add(1);
                }
                sum_sq += t * t;
                total_cnt = total_cnt.saturating_add(1);
            }
        }
        if total_cnt == 0 {
            return (1.0, 1.0);
        }
        let inv_tot = total_div_f32(1.0, total_cnt as f32, 0.0);
        ((active_cnt as f32) * inv_tot, sum_sq * inv_tot)
    }

    pub fn observe_episode_hit(&mut self, hit: bool, target_class: usize) -> bool {
        self.total_eps = self.total_eps.saturating_add(1);
        self.eps_in_dim = self.eps_in_dim.saturating_add(1);
        if target_class < self.num_classes {
            let prev_c = total_get_copy(&self.class_counts, target_class, 0.0);
            if prev_c == 0.0 {
                self.seen_classes = self.seen_classes.saturating_add(1);
            }
            let _ = total_set(&mut self.class_counts, target_class, prev_c + 1.0);
            self.total_class_obs += 1.0;
        }

        let r = if hit { 1.0_f32 } else { 0.0_f32 };
        self.fast_ema += 0.05 * (r - self.fast_ema);
        self.slow_ema += 0.006 * (r - self.slow_ema);
        self.ultra_slow_ema += 0.0008 * (r - self.ultra_slow_ema);
        self.win_total = self.win_total.saturating_add(1);
        self.block_total = self.block_total.saturating_add(1);
        if hit {
            self.win_hits = self.win_hits.saturating_add(1);
            self.block_hits = self.block_hits.saturating_add(1);
        }

        if self.active_dim >= self.max_dim {
            return false;
        }

        let check_block = (3 * self.seen_classes.max(5)).clamp(200, 1000);
        if self.block_total < check_block {
            return false;
        }
        let cur_block = total_div_f32(self.block_hits as f32, self.block_total as f32, 0.0);

        let mut emp_chance = total_div_f32(1.0, self.seen_classes.max(2) as f32, 0.5);
        if self.total_class_obs > 20.0 {
            let inv_tot = total_div_f32(1.0, self.total_class_obs, 0.0);
            let mut sum_sq = 0.0_f32;
            for c in 0..self.num_classes {
                let cc = total_get_copy(&self.class_counts, c, 0.0);
                if cc > 0.0 {
                    let p = cc * inv_tot;
                    sum_sq += p * p;
                }
            }
            emp_chance = sum_sq.clamp(0.0005, 0.99);
        }

        // P5: Размерно-зависимый dwell + проверка утилизации текущей оболочки плоскостей
        let seen = self.seen_classes.max(5);
        let (min_dwell, max_block_acc, max_slow_ema, plateau_eps) = match self.active_dim {
            2 => ((5 * seen).clamp(450, 900), 0.88_f32, 0.89_f32, 0.015_f32),
            3 => ((14 * seen).clamp(1000, 1800), 0.86_f32, 0.87_f32, 0.012_f32),
            4 => ((36 * seen).clamp(2400, 4800), 0.84_f32, 0.85_f32, 0.010_f32),
            _ => ((36 * seen).clamp(3200, 6000), 0.80_f32, 0.82_f32, 0.010_f32),
        };
        let (active_ratio, mean_shell_e) = self.current_shell_utilization();
        let shell_utilized = self.active_dim <= 2 || (active_ratio >= 0.30 && mean_shell_e >= 0.010);

        let acc_win = total_div_f32(self.win_hits as f32, self.win_total as f32, 0.0);
        let margin = acc_win - emp_chance;
        let z2_ok = (margin * margin) * (self.win_total as f32) > 25.0 * emp_chance * (1.0 - emp_chance);

        let still_climbing = self.fast_ema > self.slow_ema + 0.015;
        let slow_creep_exit = self.active_dim == 2
            && self.eps_in_dim >= min_dwell.saturating_mul(2)
            && cur_block < 0.78
            && !still_climbing;
        let is_plateau_block = cur_block < max_block_acc
            && self.slow_ema < max_slow_ema
            && !still_climbing
            && (cur_block <= self.prev_block_acc + plateau_eps || slow_creep_exit)
            && ((self.fast_ema - self.slow_ema).abs() < 0.035 || slow_creep_exit)
            && cur_block > emp_chance * 1.6
            && (cur_block - emp_chance) > 0.004
            && z2_ok
            && shell_utilized;

        if self.eps_in_dim >= min_dwell && is_plateau_block {
            self.plateau_streak = self.plateau_streak.saturating_add(1);
        } else {
            self.plateau_streak = 0;
        }
        // В d = 2 требуем 2 подряд плато-блока (исключает преждевременный выход из O(2) на циклич. группах C_7 и др.)
        let need_streak = if self.active_dim == 2 { 2usize } else { 1usize };
        let should_upgrade = self.plateau_streak >= need_streak;

        self.block_hits = 0;
        self.block_total = 0;

        if should_upgrade {
            self.upgrade_dimension_isometric()
        } else {
            self.prev_block_acc = cur_block;
            false
        }
    }

    pub fn channel_betas_into(&self, g: usize, out: &mut [f32]) {
        let gin = g.min(self.num_inputs.saturating_sub(1));
        for m in 0..self.frames {
            let p_rot = total_get_copy(&self.betas, m * self.num_inputs + gin, 0.41421356);
            let (c_p, _, _) = parity_rot_params(p_rot);
            let _ = total_set(out, m, 1.0 - c_p);
        }
    }

    pub fn step_supervised(
        &mut self,
        x: &[f32],
        has_target: bool,
        target: usize,
        lr_rot: f32,
        lr_head: f32,
    ) -> usize {
        let g = self.channel_index_from_input(x);
        let d = self.active_dim;
        let n_pl = planes_for_dim(d);
        let zero_cols = [[0.0_f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_COLS];

        // 1. Прямой проход и точные RTRL-чувствительности по всем активным столбцам
        for m in 0..self.frames {
            let ch = m * self.num_inputs + g;
            let u_prev_cols = total_get_copy(&self.state, m, zero_cols);
            let _ = total_set(&mut self.prev_state, m, u_prev_cols);

            let pl = total_get_copy(&self.planes, ch, [0.0_f32; ADAPTIVE_MAX_PLANES]);
            let hv_raw = total_get_copy(&self.h_vec, ch, [0.0_f32; ADAPTIVE_D_MAX]);
            let vh = normalized_hv_active(&hv_raw, d);
            let p_rot = total_get_copy(&self.betas, ch, 0.41421356);
            let (c_p, s_p, d1_p) = parity_rot_params(p_rot);
            let mut op = GivensParityOp::build_trig(&pl, n_pl, has_target);
            if has_target {
                op.build_q_mat(n_pl, &vh, c_p, s_p, d);
            }

            let mut hv_norm2 = 0.0_f32;
            for i in 0..ADAPTIVE_D_MAX {
                if i < d {
                    let vi = total_get_copy(&hv_raw, i, 0.0);
                    hv_norm2 += vi * vi;
                }
            }
            let inv_hv = if hv_norm2 > 1e-18 { inv_sqrt_newton_2step_f32(hv_norm2) } else { 1.0 };

            let mut next_cols = zero_cols;
            for col in 0..ADAPTIVE_MAX_COLS {
                if !is_col_active(col, d) {
                    continue;
                }
                let u_prev = total_get_copy(&u_prev_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);

                let mut chain = [[0.0_f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_PLANES + 1];
                let _ = total_set(&mut chain, 0, u_prev);
                for k in 0..ADAPTIVE_MAX_PLANES {
                    let cur = total_get_copy(&chain, k, [0.0_f32; ADAPTIVE_STATE_DIM]);
                    if k < n_pl {
                        let (p, q) = plane_pair_at(k);
                        let c = total_get_copy(&op.cs, k, 1.0);
                        let s = total_get_copy(&op.ss, k, 0.0);
                        let mut nx = cur;
                        let xp = total_get_copy(&cur, p, 0.0);
                        let xq = total_get_copy(&cur, q, 0.0);
                        let _ = total_set(&mut nx, p, c * xp - s * xq);
                        let _ = total_set(&mut nx, q, s * xp + c * xq);
                        let _ = total_set(&mut chain, k + 1, nx);
                    } else {
                        let _ = total_set(&mut chain, k + 1, cur);
                    }
                }

                let y = total_get_copy(&chain, n_pl, [0.0_f32; ADAPTIVE_STATE_DIM]);
                let mut z = apply_unitary_parity(&y, &vh, c_p, s_p, d);
                let r2 = vec_norm_active(&z, d);
                let inv_r = if r2 > 1e-18 { inv_sqrt_newton_2step_f32(r2) } else { 1.0 };
                normalize_state_active_nonzero(&mut z, d);
                let _ = total_set(&mut next_cols, col, z);

                if has_target {
                    let mut y_v = 0.0_f32;
                    for i in 0..ADAPTIVE_D_MAX {
                        if i < d {
                            y_v += total_get_copy(&vh, i, 0.0) * total_get_copy(&y, i, 0.0);
                        }
                    }
                    let y_p = total_get_copy(&y, ADAPTIVE_PARITY_AXIS, 0.0);
                    let z_v = c_p * y_v - s_p * y_p;
                    let z_p = s_p * y_v + c_p * y_p;

                    // А. Транспортировка чувствительностей для всех gx через предвычисленную Q_{g,m} (0 делений!)
                    for gx in 0..self.num_inputs {
                        let base = ((m * ADAPTIVE_MAX_COLS + col) * self.num_inputs + gx) * ADAPTIVE_PARAMS_PER_CH;
                        for p_idx in 0..ADAPTIVE_PARAMS_PER_CH {
                            let active_p = p_idx < n_pl
                                || (p_idx >= ADAPTIVE_MAX_PLANES && p_idx < ADAPTIVE_MAX_PLANES + d)
                                || (p_idx == ADAPTIVE_MAX_PLANES + ADAPTIVE_D_MAX);
                            if !active_p {
                                continue;
                            }
                            let raw_s = total_get_copy(&self.sens, base + p_idx, [0.0_f32; ADAPTIVE_STATE_DIM]);
                            let mut vec_s = op.apply_q(&raw_s, d);
                            for i in 0..ADAPTIVE_D_MAX {
                                if i < d {
                                    let v = total_get_copy(&vec_s, i, 0.0);
                                    let _ = total_set(&mut vec_s, i, (v * inv_r).clamp(-8.0, 8.0));
                                } else {
                                    let _ = total_set(&mut vec_s, i, 0.0);
                                }
                            }
                            let vp = total_get_copy(&vec_s, ADAPTIVE_PARITY_AXIS, 0.0);
                            let _ = total_set(&mut vec_s, ADAPTIVE_PARITY_AXIS, (vp * inv_r).clamp(-8.0, 8.0));
                            let _ = total_set(&mut self.sens, base + p_idx, vec_s);
                        }
                    }

                    // Б. Локальные производные для активного генератора g (с кэшем cs/ss/dcs/dss — 0 делений!)
                    let ch_base = ((m * ADAPTIVE_MAX_COLS + col) * self.num_inputs + g) * ADAPTIVE_PARAMS_PER_CH;
                    for k in 0..n_pl {
                        let (p, q) = plane_pair_at(k);
                        let dc = total_get_copy(&op.dcs, k, 0.0);
                        let ds = total_get_copy(&op.dss, k, 2.0);
                        let pre = total_get_copy(&chain, k, [0.0_f32; ADAPTIVE_STATE_DIM]);
                        let xp = total_get_copy(&pre, p, 0.0);
                        let xq = total_get_copy(&pre, q, 0.0);
                        let mut dv = [0.0_f32; ADAPTIVE_STATE_DIM];
                        let _ = total_set(&mut dv, p, dc * xp - ds * xq);
                        let _ = total_set(&mut dv, q, ds * xp + dc * xq);
                        dv = op.apply_givens_cached_from(&dv, k + 1, n_pl);
                        dv = apply_unitary_parity(&dv, &vh, c_p, s_p, d);
                        let mut cur_s = total_get_copy(&self.sens, ch_base + k, [0.0_f32; ADAPTIVE_STATE_DIM]);
                        for i in 0..ADAPTIVE_D_MAX {
                            if i < d {
                                let v = total_get_copy(&cur_s, i, 0.0) + total_get_copy(&dv, i, 0.0) * inv_r;
                                let _ = total_set(&mut cur_s, i, v.clamp(-8.0, 8.0));
                            }
                        }
                        let vp = total_get_copy(&cur_s, ADAPTIVE_PARITY_AXIS, 0.0)
                            + total_get_copy(&dv, ADAPTIVE_PARITY_AXIS, 0.0) * inv_r;
                        let _ = total_set(&mut cur_s, ADAPTIVE_PARITY_AXIS, vp.clamp(-8.0, 8.0));
                        let _ = total_set(&mut self.sens, ch_base + k, cur_s);
                    }

                    for a in 0..d {
                        let vha = total_get_copy(&vh, a, 0.0);
                        let mut d_vh = [0.0_f32; ADAPTIVE_D_MAX];
                        for i in 0..ADAPTIVE_D_MAX {
                            if i < d {
                                let vhi = total_get_copy(&vh, i, 0.0);
                                let kron = if i == a { 1.0 } else { 0.0 };
                                let _ = total_set(&mut d_vh, i, (kron - vhi * vha) * inv_hv);
                            }
                        }
                        let mut d_yv = 0.0_f32;
                        for i in 0..ADAPTIVE_D_MAX {
                            if i < d {
                                d_yv += total_get_copy(&d_vh, i, 0.0) * total_get_copy(&y, i, 0.0);
                            }
                        }
                        let mut cur_s = total_get_copy(
                            &self.sens,
                            ch_base + ADAPTIVE_MAX_PLANES + a,
                            [0.0_f32; ADAPTIVE_STATE_DIM],
                        );
                        for i in 0..ADAPTIVE_D_MAX {
                            if i < d {
                                let dz = (c_p - 1.0) * d_yv * total_get_copy(&vh, i, 0.0)
                                    + (z_v - y_v) * total_get_copy(&d_vh, i, 0.0);
                                let nv = total_get_copy(&cur_s, i, 0.0) + dz * inv_r;
                                let _ = total_set(&mut cur_s, i, nv.clamp(-8.0, 8.0));
                            }
                        }
                        let dz_p = s_p * d_yv;
                        let nvp = total_get_copy(&cur_s, ADAPTIVE_PARITY_AXIS, 0.0) + dz_p * inv_r;
                        let _ = total_set(&mut cur_s, ADAPTIVE_PARITY_AXIS, nvp.clamp(-8.0, 8.0));
                        let _ = total_set(&mut self.sens, ch_base + ADAPTIVE_MAX_PLANES + a, cur_s);
                    }

                    let mut cur_sb = total_get_copy(
                        &self.sens,
                        ch_base + ADAPTIVE_MAX_PLANES + ADAPTIVE_D_MAX,
                        [0.0_f32; ADAPTIVE_STATE_DIM],
                    );
                    let dz_v_dp = -4.0 * d1_p * z_p;
                    let dz_p_dp = 4.0 * d1_p * z_v;
                    for i in 0..ADAPTIVE_D_MAX {
                        if i < d {
                            let dz = dz_v_dp * total_get_copy(&vh, i, 0.0);
                            let nv = total_get_copy(&cur_sb, i, 0.0) + dz * inv_r;
                            let _ = total_set(&mut cur_sb, i, nv.clamp(-8.0, 8.0));
                        }
                    }
                    let nvp = total_get_copy(&cur_sb, ADAPTIVE_PARITY_AXIS, 0.0) + dz_p_dp * inv_r;
                    let _ = total_set(&mut cur_sb, ADAPTIVE_PARITY_AXIS, nvp.clamp(-8.0, 8.0));
                    let _ = total_set(&mut self.sens, ch_base + ADAPTIVE_MAX_PLANES + ADAPTIVE_D_MAX, cur_sb);
                }
            }
            let _ = total_set(&mut self.state, m, next_cols);
        }

        // 2. Комбинированный прогноз, Per-Frame Solo + сферическая метрика прототипов с отталкиванием
        self.comb_logits.fill(0.0);
        // P5: Мягкий отложенный регуляризатор чётности (Мина №2): в d=2 держит честный O(2) (lambda_par*1.0),
        // в d>=3 не защёлкивает чётность до сходимости пространственных плоскостей!
        let par_warmup = if d <= 2 {
            if self.total_eps >= 200 {
                1.0_f32
            } else {
                total_div_f32(self.total_eps as f32, 200.0, 0.0)
            }
        } else if self.slow_ema > 0.88 || self.eps_in_dim > 2600 {
            0.075_f32
        } else {
            0.012_f32
        };
        // P5: Якорный LR для старых плоскостей + прогрев для новых плоскостей (сохраняет подгруппу при d -> d+1)
        let (old_lr_mult, new_lr_mult) = if self.prev_active_planes > 0 {
            if self.eps_in_dim < 400 {
                (0.55_f32, 2.20_f32)
            } else if self.eps_in_dim < 1200 {
                (0.85_f32, 1.35_f32)
            } else {
                (1.0_f32, 1.0_f32)
            }
        } else {
            (1.0_f32, 1.0_f32)
        };
        // P5: Плавная рампа center-pull при d >= 3 (в d <= 2 выключена для точных углов O(2) на L = 500)
        let center_ramp = if d <= 2 {
            0.0_f32
        } else {
            total_div_f32(self.eps_in_dim as f32, 400.0, 1.0).min(1.0)
        };
        let eff_scale = self.proto_scale * ADAPTIVE_COL_LOGIT_SCALE;
        let solo_mix = self.solo_w;
        let comb_mix = 1.0 - solo_mix;

        for m in 0..self.frames {
            let u_cols = total_get_copy(&self.state, m, zero_cols);
            let fa = total_get_copy(&self.frame_acc_ema, m, 0.05);
            let wm = 0.25 + fa * fa;
            let base_m = m * self.num_classes;

            let mut best_fm_c = 0usize;
            let mut best_fm_l = f32::NEG_INFINITY;
            for c in 0..self.num_classes {
                let p_off = (base_m + c) * ADAPTIVE_MAX_COLS;
                let mut sum_dot = 0.0_f32;
                for col in 0..ADAPTIVE_MAX_COLS {
                    if !is_col_active(col, d) {
                        continue;
                    }
                    let u = total_get_copy(&u_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                    let pr = total_get_copy(&self.proto, p_off + col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                    sum_dot += dot_active(&u, &pr, d);
                }
                let l_val = eff_scale * sum_dot;
                let _ = total_set(&mut self.frame_logits, c, l_val);
                let cur_comb = total_get_copy(&self.comb_logits, c, 0.0);
                let _ = total_set(&mut self.comb_logits, c, cur_comb + wm * l_val);
                if c == 0 || l_val > best_fm_l {
                    best_fm_l = l_val;
                    best_fm_c = c;
                }
            }

            if has_target && target < self.num_classes {
                let hit_f = if best_fm_c == target { 1.0_f32 } else { 0.0_f32 };
                let _ = total_set(&mut self.frame_acc_ema, m, fa + 0.02 * (hit_f - fa));

                let vis = total_get_copy(&self.proto_visits, base_m + target, 0.0);
                let tgt_off = (base_m + target) * ADAPTIVE_MAX_COLS;

                // P5: Ленивая инициализация прототипа при первом визите класса (аналог NonAbelianO4Rtrl, 0 оракулов)
                if vis == 0.0 && (lr_head > 0.0 || lr_rot > 0.0) {
                    let _ = total_set(&mut self.proto_visits, base_m + target, 1.0);
                    for col in 0..ADAPTIVE_MAX_COLS {
                        if !is_col_active(col, d) {
                            continue;
                        }
                        let mut uc = total_get_copy(&u_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                        normalize_state_active_nonzero(&mut uc, d);
                        let _ = total_set(&mut self.proto, tgt_off + col, uc);
                    }
                    continue;
                }

                rational_probs_p4_into(&self.frame_logits, &mut self.frame_probs, &mut self.frame_psi);
                fill_p4_logprob_grads(
                    &self.frame_logits,
                    &self.frame_probs,
                    &self.frame_psi,
                    target,
                    &mut self.lerrs,
                );

                let mut grad_cols = zero_cols;
                let mod_w = (solo_mix + comb_mix * wm).max(0.25);
                for c in 0..self.num_classes {
                    let ec = total_get_copy(&self.lerrs, c, 0.0);
                    if ec.abs() < 1.0e-4 && c != target {
                        continue;
                    }
                    let p_off = (base_m + c) * ADAPTIVE_MAX_COLS;
                    let w_ec = mod_w * eff_scale * ec;
                    for col in 0..ADAPTIVE_MAX_COLS {
                        if !is_col_active(col, d) {
                            continue;
                        }
                        let pr_vec = total_get_copy(&self.proto, p_off + col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                        let mut gu = total_get_copy(&grad_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                        for i in 0..ADAPTIVE_D_MAX {
                            if i < d {
                                let g0 = total_get_copy(&gu, i, 0.0);
                                let _ = total_set(&mut gu, i, g0 + w_ec * total_get_copy(&pr_vec, i, 0.0));
                            }
                        }
                        let gp = total_get_copy(&gu, ADAPTIVE_PARITY_AXIS, 0.0);
                        let _ = total_set(
                            &mut gu,
                            ADAPTIVE_PARITY_AXIS,
                            gp + w_ec * total_get_copy(&pr_vec, ADAPTIVE_PARITY_AXIS, 0.0),
                        );
                        let _ = total_set(&mut grad_cols, col, gu);
                    }
                }

                // P5: Задача-независимое притяжение состояния к центру прототипа (до обновления proto)
                if self.proto_center_beta > 0.0 && center_ramp > 0.0 && lr_rot > 0.0 {
                    for col in 0..ADAPTIVE_MAX_COLS {
                        if !is_col_active(col, d) {
                            continue;
                        }
                        let col_off = total_get_copy(&self.col_visit_offset, col, 0.0);
                        let vis_col = (vis - col_off).max(0.0);
                        let tp = total_get_copy(&self.proto, tgt_off + col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                        if vis_col >= 2.0 && vec_norm_active(&tp, d) > 0.25 {
                            let uc = total_get_copy(&u_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                            let c_gate = total_div_f32(vis_col, vis_col + 4.0, 0.0);
                            let c_eff = self.proto_center_beta * center_ramp * c_gate * mod_w;
                            let mut gu = total_get_copy(&grad_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                            for i in 0..ADAPTIVE_D_MAX {
                                if i < d {
                                    let g0 = total_get_copy(&gu, i, 0.0);
                                    let pv = total_get_copy(&tp, i, 0.0);
                                    let uv = total_get_copy(&uc, i, 0.0);
                                    let _ = total_set(&mut gu, i, g0 + c_eff * (pv - uv));
                                }
                            }
                            let gp0 = total_get_copy(&gu, ADAPTIVE_PARITY_AXIS, 0.0);
                            let pvp = total_get_copy(&tp, ADAPTIVE_PARITY_AXIS, 0.0);
                            let uvp = total_get_copy(&uc, ADAPTIVE_PARITY_AXIS, 0.0);
                            let _ = total_set(&mut gu, ADAPTIVE_PARITY_AXIS, gp0 + c_eff * (pvp - uvp));
                            let _ = total_set(&mut grad_cols, col, gu);
                        }
                    }
                }

                // P5: Кросс-энтропийное разделение прототипов (раздвигает конфузоров и исключает коллапс center-pull)
                if lr_head > 0.0 {
                    for c in 0..self.num_classes {
                        let ec = total_get_copy(&self.lerrs, c, 0.0);
                        if ec.abs() < 1.0e-4 && c != target {
                            continue;
                        }
                        if c != target && total_get_copy(&self.proto_visits, base_m + c, 0.0) == 0.0 {
                            continue;
                        }
                        let st_ce = (lr_head * eff_scale * ec).clamp(-0.25, 0.25);
                        if st_ce.abs() <= 1.0e-6 {
                            continue;
                        }
                        let p_off = (base_m + c) * ADAPTIVE_MAX_COLS;
                        for col in 0..ADAPTIVE_MAX_COLS {
                            if !is_col_active(col, d) {
                                continue;
                            }
                            let mut pr_vec = total_get_copy(&self.proto, p_off + col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                            if vec_norm_active(&pr_vec, d) <= 1.0e-6 {
                                continue;
                            }
                            let uc = total_get_copy(&u_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                            for i in 0..ADAPTIVE_D_MAX {
                                if i < d {
                                    let pv = total_get_copy(&pr_vec, i, 0.0);
                                    let uv = total_get_copy(&uc, i, 0.0);
                                    let _ = total_set(&mut pr_vec, i, pv + st_ce * uv);
                                }
                            }
                            let pvp = total_get_copy(&pr_vec, ADAPTIVE_PARITY_AXIS, 0.0);
                            let uvp = total_get_copy(&uc, ADAPTIVE_PARITY_AXIS, 0.0);
                            let _ = total_set(&mut pr_vec, ADAPTIVE_PARITY_AXIS, pvp + st_ce * uvp);
                            normalize_state_active_nonzero(&mut pr_vec, d);
                            let _ = total_set(&mut self.proto, p_off + col, pr_vec);
                        }
                    }
                }

                // Счётчик посещений цели + подтягивание активных столбцов цели к u_cols
                let _ = total_set(&mut self.proto_visits, base_m + target, vis + 1.0);
                for col in 0..ADAPTIVE_MAX_COLS {
                    if !is_col_active(col, d) {
                        continue;
                    }
                    let col_off = total_get_copy(&self.col_visit_offset, col, 0.0);
                    let vis_col = (vis - col_off).max(0.0);
                    let mut tp = total_get_copy(&self.proto, tgt_off + col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                    let uc = total_get_copy(&u_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                    let alpha_vis = total_div_unit_shifted_f32(0.32, 0.02 * vis_col).max(0.005);
                    for i in 0..ADAPTIVE_D_MAX {
                        if i < d {
                            let pv = total_get_copy(&tp, i, 0.0);
                            let uv = total_get_copy(&uc, i, 0.0);
                            let _ = total_set(&mut tp, i, pv + alpha_vis * (uv - pv));
                        }
                    }
                    let pvp = total_get_copy(&tp, ADAPTIVE_PARITY_AXIS, 0.0);
                    let uvp = total_get_copy(&uc, ADAPTIVE_PARITY_AXIS, 0.0);
                    let _ = total_set(&mut tp, ADAPTIVE_PARITY_AXIS, pvp + alpha_vis * (uvp - pvp));
                    normalize_state_active_nonzero(&mut tp, d);
                    let _ = total_set(&mut self.proto, tgt_off + col, tp);
                }

                // Отталкивание ближайшего виденного конфузатора в пространстве прототипов (аналог NonAbelianO4Rtrl)
                let mut active_cnt = 1usize;
                for col in 0..4usize {
                    if col < d {
                        active_cnt += 1;
                    }
                }
                let inv_ac = total_div_f32(1.0, active_cnt as f32, 0.25);
                let mut best_sim = -2.0_f32;
                let mut best_c = target;
                let mut found_other = false;
                for c in 0..self.num_classes {
                    if c == target || total_get_copy(&self.proto_visits, base_m + c, 0.0) == 0.0 {
                        continue;
                    }
                    let c_off = (base_m + c) * ADAPTIVE_MAX_COLS;
                    let mut s_dot = 0.0_f32;
                    for col in 0..ADAPTIVE_MAX_COLS {
                        if !is_col_active(col, d) {
                            continue;
                        }
                        let pt = total_get_copy(&self.proto, tgt_off + col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                        let pc = total_get_copy(&self.proto, c_off + col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                        s_dot += dot_active(&pt, &pc, d);
                    }
                    let sim = s_dot * inv_ac;
                    if !found_other || sim > best_sim {
                        best_sim = sim;
                        best_c = c;
                        found_other = true;
                    }
                }
                if found_other && best_sim > self.proto_rep_mu {
                    let excess = total_div_f32(best_sim - self.proto_rep_mu, 1.0 - self.proto_rep_mu, 0.0);
                    let step_rep = (self.proto_rep_lr * lr_head * 20.0 * excess).clamp(0.0, 0.08);
                    let c_off = (base_m + best_c) * ADAPTIVE_MAX_COLS;
                    for col in 0..ADAPTIVE_MAX_COLS {
                        if !is_col_active(col, d) {
                            continue;
                        }
                        let mut pt = total_get_copy(&self.proto, tgt_off + col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                        let mut pc = total_get_copy(&self.proto, c_off + col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                        for i in 0..ADAPTIVE_D_MAX {
                            if i < d {
                                let wti = total_get_copy(&pt, i, 0.0);
                                let wci = total_get_copy(&pc, i, 0.0);
                                let _ = total_set(&mut pt, i, wti - step_rep * wci);
                                let _ = total_set(&mut pc, i, wci - step_rep * wti);
                            }
                        }
                        let wtp = total_get_copy(&pt, ADAPTIVE_PARITY_AXIS, 0.0);
                        let wcp = total_get_copy(&pc, ADAPTIVE_PARITY_AXIS, 0.0);
                        let _ = total_set(&mut pt, ADAPTIVE_PARITY_AXIS, wtp - step_rep * wcp);
                        let _ = total_set(&mut pc, ADAPTIVE_PARITY_AXIS, wcp - step_rep * wtp);
                        normalize_state_active_nonzero(&mut pt, d);
                        normalize_state_active_nonzero(&mut pc, d);
                        let _ = total_set(&mut self.proto, tgt_off + col, pt);
                        let _ = total_set(&mut self.proto, c_off + col, pc);
                    }
                }

                // P5: Мягкое RTRL-обновление с 3-уровневым якорным LR (old_lr_mult / new_lr_mult), моментумом и saddle escape
                for gx in 0..self.num_inputs {
                    let ch_x = m * self.num_inputs + gx;
                    let rms_base = ch_x * ADAPTIVE_PARAMS_PER_CH;

                    let mut pl = total_get_copy(&self.planes, ch_x, [0.0_f32; ADAPTIVE_MAX_PLANES]);
                    for k in 0..n_pl {
                        let mut gp = 0.0_f32;
                        for col in 0..ADAPTIVE_MAX_COLS {
                            if !is_col_active(col, d) {
                                continue;
                            }
                            let sens_base = ((m * ADAPTIVE_MAX_COLS + col) * self.num_inputs + gx) * ADAPTIVE_PARAMS_PER_CH;
                            let svec = total_get_copy(&self.sens, sens_base + k, [0.0_f32; ADAPTIVE_STATE_DIM]);
                            let gu = total_get_copy(&grad_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                            gp += dot_active(&gu, &svec, d);
                        }
                        let cur_t = total_get_copy(&pl, k, 0.0);
                        let is_new_plane = self.prev_active_planes > 0 && k >= self.prev_active_planes;
                        gp = gp.clamp(-4.0, 4.0);
                        let old_m = total_get_copy(&self.m_theta, rms_base + k, 0.0);
                        let new_m = 0.80 * old_m + 0.20 * gp;
                        let _ = total_set(&mut self.m_theta, rms_base + k, new_m);
                        let old_rms = total_get_copy(&self.rms_theta, rms_base + k, 0.04);
                        let new_rms = 0.99 * old_rms + 0.01 * gp * gp;
                        let _ = total_set(&mut self.rms_theta, rms_base + k, new_rms);
                        let saddle_th = if is_new_plane && self.eps_in_dim < 500 { 0.08_f32 } else { 0.002_f32 };
                        let saddle = cur_t.abs() <= saddle_th;
                        let lr_k = lr_rot * if is_new_plane { new_lr_mult } else { old_lr_mult };
                        let (g_num, rms_eff) = if saddle {
                            (gp, 0.02 * gp * gp + 1.0e-22)
                        } else {
                            (new_m, new_rms + 0.02)
                        };
                        let raw_step = lr_k * g_num * inv_sqrt_newton_2step_f32(rms_eff);
                        let step = if saddle { raw_step.clamp(-lr_k, lr_k) } else { raw_step.clamp(-2.5 * lr_k, 2.5 * lr_k) };
                        let _ = total_set(&mut pl, k, (cur_t + step).clamp(-3.5, 3.5));
                    }
                    let _ = total_set(&mut self.planes, ch_x, pl);

                    let mut hv = total_get_copy(&self.h_vec, ch_x, [0.0_f32; ADAPTIVE_D_MAX]);
                    for a in 0..d {
                        let p_idx = ADAPTIVE_MAX_PLANES + a;
                        let mut gp = 0.0_f32;
                        for col in 0..ADAPTIVE_MAX_COLS {
                            if !is_col_active(col, d) {
                                continue;
                            }
                            let sens_base = ((m * ADAPTIVE_MAX_COLS + col) * self.num_inputs + gx) * ADAPTIVE_PARAMS_PER_CH;
                            let svec = total_get_copy(&self.sens, sens_base + p_idx, [0.0_f32; ADAPTIVE_STATE_DIM]);
                            let gu = total_get_copy(&grad_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                            gp += dot_active(&gu, &svec, d);
                        }
                        gp = gp.clamp(-4.0, 4.0);
                        let old_m = total_get_copy(&self.m_theta, rms_base + p_idx, 0.0);
                        let new_m = 0.80 * old_m + 0.20 * gp;
                        let _ = total_set(&mut self.m_theta, rms_base + p_idx, new_m);
                        let old_rms = total_get_copy(&self.rms_theta, rms_base + p_idx, 0.04);
                        let new_rms = 0.99 * old_rms + 0.01 * gp * gp;
                        let _ = total_set(&mut self.rms_theta, rms_base + p_idx, new_rms);
                        let cur_v = total_get_copy(&hv, a, 0.0);
                        let is_new_axis = self.prev_active_dim > 0 && a >= self.prev_active_dim;
                        let saddle_th = if is_new_axis && self.eps_in_dim < 500 { 0.08_f32 } else { 0.002_f32 };
                        let saddle = cur_v.abs() <= saddle_th;
                        let lr_a = lr_rot * if is_new_axis { new_lr_mult } else { old_lr_mult };
                        let (g_num, rms_eff) = if saddle {
                            (gp, 0.02 * gp * gp + 1.0e-22)
                        } else {
                            (new_m, new_rms + 0.02)
                        };
                        let raw_step = lr_a * g_num * inv_sqrt_newton_2step_f32(rms_eff);
                        let step = if saddle { raw_step.clamp(-lr_a, lr_a) } else { raw_step.clamp(-2.5 * lr_a, 2.5 * lr_a) };
                        let _ = total_set(&mut hv, a, (cur_v + step).clamp(-2.0, 2.0));
                    }
                    hv = normalized_hv_active(&hv, d);
                    let _ = total_set(&mut self.h_vec, ch_x, hv);

                    // Замедленный шаг чётности 0.25 * lr_rot с моментумом (пространственные оси сходятся первыми)
                    let b_idx = ADAPTIVE_MAX_PLANES + ADAPTIVE_D_MAX;
                    let mut gb = 0.0_f32;
                    for col in 0..ADAPTIVE_MAX_COLS {
                        if !is_col_active(col, d) {
                            continue;
                        }
                        let sens_base = ((m * ADAPTIVE_MAX_COLS + col) * self.num_inputs + gx) * ADAPTIVE_PARAMS_PER_CH;
                        let svec_b = total_get_copy(&self.sens, sens_base + b_idx, [0.0_f32; ADAPTIVE_STATE_DIM]);
                        let gu = total_get_copy(&grad_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                        gb += dot_active(&gu, &svec_b, d);
                    }
                    let cur_p = total_get_copy(&self.betas, ch_x, 0.41421356);
                    let (cp_x, sp_x, d1_x) = parity_rot_params(cur_p);
                    let reg_grad = -8.0 * self.lambda_par * par_warmup * sp_x * cp_x * d1_x;
                    let tot_gb = (gb + reg_grad).clamp(-4.0, 4.0);
                    let old_mb = total_get_copy(&self.m_theta, rms_base + b_idx, 0.0);
                    let new_mb = 0.80 * old_mb + 0.20 * tot_gb;
                    let _ = total_set(&mut self.m_theta, rms_base + b_idx, new_mb);
                    let old_rms = total_get_copy(&self.rms_theta, rms_base + b_idx, 0.04);
                    let new_rms = 0.99 * old_rms + 0.01 * tot_gb * tot_gb;
                    let _ = total_set(&mut self.rms_theta, rms_base + b_idx, new_rms);
                    let lr_par = 0.25 * lr_rot;
                    let step_b = (lr_par * new_mb * inv_sqrt_newton_2step_f32(new_rms + 0.02)).clamp(-2.5 * lr_par, 2.5 * lr_par);
                    let _ = total_set(&mut self.betas, ch_x, (cur_p + step_b).clamp(-0.08, 1.08));
                }

                // Неабелев лапласиан коммутаторов
                if self.num_inputs >= 2 && self.comm_w > 0.0 {
                    let nxt = if g + 1 < self.num_inputs { g + 1 } else { 0 };
                    let ch_a = m * self.num_inputs + g;
                    let ch_b = m * self.num_inputs + nxt;
                    let pla = total_get_copy(&self.planes, ch_a, [0.0_f32; ADAPTIVE_MAX_PLANES]);
                    let plb = total_get_copy(&self.planes, ch_b, [0.0_f32; ADAPTIVE_MAX_PLANES]);
                    let mut sum_sq_diff = 0.0_f32;
                    for k in 0..n_pl {
                        let diff = total_get_copy(&pla, k, 0.0) - total_get_copy(&plb, k, 0.0);
                        sum_sq_diff += diff * diff;
                    }
                    if sum_sq_diff < 0.04 && n_pl > 0 {
                        let mut pla_mod = pla;
                        let k0 = total_rem_usize(g, n_pl, 0);
                        let v0 = total_get_copy(&pla_mod, k0, 0.0);
                        let nudge = if v0 >= 0.0 { lr_rot * self.comm_w } else { -lr_rot * self.comm_w };
                        let _ = total_set(&mut pla_mod, k0, (v0 + nudge).clamp(-3.5, 3.5));
                        let _ = total_set(&mut self.planes, ch_a, pla_mod);
                    }
                }
            }
        }

        let mut pred = 0usize;
        let mut best_l = f32::NEG_INFINITY;
        for c in 0..self.num_classes {
            let l = total_get_copy(&self.comb_logits, c, 0.0);
            if c == 0 || l > best_l {
                best_l = l;
                pred = c;
            }
        }
        pred
    }

    /// Экспорт обученных гиперсферических прототипов в компактный формат `Q1.14` (`[i16; 7]`, 2x меньше RAM, Lean 4 `step132`).
    pub fn export_proto_q14(&self) -> Vec<[i16; ADAPTIVE_STATE_DIM]> {
        let mut out = vec![[0i16; ADAPTIVE_STATE_DIM]; self.proto.len()];
        for (idx, p) in self.proto.iter().enumerate() {
            let mut qv = [0i16; ADAPTIVE_STATE_DIM];
            for i in 0..ADAPTIVE_STATE_DIM {
                let v = total_get_copy(p, i, 0.0).clamp(-1.0, 1.0);
                let q = (v * 16384.0_f32).round().clamp(-16384.0, 16384.0) as i16;
                let _ = total_set(&mut qv, i, q);
            }
            let _ = total_set(&mut out, idx, qv);
        }
        out
    }

    /// Read-only `k`-шаговый инференс на `Q1.14` (`i16`) прототипах (0 аллокаций на стеке, Lean 4 `step132`).
    pub fn predict_k_ahead_q14_readonly(
        &self,
        inputs: &[&[f32]],
        proto_q14: &[[i16; ADAPTIVE_STATE_DIM]],
    ) -> usize {
        let d = self.active_dim;
        let n_pl = planes_for_dim(d);
        let init_cols = init_basis_columns(d);
        let zero_cols = [[0.0_f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_COLS];
        let n_frames = self.frames.min(ADAPTIVE_MAX_FRAMES);
        let mut cur = [init_cols; ADAPTIVE_MAX_FRAMES];

        for &x_slice in inputs {
            let g = self.channel_index_from_input(x_slice);
            for m in 0..n_frames {
                let ch = m * self.num_inputs + g;
                let u_prev_cols = total_get_copy(&cur, m, zero_cols);
                let pl = total_get_copy(&self.planes, ch, [0.0_f32; ADAPTIVE_MAX_PLANES]);
                let hv_raw = total_get_copy(&self.h_vec, ch, [0.0_f32; ADAPTIVE_D_MAX]);
                let vh = normalized_hv_active(&hv_raw, d);
                let p_rot = total_get_copy(&self.betas, ch, 0.41421356);
                let (c_p, s_p, _) = parity_rot_params(p_rot);
                let op = GivensParityOp::build_trig(&pl, n_pl, false);

                let mut next_cols = zero_cols;
                for col in 0..ADAPTIVE_MAX_COLS {
                    if !is_col_active(col, d) {
                        continue;
                    }
                    let u_prev = total_get_copy(&u_prev_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                    let y = op.apply_givens_cached_from(&u_prev, 0, n_pl);
                    let mut z = apply_unitary_parity(&y, &vh, c_p, s_p, d);
                    normalize_state_active_nonzero(&mut z, d);
                    let _ = total_set(&mut next_cols, col, z);
                }
                let _ = total_set(&mut cur, m, next_cols);
            }
        }

        let inv_q14 = 6.103515625e-5_f32; // 2^-14
        let eff_scale = self.proto_scale * ADAPTIVE_COL_LOGIT_SCALE * inv_q14;
        let mut best_c = 0usize;
        let mut best_l = f32::NEG_INFINITY;
        for c in 0..self.num_classes {
            let mut sum_l = 0.0_f32;
            for m in 0..n_frames {
                let u_cols = total_get_copy(&cur, m, zero_cols);
                let fa = total_get_copy(&self.frame_acc_ema, m, 0.05);
                let ex = (fa - 0.30).max(0.0);
                let ex2 = ex * ex;
                let wm = 0.02 + ex2 * ex2;
                let p_off = (m * self.num_classes + c) * ADAPTIVE_MAX_COLS;
                let mut sum_dot = 0.0_f32;
                for col in 0..ADAPTIVE_MAX_COLS {
                    if !is_col_active(col, d) {
                        continue;
                    }
                    let u = total_get_copy(&u_cols, col, [0.0_f32; ADAPTIVE_STATE_DIM]);
                    let pr = total_get_copy(proto_q14, p_off + col, [0i16; ADAPTIVE_STATE_DIM]);
                    let [u0, u1, u2, u3, u4, u5, up] = u;
                    let [q0, q1, q2, q3, q4, q5, qp] = pr;
                    sum_dot += u0 * (q0 as f32) + u1 * (q1 as f32) + up * (qp as f32);
                    if d > 2 { sum_dot += u2 * (q2 as f32); }
                    if d > 3 { sum_dot += u3 * (q3 as f32); }
                    if d > 4 { sum_dot += u4 * (q4 as f32); }
                    if d > 5 { sum_dot += u5 * (q5 as f32); }
                }
                sum_l += wm * eff_scale * sum_dot;
            }
            if c == 0 || sum_l > best_l {
                best_l = sum_l;
                best_c = c;
            }
        }
        best_c
    }
}
