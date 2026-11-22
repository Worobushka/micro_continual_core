// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov & Sofia Sidorenko
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Benchmark Suite Step 5 (v0.4.0 Comprehensive Critique-Response & Stress Benchmarks):
//!   1. Non-Abelian Group A_5 <= SO(3) State Tracking & Length Generalization:
//!      Full 60-element A_5 word tracking AND projected 5-point orbit p_t(0) in {0..4}
//!      across L_train = 8 -> Eval L = 6, 12, 18, 30 vs Fixed-Axis Cayley, Diagonal LRU,
//!      Input-Selective SSM (Mamba-RTRL), and GRU+BPTT.
//!   2. Hardware IEEE-754 Register Bit-Flip SEU Injection (bit 30 live in state every 25 steps,
//!      enable_input_clamp = true ON for all baselines) on BOTH:
//!      - RepeatPrevious K=4 (4 symbols, pure RL +1/-1)
//!      - Continual Delayed Modular-Shift K=4 (A->B->A, y_t = (x_{t-4} + shift) mod 4)
//!      vs Deep 2L RTRRL-LRU + PH, Online Selective SSM + PH, Online Per-Step GRU + PH, 1L LRU.
//!   3. Hard Continual RL & Chaotic Partial Observability Controls (10 seeds):
//!      - Task 3A: Multi-Cue Balanced Majority Corridor L=20 (A->B->A, +/-0.35 noise)
//!      - Task 3B: Chaotic Lorenz-63 1D-Projection Regime-Switch Stream (RK4, K=8 steps ahead)
//!      - Task 3C: Position-Only CartPole 2D & +dx (A->B->A)

use std::time::Instant;

use micro_continual_core::baselines::{
    DeepRtrrlLruPhBaseline, OnlineGruBaseline, OnlineGruPerStepPhBaseline,
    OnlineSelectiveSsmPhBaseline, RtrrlLruBaseline, RtrrlLruPhBaseline, SelectiveSsmBank,
};
use micro_continual_core::math::{
    fast_dot, fast_saxpy, rand_signed_f32, rational_probs_and_psi_into, splitmix64, SpinorEncoder,
};
use micro_continual_core::rtrl::{
    AlgebraicSelectiveCayley, NonAbelianCayleyRtrl, PlasticCayleyParams, PlasticCayleyState,
};
use micro_continual_core::stream_baselines::OnlineLruStream;
use micro_continual_core::{MicroContinualCore, MicroCoreConfig};

fn mean_sd(xs: &[f64]) -> (f64, f64) {
    let n = xs.len().max(1) as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = xs.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / n;
    (m, v.sqrt())
}

// =============================================================================
// Suite 1: Non-Abelian Group A_5 State Tracking (60-element + 5-point orbit)
// =============================================================================

struct A5Group {
    perms: Vec<[u8; 5]>,
    mul_gen: [[usize; 60]; 4],
    ident_id: usize,
}

impl A5Group {
    fn compose(a: [u8; 5], b: [u8; 5]) -> [u8; 5] {
        [
            a[b[0] as usize],
            a[b[1] as usize],
            a[b[2] as usize],
            a[b[3] as usize],
            a[b[4] as usize],
        ]
    }

    fn is_even(p: [u8; 5]) -> bool {
        let mut inv = 0usize;
        for i in 0..5 {
            for j in (i + 1)..5 {
                if p[i] > p[j] {
                    inv += 1;
                }
            }
        }
        (inv % 2) == 0
    }

    fn new() -> Self {
        let mut perms = Vec::with_capacity(60);
        for a in 0..5u8 {
            for b in 0..5u8 {
                if b == a { continue; }
                for c in 0..5u8 {
                    if c == a || c == b { continue; }
                    for d in 0..5u8 {
                        if d == a || d == b || d == c { continue; }
                        let e = 10 - (a + b + c + d);
                        let p = [a, b, c, d, e];
                        if Self::is_even(p) {
                            perms.push(p);
                        }
                    }
                }
            }
        }
        perms.sort();
        assert_eq!(perms.len(), 60);

        // 4 образующих A_5: a=(0 1 2), a^-1=(0 2 1), b=(0 1 2 3 4), b^-1=(4 3 2 1 0)
        let gens: [[u8; 5]; 4] = [
            [1, 2, 0, 3, 4],
            [2, 0, 1, 3, 4],
            [1, 2, 3, 4, 0],
            [4, 0, 1, 2, 3],
        ];
        let mut mul_gen = [[0usize; 60]; 4];
        for (g_idx, &g_perm) in gens.iter().enumerate() {
            for (s_id, &cur_perm) in perms.iter().enumerate() {
                // Правое действие cur * g: p_t(0) = cur(g(0)), требует знания всей перестановки p_{t-1} in A_5
                let prod = Self::compose(cur_perm, g_perm);
                let nxt = perms.binary_search(&prod).expect("even permutation in A5");
                mul_gen[g_idx][s_id] = nxt;
            }
        }
        let ident_id = perms.binary_search(&[0, 1, 2, 3, 4]).unwrap();
        Self { perms, mul_gen, ident_id }
    }
}

struct FixedAxisCayleySupervised {
    params: PlasticCayleyParams,
    state: PlasticCayleyState,
    in_dim: usize,
    num_classes: usize,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    v_out_w: Vec<f32>,
    v_out_b: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<f32>,
    psis: Vec<f32>,
    grad_b: Vec<f32>,
}

impl FixedAxisCayleySupervised {
    fn new(sectors: usize, in_dim: usize, num_classes: usize, kappa: f32, seed: u64) -> Self {
        let mut params = PlasticCayleyParams::new_with_grid(sectors, in_dim, seed, 1);
        for a in &mut params.alpha {
            *a = (*a * 0.35).clamp(0.05, 0.25);
        }
        let state = PlasticCayleyState::new(sectors, in_dim, kappa);
        let d = sectors * 3;
        let mut rng = splitmix64(seed ^ 0xF1ED_CA71);
        let mut out_w = vec![0.0_f32; num_classes * d];
        for v in &mut out_w {
            *v = rand_signed_f32(&mut rng) * 0.10;
        }
        Self {
            params,
            state,
            in_dim,
            num_classes,
            out_w,
            out_b: vec![0.0; num_classes],
            v_out_w: vec![0.04; num_classes * d],
            v_out_b: vec![0.04; num_classes],
            logits: vec![0.0; num_classes],
            probs: vec![0.0; num_classes],
            psis: vec![1.0; num_classes],
            grad_b: vec![0.0; d],
        }
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn memory_footprint_bytes(&self) -> usize {
        self.params.memory_footprint_bytes()
            + self.state.memory_footprint_bytes()
            + (self.out_w.len()
                + self.out_b.len()
                + self.v_out_w.len()
                + self.v_out_b.len()
                + self.logits.len()
                + self.probs.len()
                + self.psis.len()
                + self.grad_b.len())
                * 4
    }

    fn step_and_learn(&mut self, raw_x: &[f32], target: Option<usize>, ro_lr: f32, rtrl_lr: f32) -> usize {
        self.state.step_forward(&self.params, raw_x);
        let d = self.params.dim();
        let mut best_c = 0usize;
        let mut best_l = f32::NEG_INFINITY;
        for c in 0..self.num_classes {
            let off = c * d;
            let lg = self.out_b[c] + fast_dot(&self.out_w[off..off + d], &self.state.b);
            self.logits[c] = lg;
            if lg > best_l {
                best_l = lg;
                best_c = c;
            }
        }
        rational_probs_and_psi_into(&self.logits, &mut self.probs, &mut self.psis);

        if let Some(y) = target {
            self.grad_b.fill(0.0);
            for c in 0..self.num_classes {
                let err_c = (self.psis[c] * (if c == y { 1.0 } else { 0.0 } - self.probs[c])).clamp(-2.0, 2.0);
                let off = c * d;
                fast_saxpy(&mut self.grad_b, &self.out_w[off..off + d], err_c);
                for i in 0..d {
                    let g = err_c * self.state.b[i];
                    let nv = 0.98 * self.v_out_w[off + i] + 0.02 * g * g;
                    self.v_out_w[off + i] = nv;
                    let st = (g / (nv + 0.0025).sqrt()).clamp(-3.0, 3.0);
                    self.out_w[off + i] = (self.out_w[off + i] * 0.99995 + ro_lr * st).clamp(-8.0, 8.0);
                }
                let nvb = 0.98 * self.v_out_b[c] + 0.02 * err_c * err_c;
                self.v_out_b[c] = nvb;
                let stb = (err_c / (nvb + 0.0025).sqrt()).clamp(-3.0, 3.0);
                self.out_b[c] = (self.out_b[c] + 0.25 * ro_lr * stb).clamp(-4.0, 4.0);
            }
            if rtrl_lr > 0.0 {
                self.state.accumulate_traces(&self.grad_b, 0.0, self.in_dim);
                self.state.apply_td_update(&mut self.params, 1.0, rtrl_lr, self.in_dim);
            }
        }
        best_c
    }
}

struct GruSequenceClassifier {
    x_dim: usize,
    h_dim: usize,
    num_classes: usize,
    wz: Vec<f32>, uz: Vec<f32>, bz: Vec<f32>,
    wr: Vec<f32>, ur: Vec<f32>, br: Vec<f32>,
    wn: Vec<f32>, un: Vec<f32>, bn: Vec<f32>,
    wo: Vec<f32>, bo: Vec<f32>,
    m_all: Vec<Vec<f32>>,
    v_all: Vec<Vec<f32>>,
    step_t: usize,
    h: Vec<f32>,
    cache: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>,
}

impl GruSequenceClassifier {
    fn init_vec(n: usize, scale: f32, rng: &mut u64) -> Vec<f32> {
        (0..n).map(|_| rand_signed_f32(rng) * scale).collect()
    }

    fn new(x_dim: usize, h_dim: usize, num_classes: usize, seed: u64) -> Self {
        let mut rng = splitmix64(seed ^ 0xA5_6400_0001);
        let sx = 0.5 / (x_dim as f32).sqrt();
        let sh = 0.5 / (h_dim as f32).sqrt();
        let sizes = [
            h_dim * x_dim, h_dim * h_dim, h_dim,
            h_dim * x_dim, h_dim * h_dim, h_dim,
            h_dim * x_dim, h_dim * h_dim, h_dim,
            num_classes * h_dim, num_classes,
        ];
        Self {
            x_dim,
            h_dim,
            num_classes,
            wz: Self::init_vec(sizes[0], sx, &mut rng),
            uz: Self::init_vec(sizes[1], sh, &mut rng),
            bz: vec![0.0; sizes[2]],
            wr: Self::init_vec(sizes[3], sx, &mut rng),
            ur: Self::init_vec(sizes[4], sh, &mut rng),
            br: vec![0.0; sizes[5]],
            wn: Self::init_vec(sizes[6], sx, &mut rng),
            un: Self::init_vec(sizes[7], sh, &mut rng),
            bn: vec![0.0; sizes[8]],
            wo: Self::init_vec(sizes[9], sh, &mut rng),
            bo: vec![0.0; sizes[10]],
            m_all: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            v_all: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            step_t: 0,
            h: vec![0.0; h_dim],
            cache: Vec::new(),
        }
    }

    fn memory_footprint_bytes(&self) -> usize {
        self.m_all.iter().map(|v| v.len()).sum::<usize>() * 12
    }

    fn reset(&mut self) {
        self.h.fill(0.0);
        self.cache.clear();
    }

    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x.clamp(-12.0, 12.0)).exp())
    }

    fn forward_step(&mut self, x: &[f32]) -> usize {
        let xd = self.x_dim;
        let hd = self.h_dim;
        let hp = self.h.clone();
        let mut z = vec![0.0_f32; hd];
        let mut r = vec![0.0_f32; hd];
        let mut n = vec![0.0_f32; hd];
        let mut hn = vec![0.0_f32; hd];
        for i in 0..hd {
            let mut sz = self.bz[i];
            let mut sr = self.br[i];
            for j in 0..xd {
                sz += self.wz[i * xd + j] * x[j];
                sr += self.wr[i * xd + j] * x[j];
            }
            for j in 0..hd {
                sz += self.uz[i * hd + j] * hp[j];
                sr += self.ur[i * hd + j] * hp[j];
            }
            z[i] = Self::sigmoid(sz);
            r[i] = Self::sigmoid(sr);
        }
        for i in 0..hd {
            let mut sn = self.bn[i];
            for j in 0..xd { sn += self.wn[i * xd + j] * x[j]; }
            for j in 0..hd { sn += self.un[i * hd + j] * (r[j] * hp[j]); }
            n[i] = sn.tanh();
            hn[i] = (1.0 - z[i]) * hp[i] + z[i] * n[i];
        }
        let mut logits = vec![0.0_f32; self.num_classes];
        let mut max_l = f32::NEG_INFINITY;
        let mut best_c = 0usize;
        for c in 0..self.num_classes {
            let mut s = self.bo[c];
            for j in 0..hd { s += self.wo[c * hd + j] * hn[j]; }
            logits[c] = s;
            if s > max_l { max_l = s; best_c = c; }
        }
        let mut probs = vec![0.0_f32; self.num_classes];
        let mut sum_e = 0.0_f32;
        for c in 0..self.num_classes { let e = (logits[c] - max_l).exp(); probs[c] = e; sum_e += e; }
        for p in &mut probs { *p /= sum_e.max(1e-8); }
        self.h = hn.clone();
        self.cache.push((x.to_vec(), hp, z, r, n, hn, probs));
        best_c
    }

    fn adam(w: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], lr: f32, t: usize) {
        let b1 = 0.9_f32;
        let b2 = 0.999_f32;
        let c1 = 1.0 - b1.powi(t as i32);
        let c2 = 1.0 - b2.powi(t as i32);
        for i in 0..w.len() {
            let gc = g[i].clamp(-2.0, 2.0);
            m[i] = b1 * m[i] + (1.0 - b1) * gc;
            v[i] = b2 * v[i] + (1.0 - b2) * gc * gc;
            w[i] = (w[i] - lr * (m[i] / c1.max(1e-6)) / ((v[i] / c2.max(1e-6)).sqrt() + 1e-6)).clamp(-5.0, 5.0);
        }
    }

    fn bptt_update(&mut self, targets: &[usize], lr: f32) {
        let t_len = self.cache.len().min(targets.len());
        if t_len == 0 { return; }
        let (xd, hd, nc) = (self.x_dim, self.h_dim, self.num_classes);
        let mut grads = [
            vec![0.0; hd * xd], vec![0.0; hd * hd], vec![0.0; hd],
            vec![0.0; hd * xd], vec![0.0; hd * hd], vec![0.0; hd],
            vec![0.0; hd * xd], vec![0.0; hd * hd], vec![0.0; hd],
            vec![0.0; nc * hd], vec![0.0; nc],
        ];
        let mut dh_next = vec![0.0_f32; hd];
        for t in (0..t_len).rev() {
            let (ref x, ref hp, ref z, ref r, ref n, ref hn, ref probs) = self.cache[t];
            let y = targets[t];
            let mut dh = dh_next.clone();
            for c in 0..nc {
                let dl = probs[c] - if c == y { 1.0 } else { 0.0 };
                grads[10][c] += dl;
                for j in 0..hd { grads[9][c * hd + j] += dl * hn[j]; dh[j] += self.wo[c * hd + j] * dl; }
            }
            let mut dr = vec![0.0_f32; hd];
            let mut dh_prev = vec![0.0_f32; hd];
            for i in 0..hd {
                let dhi = dh[i].clamp(-4.0, 4.0);
                dh_prev[i] += dhi * (1.0 - z[i]);
                let dsn = dhi * z[i] * (1.0 - n[i] * n[i]);
                let dsz = dhi * (n[i] - hp[i]) * z[i] * (1.0 - z[i]);
                grads[8][i] += dsn; grads[2][i] += dsz;
                for j in 0..xd { grads[6][i * xd + j] += dsn * x[j]; grads[0][i * xd + j] += dsz * x[j]; }
                for j in 0..hd {
                    grads[7][i * hd + j] += dsn * r[j] * hp[j];
                    dr[j] += dsn * self.un[i * hd + j] * hp[j];
                    dh_prev[j] += dsn * self.un[i * hd + j] * r[j];
                    grads[1][i * hd + j] += dsz * hp[j];
                    dh_prev[j] += dsz * self.uz[i * hd + j];
                }
            }
            for i in 0..hd {
                let dsr = dr[i] * r[i] * (1.0 - r[i]);
                grads[5][i] += dsr;
                for j in 0..xd { grads[3][i * xd + j] += dsr * x[j]; }
                for j in 0..hd { grads[4][i * hd + j] += dsr * hp[j]; dh_prev[j] += dsr * self.ur[i * hd + j]; }
            }
            dh_next = dh_prev;
        }
        self.step_t += 1;
        let st = self.step_t;
        Self::adam(&mut self.wz, &grads[0], &mut self.m_all[0], &mut self.v_all[0], lr, st);
        Self::adam(&mut self.uz, &grads[1], &mut self.m_all[1], &mut self.v_all[1], lr, st);
        Self::adam(&mut self.bz, &grads[2], &mut self.m_all[2], &mut self.v_all[2], lr, st);
        Self::adam(&mut self.wr, &grads[3], &mut self.m_all[3], &mut self.v_all[3], lr, st);
        Self::adam(&mut self.ur, &grads[4], &mut self.m_all[4], &mut self.v_all[4], lr, st);
        Self::adam(&mut self.br, &grads[5], &mut self.m_all[5], &mut self.v_all[5], lr, st);
        Self::adam(&mut self.wn, &grads[6], &mut self.m_all[6], &mut self.v_all[6], lr, st);
        Self::adam(&mut self.un, &grads[7], &mut self.m_all[7], &mut self.v_all[7], lr, st);
        Self::adam(&mut self.bn, &grads[8], &mut self.m_all[8], &mut self.v_all[8], lr, st);
        Self::adam(&mut self.wo, &grads[9], &mut self.m_all[9], &mut self.v_all[9], lr, st);
        Self::adam(&mut self.bo, &grads[10], &mut self.m_all[10], &mut self.v_all[10], lr, st);
    }
}

fn flip_bit30_precond(v: &mut f32) {
    if !v.is_finite() || v.abs() < 0.25 || v.abs() >= 2.0 {
        *v = 1.25;
    }
    *v = f32::from_bits(v.to_bits() ^ (1u32 << 30));
}

fn eval_a5_suite() {
    let a5 = A5Group::new();
    let seeds: Vec<u64> = (1..=10).collect();
    let train_eps = 4000usize;
    let eval_eps = 200usize;
    let eval_lens = [6usize, 12, 18, 30];

    let gen_word = |rng: &mut u64, len: usize| -> (Vec<usize>, Vec<usize>) {
        let mut gs = Vec::with_capacity(len);
        let mut st_ids = Vec::with_capacity(len);
        let mut cur = a5.ident_id;
        for _ in 0..len {
            *rng = splitmix64(*rng);
            let g = ((*rng >> 16) as usize) % 4;
            cur = a5.mul_gen[g][cur];
            gs.push(g);
            st_ids.push(cur);
        }
        (gs, st_ids)
    };

    #[derive(Default)]
    struct A5Row {
        full: [Vec<f64>; 4],
        orbit: [Vec<f64>; 4],
        ram: f64,
        us: Vec<f64>,
        seu: bool,
    }
    let mut r_nac_m1 = A5Row { seu: true, ..Default::default() };
    let mut r_nac_proto = A5Row { seu: true, ..Default::default() };
    let mut r_nac = A5Row { seu: true, ..Default::default() };
    let mut r_fix = A5Row { seu: true, ..Default::default() };
    let mut r_lru = A5Row { seu: true, ..Default::default() };
    let mut r_ssm = A5Row { seu: true, ..Default::default() };
    let mut r_gru = A5Row { seu: true, ..Default::default() };

    for &seed in &seeds {
        // 0a. NonAbelian 3D Cayley-RTRL (SO(3) Proto M=1 irreducible frame = 9D, 7.6 KB, 0% exp, 0 oracle)
        let mut rng_tr = splitmix64(seed ^ 0xA5_77A1);
        let mut rng_ev = splitmix64(seed ^ 0xA5_E7A1);
        let mut nac_m1 = NonAbelianCayleyRtrl::new(3, 4, 60, 0.25, seed);
        nac_m1.enable_so3_proto_metric(10.0, 1.2, 0.08);
        r_nac_m1.ram = nac_m1.memory_footprint_bytes() as f64 / 1024.0;
        let t0 = Instant::now();
        let mut steps_m1 = 0.0_f64;
        for ep in 0..train_eps {
            let len = 1 + (ep % 8);
            let (gs, ys) = gen_word(&mut rng_tr, len);
            nac_m1.reset();
            let dec = 1.0_f32 / (1.0 + 6.0 * (ep as f32) / (train_eps as f32));
            for (&g, &y) in gs.iter().zip(ys.iter()) {
                let mut x = [0.0_f32; 4];
                x[g] = 1.0;
                let _ = nac_m1.step_and_learn_supervised(&x, Some(y), 0.02 * dec, 0.020 * dec);
                steps_m1 += 1.0;
            }
        }
        r_nac_m1.us.push((t0.elapsed().as_nanos() as f64 / 1000.0) / steps_m1);
        for (li, &len_t) in eval_lens.iter().enumerate() {
            let (mut hits, mut ohits) = (0usize, 0usize);
            for _ in 0..eval_eps {
                let (gs, ys) = gen_word(&mut rng_ev, len_t);
                nac_m1.reset();
                let mut pred = 0usize;
                for &g in &gs {
                    let mut x = [0.0_f32; 4];
                    x[g] = 1.0;
                    pred = nac_m1.step_and_learn_supervised(&x, None, 0.0, 0.0);
                }
                let y_true = ys[len_t - 1];
                if pred == y_true { hits += 1; }
                if a5.perms[pred][0] == a5.perms[y_true][0] { ohits += 1; }
            }
            r_nac_m1.full[li].push(hits as f64 * 100.0 / eval_eps as f64);
            r_nac_m1.orbit[li].push(ohits as f64 * 100.0 / eval_eps as f64);
        }
        flip_bit30_precond(&mut nac_m1.u[0]);
        nac_m1.inject_weight_bit_flip(0, 30);
        nac_m1.inject_weight_bit_flip(1, 30);
        let _ = nac_m1.step_and_learn_supervised(&[f32::NAN, 1e30, 0.0, 0.0], None, 0.0, 0.0);
        if !nac_m1.b.iter().all(|v| v.is_finite())
            || !nac_m1.u.iter().all(|v| v.is_finite())
            || !nac_m1.wg.iter().all(|v| v.is_finite())
            || !nac_m1.out_w.iter().all(|v| v.is_finite())
        {
            r_nac_m1.seu = false;
        }

        // 0b. NonAbelian 3D Cayley-RTRL (SO(3) Proto-p4, task-agnostic, 0% exp, 0 group oracle) — M=4 frames (36D)
        let mut rng_tr = splitmix64(seed ^ 0xA5_77A1);
        let mut rng_ev = splitmix64(seed ^ 0xA5_E7A1);
        let mut nac_p = NonAbelianCayleyRtrl::new(12, 4, 60, 0.25, seed);
        nac_p.enable_so3_proto_metric(10.0, 1.2, 0.08);
        r_nac_proto.ram = nac_p.memory_footprint_bytes() as f64 / 1024.0;
        let t0 = Instant::now();
        let mut steps_p = 0.0_f64;
        for ep in 0..train_eps {
            let len = 1 + (ep % 8);
            let (gs, ys) = gen_word(&mut rng_tr, len);
            nac_p.reset();
            let dec = 1.0_f32 / (1.0 + 6.0 * (ep as f32) / (train_eps as f32));
            for (&g, &y) in gs.iter().zip(ys.iter()) {
                let mut x = [0.0_f32; 4];
                x[g] = 1.0;
                let _ = nac_p.step_and_learn_supervised(&x, Some(y), 0.02 * dec, 0.020 * dec);
                steps_p += 1.0;
            }
        }
        r_nac_proto.us.push((t0.elapsed().as_nanos() as f64 / 1000.0) / steps_p);
        for (li, &len_t) in eval_lens.iter().enumerate() {
            let (mut hits, mut ohits) = (0usize, 0usize);
            for _ in 0..eval_eps {
                let (gs, ys) = gen_word(&mut rng_ev, len_t);
                nac_p.reset();
                let mut pred = 0usize;
                for &g in &gs {
                    let mut x = [0.0_f32; 4];
                    x[g] = 1.0;
                    pred = nac_p.step_and_learn_supervised(&x, None, 0.0, 0.0);
                }
                let y_true = ys[len_t - 1];
                if pred == y_true { hits += 1; }
                if a5.perms[pred][0] == a5.perms[y_true][0] { ohits += 1; }
            }
            r_nac_proto.full[li].push(hits as f64 * 100.0 / eval_eps as f64);
            r_nac_proto.orbit[li].push(ohits as f64 * 100.0 / eval_eps as f64);
        }
        flip_bit30_precond(&mut nac_p.u[0]);
        nac_p.inject_weight_bit_flip(0, 30);
        nac_p.inject_weight_bit_flip(1, 30);
        let _ = nac_p.step_and_learn_supervised(&[f32::NAN, 1e30, 0.0, 0.0], None, 0.0, 0.0);
        if !nac_p.b.iter().all(|v| v.is_finite())
            || !nac_p.u.iter().all(|v| v.is_finite())
            || !nac_p.wg.iter().all(|v| v.is_finite())
            || !nac_p.out_w.iter().all(|v| v.is_finite())
        {
            r_nac_proto.seu = false;
        }

        // 1. NonAbelian 3D Cayley-RTRL (Linear Readout, v0.4.0) — M=4 SO(3) frames (12 sectors = 36D), end-to-end
        let mut rng_tr = splitmix64(seed ^ 0xA5_77A1);
        let mut rng_ev = splitmix64(seed ^ 0xA5_E7A1);
        let mut nac = NonAbelianCayleyRtrl::new(12, 4, 60, 0.25, seed);
        nac.enable_so3_frames_end_to_end();
        r_nac.ram = nac.memory_footprint_bytes() as f64 / 1024.0;
        let t0 = Instant::now();
        let mut steps_n = 0.0_f64;
        for ep in 0..train_eps {
            let len = 1 + (ep % 8);
            let (gs, ys) = gen_word(&mut rng_tr, len);
            nac.reset();
            for (&g, &y) in gs.iter().zip(ys.iter()) {
                let mut x = [0.0_f32; 4];
                x[g] = 1.0;
                let _ = nac.step_and_learn_supervised(&x, Some(y), 0.22, 0.04);
                steps_n += 1.0;
            }
        }
        r_nac.us.push((t0.elapsed().as_nanos() as f64 / 1000.0) / steps_n);
        for (li, &len_t) in eval_lens.iter().enumerate() {
            let (mut hits, mut ohits) = (0usize, 0usize);
            for _ in 0..eval_eps {
                let (gs, ys) = gen_word(&mut rng_ev, len_t);
                nac.reset();
                let mut pred = 0usize;
                for &g in &gs {
                    let mut x = [0.0_f32; 4];
                    x[g] = 1.0;
                    pred = nac.step_and_learn_supervised(&x, None, 0.0, 0.0);
                }
                let y_true = ys[len_t - 1];
                if pred == y_true { hits += 1; }
                if a5.perms[pred][0] == a5.perms[y_true][0] { ohits += 1; }
            }
            r_nac.full[li].push(hits as f64 * 100.0 / eval_eps as f64);
            r_nac.orbit[li].push(ohits as f64 * 100.0 / eval_eps as f64);
        }
        flip_bit30_precond(&mut nac.u[0]);
        nac.inject_weight_bit_flip(0, 30);
        nac.step_forward(&[f32::NAN, 1e30, 0.0, 0.0]);
        if !nac.b.iter().all(|v| v.is_finite())
            || !nac.u.iter().all(|v| v.is_finite())
            || !nac.wg.iter().all(|v| v.is_finite())
            || !nac.out_w.iter().all(|v| v.is_finite())
        {
            r_nac.seu = false;
        }

        // 2. Fixed-Axis Cayley-RTRL (v0.3.0, 12 sectors = 36D)
        let mut rng_tr = splitmix64(seed ^ 0xA5_77A1);
        let mut rng_ev = splitmix64(seed ^ 0xA5_E7A1);
        let mut fix = FixedAxisCayleySupervised::new(12, 4, 60, 0.25, seed);
        r_fix.ram = fix.memory_footprint_bytes() as f64 / 1024.0;
        let t0 = Instant::now();
        for ep in 0..train_eps {
            let len = 1 + (ep % 8);
            let (gs, ys) = gen_word(&mut rng_tr, len);
            fix.reset();
            for (&g, &y) in gs.iter().zip(ys.iter()) {
                let mut x = [0.0_f32; 4];
                x[g] = 1.0;
                let _ = fix.step_and_learn(&x, Some(y), 0.05, 0.015);
            }
        }
        r_fix.us.push((t0.elapsed().as_nanos() as f64 / 1000.0) / steps_n);
        for (li, &len_t) in eval_lens.iter().enumerate() {
            let (mut hits, mut ohits) = (0usize, 0usize);
            for _ in 0..eval_eps {
                let (gs, ys) = gen_word(&mut rng_ev, len_t);
                fix.reset();
                let mut pred = 0usize;
                for &g in &gs {
                    let mut x = [0.0_f32; 4];
                    x[g] = 1.0;
                    pred = fix.step_and_learn(&x, None, 0.0, 0.0);
                }
                let y_true = ys[len_t - 1];
                if pred == y_true { hits += 1; }
                if a5.perms[pred][0] == a5.perms[y_true][0] { ohits += 1; }
            }
            r_fix.full[li].push(hits as f64 * 100.0 / eval_eps as f64);
            r_fix.orbit[li].push(ohits as f64 * 100.0 / eval_eps as f64);
        }
        flip_bit30_precond(&mut fix.state.u[0]);
        fix.params.inject_weight_bit_flip(0, 30);
        let _ = fix.step_and_learn(&[f32::NAN, 1e30, 0.0, 0.0], None, 0.0, 0.0);
        if !fix.state.b.iter().all(|v| v.is_finite())
            || !fix.state.u.iter().all(|v| v.is_finite())
            || !fix.params.in_w.iter().all(|v| v.is_finite())
            || !fix.params.alpha.iter().all(|v| v.is_finite())
            || !fix.out_w.iter().all(|v| v.is_finite())
        {
            r_fix.seu = false;
        }

        // 3. Diagonal Complex OnlineLruStream (18 complex units = 36D)
        let mut rng_tr = splitmix64(seed ^ 0xA5_77A1);
        let mut rng_ev = splitmix64(seed ^ 0xA5_E7A1);
        let mut lru = OnlineLruStream::new(4, 18, 60, seed);
        lru.standardize = false;
        lru.lr = 0.015;
        lru.rtrl_lr = 0.008;
        r_lru.ram = lru.memory_footprint_bytes() as f64 / 1024.0;
        let t0 = Instant::now();
        for ep in 0..train_eps {
            let len = 1 + (ep % 8);
            let (gs, ys) = gen_word(&mut rng_tr, len);
            lru.reset_state();
            for (&g, &y) in gs.iter().zip(ys.iter()) {
                let mut x = [0.0_f32; 4];
                x[g] = 1.0;
                let _ = lru.predict_and_train(&x, y, true);
            }
        }
        r_lru.us.push((t0.elapsed().as_nanos() as f64 / 1000.0) / steps_n);
        for (li, &len_t) in eval_lens.iter().enumerate() {
            let (mut hits, mut ohits) = (0usize, 0usize);
            for _ in 0..eval_eps {
                let (gs, ys) = gen_word(&mut rng_ev, len_t);
                lru.reset_state();
                let mut pred = 0usize;
                for &g in &gs {
                    let mut x = [0.0_f32; 4];
                    x[g] = 1.0;
                    pred = lru.predict_and_train(&x, 0, false);
                }
                let y_true = ys[len_t - 1];
                if pred == y_true { hits += 1; }
                if a5.perms[pred][0] == a5.perms[y_true][0] { ohits += 1; }
            }
            r_lru.full[li].push(hits as f64 * 100.0 / eval_eps as f64);
            r_lru.orbit[li].push(ohits as f64 * 100.0 / eval_eps as f64);
        }
        lru.inject_register_bit_flip(0, 30);
        let _ = lru.predict_and_train(&[f32::NAN, 1e30, 0.0, 0.0], 0, false);
        if lru.had_nan {
            r_lru.seu = false;
        }

        // 4. Online Selective SSM / Mamba-RTRL (18 complex units = 36D)
        let mut rng_tr = splitmix64(seed ^ 0xA5_77A1);
        let mut rng_ev = splitmix64(seed ^ 0xA5_E7A1);
        let mut ssm = SelectiveSsmBank::new(4, 18, 60, seed);
        ssm.enable_input_clamp = false;
        r_ssm.ram = ssm.memory_footprint_bytes() as f64 / 1024.0;
        let t0 = Instant::now();
        for ep in 0..train_eps {
            let len = 1 + (ep % 8);
            let (gs, ys) = gen_word(&mut rng_tr, len);
            ssm.start_episode();
            for (&g, &y) in gs.iter().zip(ys.iter()) {
                let mut x = [0.0_f32; 4];
                x[g] = 1.0;
                let _ = ssm.step_and_learn_supervised(&x, Some(y), 0.05, 0.012);
            }
        }
        r_ssm.us.push((t0.elapsed().as_nanos() as f64 / 1000.0) / steps_n);
        for (li, &len_t) in eval_lens.iter().enumerate() {
            let (mut hits, mut ohits) = (0usize, 0usize);
            for _ in 0..eval_eps {
                let (gs, ys) = gen_word(&mut rng_ev, len_t);
                ssm.start_episode();
                let mut pred = 0usize;
                for &g in &gs {
                    let mut x = [0.0_f32; 4];
                    x[g] = 1.0;
                    pred = ssm.step_and_learn_supervised(&x, None, 0.0, 0.0);
                }
                let y_true = ys[len_t - 1];
                if pred == y_true { hits += 1; }
                if a5.perms[pred][0] == a5.perms[y_true][0] { ohits += 1; }
            }
            r_ssm.full[li].push(hits as f64 * 100.0 / eval_eps as f64);
            r_ssm.orbit[li].push(ohits as f64 * 100.0 / eval_eps as f64);
        }
        flip_bit30_precond(&mut ssm.h_re[0]);
        ssm.inject_weight_bit_flip(0, 30);
        let _ = ssm.step_and_learn_supervised(&[f32::NAN, 1e30, 0.0, 0.0], None, 0.0, 0.0);
        if ssm.had_nan || !ssm.h_re.iter().all(|v| v.is_finite()) {
            r_ssm.seu = false;
        }

        // 5. Supervised Online GRU + BPTT (H=32)
        let mut rng_tr = splitmix64(seed ^ 0xA5_77A1);
        let mut rng_ev = splitmix64(seed ^ 0xA5_E7A1);
        let mut gru = GruSequenceClassifier::new(4, 32, 60, seed);
        r_gru.ram = gru.memory_footprint_bytes() as f64 / 1024.0;
        let t0 = Instant::now();
        for ep in 0..train_eps {
            let len = 1 + (ep % 8);
            let (gs, ys) = gen_word(&mut rng_tr, len);
            gru.reset();
            for &g in &gs {
                let mut x = [0.0_f32; 4];
                x[g] = 1.0;
                let _ = gru.forward_step(&x);
            }
            gru.bptt_update(&ys, 0.004);
        }
        r_gru.us.push((t0.elapsed().as_nanos() as f64 / 1000.0) / steps_n);
        for (li, &len_t) in eval_lens.iter().enumerate() {
            let (mut hits, mut ohits) = (0usize, 0usize);
            for _ in 0..eval_eps {
                let (gs, ys) = gen_word(&mut rng_ev, len_t);
                gru.reset();
                let mut pred = 0usize;
                for &g in &gs {
                    let mut x = [0.0_f32; 4];
                    x[g] = 1.0;
                    pred = gru.forward_step(&x);
                }
                let y_true = ys[len_t - 1];
                if pred == y_true { hits += 1; }
                if a5.perms[pred][0] == a5.perms[y_true][0] { ohits += 1; }
            }
            r_gru.full[li].push(hits as f64 * 100.0 / eval_eps as f64);
            r_gru.orbit[li].push(ohits as f64 * 100.0 / eval_eps as f64);
        }
        flip_bit30_precond(&mut gru.h[0]);
        flip_bit30_precond(&mut gru.wz[0]);
        let _ = gru.forward_step(&[f32::NAN, 1e30, 0.0, 0.0]);
        if !gru.h.iter().all(|v| v.is_finite()) || !gru.wz.iter().all(|v| v.is_finite()) {
            r_gru.seu = false;
        }
    }

    println!("\n=== Suite 1 (v0.4.0): Non-Abelian Group A_5 <= SO(3) State Tracking & Length Generalization (Train L in 1..8, 10 seeds) ===");
    println!("Metric: 60-element A_5 Exact State / Projected 5-pt Orbit p_t(0) in {{0..4}}");
    println!(
        "{:<36} | {:<13} | {:<13} | {:<13} | {:<13} | {:<8} | {:<8} | {}",
        "Model", "L=6 (60 / 5)", "L=12 (60 / 5)", "L=18 (60 / 5)", "L=30 (60 / 5)", "RAM", "Lat.", "SEU"
    );
    println!("{}", "-".repeat(132));
    for (name, r) in [
        ("NonAbelian Cayley SO(3) Proto M=1", &r_nac_m1),
        ("NonAbelian Cayley SO(3) Proto M=4", &r_nac_proto),
        ("NonAbelian 3D Cayley (SO(3) Lin.)", &r_nac),
        ("Fixed-Axis Cayley-RTRL (v0.3.0)", &r_fix),
        ("Online Selective SSM (Mamba 36D)", &r_ssm),
        ("Online LRU-RTRL (Diagonal 36D)", &r_lru),
        ("Online GRU + BPTT (H=32)", &r_gru),
    ] {
        let f6 = mean_sd(&r.full[0]).0; let o6 = mean_sd(&r.orbit[0]).0;
        let f12 = mean_sd(&r.full[1]).0; let o12 = mean_sd(&r.orbit[1]).0;
        let f18 = mean_sd(&r.full[2]).0; let o18 = mean_sd(&r.orbit[2]).0;
        let f30 = mean_sd(&r.full[3]).0; let o30 = mean_sd(&r.orbit[3]).0;
        let um = mean_sd(&r.us).0;
        println!(
            "{:<36} | {:5.1}%/{:5.1}% | {:5.1}%/{:5.1}% | {:5.1}%/{:5.1}% | {:5.1}%/{:5.1}% | {:5.1} KB | {:4.2} µs | {}",
            name, f6, o6, f12, o12, f18, o18, f30, o30, r.ram, um, if r.seu { "YES" } else { "NO" }
        );
    }
    println!("{:<36} | {:5.1}%/{:5.1}% | {:5.1}%/{:5.1}% | {:5.1}%/{:5.1}% | {:5.1}%/{:5.1}% | {:5.1} KB | {:4.2} µs | {}",
        "Random Floor (1/60 / 1/5)", 1.7, 20.0, 1.7, 20.0, 1.7, 20.0, 1.7, 20.0, 0.0, 0.01, "YES");
    let (m1_30, s1_30) = mean_sd(&r_nac_m1.full[3]);
    let hi1 = r_nac_m1.full[3].iter().filter(|&&v| v >= 95.0).count();
    let max1 = r_nac_m1.full[3].iter().copied().fold(0.0_f64, f64::max);
    let (m30, s30) = mean_sd(&r_nac_proto.full[3]);
    let hi_cnt = r_nac_proto.full[3].iter().filter(|&&v| v >= 95.0).count();
    let max30 = r_nac_proto.full[3].iter().copied().fold(0.0_f64, f64::max);
    println!(
        "  [Suite 1 Proto M=1 (9D, 7.6 KB)]  L=30 Exact-60: {:.1} ± {:.1}% (max={:.1}%, seeds >=95%: {}/10)",
        m1_30, s1_30, max1, hi1
    );
    println!(
        "  [Suite 1 Proto M=4 (36D, 26.1 KB)] L=30 Exact-60: {:.1} ± {:.1}% (max={:.1}%, seeds >=95%: {}/10)",
        m30, s30, max30, hi_cnt
    );
}

// =============================================================================
// Suite 2: Hardware IEEE-754 Register Bit-Flip SEU on RepeatPrevious & Continual Modular Shift
// =============================================================================

#[derive(Clone, Copy)]
enum Suite2ModelKind {
    MicroCoreA4,
    DeepLruPh,
    DeepLruPhGuard,
    SelectiveSsmPh,
    SelectiveSsmPhGuard,
    GruPerStepPh,
    GruPerStepPhGuard,
    LruSingle,
    LruSingleGuard,
}

fn eval_suite2_task(modular_shift: bool) {
    let seeds: Vec<u64> = (1..=10).collect();
    let k_delay = 4usize;
    let tail_mean = |v: &[f64]| -> f64 {
        let k = v.len().min(30);
        v[v.len() - k..].iter().sum::<f64>() / (k as f64)
    };

    let kinds = [
        ("MicroContinualCore [A4] (v0.4.0)", Suite2ModelKind::MicroCoreA4),
        ("Deep 2L RTRRL-LRU + PH (+Clamp)", Suite2ModelKind::DeepLruPh),
        ("Deep 2L RTRRL-LRU + PH (+FinGuard)", Suite2ModelKind::DeepLruPhGuard),
        ("Online Selective SSM + PH (+Clamp)", Suite2ModelKind::SelectiveSsmPh),
        ("Online Selective SSM + PH (+FinGuard)", Suite2ModelKind::SelectiveSsmPhGuard),
        ("Online Per-Step GRU + PH (+Clamp)", Suite2ModelKind::GruPerStepPh),
        ("Online Per-Step GRU + PH (+FinGuard)", Suite2ModelKind::GruPerStepPhGuard),
        ("1L RTRRL-LRU Single (+Clamp)", Suite2ModelKind::LruSingle),
        ("1L RTRRL-LRU Single (+FinGuard)", Suite2ModelKind::LruSingleGuard),
    ];

    if !modular_shift {
        println!("\n=== Suite 2A (v0.4.0): Hardware IEEE-754 Register & Weight Bit-Flip SEU — RepeatPrevious K=4 (Input Clamp ON, 10 seeds) ===");
        println!(
            "{:<37} | {:<11} | {:<11} | {:<11} | {:<13} | {}",
            "Model", "Clean K=4", "State-SEU", "Weight-SEU", "Faults(St/Wt)", "Finite (St/Wt)"
        );
        println!("{}", "-".repeat(116));
    } else {
        println!("\n=== Suite 2B (v0.4.0): Hardware IEEE-754 Register & Weight Bit-Flip SEU — Continual Modular Shift K=4 (A->B->A, Input Clamp ON, 10 seeds) ===");
        println!(
            "{:<37} | {:<11} | {:<11} | {:<11} | {:<17} | {:<13} | {}",
            "Model", "Clean(A,B,C)", "State-SEU", "Weight-SEU", "Ret-C(Cl/St/Wt)", "Faults(St/Wt)", "Finite (St/Wt)"
        );
        println!("{}", "-".repeat(136));
    }

    for (name, kind) in kinds {
        let mut clean_acc = Vec::new();
        let mut st_seu_acc = Vec::new();
        let mut wt_seu_acc = Vec::new();
        let mut ret_c_cl = Vec::new();
        let mut ret_c_st = Vec::new();
        let mut ret_c_wt = Vec::new();
        let mut st_faults_logged = Vec::new();
        let mut wt_faults_logged = Vec::new();
        let mut st_finite_seeds = 0usize;
        let mut wt_finite_seeds = 0usize;

        for &seed in &seeds {
            for mode in 0..=2u8 {
                let mut cfg = MicroCoreConfig::numeric_stream(4, seed);
                cfg.half_sectors = 0;
                cfg.belief_lambda = 0.90;
                cfg.belief_gamma = 0.0;
                cfg.rotor_norm = 0.60;
                cfg.novelty_gate = 0.0;
                if modular_shift {
                    cfg.ph_delta = 0.14;
                    cfg.ph_threshold = 2.2;
                    cfg.ph_warmup = 25;
                    cfg.max_regimes = 4;
                } else {
                    cfg.ph_threshold = 99.0;
                    cfg.max_regimes = 2;
                }
                cfg.standardize_vec = false;
                cfg.trace_gamma = 0.0;
                cfg.linear_wd = 0.0001;
                cfg.bank_capacity = 0;
                cfg.obs_bank_capacity = 0;
                cfg.explore_ucb = 0.0;
                cfg.linear_mix = 1.0;
                cfg.linear_lr = 0.0;
                cfg.actor_critic_lr = 0.085;
                cfg.critic_lr = 0.16;
                cfg.entropy_beta = 0.003;
                cfg.rtrl_sectors = 16;
                cfg.rtrl_slow_sectors = 16;
                cfg.rtrl_max_in = 4;
                cfg.rtrl_lr = 0.012;
                cfg.rtrl_grid_mode = 1;
                cfg.rtrl_use_skip_x = true;
                cfg.rtrl_kappa = 0.65;

                let mut m_core = MicroContinualCore::new(cfg);
                m_core.encoder.freeze_to_i8();
                let use_guard = matches!(
                    kind,
                    Suite2ModelKind::DeepLruPhGuard
                        | Suite2ModelKind::SelectiveSsmPhGuard
                        | Suite2ModelKind::GruPerStepPhGuard
                        | Suite2ModelKind::LruSingleGuard
                );
                let ph_th = if modular_shift { 2.2 } else { 99.0 };
                let mut m_deep = DeepRtrrlLruPhBaseline::new(4, 24, 24, 4, 4, 0.14, ph_th, 25, seed);
                m_deep.set_hyperparams(0.0, 0.0, 0.075, 0.16, 0.012, 0.004);
                m_deep.set_state_weight_guard(use_guard);
                let mut m_ssm = OnlineSelectiveSsmPhBaseline::new(4, 48, 4, 4, 0.14, ph_th, 25, seed);
                m_ssm.set_hyperparams(0.0, 0.0, 0.075, 0.16, 0.012, 0.004);
                m_ssm.set_state_weight_guard(use_guard);
                let mut m_gru = OnlineGruPerStepPhBaseline::new(4, 32, 4, 4, 0.14, ph_th, 25, seed);
                m_gru.set_hyperparams(0.0, 0.0, 0.006);
                m_gru.set_state_weight_guard(use_guard);
                let mut m_1l = RtrrlLruBaseline::new(4, 48, 4, seed);
                m_1l.enable_input_clamp = true;
                m_1l.enable_state_weight_guard = use_guard;
                m_1l.gamma = 0.0; m_1l.trace_lambda = 0.0; m_1l.actor_lr = 0.08; m_1l.critic_lr = 0.16; m_1l.rtrl_lr = 0.012; m_1l.entropy_beta = 0.004;

                let phases: &[(usize, usize)] = if modular_shift {
                    &[(0, 140), (2, 140), (0, 80)]
                } else {
                    &[(0, 160)]
                };
                let mut rng = splitmix64(seed ^ if modular_shift { 0xBF17_9002 } else { 0xBF17_0001 });
                let ep_len = 24usize;
                let mut global_step = 0usize;
                let mut all_ep_accs = Vec::new();
                let mut phase_a_accs = Vec::new();
                let mut phase_b_accs = Vec::new();
                let mut phase_c_accs = Vec::new();
                let mut all_finite = true;

                for (p_idx, &(shift, cnt)) in phases.iter().enumerate() {
                    let mut p_accs = Vec::with_capacity(cnt);
                    for ep in 0..cnt {
                        let train = if modular_shift { true } else { ep < cnt - 30 };
                        match kind {
                            Suite2ModelKind::MicroCoreA4 => m_core.start_episode(),
                            Suite2ModelKind::DeepLruPh | Suite2ModelKind::DeepLruPhGuard => m_deep.start_episode(),
                            Suite2ModelKind::SelectiveSsmPh | Suite2ModelKind::SelectiveSsmPhGuard => m_ssm.start_episode(),
                            Suite2ModelKind::GruPerStepPh | Suite2ModelKind::GruPerStepPhGuard => m_gru.start_episode(),
                            Suite2ModelKind::LruSingle | Suite2ModelKind::LruSingleGuard => m_1l.start_episode(),
                        }
                        let mut hist = Vec::with_capacity(ep_len);
                        let (mut hits, mut eval_n, mut ep_ret) = (0usize, 0usize, 0.0_f32);
                        for t in 0..ep_len {
                            rng = splitmix64(rng);
                            let sym = ((rng >> 16) as usize) % 4;
                            hist.push(sym);
                            let mut obs = [0.0_f32; 4];
                            obs[sym] = 1.0;

                            if mode == 1 && global_step > 20 && (global_step % 25 == 0) {
                                let kidx = global_step / 25;
                                let precond = |v: &mut f32| {
                                    if !v.is_finite() || v.abs() >= 2.0 || v.abs() < 0.25 {
                                        *v = 1.25;
                                    }
                                };
                                match kind {
                                    Suite2ModelKind::MicroCoreA4 => {
                                        let nf = m_core.rtrl_state.u.len();
                                        let ns = m_core.rtrl_slow_state.u.len();
                                        let tot = (nf + ns).max(1);
                                        let ridx = kidx % tot;
                                        if ridx < nf {
                                            precond(&mut m_core.rtrl_state.u[ridx]);
                                        } else if ridx - nf < ns {
                                            precond(&mut m_core.rtrl_slow_state.u[ridx - nf]);
                                        }
                                        m_core.inject_register_bit_flip(ridx, 30);
                                    }
                                    Suite2ModelKind::DeepLruPh | Suite2ModelKind::DeepLruPhGuard => {
                                        let act = m_deep.active_regime();
                                        let nf = 2 * m_deep.banks[act].fast.units;
                                        let ns = 2 * m_deep.banks[act].slow.units;
                                        let ridx = kidx % (nf + ns).max(1);
                                        if ridx < nf {
                                            let u = ridx / 2;
                                            if (ridx & 1) == 0 { precond(&mut m_deep.banks[act].fast.h_re[u]); }
                                            else { precond(&mut m_deep.banks[act].fast.h_im[u]); }
                                        } else {
                                            let sr = ridx - nf;
                                            let u = sr / 2;
                                            if (sr & 1) == 0 { precond(&mut m_deep.banks[act].slow.h_re[u]); }
                                            else { precond(&mut m_deep.banks[act].slow.h_im[u]); }
                                        }
                                        m_deep.inject_register_bit_flip(ridx, 30);
                                    }
                                    Suite2ModelKind::SelectiveSsmPh | Suite2ModelKind::SelectiveSsmPhGuard => {
                                        let act = m_ssm.active_regime();
                                        let n = 2 * m_ssm.banks[act].units;
                                        let ridx = kidx % n.max(1);
                                        let u = ridx / 2;
                                        if (ridx & 1) == 0 { precond(&mut m_ssm.banks[act].h_re[u]); }
                                        else { precond(&mut m_ssm.banks[act].h_im[u]); }
                                        m_ssm.inject_register_bit_flip(ridx, 30);
                                    }
                                    Suite2ModelKind::GruPerStepPh | Suite2ModelKind::GruPerStepPhGuard => {
                                        let act = m_gru.active_regime();
                                        let ridx = kidx % m_gru.banks[act].gru.h_cur.len().max(1);
                                        precond(&mut m_gru.banks[act].gru.h_cur[ridx]);
                                        m_gru.inject_register_bit_flip(ridx, 30);
                                    }
                                    Suite2ModelKind::LruSingle | Suite2ModelKind::LruSingleGuard => {
                                        let ridx = kidx % (2 * m_1l.units).max(1);
                                        let u = ridx / 2;
                                        if (ridx & 1) == 0 { precond(&mut m_1l.h_re[u]); }
                                        else { precond(&mut m_1l.h_im[u]); }
                                        m_1l.inject_register_bit_flip(ridx, 30);
                                    }
                                }
                            } else if mode == 2 && global_step > 20 && (global_step % 25 == 0) {
                                let kidx = global_step / 25;
                                match kind {
                                    Suite2ModelKind::MicroCoreA4 => m_core.inject_weight_bit_flip(kidx, 30),
                                    Suite2ModelKind::DeepLruPh | Suite2ModelKind::DeepLruPhGuard => m_deep.inject_weight_bit_flip(kidx, 30),
                                    Suite2ModelKind::SelectiveSsmPh | Suite2ModelKind::SelectiveSsmPhGuard => m_ssm.inject_weight_bit_flip(kidx, 30),
                                    Suite2ModelKind::GruPerStepPh | Suite2ModelKind::GruPerStepPhGuard => m_gru.inject_weight_bit_flip(kidx, 30),
                                    Suite2ModelKind::LruSingle | Suite2ModelKind::LruSingleGuard => m_1l.inject_weight_bit_flip(kidx, 30),
                                }
                            }

                            let pred = match kind {
                                Suite2ModelKind::MicroCoreA4 => {
                                    let pr = m_core.decide_vec(&obs, train, false, false);
                                    if !pr.get().1.is_finite()
                                        || !m_core.rtrl_state.u.iter().all(|v| v.is_finite())
                                        || !m_core.rtrl_state.b.iter().all(|v| v.is_finite())
                                        || !m_core.rtrl_slow_state.u.iter().all(|v| v.is_finite())
                                        || !m_core.rtrl_slow_state.b.iter().all(|v| v.is_finite())
                                    {
                                        all_finite = false;
                                    }
                                    pr.get().0
                                }
                                Suite2ModelKind::DeepLruPh | Suite2ModelKind::DeepLruPhGuard => m_deep.decide_vec(&obs, train, false, false),
                                Suite2ModelKind::SelectiveSsmPh | Suite2ModelKind::SelectiveSsmPhGuard => m_ssm.decide_vec(&obs, train, false, false),
                                Suite2ModelKind::GruPerStepPh | Suite2ModelKind::GruPerStepPhGuard => m_gru.decide_vec(&obs, train, false, false),
                                Suite2ModelKind::LruSingle | Suite2ModelKind::LruSingleGuard => m_1l.decide_vec(&obs, train, false, false),
                            };
                            global_step += 1;

                            if t >= k_delay {
                                let target_sym = if modular_shift {
                                    (hist[t - k_delay] + shift) % 4
                                } else {
                                    hist[t - k_delay]
                                };
                                let ok = pred == target_sym;
                                if ok { hits += 1; }
                                eval_n += 1;
                                let r = if ok { 1.0_f32 } else { -1.0_f32 };
                                match kind {
                                    Suite2ModelKind::MicroCoreA4 => m_core.observe_step_reward(r),
                                    Suite2ModelKind::DeepLruPh | Suite2ModelKind::DeepLruPhGuard => m_deep.observe_step_reward(r),
                                    Suite2ModelKind::SelectiveSsmPh | Suite2ModelKind::SelectiveSsmPhGuard => m_ssm.observe_step_reward(r),
                                    Suite2ModelKind::GruPerStepPh | Suite2ModelKind::GruPerStepPhGuard => m_gru.observe_step_reward(r),
                                    Suite2ModelKind::LruSingle | Suite2ModelKind::LruSingleGuard => m_1l.observe_step_reward(r),
                                }
                                ep_ret += r;
                            } else {
                                match kind {
                                    Suite2ModelKind::MicroCoreA4 => m_core.observe_step_reward(0.0),
                                    Suite2ModelKind::DeepLruPh | Suite2ModelKind::DeepLruPhGuard => m_deep.observe_step_reward(0.0),
                                    Suite2ModelKind::SelectiveSsmPh | Suite2ModelKind::SelectiveSsmPhGuard => m_ssm.observe_step_reward(0.0),
                                    Suite2ModelKind::GruPerStepPh | Suite2ModelKind::GruPerStepPhGuard => m_gru.observe_step_reward(0.0),
                                    Suite2ModelKind::LruSingle | Suite2ModelKind::LruSingleGuard => m_1l.observe_step_reward(0.0),
                                }
                            }
                        }
                        let norm_r = ((ep_ret / (eval_n.max(1) as f32)) + 1.0) * 0.5;
                        match kind {
                            Suite2ModelKind::MicroCoreA4 => { let _ = m_core.end_episode(norm_r * 2.0 - 1.0, train); }
                            Suite2ModelKind::DeepLruPh | Suite2ModelKind::DeepLruPhGuard => m_deep.end_episode_with_score(norm_r, false, train),
                            Suite2ModelKind::SelectiveSsmPh | Suite2ModelKind::SelectiveSsmPhGuard => m_ssm.end_episode_with_score(norm_r, false, train),
                            Suite2ModelKind::GruPerStepPh | Suite2ModelKind::GruPerStepPhGuard => m_gru.end_episode_with_score(norm_r, false, train),
                            Suite2ModelKind::LruSingle | Suite2ModelKind::LruSingleGuard => m_1l.end_episode(train),
                        }
                        let acc = hits as f64 * 100.0 / eval_n.max(1) as f64;
                        all_ep_accs.push(acc);
                        p_accs.push(acc);
                    }
                    match p_idx {
                        0 => phase_a_accs = p_accs,
                        1 => phase_b_accs = p_accs,
                        _ => phase_c_accs = p_accs,
                    }
                }

                let fin_score = if modular_shift {
                    (tail_mean(&phase_a_accs) + tail_mean(&phase_b_accs) + tail_mean(&phase_c_accs)) / 3.0
                } else {
                    tail_mean(&all_ep_accs)
                };
                let ret_c_val = if modular_shift && !phase_c_accs.is_empty() {
                    tail_mean(&phase_c_accs)
                } else {
                    0.0
                };

                if mode == 0 {
                    clean_acc.push(fin_score);
                    if modular_shift { ret_c_cl.push(ret_c_val); }
                } else {
                    let survived = match kind {
                        Suite2ModelKind::MicroCoreA4 => all_finite,
                        Suite2ModelKind::DeepLruPh | Suite2ModelKind::DeepLruPhGuard => !m_deep.had_nan(),
                        Suite2ModelKind::SelectiveSsmPh | Suite2ModelKind::SelectiveSsmPhGuard => !m_ssm.had_nan(),
                        Suite2ModelKind::GruPerStepPh | Suite2ModelKind::GruPerStepPhGuard => !m_gru.had_nan(),
                        Suite2ModelKind::LruSingle | Suite2ModelKind::LruSingleGuard => !m_1l.ever_had_nan,
                    };
                    let fc = match kind {
                        Suite2ModelKind::MicroCoreA4 => m_core.total_seu_faults() as f64,
                        Suite2ModelKind::DeepLruPh | Suite2ModelKind::DeepLruPhGuard => m_deep.total_seu_faults() as f64,
                        Suite2ModelKind::SelectiveSsmPh | Suite2ModelKind::SelectiveSsmPhGuard => m_ssm.total_seu_faults() as f64,
                        Suite2ModelKind::GruPerStepPh | Suite2ModelKind::GruPerStepPhGuard => m_gru.total_seu_faults() as f64,
                        Suite2ModelKind::LruSingle | Suite2ModelKind::LruSingleGuard => m_1l.total_seu_faults() as f64,
                    };
                    if mode == 1 {
                        st_seu_acc.push(fin_score);
                        if modular_shift { ret_c_st.push(ret_c_val); }
                        if survived { st_finite_seeds += 1; }
                        st_faults_logged.push(fc);
                    } else {
                        wt_seu_acc.push(fin_score);
                        if modular_shift { ret_c_wt.push(ret_c_val); }
                        if survived { wt_finite_seeds += 1; }
                        wt_faults_logged.push(fc);
                    }
                }
            }
        }

        let (cm, cs) = mean_sd(&clean_acc);
        let (sm, ss) = mean_sd(&st_seu_acc);
        let (wm, ws) = mean_sd(&wt_seu_acc);
        let sfm = mean_sd(&st_faults_logged).0;
        let wfm = mean_sd(&wt_faults_logged).0;
        if !modular_shift {
            println!(
                "{:<37} | {:5.1}±{:4.1}% | {:5.1}±{:4.1}% | {:5.1}±{:4.1}% | {:5.1}/{:5.1}   | {:2}/10 / {:2}/10",
                name, cm, cs, sm, ss, wm, ws, sfm, wfm, st_finite_seeds, wt_finite_seeds
            );
        } else {
            let rcc = mean_sd(&ret_c_cl).0;
            let rcs = mean_sd(&ret_c_st).0;
            let rcw = mean_sd(&ret_c_wt).0;
            println!(
                "{:<37} | {:5.1}±{:4.1}% | {:5.1}±{:4.1}% | {:5.1}±{:4.1}% | {:4.1}%/{:4.1}%/{:4.1}% | {:5.1}/{:5.1}   | {:2}/10 / {:2}/10",
                name, cm, cs, sm, ss, wm, ws, rcc, rcs, rcw, sfm, wfm, st_finite_seeds, wt_finite_seeds
            );
        }
    }
}

// =============================================================================
// Suite 3: Hard Continual RL & Chaotic Partial Observability vs Stronger Baselines
// =============================================================================

struct SimpleCartPole {
    x: f32,
    xdot: f32,
    theta: f32,
    theta_dot: f32,
    steps: usize,
}

impl SimpleCartPole {
    fn new(rng: &mut u64) -> Self {
        Self {
            x: rand_signed_f32(rng) * 0.04,
            xdot: rand_signed_f32(rng) * 0.04,
            theta: rand_signed_f32(rng) * 0.04,
            theta_dot: rand_signed_f32(rng) * 0.04,
            steps: 0,
        }
    }

    fn obs(&self) -> [f32; 2] {
        [self.x / 2.4, self.theta / 0.2095]
    }

    fn step(&mut self, action: usize, flipped: bool) -> bool {
        let g = 9.8_f32;
        let mc = 1.0_f32;
        let mp = 0.1_f32;
        let mt = mc + mp;
        let l = 0.5_f32;
        let mpl = mp * l;
        let tau = 0.02_f32;
        let raw_dir = if action == 1 { 1.0_f32 } else { -1.0_f32 };
        let force = 10.0 * if flipped { -raw_dir } else { raw_dir };
        let ct = self.theta.cos();
        let st = self.theta.sin();
        let tmp = (force + mpl * self.theta_dot * self.theta_dot * st) / mt;
        let th_acc = (g * st - ct * tmp) / (l * (4.0 / 3.0 - mp * ct * ct / mt));
        let x_acc = tmp - mpl * th_acc * ct / mt;
        self.x += tau * self.xdot;
        self.xdot += tau * x_acc;
        self.theta += tau * self.theta_dot;
        self.theta_dot += tau * th_acc;
        self.steps += 1;
        self.x.abs() > 2.4 || self.theta.abs() > 0.2095 || self.steps >= 200
    }
}

fn eval_suite3_hard_benchmarks() {
    let seeds: Vec<u64> = (1..=10).collect();
    let tail_40 = |v: &[f64]| v[v.len() - 40..].iter().sum::<f64>() / 40.0;

    // -------------------------------------------------------------------------
    // Task 3A: Multi-Cue Balanced Majority Corridor L=20 (A->B->A, noise +/-0.35)
    // -------------------------------------------------------------------------
    #[derive(Clone, Copy)]
    enum HardRlKind {
        MicroCoreRtrl,
        AlgSelCayley,
        MicroCore,
        DeepLruPh,
        SelectiveSsmPh,
        GruPerStepPh,
        Lru1LPh,
    }
    let models_3a = [
        ("MicroCore (Cayley-RTRL, gate=0.0)", HardRlKind::MicroCoreRtrl),
        ("Alg. Sel. Cayley (0% exp, gate=0.0)", HardRlKind::AlgSelCayley),
        ("MicroCore [Band-0 + gate=0.28 Abl.]", HardRlKind::MicroCore),
        ("Deep 2L RTRRL-LRU + PH + Clamp", HardRlKind::DeepLruPh),
        ("Online Selective SSM + PH + Clamp", HardRlKind::SelectiveSsmPh),
        ("Online Per-Step GRU + PH + Clamp", HardRlKind::GruPerStepPh),
        ("1L RTRRL-LRU + Matched PH + Clamp", HardRlKind::Lru1LPh),
    ];

    println!("\n=== Suite 3A (v0.4.0): Multi-Cue Balanced Majority Corridor L=20 (A->B->A, noise +/-0.35, 10 seeds) ===");
    println!("{:<37} | {:<13} | {:<13} | {}", "Model", "Majority Sc", "Majority Ret", "RAM");
    println!("{}", "-".repeat(82));

    for (name, kind) in models_3a {
        let mut scores = Vec::new();
        let mut rets = Vec::new();
        let mut ram_kb = 0.0;

        for &seed in &seeds {
            let mut cfg = MicroCoreConfig::numeric_stream(2, seed);
            cfg.half_sectors = 32;
            cfg.belief_lambda = 0.96;
            cfg.belief_gamma = 0.45;
            cfg.rotor_norm = 0.08;
            cfg.novelty_gate = 0.28;
            cfg.linear_lr = 0.20;
            cfg.linear_mix = 0.65;
            cfg.bank_capacity = 48;
            cfg.obs_bank_capacity = 16;
            cfg.explore_ucb = 0.35;
            cfg.trace_gamma = 0.96;
            cfg.rtrl_sectors = 0;
            cfg.ph_delta = 0.18;
            cfg.ph_threshold = 2.6;
            cfg.ph_warmup = 30;
            let mut core = MicroContinualCore::new(cfg);
            core.encoder.freeze_to_i8();

            let mut cfg_r = MicroCoreConfig::numeric_stream(2, seed);
            cfg_r.half_sectors = 16;
            cfg_r.belief_lambda = 0.88;
            cfg_r.belief_gamma = 0.30;
            cfg_r.rotor_norm = 0.12;
            cfg_r.novelty_gate = 0.0;
            cfg_r.linear_lr = 0.0;
            cfg_r.linear_mix = 1.0;
            cfg_r.bank_capacity = 0;
            cfg_r.obs_bank_capacity = 0;
            cfg_r.explore_ucb = 0.0;
            cfg_r.trace_gamma = 0.96;
            cfg_r.actor_critic_lr = 0.055;
            cfg_r.critic_lr = 0.14;
            cfg_r.entropy_beta = 0.008;
            cfg_r.rtrl_sectors = 16;
            cfg_r.rtrl_slow_sectors = 12;
            cfg_r.rtrl_max_in = 3;
            cfg_r.rtrl_lr = 0.012;
            cfg_r.rtrl_grid_mode = 1;
            cfg_r.rtrl_use_skip_x = true;
            cfg_r.rtrl_kappa = 0.65;
            cfg_r.ph_delta = 0.18;
            cfg_r.ph_threshold = 2.6;
            cfg_r.ph_warmup = 30;
            let mut core_rtrl = MicroContinualCore::new(cfg_r);
            core_rtrl.encoder.freeze_to_i8();

            let mut deep = DeepRtrrlLruPhBaseline::new(3, 18, 14, 2, 4, 0.18, 2.6, 30, seed);
            deep.set_hyperparams(0.96, 0.85, 0.055, 0.14, 0.008, 0.008);
            deep.set_probe_accept_threshold(0.45);
            let mut ssm = OnlineSelectiveSsmPhBaseline::new(3, 32, 2, 4, 0.18, 2.6, 30, seed);
            ssm.set_hyperparams(0.96, 0.85, 0.055, 0.14, 0.008, 0.008);
            ssm.set_probe_accept_threshold(0.45);
            let mut gru = OnlineGruPerStepPhBaseline::new(3, 32, 2, 4, 0.18, 2.6, 30, seed);
            gru.set_hyperparams(0.96, 0.85, 0.005);
            gru.set_probe_accept_threshold(0.45);
            let mut lph = RtrrlLruPhBaseline::new(3, 24, 2, 4, 0.18, 2.6, 30, seed);
            lph.set_hyperparams(0.96, 0.85, 0.055, 0.14, 0.008, 0.008);
            lph.set_probe_accept_threshold(0.45);
            let mut asc = AlgebraicSelectiveCayley::new(3, 16, 2, true, seed);
            asc.enable_rl_mode();

            ram_kb = match kind {
                HardRlKind::MicroCoreRtrl => core_rtrl.memory_footprint_bytes(),
                HardRlKind::AlgSelCayley => asc.memory_footprint_bytes(),
                HardRlKind::MicroCore => core.memory_footprint_bytes(),
                HardRlKind::DeepLruPh => deep.memory_footprint_bytes(),
                HardRlKind::SelectiveSsmPh => ssm.memory_footprint_bytes(),
                HardRlKind::GruPerStepPh => gru.memory_footprint_bytes(),
                _ => lph.memory_footprint_bytes(),
            } as f64 / 1024.0;

            let mut rng = splitmix64(seed ^ 0x3A_7A1E);
            let mut phases_acc = Vec::new();
            for (flipped, cnt) in [(false, 140usize), (true, 140), (false, 80)] {
                let mut res = Vec::with_capacity(cnt);
                for _ in 0..cnt {
                    rng = splitmix64(rng);
                    let maj = if (rng & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
                    let opp_idx = ((rng >> 2) as usize) % 3;
                    let c1 = if opp_idx == 0 { -maj } else { maj };
                    let c2 = if opp_idx == 1 { -maj } else { maj };
                    let c3 = if opp_idx == 2 { -maj } else { maj };
                    let t2 = 5 + (((rng >> 8) as usize) % 5);  // [5..9]
                    let t3 = 12 + (((rng >> 12) as usize) % 5); // [12..16]
                    let base_maj = (c1 + c2 + c3) > 0.0;
                    let expected = if flipped { usize::from(!base_maj) } else { usize::from(base_maj) };

                    match kind {
                        HardRlKind::MicroCoreRtrl => core_rtrl.start_episode(),
                        HardRlKind::AlgSelCayley => asc.reset(),
                        HardRlKind::MicroCore => core.start_episode(),
                        HardRlKind::DeepLruPh => deep.start_episode(),
                        HardRlKind::SelectiveSsmPh => ssm.start_episode(),
                        HardRlKind::GruPerStepPh => gru.start_episode(),
                        _ => lph.start_episode(),
                    }
                    for t in 0..20usize {
                        let cue_sig = if t == 0 { c1 } else if t == t2 { c2 } else if t == t3 { c3 } else { 0.0 };
                        let nz = rand_signed_f32(&mut rng) * 0.35;
                        let obs = [cue_sig, 0.0, nz];
                        match kind {
                            HardRlKind::MicroCoreRtrl => { let _ = core_rtrl.decide_vec(&obs, true, false, false); core_rtrl.observe_step_reward(0.0); }
                            HardRlKind::AlgSelCayley => { let _ = asc.decide_rl(&obs, true); asc.observe_step_reward_rl(0.0); }
                            HardRlKind::MicroCore => { let _ = core.decide_vec(&obs, true, false, false); core.observe_step_reward(0.0); }
                            HardRlKind::DeepLruPh => { let _ = deep.decide_vec(&obs, true, false, false); deep.observe_step_reward(0.0); }
                            HardRlKind::SelectiveSsmPh => { let _ = ssm.decide_vec(&obs, true, false, false); ssm.observe_step_reward(0.0); }
                            HardRlKind::GruPerStepPh => { let _ = gru.decide_vec(&obs, true, false, false); gru.observe_step_reward(0.0); }
                            _ => { let _ = lph.decide_vec(&obs, true, false, false); lph.observe_step_reward(0.0); }
                        }
                    }
                    let obs_j = [0.0_f32, 1.0, 0.0];
                    let act = match kind {
                        HardRlKind::MicroCoreRtrl => core_rtrl.decide_vec(&obs_j, true, false, false).get().0,
                        HardRlKind::AlgSelCayley => asc.decide_rl(&obs_j, true),
                        HardRlKind::MicroCore => core.decide_vec(&obs_j, true, false, false).get().0,
                        HardRlKind::DeepLruPh => deep.decide_vec(&obs_j, true, false, false),
                        HardRlKind::SelectiveSsmPh => ssm.decide_vec(&obs_j, true, false, false),
                        HardRlKind::GruPerStepPh => gru.decide_vec(&obs_j, true, false, false),
                        _ => lph.decide_vec(&obs_j, true, false, false),
                    };
                    let ok = act == expected;
                    let r = if ok { 1.0_f32 } else { -1.0_f32 };
                    match kind {
                        HardRlKind::MicroCoreRtrl => { core_rtrl.observe_step_reward(r); let _ = core_rtrl.end_episode(r, true); }
                        HardRlKind::AlgSelCayley => { asc.end_episode_rl(r, true); }
                        HardRlKind::MicroCore => { core.observe_step_reward(r); let _ = core.end_episode(r, true); }
                        HardRlKind::DeepLruPh => { deep.observe_step_reward(r); deep.end_episode_with_score(if ok { 1.0 } else { 0.0 }, false, true); }
                        HardRlKind::SelectiveSsmPh => { ssm.observe_step_reward(r); ssm.end_episode_with_score(if ok { 1.0 } else { 0.0 }, false, true); }
                        HardRlKind::GruPerStepPh => { gru.observe_step_reward(r); gru.end_episode_with_score(if ok { 1.0 } else { 0.0 }, false, true); }
                        _ => { lph.observe_step_reward(r); lph.end_episode_with_score(if ok { 1.0 } else { 0.0 }, false, true); }
                    }
                    res.push(if ok { 100.0 } else { 0.0 });
                }
                phases_acc.push(res);
            }
            scores.push((tail_40(&phases_acc[0]) + tail_40(&phases_acc[1])) * 0.5);
            rets.push(tail_40(&phases_acc[2]));
        }
        let (sm, ss) = mean_sd(&scores);
        let (rm, rs) = mean_sd(&rets);
        println!("{:<37} | {:5.1}±{:4.1}% | {:5.1}±{:4.1}% | {:5.1} KB", name, sm, ss, rm, rs, ram_kb);
    }

    // -------------------------------------------------------------------------
    // Task 3B: Chaotic Lorenz-63 1D-Projection Regime-Switch Stream (RK4, K=8 ahead)
    // -------------------------------------------------------------------------
    #[derive(Clone, Copy)]
    enum LorenzKind {
        MicroCore,
        DeepLruPh,
        SelectiveSsmPh,
        GruPerStepPh,
        LruSingle,
    }
    let models_3b = [
        ("MicroContinualCore (+PH, 60D)", LorenzKind::MicroCore),
        ("Deep 2L RTRRL-LRU + PH + Clamp (60D)", LorenzKind::DeepLruPh),
        ("Online Selective SSM + PH (60D)", LorenzKind::SelectiveSsmPh),
        ("Online Per-Step GRU + PH (60D,BPTT4)", LorenzKind::GruPerStepPh),
        ("Online LRU-RTRL Single-Bank (60D)", LorenzKind::LruSingle),
    ];

    println!("\n=== Suite 3B (v0.4.0): Chaotic Lorenz-63 1D-Projection Stream (RK4, causal std, win=1, matched 60D, sign(x_{{t+8}}), dt=0.06, 2500 -> 2500 Inv -> 1000 Ret, 10 seeds) ===");
    println!("{:<38} | {:<13} | {:<13} | {:<8} | {}", "Model", "Stream Acc", "Return Acc", "KappaT", "RAM");
    println!("{}", "-".repeat(95));

    let rk4_step = |x: &mut f64, y: &mut f64, z: &mut f64| {
        let (sigma, rho, beta, dt) = (10.0_f64, 28.0_f64, 8.0 / 3.0, 0.02_f64);
        let deriv = |xx: f64, yy: f64, zz: f64| -> (f64, f64, f64) {
            (sigma * (yy - xx), xx * (rho - zz) - yy, xx * yy - beta * zz)
        };
        let (k1x, k1y, k1z) = deriv(*x, *y, *z);
        let (k2x, k2y, k2z) = deriv(*x + 0.5 * dt * k1x, *y + 0.5 * dt * k1y, *z + 0.5 * dt * k1z);
        let (k3x, k3y, k3z) = deriv(*x + 0.5 * dt * k2x, *y + 0.5 * dt * k2y, *z + 0.5 * dt * k2z);
        let (k4x, k4y, k4z) = deriv(*x + dt * k3x, *y + dt * k3y, *z + dt * k3z);
        *x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        *y += (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
        *z += (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    };

    for (name, kind) in models_3b {
        let mut all_accs = Vec::new();
        let mut ret_accs = Vec::new();
        let mut kappas = Vec::new();
        let mut ram_kb = 0.0_f64;

        for &seed in &seeds {
            let mut lx = 1.0_f64 + 0.35 * (seed as f64);
            let mut ly = -1.2_f64 + 0.15 * (seed as f64);
            let mut lz = 21.0_f64;
            for _ in 0..400 {
                rk4_step(&mut lx, &mut ly, &mut lz);
            }
            let total_n = 6000usize;
            let k_ahead = 8usize;
            let mut xs = Vec::with_capacity(total_n + k_ahead + 2);
            for _ in 0..(total_n + k_ahead + 2) {
                xs.push(lx as f32);
                for _ in 0..3usize {
                    rk4_step(&mut lx, &mut ly, &mut lz);
                }
            }

            // Строгое равенство размерности состояния: 10*3 (Band 1) + 10*3 (Band 2) = 60D
            let mut cfg = MicroCoreConfig::numeric_stream(2, seed);
            cfg.half_sectors = 0;
            cfg.rtrl_sectors = 10;
            cfg.rtrl_slow_sectors = 10;
            cfg.rtrl_max_in = 1;
            cfg.rtrl_lr = 0.012;
            cfg.rtrl_grid_mode = 1;
            cfg.rtrl_use_skip_x = true;
            cfg.standardize_vec = false;
            cfg.linear_lr = 0.06;
            cfg.ph_delta = 0.04;
            cfg.ph_threshold = 2.2;
            cfg.ph_warmup = 4;
            cfg.max_regimes = 4;
            let mut core = MicroContinualCore::new(cfg);
            core.encoder.freeze_to_i8();
            core.start_episode();

            let mut deep = DeepRtrrlLruPhBaseline::new(1, 16, 14, 2, 4, 0.04, 2.2, 4, seed);
            deep.set_probe_accept_threshold(0.60);
            let mut ssm = OnlineSelectiveSsmPhBaseline::new(1, 30, 2, 4, 0.04, 2.2, 4, seed);
            ssm.set_probe_accept_threshold(0.60);
            let mut gru = OnlineGruPerStepPhBaseline::new_supervised(1, 60, 2, 4, 4, 0.04, 2.2, 4, seed);
            gru.set_hyperparams(0.95, 0.85, 0.004);
            gru.set_probe_accept_threshold(0.60);
            let mut lru1 = OnlineLruStream::new(1, 30, 2, seed);
            lru1.standardize = false;
            lru1.lr = 0.015;
            lru1.rtrl_lr = 0.008;

            ram_kb = match kind {
                LorenzKind::MicroCore => core.memory_footprint_bytes(),
                LorenzKind::DeepLruPh => deep.memory_footprint_bytes(),
                LorenzKind::SelectiveSsmPh => ssm.memory_footprint_bytes(),
                LorenzKind::GruPerStepPh => gru.memory_footprint_bytes(),
                LorenzKind::LruSingle => lru1.memory_footprint_bytes(),
            } as f64 / 1024.0;

            let mut hits = 0usize;
            let mut persist_hits = 0usize;
            let mut ret_hits = 0usize;
            let mut wf_n = 0usize;
            let mut wf_mean = 0.0_f64;
            let mut wf_m2 = 0.0_f64;
            let mut prev_y = 1usize;

            for t in 0..total_n {
                wf_n += 1;
                let x_cur = xs[t] as f64;
                let delta = x_cur - wf_mean;
                wf_mean += delta / wf_n as f64;
                let delta2 = x_cur - wf_mean;
                wf_m2 += delta * delta2;
                let std_cur = if wf_n >= 2 { (wf_m2 / (wf_n - 1) as f64).sqrt().max(0.25) } else { 8.0 };
                let z_t = ((x_cur - wf_mean) / std_cur).clamp(-4.0, 4.0) as f32;
                let obs = [z_t];

                let inv = t >= 2500 && t < 5000;
                let y = if inv { usize::from(xs[t + k_ahead] < 0.0) } else { usize::from(xs[t + k_ahead] >= 0.0) };
                // Строго каузальная персистенция Бифета: прошлый наблюдаемый таргет y_{t-1} без утечки inv
                let y_persist = if t > 0 { prev_y } else { 1 };
                if y_persist == y { persist_hits += 1; }

                let pred = match kind {
                    LorenzKind::MicroCore => {
                        let p = core.decide_vec(&obs, true, false, false).get().0;
                        core.end_supervised_step(p, y);
                        p
                    }
                    LorenzKind::DeepLruPh => deep.step_supervised(&obs, y, 0.04, 0.01, true, 1),
                    LorenzKind::SelectiveSsmPh => ssm.step_supervised(&obs, y, 0.04, 0.01, true, 1),
                    LorenzKind::GruPerStepPh => gru.step_supervised(&obs, y, true, 1),
                    LorenzKind::LruSingle => lru1.predict_and_train(&obs, y, true),
                };
                if pred == y {
                    hits += 1;
                    if t >= 5000 { ret_hits += 1; }
                }
                prev_y = y;
            }
            let acc = hits as f64 * 100.0 / total_n as f64;
            let p_acc = persist_hits as f64 / total_n as f64;
            let kappa = ((acc / 100.0) - p_acc) / (1.0 - p_acc).max(1e-6);
            all_accs.push(acc);
            ret_accs.push(ret_hits as f64 * 100.0 / 1000.0);
            kappas.push(kappa);
        }
        let (am, asd) = mean_sd(&all_accs);
        let (rm, rsd) = mean_sd(&ret_accs);
        let km = mean_sd(&kappas).0;
        println!("{:<38} | {:5.1}±{:4.1}% | {:5.1}±{:4.1}% | {:+6.3} | {:5.1} KB", name, am, asd, rm, rsd, km, ram_kb);
    }

    // -------------------------------------------------------------------------
    // Task 3C: Position-Only CartPole 2D & +dx (A->B->A) — Full Apples-to-Apples & Frozen-Eval Audit
    // -------------------------------------------------------------------------
    #[derive(Clone, Copy)]
    enum CartPoleKind {
        MicroCore,
        MicroCorePure2L,
        DeepLruPh,
        DeepLruPhSpinor,
        Lru1LPh,
        Lru1LPhSpinor,
        SelectiveSsmPh,
        GruEpisodicBptt,
        GruPerStepPh,
        Lru1LSingle,
    }
    let models_3c = [
        ("MicroCore (B0+2L, +PH, 96D)", CartPoleKind::MicroCore),
        ("MicroCore PURE 2L+Skip (B0=0, 96D)", CartPoleKind::MicroCorePure2L),
        ("Deep 2L RTRRL-LRU + PH (96D)", CartPoleKind::DeepLruPh),
        ("Deep 2L LRU + PH (+Spinor 48D)", CartPoleKind::DeepLruPhSpinor),
        ("1L RTRRL-LRU + PH (96D)", CartPoleKind::Lru1LPh),
        ("1L LRU + PH (+Spinor 48D)", CartPoleKind::Lru1LPhSpinor),
        ("Online Selective SSM + PH (96D)", CartPoleKind::SelectiveSsmPh),
        ("Online Episodic GRU + A2C (32D)", CartPoleKind::GruEpisodicBptt),
        ("Online Per-Step GRU + PH (32D)", CartPoleKind::GruPerStepPh),
        ("1L RTRRL-LRU Single-Bank (96D)", CartPoleKind::Lru1LSingle),
    ];

    {
        let mut srng = splitmix64(12345);
        let (mut srnd, mut ssig, mut spd) = (0.0_f64, 0.0_f64, 0.0_f64);
        for _ in 0..500 {
            let mut e1 = SimpleCartPole::new(&mut srng);
            loop { srng = splitmix64(srng); if e1.step((srng & 1) as usize, false) { break; } }
            srnd += e1.steps as f64;
            let mut e2 = SimpleCartPole::new(&mut srng);
            loop { if e2.step(usize::from(e2.theta > 0.0), false) { break; } }
            ssig += e2.steps as f64;
            let mut e3 = SimpleCartPole::new(&mut srng);
            loop {
                let u = 0.1 * e3.x + 0.2 * e3.xdot + 1.0 * e3.theta + 0.25 * e3.theta_dot;
                if e3.step(usize::from(u > 0.0), false) { break; }
            }
            spd += e3.steps as f64;
        }
        println!("\n=== Suite 3C (v0.4.0): Position-Only CartPole A->B->A (Matched 96D + Raw/Spinor Controls + Frozen Eval, 10 seeds) ===");
        println!("Physics Sanity (500 eps): Random={:.1} st | Memoryless sign(theta)={:.1} st | Oracle 4D-PD={:.1} st", srnd / 500.0, ssig / 500.0, spd / 500.0);
    }
    println!(
        "{:<35} | {:<11} | {:<11} | {:<11} | {:<11} | {:<11} | {:<11} | {}",
        "Model", "2D TrSc", "2D TrRet", "2D Ev(S/R)", "+dx TrSc", "+dx TrRet", "+dx Ev(S/R)", "RAM(4D)"
    );
    println!("{}", "-".repeat(130));

    for (name, kind) in models_3c {
        let mut s2d = Vec::new(); let mut r2d = Vec::new();
        let mut ev2s = Vec::new(); let mut ev2r = Vec::new();
        let mut sdx = Vec::new(); let mut rdx = Vec::new();
        let mut evds = Vec::new(); let mut evdr = Vec::new();
        let mut ram_kb = 0.0_f64;
        let use_spinor = matches!(kind, CartPoleKind::DeepLruPhSpinor | CartPoleKind::Lru1LPhSpinor);
        let pure_2l = matches!(kind, CartPoleKind::MicroCorePure2L);

        for use_dx in [false, true] {
            let in_dim = if use_dx { 4 } else { 2 };
            let lru_in = if use_spinor { 48 } else { in_dim };
            for &seed in &seeds {
                let enc = SpinorEncoder::new(8, 0);
                let mut enc_buf = vec![0.0_f32; 48];
                // Строгое равенство размерности состояния:
                // MicroCore: 8*6 (Band 0) + 10*3 (Band 1) + 6*3 (Band 2) = 96D
                // MicroCorePure2L: 0 (Band 0) + 16*3 (Band 1) + 16*3 (Band 2) = 96D
                let mut cfg = MicroCoreConfig::numeric_stream(2, seed);
                cfg.belief_lambda = 0.60; cfg.belief_gamma = if pure_2l { 0.0 } else { 0.30 };
                cfg.rotor_norm = 0.14; cfg.novelty_gate = 0.0; cfg.standardize_vec = false;
                cfg.delta_obs_scale = if use_dx { 10.0 } else { 0.0 };
                cfg.trace_capacity = 0; cfg.trace_gamma = 0.95; cfg.linear_lr = 0.0;
                cfg.linear_mix = 1.0; cfg.linear_wd = 0.0001; cfg.actor_critic_lr = 0.055;
                cfg.critic_lr = 0.14; cfg.entropy_beta = 0.008; cfg.bank_capacity = 0;
                cfg.obs_bank_capacity = 0; cfg.explore_ucb = 0.0; cfg.ph_delta = 0.12;
                cfg.ph_threshold = 1.8; cfg.ph_warmup = 28; cfg.max_regimes = 4;
                cfg.half_sectors = if pure_2l { 0 } else { 8 };
                cfg.rtrl_sectors = if pure_2l { 16 } else { 10 };
                cfg.rtrl_slow_sectors = if pure_2l { 16 } else { 6 };
                cfg.rtrl_max_in = in_dim; cfg.rtrl_lr = 0.006; cfg.rtrl_grid_mode = 1;
                cfg.rtrl_use_skip_x = pure_2l; cfg.rtrl_kappa = 0.5;
                let mut core = MicroContinualCore::new(cfg);
                core.encoder.freeze_to_i8();

                let mut deep = DeepRtrrlLruPhBaseline::new(lru_in, 24, 24, 2, 4, 0.12, 1.8, 28, seed);
                deep.set_hyperparams(0.95, 0.80, 0.055, 0.14, 0.006, 0.008);
                let mut ssm = OnlineSelectiveSsmPhBaseline::new(in_dim, 48, 2, 4, 0.12, 1.8, 28, seed);
                ssm.set_hyperparams(0.95, 0.80, 0.055, 0.14, 0.006, 0.008);
                let mut gru_ep = OnlineGruBaseline::new(in_dim, 32, 2, seed);
                gru_ep.lr = 0.008;
                gru_ep.gamma = 0.96;
                let mut gru = OnlineGruPerStepPhBaseline::new(in_dim, 32, 2, 4, 0.12, 1.8, 28, seed);
                gru.set_hyperparams(0.95, 0.80, 0.002);
                let mut lph = RtrrlLruPhBaseline::new(lru_in, 48, 2, 4, 0.12, 1.8, 28, seed);
                lph.set_hyperparams(0.95, 0.80, 0.055, 0.14, 0.006, 0.008);
                let mut l1b = RtrrlLruBaseline::new(in_dim, 48, 2, seed);
                l1b.enable_input_clamp = true;
                l1b.gamma = 0.95; l1b.trace_lambda = 0.80; l1b.actor_lr = 0.055; l1b.critic_lr = 0.14; l1b.rtrl_lr = 0.006; l1b.entropy_beta = 0.008;

                if use_dx {
                    ram_kb = match kind {
                        CartPoleKind::MicroCore | CartPoleKind::MicroCorePure2L => core.memory_footprint_bytes(),
                        CartPoleKind::DeepLruPh | CartPoleKind::DeepLruPhSpinor => deep.memory_footprint_bytes(),
                        CartPoleKind::SelectiveSsmPh => ssm.memory_footprint_bytes(),
                        CartPoleKind::GruEpisodicBptt => gru_ep.memory_footprint_bytes(),
                        CartPoleKind::GruPerStepPh => gru.memory_footprint_bytes(),
                        CartPoleKind::Lru1LPh | CartPoleKind::Lru1LPhSpinor => lph.memory_footprint_bytes(),
                        CartPoleKind::Lru1LSingle => l1b.memory_footprint_bytes(),
                    } as f64 / 1024.0;
                }

                let mut rng = splitmix64(seed ^ 0xCA27_901E);
                let mut phase_s = Vec::new();
                let mut phase_ev = Vec::new();
                for (flipped, cnt) in [(false, 140usize), (true, 140), (false, 80)] {
                    let mut surv = Vec::with_capacity(cnt);
                    for _ in 0..cnt {
                        let mut env = SimpleCartPole::new(&mut rng);
                        match kind {
                            CartPoleKind::MicroCore | CartPoleKind::MicroCorePure2L => core.start_episode(),
                            CartPoleKind::DeepLruPh | CartPoleKind::DeepLruPhSpinor => deep.start_episode(),
                            CartPoleKind::SelectiveSsmPh => ssm.start_episode(),
                            CartPoleKind::GruEpisodicBptt => gru_ep.start_episode(),
                            CartPoleKind::GruPerStepPh => gru.start_episode(),
                            CartPoleKind::Lru1LPh | CartPoleKind::Lru1LPhSpinor => lph.start_episode(),
                            CartPoleKind::Lru1LSingle => l1b.start_episode(),
                        }
                        let mut prev = [0.0_f32; 2];
                        let mut has_prev = false;
                        let mut ep_ret = 0.0_f32;
                        loop {
                            let o = env.obs();
                            let act = match kind {
                                CartPoleKind::MicroCore | CartPoleKind::MicroCorePure2L => core.decide_vec(&o, true, false, false).get().0,
                                _ => {
                                    let raw: Vec<f32> = if use_dx {
                                        let dx0 = if has_prev { (o[0] - prev[0]) * 10.0 } else { 0.0 };
                                        let dx1 = if has_prev { (o[1] - prev[1]) * 10.0 } else { 0.0 };
                                        prev = o; has_prev = true;
                                        vec![o[0], o[1], dx0.clamp(-4.0, 4.0), dx1.clamp(-4.0, 4.0)]
                                    } else {
                                        vec![o[0], o[1]]
                                    };
                                    let feat: &[f32] = if use_spinor {
                                        enc.encode_vec_into(&raw, &mut enc_buf);
                                        &enc_buf
                                    } else {
                                        &raw
                                    };
                                    match kind {
                                        CartPoleKind::DeepLruPh | CartPoleKind::DeepLruPhSpinor => deep.decide_vec(feat, true, false, false),
                                        CartPoleKind::SelectiveSsmPh => ssm.decide_vec(feat, true, false, false),
                                        CartPoleKind::GruEpisodicBptt => gru_ep.decide_vec(feat, true, false, false),
                                        CartPoleKind::GruPerStepPh => gru.decide_vec(feat, true, false, false),
                                        CartPoleKind::Lru1LPh | CartPoleKind::Lru1LPhSpinor => lph.decide_vec(feat, true, false, false),
                                        CartPoleKind::Lru1LSingle => l1b.decide_vec(feat, true, false, false),
                                        CartPoleKind::MicroCore | CartPoleKind::MicroCorePure2L => 0,
                                    }
                                }
                            };
                            let done = env.step(act, flipped);
                            let step_r = if done && env.steps < 200 { -1.0_f32 } else { 0.06_f32 };
                            ep_ret += step_r;
                            match kind {
                                CartPoleKind::MicroCore | CartPoleKind::MicroCorePure2L => core.observe_step_reward(step_r),
                                CartPoleKind::DeepLruPh | CartPoleKind::DeepLruPhSpinor => deep.observe_step_reward(step_r),
                                CartPoleKind::SelectiveSsmPh => ssm.observe_step_reward(step_r),
                                CartPoleKind::GruEpisodicBptt => gru_ep.observe_step_reward(step_r),
                                CartPoleKind::GruPerStepPh => gru.observe_step_reward(step_r),
                                CartPoleKind::Lru1LPh | CartPoleKind::Lru1LPhSpinor => lph.observe_step_reward(step_r),
                                CartPoleKind::Lru1LSingle => l1b.observe_step_reward(step_r),
                            }
                            if done { break; }
                        }
                        let trunc = env.steps >= 200;
                        let norm_01 = ((env.steps as f32) / 70.0).clamp(0.0, 1.0);
                        match kind {
                            CartPoleKind::MicroCore | CartPoleKind::MicroCorePure2L => { let _ = core.end_episode_truncated(norm_01 * 2.0 - 1.0, trunc, true); }
                            CartPoleKind::DeepLruPh | CartPoleKind::DeepLruPhSpinor => deep.end_episode_with_score(norm_01, trunc, true),
                            CartPoleKind::SelectiveSsmPh => ssm.end_episode_with_score(norm_01, trunc, true),
                            CartPoleKind::GruEpisodicBptt => gru_ep.end_episode_truncated(ep_ret, trunc, true),
                            CartPoleKind::GruPerStepPh => gru.end_episode_with_score(norm_01, trunc, true),
                            CartPoleKind::Lru1LPh | CartPoleKind::Lru1LPhSpinor => lph.end_episode_with_score(norm_01, trunc, true),
                            CartPoleKind::Lru1LSingle => l1b.end_episode_truncated(true, trunc),
                        }
                        surv.push(env.steps as f64);
                    }
                    phase_s.push(surv);
                    // Замороженная детерминированная оценка политики (30 эп., train_mode = false)
                    let mut erng = splitmix64(rng ^ 0xE7A1_0001);
                    let mut esum = 0.0_f64;
                    for _ in 0..30usize {
                        let mut env = SimpleCartPole::new(&mut erng);
                        match kind {
                            CartPoleKind::MicroCore | CartPoleKind::MicroCorePure2L => core.start_episode(),
                            CartPoleKind::DeepLruPh | CartPoleKind::DeepLruPhSpinor => deep.start_episode(),
                            CartPoleKind::SelectiveSsmPh => ssm.start_episode(),
                            CartPoleKind::GruEpisodicBptt => gru_ep.start_episode(),
                            CartPoleKind::GruPerStepPh => gru.start_episode(),
                            CartPoleKind::Lru1LPh | CartPoleKind::Lru1LPhSpinor => lph.start_episode(),
                            CartPoleKind::Lru1LSingle => l1b.start_episode(),
                        }
                        let mut prev = [0.0_f32; 2];
                        let mut has_prev = false;
                        loop {
                            let o = env.obs();
                            let act = match kind {
                                CartPoleKind::MicroCore | CartPoleKind::MicroCorePure2L => core.decide_vec(&o, false, false, false).get().0,
                                _ => {
                                    let raw: Vec<f32> = if use_dx {
                                        let dx0 = if has_prev { (o[0] - prev[0]) * 10.0 } else { 0.0 };
                                        let dx1 = if has_prev { (o[1] - prev[1]) * 10.0 } else { 0.0 };
                                        prev = o; has_prev = true;
                                        vec![o[0], o[1], dx0.clamp(-4.0, 4.0), dx1.clamp(-4.0, 4.0)]
                                    } else {
                                        vec![o[0], o[1]]
                                    };
                                    let feat: &[f32] = if use_spinor {
                                        enc.encode_vec_into(&raw, &mut enc_buf);
                                        &enc_buf
                                    } else {
                                        &raw
                                    };
                                    match kind {
                                        CartPoleKind::DeepLruPh | CartPoleKind::DeepLruPhSpinor => deep.decide_vec(feat, false, false, false),
                                        CartPoleKind::SelectiveSsmPh => ssm.decide_vec(feat, false, false, false),
                                        CartPoleKind::GruEpisodicBptt => gru_ep.decide_vec(feat, false, false, false),
                                        CartPoleKind::GruPerStepPh => gru.decide_vec(feat, false, false, false),
                                        CartPoleKind::Lru1LPh | CartPoleKind::Lru1LPhSpinor => lph.decide_vec(feat, false, false, false),
                                        CartPoleKind::Lru1LSingle => l1b.decide_vec(feat, false, false, false),
                                        CartPoleKind::MicroCore | CartPoleKind::MicroCorePure2L => 0,
                                    }
                                }
                            };
                            if env.step(act, flipped) { break; }
                        }
                        esum += env.steps as f64;
                    }
                    phase_ev.push(esum / 30.0);
                }
                let sc = (tail_40(&phase_s[0]) + tail_40(&phase_s[1])) * 0.5;
                let rt = tail_40(&phase_s[2]);
                let ev_sc = (phase_ev[0] + phase_ev[1]) * 0.5;
                let ev_rt = phase_ev[2];
                if use_dx {
                    sdx.push(sc); rdx.push(rt); evds.push(ev_sc); evdr.push(ev_rt);
                } else {
                    s2d.push(sc); r2d.push(rt); ev2s.push(ev_sc); ev2r.push(ev_rt);
                }
            }
        }
        let (s2m, s2s) = mean_sd(&s2d); let (r2m, r2s) = mean_sd(&r2d);
        let (sdm, sds) = mean_sd(&sdx); let (rdm, rds) = mean_sd(&rdx);
        println!(
            "{:<35} | {:5.1}±{:4.1} | {:5.1}±{:4.1} | {:5.1}/{:5.1} | {:5.1}±{:4.1} | {:5.1}±{:4.1} | {:5.1}/{:5.1} | {:5.1}KB",
            name, s2m, s2s, r2m, r2s, mean_sd(&ev2s).0, mean_sd(&ev2r).0,
            sdm, sds, rdm, rds, mean_sd(&evds).0, mean_sd(&evdr).0, ram_kb
        );
    }
}

fn main() {
    eval_a5_suite();
    eval_suite2_task(false);
    eval_suite2_task(true);
    eval_suite3_hard_benchmarks();
}
