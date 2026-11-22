// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov & Sofia Sidorenko
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Benchmark Suite Step 6 (v0.4.0 — 5-Axis Architectural & Systemic Scalability Suite):
//!   Axis 1: Width Scaling Laws (S in {4..256}, D=3S in {12..768}):
//!           1A. Compute & Memory O(D) Cayley-RTRL vs O(D^2) Online GRU + power-law exponents
//!           1B. Accuracy Scaling Law on A_5 (L=30), RepeatPrevious (K=8), Lorenz-63 (K=8)
//!   Axis 2: Depth Scaling Without BPTT (1L -> 2L -> 3L -> 4L local RTRL + DFA/residual):
//!           2A. Symmetric Group S_5 (120 permutations, odd + even generators, Eval L=10)
//!           2B. Hierarchical Bracket Matching Dyck-2 (nesting depth 3..5)
//!           2C. High-Dimensional Spatiotemporal Chaos Lorenz-96 (N=16, F=8.0, sparse obs)
//!   Axis 3: Horizon / Temporal Extrapolation Scaling:
//!           3A. A_5 <= SO(3) Ultra-Long Extrapolation (Train L<=8 -> Eval L=6..500) + f32 drift
//!           3B. RepeatPrevious Deep Delay Horizon K in {2, 4, 8, 12, 16, 24}
//!   Axis 4: Input Dimensionality & Continual Regime Count Scaling:
//!           4A. Lorenz-96 N=d_in in {4, 16, 32, 64, 128, 256}: Direct Cayley vs Count-Sketch vs GRU
//!           4B. R in {2, 4, 8, 16, 32} Regimes: Cyclical Return Retention, RAM(R), Latency(R)
//!   Axis 5: Systemic Swarm Scaling (N in {1..10000} agents) & L1/L2/L3 Cache Boundary Profile
//!           + Multi-Core Throughput (Million Agent-Steps / sec) vs Online SSM, LRU, GRU

use std::time::Instant;

use micro_continual_core::baselines::{GruPerStepBank, SelectiveSsmBank};
use micro_continual_core::core::{MicroContinualCore, MicroCoreConfig};
use micro_continual_core::math::{
    fast_dot, fast_saxpy, rand_signed_f32, rational_probs_and_psi_into, splitmix64,
};
use micro_continual_core::rtrl::{
    AlgebraicSelectiveCayley, NonAbelianCayleyRtrl, PlasticCayleyParams, PlasticCayleyState,
};
use micro_continual_core::stream_baselines::OnlineLruStream;

fn mean_sd(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    let var = xs.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / xs.len() as f64;
    (m, var.sqrt())
}

fn loglog_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0;
    }
    let lxs: Vec<f64> = xs[..n].iter().map(|&v| v.max(1e-9).ln()).collect();
    let lys: Vec<f64> = ys[..n].iter().map(|&v| v.max(1e-9).ln()).collect();
    let mx = lxs.iter().sum::<f64>() / n as f64;
    let my = lys.iter().sum::<f64>() / n as f64;
    let num: f64 = lxs.iter().zip(lys.iter()).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = lxs.iter().map(|x| (x - mx) * (x - mx)).sum();
    if den.abs() < 1e-12 {
        0.0
    } else {
        num / den
    }
}

// ============================================================================
// Supervised Single-Layer Plastic Cayley-RTRL Block (used in Axes 1, 2, 3B, 4A, 5)
// ============================================================================
struct CayleyBlockSupervised {
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

impl CayleyBlockSupervised {
    fn new(
        sectors: usize,
        in_dim: usize,
        num_classes: usize,
        kappa: f32,
        band_mode: u8,
        alpha_scale: f32,
        seed: u64,
    ) -> Self {
        let mut params = PlasticCayleyParams::new_with_grid(sectors, in_dim, seed, band_mode);
        for a in &mut params.alpha {
            *a = (*a * alpha_scale).clamp(0.03, 2.2);
        }
        let state = PlasticCayleyState::new(sectors, in_dim, kappa);
        let d = params.dim();
        let mut rng = splitmix64(seed ^ 0xCA71_E601);
        let scale = (1.0 / (d.max(1) as f32).sqrt()).clamp(0.02, 0.20);
        let out_w: Vec<f32> = (0..num_classes * d)
            .map(|_| rand_signed_f32(&mut rng) * scale)
            .collect();
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

    #[allow(dead_code)]
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

    #[inline]
    fn compute_logits(&mut self) {
        let d = self.params.dim();
        for c in 0..self.num_classes {
            let off = c * d;
            self.logits[c] =
                self.out_b[c] + fast_dot(&self.out_w[off..off + d], &self.state.b);
        }
    }

    fn step_forward_logits(&mut self, raw_x: &[f32]) {
        self.state.step_forward(&self.params, raw_x);
        self.compute_logits();
    }

    fn apply_target_err(
        &mut self,
        probs: &[f32],
        psis: &[f32],
        y: usize,
        ro_lr: f32,
        rtrl_lr: f32,
        trace_decay: f32,
    ) {
        let d = self.params.dim();
        self.grad_b.fill(0.0);
        for c in 0..self.num_classes {
            let tgt = if c == y { 1.0 } else { 0.0 };
            let err_c = (psis[c] * (tgt - probs[c])).clamp(-2.0, 2.0);
            let off = c * d;
            fast_saxpy(&mut self.grad_b, &self.out_w[off..off + d], err_c);
            for i in 0..d {
                let g = err_c * self.state.b[i];
                let nv = 0.98 * self.v_out_w[off + i] + 0.02 * g * g;
                self.v_out_w[off + i] = nv;
                let st = (g / (nv + 0.0025).sqrt()).clamp(-3.0, 3.0);
                self.out_w[off + i] =
                    (self.out_w[off + i] * 0.99995 + ro_lr * st).clamp(-8.0, 8.0);
            }
            let nvb = 0.98 * self.v_out_b[c] + 0.02 * err_c * err_c;
            self.v_out_b[c] = nvb;
            let stb = (err_c / (nvb + 0.0025).sqrt()).clamp(-3.0, 3.0);
            self.out_b[c] = (self.out_b[c] + 0.25 * ro_lr * stb).clamp(-4.0, 4.0);
        }
        if rtrl_lr > 0.0 {
            self.state
                .accumulate_traces(&self.grad_b, trace_decay, self.in_dim);
            self.state
                .apply_td_update(&mut self.params, 1.0, rtrl_lr, self.in_dim);
        }
    }

    fn step_and_learn(
        &mut self,
        raw_x: &[f32],
        target: Option<usize>,
        ro_lr: f32,
        rtrl_lr: f32,
        trace_decay: f32,
    ) -> usize {
        self.step_forward_logits(raw_x);
        let mut best_c = 0usize;
        let mut best_l = f32::NEG_INFINITY;
        for c in 0..self.num_classes {
            if self.logits[c] > best_l {
                best_l = self.logits[c];
                best_c = c;
            }
        }
        rational_probs_and_psi_into(&self.logits, &mut self.probs, &mut self.psis);
        if let Some(y) = target {
            let probs = std::mem::take(&mut self.probs);
            let psis = std::mem::take(&mut self.psis);
            self.apply_target_err(&probs, &psis, y, ro_lr, rtrl_lr, trace_decay);
            self.probs = probs;
            self.psis = psis;
        }
        best_c
    }
}

// ============================================================================
// Deep Compositional Local-RTRL Stack (Zero Inter-Layer BPTT — Axis 2)
// ============================================================================
struct DeepCayleyStack {
    layers: Vec<CayleyBlockSupervised>,
    raw_dim: usize,
    bridge_dim: usize,
    num_classes: usize,
    cum_logits: Vec<f32>,
    probs: Vec<f32>,
    psis: Vec<f32>,
    bridge_buf: Vec<f32>,
    in_buf: Vec<f32>,
}

impl DeepCayleyStack {
    fn new(
        depth: usize,
        sectors_per_layer: usize,
        raw_dim: usize,
        bridge_dim: usize,
        num_classes: usize,
        alpha_scale: f32,
        seed: u64,
    ) -> Self {
        let mut layers = Vec::with_capacity(depth);
        for l in 0..depth {
            let in_d = if l == 0 {
                raw_dim
            } else {
                raw_dim + bridge_dim
            };
            let b_mode = if l == 0 { 1 } else if l == 1 { 0 } else { 2 };
            let a_sc = if l == 0 {
                alpha_scale * 1.15
            } else {
                alpha_scale * (0.85_f32).powi(l as i32)
            };
            layers.push(CayleyBlockSupervised::new(
                sectors_per_layer,
                in_d,
                num_classes,
                0.65,
                b_mode,
                a_sc,
                seed ^ (0xD339_1000 + (l as u64) * 0x9E37_79B9),
            ));
        }
        Self {
            layers,
            raw_dim,
            bridge_dim,
            num_classes,
            cum_logits: vec![0.0; num_classes],
            probs: vec![0.0; num_classes],
            psis: vec![1.0; num_classes],
            bridge_buf: vec![0.0; bridge_dim],
            in_buf: vec![0.0; raw_dim + bridge_dim],
        }
    }

    fn reset(&mut self) {
        for l in &mut self.layers {
            l.reset();
        }
    }

    #[allow(dead_code)]
    fn memory_footprint_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.memory_footprint_bytes())
            .sum::<usize>()
            + (self.cum_logits.len()
                + self.probs.len()
                + self.psis.len()
                + self.bridge_buf.len()
                + self.in_buf.len())
                * 4
    }

    /// Извлекает нелинейный 8-мерный мост (средние проекции + спинорные парные произведения) без трансценденций.
    fn extract_bridge(prev_b: &[f32], bridge: &mut [f32]) {
        bridge.fill(0.0);
        if prev_b.is_empty() || bridge.is_empty() {
            return;
        }
        let d = prev_b.len();
        let half_br = bridge.len() / 2;
        for j in 0..half_br {
            let start = (j * d) / half_br.max(1);
            let end = (((j + 1) * d) / half_br.max(1)).max(start + 1).min(d);
            let mut sum = 0.0_f32;
            for idx in start..end {
                sum += prev_b[idx];
            }
            bridge[j] = (sum / ((end - start) as f32).sqrt()).clamp(-2.5, 2.5);
        }
        for j in half_br..bridge.len() {
            let a = (j * 3) % d;
            let b = (j * 5 + 1) % d;
            let c = (j * 7 + 2) % d;
            // Билинейное взаимодействие между секторами (критично для Lorenz-96 и вложенных стеков!)
            bridge[j] = (1.5 * prev_b[a] * prev_b[b] + 0.5 * prev_b[c]).clamp(-2.5, 2.5);
        }
    }

    fn step_and_learn(
        &mut self,
        raw_x: &[f32],
        target: Option<usize>,
        ro_lr: f32,
        rtrl_lr: f32,
        trace_decay: f32,
    ) -> usize {
        self.cum_logits.fill(0.0);
        let depth = self.layers.len();
        let inv_sqrt_d = 1.0_f32 / (depth.max(1) as f32).sqrt();

        // 1. Полный прямой проход слоёв 0..depth и объединённый глобальный логит (0 аллокаций!)
        for l in 0..depth {
            if l == 0 {
                self.layers[0].step_forward_logits(raw_x);
            } else {
                Self::extract_bridge(&self.layers[l - 1].state.b, &mut self.bridge_buf);
                self.in_buf.fill(0.0);
                for i in 0..self.raw_dim.min(raw_x.len()) {
                    self.in_buf[i] = raw_x[i];
                }
                for i in 0..self.bridge_dim {
                    self.in_buf[self.raw_dim + i] = self.bridge_buf[i];
                }
                self.layers[l].step_forward_logits(&self.in_buf);
            }
            for c in 0..self.num_classes {
                self.cum_logits[c] += inv_sqrt_d * self.layers[l].logits[c];
            }
        }

        let mut best_c = 0usize;
        let mut best_l = f32::NEG_INFINITY;
        for c in 0..self.num_classes {
            if self.cum_logits[c] > best_l {
                best_l = self.cum_logits[c];
                best_c = c;
            }
        }
        rational_probs_and_psi_into(&self.cum_logits, &mut self.probs, &mut self.psis);

        // 2. Прямое выравнивание обратной связи (DFA): единая глобальная ошибка идёт прямо в каждый слой
        //    без межслойного обратного распространения во времени (Zero Inter-Layer BPTT!)
        if let Some(y) = target {
            for l in 0..depth {
                self.layers[l].apply_target_err(
                    &self.probs,
                    &self.psis,
                    y,
                    ro_lr * inv_sqrt_d,
                    rtrl_lr,
                    trace_decay,
                );
            }
        }
        best_c
    }
}

// ============================================================================
// Deep Non-Abelian Stack (for S_5 = 120 elements, Zero Inter-Layer BPTT)
// ============================================================================
struct DeepNonAbelianStack {
    layers: Vec<NonAbelianCayleyRtrl>,
    bridge_dim: usize,
    raw_dim: usize,
    num_classes: usize,
    cum_logits: Vec<f32>,
    probs: Vec<f32>,
    psis: Vec<f32>,
    bridge_buf: Vec<f32>,
    in_buf: Vec<f32>,
}

impl DeepNonAbelianStack {
    fn new(
        depth: usize,
        sectors_per_layer: usize,
        raw_dim: usize,
        bridge_dim: usize,
        num_classes: usize,
        seed: u64,
    ) -> Self {
        let mut layers = Vec::with_capacity(depth);
        for l in 0..depth {
            let in_d = if l == 0 {
                raw_dim
            } else {
                raw_dim + bridge_dim
            };
            let mut nac = NonAbelianCayleyRtrl::new(
                sectors_per_layer,
                in_d,
                num_classes,
                0.25,
                seed ^ (0x55_1200_0000 + (l as u64) * 0x9E37_79B9),
            );
            nac.enable_so3_frames_end_to_end();
            layers.push(nac);
        }
        Self {
            layers,
            bridge_dim,
            raw_dim,
            num_classes,
            cum_logits: vec![0.0; num_classes],
            probs: vec![0.0; num_classes],
            psis: vec![1.0; num_classes],
            bridge_buf: vec![0.0; bridge_dim],
            in_buf: vec![0.0; raw_dim + bridge_dim],
        }
    }

    fn reset(&mut self) {
        for l in &mut self.layers {
            l.reset();
        }
    }

    #[allow(dead_code)]
    fn memory_footprint_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.memory_footprint_bytes())
            .sum::<usize>()
            + (self.cum_logits.len()
                + self.probs.len()
                + self.psis.len()
                + self.bridge_buf.len()
                + self.in_buf.len())
                * 4
    }

    fn step_and_learn(
        &mut self,
        raw_x: &[f32],
        target: Option<usize>,
        ro_lr: f32,
        rtrl_lr: f32,
    ) -> usize {
        self.cum_logits.fill(0.0);
        let depth = self.layers.len();

        for l in 0..depth {
            let ro = if target.is_some() {
                ro_lr / ((l + 1) as f32).sqrt()
            } else {
                0.0
            };
            let rl = if target.is_some() {
                rtrl_lr / ((l + 1) as f32).sqrt()
            } else {
                0.0
            };
            if l == 0 {
                let _ = self.layers[0].step_and_learn_supervised(raw_x, target, ro, rl);
            } else {
                DeepCayleyStack::extract_bridge(&self.layers[l - 1].b, &mut self.bridge_buf);
                self.in_buf.fill(0.0);
                for i in 0..self.raw_dim.min(raw_x.len()) {
                    self.in_buf[i] = raw_x[i];
                }
                for i in 0..self.bridge_dim {
                    self.in_buf[self.raw_dim + i] = self.bridge_buf[i];
                }
                let _ = self.layers[l].step_and_learn_supervised(&self.in_buf, target, ro, rl);
            }

            let d = self.layers[l].sectors * 3;
            for c in 0..self.num_classes {
                let off = c * d;
                let lg = self.layers[l].out_b[c]
                    + fast_dot(&self.layers[l].out_w[off..off + d], &self.layers[l].b);
                self.cum_logits[c] += lg;
            }
        }
        rational_probs_and_psi_into(&self.cum_logits, &mut self.probs, &mut self.psis);
        let mut best_c = 0usize;
        let mut best_l = f32::NEG_INFINITY;
        for c in 0..self.num_classes {
            if self.cum_logits[c] > best_l {
                best_l = self.cum_logits[c];
                best_c = c;
            }
        }
        best_c
    }
}

// ============================================================================
// Discrete Finite Groups A_5 (60 elems) and S_5 (120 elems)
// ============================================================================
struct PermGroup {
    perms: Vec<[u8; 5]>,
    even_flags: Vec<bool>,
    mul_gen: Vec<Vec<usize>>,
    ident_id: usize,
}

impl PermGroup {
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

    fn a5() -> Self {
        let gens: [[u8; 5]; 4] = [
            [1, 2, 0, 3, 4], // (0 1 2)
            [2, 0, 1, 3, 4], // (0 2 1)
            [1, 2, 3, 4, 0], // (0 1 2 3 4)
            [4, 0, 1, 2, 3], // (4 3 2 1 0)
        ];
        Self::build(true, &gens)
    }

    fn s5() -> Self {
        // S_5 образующие: (0 1), (1 2), (0 1 2 3 4), (4 3 2 1 0)
        let gens: [[u8; 5]; 4] = [
            [1, 0, 2, 3, 4], // (0 1) — нечётная транспозиция
            [0, 2, 1, 3, 4], // (1 2) — нечётная транспозиция
            [1, 2, 3, 4, 0], // (0 1 2 3 4) — 5-цикл
            [4, 0, 1, 2, 3], // (4 3 2 1 0) — обратный 5-цикл
        ];
        Self::build(false, &gens)
    }

    fn build(only_even: bool, gens: &[[u8; 5]]) -> Self {
        let mut perms = Vec::new();
        for a in 0..5u8 {
            for b in 0..5u8 {
                if b == a { continue; }
                for c in 0..5u8 {
                    if c == a || c == b { continue; }
                    for d in 0..5u8 {
                        if d == a || d == b || d == c { continue; }
                        let e = 10 - (a + b + c + d);
                        let p = [a, b, c, d, e];
                        if !only_even || Self::is_even(p) {
                            perms.push(p);
                        }
                    }
                }
            }
        }
        perms.sort();
        let even_flags: Vec<bool> = perms.iter().map(|&p| Self::is_even(p)).collect();
        let mut mul_gen = Vec::with_capacity(gens.len());
        for &g_perm in gens {
            let mut row = Vec::with_capacity(perms.len());
            for &cur_perm in &perms {
                let prod = Self::compose(cur_perm, g_perm);
                let nxt = perms.binary_search(&prod).unwrap();
                row.push(nxt);
            }
            mul_gen.push(row);
        }
        let ident_id = perms.binary_search(&[0, 1, 2, 3, 4]).unwrap();
        Self {
            perms,
            even_flags,
            mul_gen,
            ident_id,
        }
    }

    fn gen_word(&self, rng: &mut u64, len: usize) -> (Vec<usize>, Vec<usize>) {
        let mut gs = Vec::with_capacity(len);
        let mut st_ids = Vec::with_capacity(len);
        let mut cur = self.ident_id;
        let ng = self.mul_gen.len();
        for _ in 0..len {
            *rng = splitmix64(*rng);
            let g = ((*rng >> 16) as usize) % ng;
            cur = self.mul_gen[g][cur];
            gs.push(g);
            st_ids.push(cur);
        }
        (gs, st_ids)
    }
}

// ============================================================================
// GRU + BPTT Sequence Classifier (for Axis 3A Ultra-Long Horizon Baseline)
// ============================================================================
struct GruBpttSeq {
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

impl GruBpttSeq {
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

    #[allow(dead_code)]
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

    fn forward_step(&mut self, x: &[f32], record: bool) -> usize {
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
        self.h = hn.clone();
        if record {
            let mut probs = vec![0.0_f32; self.num_classes];
            let mut sum_e = 0.0_f32;
            for c in 0..self.num_classes {
                let e = (logits[c] - max_l).exp();
                probs[c] = e;
                sum_e += e;
            }
            for p in &mut probs { *p /= sum_e.max(1e-8); }
            self.cache.push((x.to_vec(), hp, z, r, n, hn, probs));
        }
        best_c
    }

    fn adam(w: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], lr: f32, t: usize) {
        let (b1, b2) = (0.9_f32, 0.999_f32);
        let c1 = 1.0 - b1.powi(t as i32);
        let c2 = 1.0 - b2.powi(t as i32);
        for i in 0..w.len() {
            let gc = g[i].clamp(-2.0, 2.0);
            m[i] = b1 * m[i] + (1.0 - b1) * gc;
            v[i] = b2 * v[i] + (1.0 - b2) * gc * gc;
            w[i] = (w[i] - lr * (m[i] / c1.max(1e-6)) / ((v[i] / c2.max(1e-6)).sqrt() + 1e-6))
                .clamp(-5.0, 5.0);
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
                for j in 0..hd {
                    grads[9][c * hd + j] += dl * hn[j];
                    dh[j] += self.wo[c * hd + j] * dl;
                }
            }
            let mut dr = vec![0.0_f32; hd];
            let mut dh_prev = vec![0.0_f32; hd];
            for i in 0..hd {
                let dhi = dh[i].clamp(-4.0, 4.0);
                dh_prev[i] += dhi * (1.0 - z[i]);
                let dsn = dhi * z[i] * (1.0 - n[i] * n[i]);
                let dsz = dhi * (n[i] - hp[i]) * z[i] * (1.0 - z[i]);
                grads[8][i] += dsn;
                grads[2][i] += dsz;
                for j in 0..xd {
                    grads[6][i * xd + j] += dsn * x[j];
                    grads[0][i * xd + j] += dsz * x[j];
                }
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
                for j in 0..hd {
                    grads[4][i * hd + j] += dsr * hp[j];
                    dh_prev[j] += dsr * self.ur[i * hd + j];
                }
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

// ============================================================================
// Dynamical Systems: Lorenz-63 & Coupled Spatiotemporal Lorenz-96 (RK4)
// ============================================================================
fn generate_lorenz63(n_steps: usize, k_ahead: usize, seed: u64) -> Vec<f32> {
    let mut lx = 1.0_f64 + 0.35 * (seed as f64);
    let mut ly = -1.2_f64 + 0.15 * (seed as f64);
    let mut lz = 21.0_f64;
    let (sigma, rho, beta, dt) = (10.0_f64, 28.0_f64, 8.0 / 3.0, 0.02_f64);
    let rk4_sub = |x: &mut f64, y: &mut f64, z: &mut f64| {
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
    for _ in 0..400 {
        rk4_sub(&mut lx, &mut ly, &mut lz);
    }
    let total = n_steps + k_ahead + 2;
    let mut xs = Vec::with_capacity(total);
    for _ in 0..total {
        xs.push((lx / 20.0) as f32);
        for _ in 0..3 {
            rk4_sub(&mut lx, &mut ly, &mut lz);
        }
    }
    xs
}

fn lorenz96_rhs(x: &[f64], f_force: f64, out: &mut [f64]) {
    let n = x.len();
    for i in 0..n {
        let ip1 = (i + 1) % n;
        let im1 = (i + n - 1) % n;
        let im2 = (i + n - 2) % n;
        out[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + f_force;
    }
}

fn generate_lorenz96(n_vars: usize, n_steps: usize, k_ahead: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut x = vec![8.0_f64; n_vars];
    let mut rng = splitmix64(seed ^ 0x96_102E);
    for v in &mut x {
        *v += 0.25 * rand_signed_f32(&mut rng) as f64;
    }
    let dt = 0.015_f64;
    let f_force = 8.0_f64;
    let mut k1 = vec![0.0; n_vars];
    let mut k2 = vec![0.0; n_vars];
    let mut k3 = vec![0.0; n_vars];
    let mut k4 = vec![0.0; n_vars];
    let mut tmp = vec![0.0; n_vars];

    let mut rk4_step = |state: &mut Vec<f64>| {
        lorenz96_rhs(state, f_force, &mut k1);
        for i in 0..n_vars { tmp[i] = state[i] + 0.5 * dt * k1[i]; }
        lorenz96_rhs(&tmp, f_force, &mut k2);
        for i in 0..n_vars { tmp[i] = state[i] + 0.5 * dt * k2[i]; }
        lorenz96_rhs(&tmp, f_force, &mut k3);
        for i in 0..n_vars { tmp[i] = state[i] + dt * k3[i]; }
        lorenz96_rhs(&tmp, f_force, &mut k4);
        for i in 0..n_vars {
            state[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    };

    for _ in 0..300 {
        rk4_step(&mut x);
    }
    let total = n_steps + k_ahead + 2;
    let mut traj = Vec::with_capacity(total);
    for _ in 0..total {
        let centered: Vec<f32> = x.iter().map(|&v| ((v - 2.3) / 6.0) as f32).collect();
        traj.push(centered);
        rk4_step(&mut x);
        rk4_step(&mut x);
    }
    traj
}

// ============================================================================
// Axis 1: Width Scaling Laws (1A Compute/RAM O(D) vs O(D^2) + 1B Accuracy Law)
// ============================================================================
fn run_axis1_width(sink: &mut usize) {
    println!("\n============================================================================================");
    println!("AXIS 1A: Width Compute & Memory Scaling Laws (S in 4..256, D=3S in 12..768 vs SSM / LRU / Sup-GRU)");
    println!("============================================================================================");
    println!(
        "{:<5} | {:<5} | {:<10} | {:<10} | {:<10} | {:<10} | {:<9} | {:<9} | {:<9} | {:<9}",
        "S", "Dim D", "Cayley RAM", "SSM RAM", "LRU RAM", "GRU RAM", "Cayley µs", "SSM µs", "LRU µs", "GRU µs"
    );
    println!("{}", "-".repeat(118));

    let s_list = [4usize, 8, 16, 32, 64, 128, 256];
    let d_in = 8usize;
    let nc = 4usize;

    let mut d_vec = Vec::new();
    let mut cay_ram = Vec::new();
    let mut ssm_ram = Vec::new();
    let mut lru_ram = Vec::new();
    let mut gru_ram = Vec::new();
    let mut cay_us = Vec::new();
    let mut ssm_us = Vec::new();
    let mut lru_us = Vec::new();
    let mut gru_us = Vec::new();

    for &s in &s_list {
        let d = s * 3;
        let units = d / 2; // 2*units == d
        let mut cay = CayleyBlockSupervised::new(s, d_in, nc, 0.65, 1, 0.85, 77 + s as u64);
        let mut ssm = SelectiveSsmBank::new(d_in, units, nc, 77 + s as u64);
        let mut lru = OnlineLruStream::new(d_in, units, nc, 77 + s as u64);
        let mut gru = GruPerStepBank::new_supervised(d_in, d, nc, 1, 77 + s as u64);

        let cram = cay.memory_footprint_bytes() as f64 / 1024.0;
        let sram = ssm.memory_footprint_bytes() as f64 / 1024.0;
        let lram = lru.memory_footprint_bytes() as f64 / 1024.0;
        let gram = gru.memory_footprint_bytes() as f64 / 1024.0;

        let n_steps = if d >= 384 {
            350usize
        } else if d >= 192 {
            800usize
        } else {
            2400usize
        };
        let x = [0.4_f32, -0.3, 0.7, 0.1, -0.5, 0.2, 0.6, -0.4];

        for t in 0..40usize {
            *sink ^= cay.step_and_learn(&x, Some(t % nc), 0.04, 0.01, 0.5);
            *sink ^= ssm.step_and_learn_supervised(&x, Some(t % nc), 0.04, 0.01);
            *sink ^= lru.predict_and_train(&x, t % nc, true);
            *sink ^= gru.step_and_learn_supervised(&x, Some(t % nc));
        }

        let t0 = Instant::now();
        for t in 0..n_steps {
            *sink ^= cay.step_and_learn(&x, Some(t % nc), 0.04, 0.01, 0.5);
        }
        let cus = (t0.elapsed().as_nanos() as f64 / 1000.0) / n_steps as f64;

        let ts = Instant::now();
        for t in 0..n_steps {
            *sink ^= ssm.step_and_learn_supervised(&x, Some(t % nc), 0.04, 0.01);
        }
        let sus = (ts.elapsed().as_nanos() as f64 / 1000.0) / n_steps as f64;

        let tl = Instant::now();
        for t in 0..n_steps {
            *sink ^= lru.predict_and_train(&x, t % nc, true);
        }
        let lus = (tl.elapsed().as_nanos() as f64 / 1000.0) / n_steps as f64;

        let t1 = Instant::now();
        for t in 0..n_steps {
            *sink ^= gru.step_and_learn_supervised(&x, Some(t % nc));
        }
        let gus = (t1.elapsed().as_nanos() as f64 / 1000.0) / n_steps as f64;

        println!(
            "{:<5} | {:<5} | {:>7.1} KB | {:>7.1} KB | {:>7.1} KB | {:>7.1} KB | {:>6.2} µs | {:>6.2} µs | {:>6.2} µs | {:>6.2} µs",
            s, d, cram, sram, lram, gram, cus, sus, lus, gus
        );
        d_vec.push(d as f64);
        cay_ram.push(cram);
        ssm_ram.push(sram);
        lru_ram.push(lram);
        gru_ram.push(gram);
        cay_us.push(cus);
        ssm_us.push(sus);
        lru_us.push(lus);
        gru_us.push(gus);
    }
    println!(
        "Empirical Power-Law Exponents (D=12->768): RAM ~ Cayley D^{:.2} / SSM D^{:.2} / LRU D^{:.2} / GRU D^{:.2} | Lat ~ Cayley D^{:.2} / SSM D^{:.2} / LRU D^{:.2} / GRU D^{:.2}",
        loglog_slope(&d_vec, &cay_ram),
        loglog_slope(&d_vec, &ssm_ram),
        loglog_slope(&d_vec, &lru_ram),
        loglog_slope(&d_vec, &gru_ram),
        loglog_slope(&d_vec, &cay_us),
        loglog_slope(&d_vec, &ssm_us),
        loglog_slope(&d_vec, &lru_us),
        loglog_slope(&d_vec, &gru_us)
    );

    println!("\n============================================================================================");
    println!("AXIS 1B: Accuracy Scaling Law vs State Width (5 seeds, Equal Step Budgets)");
    println!("============================================================================================");
    let seeds: [u64; 5] = [1, 2, 3, 4, 5];

    // 1B-1: A_5 Length Generalization at L=30 (Train strictly L in 1..8)
    println!("\n[1B-1] A_5 <= SO(3) Length Generalization at L=30 across M frames (S=3*M, Train L<=8, Linear vs Proto-SO(3)):");
    println!(
        "{:<12} | {:<14} | {:<14} | {:<14} | {:<10}",
        "S (Frames M)", "Lin. Exact 60", "Proto Exact 60", "Proto Orbit 5", "RAM (KB)"
    );
    println!("{}", "-".repeat(78));
    let a5 = PermGroup::a5();
    for &s in &[3usize, 6, 12, 24, 48] {
        let mut exacts_lin = Vec::new();
        let mut exacts_pr = Vec::new();
        let mut orbits_pr = Vec::new();
        let mut ram_kb = 0.0;
        for &seed in &seeds {
            let mut rng_tr = splitmix64(seed ^ 0x1B_A501);
            let mut rng_ev = splitmix64(seed ^ 0x1B_A502);
            let mut nac = NonAbelianCayleyRtrl::new(s, 4, 60, 0.25, seed);
            nac.enable_so3_frames_end_to_end();
            for ep in 0..4000usize {
                let len = 1 + (ep % 8);
                let (gs, ys) = a5.gen_word(&mut rng_tr, len);
                nac.reset();
                for (&g, &y) in gs.iter().zip(ys.iter()) {
                    let mut x = [0.0_f32; 4];
                    x[g] = 1.0;
                    let _ = nac.step_and_learn_supervised(&x, Some(y), 0.22, 0.04);
                }
            }
            let mut hits_l = 0usize;
            for _ in 0..160usize {
                let (gs, ys) = a5.gen_word(&mut rng_ev, 30);
                nac.reset();
                let mut pred = 0usize;
                for &g in &gs {
                    let mut x = [0.0_f32; 4];
                    x[g] = 1.0;
                    pred = nac.step_and_learn_supervised(&x, None, 0.0, 0.0);
                }
                if pred == ys[29] { hits_l += 1; }
            }
            exacts_lin.push(hits_l as f64 * 100.0 / 160.0);

            let mut rng_tr2 = splitmix64(seed ^ 0x1B_A501);
            let mut rng_ev2 = splitmix64(seed ^ 0x1B_A502);
            let mut nac_p = NonAbelianCayleyRtrl::new(s, 4, 60, 0.25, seed);
            nac_p.enable_so3_proto_metric(10.0, 1.2, 0.08);
            ram_kb = nac_p.memory_footprint_bytes() as f64 / 1024.0;
            for ep in 0..4000usize {
                let len = 1 + (ep % 8);
                let (gs, ys) = a5.gen_word(&mut rng_tr2, len);
                nac_p.reset();
                let dec = 1.0_f32 / (1.0 + 6.0 * (ep as f32) / 4000.0);
                for (&g, &y) in gs.iter().zip(ys.iter()) {
                    let mut x = [0.0_f32; 4];
                    x[g] = 1.0;
                    let _ = nac_p.step_and_learn_supervised(&x, Some(y), 0.02 * dec, 0.020 * dec);
                }
            }
            let (mut hits_p, mut ohits_p) = (0usize, 0usize);
            for _ in 0..160usize {
                let (gs, ys) = a5.gen_word(&mut rng_ev2, 30);
                nac_p.reset();
                let mut pred = 0usize;
                for &g in &gs {
                    let mut x = [0.0_f32; 4];
                    x[g] = 1.0;
                    pred = nac_p.step_and_learn_supervised(&x, None, 0.0, 0.0);
                }
                let yt = ys[29];
                if pred == yt { hits_p += 1; }
                if a5.perms[pred][0] == a5.perms[yt][0] { ohits_p += 1; }
            }
            exacts_pr.push(hits_p as f64 * 100.0 / 160.0);
            orbits_pr.push(ohits_p as f64 * 100.0 / 160.0);
        }
        let (lm, ls) = mean_sd(&exacts_lin);
        let (pm, ps) = mean_sd(&exacts_pr);
        let (om, os) = mean_sd(&orbits_pr);
        println!(
            "S={:<2} (M={:<2})   | {:5.1}±{:4.1}%    | {:5.1}±{:4.1}%    | {:5.1}±{:4.1}%    | {:>7.1} KB",
            s, s / 3, lm, ls, pm, ps, om, os, ram_kb
        );
    }

    // 1B-2: RepeatPrevious K=8 & 1B-3: Lorenz-63 K=8 across S in {4, 8, 16, 32, 64}
    println!("\n[1B-2 & 1B-3] RepeatPrevious K=8 & Chaotic Lorenz-63 K=8 across S in 4..64:");
    println!("{:<10} | {:<14} | {:<14} | {:<10}", "S (Dim D)", "RepeatPrev K=8", "Lorenz-63 K=8", "RAM (KB)");
    println!("{}", "-".repeat(60));
    for &s in &[4usize, 8, 16, 32, 64] {
        let mut rp_accs = Vec::new();
        let mut l63_accs = Vec::new();
        let mut ram_kb = 0.0;
        for &seed in &seeds {
            // RepeatPrevious K=8 (4 symbols, lag=8, ep_len=32, 240 train + 60 eval)
            let mut cay_rp = CayleyBlockSupervised::new(s, 4, 4, 0.65, 2, 0.55, seed ^ 0x1B_7780);
            ram_kb = cay_rp.memory_footprint_bytes() as f64 / 1024.0;
            let mut rng = splitmix64(seed ^ 0x1B_8801);
            let k_lag = 8usize;
            let (mut eval_h, mut eval_n) = (0usize, 0usize);
            for ep in 0..300usize {
                let train = ep < 240;
                cay_rp.reset();
                let mut hist = Vec::with_capacity(32);
                for t in 0..32usize {
                    rng = splitmix64(rng);
                    let sym = ((rng >> 16) as usize) % 4;
                    hist.push(sym);
                    let mut x = [0.0_f32; 4];
                    x[sym] = 1.0;
                    if t >= k_lag {
                        let y = hist[t - k_lag];
                        let pred = cay_rp.step_and_learn(
                            &x,
                            if train { Some(y) } else { None },
                            0.10,
                            0.015,
                            0.0,
                        );
                        if !train {
                            eval_n += 1;
                            if pred == y { eval_h += 1; }
                        }
                    } else {
                        let _ = cay_rp.step_and_learn(&x, None, 0.0, 0.0, 0.0);
                    }
                }
            }
            rp_accs.push(eval_h as f64 * 100.0 / eval_n.max(1) as f64);

            // Lorenz-63 K=8 (online 3500 steps, score on last 2000)
            let xs = generate_lorenz63(3500, 8, seed);
            let mut cay_l = CayleyBlockSupervised::new(s, 1, 2, 0.65, 1, 0.90, seed ^ 0x1B_6308);
            let (mut l_h, mut l_n) = (0usize, 0usize);
            for t in 0..3500usize {
                let x = [xs[t]];
                let y = usize::from(xs[t + 8] >= 0.0);
                let pred = cay_l.step_and_learn(&x, Some(y), 0.08, 0.015, 0.3);
                if t >= 1500 {
                    l_n += 1;
                    if pred == y { l_h += 1; }
                }
            }
            l63_accs.push(l_h as f64 * 100.0 / l_n as f64);
        }
        let (rm, rs) = mean_sd(&rp_accs);
        let (lm, ls) = mean_sd(&l63_accs);
        println!("S={:<2} ({:<3}D)| {:5.1}±{:4.1}%    | {:5.1}±{:4.1}%    | {:>7.1} KB", s, s * 3, rm, rs, lm, ls, ram_kb);
    }
}

// ============================================================================
// Axis 2: Depth Scaling Without BPTT (1L -> 2L -> 3L -> 4L)
// ============================================================================
fn gen_dyck2_seq(rng: &mut u64, len: usize, max_depth: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    // Токены: 0:'(', 1:')', 2:'[', 3:']'; Target y_t in {0: top '(', 1: top '[', 2: empty}
    let mut tokens = Vec::with_capacity(len);
    let mut targets = Vec::with_capacity(len);
    let mut depths = Vec::with_capacity(len);
    let mut stack: Vec<u8> = Vec::new();

    for t in 0..len {
        *rng = splitmix64(*rng);
        let rem = len - t;
        let can_push = stack.len() < max_depth && stack.len() + 1 <= rem;
        let can_pop = !stack.is_empty();
        let do_push = if !can_pop {
            true
        } else if !can_push {
            false
        } else {
            ((*rng >> 12) & 3) != 0
        };
        let tok = if do_push {
            let open_t = ((*rng >> 18) as usize) & 1; // 0 for '(', 1 for '['
            stack.push(open_t as u8);
            if open_t == 0 { 0usize } else { 2usize }
        } else {
            let top = stack.pop().unwrap();
            if top == 0 { 1usize } else { 3usize }
        };
        let d_after = stack.len();
        let tgt = match stack.last().copied() {
            Some(0) => 0usize,
            Some(_) => 1usize,
            None => 2usize,
        };
        tokens.push(tok);
        targets.push(tgt);
        depths.push(d_after);
    }
    (tokens, targets, depths)
}

fn run_axis2_depth() {
    println!("\n============================================================================================");
    println!("AXIS 2: Depth Scaling Without BPTT (1L -> 2L -> 3L -> 4L Local RTRL + Residual/DFA, 5 seeds)");
    println!("============================================================================================");
    let seeds: [u64; 5] = [1, 2, 3, 4, 5];

    // 2A: Symmetric Group S_5 (120 permutations, odd transpositions + 5-cycles, Eval L=10)
    println!("\n[2A] Symmetric Group S_5 (120 elements, odd+even gens, Train L<=6 -> Eval L=10):");
    println!("{:<28} | {:<13} | {:<13} | {:<13} | {:<9}", "Architecture", "S5 Exact 120", "S5 Orbit 5", "Parity Z_2", "RAM (KB)");
    println!("{}", "-".repeat(86));
    let s5 = PermGroup::s5();

    let s5_configs: [(&str, usize, usize); 5] = [
        ("1L NonAbelian Cayley (S=9)", 1, 9),
        ("2L Local-RTRL Stack (2xS=9)", 2, 9),
        ("3L Local-RTRL Stack (3xS=9)", 3, 9),
        ("4L Local-RTRL Stack (4xS=9)", 4, 9),
        ("1L Width-Matched (S=36)", 1, 36),
    ];
    for &(label, depth, spl) in &s5_configs {
        let mut ex_v = Vec::new();
        let mut or_v = Vec::new();
        let mut pa_v = Vec::new();
        let mut ram_kb = 0.0;
        for &seed in &seeds {
            let mut rng_tr = splitmix64(seed ^ 0x55_77A1);
            let mut rng_ev = splitmix64(seed ^ 0x55_E7A1);
            let mut net = DeepNonAbelianStack::new(depth, spl, 4, 8, 120, seed);
            ram_kb = net.memory_footprint_bytes() as f64 / 1024.0;
            for ep in 0..3200usize {
                let len = 1 + (ep % 6);
                let (gs, ys) = s5.gen_word(&mut rng_tr, len);
                net.reset();
                for (&g, &y) in gs.iter().zip(ys.iter()) {
                    let mut x = [0.0_f32; 4];
                    x[g] = 1.0;
                    let _ = net.step_and_learn(&x, Some(y), 0.22, 0.04);
                }
            }
            let (mut hits, mut ohits, mut phits) = (0usize, 0usize, 0usize);
            for _ in 0..160usize {
                let (gs, ys) = s5.gen_word(&mut rng_ev, 10);
                net.reset();
                let mut pred = 0usize;
                for &g in &gs {
                    let mut x = [0.0_f32; 4];
                    x[g] = 1.0;
                    pred = net.step_and_learn(&x, None, 0.0, 0.0);
                }
                let yt = ys[9];
                if pred == yt { hits += 1; }
                if s5.perms[pred][0] == s5.perms[yt][0] { ohits += 1; }
                if s5.even_flags[pred] == s5.even_flags[yt] { phits += 1; }
            }
            ex_v.push(hits as f64 * 100.0 / 160.0);
            or_v.push(ohits as f64 * 100.0 / 160.0);
            pa_v.push(phits as f64 * 100.0 / 160.0);
        }
        let (em, es) = mean_sd(&ex_v);
        let (om, os) = mean_sd(&or_v);
        let (pm, ps) = mean_sd(&pa_v);
        println!(
            "{:<28} | {:5.1}±{:4.1}%  | {:5.1}±{:4.1}%  | {:5.1}±{:4.1}%  | {:>6.1} KB",
            label, em, es, om, os, pm, ps, ram_kb
        );
    }

    // 2B: Hierarchical Bracket Matching Dyck-2 (depth 3..5) & 2C: Coupled Lorenz-96 (N=16, F=8.0)
    println!("\n[2B & 2C] Hierarchical Dyck-2 (Nesting Depth 3..5) & Spatiotemporal Chaos Lorenz-96 (N=16, F=8.0):");
    println!(
        "{:<28} | {:<13} | {:<13} | {:<13} | {:<9} | {:<8}",
        "Architecture (Zero BPTT)", "Dyck-2 All", "Dyck Depth>=3", "Lorenz-96", "RAM (KB)", "Lat µs"
    );
    println!("{}", "-".repeat(98));

    let depth_configs: [(&str, usize, usize, bool); 6] = [
        ("1L Cayley-RTRL (S=12)", 1, 12, false),
        ("2L Local-RTRL (2xS=12)", 2, 12, false),
        ("3L Local-RTRL (3xS=12)", 3, 12, false),
        ("4L Local-RTRL (4xS=12)", 4, 12, false),
        ("1L Width-Matched (S=36)", 1, 36, false),
        ("Online Per-Step GRU (H=48)", 1, 48, true),
    ];

    for &(label, depth, spl, is_gru) in &depth_configs {
        let mut d2_all = Vec::new();
        let mut d2_deep = Vec::new();
        let mut l96_v = Vec::new();
        let mut lat_v = Vec::new();
        let mut ram_kb = 0.0;

        for &seed in &seeds {
            // 2B: Dyck-2 (len=24, max_depth=5, 1600 train + 400 eval)
            let mut rng_d = splitmix64(seed ^ 0x2B_DC20);
            let (mut ah, mut an, mut dh, mut dn) = (0usize, 0usize, 0usize, 0usize);
            if is_gru {
                let mut gru = GruPerStepBank::new_supervised(4, spl, 3, 4, seed);
                for ep in 0..2000usize {
                    let train = ep < 1600;
                    let (toks, tgts, deps) = gen_dyck2_seq(&mut rng_d, 24, 5);
                    gru.start_episode();
                    for t in 0..24usize {
                        let mut x = [0.0_f32; 4];
                        x[toks[t]] = 1.0;
                        let pred = gru.step_and_learn_supervised(&x, if train { Some(tgts[t]) } else { None });
                        if !train {
                            an += 1;
                            if pred == tgts[t] { ah += 1; }
                            if deps[t] >= 3 {
                                dn += 1;
                                if pred == tgts[t] { dh += 1; }
                            }
                        }
                    }
                }
            } else {
                let mut net = DeepCayleyStack::new(depth, spl, 4, 8, 3, 0.72, seed ^ 0x2B_DC20);
                for ep in 0..2000usize {
                    let train = ep < 1600;
                    let (toks, tgts, deps) = gen_dyck2_seq(&mut rng_d, 24, 5);
                    net.reset();
                    for t in 0..24usize {
                        let mut x = [0.0_f32; 4];
                        x[toks[t]] = 1.0;
                        let pred = net.step_and_learn(&x, if train { Some(tgts[t]) } else { None }, 0.09, 0.015, 0.0);
                        if !train {
                            an += 1;
                            if pred == tgts[t] { ah += 1; }
                            if deps[t] >= 3 {
                                dn += 1;
                                if pred == tgts[t] { dh += 1; }
                            }
                        }
                    }
                }
            }
            d2_all.push(ah as f64 * 100.0 / an as f64);
            d2_deep.push(dh as f64 * 100.0 / dn.max(1) as f64);

            // 2C: Lorenz-96 (N=16 variables, observe 8 even channels, predict sign(x_{0, t+6}), 4000 steps)
            let traj = generate_lorenz96(16, 4000, 6, seed);
            let (mut lh, mut ln) = (0usize, 0usize);
            let t0 = Instant::now();
            if is_gru {
                let mut gru = GruPerStepBank::new_supervised(8, spl, 2, 4, seed ^ 0x2C_9600);
                ram_kb = gru.memory_footprint_bytes() as f64 / 1024.0;
                for t in 0..4000usize {
                    let obs = [
                        traj[t][0], traj[t][2], traj[t][4], traj[t][6],
                        traj[t][8], traj[t][10], traj[t][12], traj[t][14],
                    ];
                    let y = usize::from(traj[t + 6][0] >= 0.0);
                    let pred = gru.step_and_learn_supervised(&obs, Some(y));
                    if t >= 1500 {
                        ln += 1;
                        if pred == y { lh += 1; }
                    }
                }
            } else {
                let mut net = DeepCayleyStack::new(depth, spl, 8, 8, 2, 0.85, seed ^ 0x2C_9600);
                ram_kb = net.memory_footprint_bytes() as f64 / 1024.0;
                for t in 0..4000usize {
                    let obs = [
                        traj[t][0], traj[t][2], traj[t][4], traj[t][6],
                        traj[t][8], traj[t][10], traj[t][12], traj[t][14],
                    ];
                    let y = usize::from(traj[t + 6][0] >= 0.0);
                    let pred = net.step_and_learn(&obs, Some(y), 0.06, 0.014, 0.25);
                    if t >= 1500 {
                        ln += 1;
                        if pred == y { lh += 1; }
                    }
                }
            }
            lat_v.push((t0.elapsed().as_nanos() as f64 / 1000.0) / 4000.0);
            l96_v.push(lh as f64 * 100.0 / ln as f64);
        }
        let (am, asd) = mean_sd(&d2_all);
        let (dm, dsd) = mean_sd(&d2_deep);
        let (lm, lsd) = mean_sd(&l96_v);
        let (um, _) = mean_sd(&lat_v);
        println!(
            "{:<28} | {:5.1}±{:4.1}%  | {:5.1}±{:4.1}%  | {:5.1}±{:4.1}%  | {:>6.1} KB | {:>6.2} µs",
            label, am, asd, dm, dsd, lm, lsd, ram_kb, um
        );
    }
}

// ============================================================================
// Axis 3: Horizon / Temporal Extrapolation Scaling (L = 6 -> 500, K = 2 -> 24)
// ============================================================================
fn run_axis3_horizon() {
    println!("\n============================================================================================");
    println!("AXIS 3A: A_5 <= SO(3) Ultra-Long Horizon Extrapolation (Train strictly L<=8 -> Eval L=6..500, 5 seeds)");
    println!("============================================================================================");
    let a5 = PermGroup::a5();
    let seeds: [u64; 5] = [1, 2, 3, 4, 5];
    let horizons = [6usize, 12, 18, 30, 50, 100, 200, 500];

    #[derive(Clone, Copy)]
    enum H3Model {
        NacProtoM1,
        NacProtoM4,
        NacProtoM8,
        NacEndToEnd,
        SelectiveSsm,
        GruBptt,
    }
    let models = [
        ("NonAbelian Cayley SO(3) Proto M=1", H3Model::NacProtoM1),
        ("NonAbelian Cayley SO(3) Proto M=4", H3Model::NacProtoM4),
        ("NonAbelian Cayley SO(3) Proto M=8", H3Model::NacProtoM8),
        ("NonAbelian Cayley SO(3) (Lin.)", H3Model::NacEndToEnd),
        ("Online Selective SSM (36D)", H3Model::SelectiveSsm),
        ("Online GRU + BPTT (H=32)", H3Model::GruBptt),
    ];

    println!(
        "{:<34} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7}",
        "Model (60-elem Exact %)", "L=6", "L=12", "L=18", "L=30", "L=50", "L=100", "L=200", "L=500"
    );
    println!("{}", "-".repeat(110));

    let mut max_sphere_err_500 = 0.0_f64;
    let mut max_orth_err_500 = 0.0_f64;

    for (name, kind) in models {
        let mut h_accs: [Vec<f64>; 8] = Default::default();
        for &seed in &seeds {
            let mut rng_tr = splitmix64(seed ^ 0x3A_A500);
            let mut rng_ev = splitmix64(seed ^ 0x3A_E700);
            match kind {
                H3Model::NacProtoM1 | H3Model::NacProtoM4 | H3Model::NacProtoM8 => {
                    let s_cnt = match kind {
                        H3Model::NacProtoM1 => 3usize,
                        H3Model::NacProtoM8 => 24usize,
                        _ => 12usize,
                    };
                    let mut nac = NonAbelianCayleyRtrl::new(s_cnt, 4, 60, 0.25, seed);
                    nac.enable_so3_proto_metric(10.0, 1.2, 0.08);
                    for ep in 0..4000usize {
                        let len = 1 + (ep % 8);
                        let (gs, ys) = a5.gen_word(&mut rng_tr, len);
                        nac.reset();
                        let dec = 1.0_f32 / (1.0 + 6.0 * (ep as f32) / 4000.0);
                        for (&g, &y) in gs.iter().zip(ys.iter()) {
                            let mut x = [0.0_f32; 4];
                            x[g] = 1.0;
                            let _ = nac.step_and_learn_supervised(&x, Some(y), 0.02 * dec, 0.020 * dec);
                        }
                    }
                    for (hi, &len_h) in horizons.iter().enumerate() {
                        let n_eval = if len_h >= 200 { 60usize } else { 100usize };
                        let mut hits = 0usize;
                        for _ in 0..n_eval {
                            let (gs, ys) = a5.gen_word(&mut rng_ev, len_h);
                            nac.reset();
                            let mut pred = 0usize;
                            for &g in &gs {
                                let mut x = [0.0_f32; 4];
                                x[g] = 1.0;
                                pred = nac.step_and_learn_supervised(&x, None, 0.0, 0.0);
                            }
                            if pred == ys[len_h - 1] { hits += 1; }
                            if len_h == 500 && matches!(kind, H3Model::NacProtoM4) {
                                for s in 0..nac.sectors {
                                    let off = s * 3;
                                    let n2 = (nac.u[off] * nac.u[off]
                                        + nac.u[off + 1] * nac.u[off + 1]
                                        + nac.u[off + 2] * nac.u[off + 2])
                                        .sqrt() as f64;
                                    max_sphere_err_500 = max_sphere_err_500.max((n2 - 1.0).abs());
                                }
                                max_orth_err_500 = max_orth_err_500.max(nac.max_orth_drift() as f64);
                            }
                        }
                        h_accs[hi].push(hits as f64 * 100.0 / n_eval as f64);
                    }
                }
                H3Model::NacEndToEnd => {
                    let mut nac = NonAbelianCayleyRtrl::new(12, 4, 60, 0.25, seed);
                    nac.enable_so3_frames_end_to_end();
                    for ep in 0..4000usize {
                        let len = 1 + (ep % 8);
                        let (gs, ys) = a5.gen_word(&mut rng_tr, len);
                        nac.reset();
                        for (&g, &y) in gs.iter().zip(ys.iter()) {
                            let mut x = [0.0_f32; 4];
                            x[g] = 1.0;
                            let _ = nac.step_and_learn_supervised(&x, Some(y), 0.22, 0.04);
                        }
                    }
                    for (hi, &len_h) in horizons.iter().enumerate() {
                        let n_eval = if len_h >= 200 { 60usize } else { 100usize };
                        let mut hits = 0usize;
                        for _ in 0..n_eval {
                            let (gs, ys) = a5.gen_word(&mut rng_ev, len_h);
                            nac.reset();
                            let mut pred = 0usize;
                            for &g in &gs {
                                let mut x = [0.0_f32; 4];
                                x[g] = 1.0;
                                pred = nac.step_and_learn_supervised(&x, None, 0.0, 0.0);
                            }
                            if pred == ys[len_h - 1] { hits += 1; }
                            if len_h == 500 {
                                for s in 0..nac.sectors {
                                    let off = s * 3;
                                    let n2 = (nac.u[off] * nac.u[off]
                                        + nac.u[off + 1] * nac.u[off + 1]
                                        + nac.u[off + 2] * nac.u[off + 2])
                                        .sqrt() as f64;
                                    max_sphere_err_500 = max_sphere_err_500.max((n2 - 1.0).abs());
                                }
                                let n_frames = nac.sectors / 3;
                                for m in 0..n_frames {
                                    let b = m * 9;
                                    let mut fro2 = 0.0_f64;
                                    for r in 0..3 {
                                        for c in 0..3 {
                                            let d = nac.u[b + r * 3] as f64 * nac.u[b + c * 3] as f64
                                                + nac.u[b + r * 3 + 1] as f64 * nac.u[b + c * 3 + 1] as f64
                                                + nac.u[b + r * 3 + 2] as f64 * nac.u[b + c * 3 + 2] as f64;
                                            let diff = d - if r == c { 1.0 } else { 0.0 };
                                            fro2 += diff * diff;
                                        }
                                    }
                                    max_orth_err_500 = max_orth_err_500.max(fro2.sqrt());
                                }
                            }
                        }
                        h_accs[hi].push(hits as f64 * 100.0 / n_eval as f64);
                    }
                }
                H3Model::SelectiveSsm => {
                    let mut ssm = SelectiveSsmBank::new(4, 18, 60, seed);
                    ssm.enable_input_clamp = false;
                    for ep in 0..4000usize {
                        let len = 1 + (ep % 8);
                        let (gs, ys) = a5.gen_word(&mut rng_tr, len);
                        ssm.start_episode();
                        for (&g, &y) in gs.iter().zip(ys.iter()) {
                            let mut x = [0.0_f32; 4];
                            x[g] = 1.0;
                            let _ = ssm.step_and_learn_supervised(&x, Some(y), 0.05, 0.012);
                        }
                    }
                    for (hi, &len_h) in horizons.iter().enumerate() {
                        let n_eval = if len_h >= 200 { 60usize } else { 100usize };
                        let mut hits = 0usize;
                        for _ in 0..n_eval {
                            let (gs, ys) = a5.gen_word(&mut rng_ev, len_h);
                            ssm.start_episode();
                            let mut pred = 0usize;
                            for &g in &gs {
                                let mut x = [0.0_f32; 4];
                                x[g] = 1.0;
                                pred = ssm.step_and_learn_supervised(&x, None, 0.0, 0.0);
                            }
                            if pred == ys[len_h - 1] { hits += 1; }
                        }
                        h_accs[hi].push(hits as f64 * 100.0 / n_eval as f64);
                    }
                }
                H3Model::GruBptt => {
                    let mut gru = GruBpttSeq::new(4, 32, 60, seed);
                    for ep in 0..4000usize {
                        let len = 1 + (ep % 8);
                        let (gs, ys) = a5.gen_word(&mut rng_tr, len);
                        gru.reset();
                        for &g in &gs {
                            let mut x = [0.0_f32; 4];
                            x[g] = 1.0;
                            let _ = gru.forward_step(&x, true);
                        }
                        gru.bptt_update(&ys, 0.008);
                    }
                    for (hi, &len_h) in horizons.iter().enumerate() {
                        let n_eval = if len_h >= 200 { 60usize } else { 100usize };
                        let mut hits = 0usize;
                        for _ in 0..n_eval {
                            let (gs, ys) = a5.gen_word(&mut rng_ev, len_h);
                            gru.reset();
                            let mut pred = 0usize;
                            for &g in &gs {
                                let mut x = [0.0_f32; 4];
                                x[g] = 1.0;
                                pred = gru.forward_step(&x, false);
                            }
                            if pred == ys[len_h - 1] { hits += 1; }
                        }
                        h_accs[hi].push(hits as f64 * 100.0 / n_eval as f64);
                    }
                }
            }
        }
        println!(
            "{:<34} | {:5.1}%  | {:5.1}%  | {:5.1}%  | {:5.1}%  | {:5.1}%  | {:5.1}%  | {:5.1}%  | {:5.1}%",
            name,
            mean_sd(&h_accs[0]).0,
            mean_sd(&h_accs[1]).0,
            mean_sd(&h_accs[2]).0,
            mean_sd(&h_accs[3]).0,
            mean_sd(&h_accs[4]).0,
            mean_sd(&h_accs[5]).0,
            mean_sd(&h_accs[6]).0,
            mean_sd(&h_accs[7]).0,
        );
    }

    println!(
        "Live Unnormalized f32 Drift at L=500: max | ||u_500||_2 - 1 | = {:.2e} | max ||Q_500^T Q_500 - I_3||_F = {:.2e}",
        max_sphere_err_500, max_orth_err_500
    );

    // Axis 3B: RepeatPrevious Deep Delay Horizon Scaling K in {2, 4, 8, 12, 16, 24}
    println!("\n============================================================================================");
    println!("AXIS 3B: RepeatPrevious Deep Delay Horizon Scaling (K in 2..24, 5 seeds, 0% exp vs Transc.)");
    println!("============================================================================================");
    println!(
        "{:<7} | {:<13} | {:<13} | {:<13} | {:<13} | {:<13} | {:<9}",
        "Delay K", "3D Cayley b=1", "2D Rat.(36D)", "Alg.Sel.(36D)", "Sel-SSM (36D)", "GRU (H=32)", "2D RAM"
    );
    println!("{}", "-".repeat(100));

    for &k_lag in &[2usize, 4, 8, 12, 16, 24] {
        let s_cay = (k_lag * 2).max(16);
        let mut cay_v = Vec::new();
        let mut c2d_v = Vec::new();
        let mut asc_v = Vec::new();
        let mut ssm_v = Vec::new();
        let mut gru_v = Vec::new();
        let mut c2d_ram = 0.0;
        let ep_len = k_lag + 20;

        for &seed in &seeds {
            let mut cay = CayleyBlockSupervised::new(s_cay, 4, 4, 0.65, 1, 0.38, seed ^ (k_lag as u64));
            let mut c2d = AlgebraicSelectiveCayley::new(4, 18, 4, false, seed ^ (k_lag as u64));
            c2d_ram = c2d.memory_footprint_bytes() as f64 / 1024.0;
            let mut asc = AlgebraicSelectiveCayley::new(4, 18, 4, true, seed ^ (k_lag as u64));
            let mut ssm = SelectiveSsmBank::new(4, 18, 4, seed ^ (k_lag as u64));
            ssm.enable_input_clamp = true;
            let mut gru = GruPerStepBank::new_supervised(4, 32, 4, 4, seed ^ (k_lag as u64));

            let rng_base = splitmix64(seed ^ 0x3B_1100 + k_lag as u64);
            let mut rng_c = rng_base;
            let mut rng_2d = rng_base;
            let mut rng_as = rng_base;
            let mut rng_s = rng_base;
            let mut rng_g = rng_base;

            let mut eval_model = |rng: &mut u64, mode: usize| -> f64 {
                let (mut hits, mut cnt) = (0usize, 0usize);
                for ep in 0..310usize {
                    let train = ep < 250;
                    match mode {
                        0 => cay.reset(),
                        1 => c2d.reset(),
                        2 => asc.reset(),
                        3 => ssm.start_episode(),
                        _ => gru.start_episode(),
                    }
                    let mut hist = Vec::with_capacity(ep_len);
                    for t in 0..ep_len {
                        *rng = splitmix64(*rng);
                        let sym = ((*rng >> 16) as usize) % 4;
                        hist.push(sym);
                        let mut x = [0.0_f32; 4];
                        x[sym] = 1.0;
                        let target = if t >= k_lag { Some(hist[t - k_lag]) } else { None };
                        let learn_t = if train { target } else { None };
                        let pred = match mode {
                            0 => cay.step_and_learn(&x, learn_t, 0.10, 0.015, 0.0),
                            1 => c2d.step_and_learn_supervised(&x, learn_t, 0.05, 0.015),
                            2 => asc.step_and_learn_supervised(&x, learn_t, 0.05, 0.015),
                            3 => ssm.step_and_learn_supervised(&x, learn_t, 0.05, 0.012),
                            _ => gru.step_and_learn_supervised(&x, learn_t),
                        };
                        if !train {
                            if let Some(y) = target {
                                cnt += 1;
                                if pred == y { hits += 1; }
                            }
                        }
                    }
                }
                hits as f64 * 100.0 / cnt.max(1) as f64
            };
            cay_v.push(eval_model(&mut rng_c, 0));
            c2d_v.push(eval_model(&mut rng_2d, 1));
            asc_v.push(eval_model(&mut rng_as, 2));
            ssm_v.push(eval_model(&mut rng_s, 3));
            gru_v.push(eval_model(&mut rng_g, 4));
        }
        println!(
            "K={:<5} | {:5.1}±{:4.1}%   | {:5.1}±{:4.1}%   | {:5.1}±{:4.1}%   | {:5.1}±{:4.1}%   | {:5.1}±{:4.1}%   | {:>6.1} KB",
            k_lag,
            mean_sd(&cay_v).0, mean_sd(&cay_v).1,
            mean_sd(&c2d_v).0, mean_sd(&c2d_v).1,
            mean_sd(&asc_v).0, mean_sd(&asc_v).1,
            mean_sd(&ssm_v).0, mean_sd(&ssm_v).1,
            mean_sd(&gru_v).0, mean_sd(&gru_v).1,
            c2d_ram,
        );
    }
}

// ============================================================================
// Axis 4: Input Dimensionality (d_in = 4..256) & Regime Count (R = 2..32)
// ============================================================================
fn run_axis4_input_and_regimes() {
    println!("\n============================================================================================");
    println!("AXIS 4A: Input Dimensionality Scaling (True N=d_in Lorenz-96 in 4..256: Direct vs Count-Sketch vs Sup-GRU, 5 seeds)");
    println!("============================================================================================");
    println!(
        "{:<5} | {:<21} | {:<21} | {:<21}",
        "d_in", "Direct Cayley S=16", "Proj(d->8) Cayley S=16", "Online GRU (H=32)"
    );
    println!(
        "{:<5} | {:<21} | {:<21} | {:<21}",
        "", "Acc / RAM / Lat µs", "Acc / RAM / Lat µs", "Acc / RAM / Lat µs"
    );
    println!("{}", "-".repeat(78));

    let seeds: [u64; 5] = [1, 2, 3, 4, 5];
    for &d_in in &[4usize, 16, 32, 64, 128, 256] {
        let mut acc_dir = Vec::new();
        let mut lat_dir = Vec::new();
        let mut ram_dir = 0.0;
        let mut acc_prj = Vec::new();
        let mut lat_prj = Vec::new();
        let mut ram_prj = 0.0;
        let mut acc_gru = Vec::new();
        let mut lat_gru = Vec::new();
        let mut ram_gru = 0.0;

        for &seed in &seeds {
            // Истинная N=d_in-мерная хаотическая динамика Лоренца-96 (RK4, F=8.0) без малорангового вложения
            let traj = generate_lorenz96(d_in, 2500, 4, seed);
            let mut obs_stream = Vec::with_capacity(2500);
            for t in 0..2500usize {
                let ch: Vec<f32> = traj[t].iter().map(|&v| v.clamp(-3.0, 3.0)).collect();
                obs_stream.push(ch);
            }

            // 1. Direct Cayley-RTRL (S=16, полный d_in)
            let mut cay_d = CayleyBlockSupervised::new(16, d_in, 2, 0.65, 1, 0.85, seed ^ 0x4A_001);
            ram_dir = cay_d.memory_footprint_bytes() as f64 / 1024.0;
            let (mut hd, mut nd) = (0usize, 0usize);
            let t0 = Instant::now();
            for t in 0..2500usize {
                let y = usize::from(traj[t + 4][0] >= 0.0);
                let pred = cay_d.step_and_learn(&obs_stream[t], Some(y), 0.06, 0.012, 0.2);
                if t >= 1000 { nd += 1; if pred == y { hd += 1; } }
            }
            lat_dir.push((t0.elapsed().as_nanos() as f64 / 1000.0) / 2500.0);
            acc_dir.push(hd as f64 * 100.0 / nd as f64);

            // 2. Low-Rank Count-Sketch / Orthogonal Hashing Projector (d_in -> r=8) + Cayley-RTRL(S=16, 8D)
            let r_bot = 8usize;
            let mut idx_map = vec![(0usize, 0usize, 1.0_f32, 1.0_f32); d_in];
            let mut rng_p = splitmix64(seed ^ 0x4A_9900);
            let inv_norm = 1.0 / ((d_in.max(1) as f32) / r_bot as f32).sqrt();
            for i in 0..d_in {
                rng_p = splitmix64(rng_p);
                let b1 = ((rng_p >> 8) as usize) % r_bot;
                let b2 = (b1 + 1 + (((rng_p >> 16) as usize) % (r_bot - 1))) % r_bot;
                let s1 = if ((rng_p >> 24) & 1) == 0 { inv_norm } else { -inv_norm };
                let s2 = if ((rng_p >> 25) & 1) == 0 { inv_norm } else { -inv_norm };
                idx_map[i] = (b1, b2, s1, s2);
            }
            let mut cay_p = CayleyBlockSupervised::new(16, r_bot, 2, 0.65, 1, 0.85, seed ^ 0x4A_002);
            ram_prj = (cay_p.memory_footprint_bytes() + d_in * 16) as f64 / 1024.0;
            let mut proj_buf = [0.0_f32; 8];
            let (mut hp, mut np) = (0usize, 0usize);
            let t1 = Instant::now();
            for t in 0..2500usize {
                proj_buf.fill(0.0);
                let obs = &obs_stream[t];
                for i in 0..d_in {
                    let v = obs[i];
                    let (b1, b2, s1, s2) = idx_map[i];
                    proj_buf[b1] += s1 * v;
                    proj_buf[b2] += s2 * v;
                }
                let y = usize::from(traj[t + 4][0] >= 0.0);
                let pred = cay_p.step_and_learn(&proj_buf, Some(y), 0.06, 0.012, 0.2);
                if t >= 1000 { np += 1; if pred == y { hp += 1; } }
            }
            lat_prj.push((t1.elapsed().as_nanos() as f64 / 1000.0) / 2500.0);
            acc_prj.push(hp as f64 * 100.0 / np as f64);

            // 3. Online Per-Step Supervised GRU (H=32, BPTT4)
            let mut gru = GruPerStepBank::new_supervised(d_in, 32, 2, 4, seed ^ 0x4A_003);
            ram_gru = gru.memory_footprint_bytes() as f64 / 1024.0;
            let (mut hg, mut ng) = (0usize, 0usize);
            let t2 = Instant::now();
            for t in 0..2500usize {
                let y = usize::from(traj[t + 4][0] >= 0.0);
                let pred = gru.step_and_learn_supervised(&obs_stream[t], Some(y));
                if t >= 1000 { ng += 1; if pred == y { hg += 1; } }
            }
            lat_gru.push((t2.elapsed().as_nanos() as f64 / 1000.0) / 2500.0);
            acc_gru.push(hg as f64 * 100.0 / ng as f64);
        }

        println!(
            "d={:<3} | {:4.1}%/{:5.1}K/{:4.1}µs | {:4.1}%/{:5.1}K/{:4.1}µs | {:4.1}%/{:5.1}K/{:4.1}µs",
            d_in,
            mean_sd(&acc_dir).0, ram_dir, mean_sd(&lat_dir).0,
            mean_sd(&acc_prj).0, ram_prj, mean_sd(&lat_prj).0,
            mean_sd(&acc_gru).0, ram_gru, mean_sd(&lat_gru).0,
        );
    }

    // Axis 4B: Continual Regime Count R in {2, 4, 8, 16, 32}
    println!("\n============================================================================================");
    println!("AXIS 4B: Continual Regime Count Scaling (R in 2..32 Cyclical Regimes, Seq vs Par6 Return, 5 seeds)");
    println!("============================================================================================");
    println!(
        "{:<6} | {:<14} | {:<14} | {:<14} | {:<14} | {:<10} | {:<8}",
        "R", "Core(Seq) Str", "Core(Seq) Ret", "Core(Par6) Ret", "1-Bank Return", "Core RAM", "Lat µs"
    );
    println!("{}", "-".repeat(104));

    for &n_reg in &[2usize, 4, 8, 16, 32] {
        let mut core_str = Vec::new();
        let mut core_ret = Vec::new();
        let mut par_ret = Vec::new();
        let mut one_ret = Vec::new();
        let mut lat_v = Vec::new();
        let mut ram_kb = 0.0;

        for &seed in &seeds {
            // Строим R различных плотных 6D-режимов (уникальные ортогональные/знаковые гиперплоскости на S^5)
            let mut reg_w = Vec::with_capacity(n_reg);
            let mut rng_r = splitmix64(seed ^ 0x4B_0000);
            for r in 0..n_reg {
                let mut w = [0.0_f32; 6];
                for j in 0..6usize {
                    rng_r = splitmix64(rng_r);
                    w[j] = rand_signed_f32(&mut rng_r);
                }
                let nm = (w.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-5);
                for v in &mut w { *v /= nm; }
                // Гарантируем смену знака относительно предыдущего режима
                if r > 0 {
                    let prev: [f32; 6] = reg_w[r - 1];
                    let dot: f32 = w.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                    if dot > -0.2 {
                        for j in 0..6 { w[j] = -w[j]; }
                    }
                }
                reg_w.push(w);
            }
            let steps_per_reg = 450usize;
            let xs = generate_lorenz63(steps_per_reg * n_reg + 1000, 6, seed);

            let mut cfg = MicroCoreConfig::numeric_stream(2, seed);
            cfg.half_sectors = 8;
            cfg.belief_lambda = 0.85;
            cfg.rotor_norm = 0.25;
            cfg.rtrl_sectors = 10;
            cfg.rtrl_slow_sectors = 6;
            cfg.rtrl_max_in = 6;
            cfg.rtrl_lr = 0.012;
            cfg.rtrl_grid_mode = 1;
            cfg.rtrl_use_skip_x = true;
            cfg.linear_lr = 0.08;
            cfg.ph_delta = 0.06;
            cfg.ph_threshold = 2.0;
            cfg.ph_warmup = 4;
            cfg.max_regimes = n_reg;
            let mut core = MicroContinualCore::new(cfg);
            core.encoder.freeze_to_i8();
            core.start_episode();

            let mut cfg_par = cfg;
            cfg_par.parallel_bank_probe = true;
            let mut core_par = MicroContinualCore::new(cfg_par);
            core_par.encoder.freeze_to_i8();
            core_par.start_episode();
            ram_kb = core_par.memory_footprint_bytes() as f64 / 1024.0;

            let mut single_cfg = cfg;
            single_cfg.ph_threshold = 9999.0; // отключаем переключение (1 активный банк)
            single_cfg.max_regimes = 2;
            let mut single = MicroContinualCore::new(single_cfg);
            single.encoder.freeze_to_i8();
            single.start_episode();

            let (mut sh, mut sn) = (0usize, 0usize);
            let t0 = Instant::now();
            let mut step_idx = 0usize;
            for r in 0..n_reg {
                let w = reg_w[r];
                for s_t in 0..steps_per_reg {
                    let obs = [
                        xs[step_idx],
                        xs[step_idx + 1],
                        xs[step_idx + 2],
                        xs[step_idx + 3],
                        xs[step_idx + 4],
                        xs[step_idx + 5],
                    ];
                    let score = obs[0] * w[0] + obs[1] * w[1] + obs[2] * w[2]
                        + obs[3] * w[3] + obs[4] * w[4] + obs[5] * w[5];
                    let y = usize::from(score >= 0.0);
                    let pc = core.decide_vec(&obs, true, false, false).get().0;
                    core.end_supervised_step(pc, y);
                    if s_t >= 150 {
                        sn += 1;
                        if pc == y { sh += 1; }
                    }
                    step_idx += 1;
                }
            }
            lat_v.push((t0.elapsed().as_nanos() as f64 / 1000.0) / step_idx.max(1) as f64);
            core_str.push(sh as f64 * 100.0 / sn as f64);

            // Фаза быстрого возврата для core (Seq)
            let check_cnt = n_reg.min(8);
            let (mut rh_c, mut rh_p, mut rh_1, mut rn) = (0usize, 0usize, 0usize, 0usize);
            let ret_start_idx = step_idx;
            for k in 0..check_cnt {
                let r = (k * n_reg) / check_cnt;
                let w = reg_w[r];
                for _s_t in 0..50usize {
                    let obs = [
                        xs[step_idx],
                        xs[step_idx + 1],
                        xs[step_idx + 2],
                        xs[step_idx + 3],
                        xs[step_idx + 4],
                        xs[step_idx + 5],
                    ];
                    let score = obs[0] * w[0] + obs[1] * w[1] + obs[2] * w[2]
                        + obs[3] * w[3] + obs[4] * w[4] + obs[5] * w[5];
                    let y = usize::from(score >= 0.0);
                    let pc = core.decide_vec(&obs, true, false, false).get().0;
                    core.end_supervised_step(pc, y);
                    rn += 1;
                    if pc == y { rh_c += 1; }
                    step_idx += 1;
                }
            }

            // Независимый цикл для core_par (Parallel 8-step Read-Only Bank Match)
            let mut p_idx = 0usize;
            for r in 0..n_reg {
                let w = reg_w[r];
                for _s_t in 0..steps_per_reg {
                    let obs = [
                        xs[p_idx], xs[p_idx + 1], xs[p_idx + 2],
                        xs[p_idx + 3], xs[p_idx + 4], xs[p_idx + 5],
                    ];
                    let score = obs[0] * w[0] + obs[1] * w[1] + obs[2] * w[2]
                        + obs[3] * w[3] + obs[4] * w[4] + obs[5] * w[5];
                    let y = usize::from(score >= 0.0);
                    let pp = core_par.decide_vec(&obs, true, false, false).get().0;
                    core_par.end_supervised_step(pp, y);
                    p_idx += 1;
                }
            }
            debug_assert_eq!(p_idx, ret_start_idx);
            for k in 0..check_cnt {
                let r = (k * n_reg) / check_cnt;
                let w = reg_w[r];
                for _s_t in 0..50usize {
                    let obs = [
                        xs[p_idx], xs[p_idx + 1], xs[p_idx + 2],
                        xs[p_idx + 3], xs[p_idx + 4], xs[p_idx + 5],
                    ];
                    let score = obs[0] * w[0] + obs[1] * w[1] + obs[2] * w[2]
                        + obs[3] * w[3] + obs[4] * w[4] + obs[5] * w[5];
                    let y = usize::from(score >= 0.0);
                    let pp = core_par.decide_vec(&obs, true, false, false).get().0;
                    core_par.end_supervised_step(pp, y);
                    if pp == y { rh_p += 1; }
                    p_idx += 1;
                }
            }

            // Независимый цикл для 1-Bank контрольной модели (вне таймера core)
            let mut s_idx = 0usize;
            for r in 0..n_reg {
                let w = reg_w[r];
                for _s_t in 0..steps_per_reg {
                    let obs = [
                        xs[s_idx], xs[s_idx + 1], xs[s_idx + 2],
                        xs[s_idx + 3], xs[s_idx + 4], xs[s_idx + 5],
                    ];
                    let score = obs[0] * w[0] + obs[1] * w[1] + obs[2] * w[2]
                        + obs[3] * w[3] + obs[4] * w[4] + obs[5] * w[5];
                    let y = usize::from(score >= 0.0);
                    let ps = single.decide_vec(&obs, true, false, false).get().0;
                    single.end_supervised_step(ps, y);
                    s_idx += 1;
                }
            }
            debug_assert_eq!(s_idx, ret_start_idx);
            for k in 0..check_cnt {
                let r = (k * n_reg) / check_cnt;
                let w = reg_w[r];
                for _s_t in 0..50usize {
                    let obs = [
                        xs[s_idx], xs[s_idx + 1], xs[s_idx + 2],
                        xs[s_idx + 3], xs[s_idx + 4], xs[s_idx + 5],
                    ];
                    let score = obs[0] * w[0] + obs[1] * w[1] + obs[2] * w[2]
                        + obs[3] * w[3] + obs[4] * w[4] + obs[5] * w[5];
                    let y = usize::from(score >= 0.0);
                    let ps = single.decide_vec(&obs, true, false, false).get().0;
                    single.end_supervised_step(ps, y);
                    if ps == y { rh_1 += 1; }
                    s_idx += 1;
                }
            }
            core_ret.push(rh_c as f64 * 100.0 / rn.max(1) as f64);
            par_ret.push(rh_p as f64 * 100.0 / rn.max(1) as f64);
            one_ret.push(rh_1 as f64 * 100.0 / rn.max(1) as f64);
        }
        println!(
            "R={:<3} | {:5.1}±{:4.1}%   | {:5.1}±{:4.1}%   | {:5.1}±{:4.1}%   | {:5.1}±{:4.1}%   | {:>7.1} KB | {:>6.2} µs",
            n_reg,
            mean_sd(&core_str).0, mean_sd(&core_str).1,
            mean_sd(&core_ret).0, mean_sd(&core_ret).1,
            mean_sd(&par_ret).0, mean_sd(&par_ret).1,
            mean_sd(&one_ret).0, mean_sd(&one_ret).1,
            ram_kb,
            mean_sd(&lat_v).0
        );
    }
}

// ============================================================================
// Axis 5: Systemic Swarm Scaling (N = 1..10,000 Agents) & L1/L2/L3 Cache Boundary Profile
// ============================================================================
fn run_axis5_swarm(sink: &mut usize) {
    println!("\n============================================================================================");
    println!("AXIS 5: Systemic Swarm Scaling (N = 1..10,000 Agents) & Physical L1/L2/L3 Cache Boundary Profile");
    println!("Config per agent: State D=48 (Cayley S=16, Sel-SSM U=24, LRU U=24, Sup-GRU H=48), d_in=8, C=4");
    println!("============================================================================================");
    println!(
        "{:<7} | {:<10} | {:<10} | {:<10} | {:<10} | {:<9} | {:<9} | {:<9} | {:<9} | {}",
        "Agents", "Cay RAM", "SSM RAM", "LRU RAM", "GRU RAM", "Cay µs", "SSM µs", "LRU µs", "GRU µs", "Cay Cache"
    );
    println!("{}", "-".repeat(126));

    let x = [0.3_f32, -0.5, 0.2, 0.7, -0.1, 0.4, -0.6, 0.2];
    for &n_agents in &[1usize, 4, 16, 64, 256, 1024, 4096, 10000] {
        let mut cays: Vec<CayleyBlockSupervised> = (0..n_agents)
            .map(|i| CayleyBlockSupervised::new(16, 8, 4, 0.65, 1, 0.85, 500 + i as u64))
            .collect();
        let mut ssms: Vec<SelectiveSsmBank> = (0..n_agents)
            .map(|i| SelectiveSsmBank::new(8, 24, 4, 500 + i as u64))
            .collect();
        let mut lrus: Vec<OnlineLruStream> = (0..n_agents)
            .map(|i| OnlineLruStream::new(8, 24, 4, 500 + i as u64))
            .collect();
        let mut grus: Vec<GruPerStepBank> = (0..n_agents)
            .map(|i| GruPerStepBank::new_supervised(8, 48, 4, 1, 500 + i as u64))
            .collect();

        let cram_bytes = cays[0].memory_footprint_bytes() * n_agents;
        let sram_bytes = ssms[0].memory_footprint_bytes() * n_agents;
        let lram_bytes = lrus[0].memory_footprint_bytes() * n_agents;
        let gram_bytes = grus[0].memory_footprint_bytes() * n_agents;

        let sweeps = if n_agents <= 4 {
            2000usize
        } else if n_agents <= 64 {
            300usize
        } else if n_agents <= 1024 {
            30usize
        } else if n_agents <= 4096 {
            10usize
        } else {
            4usize
        };

        for a in cays.iter_mut() { *sink ^= a.step_and_learn(&x, Some(1), 0.04, 0.01, 0.3); }
        for s in ssms.iter_mut() { *sink ^= s.step_and_learn_supervised(&x, Some(1), 0.04, 0.01); }
        for l in lrus.iter_mut() { *sink ^= l.predict_and_train(&x, 1, true); }
        for g in grus.iter_mut() { *sink ^= g.step_and_learn_supervised(&x, Some(1)); }

        let t0 = Instant::now();
        for sw in 0..sweeps {
            let tgt = Some(sw % 4);
            for a in cays.iter_mut() {
                *sink ^= a.step_and_learn(&x, tgt, 0.04, 0.01, 0.3);
            }
        }
        let c_us = (t0.elapsed().as_nanos() as f64 / 1000.0) / (n_agents * sweeps) as f64;

        let ts = Instant::now();
        for sw in 0..sweeps {
            let tgt = Some(sw % 4);
            for s in ssms.iter_mut() {
                *sink ^= s.step_and_learn_supervised(&x, tgt, 0.04, 0.01);
            }
        }
        let s_us = (ts.elapsed().as_nanos() as f64 / 1000.0) / (n_agents * sweeps) as f64;

        let tl = Instant::now();
        for sw in 0..sweeps {
            let tgt = sw % 4;
            for l in lrus.iter_mut() {
                *sink ^= l.predict_and_train(&x, tgt, true);
            }
        }
        let l_us = (tl.elapsed().as_nanos() as f64 / 1000.0) / (n_agents * sweeps) as f64;

        let t1 = Instant::now();
        for sw in 0..sweeps {
            let tgt = Some(sw % 4);
            for g in grus.iter_mut() {
                *sink ^= g.step_and_learn_supervised(&x, tgt);
            }
        }
        let g_us = (t1.elapsed().as_nanos() as f64 / 1000.0) / (n_agents * sweeps) as f64;

        let tier = if cram_bytes <= 32 * 1024 {
            "L1d (<=32 KB)"
        } else if cram_bytes <= 512 * 1024 {
            "L2 (<=512 KB)"
        } else if cram_bytes <= 96 * 1024 * 1024 {
            "L3 V-Cache (<=96 MB)"
        } else {
            "DRAM (>96 MB)"
        };

        let fmt_bytes = |b: usize| -> String {
            if b < 1024 * 1024 {
                format!("{:>6.1} KB", b as f64 / 1024.0)
            } else {
                format!("{:>6.2} MB", b as f64 / (1024.0 * 1024.0))
            }
        };

        println!(
            "{:<7} | {:<10} | {:<10} | {:<10} | {:<10} | {:>6.2} µs | {:>6.2} µs | {:>6.2} µs | {:>6.2} µs | {}",
            n_agents,
            fmt_bytes(cram_bytes),
            fmt_bytes(sram_bytes),
            fmt_bytes(lram_bytes),
            fmt_bytes(gram_bytes),
            c_us, s_us, l_us, g_us,
            tier
        );
    }

    // Multi-Core Swarm Throughput (N=4096 simultaneous online agents via std::thread::scope)
    let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4);
    let n_swarm = 4096usize;
    let sweeps_mc = 25usize;
    let mut cays_mc: Vec<CayleyBlockSupervised> = (0..n_swarm)
        .map(|i| CayleyBlockSupervised::new(16, 8, 4, 0.65, 1, 0.85, 9000 + i as u64))
        .collect();
    let mut ssms_mc: Vec<SelectiveSsmBank> = (0..n_swarm)
        .map(|i| SelectiveSsmBank::new(8, 24, 4, 9000 + i as u64))
        .collect();
    let mut lrus_mc: Vec<OnlineLruStream> = (0..n_swarm)
        .map(|i| OnlineLruStream::new(8, 24, 4, 9000 + i as u64))
        .collect();
    let mut grus_mc: Vec<GruPerStepBank> = (0..n_swarm)
        .map(|i| GruPerStepBank::new_supervised(8, 48, 4, 1, 9000 + i as u64))
        .collect();

    let chunk_c = (n_swarm + threads - 1) / threads;
    let t_mc_c = Instant::now();
    std::thread::scope(|sc| {
        for chunk in cays_mc.chunks_mut(chunk_c) {
            sc.spawn(move || {
                let mut local = 0usize;
                for sw in 0..sweeps_mc {
                    let tgt = Some(sw % 4);
                    for a in chunk.iter_mut() {
                        local ^= a.step_and_learn(&x, tgt, 0.04, 0.01, 0.3);
                    }
                }
                std::hint::black_box(local);
            });
        }
    });
    let sec_c = t_mc_c.elapsed().as_secs_f64();

    let t_mc_s = Instant::now();
    std::thread::scope(|sc| {
        for chunk in ssms_mc.chunks_mut(chunk_c) {
            sc.spawn(move || {
                let mut local = 0usize;
                for sw in 0..sweeps_mc {
                    let tgt = Some(sw % 4);
                    for s in chunk.iter_mut() {
                        local ^= s.step_and_learn_supervised(&x, tgt, 0.04, 0.01);
                    }
                }
                std::hint::black_box(local);
            });
        }
    });
    let sec_s = t_mc_s.elapsed().as_secs_f64();

    let t_mc_l = Instant::now();
    std::thread::scope(|sc| {
        for chunk in lrus_mc.chunks_mut(chunk_c) {
            sc.spawn(move || {
                let mut local = 0usize;
                for sw in 0..sweeps_mc {
                    let tgt = sw % 4;
                    for l in chunk.iter_mut() {
                        local ^= l.predict_and_train(&x, tgt, true);
                    }
                }
                std::hint::black_box(local);
            });
        }
    });
    let sec_l = t_mc_l.elapsed().as_secs_f64();

    let t_mc_g = Instant::now();
    std::thread::scope(|sc| {
        for chunk in grus_mc.chunks_mut(chunk_c) {
            sc.spawn(move || {
                let mut local = 0usize;
                for sw in 0..sweeps_mc {
                    let tgt = Some(sw % 4);
                    for g in chunk.iter_mut() {
                        local ^= g.step_and_learn_supervised(&x, tgt);
                    }
                }
                std::hint::black_box(local);
            });
        }
    });
    let sec_g = t_mc_g.elapsed().as_secs_f64();

    let msteps_c = ((n_swarm * sweeps_mc) as f64 / 1e6) / sec_c;
    let msteps_s = ((n_swarm * sweeps_mc) as f64 / 1e6) / sec_s;
    let msteps_l = ((n_swarm * sweeps_mc) as f64 / 1e6) / sec_l;
    let msteps_g = ((n_swarm * sweeps_mc) as f64 / 1e6) / sec_g;
    println!(
        "\nMulti-Core Throughput ({} threads, N={} simultaneous agents): Cayley-RTRL = {:.2} M/s ({:.1} MB) | Sel-SSM = {:.2} M/s ({:.1} MB) | LRU = {:.2} M/s ({:.1} MB) | Sup-GRU = {:.2} M/s ({:.1} MB)",
        threads,
        n_swarm,
        msteps_c, (cays_mc[0].memory_footprint_bytes() * n_swarm) as f64 / (1024.0 * 1024.0),
        msteps_s, (ssms_mc[0].memory_footprint_bytes() * n_swarm) as f64 / (1024.0 * 1024.0),
        msteps_l, (lrus_mc[0].memory_footprint_bytes() * n_swarm) as f64 / (1024.0 * 1024.0),
        msteps_g, (grus_mc[0].memory_footprint_bytes() * n_swarm) as f64 / (1024.0 * 1024.0),
    );
}

fn main() {
    let total_t0 = Instant::now();
    let mut sink = 0usize;

    run_axis1_width(&mut sink);
    run_axis2_depth();
    run_axis3_horizon();
    run_axis4_input_and_regimes();
    run_axis5_swarm(&mut sink);

    std::hint::black_box(sink);
    println!(
        "\nCompleted full 5-Axis Scalability Suite in {:.2}s (checksum={}).",
        total_t0.elapsed().as_secs_f64(),
        sink
    );
}
