// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Benchmark Step 9: Prime-Field Modular Arithmetic (F_997 & F_9973)
//! Across 3 Generations: v0.4.0 vs v0.4.1 vs v0.5.0 (+ OnlineMlpAdam reference)
//!
//! Zero-Oracle protocol (0% sin/cos/Fourier, 0% mod-p feature engineering):
//! - Programmatic verification of primality for p = 997 (10-bit, g = 7) and p = 9973 (14-bit, g = 11)
//!   and primitive-root bijection g^k <-> log_g(x).
//! - Section 1 (Bit-Streamed LSB->MSB + normalized operands [ba, bb, a/p-0.5, b/p-0.5, tf, is_last],
//!   strictly disjoint 80/20 Train/Holdout):
//!     Add: (a + b) mod p | Mul: (a * b) mod p | Pow_g: g^k mod p | Pow_ab: a^b mod p | DLog: log_g(x) mod (p-1)
//!     + Label-Shuffle control (per-bit majority floor: 50.75% on F_997, 53.87% on F_9973).
//! - Section 2 (Direct Residue Regression y/p, 16 raw features without oracles, Disjoint Holdout + Bit-30 SEU):
//!     RMSE across all 5 tasks (uniform random floor = 1/sqrt(12) ≈ 0.2887).
//! - Section 3 (Order-12 Multiplicative Subgroup <g^{(p-1)/12}> <= F_p^* — isomorphic to C_12 under
//!   generator-action inputs [1,0]/[0,1], train L in 1..=8 -> extrapolation L = 6, 30, 100, 500):
//!     v0.4.0 vs v0.4.1 vs v0.5.0 (+ Label-Shuffle control 8.3%).

use std::time::Instant;

use micro_continual_core::fixed_point::{FixedCayleyCore, Q39_24};
use micro_continual_core::math::splitmix64;
use micro_continual_core::stream_baselines::OnlineMlpAdam;
use micro_continual_core::{
    AdaptiveOrthogonalRtrl, AlgebraicSelectiveCayley, MicroContinualCore, MicroCoreConfig,
    NonAbelianCayleyRtrl, NonAbelianO4Rtrl, UnifiedCayleyConfig, UnifiedCayleyCore,
};

// =============================================================================
// 0. Exact Prime-Field Arithmetic Utilities (p = 997, p = 9973)
// =============================================================================

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3u64;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut acc = 1u64 % modulus;
    base %= modulus;
    while exp > 0 {
        if (exp & 1) == 1 {
            acc = ((acc as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exp >>= 1;
    }
    acc
}

fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut d = 2u64;
    while d * d <= n {
        if n % d == 0 {
            out.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += if d == 2 { 1 } else { 2 };
    }
    if n > 1 {
        out.push(n);
    }
    out
}

fn find_smallest_primitive_root(p: u64) -> u64 {
    let phi = p - 1;
    let factors = prime_factors(phi);
    for g in 2..p {
        let mut ok = true;
        for &q in &factors {
            if mod_pow(g, phi / q, p) == 1 {
                ok = false;
                break;
            }
        }
        if ok {
            return g;
        }
    }
    panic!("No primitive root found for prime {p}");
}

struct PrimeField {
    p: u64,
    bits: usize,
    g: u64,
    pow_tbl: Vec<u64>,  // pow_tbl[k] = g^k mod p, k in 0..p-1
    dlog_tbl: Vec<u64>, // dlog_tbl[x] = k for x in 1..p
}

impl PrimeField {
    fn new(p: u64) -> Self {
        assert!(is_prime(p), "{p} must be prime");
        let g = find_smallest_primitive_root(p);
        let phi = (p - 1) as usize;
        let mut pow_tbl = vec![0u64; phi];
        let mut dlog_tbl = vec![u64::MAX; p as usize];
        let mut cur = 1u64;
        for k in 0..phi {
            pow_tbl[k] = cur;
            assert_eq!(dlog_tbl[cur as usize], u64::MAX, "Duplicate residue in primitive root table");
            dlog_tbl[cur as usize] = k as u64;
            cur = (cur * g) % p;
        }
        assert_eq!(cur, 1, "g^(p-1) must be 1 mod p");
        for x in 1..(p as usize) {
            assert_ne!(dlog_tbl[x], u64::MAX, "Primitive root failed bijection at x={x}");
        }
        let bits = 64 - (p - 1).leading_zeros() as usize;
        Self { p, bits, g, pow_tbl, dlog_tbl }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArithOp {
    Add,   // (a + b) mod p
    Mul,   // (a * b) mod p
    PowG,  // g^k mod p
    PowAB, // a^b mod p (a in 1..p, b in 0..p-1)
    DLog,  // log_g(x) mod (p-1), x in 1..p
}

impl ArithOp {
    fn name(self) -> &'static str {
        match self {
            ArithOp::Add => "Add   (a+b) mod p  ",
            ArithOp::Mul => "Mul   (a*b) mod p  ",
            ArithOp::PowG => "PowG  g^k mod p    ",
            ArithOp::PowAB => "PowAB a^b mod p    ",
            ArithOp::DLog => "DLog  log_g(x)     ",
        }
    }

    fn is_unary(self) -> bool {
        matches!(self, ArithOp::PowG | ArithOp::DLog)
    }

    fn eval(self, f: &PrimeField, a: u64, b: u64) -> u64 {
        match self {
            ArithOp::Add => (a + b) % f.p,
            ArithOp::Mul => (a * b) % f.p,
            ArithOp::PowG => f.pow_tbl[(a % (f.p - 1)) as usize],
            ArithOp::PowAB => {
                let base = if a == 0 { 1 } else { a % f.p };
                mod_pow(base, b % (f.p - 1), f.p)
            }
            ArithOp::DLog => {
                let x = if a == 0 { 1 } else { a % f.p };
                f.dlog_tbl[x as usize]
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Sample {
    a: u64,
    b: u64,
    y: u64,
}

/// Строго непересекающееся разбиение 80% Train / 20% Holdout по хешу операндов `(a, b)`.
fn is_holdout_pair(op: ArithOp, a: u64, b: u64) -> bool {
    let key = if op.is_unary() {
        splitmix64(a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x55AA_1234)
    } else {
        splitmix64(((a << 20) ^ b).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA55A_5678)
    };
    (key % 5) == 0
}

fn generate_disjoint_dataset(
    f: &PrimeField,
    op: ArithOp,
    n_train: usize,
    n_eval: usize,
    seed: u64,
    shuffle_labels: bool,
) -> (Vec<Sample>, Vec<Sample>) {
    let mut rng = splitmix64(seed ^ (f.p << 16) ^ ((op as u64) << 4));
    let mut train = Vec::with_capacity(n_train);
    let mut eval = Vec::with_capacity(n_eval);

    let draw_ab = |r: &mut u64| -> (u64, u64) {
        *r = splitmix64(*r);
        let r1 = *r;
        *r = splitmix64(*r);
        let r2 = *r;
        match op {
            ArithOp::Add | ArithOp::Mul => (r1 % f.p, r2 % f.p),
            ArithOp::PowG => (r1 % (f.p - 1), 0),
            ArithOp::PowAB => (1 + (r1 % (f.p - 1)), r2 % (f.p - 1)),
            ArithOp::DLog => (1 + (r1 % (f.p - 1)), 0),
        }
    };

    while train.len() < n_train || eval.len() < n_eval {
        let (a, b) = draw_ab(&mut rng);
        let hold = is_holdout_pair(op, a, b);
        if hold && eval.len() < n_eval {
            let mut y = op.eval(f, a, b);
            if shuffle_labels {
                rng = splitmix64(rng);
                y = rng % f.p;
            }
            eval.push(Sample { a, b, y });
        } else if !hold && train.len() < n_train {
            let mut y = op.eval(f, a, b);
            if shuffle_labels {
                rng = splitmix64(rng);
                y = rng % f.p;
            }
            train.push(Sample { a, b, y });
        }
    }
    (train, eval)
}

#[inline]
fn bit_pm1(v: u64, pos: usize) -> f32 {
    if ((v >> pos) & 1) == 1 { 1.0_f32 } else { -1.0_f32 }
}

/// 6D нуль-оракульный вход для побитового потокового шага `t in 0..B` (LSB -> MSB).
fn encode_bit_serial_step(f: &PrimeField, op: ArithOp, s: Sample, t: usize) -> [f32; 6] {
    let ba = bit_pm1(s.a, t);
    let bb = if op.is_unary() { 0.0_f32 } else { bit_pm1(s.b, t) };
    let an = (s.a as f32) / (f.p as f32) - 0.5;
    let bn = if op.is_unary() { 0.0_f32 } else { (s.b as f32) / (f.p as f32) - 0.5 };
    let tf = (t as f32) / ((f.bits.saturating_sub(1).max(1)) as f32) - 0.5;
    let is_last = if t + 1 == f.bits { 1.0_f32 } else { -1.0_f32 };
    [ba, bb, an, bn, tf, is_last]
}

/// 16D нуль-оракульный вход для прямого одношагового прогноза вычета `y / p`.
fn encode_direct_16(f: &PrimeField, op: ArithOp, s: Sample) -> [f32; 16] {
    let mut x = [0.0_f32; 16];
    let inv_p = 1.0_f32 / (f.p as f32);
    x[0] = (s.a as f32) * inv_p - 0.5;
    if op.is_unary() {
        x[1] = 0.0;
        let nb = f.bits.min(14);
        for b in 0..nb {
            x[2 + b] = bit_pm1(s.a, b);
        }
    } else {
        x[1] = (s.b as f32) * inv_p - 0.5;
        for b in 0..7usize {
            x[2 + b] = bit_pm1(s.a, b);
            x[9 + b] = bit_pm1(s.b, b);
        }
    }
    x
}

// =============================================================================
// Unified Model Wrapper across v0.4.0, v0.4.1, v0.5.0 (+ OnlineMlpAdam)
// =============================================================================

#[derive(Clone, Copy)]
enum ModelKind {
    V040MicroCore,
    V040AlgSelCayley,
    V041MicroCorePure2L,
    V041NonAbelianSo3,
    V050UnifiedOneCfg,
    V050NonAbelianO4,
    V050FixedQ824,
    BaselineMlpAdam,
}

impl ModelKind {
    fn label(self) -> &'static str {
        match self {
            ModelKind::V040MicroCore => "v0.4.0 MicroCore (B0+Kerr) ",
            ModelKind::V040AlgSelCayley => "v0.4.0 Alg.Sel.Cayley 2D   ",
            ModelKind::V041MicroCorePure2L => "v0.4.1 MicroCore PURE 2L   ",
            ModelKind::V041NonAbelianSo3 => "v0.4.1 NonAbelian SO(3)    ",
            ModelKind::V050UnifiedOneCfg => "v0.5.1 Unified One-Config  ",
            ModelKind::V050NonAbelianO4 => "v0.5.0 NonAbelian O(4) M=1 ",
            ModelKind::V050FixedQ824 => "v0.5.0 Fixed Q8.24 (0-FPU) ",
            ModelKind::BaselineMlpAdam => "Ref    OnlineMlpAdam H=32  ",
        }
    }
}

enum PrimeModel {
    Micro(MicroContinualCore),
    AlgSel(AlgebraicSelectiveCayley),
    So3(NonAbelianCayleyRtrl),
    O4(NonAbelianO4Rtrl),
    Unified(UnifiedCayleyCore),
    Fixed(FixedCayleyCore),
    Mlp(OnlineMlpAdam),
}

fn fixed_memory_bytes(m: &FixedCayleyCore) -> usize {
    (m.norm_mean.len() + m.norm_m2.len()) * 8
        + (m.x_norm.len()
            + m.b_state.len()
            + m.alpha.len()
            + m.omega.len()
            + m.lc_cache.len()
            + m.ls_cache.len()
            + m.in_w.len()
            + m.s_inw.len()
            + m.feat.len()
            + m.actor_w.len()
            + m.actor_b.len()
            + m.reg_w.len()
            + m.logits.len()
            + m.probs.len()
            + m.psis.len())
            * 4
        + 64
}

impl PrimeModel {
    fn new(kind: ModelKind, in_dim: usize, num_classes: usize, seed: u64) -> Self {
        match kind {
            ModelKind::V040MicroCore => {
                let mut cfg = MicroCoreConfig::numeric_stream(num_classes.max(2), seed);
                cfg.max_regimes = 1;
                cfg.linear_lr = 0.06;
                cfg.linear_mix = 0.90;
                PrimeModel::Micro(MicroContinualCore::new(cfg))
            }
            ModelKind::V040AlgSelCayley => {
                PrimeModel::AlgSel(AlgebraicSelectiveCayley::new(in_dim, 16, num_classes.max(2), true, seed))
            }
            ModelKind::V041MicroCorePure2L => {
                let mut cfg = MicroCoreConfig::numeric_stream(num_classes.max(2), seed);
                cfg.half_sectors = 0;
                cfg.obs_bank_capacity = 0;
                cfg.bank_capacity = 0;
                cfg.max_regimes = 1;
                cfg.linear_lr = 0.055;
                cfg.linear_mix = 1.0;
                cfg.rtrl_sectors = 12;
                cfg.rtrl_slow_sectors = 6;
                cfg.rtrl_max_in = in_dim;
                cfg.rtrl_lr = 0.012;
                cfg.rtrl_grid_mode = 1;
                cfg.rtrl_use_skip_x = true;
                PrimeModel::Micro(MicroContinualCore::new(cfg))
            }
            ModelKind::V041NonAbelianSo3 => {
                PrimeModel::So3(NonAbelianCayleyRtrl::new(12, in_dim, num_classes.max(2), 0.5, seed))
            }
            ModelKind::V050UnifiedOneCfg => {
                PrimeModel::Unified(UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config(
                    num_classes.max(2),
                    in_dim,
                    seed,
                )))
            }
            ModelKind::V050NonAbelianO4 => {
                PrimeModel::O4(NonAbelianO4Rtrl::new(1, in_dim, num_classes.max(2), seed))
            }
            ModelKind::V050FixedQ824 => {
                PrimeModel::Fixed(FixedCayleyCore::new(in_dim, num_classes.max(2), seed))
            }
            ModelKind::BaselineMlpAdam => {
                let mut m = OnlineMlpAdam::new(in_dim, 32, num_classes.max(2), seed);
                m.lr = 0.008;
                PrimeModel::Mlp(m)
            }
        }
    }

    fn reset_episode(&mut self) {
        match self {
            PrimeModel::Micro(m) => m.start_episode(),
            PrimeModel::AlgSel(m) => m.reset(),
            PrimeModel::So3(m) => m.reset(),
            PrimeModel::O4(m) => m.reset(),
            PrimeModel::Unified(m) => m.start_episode(),
            PrimeModel::Fixed(m) => m.reset_episode(),
            PrimeModel::Mlp(_) => {}
        }
    }

    fn step_class(&mut self, x: &[f32], y: usize, train: bool) -> usize {
        match self {
            PrimeModel::Micro(m) => {
                let pred = m.decide_vec(x, train, false, false).get().0;
                if train {
                    let _ = m.end_supervised_step(pred, y);
                }
                pred
            }
            PrimeModel::AlgSel(m) => {
                m.step_and_learn_supervised(x, if train { Some(y) } else { None }, 0.035, 0.01)
            }
            PrimeModel::So3(m) => {
                m.step_and_learn_supervised(x, if train { Some(y) } else { None }, 0.03, 0.01)
            }
            PrimeModel::O4(m) => {
                m.step_and_learn_supervised(x, if train { Some(y) } else { None }, 0.03, 0.012)
            }
            PrimeModel::Unified(m) => m.step_supervised_class(x, y, true, train, true).0,
            PrimeModel::Fixed(m) => {
                let mut xq = [Q39_24(0); 24];
                let n = x.len().min(24);
                for i in 0..n {
                    xq[i] = Q39_24::from_f32_boundary(x[i]);
                }
                m.step_supervised_class_q(&xq[..n], y, train)
            }
            PrimeModel::Mlp(m) => m.predict_and_train(x, y, train),
        }
    }

    fn step_reg(&mut self, x: &[f32], y: f32, train: bool, seu_inject: bool) -> f32 {
        match self {
            PrimeModel::Micro(m) => {
                if seu_inject {
                    m.inject_weight_bit_flip(3, 30);
                }
                let pred = *m.decide_reg_vec(x, train, seu_inject, false).get();
                if train {
                    let _ = m.end_reg_step(y);
                }
                pred
            }
            PrimeModel::AlgSel(m) => {
                m.step_and_learn_regression(x, if train { Some(y) } else { None }, 0.03, 0.008)
            }
            PrimeModel::So3(_) | PrimeModel::O4(_) => 0.5,
            PrimeModel::Unified(m) => {
                if seu_inject {
                    m.inject_weight_bit30_seu(0);
                }
                m.step_supervised_reg(x, y, true, train, true)
            }
            PrimeModel::Fixed(m) => {
                let mut xq = [Q39_24(0); 24];
                let n = x.len().min(24);
                for i in 0..n {
                    xq[i] = Q39_24::from_f32_boundary(x[i]);
                }
                m.step_supervised_reg_q(&xq[..n], Q39_24::from_f32_boundary(y), train).to_f32_boundary()
            }
            PrimeModel::Mlp(m) => m.step_and_learn_regression(x, if train { Some(y) } else { None }),
        }
    }

    fn memory_kb(&self) -> f64 {
        let bytes = match self {
            PrimeModel::Micro(m) => m.memory_footprint_bytes(),
            PrimeModel::AlgSel(m) => m.memory_footprint_bytes(),
            PrimeModel::So3(m) => m.memory_footprint_bytes(),
            PrimeModel::O4(m) => m.memory_footprint_bytes(),
            PrimeModel::Unified(m) => m.memory_footprint_bytes(),
            PrimeModel::Fixed(m) => fixed_memory_bytes(m),
            PrimeModel::Mlp(m) => m.memory_footprint_bytes(),
        };
        bytes as f64 / 1024.0
    }
}

fn mean_std(vals: &[f64]) -> (f64, f64) {
    let n = vals.len().max(1) as f64;
    let mu = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / n;
    (mu, var.sqrt())
}

// =============================================================================
// Section 1: Bit-Serial Causal Streaming (LSB -> MSB) on F_997 & F_9973
// =============================================================================

fn run_bit_serial_suite(f: &PrimeField, seeds: &[u64]) {
    println!(
        "\n=========================================================================================="
    );
    println!(
        " SECTION 1: BIT-SERIAL STREAMING (LSB->MSB, B={} bits) on F_{} (g={}) — DISJOINT HOLDOUT",
        f.bits, f.p, f.g
    );
    println!(
        "=========================================================================================="
    );

    let ops = [ArithOp::Add, ArithOp::Mul, ArithOp::PowG, ArithOp::PowAB, ArithOp::DLog];
    let models = [
        ModelKind::V040MicroCore,
        ModelKind::V040AlgSelCayley,
        ModelKind::V041MicroCorePure2L,
        ModelKind::V041NonAbelianSo3,
        ModelKind::V050UnifiedOneCfg,
        ModelKind::V050NonAbelianO4,
        ModelKind::V050FixedQ824,
        ModelKind::BaselineMlpAdam,
    ];

    let n_train = 1200usize;
    let n_eval = 300usize;

    for &op in &ops {
        println!("\n--- F_{} | {} (train={} pairs, frozen disjoint holdout={}, {} seeds) ---",
            f.p, op.name().trim(), n_train, n_eval, seeds.len());
        println!(
            "{:<29} | {:>7} | {:>14} | {:>14} | {:>10}",
            "Model", "RAM(KB)", "Holdout AllBit", "Holdout Bit-0", "Shuf Ctrl"
        );
        for &mk in &models {
            let mut all_accs = Vec::with_capacity(seeds.len());
            let mut b0_accs = Vec::with_capacity(seeds.len());
            let mut ram_kb = 0.0_f64;

            for &seed in seeds {
                let (train_ds, eval_ds) = generate_disjoint_dataset(f, op, n_train, n_eval, seed, false);
                let mut m = PrimeModel::new(mk, 6, 2, seed);
                ram_kb = m.memory_kb();

                for &s in &train_ds {
                    m.reset_episode();
                    for t in 0..f.bits {
                        let x = encode_bit_serial_step(f, op, s, t);
                        let ybit = ((s.y >> t) & 1) as usize;
                        let _ = m.step_class(&x, ybit, true);
                    }
                }
                let mut hits = 0usize;
                let mut tot = 0usize;
                let mut b0_hits = 0usize;
                for &s in &eval_ds {
                    m.reset_episode();
                    for t in 0..f.bits {
                        let x = encode_bit_serial_step(f, op, s, t);
                        let ybit = ((s.y >> t) & 1) as usize;
                        let pred = m.step_class(&x, ybit, false);
                        if pred == ybit {
                            hits += 1;
                            if t == 0 {
                                b0_hits += 1;
                            }
                        }
                        tot += 1;
                    }
                }
                all_accs.push(100.0 * (hits as f64) / (tot.max(1) as f64));
                b0_accs.push(100.0 * (b0_hits as f64) / (eval_ds.len().max(1) as f64));
            }

            // Label-Shuffle anti-leakage control (seed 1) for v0.5.0 & v0.4.1 (incl. SO(3)) & MLP
            let shuf_str = if matches!(
                mk,
                ModelKind::V050UnifiedOneCfg
                    | ModelKind::V050NonAbelianO4
                    | ModelKind::V041MicroCorePure2L
                    | ModelKind::V041NonAbelianSo3
                    | ModelKind::BaselineMlpAdam
            ) {
                let (tr_s, ev_s) = generate_disjoint_dataset(f, op, n_train, n_eval, 999, true);
                let mut m = PrimeModel::new(mk, 6, 2, 999);
                for &s in &tr_s {
                    m.reset_episode();
                    for t in 0..f.bits {
                        let x = encode_bit_serial_step(f, op, s, t);
                        let _ = m.step_class(&x, ((s.y >> t) & 1) as usize, true);
                    }
                }
                let mut hits = 0usize;
                let mut tot = 0usize;
                for &s in &ev_s {
                    m.reset_episode();
                    for t in 0..f.bits {
                        let x = encode_bit_serial_step(f, op, s, t);
                        let ybit = ((s.y >> t) & 1) as usize;
                        if m.step_class(&x, ybit, false) == ybit {
                            hits += 1;
                        }
                        tot += 1;
                    }
                }
                format!("{:5.1}%", 100.0 * (hits as f64) / (tot.max(1) as f64))
            } else {
                "   -- ".to_string()
            };

            let (am, asd) = mean_std(&all_accs);
            let (b0m, b0sd) = mean_std(&b0_accs);
            println!(
                "{:<29} | {:7.1} | {:6.2}±{:5.2}% | {:6.2}±{:5.2}% | {:>10}",
                mk.label(),
                ram_kb,
                am,
                asd,
                b0m,
                b0sd,
                shuf_str
            );
        }
    }
}

// =============================================================================
// Section 2: Direct Residue Regression (y / p) on F_997 & F_9973 + Bit-30 SEU
// =============================================================================

fn run_direct_residue_suite(f: &PrimeField, seeds: &[u64]) {
    println!(
        "\n=========================================================================================="
    );
    println!(
        " SECTION 2: DIRECT RESIDUE REGRESSION (y/p, 16 raw features, 0 oracles) on F_{} (chance=0.2887)",
        f.p
    );
    println!(
        "=========================================================================================="
    );

    let ops = [ArithOp::Add, ArithOp::Mul, ArithOp::PowG, ArithOp::PowAB, ArithOp::DLog];
    let models = [
        ModelKind::V040MicroCore,
        ModelKind::V040AlgSelCayley,
        ModelKind::V041MicroCorePure2L,
        ModelKind::V050UnifiedOneCfg,
        ModelKind::V050FixedQ824,
        ModelKind::BaselineMlpAdam,
    ];
    let n_train = 2500usize;
    let n_eval = 500usize;
    let inv_p = 1.0_f32 / (f.p as f32);

    println!(
        "{:<29} | {:>7} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12}",
        "Model", "RAM(KB)", "Add RMSE", "Mul RMSE", "PowG RMSE", "PowAB RMSE", "DLog RMSE"
    );
    for &mk in &models {
        let mut op_rmses = [0.0_f64; 5];
        let mut ram_kb = 0.0_f64;
        for (oi, &op) in ops.iter().enumerate() {
            let mut rmses = Vec::with_capacity(seeds.len());
            for &seed in seeds {
                let (tr, ev) = generate_disjoint_dataset(f, op, n_train, n_eval, seed, false);
                let mut m = PrimeModel::new(mk, 16, 2, seed);
                ram_kb = m.memory_kb();
                for &s in &tr {
                    m.reset_episode();
                    let x = encode_direct_16(f, op, s);
                    let y = (s.y as f32) * inv_p;
                    let _ = m.step_reg(&x, y, true, false);
                }
                let mut sq_err = 0.0_f64;
                for &s in &ev {
                    m.reset_episode();
                    let x = encode_direct_16(f, op, s);
                    let y = (s.y as f32) * inv_p;
                    let pred = m.step_reg(&x, y, false, false).clamp(0.0, 1.0);
                    let d = (pred - y) as f64;
                    sq_err += d * d;
                }
                rmses.push((sq_err / (ev.len().max(1) as f64)).sqrt());
            }
            op_rmses[oi] = mean_std(&rmses).0;
        }
        println!(
            "{:<29} | {:7.1} | {:12.4} | {:12.4} | {:12.4} | {:12.4} | {:12.4}",
            mk.label(),
            ram_kb,
            op_rmses[0],
            op_rmses[1],
            op_rmses[2],
            op_rmses[3],
            op_rmses[4]
        );
    }

    // Bit-30 State+Weight SEU check on Add (a+b) mod p for v0.5.0 Unified & v0.4.1 MicroCore
    let (tr, ev) = generate_disjoint_dataset(f, ArithOp::Add, n_train, n_eval, 7, false);
    for mk in [ModelKind::V041MicroCorePure2L, ModelKind::V050UnifiedOneCfg] {
        let mut m = PrimeModel::new(mk, 16, 2, 7);
        let mut any_nan = false;
        for (idx, &s) in tr.iter().enumerate() {
            m.reset_episode();
            let x = encode_direct_16(f, ArithOp::Add, s);
            let y = (s.y as f32) * inv_p;
            let p = m.step_reg(&x, y, true, (idx % 125) == 60);
            if !p.is_finite() {
                any_nan = true;
            }
        }
        let mut sq = 0.0_f64;
        for &s in &ev {
            m.reset_episode();
            let x = encode_direct_16(f, ArithOp::Add, s);
            let y = (s.y as f32) * inv_p;
            let p = m.step_reg(&x, y, false, false);
            if !p.is_finite() {
                any_nan = true;
            }
            let d = (p.clamp(0.0, 1.0) - y) as f64;
            sq += d * d;
        }
        println!(
            "  -> Bit-30 State+Weight SEU on F_{} Add ({}): Holdout RMSE={:.4} (any_nan={})",
            f.p,
            mk.label().trim(),
            (sq / (ev.len() as f64)).sqrt(),
            any_nan
        );
    }
}

// =============================================================================
// Section 3: Cyclic Multiplicative Subgroup Exponentiation & DLog in F_p^*
// (Order-12 subgroup <g^{(p-1)/12}> in F_997^* and F_9973^*, train L<=8 -> L=500!)
// =============================================================================

fn run_cyclic_subgroup_pow_dlog(f: &PrimeField, seeds: &[u64]) {
    let order = 12usize;
    assert_eq!((f.p - 1) % (order as u64), 0, "p-1 must be divisible by 12");
    let step_exp = (f.p - 1) / (order as u64);
    let gen_sub = mod_pow(f.g, step_exp, f.p);
    let gen_inv = mod_pow(gen_sub, (order - 1) as u64, f.p);
    assert_eq!((gen_sub * gen_inv) % f.p, 1);

    // Exact subgroup table: sub[k] = gen_sub^k mod p; log_sub[res] = k
    let mut sub_elems = vec![0u64; order];
    let mut cur = 1u64;
    for k in 0..order {
        sub_elems[k] = cur;
        cur = (cur * gen_sub) % f.p;
    }
    assert_eq!(cur, 1);

    println!(
        "\n=========================================================================================="
    );
    println!(
        " SECTION 3: ORDER-12 SUBGROUP <{}> <= F_{}^* (~= C_12 generator actions, train L=1..=8 -> L=500)",
        gen_sub, f.p
    );
    println!(
        "=========================================================================================="
    );
    let horizons = [6usize, 30, 100, 500];
    println!(
        "{:<30} | {:>7} | {:>12} | {:>12} | {:>12} | {:>12}",
        "Model", "RAM(KB)", "L=6", "L=30", "L=100", "L=500"
    );

    // Actions: 0 => multiply by gen_sub (k -> k+1 mod 12), 1 => multiply by gen_inv (k -> k-1 mod 12)
    let x_mul_g = [1.0_f32, 0.0];
    let x_mul_inv = [0.0_f32, 1.0];
    let n_train_eps = 1800usize;
    let n_eval_eps = 48usize;
    let p_salt = f.p.wrapping_mul(0x9E37_79B9);

    // 1) v0.4.0 Alg.Sel.Cayley 2D (in_dim=2, units=18 -> 36D state, raw one-hot inputs)
    {
        let mut res = vec![Vec::new(); horizons.len()];
        let mut ram = 0.0;
        for &seed in seeds {
            let mut m = AlgebraicSelectiveCayley::new(2, 18, order, true, seed).with_standardize(false);
            ram = m.memory_footprint_bytes() as f64 / 1024.0;
            let mut rng = splitmix64(seed ^ 0xC001_0997 ^ p_salt);
            for _ in 0..n_train_eps {
                m.reset();
                rng = splitmix64(rng);
                let len = 1 + (rng as usize % 8);
                let mut val = 1u64;
                for t in 0..len {
                    rng = splitmix64(rng);
                    let act = (rng & 1) as usize;
                    val = if act == 0 { (val * gen_sub) % f.p } else { (val * gen_inv) % f.p };
                    let y = sub_elems.iter().position(|&e| e == val).unwrap();
                    let x = if act == 0 { &x_mul_g } else { &x_mul_inv };
                    let _ = m.step_and_learn_supervised(x, if t + 1 == len { Some(y) } else { None }, 0.04, 0.015);
                }
            }
            for (hi, &h) in horizons.iter().enumerate() {
                let mut hits = 0usize;
                let mut erng = splitmix64(seed ^ (h as u64 * 0x10001) ^ p_salt);
                for _ in 0..n_eval_eps {
                    m.reset();
                    let mut val = 1u64;
                    let mut pred = 0usize;
                    for _ in 0..h {
                        erng = splitmix64(erng);
                        let act = (erng & 1) as usize;
                        val = if act == 0 { (val * gen_sub) % f.p } else { (val * gen_inv) % f.p };
                        let x = if act == 0 { &x_mul_g } else { &x_mul_inv };
                        pred = m.step_and_learn_supervised(x, None, 0.0, 0.0);
                    }
                    let y = sub_elems.iter().position(|&e| e == val).unwrap();
                    if pred == y { hits += 1; }
                }
                res[hi].push(100.0 * hits as f64 / n_eval_eps as f64);
            }
        }
        print!("{:<30} | {:7.1}", "v0.4.0 Alg.Sel.Cayley 2D", ram);
        for r in &res { let (mu, sd) = mean_std(r); print!(" | {:5.1}±{:4.1}%", mu, sd); }
        println!();
    }

    // 2) v0.5.0 NonAbelianCayleyRtrl SO(3)+Pillar2
    {
        let mut res = vec![Vec::new(); horizons.len()];
        let mut ram = 0.0;
        for &seed in seeds {
            let mut m = NonAbelianCayleyRtrl::new(9, 2, order, 0.5, seed);
            m.enable_so3_frames_end_to_end();
            m.enable_so3_proto_metric_v050(12.0, 1.2, 0.08, 0.55, 0.04, seed);
            ram = m.memory_footprint_bytes() as f64 / 1024.0;
            let mut rng = splitmix64(seed ^ 0xC001_0997 ^ p_salt);
            for _ in 0..n_train_eps {
                m.reset();
                rng = splitmix64(rng);
                let len = 1 + (rng as usize % 8);
                let mut val = 1u64;
                for t in 0..len {
                    rng = splitmix64(rng);
                    let act = (rng & 1) as usize;
                    val = if act == 0 { (val * gen_sub) % f.p } else { (val * gen_inv) % f.p };
                    let y = sub_elems.iter().position(|&e| e == val).unwrap();
                    let x = if act == 0 { &x_mul_g } else { &x_mul_inv };
                    let _ = m.step_and_learn_supervised(x, if t + 1 == len { Some(y) } else { None }, 0.04, 0.02);
                }
            }
            for (hi, &h) in horizons.iter().enumerate() {
                let mut hits = 0usize;
                let mut erng = splitmix64(seed ^ (h as u64 * 0x10001) ^ p_salt);
                for _ in 0..n_eval_eps {
                    m.reset();
                    let mut val = 1u64;
                    let mut pred = 0usize;
                    for _ in 0..h {
                        erng = splitmix64(erng);
                        let act = (erng & 1) as usize;
                        val = if act == 0 { (val * gen_sub) % f.p } else { (val * gen_inv) % f.p };
                        let x = if act == 0 { &x_mul_g } else { &x_mul_inv };
                        pred = m.step_and_learn_supervised(x, None, 0.0, 0.0);
                    }
                    let y = sub_elems.iter().position(|&e| e == val).unwrap();
                    if pred == y { hits += 1; }
                }
                res[hi].push(100.0 * hits as f64 / n_eval_eps as f64);
            }
        }
        print!("{:<30} | {:7.1}", "v0.5.0 SO(3)+Pillar2 M=3", ram);
        for r in &res { let (mu, sd) = mean_std(r); print!(" | {:5.1}±{:4.1}%", mu, sd); }
        println!();
    }

    // 3) v0.5.0 NonAbelianO4Rtrl (optimized + Q1.14 inference parity check!)
    {
        let mut res = vec![Vec::new(); horizons.len()];
        let mut res_q14 = vec![Vec::new(); horizons.len()];
        let mut ram = 0.0;
        let mut ram_q14 = 0.0;
        for &seed in seeds {
            let mut m = NonAbelianO4Rtrl::new(2, 2, order, seed);
            m.enable_o4_proto_metric(12.0, 1.4, 0.08, 0.55, 0.04);
            ram = m.memory_footprint_bytes() as f64 / 1024.0;
            let mut rng = splitmix64(seed ^ 0xC001_0997 ^ p_salt);
            for _ in 0..n_train_eps {
                m.reset();
                rng = splitmix64(rng);
                let len = 1 + (rng as usize % 8);
                let mut val = 1u64;
                for t in 0..len {
                    rng = splitmix64(rng);
                    let act = (rng & 1) as usize;
                    val = if act == 0 { (val * gen_sub) % f.p } else { (val * gen_inv) % f.p };
                    let y = sub_elems.iter().position(|&e| e == val).unwrap();
                    let x = if act == 0 { &x_mul_g } else { &x_mul_inv };
                    let _ = m.step_and_learn_supervised(x, if t + 1 == len { Some(y) } else { None }, 0.04, 0.025);
                }
            }
            let proto_q14 = m.export_proto_q14();
            let saved_bytes = m.out_w.len() * 4 - proto_q14.len() * 2;
            ram_q14 = (m.memory_footprint_bytes().saturating_sub(saved_bytes)) as f64 / 1024.0;
            for (hi, &h) in horizons.iter().enumerate() {
                let mut hits = 0usize;
                let mut hits_q = 0usize;
                let mut erng = splitmix64(seed ^ (h as u64 * 0x10001) ^ p_salt);
                for _ in 0..n_eval_eps {
                    m.reset();
                    let mut val = 1u64;
                    let mut win: Vec<&[f32]> = Vec::with_capacity(h);
                    for _ in 0..h {
                        erng = splitmix64(erng);
                        let act = (erng & 1) as usize;
                        val = if act == 0 { (val * gen_sub) % f.p } else { (val * gen_inv) % f.p };
                        win.push(if act == 0 { &x_mul_g } else { &x_mul_inv });
                    }
                    let y = sub_elems.iter().position(|&e| e == val).unwrap();
                    if m.predict_k_ahead_readonly(&win) == y { hits += 1; }
                    if m.predict_k_ahead_q14_readonly(&win, &proto_q14) == y { hits_q += 1; }
                }
                res[hi].push(100.0 * hits as f64 / n_eval_eps as f64);
                res_q14[hi].push(100.0 * hits_q as f64 / n_eval_eps as f64);
            }
        }
        print!("{:<30} | {:7.1}", "v0.5.0 NonAbelianO4Rtrl M=2", ram);
        for r in &res { let (mu, sd) = mean_std(r); print!(" | {:5.1}±{:4.1}%", mu, sd); }
        println!();
        print!("{:<30} | {:7.1}", "v0.5.0 O4 Q1.14 (i16 deploy)", ram_q14);
        for r in &res_q14 { let (mu, sd) = mean_std(r); print!(" | {:5.1}±{:4.1}%", mu, sd); }
        println!();
    }

    // 4) v0.5.0 AdaptiveOrthogonalRtrl O(d)->SO(d+1) + Label-Shuffle control
    for &shuf in &[false, true] {
        let mut res = vec![Vec::new(); horizons.len()];
        let mut dims = Vec::new();
        let mut ram = 0.0;
        for &seed in seeds {
            let mut m = AdaptiveOrthogonalRtrl::new(3, 2, order, seed);
            ram = m.memory_footprint_bytes() as f64 / 1024.0;
            let mut rng = splitmix64(seed ^ 0xC001_0997 ^ p_salt);
            for _ in 0..n_train_eps {
                m.reset();
                rng = splitmix64(rng);
                let len = 1 + (rng as usize % 8);
                let mut val = 1u64;
                let mut pred = 0usize;
                for t in 0..len {
                    rng = splitmix64(rng);
                    let act = (rng & 1) as usize;
                    val = if act == 0 { (val * gen_sub) % f.p } else { (val * gen_inv) % f.p };
                    let y = if shuf {
                        rng = splitmix64(rng);
                        (rng as usize) % order
                    } else {
                        sub_elems.iter().position(|&e| e == val).unwrap()
                    };
                    let x = if act == 0 { &x_mul_g } else { &x_mul_inv };
                    pred = m.step_supervised(x, t + 1 == len, y, 0.03, 0.04);
                }
                let y_fin = if shuf {
                    rng = splitmix64(rng);
                    (rng as usize) % order
                } else {
                    sub_elems.iter().position(|&e| e == val).unwrap()
                };
                let _ = m.observe_episode_hit(pred == y_fin, y_fin);
            }
            dims.push(m.active_dim as f64);
            for (hi, &h) in horizons.iter().enumerate() {
                let mut hits = 0usize;
                let mut erng = splitmix64(seed ^ (h as u64 * 0x10001) ^ p_salt);
                for _ in 0..n_eval_eps {
                    let mut val = 1u64;
                    let mut win: Vec<&[f32]> = Vec::with_capacity(h);
                    for _ in 0..h {
                        erng = splitmix64(erng);
                        let act = (erng & 1) as usize;
                        val = if act == 0 { (val * gen_sub) % f.p } else { (val * gen_inv) % f.p };
                        win.push(if act == 0 { &x_mul_g } else { &x_mul_inv });
                    }
                    let y = sub_elems.iter().position(|&e| e == val).unwrap();
                    if m.predict_k_ahead_readonly(&win) == y { hits += 1; }
                }
                res[hi].push(100.0 * hits as f64 / n_eval_eps as f64);
            }
        }
        let (dm, _) = mean_std(&dims);
        let lbl = if shuf {
            format!("Control Label-Shuffle (d={:.1})", dm)
        } else {
            format!("v0.5.1 AdaptiveO(d) (d={:.1})", dm)
        };
        print!("{:<30} | {:7.1}", lbl, ram);
        for r in &res { let (mu, sd) = mean_std(r); print!(" | {:5.1}±{:4.1}%", mu, sd); }
        println!();
    }
}

fn main() {
    let t0 = Instant::now();
    let f997 = PrimeField::new(997);
    let f9973 = PrimeField::new(9973);

    println!("==========================================================================================");
    println!(" BENCH STEP 9: PRIME-FIELD MODULAR ARITHMETIC (F_997 & F_9973) — v0.4.0 vs v0.4.1 vs v0.5.0/v0.5.1");
    println!("==========================================================================================");
    println!("Verified F_997 : prime=true, bits={}, primitive_root g={} (order 996 = 2^2 * 3 * 83, bijection OK)", f997.bits, f997.g);
    println!("Verified F_9973: prime=true, bits={}, primitive_root g={} (order 9972 = 2^2 * 3^2 * 277, bijection OK)", f9973.bits, f9973.g);

    let seeds = [1u64, 2, 3, 4, 5];

    run_bit_serial_suite(&f997, &seeds);
    run_bit_serial_suite(&f9973, &seeds);

    run_direct_residue_suite(&f997, &seeds);
    run_direct_residue_suite(&f9973, &seeds);

    run_cyclic_subgroup_pow_dlog(&f997, &seeds);
    run_cyclic_subgroup_pow_dlog(&f9973, &seeds);

    println!("\nBenchmark Step 9 (p997 & p9973) finished in {:.2}s.", t0.elapsed().as_secs_f64());
}
