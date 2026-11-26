// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Комплексный честный бенчмарк-контур `v0.5.0` (`Suite A / B1 / B2 / B / C / D`).
//! Запуск:
//!   cargo run --release --bin bench_step8_v050 -- --suite a
//!   cargo run --release --bin bench_step8_v050 -- --suite b1|b2|b3|b|c|d
//!   cargo run --release --bin bench_step8_v050 -- --all

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::thread;
use std::time::Instant;

use micro_continual_core::math::{rand_signed_f32, rand_unit_f32, splitmix64, SpinorEncoder};
use micro_continual_core::rtrl::{
    AlgebraicSelectiveCayley, NonAbelianCayleyRtrl, NonAbelianO4Rtrl,
};
use micro_continual_core::{
    q16_mul, q24_mul, q8_24_inv_sqrt, unlabeled_domain_cloze_logs_noisy, AdaptiveOrthogonalRtrl,
    FixedCayleyCore, FixedO4Rotor, HardPomdpConfig, HardPomdpV2, HardRegime, PageHinkleyQ39_24,
    ParaphraseSplit, Q16_16, Q39_24, Q8_24, UnifiedCayleyConfig, UnifiedCayleyCore,
    NUM_ENV_ACTIONS,
};

fn mean_sd(vals: &[f64]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    let n = vals.len() as f64;
    let m = vals.iter().sum::<f64>() / n;
    if vals.len() < 2 {
        return (m, 0.0);
    }
    let var = vals.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / ((vals.len() - 1) as f64);
    (m, var.sqrt())
}

// ---------- Группы перестановок (честные, без подбора под модель) ----------

#[derive(Clone)]
struct PermGroup {
    num_elems: usize,
    mul_gen: Vec<Vec<usize>>,
    ident_id: usize,
}

impl PermGroup {
    fn compose(a: &[u8], b: &[u8]) -> Vec<u8> {
        b.iter().map(|&idx| a[idx as usize]).collect()
    }

    fn is_even(p: &[u8]) -> bool {
        let mut inv = 0usize;
        for i in 0..p.len() {
            for j in (i + 1)..p.len() {
                if p[i] > p[j] {
                    inv += 1;
                }
            }
        }
        (inv % 2) == 0
    }

    fn c5() -> Self {
        let mut mul_gen = vec![vec![0usize; 5], vec![0usize; 5]];
        for s in 0..5usize {
            mul_gen[0][s] = (s + 1) % 5;
            mul_gen[1][s] = (s + 4) % 5;
        }
        Self {
            num_elems: 5,
            mul_gen,
            ident_id: 0,
        }
    }

    fn a5() -> Self {
        let gens: Vec<Vec<u8>> = vec![
            vec![1, 2, 0, 3, 4], // (0 1 2)
            vec![2, 0, 1, 3, 4], // (0 2 1)
            vec![1, 2, 3, 4, 0], // (0 1 2 3 4)
            vec![4, 0, 1, 2, 3], // (4 3 2 1 0)
        ];
        Self::build(5, true, &gens)
    }

    fn s5() -> Self {
        let gens: Vec<Vec<u8>> = vec![
            vec![1, 0, 2, 3, 4], // (0 1) — нечётная
            vec![0, 2, 1, 3, 4], // (1 2) — нечётная
            vec![1, 2, 3, 4, 0], // (0 1 2 3 4) — чётная
            vec![4, 0, 1, 2, 3], // (4 3 2 1 0) — чётная
        ];
        Self::build(5, false, &gens)
    }

    fn s6() -> Self {
        let gens: Vec<Vec<u8>> = vec![
            vec![1, 0, 2, 3, 4, 5], // (0 1) — нечётная
            vec![5, 1, 2, 3, 4, 0], // (0 5) — нечётная
            vec![1, 2, 3, 4, 0, 5], // (0 1 2 3 4) — чётная
            vec![4, 0, 1, 2, 3, 5], // (4 3 2 1 0) — чётная
        ];
        let g = Self::build(6, false, &gens);
        assert_eq!(g.num_elems, 720);
        g
    }

    fn build(n: usize, only_even: bool, gens: &[Vec<u8>]) -> Self {
        let mut perms: Vec<Vec<u8>> = Vec::new();
        let mut cur: Vec<u8> = (0..(n as u8)).collect();
        fn heap_perm(k: usize, arr: &mut Vec<u8>, only_even: bool, out: &mut Vec<Vec<u8>>) {
            if k == 1 {
                if !only_even || PermGroup::is_even(arr) {
                    out.push(arr.clone());
                }
                return;
            }
            heap_perm(k - 1, arr, only_even, out);
            for i in 0..(k - 1) {
                if (k % 2) == 0 {
                    arr.swap(i, k - 1);
                } else {
                    arr.swap(0, k - 1);
                }
                heap_perm(k - 1, arr, only_even, out);
            }
        }
        heap_perm(n, &mut cur, only_even, &mut perms);
        perms.sort();

        let ident_vec: Vec<u8> = (0..(n as u8)).collect();
        let ident_id = perms.binary_search(&ident_vec).unwrap();
        let mut mul_gen = Vec::with_capacity(gens.len());
        for g in gens {
            let mut row = Vec::with_capacity(perms.len());
            for p in &perms {
                let prod = Self::compose(p, g);
                row.push(perms.binary_search(&prod).unwrap());
            }
            mul_gen.push(row);
        }
        Self {
            num_elems: perms.len(),
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

// ---------- Контрольный классификатор GRU BPTT ----------

struct GruSequenceClassifier {
    x_dim: usize,
    h_dim: usize,
    num_classes: usize,
    wz: Vec<f32>,
    uz: Vec<f32>,
    bz: Vec<f32>,
    wr: Vec<f32>,
    ur: Vec<f32>,
    br: Vec<f32>,
    wn: Vec<f32>,
    un: Vec<f32>,
    bn: Vec<f32>,
    wo: Vec<f32>,
    bo: Vec<f32>,
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
            h_dim * x_dim,
            h_dim * h_dim,
            h_dim,
            h_dim * x_dim,
            h_dim * h_dim,
            h_dim,
            h_dim * x_dim,
            h_dim * h_dim,
            h_dim,
            num_classes * h_dim,
            num_classes,
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
            for j in 0..xd {
                sn += self.wn[i * xd + j] * x[j];
            }
            for j in 0..hd {
                sn += self.un[i * hd + j] * (r[j] * hp[j]);
            }
            n[i] = sn.tanh();
            hn[i] = (1.0 - z[i]) * hp[i] + z[i] * n[i];
        }
        let mut logits = vec![0.0_f32; self.num_classes];
        let mut max_l = f32::NEG_INFINITY;
        let mut best_c = 0usize;
        for c in 0..self.num_classes {
            let mut s = self.bo[c];
            for j in 0..hd {
                s += self.wo[c * hd + j] * hn[j];
            }
            logits[c] = s;
            if s > max_l {
                max_l = s;
                best_c = c;
            }
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
            for p in &mut probs {
                *p /= sum_e.max(1e-8);
            }
            self.cache.push((x.to_vec(), hp, z, r, n, hn, probs));
        }
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
            w[i] = (w[i] - lr * (m[i] / c1.max(1e-6)) / ((v[i] / c2.max(1e-6)).sqrt() + 1e-6))
                .clamp(-5.0, 5.0);
        }
    }

    fn bptt_update(&mut self, targets: &[usize], lr: f32) {
        let t_len = self.cache.len().min(targets.len());
        if t_len == 0 {
            return;
        }
        let (xd, hd, nc) = (self.x_dim, self.h_dim, self.num_classes);
        let mut grads = [
            vec![0.0; hd * xd],
            vec![0.0; hd * hd],
            vec![0.0; hd],
            vec![0.0; hd * xd],
            vec![0.0; hd * hd],
            vec![0.0; hd],
            vec![0.0; hd * xd],
            vec![0.0; hd * hd],
            vec![0.0; hd],
            vec![0.0; nc * hd],
            vec![0.0; nc],
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
                for j in 0..xd {
                    grads[3][i * xd + j] += dsr * x[j];
                }
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
// Suite A: Non-Abelian S_5 (O(4) -> SO(5) Parity Bridge) & A_5 Pillar 2
// ============================================================================

#[derive(Clone)]
struct O4SeedOut {
    accs: Vec<f64>,
    max_orth: f32,
    max_sph: f32,
    max_rq: f32,
    odd_beta: f32,
    even_beta: f32,
    seu_l100_acc: f64,
    seu_nan: bool,
    mem_bytes: usize,
}

fn eval_o4_model(
    group: &PermGroup,
    frames: usize,
    train_eps: usize,
    max_train_len: usize,
    dec_k: f32,
    eval_lens: &[usize],
    seed: u64,
    shuffle_labels: bool,
    run_seu: bool,
    odd_channels: &[usize],
    even_channels: &[usize],
) -> O4SeedOut {
    let ng = group.mul_gen.len();
    let nc = group.num_elems;
    let mut net = NonAbelianO4Rtrl::new(frames, ng, nc, seed);
    net.enable_o4_proto_metric(12.0, 1.4, 0.08, 0.55, 0.0);

    let mut rng = splitmix64(seed ^ 0x5555_00A1);
    let mut x = vec![0.0_f32; ng];
    for ep in 0..train_eps {
        let len = 1 + (ep % max_train_len);
        let (gs, st_ids) = group.gen_word(&mut rng, len);
        net.reset();
        let dec = 1.0_f32 / (1.0 + dec_k * (ep as f32) / (train_eps as f32));
        for t in 0..len {
            x.fill(0.0);
            x[gs[t]] = 1.0;
            let y = if shuffle_labels {
                rng = splitmix64(rng);
                ((rng >> 16) as usize) % nc
            } else {
                st_ids[t]
            };
            let _ = net.step_and_learn_supervised(&x, Some(y), 0.025 * dec, 0.025 * dec);
        }
    }

    let onehot: Vec<Vec<f32>> = (0..ng)
        .map(|g| {
            let mut v = vec![0.0_f32; ng];
            v[g] = 1.0;
            v
        })
        .collect();

    let eval_eps = 200usize;
    let mut accs = Vec::with_capacity(eval_lens.len());
    let mut max_orth = 0.0_f32;
    let mut max_sph = 0.0_f32;

    for &el in eval_lens {
        let mut hits = 0usize;
        let mut ev_rng = splitmix64(seed.wrapping_mul(100_003) ^ (el as u64));
        for ep_i in 0..eval_eps {
            let (gs, st_ids) = group.gen_word(&mut ev_rng, el);
            net.reset();
            let refs: Vec<&[f32]> = gs.iter().map(|&g| onehot[g].as_slice()).collect();
            let pred = net.predict_k_ahead_readonly(&refs);
            if pred == st_ids[el - 1] {
                hits += 1;
            }
            if ep_i < 20 {
                net.reset();
                for &r in &refs {
                    net.step_forward(r);
                }
                max_orth = max_orth.max(net.max_orth_drift_4d());
                max_sph = max_sph.max(net.max_sphere_drift_4d());
            }
        }
        accs.push((hits as f64) * 100.0 / (eval_eps as f64));
    }

    // Честный базисно-инвариантный замер чётности по нетривиальным верным фреймам (без оракула)
    let mut ch_betas = vec![vec![0.0_f32; frames]; ng];
    for g in 0..ng {
        net.channel_betas_into(g, &mut ch_betas[g]);
    }
    let parity_spread = |m: usize| -> f32 {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for g in 0..ng {
            let v = ch_betas[g][m];
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        mx - mn
    };
    let mut rqs = Vec::with_capacity(frames);
    for m in 0..frames {
        rqs.push(net.frame_parity_line_rq(m));
    }
    let max_rq = rqs.iter().copied().fold(0.0_f32, f32::max);
    let mut use_frames: Vec<usize> = (0..frames)
        .filter(|&m| rqs[m] > 0.97 && net.frame_acc_ema[m] > 0.85 && parity_spread(m) > 0.5)
        .collect();
    if use_frames.is_empty() {
        use_frames = (0..frames)
            .filter(|&m| rqs[m] > 0.97 && net.frame_acc_ema[m] > 0.85)
            .collect();
    }
    if use_frames.is_empty() {
        use_frames = (0..frames)
            .filter(|&m| rqs[m] > 0.97 && net.frame_acc_ema[m] > 0.70)
            .collect();
    }
    if use_frames.is_empty() {
        let best_m = (0..frames)
            .max_by(|&a, &b| rqs[a].partial_cmp(&rqs[b]).unwrap())
            .unwrap_or(0);
        use_frames.push(best_m);
    }
    let avg_channels = |chs: &[usize]| -> f32 {
        if chs.is_empty() {
            return 0.0;
        }
        let mut s = 0.0_f32;
        let mut cnt = 0usize;
        for &ch in chs {
            for &m in &use_frames {
                s += ch_betas[ch][m];
                cnt += 1;
            }
        }
        if cnt > 0 { s / (cnt as f32) } else { 0.0 }
    };
    let odd_beta = avg_channels(odd_channels);
    let even_beta = avg_channels(even_channels);

    // Стресс-тест Bit-30 State + Weight SEU на L=100
    let mut seu_l100_acc = 0.0_f64;
    let mut seu_nan = false;
    if run_seu {
        let mut hits = 0usize;
        let seu_eps = 100usize;
        let mut ev_rng = splitmix64(seed.wrapping_mul(999_983) ^ 0xB1730);
        for ep_i in 0..seu_eps {
            let mut copy = net.clone();
            copy.reset();
            let (gs, st_ids) = group.gen_word(&mut ev_rng, 100);
            for step in 0..100usize {
                if step % 50 == 25 {
                    if ((ep_i + step / 50) & 1) == 0 {
                        copy.inject_register_bit_flip(ep_i.wrapping_mul(17).wrapping_add(step), 30);
                    } else {
                        copy.inject_weight_bit_flip(ep_i.wrapping_mul(13).wrapping_add(step), 30);
                    }
                }
                copy.step_forward(&onehot[gs[step]]);
            }
            if !copy.max_orth_drift_4d().is_finite()
                || !copy.max_sphere_drift_4d().is_finite()
                || copy.u.iter().any(|v| !v.is_finite())
            {
                seu_nan = true;
            }
            if copy.predict_class() == st_ids[99] {
                hits += 1;
            }
        }
        seu_l100_acc = (hits as f64) * 100.0 / (seu_eps as f64);
    }

    O4SeedOut {
        accs,
        max_orth,
        max_sph,
        max_rq,
        odd_beta,
        even_beta,
        seu_l100_acc,
        seu_nan,
        mem_bytes: net.memory_footprint_bytes(),
    }
}

fn eval_so3_model(
    group: &PermGroup,
    frames: usize,
    v050: bool,
    tau: f32,
    center_beta: f32,
    rep_mu: f32,
    rep_lr: f32,
    train_eps: usize,
    max_train_len: usize,
    dec_k: f32,
    eval_lens: &[usize],
    seed: u64,
) -> (Vec<f64>, f32, usize) {
    let ng = group.mul_gen.len();
    let nc = group.num_elems;
    let mut net = NonAbelianCayleyRtrl::new(frames * 3, ng, nc, 1.0, seed);
    if v050 {
        net.enable_so3_proto_metric_v050(tau, center_beta, 0.08, rep_mu, rep_lr, seed);
    } else {
        net.enable_so3_proto_metric(tau, center_beta, 0.08);
    }

    let mut rng = splitmix64(seed ^ 0x5030_00A1);
    let mut x = vec![0.0_f32; ng];
    for ep in 0..train_eps {
        let len = 1 + (ep % max_train_len);
        let (gs, st_ids) = group.gen_word(&mut rng, len);
        net.reset();
        let dec = 1.0_f32 / (1.0 + dec_k * (ep as f32) / (train_eps as f32));
        for t in 0..len {
            x.fill(0.0);
            x[gs[t]] = 1.0;
            let _ = net.step_and_learn_supervised(&x, Some(st_ids[t]), 0.025 * dec, 0.025 * dec);
        }
    }

    let onehot: Vec<Vec<f32>> = (0..ng)
        .map(|g| {
            let mut v = vec![0.0_f32; ng];
            v[g] = 1.0;
            v
        })
        .collect();
    let eval_eps = 200usize;
    let mut accs = Vec::with_capacity(eval_lens.len());
    let mut max_orth = 0.0_f32;
    for &el in eval_lens {
        let mut hits = 0usize;
        let mut ev_rng = splitmix64(seed.wrapping_mul(100_003) ^ (el as u64));
        for ep_i in 0..eval_eps {
            let (gs, st_ids) = group.gen_word(&mut ev_rng, el);
            net.reset();
            for &g in &gs {
                net.step_forward(&onehot[g]);
            }
            if net.predict_class() == st_ids[el - 1] {
                hits += 1;
            }
            if ep_i < 20 {
                max_orth = max_orth.max(net.max_orth_drift());
            }
        }
        accs.push((hits as f64) * 100.0 / (eval_eps as f64));
    }
    (accs, max_orth, net.memory_footprint_bytes())
}

fn eval_selective_2d_model(
    group: &PermGroup,
    units: usize,
    train_eps: usize,
    max_train_len: usize,
    dec_k: f32,
    eval_lens: &[usize],
    seed: u64,
) -> (Vec<f64>, usize) {
    let ng = group.mul_gen.len();
    let nc = group.num_elems;
    let mut net = AlgebraicSelectiveCayley::new(ng, units, nc, true, seed).with_standardize(false);
    let mut rng = splitmix64(seed ^ 0x2D2D_00A1);
    let mut x = vec![0.0_f32; ng];
    for ep in 0..train_eps {
        let len = 1 + (ep % max_train_len);
        let (gs, st_ids) = group.gen_word(&mut rng, len);
        net.reset();
        let dec = 1.0_f32 / (1.0 + dec_k * (ep as f32) / (train_eps as f32));
        for t in 0..len {
            x.fill(0.0);
            x[gs[t]] = 1.0;
            let _ = net.step_and_learn_supervised(&x, Some(st_ids[t]), 0.03 * dec, 0.02 * dec);
        }
    }
    let onehot: Vec<Vec<f32>> = (0..ng)
        .map(|g| {
            let mut v = vec![0.0_f32; ng];
            v[g] = 1.0;
            v
        })
        .collect();
    let eval_eps = 200usize;
    let mut accs = Vec::with_capacity(eval_lens.len());
    for &el in eval_lens {
        let mut hits = 0usize;
        let mut ev_rng = splitmix64(seed.wrapping_mul(100_003) ^ (el as u64));
        for _ in 0..eval_eps {
            let (gs, st_ids) = group.gen_word(&mut ev_rng, el);
            net.reset();
            let mut pred = 0usize;
            for &g in &gs {
                pred = net.step_and_learn_supervised(&onehot[g], None, 0.0, 0.0);
            }
            if pred == st_ids[el - 1] {
                hits += 1;
            }
        }
        accs.push((hits as f64) * 100.0 / (eval_eps as f64));
    }
    (accs, net.memory_footprint_bytes())
}

fn eval_gru_model(
    group: &PermGroup,
    h_dim: usize,
    train_eps: usize,
    max_train_len: usize,
    dec_k: f32,
    eval_lens: &[usize],
    seed: u64,
) -> (Vec<f64>, usize) {
    let ng = group.mul_gen.len();
    let nc = group.num_elems;
    let mut gru = GruSequenceClassifier::new(ng, h_dim, nc, seed);
    let mut rng = splitmix64(seed ^ 0x6700_00A1);
    let mut x = vec![0.0_f32; ng];
    for ep in 0..train_eps {
        let len = 1 + (ep % max_train_len);
        let (gs, st_ids) = group.gen_word(&mut rng, len);
        gru.reset();
        let dec = 1.0_f32 / (1.0 + dec_k * (ep as f32) / (train_eps as f32));
        for t in 0..len {
            x.fill(0.0);
            x[gs[t]] = 1.0;
            let _ = gru.forward_step(&x, true);
        }
        gru.bptt_update(&st_ids, 0.005 * dec);
    }
    let onehot: Vec<Vec<f32>> = (0..ng)
        .map(|g| {
            let mut v = vec![0.0_f32; ng];
            v[g] = 1.0;
            v
        })
        .collect();
    let eval_eps = 200usize;
    let mut accs = Vec::with_capacity(eval_lens.len());
    for &el in eval_lens {
        let mut hits = 0usize;
        let mut ev_rng = splitmix64(seed.wrapping_mul(100_003) ^ (el as u64));
        for _ in 0..eval_eps {
            let (gs, st_ids) = group.gen_word(&mut ev_rng, el);
            gru.reset();
            let mut pred = 0usize;
            for &g in &gs {
                pred = gru.forward_step(&onehot[g], false);
            }
            if pred == st_ids[el - 1] {
                hits += 1;
            }
        }
        accs.push((hits as f64) * 100.0 / (eval_eps as f64));
    }
    (accs, gru.memory_footprint_bytes())
}

fn format_accs(per_seed: &[Vec<f64>]) -> String {
    if per_seed.is_empty() {
        return String::new();
    }
    let k = per_seed[0].len();
    let mut parts = Vec::with_capacity(k);
    for idx in 0..k {
        let col: Vec<f64> = per_seed.iter().map(|v| v[idx]).collect();
        let (m, s) = mean_sd(&col);
        parts.push(format!("{m:5.1}±{s:4.1}%"));
    }
    parts.join(" | ")
}

fn run_suite_a() {
    let t0 = Instant::now();
    println!("==========================================================================================");
    println!(" SUITE A: NON-ABELIAN O(4) -> SO(5) PARITY BRIDGE (S_5) & A_5 PILLAR-2 (10 SEEDS)");
    println!("==========================================================================================");

    let s5_box = PermGroup::s5();
    let s5 = &s5_box;
    let s5_lens_arr = [6usize, 12, 30, 100, 250, 500];
    let s5_lens = &s5_lens_arr[..];
    let s5_train_eps = 12000usize;
    let seeds: Vec<u64> = (1..=10).collect();

    println!(
        "\n--- Suite A1: S_5 (120 classes, train L in 1..=10, {} eps) ---",
        s5_train_eps
    );
    println!("Horizons: {:?}", s5_lens);

    let o4_m3: Vec<O4SeedOut> = thread::scope(|s| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&sd| {
                s.spawn(move || {
                    eval_o4_model(
                        s5, 3, s5_train_eps, 10, 3.5, s5_lens, sd, false, true, &[0, 1], &[2, 3],
                    )
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let o4_m4: Vec<O4SeedOut> = thread::scope(|s| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&sd| {
                s.spawn(move || {
                    eval_o4_model(
                        s5, 4, s5_train_eps, 10, 3.5, s5_lens, sd, false, true, &[0, 1], &[2, 3],
                    )
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let o4_shuf: Vec<O4SeedOut> = thread::scope(|s| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&sd| {
                s.spawn(move || {
                    eval_o4_model(
                        s5, 3, s5_train_eps, 10, 3.5, s5_lens, sd, true, false, &[0, 1], &[2, 3],
                    )
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let so3_m4_v04: Vec<(Vec<f64>, f32, usize)> = thread::scope(|s| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&sd| {
                s.spawn(move || {
                    eval_so3_model(
                        s5, 4, false, 12.0, 1.4, 0.20, 0.0, s5_train_eps, 10, 3.5, s5_lens, sd,
                    )
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let so3_m6_v05: Vec<(Vec<f64>, f32, usize)> = thread::scope(|s| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&sd| {
                s.spawn(move || {
                    eval_so3_model(
                        s5, 6, true, 12.0, 1.4, 0.65, 0.01, s5_train_eps, 10, 3.5, s5_lens, sd,
                    )
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let sel_2d: Vec<(Vec<f64>, usize)> = thread::scope(|s| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&sd| {
                s.spawn(move || eval_selective_2d_model(s5, 38, s5_train_eps, 10, 3.5, s5_lens, sd))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let gru_64: Vec<(Vec<f64>, usize)> = thread::scope(|s| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&sd| s.spawn(move || eval_gru_model(s5, 64, s5_train_eps, 10, 3.5, s5_lens, sd)))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let print_o4_stats = |label: &str, res: &[O4SeedOut]| {
        let accs: Vec<Vec<f64>> = res.iter().map(|r| r.accs.clone()).collect();
        let (odd_m, odd_s) = mean_sd(&res.iter().map(|r| r.odd_beta as f64).collect::<Vec<_>>());
        let (ev_m, ev_s) = mean_sd(&res.iter().map(|r| r.even_beta as f64).collect::<Vec<_>>());
        let (rq_m, _) = mean_sd(&res.iter().map(|r| r.max_rq as f64).collect::<Vec<_>>());
        let max_orth = res.iter().map(|r| r.max_orth).fold(0.0_f32, f32::max);
        let max_sph = res.iter().map(|r| r.max_sph).fold(0.0_f32, f32::max);
        println!(
            "{:<32} ({:5.1} KB) | {}",
            label,
            res[0].mem_bytes as f64 / 1024.0,
            format_accs(&accs)
        );
        println!(
            "   -> drift: orth={max_orth:.2e} sph={max_sph:.2e} | max_rq={rq_m:.5} | odd_beta={odd_m:.3}±{odd_s:.3} even_beta={ev_m:.3}±{ev_s:.3}"
        );
    };

    print_o4_stats("NonAbelianO4Rtrl M=3 D=75", &o4_m3);
    let (seu_m3, seu_s3) = mean_sd(&o4_m3.iter().map(|r| r.seu_l100_acc).collect::<Vec<_>>());
    let any_nan3 = o4_m3.iter().any(|r| r.seu_nan);
    println!(
        "   -> Bit-30 State+Weight SEU @L=100 (M=3): {seu_m3:.1}±{seu_s3:.1}% (any_nan={any_nan3})"
    );
    print_o4_stats("NonAbelianO4Rtrl M=4 D=100", &o4_m4);
    let (seu_m4, seu_s4) = mean_sd(&o4_m4.iter().map(|r| r.seu_l100_acc).collect::<Vec<_>>());
    let any_nan4 = o4_m4.iter().any(|r| r.seu_nan);
    println!(
        "   -> Bit-30 State+Weight SEU @L=100 (M=4): {seu_m4:.1}±{seu_s4:.1}% (any_nan={any_nan4})"
    );

    println!(
        "{:<32} ({:5.1} KB) | {}",
        "SO(3) v0.4.0 M=4 D=36",
        so3_m4_v04[0].2 as f64 / 1024.0,
        format_accs(&so3_m4_v04.iter().map(|r| r.0.clone()).collect::<Vec<_>>())
    );
    println!(
        "{:<32} ({:5.1} KB) | {}",
        "SO(3)+Pillar2 v0.5.0 M=6 D=54",
        so3_m6_v05[0].2 as f64 / 1024.0,
        format_accs(&so3_m6_v05.iter().map(|r| r.0.clone()).collect::<Vec<_>>())
    );
    println!(
        "{:<32} ({:5.1} KB) | {}",
        "Selective2D U=38 D=76",
        sel_2d[0].1 as f64 / 1024.0,
        format_accs(&sel_2d.iter().map(|r| r.0.clone()).collect::<Vec<_>>())
    );
    println!(
        "{:<32} ({:5.1} KB) | {}",
        "GRU-BPTT H=64",
        gru_64[0].1 as f64 / 1024.0,
        format_accs(&gru_64.iter().map(|r| r.0.clone()).collect::<Vec<_>>())
    );
    print_o4_stats("Label-Shuffle Control (M=3)", &o4_shuf);

    // Suite A2: A_5
    let a5_box = PermGroup::a5();
    let a5 = &a5_box;
    let a5_lens_arr = [6usize, 30, 100, 500];
    let a5_lens = &a5_lens_arr[..];
    let a5_train_eps = 5000usize;
    println!("\n--- Suite A2: A_5 (60 classes, train L in 1..=8, {} eps) ---", a5_train_eps);
    println!("Horizons: {:?}", a5_lens);

    let a5_so3_v04: Vec<(Vec<f64>, f32, usize)> = thread::scope(|s| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&sd| {
                s.spawn(move || {
                    eval_so3_model(a5, 4, false, 10.0, 1.2, 0.20, 0.0, a5_train_eps, 8, 5.0, a5_lens, sd)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let a5_so3_v05: Vec<(Vec<f64>, f32, usize)> = thread::scope(|s| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&sd| {
                s.spawn(move || {
                    eval_so3_model(a5, 4, true, 10.0, 1.2, 0.65, 0.01, a5_train_eps, 8, 5.0, a5_lens, sd)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let a5_o4_m3: Vec<O4SeedOut> = thread::scope(|s| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&sd| {
                s.spawn(move || {
                    eval_o4_model(
                        a5, 3, a5_train_eps, 8, 5.0, a5_lens, sd, false, false, &[], &[0, 1, 2, 3],
                    )
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });

    println!(
        "{:<32} ({:5.1} KB) | {}",
        "SO(3) v0.4.0 M=4 D=36",
        a5_so3_v04[0].2 as f64 / 1024.0,
        format_accs(&a5_so3_v04.iter().map(|r| r.0.clone()).collect::<Vec<_>>())
    );
    println!(
        "{:<32} ({:5.1} KB) | {}",
        "SO(3)+Pillar2 v0.5.0 M=4 D=36",
        a5_so3_v05[0].2 as f64 / 1024.0,
        format_accs(&a5_so3_v05.iter().map(|r| r.0.clone()).collect::<Vec<_>>())
    );
    print_o4_stats("NonAbelianO4Rtrl M=3 D=75", &a5_o4_m3);

    println!("\nSuite A finished in {:.2}s.", t0.elapsed().as_secs_f64());
}

// ---------- База задач для Suite B (B1 -> B2 -> B3 One-Config) ----------

const CMAPSS_RUL_MAX: f32 = 125.0;
const CMAPSS_CRIT_RUL: f32 = 30.0;

fn try_load_elec2_data() -> Option<Vec<(Vec<f32>, usize)>> {
    let candidates = [
        "electricity-normalized.arff",
        "../micro_continual_core/electricity-normalized.arff",
        "electricity.csv",
        "elecNormNew.csv",
    ];
    for p in candidates {
        if let Ok(text) = std::fs::read_to_string(Path::new(p)) {
            let mut out = Vec::new();
            let is_arff = p.ends_with(".arff") || text.contains("@relation");
            let mut in_data = !is_arff;
            for (li, line) in text.lines().enumerate() {
                let tr = line.trim();
                if tr.is_empty() || tr.starts_with('%') {
                    continue;
                }
                if is_arff && !in_data {
                    if tr.eq_ignore_ascii_case("@data") {
                        in_data = true;
                    }
                    continue;
                }
                if !is_arff && li == 0 && tr.chars().any(|c| c.is_alphabetic()) {
                    continue;
                }
                let cols: Vec<&str> = tr.split(',').collect();
                if cols.len() < 4 {
                    continue;
                }
                let mut feats = Vec::with_capacity(cols.len() - 1);
                let mut ok = true;
                for &c in &cols[..cols.len() - 1] {
                    let clean = c.trim().trim_matches('\'').trim_matches('"');
                    if let Ok(v) = clean.parse::<f32>() {
                        feats.push(v);
                    } else {
                        ok = false;
                        break;
                    }
                }
                if !ok || feats.is_empty() {
                    continue;
                }
                let lbl = cols[cols.len() - 1]
                    .trim()
                    .trim_matches('\'')
                    .trim_matches('"')
                    .to_ascii_lowercase();
                let y = if lbl == "up" || lbl == "1" || lbl == "true" { 1 } else { 0 };
                out.push((feats, y));
            }
            if out.len() >= 1000 {
                return Some(out);
            }
        }
    }
    None
}

fn augment_elec2_yprev(data: &[(Vec<f32>, usize)]) -> Vec<(Vec<f32>, usize)> {
    let mut out = Vec::with_capacity(data.len());
    let mut py = 0.0_f32;
    for (x, y) in data {
        let mut aug = Vec::with_capacity(x.len() + 1);
        aug.extend_from_slice(x);
        aug.push(py);
        out.push((aug, *y));
        py = if *y == 1 { 1.0 } else { -1.0 };
    }
    out
}

fn shuffle_data_labels(data: &[(Vec<f32>, usize)], seed: u64) -> Vec<(Vec<f32>, usize)> {
    let mut ys: Vec<usize> = data.iter().map(|d| d.1).collect();
    let mut rng = splitmix64(seed ^ 0x540F_F1E1);
    for i in (1..ys.len()).rev() {
        rng = splitmix64(rng);
        let j = (rng as usize) % (i + 1);
        ys.swap(i, j);
    }
    data.iter().zip(ys.into_iter()).map(|((x, _), y)| (x.clone(), y)).collect()
}

#[derive(Clone)]
struct CmapssUnit {
    feats: Vec<Vec<f32>>,
    true_rul: Vec<f32>,
    end_rul: f32,
}

struct CmapssData {
    name: &'static str,
    in_dim: usize,
    train: Vec<CmapssUnit>,
    test: Vec<CmapssUnit>,
}

fn load_cmapss_causal(dir: &Path, name: &'static str, multi_regime: bool, raw_mode: bool) -> CmapssData {
    let parse_rows = |p: &Path| -> Vec<(usize, usize, [f32; 3], [f32; 21])> {
        let f = File::open(p).unwrap_or_else(|e| panic!("Cannot open {}: {e}", p.display()));
        let mut rows = Vec::new();
        for l in BufReader::new(f).lines().map_while(Result::ok) {
            let parts: Vec<&str> = l.split_whitespace().collect();
            if parts.len() < 26 {
                continue;
            }
            let uid: usize = parts[0].parse().unwrap();
            let cyc: usize = parts[1].parse().unwrap();
            let mut ops = [0.0_f32; 3];
            let mut sns = [0.0_f32; 21];
            for i in 0..3 { ops[i] = parts[2 + i].parse().unwrap(); }
            for i in 0..21 { sns[i] = parts[5 + i].parse().unwrap(); }
            rows.push((uid, cyc, ops, sns));
        }
        rows
    };
    let tr_rows = parse_rows(&dir.join(format!("train_{name}.txt")));
    let te_rows = parse_rows(&dir.join(format!("test_{name}.txt")));
    let ruls: Vec<f32> = std::fs::read_to_string(dir.join(format!("RUL_{name}.txt")))
        .unwrap()
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<f32>().unwrap())
        .collect();

    let n = tr_rows.len().max(1) as f64;
    let (centroids, op_mean, op_std, s_mean, s_std, raw_mean, raw_std) = if raw_mode {
        let mut mean = [0.0_f64; 24];
        let mut var = [0.0_f64; 24];
        for &(_, _, ops, sns) in &tr_rows {
            for j in 0..3 { mean[j] += ops[j] as f64; }
            for j in 0..21 { mean[3 + j] += sns[j] as f64; }
        }
        for m in &mut mean { *m /= n; }
        for &(_, _, ops, sns) in &tr_rows {
            for j in 0..3 { let d = ops[j] as f64 - mean[j]; var[j] += d * d; }
            for j in 0..21 { let d = sns[j] as f64 - mean[3 + j]; var[3 + j] += d * d; }
        }
        let mut sd = [1.0_f64; 24];
        for j in 0..24 { sd[j] = (var[j] / n).sqrt(); }
        (vec![[0.0; 3]], [0.0; 3], [1.0; 3], vec![[0.0; 21]], vec![[1.0; 21]], mean, sd)
    } else if !multi_regime {
        let mut sm = [0.0_f64; 21];
        let mut sv = [0.0_f64; 21];
        for &(_, _, _, sns) in &tr_rows {
            for j in 0..21 { sm[j] += sns[j] as f64; }
        }
        for m in &mut sm { *m /= n; }
        for &(_, _, _, sns) in &tr_rows {
            for j in 0..21 { let d = sns[j] as f64 - sm[j]; sv[j] += d * d; }
        }
        let mut sd = [0.0_f64; 21];
        for j in 0..21 { sd[j] = (sv[j] / n).sqrt(); }
        (vec![[0.0; 3]], [0.0; 3], [1.0; 3], vec![sm], vec![sd], [0.0; 24], [1.0; 24])
    } else {
        let mut om = [0.0_f64; 3];
        let mut ov = [0.0_f64; 3];
        for &(_, _, ops, _) in &tr_rows { for j in 0..3 { om[j] += ops[j] as f64; } }
        for m in &mut om { *m /= n; }
        for &(_, _, ops, _) in &tr_rows { for j in 0..3 { let d = ops[j] as f64 - om[j]; ov[j] += d * d; } }
        let mut os = [1.0_f64; 3];
        for j in 0..3 { let s = (ov[j] / n).sqrt(); os[j] = if s > 1e-6 { s } else { 1.0 }; }
        let ops_n: Vec<[f64; 3]> = tr_rows.iter().map(|r| [(r.2[0] as f64 - om[0]) / os[0], (r.2[1] as f64 - om[1]) / os[1], (r.2[2] as f64 - om[2]) / os[2]]).collect();
        let d2 = |a: &[f64; 3], b: &[f64; 3]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
        let k = 6usize;
        let mut ctr = vec![ops_n[0]];
        for _ in 1..k {
            let mut best_i = 0; let mut best_d = -1.0_f64;
            for (idx, p) in ops_n.iter().enumerate() {
                let md = ctr.iter().map(|c| d2(p, c)).fold(f64::INFINITY, f64::min);
                if md > best_d { best_d = md; best_i = idx; }
            }
            ctr.push(ops_n[best_i]);
        }
        for _ in 0..25 {
            let mut sums = vec![[0.0_f64; 3]; k]; let mut cnts = vec![0usize; k];
            for p in &ops_n {
                let mut bi = 0; let mut bd = f64::INFINITY;
                for (ci, c) in ctr.iter().enumerate() { let d = d2(p, c); if d < bd { bd = d; bi = ci; } }
                cnts[bi] += 1;
                for j in 0..3 { sums[bi][j] += p[j]; }
            }
            for ci in 0..k { if cnts[ci] > 0 { for j in 0..3 { ctr[ci][j] = sums[ci][j] / cnts[ci] as f64; } } }
        }
        ctr.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        let mut sm = vec![[0.0_f64; 21]; k]; let mut sv = vec![[0.0_f64; 21]; k]; let mut sc = vec![0usize; k];
        for (i, r) in tr_rows.iter().enumerate() {
            let p = ops_n[i];
            let mut bi = 0; let mut bd = f64::INFINITY;
            for (ci, c) in ctr.iter().enumerate() { let d = d2(&p, c); if d < bd { bd = d; bi = ci; } }
            sc[bi] += 1;
            for j in 0..21 { sm[bi][j] += r.3[j] as f64; }
        }
        for ci in 0..k { for j in 0..21 { sm[ci][j] /= sc[ci].max(1) as f64; } }
        for r in &tr_rows {
            let p = [(r.2[0] as f64 - om[0]) / os[0], (r.2[1] as f64 - om[1]) / os[1], (r.2[2] as f64 - om[2]) / os[2]];
            let mut bi = 0; let mut bd = f64::INFINITY;
            for (ci, c) in ctr.iter().enumerate() { let d = d2(&p, c); if d < bd { bd = d; bi = ci; } }
            for j in 0..21 { let d = r.3[j] as f64 - sm[bi][j]; sv[bi][j] += d * d; }
        }
        let mut ssd = vec![[0.0_f64; 21]; k];
        for ci in 0..k { for j in 0..21 { ssd[ci][j] = (sv[ci][j] / sc[ci].max(1) as f64).sqrt(); } }
        (ctr, om, os, sm, ssd, [0.0; 24], [1.0; 24])
    };

    let xform = |ops: &[f32; 3], sns: &[f32; 21]| -> Vec<f32> {
        if raw_mode {
            let mut v = vec![0.0_f32; 24];
            for j in 0..3 {
                v[j] = if raw_std[j] < 1e-6 { 0.0 } else { ((ops[j] as f64 - raw_mean[j]) / raw_std[j]) as f32 };
            }
            for j in 0..21 {
                v[3 + j] = if raw_std[3 + j] < 1e-6 { 0.0 } else { ((sns[j] as f64 - raw_mean[3 + j]) / raw_std[3 + j]) as f32 };
            }
            return v;
        }
        let reg = if centroids.len() <= 1 {
            0
        } else {
            let p = [(ops[0] as f64 - op_mean[0]) / op_std[0], (ops[1] as f64 - op_mean[1]) / op_std[1], (ops[2] as f64 - op_mean[2]) / op_std[2]];
            let mut bi = 0; let mut bd = f64::INFINITY;
            for (ci, c) in centroids.iter().enumerate() {
                let d = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2);
                if d < bd { bd = d; bi = ci; }
            }
            bi
        };
        (0..21).map(|j| {
            let st = s_std[reg][j];
            if st < 1e-6 { 0.0 } else { ((sns[j] as f64 - s_mean[reg][j]) / st) as f32 }
        }).collect()
    };

    let build_units = |rows: &[(usize, usize, [f32; 3], [f32; 21])], ruls_opt: Option<&[f32]>| -> Vec<CmapssUnit> {
        let max_u = rows.iter().map(|r| r.0).max().unwrap_or(0);
        let mut by_u = vec![Vec::new(); max_u + 1];
        for r in rows { by_u[r.0].push(r); }
        let mut out = Vec::new();
        for u in 1..=max_u {
            if by_u[u].is_empty() { continue; }
            by_u[u].sort_by_key(|r| r.1);
            let max_c = by_u[u].last().unwrap().1;
            let er = ruls_opt.map_or(0.0, |v| v[u - 1]);
            let feats = by_u[u].iter().map(|r| xform(&r.2, &r.3)).collect();
            let tr = by_u[u].iter().map(|r| (er + (max_c - r.1) as f32).min(CMAPSS_RUL_MAX)).collect();
            out.push(CmapssUnit { feats, true_rul: tr, end_rul: er.min(CMAPSS_RUL_MAX) });
        }
        out
    };

    CmapssData {
        name,
        in_dim: if raw_mode { 24 } else { 21 },
        train: build_units(&tr_rows, None),
        test: build_units(&te_rows, Some(&ruls)),
    }
}

fn eval_elec2_preset(data: &[(Vec<f32>, usize)], cfg_fn: fn(usize, usize, u64) -> UnifiedCayleyConfig, seeds: &[u64]) -> (f64, f64, f64, f64) {
    let d_in = data[0].0.len();
    let res: Vec<(f64, f64, f64)> = thread::scope(|s| {
        let hs: Vec<_> = seeds.iter().map(|&sd| {
            s.spawn(move || {
                let mut core = UnifiedCayleyCore::new(cfg_fn(2, d_in, sd));
                core.start_episode();
                let mut hits = 0usize;
                let t0 = Instant::now();
                for (x, y) in data {
                    let (p, _) = core.step_supervised_class(x, *y, true, true, false);
                    if p == *y { hits += 1; }
                }
                let lat_us = t0.elapsed().as_secs_f64() * 1e6 / data.len() as f64;
                (hits as f64 * 100.0 / data.len() as f64, core.memory_footprint_bytes() as f64 / 1024.0, lat_us)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let accs: Vec<f64> = res.iter().map(|r| r.0).collect();
    let lats: Vec<f64> = res.iter().map(|r| r.2).collect();
    let (m, sd) = mean_sd(&accs);
    let (lm, _) = mean_sd(&lats);
    (m, sd, res[0].1, lm)
}

#[derive(Clone, Default)]
struct NasaEvalOut {
    rmse_m: f64,
    rmse_sd: f64,
    s_score: f64,
    ew_recall: f64,
    ram_kb: f64,
    lat_us: f64,
    any_nan: bool,
}

fn eval_nasa_preset(ds: &CmapssData, passes: usize, cfg_fn: fn(usize, usize, u64) -> UnifiedCayleyConfig, seu_mode: bool, shuffle_rul: bool, seeds: &[u64]) -> NasaEvalOut {
    let mut flat_ruls: Vec<f32> = Vec::new();
    if shuffle_rul {
        for u in &ds.train {
            for &r in &u.true_rul { flat_ruls.push(r / CMAPSS_RUL_MAX); }
        }
        let mut rng = splitmix64(0x901_580F);
        for i in (1..flat_ruls.len()).rev() {
            rng = splitmix64(rng);
            let j = (rng as usize) % (i + 1);
            flat_ruls.swap(i, j);
        }
    }

    let res: Vec<(f64, f64, f64, f64, f64, bool)> = thread::scope(|s| {
        let hs: Vec<_> = seeds.iter().map(|&sd| {
            let ds_ref = ds;
            let shuf_ref = &flat_ruls;
            s.spawn(move || {
                let mut core = UnifiedCayleyCore::new(cfg_fn(2, ds_ref.in_dim, sd));
                let mut total_cyc = 0usize;
                let t0 = Instant::now();
                let mut flat_pos = 0usize;
                for pass in 0..passes {
                    let mut order: Vec<usize> = (0..ds_ref.train.len()).collect();
                    let mut rng = splitmix64(sd.wrapping_mul(131) ^ (pass as u64 + 17));
                    for i in (1..order.len()).rev() {
                        rng = splitmix64(rng);
                        let j = (rng as usize) % (i + 1);
                        order.swap(i, j);
                    }
                    for &uid in &order {
                        let unit = &ds_ref.train[uid];
                        core.start_episode();
                        for (ci, feat) in unit.feats.iter().enumerate() {
                            let yn = if shuffle_rul {
                                let v = shuf_ref[flat_pos % shuf_ref.len()];
                                flat_pos += 1;
                                v
                            } else {
                                unit.true_rul[ci] / CMAPSS_RUL_MAX
                            };
                            let _ = core.step_supervised_reg(feat, yn, true, true, false);
                            total_cyc += 1;
                        }
                    }
                }
                let lat_us = t0.elapsed().as_secs_f64() * 1e6 / total_cyc.max(1) as f64;
                let mut se = 0.0_f64; let mut ss = 0.0_f64;
                let mut tp = 0usize; let mut fn_c = 0usize;
                let mut any_nan = false;
                for (u_idx, unit) in ds_ref.test.iter().enumerate() {
                    let mut m = core.clone();
                    m.start_episode();
                    let mut last_p = 0.0_f32;
                    for (ci, feat) in unit.feats.iter().enumerate() {
                        let inj = seu_mode && ci >= 9 && ((ci - 9) % 25 == 0);
                        if inj {
                            m.inject_weight_bit30_seu(u_idx % m.slots.len());
                        }
                        last_p = m.step_supervised_reg(feat, 0.0, false, false, seu_mode);
                    }
                    if !last_p.is_finite() { any_nan = true; }
                    let p_rul = (last_p.clamp(0.0, 1.0) * CMAPSS_RUL_MAX) as f64;
                    let y = unit.end_rul as f64;
                    let d = p_rul - y;
                    se += d * d;
                    ss += if d < 0.0 { (-d / 13.0).exp() - 1.0 } else { (d / 10.0).exp() - 1.0 };
                    if y <= CMAPSS_CRIT_RUL as f64 {
                        if p_rul <= CMAPSS_CRIT_RUL as f64 { tp += 1; } else { fn_c += 1; }
                    }
                }
                let rmse = (se / ds_ref.test.len().max(1) as f64).sqrt();
                let rec = if tp + fn_c > 0 { tp as f64 * 100.0 / (tp + fn_c) as f64 } else { 0.0 };
                (rmse, ss, rec, core.memory_footprint_bytes() as f64 / 1024.0, lat_us, any_nan)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let rmses: Vec<f64> = res.iter().map(|r| r.0).collect();
    let sss: Vec<f64> = res.iter().map(|r| r.1).collect();
    let recs: Vec<f64> = res.iter().map(|r| r.2).collect();
    let lats: Vec<f64> = res.iter().map(|r| r.4).collect();
    let (rm, rsd) = mean_sd(&rmses);
    NasaEvalOut {
        rmse_m: rm,
        rmse_sd: rsd,
        s_score: mean_sd(&sss).0,
        ew_recall: mean_sd(&recs).0,
        ram_kb: res[0].3,
        lat_us: mean_sd(&lats).0,
        any_nan: res.iter().any(|r| r.5),
    }
}

struct CartPoleSim {
    x: f32,
    xdot: f32,
    theta: f32,
    theta_dot: f32,
    steps: usize,
}

impl CartPoleSim {
    fn new(rng: &mut u64) -> Self {
        Self {
            x: rand_signed_f32(rng) * 0.04,
            xdot: rand_signed_f32(rng) * 0.04,
            theta: rand_signed_f32(rng) * 0.04,
            theta_dot: rand_signed_f32(rng) * 0.04,
            steps: 0,
        }
    }
    fn obs(&self) -> [f32; 2] { [self.x / 2.4, self.theta / 0.2095] }
    fn step(&mut self, action: usize, flipped: bool) -> bool {
        let g = 9.8_f32; let mc = 1.0_f32; let mp = 0.1_f32; let mt = mc + mp;
        let l = 0.5_f32; let mpl = mp * l; let tau = 0.02_f32;
        let raw_dir = if action == 1 { 1.0_f32 } else { -1.0_f32 };
        let force = 10.0 * if flipped { -raw_dir } else { raw_dir };
        let ct = self.theta.cos(); let st = self.theta.sin();
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

#[derive(Clone, Default)]
struct CartPoleOut {
    tr_score_m: f64,
    tr_score_sd: f64,
    tr_ret_m: f64,
    tr_ret_sd: f64,
    ev_score: f64,
    ev_ret: f64,
    ram_kb: f64,
}

fn eval_cartpole_preset(use_dx: bool, cfg_fn: fn(usize, usize, u64) -> UnifiedCayleyConfig, seeds: &[u64]) -> CartPoleOut {
    let in_dim = if use_dx { 4 } else { 2 };
    let res: Vec<(f64, f64, f64, f64, f64)> = thread::scope(|s| {
        let hs: Vec<_> = seeds.iter().map(|&sd| {
            s.spawn(move || {
                let mut core = UnifiedCayleyCore::new(cfg_fn(2, in_dim, sd));
                let mut rng = splitmix64(sd ^ 0xCA27_901E);
                let mut phase_tr = Vec::new();
                let mut phase_ev = Vec::new();
                for (flipped, cnt) in [(false, 140usize), (true, 140), (false, 80)] {
                    let mut surv = Vec::with_capacity(cnt);
                    for _ in 0..cnt {
                        let mut env = CartPoleSim::new(&mut rng);
                        core.start_episode();
                        let mut prev = [0.0_f32; 2];
                        let mut has_prev = false;
                        loop {
                            let o = env.obs();
                            let raw_vec: Vec<f32> = if use_dx {
                                let dx0 = if has_prev { (o[0] - prev[0]) * 10.0 } else { 0.0 };
                                let dx1 = if has_prev { (o[1] - prev[1]) * 10.0 } else { 0.0 };
                                prev = o; has_prev = true;
                                vec![o[0], o[1], dx0.clamp(-4.0, 4.0), dx1.clamp(-4.0, 4.0)]
                            } else {
                                vec![o[0], o[1]]
                            };
                            let act = core.decide_rl(&raw_vec, true, false, false).0;
                            let done = env.step(act, flipped);
                            let r = if done && env.steps < 200 { -1.0_f32 } else { 0.06_f32 };
                            core.observe_rl_reward(r, &[], done, true);
                            if done { break; }
                        }
                        let norm_01 = ((env.steps as f32) / 140.0).clamp(0.0, 1.0);
                        let _ = core.end_episode_score(norm_01, true);
                        surv.push(env.steps as f64);
                    }
                    let tail40 = surv[surv.len() - 40..].iter().sum::<f64>() / 40.0;
                    phase_tr.push(tail40);

                    let mut erng = splitmix64(rng ^ 0xE7A1_0001);
                    let mut esum = 0.0_f64;
                    for _ in 0..30usize {
                        let mut env = CartPoleSim::new(&mut erng);
                        core.start_episode();
                        let mut prev = [0.0_f32; 2];
                        let mut has_prev = false;
                        loop {
                            let o = env.obs();
                            let raw_vec: Vec<f32> = if use_dx {
                                let dx0 = if has_prev { (o[0] - prev[0]) * 10.0 } else { 0.0 };
                                let dx1 = if has_prev { (o[1] - prev[1]) * 10.0 } else { 0.0 };
                                prev = o; has_prev = true;
                                vec![o[0], o[1], dx0.clamp(-4.0, 4.0), dx1.clamp(-4.0, 4.0)]
                            } else {
                                vec![o[0], o[1]]
                            };
                            let act = core.decide_rl(&raw_vec, false, false, false).0;
                            if env.step(act, flipped) { break; }
                        }
                        esum += env.steps as f64;
                    }
                    phase_ev.push(esum / 30.0);
                }
                let tr_sc = 0.5 * (phase_tr[0] + phase_tr[1]);
                let tr_ret = phase_tr[2];
                let ev_sc = 0.5 * (phase_ev[0] + phase_ev[1]);
                let ev_ret = phase_ev[2];
                (tr_sc, tr_ret, ev_sc, ev_ret, core.memory_footprint_bytes() as f64 / 1024.0)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let tr_scs: Vec<f64> = res.iter().map(|r| r.0).collect();
    let tr_rts: Vec<f64> = res.iter().map(|r| r.1).collect();
    let ev_scs: Vec<f64> = res.iter().map(|r| r.2).collect();
    let ev_rts: Vec<f64> = res.iter().map(|r| r.3).collect();
    CartPoleOut {
        tr_score_m: mean_sd(&tr_scs).0,
        tr_score_sd: mean_sd(&tr_scs).1,
        tr_ret_m: mean_sd(&tr_rts).0,
        tr_ret_sd: mean_sd(&tr_rts).1,
        ev_score: mean_sd(&ev_scs).0,
        ev_ret: mean_sd(&ev_rts).0,
        ram_kb: res[0].4,
    }
}

fn run_suite_b1() {
    let t0 = Instant::now();
    println!("==========================================================================================");
    println!(" SUITE B1 (Checkpoint 1): MasterEq + 2D/3D + Newton RMSNorm + BandGates + var+1");
    println!("==========================================================================================");
    let seeds: Vec<u64> = (1..=10).collect();

    if let Some(elec) = try_load_elec2_data() {
        let elec_y = augment_elec2_yprev(&elec);
        let elec_shuf = shuffle_data_labels(&elec, 42);
        let r_raw = eval_elec2_preset(&elec, UnifiedCayleyConfig::checkpoint_b1, &seeds[..5]);
        let r_yp = eval_elec2_preset(&elec_y, UnifiedCayleyConfig::checkpoint_b1, &seeds[..5]);
        let r_sh = eval_elec2_preset(&elec_shuf, UnifiedCayleyConfig::checkpoint_b1, &seeds[..5]);
        println!("\n--- [B1-1] ELEC2 ({} rows, 5 seeds, RAM={:.1} KB) ---", elec.len(), r_raw.2);
        println!("  ELEC2 raw      : {:.2}±{:.2}% ({:.2} us/step)", r_raw.0, r_raw.1, r_raw.3);
        println!("  ELEC2 +y_{{t-1}}: {:.2}±{:.2}% ({:.2} us/step)", r_yp.0, r_yp.1, r_yp.3);
        println!("  ELEC2 Shuffled : {:.2}±{:.2}%", r_sh.0, r_sh.1);
    }

    let cmapss_dir = Path::new("CMAPSSData");
    if cmapss_dir.exists() {
        println!("\n--- [B1-2] NASA C-MAPSS FD001 & FD004 (1-pass & 3-pass, 10 seeds) ---");
        let fd1 = load_cmapss_causal(cmapss_dir, "FD001", false, false);
        let fd4 = load_cmapss_causal(cmapss_dir, "FD004", true, false);
        let fd4_raw = load_cmapss_causal(cmapss_dir, "FD004", false, true);
        for ds in [&fd1, &fd4, &fd4_raw] {
            for &passes in &[1usize, 3usize] {
                let r = eval_nasa_preset(ds, passes, UnifiedCayleyConfig::checkpoint_b1, false, false, &seeds);
                let tag = if ds.in_dim == 24 { "FD004-RAW" } else { ds.name };
                println!(
                    "  {:<9} {}-pass | RMSE={:5.2}±{:4.2} cyc | S-Scr={:7.1} | EW-Rec={:5.1}% | RAM={:5.1}KB | {:.2} us",
                    tag, passes, r.rmse_m, r.rmse_sd, r.s_score, r.ew_recall, r.ram_kb, r.lat_us
                );
            }
        }
        let r_seu = eval_nasa_preset(&fd4, 3, UnifiedCayleyConfig::checkpoint_b1, true, false, &seeds);
        let r_shuf = eval_nasa_preset(&fd1, 3, UnifiedCayleyConfig::checkpoint_b1, false, true, &seeds);
        println!("  FD004 Bit-30 SEU (3-pass): RMSE={:5.2}±{:4.2} | EW-Rec={:5.1}% | any_nan={}", r_seu.rmse_m, r_seu.rmse_sd, r_seu.ew_recall, r_seu.any_nan);
        println!("  FD001 RUL-Shuffle Control: RMSE={:5.2}±{:4.2} (chance floor)", r_shuf.rmse_m, r_shuf.rmse_sd);
    }

    println!("\n--- [B1-3] Position-Only CartPole 2D & +dx (A->B->A, Train & Frozen Eval, 10 seeds) ---");
    let cp2d = eval_cartpole_preset(false, UnifiedCayleyConfig::checkpoint_b1, &seeds);
    let cpdx = eval_cartpole_preset(true, UnifiedCayleyConfig::checkpoint_b1, &seeds);
    println!(
        "  CartPole 2D (no dx): TrSc={:5.1}±{:4.1} | TrRet={:5.1}±{:4.1} | Ev(S/R)={:5.1}/{:5.1} | RAM={:5.1}KB",
        cp2d.tr_score_m, cp2d.tr_score_sd, cp2d.tr_ret_m, cp2d.tr_ret_sd, cp2d.ev_score, cp2d.ev_ret, cp2d.ram_kb
    );
    println!(
        "  CartPole +dx       : TrSc={:5.1}±{:4.1} | TrRet={:5.1}±{:4.1} | Ev(S/R)={:5.1}/{:5.1} | RAM={:5.1}KB",
        cpdx.tr_score_m, cpdx.tr_score_sd, cpdx.tr_ret_m, cpdx.tr_ret_sd, cpdx.ev_score, cpdx.ev_ret, cpdx.ram_kb
    );
    println!("\nSuite B1 finished in {:.2}s.", t0.elapsed().as_secs_f64());
}

// ---------- Дополнительные эталонные задачи Stage B2 / B3 ----------

fn eval_repeat_prev_preset(k_delay: usize, cfg_fn: fn(usize, usize, u64) -> UnifiedCayleyConfig, seeds: &[u64]) -> (f64, f64, f64) {
    let ep_len = (k_delay + 18).max(24);
    let n_eps = 220usize + k_delay * 5;
    let res: Vec<(f64, f64)> = thread::scope(|s| {
        let hs: Vec<_> = seeds.iter().map(|&sd| {
            s.spawn(move || {
                let mut core = UnifiedCayleyCore::new(cfg_fn(4, 4, sd));
                let mut rng = splitmix64(sd ^ 0x2E9E_A700 ^ (k_delay as u64));
                let mut ep_accs = Vec::with_capacity(n_eps);
                for ep in 0..n_eps {
                    let train_flag = ep < n_eps - 30;
                    core.start_episode();
                    let mut hist = Vec::with_capacity(ep_len);
                    let mut hits = 0usize;
                    let mut eval_n = 0usize;
                    for t in 0..ep_len {
                        rng = splitmix64(rng);
                        let sym = ((rng >> 16) as usize) % 4;
                        hist.push(sym);
                        let mut obs = [0.0_f32; 4];
                        obs[sym] = 1.0;
                        let pred = core.decide_rl(&obs, train_flag, false, false).0;
                        let r = if t >= k_delay {
                            let ok = pred == hist[t - k_delay];
                            if ok { hits += 1; }
                            eval_n += 1;
                            if ok { 1.0_f32 } else { -1.0_f32 }
                        } else {
                            0.0_f32
                        };
                        core.observe_rl_reward(r, &[], t + 1 == ep_len, train_flag);
                    }
                    let acc = hits as f64 * 100.0 / eval_n.max(1) as f64;
                    let _ = core.end_episode_score((acc / 100.0) as f32, train_flag);
                    ep_accs.push(acc);
                }
                let ev_tail = ep_accs[ep_accs.len() - 30..].iter().sum::<f64>() / 30.0;
                (ev_tail, core.memory_footprint_bytes() as f64 / 1024.0)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let accs: Vec<f64> = res.iter().map(|r| r.0).collect();
    let (m, sd) = mean_sd(&accs);
    (m, sd, res[0].1)
}

fn gen_dyck2_canonical(rng: &mut u64, len: usize, max_depth: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut tokens = Vec::with_capacity(len);
    let mut targets = Vec::with_capacity(len);
    let mut depths = Vec::with_capacity(len);
    let mut stack: Vec<u8> = Vec::new();
    for t in 0..len {
        *rng = splitmix64(*rng);
        let rem = len - t;
        let can_push = stack.len() < max_depth && stack.len() + 1 <= rem;
        let can_pop = !stack.is_empty();
        let do_push = if !can_pop { true } else if !can_push { false } else { ((*rng >> 12) & 3) != 0 };
        let tok = if do_push {
            let open_t = ((*rng >> 18) as usize) & 1;
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

fn eval_dyck2_preset(cfg_fn: fn(usize, usize, u64) -> UnifiedCayleyConfig, shuffle_labels: bool, seeds: &[u64]) -> (f64, f64, f64, f64, f64) {
    let res: Vec<(f64, f64, f64)> = thread::scope(|s| {
        let hs: Vec<_> = seeds.iter().map(|&sd| {
            s.spawn(move || {
                let mut core = UnifiedCayleyCore::new(cfg_fn(3, 4, sd));
                let mut rng = splitmix64(sd ^ 0xD7C2_0001);
                for _ in 0..1600usize {
                    let (toks, tgts, _) = gen_dyck2_canonical(&mut rng, 24, 5);
                    core.start_episode();
                    for t in 0..toks.len() {
                        let mut x = [0.0_f32; 4];
                        x[toks[t]] = 1.0;
                        let y = if shuffle_labels { rng = splitmix64(rng); (rng as usize) % 3 } else { tgts[t] };
                        let _ = core.step_supervised_class(&x, y, true, true, false);
                    }
                }
                let mut all_hits = 0usize; let mut all_n = 0usize;
                let mut d3_hits = 0usize; let mut d3_n = 0usize;
                for _ in 0..400usize {
                    let (toks, tgts, deps) = gen_dyck2_canonical(&mut rng, 24, 5);
                    core.start_episode();
                    for t in 0..toks.len() {
                        let mut x = [0.0_f32; 4];
                        x[toks[t]] = 1.0;
                        let pred = core.step_supervised_class(&x, 0, false, false, false).0;
                        let y = if shuffle_labels { rng = splitmix64(rng); (rng as usize) % 3 } else { tgts[t] };
                        if pred == y { all_hits += 1; }
                        all_n += 1;
                        if deps[t] >= 3 {
                            if pred == y { d3_hits += 1; }
                            d3_n += 1;
                        }
                    }
                }
                (
                    all_hits as f64 * 100.0 / all_n.max(1) as f64,
                    d3_hits as f64 * 100.0 / d3_n.max(1) as f64,
                    core.memory_footprint_bytes() as f64 / 1024.0,
                )
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let all_v: Vec<f64> = res.iter().map(|r| r.0).collect();
    let d3_v: Vec<f64> = res.iter().map(|r| r.1).collect();
    (mean_sd(&all_v).0, mean_sd(&all_v).1, mean_sd(&d3_v).0, mean_sd(&d3_v).1, res[0].2)
}

fn gen_wave_ep_canon(rng: &mut u64, wa: f32, wb: f32, wc: f32, ep_len: usize, sign_flip: bool) -> (Vec<f32>, Vec<usize>) {
    let pi2 = std::f32::consts::TAU;
    let a_sign0 = if rand_unit_f32(rng) < 0.5 { 1.0_f32 } else { -1.0_f32 };
    let flip_at = 16usize + ((splitmix64(*rng) as usize) % 12);
    *rng = splitmix64(*rng);
    let phi_b = rand_unit_f32(rng) * pi2;
    let phi_c = rand_unit_f32(rng) * pi2;
    let eval_wave = |t: usize| -> f32 {
        let sa = if t < flip_at { a_sign0 } else { -a_sign0 };
        let sb = (pi2 * (t as f32) / 8.0 + phi_b).cos();
        let sc = (pi2 * (t as f32) / 3.0 + phi_c).cos();
        wa * sa + wb * sb + wc * sc
    };
    let mut obs = Vec::with_capacity(ep_len);
    let mut tgts = Vec::with_capacity(ep_len);
    for t in 0..ep_len {
        obs.push(eval_wave(t) + rand_signed_f32(rng) * 0.06);
        let raw_lbl = if eval_wave(t + 1) >= 0.0 { 1usize } else { 0usize };
        tgts.push(if sign_flip { 1 - raw_lbl } else { raw_lbl });
    }
    (obs, tgts)
}

fn eval_waves_preset(cfg_fn: fn(usize, usize, u64) -> UnifiedCayleyConfig, seeds: &[u64]) -> (f64, f64, f64, f64) {
    let res: Vec<(f64, f64, f64, f64)> = thread::scope(|s| {
        let hs: Vec<_> = seeds.iter().map(|&sd| {
            s.spawn(move || {
                // Zero-Shot ABC -> {A, B, C, AB}
                let mut core = UnifiedCayleyCore::new(cfg_fn(2, 1, sd));
                let mut rng = splitmix64(sd ^ 0xBA7E_509E);
                let ep_len = 48usize;
                for _ in 0..220usize {
                    let (obs, tgts) = gen_wave_ep_canon(&mut rng, 1.0, 1.0, 1.0, ep_len, false);
                    core.start_episode();
                    for t in 0..ep_len {
                        let _ = core.step_supervised_class(&[obs[t]], tgts[t], t >= 6, true, false);
                    }
                }
                let cfgs = [(1.0_f32, 1.0_f32, 1.0_f32), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)];
                let mut zs_scores = [0.0_f64; 5];
                let mut erng = splitmix64(sd ^ 0xE7A1_0001);
                for (ci, &(wa, wb, wc)) in cfgs.iter().enumerate() {
                    let mut hits = 0usize; let mut tot = 0usize;
                    for _ in 0..40usize {
                        let (obs, tgts) = gen_wave_ep_canon(&mut erng, wa, wb, wc, ep_len, false);
                        core.start_episode();
                        for t in 0..ep_len {
                            let p = core.step_supervised_class(&[obs[t]], 0, false, false, false).0;
                            if t >= 6 { if p == tgts[t] { hits += 1; } tot += 1; }
                        }
                    }
                    zs_scores[ci] = hits as f64 * 100.0 / tot.max(1) as f64;
                }
                let zs_mean = (zs_scores[1] + zs_scores[2] + zs_scores[3] + zs_scores[4]) * 0.25;

                // Continual Flip A -> B -> A
                let mut ccore = UnifiedCayleyCore::new(cfg_fn(2, 1, sd ^ 0xF119));
                let mut crng = splitmix64(sd ^ 0xC071_5CA1);
                let mut run_reg = |flip: bool, cnt: usize| -> Vec<f64> {
                    let mut accs = Vec::with_capacity(cnt);
                    for _ in 0..cnt {
                        let (obs, tgts) = gen_wave_ep_canon(&mut crng, 1.0, 1.0, 1.0, ep_len, flip);
                        ccore.start_episode();
                        let mut hits = 0usize; let mut n = 0usize;
                        for t in 0..ep_len {
                            let p = ccore.step_supervised_class(&[obs[t]], tgts[t], t >= 6, true, false).0;
                            if t >= 6 { if p == tgts[t] { hits += 1; } n += 1; }
                        }
                        let acc = hits as f64 * 100.0 / n.max(1) as f64;
                        let _ = ccore.end_episode_score((acc / 100.0) as f32, true);
                        accs.push(acc);
                    }
                    accs
                };
                let ra = run_reg(false, 180);
                let rb = run_reg(true, 180);
                let rret = run_reg(false, 40);
                let tail40 = |v: &[f64]| v[v.len() - 40..].iter().sum::<f64>() / 40.0;
                let cont_sc = 0.5 * (tail40(&ra) + tail40(&rb));
                let ret12 = rret[..12].iter().sum::<f64>() / 12.0;
                (zs_scores[0], zs_mean, cont_sc, ret12)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let abc: Vec<f64> = res.iter().map(|r| r.0).collect();
    let zs: Vec<f64> = res.iter().map(|r| r.1).collect();
    let cs: Vec<f64> = res.iter().map(|r| r.2).collect();
    let rt: Vec<f64> = res.iter().map(|r| r.3).collect();
    (mean_sd(&abc).0, mean_sd(&zs).0, mean_sd(&cs).0, mean_sd(&rt).0)
}

fn eval_hard_pomdp_preset(sparse: bool, cfg_fn: fn(usize, usize, u64) -> UnifiedCayleyConfig, seeds: &[u64]) -> (f64, f64, f64, f64) {
    let logs = unlabeled_domain_cloze_logs_noisy();
    let res: Vec<(f64, f64, f64, f64)> = thread::scope(|s| {
        let hs: Vec<_> = seeds.iter().map(|&sd| {
            let logs_ref = &logs;
            s.spawn(move || {
                let mut enc = SpinorEncoder::new(4, 6); // 24D семантически-структурное кодирование
                enc.adapt_unlabeled_cloze(logs_ref, 35);
                enc.freeze_to_i8();
                let mut fv = [0.0_f32; 24];
                let mut tmp_slots: Vec<(usize, f32)> = Vec::with_capacity(16);
                let mut tmp_sem = [0.0_f32; 12];

                let mut core = UnifiedCayleyCore::new(cfg_fn(NUM_ENV_ACTIONS, 24, sd));
                let ep_n = if sparse { 320usize } else { 220usize };
                let mut run_phase = |reg: HardRegime, sp: ParaphraseSplit, cnt: usize, base_sd: u64, train: bool| -> Vec<bool> {
                    let mut solved = Vec::with_capacity(cnt);
                    for i in 0..cnt {
                        let mut cfg = HardPomdpConfig::strict_honest(reg, 15, sp);
                        cfg.sparse_reward = sparse;
                        let mut ep = HardPomdpV2::generate(base_sd + i as u64, cfg);
                        core.start_episode();
                        let mut obs = ep.initial_observation();
                        let mut ep_sol = false;
                        while ep.step_idx < ep.config.max_steps {
                            enc.encode_text_into(&obs, &mut fv, &mut tmp_slots, &mut tmp_sem);
                            let act = core.decide_rl(&fv, train, false, false).0;
                            let sr = ep.step(act);
                            let out = sr.get();
                            core.observe_rl_reward(out.reward, &[], out.done, train);
                            if out.done {
                                ep_sol = out.solved;
                                break;
                            }
                            obs = out.observation.clone();
                        }
                        let _ = core.end_episode_score(if ep_sol { 1.0 } else { 0.0 }, train);
                        solved.push(ep_sol);
                    }
                    solved
                };
                let sb = sd * 100_000;
                let sa = run_phase(HardRegime::Alpha, ParaphraseSplit::CanonicalTrain, ep_n, sb + 1, true);
                let sh = run_phase(HardRegime::Alpha, ParaphraseSplit::UnseenHoldout, 60, sb + 10_000, false);
                let s_b = run_phase(HardRegime::Beta, ParaphraseSplit::CanonicalTrain, ep_n, sb + 20_000, true);
                let sg = run_phase(HardRegime::Gamma, ParaphraseSplit::CanonicalTrain, ep_n, sb + 30_000, true);
                let sr = run_phase(HardRegime::Alpha, ParaphraseSplit::CanonicalTrain, 100, sb + 40_000, true);
                let tail80 = |v: &[bool]| -> f64 {
                    let k = v.len().min(80);
                    v[v.len() - k..].iter().filter(|&&b| b).count() as f64 * 100.0 / k as f64
                };
                let online = (tail80(&sa) + tail80(&s_b) + tail80(&sg)) / 3.0;
                let ret = tail80(&sr);
                let hold = sh.iter().filter(|&&b| b).count() as f64 * 100.0 / sh.len() as f64;
                (online, ret, hold, (core.memory_footprint_bytes() + enc.memory_footprint_bytes()) as f64 / 1024.0)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let on: Vec<f64> = res.iter().map(|r| r.0).collect();
    let rt: Vec<f64> = res.iter().map(|r| r.1).collect();
    let hd: Vec<f64> = res.iter().map(|r| r.2).collect();
    (mean_sd(&on).0, mean_sd(&rt).0, mean_sd(&hd).0, res[0].3)
}

fn eval_tmaze_preset(corridor_len: usize, cfg_fn: fn(usize, usize, u64) -> UnifiedCayleyConfig, seeds: &[u64]) -> (f64, f64, f64) {
    let noise_amp = if corridor_len >= 10 { 0.35_f32 } else { 0.06_f32 };
    let res: Vec<(f64, f64, f64)> = thread::scope(|s| {
        let hs: Vec<_> = seeds.iter().map(|&sd| {
            s.spawn(move || {
                let mut core = UnifiedCayleyCore::new(cfg_fn(2, 3, sd));
                let mut rng = splitmix64(sd ^ 0x7A1E_0001 ^ (corridor_len as u64));
                let mut run_reg = |flipped: bool, n_eps: usize| -> Vec<f64> {
                    let mut out = Vec::with_capacity(n_eps);
                    for _ in 0..n_eps {
                        rng = splitmix64(rng);
                        let cue = (rng & 1) as usize;
                        let expected = if flipped { 1 - cue } else { cue };
                        core.start_episode();
                        let obs0 = [if cue == 0 { -1.0_f32 } else { 1.0_f32 }, 0.0, 0.0];
                        let _ = core.decide_rl(&obs0, true, false, false);
                        core.observe_rl_reward(0.0, &[], false, true);
                        for _ in 0..corridor_len {
                            let nz = rand_signed_f32(&mut rng) * noise_amp;
                            let _ = core.decide_rl(&[0.0, 0.0, nz], true, false, false);
                            core.observe_rl_reward(0.0, &[], false, true);
                        }
                        let act = core.decide_rl(&[0.0, 1.0, 0.0], true, false, false).0;
                        let ok = act == expected;
                        let r = if ok { 1.0_f32 } else { -1.0_f32 };
                        core.observe_rl_reward(r, &[], true, true);
                        let _ = core.end_episode_score(if ok { 1.0 } else { 0.0 }, true);
                        out.push(if ok { 1.0 } else { 0.0 });
                    }
                    out
                };
                let ra = run_reg(false, 120);
                let rb = run_reg(true, 120);
                let rret = run_reg(false, 80);
                let tail40 = |v: &[f64]| v[v.len() - 40..].iter().sum::<f64>() * 100.0 / 40.0;
                (0.5 * (tail40(&ra) + tail40(&rb)), tail40(&rret), core.memory_footprint_bytes() as f64 / 1024.0)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let sc: Vec<f64> = res.iter().map(|r| r.0).collect();
    let rt: Vec<f64> = res.iter().map(|r| r.1).collect();
    (mean_sd(&sc).0, mean_sd(&rt).0, res[0].2)
}

fn run_suite_b2() {
    let t0 = Instant::now();
    println!("==========================================================================================");
    println!(" SUITE B2 (Checkpoint 2): + Selectivity beta_k + Local Feedback z_{{t,k}}(u_{{t-1}})");
    println!("==========================================================================================");
    let seeds: Vec<u64> = (1..=10).collect();

    println!("\n--- [B2-1] Working Memory RepeatPrevious (RL +1/-1, 4 symbols, 10 seeds) ---");
    for &k in &[4usize, 8, 16, 24] {
        let (b1_m, _, _) = eval_repeat_prev_preset(k, UnifiedCayleyConfig::checkpoint_b1, &seeds);
        let (b2_m, b2_sd, ram) = eval_repeat_prev_preset(k, UnifiedCayleyConfig::checkpoint_b2, &seeds);
        println!("  K={:<2} | B1={:5.1}% | B2={:5.1}±{:4.1}% | RAM={:5.1}KB", k, b1_m, b2_m, b2_sd, ram);
    }

    println!("\n--- [B2-2] Hierarchical Dyck-2 (Depth 3..5, len=24, 1600 train / 400 eval, 10 seeds) ---");
    let (da, da_sd, dd3, dd3_sd, dram) = eval_dyck2_preset(UnifiedCayleyConfig::checkpoint_b2, false, &seeds);
    let (dsh, _, _, _, _) = eval_dyck2_preset(UnifiedCayleyConfig::checkpoint_b2, true, &seeds[..3]);
    println!("  Dyck-2 B2      : All={:5.1}±{:4.1}% | Depth>=3={:5.1}±{:4.1}% | RAM={:5.1}KB", da, da_sd, dd3, dd3_sd, dram);
    println!("  Dyck-2 Shuffled: All={:5.1}% (chance=33.3%)", dsh);

    println!("\n--- [B2-3] Multi-Scale Waves (ABC + Zero-Shot + Flip A->B->A, 10 seeds) ---");
    let (wabc, wzs, wcs, wrt) = eval_waves_preset(UnifiedCayleyConfig::checkpoint_b2, &seeds);
    println!("  Waves B2: ABC={:5.1}% | ZeroShot={:5.1}% | FlipSc={:5.1}% | Ret12={:5.1}%", wabc, wzs, wcs, wrt);

    println!("\n--- [B2-4] Non-Stationary T-Maze (L=5 ±0.06 & L=15 ±0.35, A->B->A, sparse, 10 seeds) ---");
    for &l in &[5usize, 15] {
        let (sc, rt, ram) = eval_tmaze_preset(l, UnifiedCayleyConfig::checkpoint_b2, &seeds);
        println!("  T-Maze L={:<2} B2: Score={:5.1}% | Ret={:5.1}% | RAM={:5.1}KB", l, sc, rt, ram);
    }

    println!("\n--- [B2-5] HardPOMDP v2 Strict Honest (Dense & Sparse, FNV-1a 24D, 10 seeds) ---");
    let (hd_on, hd_rt, hd_ho, hd_ram) = eval_hard_pomdp_preset(false, UnifiedCayleyConfig::checkpoint_b2, &seeds);
    let (hs_on, hs_rt, hs_ho, _) = eval_hard_pomdp_preset(true, UnifiedCayleyConfig::checkpoint_b2, &seeds);
    println!("  HardPOMDP Dense  B2: Online={:5.1}% | Ret={:5.1}% | Hold={:5.1}% | RAM={:5.1}KB", hd_on, hd_rt, hd_ho, hd_ram);
    println!("  HardPOMDP Sparse B2: Online={:5.1}% | Ret={:5.1}% | Hold={:5.1}%", hs_on, hs_rt, hs_ho);

    println!("\n--- [B2-6] Regression Check on B1 Tasks (ELEC2, NASA, CartPole) ---");
    if let Some(elec) = try_load_elec2_data() {
        let r_raw = eval_elec2_preset(&elec, UnifiedCayleyConfig::checkpoint_b2, &seeds[..5]);
        println!("  ELEC2 raw B2: {:.2}±{:.2}% ({:.2} us)", r_raw.0, r_raw.1, r_raw.3);
    }
    let cmapss_dir = Path::new("CMAPSSData");
    if cmapss_dir.exists() {
        let fd1 = load_cmapss_causal(cmapss_dir, "FD001", false, false);
        let fd4 = load_cmapss_causal(cmapss_dir, "FD004", true, false);
        let r1 = eval_nasa_preset(&fd1, 3, UnifiedCayleyConfig::checkpoint_b2, false, false, &seeds);
        let r4 = eval_nasa_preset(&fd4, 3, UnifiedCayleyConfig::checkpoint_b2, false, false, &seeds);
        println!("  NASA FD001 3p B2: RMSE={:5.2}±{:4.2} | FD004 3p B2: RMSE={:5.2}±{:4.2}", r1.rmse_m, r1.rmse_sd, r4.rmse_m, r4.rmse_sd);
    }
    let cp2 = eval_cartpole_preset(false, UnifiedCayleyConfig::checkpoint_b2, &seeds);
    let cpd = eval_cartpole_preset(true, UnifiedCayleyConfig::checkpoint_b2, &seeds);
    println!("  CartPole 2D B2: TrSc={:5.1} Ev={:5.1}/{:5.1} | +dx: TrSc={:5.1} Ev={:5.1}/{:5.1}", cp2.tr_score_m, cp2.ev_score, cp2.ev_ret, cpd.tr_score_m, cpd.ev_score, cpd.ev_ret);

    println!("\nSuite B2 finished in {:.2}s.", t0.elapsed().as_secs_f64());
}

fn eval_causal_prefix_invariance_6000() -> (usize, usize) {
    // 100 траекторий x 60 шагов = 6000 префиксов с NaN в будущем [t+1..]
    let mut passed = 0usize;
    let total = 6000usize;
    let base_cfg = UnifiedCayleyConfig::one_config(3, 4, 777);
    let mut rng = splitmix64(0xCA05_A100);
    for ep in 0..100usize {
        let len = 60usize;
        let mut seq = Vec::with_capacity(len);
        for _ in 0..len {
            seq.push([
                rand_signed_f32(&mut rng),
                rand_signed_f32(&mut rng),
                rand_signed_f32(&mut rng),
                rand_signed_f32(&mut rng),
            ]);
        }
        let mut ref_core = UnifiedCayleyCore::new(base_cfg);
        ref_core.start_episode();
        let mut ref_bits: Vec<(usize, u32)> = Vec::with_capacity(len);
        for t in 0..len {
            let out = match ep % 3 {
                0 => {
                    let (c, p) = ref_core.step_supervised_class(&seq[t], t % 3, true, true, false);
                    (c, p.to_bits())
                }
                1 => {
                    let r = ref_core.step_supervised_reg(&seq[t], 0.5, true, true, false);
                    (0, r.to_bits())
                }
                _ => {
                    let (a, p) = ref_core.decide_rl(&seq[t], true, false, false);
                    ref_core.observe_rl_reward(0.1, &[], false, true);
                    (a, p.to_bits())
                }
            };
            ref_bits.push(out);
        }

        // Каждая точка отсечения t проверяется отдельно: шаги t+1..len заменяются на NaN в потоке-двойнике
        for t_cut in 0..len {
            let mut probe_core = UnifiedCayleyCore::new(base_cfg);
            probe_core.start_episode();
            let mut match_all = true;
            for t in 0..len {
                let nan_x = [f32::NAN; 4];
                let inp = if t <= t_cut { &seq[t] } else { &nan_x };
                let out = match ep % 3 {
                    0 => {
                        let (c, p) = probe_core.step_supervised_class(inp, t % 3, t <= t_cut, t <= t_cut, false);
                        (c, p.to_bits())
                    }
                    1 => {
                        let r = probe_core.step_supervised_reg(inp, 0.5, t <= t_cut, t <= t_cut, false);
                        (0, r.to_bits())
                    }
                    _ => {
                        let (a, p) = probe_core.decide_rl(inp, t <= t_cut, false, false);
                        if t <= t_cut {
                            probe_core.observe_rl_reward(0.1, &[], false, true);
                        }
                        (a, p.to_bits())
                    }
                };
                if t <= t_cut && out != ref_bits[t] {
                    match_all = false;
                    break;
                }
            }
            if match_all {
                passed += 1;
            }
        }
    }
    (passed, total)
}

fn run_suite_b() {
    let t0 = Instant::now();
    println!("==========================================================================================");
    println!(" SUITE B (One-Config Challenge + Track 2 Unitary + 6000/6000 Causal + Shuffle/SEU)");
    println!("==========================================================================================");
    let seeds: Vec<u64> = (1..=10).collect();

    println!("\n--- Track 1: Ablation Ladder B1 -> B2 -> B3 (One-Config, ONE hyperparameter set!) ---");
    let presets: [(&str, fn(usize, usize, u64) -> UnifiedCayleyConfig); 3] = [
        ("B1 Base", UnifiedCayleyConfig::checkpoint_b1),
        ("B2 +Select+LocalZ", UnifiedCayleyConfig::checkpoint_b2),
        ("B3 One-Config", UnifiedCayleyConfig::one_config),
    ];

    let elec_opt = try_load_elec2_data();
    let cmapss_dir = Path::new("CMAPSSData");
    let fd1_opt = if cmapss_dir.exists() { Some(load_cmapss_causal(cmapss_dir, "FD001", false, false)) } else { None };
    let fd4_opt = if cmapss_dir.exists() { Some(load_cmapss_causal(cmapss_dir, "FD004", true, false)) } else { None };

    for &(name, cfg_fn) in &presets {
        let e_raw = elec_opt.as_ref().map(|d| eval_elec2_preset(d, cfg_fn, &seeds[..5])).unwrap_or((0.0, 0.0, 0.0, 0.0));
        let r_fd1 = fd1_opt.as_ref().map(|d| eval_nasa_preset(d, 3, cfg_fn, false, false, &seeds)).unwrap_or_default();
        let r_fd4 = fd4_opt.as_ref().map(|d| eval_nasa_preset(d, 3, cfg_fn, false, false, &seeds)).unwrap_or_default();
        let cp2d = eval_cartpole_preset(false, cfg_fn, &seeds);
        let cpdx = eval_cartpole_preset(true, cfg_fn, &seeds);
        let (rp4, _, _) = eval_repeat_prev_preset(4, cfg_fn, &seeds);
        let (rp8, _, _) = eval_repeat_prev_preset(8, cfg_fn, &seeds);
        let (dy_all, _, dy_d3, _, _) = eval_dyck2_preset(cfg_fn, false, &seeds);
        let (wv_abc, wv_zs, wv_fl, wv_rt) = eval_waves_preset(cfg_fn, &seeds);
        let (tm15, tm15_rt, _) = eval_tmaze_preset(15, cfg_fn, &seeds);
        let (hp_on, hp_rt, hp_ho, _) = eval_hard_pomdp_preset(false, cfg_fn, &seeds);

        println!(
            "[{:<18}] ELEC2={:.2}% | FD001={:5.2} FD004={:5.2} | CP2D={:5.1}({:5.1}/{:5.1}) CP+dx={:5.1}({:5.1}/{:5.1}) | RPrev4/8={:4.1}%/{:4.1}% | Dyck={:4.1}%(d>=3:{:4.1}%) | Waves(ABC={:4.1},ZS={:4.1},Fl={:4.1},R12={:4.1}) | TM15={:4.1}/{:4.1}% | HPOMDP(On={:4.1}%,Ret={:4.1}%,Hold={:4.1}%)",
            name,
            e_raw.0,
            r_fd1.rmse_m, r_fd4.rmse_m,
            cp2d.tr_score_m, cp2d.ev_score, cp2d.ev_ret,
            cpdx.tr_score_m, cpdx.ev_score, cpdx.ev_ret,
            rp4, rp8,
            dy_all, dy_d3,
            wv_abc, wv_zs, wv_fl, wv_rt,
            tm15, tm15_rt,
            hp_on, hp_rt, hp_ho
        );
    }

    println!("\n--- Track 1 Full Audit under B3 (One-Config) ---");
    if let Some(ref elec) = elec_opt {
        let elec_y = augment_elec2_yprev(elec);
        let elec_shuf = shuffle_data_labels(elec, 42);
        let ey = eval_elec2_preset(&elec_y, UnifiedCayleyConfig::one_config, &seeds[..5]);
        let es = eval_elec2_preset(&elec_shuf, UnifiedCayleyConfig::one_config, &seeds[..5]);
        println!("  ELEC2 +y_{{t-1}}={:.2}±{:.2}% | Label-Shuffle={:.2}±{:.2}% (RAM={:.1} KB)", ey.0, ey.1, es.0, es.1, ey.2);
    }
    if let (Some(fd1), Some(fd4)) = (&fd1_opt, &fd4_opt) {
        let r_seu = eval_nasa_preset(fd4, 3, UnifiedCayleyConfig::one_config, true, false, &seeds);
        let r_sh = eval_nasa_preset(fd1, 3, UnifiedCayleyConfig::one_config, false, true, &seeds);
        println!("  NASA FD004 Bit-30 SEU (3-pass): RMSE={:5.2}±{:4.2} cyc | EW-Rec={:5.1}% | any_nan={}", r_seu.rmse_m, r_seu.rmse_sd, r_seu.ew_recall, r_seu.any_nan);
        println!("  NASA FD001 RUL-Shuffle Control: RMSE={:5.2}±{:4.2} cyc", r_sh.rmse_m, r_sh.rmse_sd);
    }
    for &k in &[16usize, 24] {
        let (m, sd, ram) = eval_repeat_prev_preset(k, UnifiedCayleyConfig::one_config, &seeds);
        println!("  RepeatPrevious K={:<2} (B3): {:5.1}±{:4.1}% | RAM={:5.1} KB", k, m, sd, ram);
    }
    let (dy_sh, _, _, _, _) = eval_dyck2_preset(UnifiedCayleyConfig::one_config, true, &seeds[..3]);
    println!("  Dyck-2 Label-Shuffle (B3): {:5.1}%", dy_sh);
    let (tm5, tm5_rt, _) = eval_tmaze_preset(5, UnifiedCayleyConfig::one_config, &seeds);
    println!("  T-Maze L=5 (B3): Score={:5.1}% | Ret={:5.1}%", tm5, tm5_rt);
    let (hs_on, hs_rt, hs_ho, _) = eval_hard_pomdp_preset(true, UnifiedCayleyConfig::one_config, &seeds);
    println!("  HardPOMDP Sparse (B3): Online={:5.1}% | Ret={:5.1}% | Hold={:5.1}%", hs_on, hs_rt, hs_ho);

    println!("\n--- Track 2: Unitary Limit (sigma=0: Pure Orthogonal Memory & Non-Abelian Groups -> L=500) ---");
    let (u_rp24, u_rp24_sd, u_ram) = eval_repeat_prev_preset(24, UnifiedCayleyConfig::unitary_limit_cfg, &seeds);
    println!("  UnitaryLimit RepeatPrev K=24 (alpha=0 -> lam=1.0): {:5.1}±{:4.1}% | RAM={:5.1} KB", u_rp24, u_rp24_sd, u_ram);
    let a5 = PermGroup::a5();
    let s5 = PermGroup::s5();
    let a5_ref = &a5;
    let s5_ref = &s5;
    let t2_lens = &[6usize, 30, 100, 500];
    let a5_u: Vec<_> = thread::scope(|s| {
        seeds.iter().map(|&sd| s.spawn(move || eval_so3_model(a5_ref, 4, true, 10.0, 1.2, 0.65, 0.01, 5000, 8, 5.0, t2_lens, sd))).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).collect()
    });
    println!("  Track 2 A_5 Unitary SO(3)+Pillar2 [6,30,100,500] ({:.1} KB): {}", a5_u[0].2 as f64 / 1024.0, format_accs(&a5_u.iter().map(|r| r.0.clone()).collect::<Vec<_>>()));
    let s5_u: Vec<_> = thread::scope(|s| {
        seeds.iter().map(|&sd| s.spawn(move || eval_o4_model(s5_ref, 3, 12000, 10, 5.0, t2_lens, sd, false, false, &[0, 1], &[2, 3]))).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).collect()
    });
    println!("  Track 2 S_5 Unitary Band-3 O(4)->SO(5) [6,30,100,500] ({:.1} KB): {}", s5_u[0].mem_bytes as f64 / 1024.0, format_accs(&s5_u.iter().map(|r| r.accs.clone()).collect::<Vec<_>>()));

    println!("\n--- Causal Prefix Invariance Audit (future replaced with NaN) ---");
    let (pinv_ok, pinv_tot) = eval_causal_prefix_invariance_6000();
    println!("  Causal Prefix Invariance: {}/{} bit-identical", pinv_ok, pinv_tot);

    println!("\nSuite B finished in {:.2}s.", t0.elapsed().as_secs_f64());
}

fn run_suite_c() {
    let t0 = Instant::now();
    println!("==========================================================================================");
    println!(" SUITE C: No-FPU Fixed-Point Q8.24/Q39.24 Determinism & Digest Verification");
    println!("==========================================================================================");
    #[cfg(feature = "strict_deterministic")]
    println!("  Build mode: strict_deterministic (SIMD/FMA disabled)");
    #[cfg(not(feature = "strict_deterministic"))]
    println!("  Build mode: default release (x86_64 SIMD enabled for float path; Q8.24 path 100% integer)");

    // 1. q8_24_inv_sqrt по 10+ порядкам & FixedO4Rotor 10k-step drift
    let scales = [1.0f32 / 16000.0, 1.0e-3, 0.25, 1.0, 125.0, 4096.0, 1.0e6];
    let mut max_rel_err = 0.0_f64;
    let mut inv_sqrt_digest: u64 = 0xCBF2_9CE4_8422_2325;
    for &a in &scales {
        let y_q = q8_24_inv_sqrt(Q39_24::from_f32_boundary(a));
        let y_f = y_q.to_f32_boundary() as f64;
        let exp = 1.0_f64 / (a as f64).sqrt();
        max_rel_err = max_rel_err.max((y_f - exp).abs() / exp);
        inv_sqrt_digest = inv_sqrt_digest.wrapping_mul(0x0100_0000_01B3) ^ (y_q.0 as u64);
    }

    let mut rotor_unnorm = FixedO4Rotor::new(42, Q8_24::from_f32_boundary(1.0));
    let mut rotor_renorm = FixedO4Rotor::new(42, Q8_24::from_f32_boundary(1.0));
    for _ in 0..10_000usize {
        rotor_unnorm.step_unnormalized();
        rotor_renorm.step_and_renorm();
    }
    let rotor_unnorm_err = (rotor_unnorm.sphere_norm_sq_q39().to_f32_boundary() as f64 - 1.0).abs();
    let rotor_renorm_err = (rotor_renorm.sphere_norm_sq_q39().to_f32_boundary() as f64 - 1.0).abs();

    // 2. Q8.24 vs Q16.16 контроль малого шага lr = 1e-4 * 0.05 & PageHinkleyQ39_24 при RUL=125.0
    let step_q24 = q24_mul(Q8_24::from_f32_boundary(1.0e-4), Q8_24::from_f32_boundary(0.05));
    let step_q16 = q16_mul(Q16_16::from_f32_boundary(1.0e-4), Q16_16::from_f32_boundary(0.05));
    let mut ph = PageHinkleyQ39_24::new(
        Q39_24::from_f32_boundary(2.0),
        Q39_24::from_f32_boundary(60.0),
        8,
    );
    for _ in 0..16 {
        let _ = ph.observe(Q39_24::from_f32_boundary(125.0));
    }
    let mu_125 = ph.mu_bank.to_f32_boundary();
    let mut ph_drift_steps = 0usize;
    for k in 1..=16usize {
        if ph.observe(Q39_24::from_f32_boundary(40.0)) {
            ph_drift_steps = k;
            break;
        }
    }

    println!("\n--- [C-1] Integer Primitives & Controls ---");
    println!(
        "  q8_24_inv_sqrt [1/16000..1e6] max_rel_err={:.3e} | digest=0x{:016x}",
        max_rel_err, inv_sqrt_digest
    );
    println!(
        "  FixedO4Rotor (4-HH SO(5), beta=1.0) 10k-step |norm^2 - 1|: unnorm={:.3e} | renorm={:.3e}",
        rotor_unnorm_err, rotor_renorm_err
    );
    println!(
        "  lr=1e-4 * grad=0.05 control: Q8.24 raw={} ({:.3e}) vs Q16.16 raw={} ({:.3e})",
        step_q24.0,
        step_q24.to_f32_boundary(),
        step_q16.0,
        step_q16.0 as f64 / 65536.0
    );
    println!(
        "  PageHinkleyQ39_24 @ RUL=125.0 (no 128.0 cap): mu_bank={:.3}, drift_40_alarm_in={} steps",
        mu_125, ph_drift_steps
    );

    // 3. ELEC2 через FixedCayleyCore
    let seeds: Vec<u64> = (1..=5).collect();
    if let Some(elec) = try_load_elec2_data() {
        let d_in = elec[0].0.len();
        let elec_q: Vec<(Vec<Q39_24>, usize)> = elec
            .iter()
            .map(|(x, y)| (x.iter().map(|&v| Q39_24::from_f32_boundary(v)).collect(), *y))
            .collect();
        let res: Vec<(f64, f64, u64)> = thread::scope(|s| {
            let hs: Vec<_> = seeds
                .iter()
                .map(|&sd| {
                    let data_ref = &elec_q;
                    s.spawn(move || {
                        let mut core = FixedCayleyCore::new(d_in, 2, sd);
                        core.reset_episode();
                        let t_s = Instant::now();
                        let mut hits = 0usize;
                        for (qx, y) in data_ref {
                            let p = core.step_supervised_class_q(qx, *y, true);
                            if p == *y {
                                hits += 1;
                            }
                        }
                        let us = t_s.elapsed().as_secs_f64() * 1e6 / data_ref.len() as f64;
                        (hits as f64 * 100.0 / data_ref.len() as f64, us, core.state_digest_fnv1a())
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let accs: Vec<f64> = res.iter().map(|r| r.0).collect();
        let lats: Vec<f64> = res.iter().map(|r| r.1).collect();
        let (am, asd) = mean_sd(&accs);
        let (lm, _) = mean_sd(&lats);
        println!("\n--- [C-2] FixedCayleyCore (Q8.24/Q39.24, 0 FPU) on ELEC2 ({} rows, 5 seeds) ---", elec.len());
        println!(
            "  ELEC2 Q8.24: {:.2}±{:.2}% ({:.2} us/step) | Digest(seed=1)=0x{:016x}",
            am, asd, lm, res[0].2
        );
    }

    // 4. NASA C-MAPSS FD001 через FixedCayleyCore
    let cmapss_dir = Path::new("CMAPSSData");
    if cmapss_dir.exists() {
        let fd1 = load_cmapss_causal(cmapss_dir, "FD001", false, false);
        let train_q: Vec<(Vec<Vec<Q39_24>>, Vec<Q39_24>)> = fd1
            .train
            .iter()
            .map(|u| {
                (
                    u.feats
                        .iter()
                        .map(|f| f.iter().map(|&v| Q39_24::from_f32_boundary(v)).collect())
                        .collect(),
                    u.true_rul
                        .iter()
                        .map(|&r| Q39_24::from_f32_boundary(r / CMAPSS_RUL_MAX))
                        .collect(),
                )
            })
            .collect();
        let test_q: Vec<(Vec<Vec<Q39_24>>, f32)> = fd1
            .test
            .iter()
            .map(|u| {
                (
                    u.feats
                        .iter()
                        .map(|f| f.iter().map(|&v| Q39_24::from_f32_boundary(v)).collect())
                        .collect(),
                    u.end_rul,
                )
            })
            .collect();

        let res: Vec<(f64, f64, f64, f64, u64)> = thread::scope(|s| {
            let hs: Vec<_> = seeds
                .iter()
                .map(|&sd| {
                    let tr_ref = &train_q;
                    let te_ref = &test_q;
                    let in_d = fd1.in_dim;
                    s.spawn(move || {
                        let mut core = FixedCayleyCore::new(in_d, 2, sd);
                        let mut cyc = 0usize;
                        let t_s = Instant::now();
                        for pass in 0..3usize {
                            let mut order: Vec<usize> = (0..tr_ref.len()).collect();
                            let mut rng = splitmix64(sd.wrapping_mul(131) ^ (pass as u64 + 17));
                            for i in (1..order.len()).rev() {
                                rng = splitmix64(rng);
                                let j = (rng as usize) % (i + 1);
                                order.swap(i, j);
                            }
                            for &uid in &order {
                                core.reset_episode();
                                for (qx, qy) in tr_ref[uid].0.iter().zip(tr_ref[uid].1.iter()) {
                                    let _ = core.step_supervised_reg_q(qx, *qy, true);
                                    cyc += 1;
                                }
                            }
                        }
                        let us = t_s.elapsed().as_secs_f64() * 1e6 / cyc.max(1) as f64;
                        let mut se = 0.0_f64;
                        let mut ss = 0.0_f64;
                        let mut tp = 0usize;
                        let mut fn_c = 0usize;
                        for (qfeats, end_rul) in te_ref {
                            let mut m = core.clone();
                            m.reset_episode();
                            let mut last_p = Q39_24(0);
                            for qx in qfeats {
                                last_p = m.step_supervised_reg_q(qx, Q39_24(0), false);
                            }
                            let p_rul = (last_p.to_f32_boundary().clamp(0.0, 1.0) * CMAPSS_RUL_MAX) as f64;
                            let y = *end_rul as f64;
                            let d = p_rul - y;
                            se += d * d;
                            ss += if d < 0.0 { (-d / 13.0).exp() - 1.0 } else { (d / 10.0).exp() - 1.0 };
                            if y <= CMAPSS_CRIT_RUL as f64 {
                                if p_rul <= CMAPSS_CRIT_RUL as f64 { tp += 1; } else { fn_c += 1; }
                            }
                        }
                        let rmse = (se / te_ref.len().max(1) as f64).sqrt();
                        let rec = if tp + fn_c > 0 { tp as f64 * 100.0 / (tp + fn_c) as f64 } else { 0.0 };
                        (rmse, ss, rec, us, core.state_digest_fnv1a())
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let rmses: Vec<f64> = res.iter().map(|r| r.0).collect();
        let (rm, rsd) = mean_sd(&rmses);
        let (sm, _) = mean_sd(&res.iter().map(|r| r.1).collect::<Vec<_>>());
        let (recm, _) = mean_sd(&res.iter().map(|r| r.2).collect::<Vec<_>>());
        let (lm, _) = mean_sd(&res.iter().map(|r| r.3).collect::<Vec<_>>());
        println!("\n--- [C-3] FixedCayleyCore on NASA C-MAPSS FD001 (3-pass, 5 seeds) ---");
        println!(
            "  FD001 Q8.24: RMSE={:5.2}±{:4.2} cyc | S-Scr={:7.1} | EW-Rec={:5.1}% | {:.2} us | Digest(seed=1)=0x{:016x}",
            rm, rsd, sm, recm, lm, res[0].4
        );
    }

    // 5. Сырой NASA-стресс поток амплитуды 125.0..175.0 (проверка Q39.24 без потолка 128.0)
    let mut stress = FixedCayleyCore::new(4, 2, 777);
    for t in 0..200usize {
        let qx = [
            Q39_24::from_f32_boundary(125.0 + (t as f32) * 0.25),
            Q39_24::from_f32_boundary(((t % 7) as f32 - 3.0) * 20.0),
            Q39_24::from_f32_boundary(0.5),
            Q39_24::from_f32_boundary(-12.0),
        ];
        let _ = stress.step_supervised_class_q(&qx, t & 1, true);
        let _ = stress.step_supervised_reg_q(&qx, Q39_24::from_f32_boundary((200 - t) as f32 * 0.5), true);
    }
    println!(
        "\n--- [C-4] Raw Amplitude Stress (125.0..175.0 in Q39.24) ---\n  Digest(stress)=0x{:016x}",
        stress.state_digest_fnv1a()
    );
    println!("\nSuite C finished in {:.2}s.", t0.elapsed().as_secs_f64());
}

struct AdaptiveSeedOut {
    accs: Vec<f64>,
    final_dim: usize,
    ram_bytes: usize,
    escalations: usize,
    min_ep_visits: usize,
    med_ep_visits: usize,
    mean_ep_visits: f64,
    min_step_visits: usize,
}

fn eval_adaptive_od_model(
    group: &PermGroup,
    frames: usize,
    train_eps: usize,
    max_train_len: usize,
    dec_k: f32,
    eval_lens: &[usize],
    seed: u64,
    shuffle_labels: bool,
) -> AdaptiveSeedOut {
    let ng = group.mul_gen.len();
    let nc = group.num_elems;
    let mut net = AdaptiveOrthogonalRtrl::new(frames, ng, nc, seed);

    let mut rng = splitmix64(seed ^ 0xAD0D_5600_7200);
    let mut x = vec![0.0_f32; ng];
    let mut ep_visits = vec![0usize; nc];
    let mut step_visits = vec![0usize; nc];
    let mut escalations = 0usize;

    for ep in 0..train_eps {
        let len = 1 + (ep % max_train_len);
        let (gs, st_ids) = group.gen_word(&mut rng, len);
        net.reset();
        let dec = 1.0_f32 / (1.0 + dec_k * (ep as f32) / (train_eps as f32));
        let mut last_pred = 0usize;
        let mut last_y = 0usize;
        for t in 0..len {
            x.fill(0.0);
            x[gs[t]] = 1.0;
            let y = if shuffle_labels {
                rng = splitmix64(rng);
                ((rng >> 16) as usize) % nc
            } else {
                st_ids[t]
            };
            last_pred = net.step_supervised(&x, true, y, 0.025 * dec, 0.025 * dec);
            last_y = y;
            step_visits[y] += 1;
        }
        ep_visits[last_y] += 1;
        if net.observe_episode_hit(last_pred == last_y, last_y) {
            escalations += 1;
        }
    }

    let onehot: Vec<Vec<f32>> = (0..ng)
        .map(|g| {
            let mut v = vec![0.0_f32; ng];
            v[g] = 1.0;
            v
        })
        .collect();

    let eval_eps = 200usize;
    let mut accs = Vec::with_capacity(eval_lens.len());
    for &el in eval_lens {
        let mut hits = 0usize;
        let mut ev_rng = splitmix64(seed.wrapping_mul(100_003) ^ (el as u64));
        for _ in 0..eval_eps {
            let (gs, st_ids) = group.gen_word(&mut ev_rng, el);
            let refs: Vec<&[f32]> = gs.iter().map(|&g| onehot[g].as_slice()).collect();
            let pred = net.predict_k_ahead_readonly(&refs);
            if pred == st_ids[el - 1] {
                hits += 1;
            }
        }
        accs.push((hits as f64) * 100.0 / (eval_eps as f64));
    }

    let mut ep_sorted = ep_visits.clone();
    ep_sorted.sort_unstable();
    let min_ep = ep_sorted[0];
    let med_ep = ep_sorted[nc / 2];
    let mean_ep = ep_visits.iter().sum::<usize>() as f64 / nc as f64;
    let min_step = *step_visits.iter().min().unwrap_or(&0);

    AdaptiveSeedOut {
        accs,
        final_dim: net.active_dim,
        ram_bytes: net.memory_footprint_bytes(),
        escalations,
        min_ep_visits: min_ep,
        med_ep_visits: med_ep,
        mean_ep_visits: mean_ep,
        min_step_visits: min_step,
    }
}

fn summarize_suite_d(label: &str, outs: &[AdaptiveSeedOut], log_visits: bool) {
    let accs: Vec<Vec<f64>> = outs.iter().map(|o| o.accs.clone()).collect();
    let dims: Vec<f64> = outs.iter().map(|o| o.final_dim as f64).collect();
    let escs: Vec<f64> = outs.iter().map(|o| o.escalations as f64).collect();
    let (dm, dsd) = mean_sd(&dims);
    let (em, _) = mean_sd(&escs);
    let min_d = outs.iter().map(|o| o.final_dim).min().unwrap_or(0);
    let max_d = outs.iter().map(|o| o.final_dim).max().unwrap_or(0);
    let ram_kb = outs[0].ram_bytes as f64 / 1024.0;
    println!(
        "  {:<26} | d={:.2}±{:.2} [{}..{}] (esc={:.1}) | RAM={:5.1} KB | [6,12,30,100,250,500]: {}",
        label,
        dm,
        dsd,
        min_d,
        max_d,
        em,
        ram_kb,
        format_accs(&accs)
    );
    if log_visits {
        let min_ev = outs.iter().map(|o| o.min_ep_visits).min().unwrap_or(0);
        let med_ev = outs[0].med_ep_visits;
        let mean_ev = outs[0].mean_ep_visits;
        let min_sv = outs.iter().map(|o| o.min_step_visits).min().unwrap_or(0);
        println!(
            "    -> Class visit stats: end_ep(min={}, med={}, mean={:.1}) | total_step(min={})",
            min_ev, med_ev, mean_ev, min_sv
        );
    }
}

fn run_suite_d() {
    let t0 = Instant::now();
    println!("==========================================================================================");
    println!(" SUITE D: AdaptiveOrthogonalRtrl O(d)->SO(d+1) Ladder C_5 -> A_5 -> S_5 -> S_6");
    println!("==========================================================================================");
    let seeds: Vec<u64> = (1..=10).collect();
    let lens = &[6usize, 12, 30, 100, 250, 500];

    let c5 = PermGroup::c5();
    let a5 = PermGroup::a5();
    let s5 = PermGroup::s5();
    let s6 = PermGroup::s6();

    let r_c5: Vec<_> = thread::scope(|s| {
        seeds.iter().map(|&sd| {
            let g = &c5;
            s.spawn(move || eval_adaptive_od_model(g, 2, 2_500, 10, 5.0, lens, sd, false))
        }).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).collect()
    });
    summarize_suite_d("C_5 (5 cls, 2.5k eps)", &r_c5, false);

    let r_a5: Vec<_> = thread::scope(|s| {
        seeds.iter().map(|&sd| {
            let g = &a5;
            s.spawn(move || eval_adaptive_od_model(g, 2, 6_000, 10, 5.0, lens, sd, false))
        }).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).collect()
    });
    summarize_suite_d("A_5 (60 cls, 6k eps)", &r_a5, false);

    let r_s5: Vec<_> = thread::scope(|s| {
        seeds.iter().map(|&sd| {
            let g = &s5;
            s.spawn(move || eval_adaptive_od_model(g, 2, 12_000, 10, 5.0, lens, sd, false))
        }).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).collect()
    });
    summarize_suite_d("S_5 (120 cls, 12k eps)", &r_s5, false);

    let r_s6: Vec<_> = thread::scope(|s| {
        seeds.iter().map(|&sd| {
            let g = &s6;
            s.spawn(move || eval_adaptive_od_model(g, 2, 48_000, 24, 5.0, lens, sd, false))
        }).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).collect()
    });
    summarize_suite_d("S_6 (720 cls, 48k eps)", &r_s6, true);

    let r_shuf: Vec<_> = thread::scope(|s| {
        seeds.iter().map(|&sd| {
            let g = &s5;
            s.spawn(move || eval_adaptive_od_model(g, 2, 12_000, 10, 5.0, lens, sd, true))
        }).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).collect()
    });
    summarize_suite_d("Label-Shuffle Control", &r_shuf, false);

    println!("\nSuite D finished in {:.2}s.", t0.elapsed().as_secs_f64());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut run_a = false;
    let mut run_b1 = false;
    let mut run_b2 = false;
    let mut run_b3 = false;
    let mut run_c = false;
    let mut run_d = false;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--all" => {
                run_a = true;
                run_b1 = true;
                run_b2 = true;
                run_b3 = true;
                run_c = true;
                run_d = true;
            }
            "--suite" if i + 1 < args.len() => {
                i += 1;
                match args[i].as_str() {
                    "a" | "A" => run_a = true,
                    "b1" | "B1" => run_b1 = true,
                    "b2" | "B2" => run_b2 = true,
                    "b3" | "B3" | "b" | "B" => run_b3 = true,
                    "c" | "C" => run_c = true,
                    "d" | "D" => run_d = true,
                    other => eprintln!("Unknown suite: {other}"),
                }
            }
            _ => {}
        }
        i += 1;
    }

    if !(run_a || run_b1 || run_b2 || run_b3 || run_c || run_d) {
        run_a = true;
    }

    if run_a {
        run_suite_a();
    }
    if run_b1 {
        run_suite_b1();
    }
    if run_b2 {
        run_suite_b2();
    }
    if run_b3 {
        run_suite_b();
    }
    if run_c {
        run_suite_c();
    }
    if run_d {
        run_suite_d();
    }
}
