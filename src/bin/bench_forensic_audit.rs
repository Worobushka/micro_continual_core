// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Независимый форензик-аудит:
//! 1. Label-Shuffle Tests (ELEC2, ELEC2+y_{t-1}, Lorenz-63, Lorenz-96 N=16/256) — проверка отсутствия утечки будущего.
//! 2. Диагностика коренных причин КАЖДОГО провала MicroContinualCore на бенчмарках:
//!    - Axis 3B (RepeatPrevious supervised: band_mode / alpha_scale / 1L vs 2L / skip_x / gating)
//!    - ELEC2 +y_{t-1} (PH switching vs single bank vs Band-0 interference vs readout optimizer)
//!    - T-Maze без novelty_gate (SNR затухания cue под шумом коридора)
//!    - A_5 экстраполяция L=18..30 (нарушение групповых соотношений a^3=b^5=(ab)^2=I у выученных роторов SO(3))
//!    - Axis 4B R=8..32 (почему PH проигрывает 1-Bank при большом числе режимов: стоимость последовательного probe)
//!    - Online Per-Step GRU + PH в RL (конфликт BPTT+eligibility traces vs TD(0)+BPTT vs 1-step)

use std::path::Path;

use micro_continual_core::baselines::{
    DeepRtrrlLruPhBaseline, GruPerStepBank, OnlineGruPerStepPhBaseline, SelectiveSsmBank,
};
use micro_continual_core::math::{
    fast_dot, fast_saxpy, rand_signed_f32, rational_probs_and_psi_into, splitmix64,
};
use micro_continual_core::rtrl::{
    AlgebraicSelectiveCayley, NonAbelianCayleyRtrl, PlasticCayleyParams, PlasticCayleyState,
};
use micro_continual_core::stream_baselines::{NoChangePersistent, OnlineLruStream, OnlineMlpAdam};
use micro_continual_core::{MicroContinualCore, MicroCoreConfig};

fn try_load_elec2() -> Option<Vec<(Vec<f32>, usize)>> {
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

fn shuffle_labels(data: &[(Vec<f32>, usize)], seed: u64) -> Vec<(Vec<f32>, usize)> {
    let mut ys: Vec<usize> = data.iter().map(|d| d.1).collect();
    let mut rng = splitmix64(seed ^ 0x540F_F1E1);
    for i in (1..ys.len()).rev() {
        rng = splitmix64(rng);
        let j = (rng as usize) % (i + 1);
        ys.swap(i, j);
    }
    data.iter()
        .zip(ys.into_iter())
        .map(|((x, _), y)| (x.clone(), y))
        .collect()
}

fn augment_yprev(data: &[(Vec<f32>, usize)]) -> Vec<(Vec<f32>, usize)> {
    let mut out = Vec::with_capacity(data.len());
    let mut prev_y = 0.0_f32;
    for (x, y) in data {
        let mut aug = Vec::with_capacity(x.len() + 1);
        aug.extend_from_slice(x);
        aug.push(prev_y);
        out.push((aug, *y));
        prev_y = if *y == 1 { 1.0_f32 } else { -1.0_f32 };
    }
    out
}

fn generate_lorenz96(n: usize, steps: usize, k_ahead: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut st = vec![8.0_f64; n];
    let mut rng = splitmix64(seed ^ 0x1096_0000);
    for v in &mut st {
        rng = splitmix64(rng);
        *v += (rand_signed_f32(&mut rng) as f64) * 0.25;
    }
    let dt = 0.01_f64;
    let mut k1 = vec![0.0_f64; n];
    let mut k2 = vec![0.0_f64; n];
    let mut k3 = vec![0.0_f64; n];
    let mut k4 = vec![0.0_f64; n];
    let mut tmp = vec![0.0_f64; n];
    let rhs = |s: &[f64], out: &mut [f64]| {
        let m = s.len();
        for i in 0..m {
            let im2 = (i + m - 2) % m;
            let im1 = (i + m - 1) % m;
            let ip1 = (i + 1) % m;
            out[i] = (s[ip1] - s[im2]) * s[im1] - s[i] + 8.0;
        }
    };
    let mut rk4 = |s: &mut [f64]| {
        rhs(s, &mut k1);
        for i in 0..n { tmp[i] = s[i] + 0.5 * dt * k1[i]; }
        rhs(&tmp, &mut k2);
        for i in 0..n { tmp[i] = s[i] + 0.5 * dt * k2[i]; }
        rhs(&tmp, &mut k3);
        for i in 0..n { tmp[i] = s[i] + dt * k3[i]; }
        rhs(&tmp, &mut k4);
        for i in 0..n {
            s[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    };
    for _ in 0..300 { rk4(&mut st); }
    let mut traj = Vec::with_capacity(steps + k_ahead + 2);
    for _ in 0..(steps + k_ahead + 2) {
        traj.push(st.iter().map(|&v| ((v - 2.3) / 3.6) as f32).collect());
        for _ in 0..5 { rk4(&mut st); }
    }
    traj
}

// =============================================================================
// PART A: LABEL-SHUFFLE LEAKAGE TESTS
// =============================================================================
fn run_part_a_label_shuffle() {
    println!("==========================================================================================");
    println!("PART A: LABEL-SHUFFLE LEAKAGE VERIFICATION (Original vs Shuffled Labels)");
    println!("==========================================================================================");

    if let Some(elec) = try_load_elec2() {
        let up_pct = elec.iter().filter(|d| d.1 == 1).count() as f64 * 100.0 / elec.len() as f64;
        let maj_pct = up_pct.max(100.0 - up_pct);
        println!(
            "\n[A1] ELEC2 ({} rows, UP={:.2}%, DOWN={:.2}%, Majority floor={:.2}%):",
            elec.len(),
            up_pct,
            100.0 - up_pct,
            maj_pct
        );
        let elec_shuf = shuffle_labels(&elec, 42);
        let elec_y = augment_yprev(&elec);
        let elec_shuf_y = augment_yprev(&elec_shuf);

        let eval_micro_rtrl = |data: &[(Vec<f32>, usize)], ph_off: bool, no_band0: bool| -> f64 {
            let in_dim = data[0].0.len().min(12);
            let mut cfg = MicroCoreConfig::numeric_stream(2, 1);
            cfg.half_sectors = if no_band0 { 0 } else { 12 };
            cfg.belief_lambda = 0.25;
            cfg.rotor_norm = 0.10;
            cfg.linear_lr = 0.06;
            cfg.linear_mix = 1.0;
            cfg.linear_wd = 0.0005;
            cfg.trace_gamma = 0.0;
            cfg.rtrl_sectors = 12;
            cfg.rtrl_slow_sectors = 10;
            cfg.rtrl_max_in = in_dim;
            cfg.rtrl_lr = 0.004;
            cfg.rtrl_grid_mode = 1;
            cfg.rtrl_use_skip_x = true;
            cfg.rtrl_kappa = 0.65;
            if ph_off {
                cfg.ph_threshold = 9999.0;
                cfg.max_regimes = 2;
            }
            let mut core = MicroContinualCore::new(cfg);
            core.encoder.freeze_to_i8();
            core.start_episode();
            let mut hits = 0usize;
            for (x, y) in data {
                let p = core.decide_vec(x, false, false, false).get().0;
                if p == *y { hits += 1; }
                let _ = core.end_supervised_step(p, *y);
            }
            hits as f64 * 100.0 / data.len() as f64
        };

        let eval_lru = |data: &[(Vec<f32>, usize)]| -> f64 {
            let mut lru = OnlineLruStream::new(data[0].0.len(), 32, 2, 1);
            let mut hits = 0usize;
            for (x, y) in data {
                let p = lru.predict_and_train(x, *y, true);
                if p == *y { hits += 1; }
            }
            hits as f64 * 100.0 / data.len() as f64
        };

        let eval_mlp = |data: &[(Vec<f32>, usize)]| -> f64 {
            let mut mlp = OnlineMlpAdam::new(data[0].0.len(), 32, 2, 1);
            let mut hits = 0usize;
            for (x, y) in data {
                let p = mlp.predict_and_train(x, *y, true);
                if p == *y { hits += 1; }
            }
            hits as f64 * 100.0 / data.len() as f64
        };

        let eval_nc = |data: &[(Vec<f32>, usize)]| -> f64 {
            let mut nc = NoChangePersistent::new();
            let mut hits = 0usize;
            for (x, y) in data {
                let p = nc.predict_and_train(x, *y, true);
                if p == *y { hits += 1; }
            }
            hits as f64 * 100.0 / data.len() as f64
        };

        let eval_alg = |data: &[(Vec<f32>, usize)], selective: bool| -> f64 {
            let mut m = AlgebraicSelectiveCayley::new(data[0].0.len(), 32, 2, selective, 1)
                .with_standardize(true);
            let mut hits = 0usize;
            for (x, y) in data {
                let p = m.step_and_learn_supervised(x, Some(*y), 0.015, 0.005);
                if p == *y { hits += 1; }
            }
            hits as f64 * 100.0 / data.len() as f64
        };

        println!("{:<32} | {:>10} | {:>12}", "Model", "Original", "Shuffled y");
        println!("{}", "-".repeat(62));
        println!("{:<32} | {:9.2}% | {:11.2}%", "MicroCore+2L-RTRL (raw)", eval_micro_rtrl(&elec, false, false), eval_micro_rtrl(&elec_shuf, false, false));
        println!("{:<32} | {:9.2}% | {:11.2}%", "MicroCore+2L-RTRL (+y_{t-1})", eval_micro_rtrl(&elec_y, false, false), eval_micro_rtrl(&elec_shuf_y, false, false));
        println!("{:<32} | {:9.2}% | {:11.2}%", "2D Rat. Cayley (0% exp, raw)", eval_alg(&elec, false), eval_alg(&elec_shuf, false));
        println!("{:<32} | {:9.2}% | {:11.2}%", "Alg.Sel.Cayley (0% exp, raw)", eval_alg(&elec, true), eval_alg(&elec_shuf, true));
        println!("{:<32} | {:9.2}% | {:11.2}%", "Alg.Sel.Cayley (0% exp,+y_{t-1})", eval_alg(&elec_y, true), eval_alg(&elec_shuf_y, true));
        println!("{:<32} | {:9.2}% | {:11.2}%", "Online LRU-RTRL (raw)", eval_lru(&elec), eval_lru(&elec_shuf));
        println!("{:<32} | {:9.2}% | {:11.2}%", "Online LRU-RTRL (+y_{t-1})", eval_lru(&elec_y), eval_lru(&elec_shuf_y));
        println!("{:<32} | {:9.2}% | {:11.2}%", "Online MLP (H=32)+Adam (raw)", eval_mlp(&elec), eval_mlp(&elec_shuf));
        println!("{:<32} | {:9.2}% | {:11.2}%", "No-Change Persistent", eval_nc(&elec), eval_nc(&elec_shuf));

        // Diagnostic 2 right here while ELEC2 is already loaded!
        println!("\n[DIAG 2] Why MicroCore+RTRL (+y_{{t-1}}) trails Online LRU-RTRL (+y_{{t-1}}) on ELEC2:");
        println!("  MicroCore+RTRL (+y_{{t-1}}, default PH + Band0)  : {:.2}%", eval_micro_rtrl(&elec_y, false, false));
        println!("  MicroCore+RTRL (+y_{{t-1}}, PH OFF / 1-bank)     : {:.2}%", eval_micro_rtrl(&elec_y, true, false));
        println!("  MicroCore+RTRL (+y_{{t-1}}, PH OFF + no Band-0)  : {:.2}%", eval_micro_rtrl(&elec_y, true, true));
        println!("  MicroCore+RTRL (raw, PH OFF / 1-bank)            : {:.2}%", eval_micro_rtrl(&elec, true, false));
        println!("  MicroCore+RTRL (raw, PH OFF + no Band-0)         : {:.2}%", eval_micro_rtrl(&elec, true, true));
        println!("  Online LRU-RTRL (+y_{{t-1}}, Adam+Softmax)       : {:.2}%", eval_lru(&elec_y));
    }

    // [A2] Lorenz-63 (matched 60D) Original vs Shuffled y
    println!("\n[A2] Lorenz-63 1D-Projection Stream (6000 steps, K=8, 3-regime, matched 60D):");
    let rk4_step = |x: &mut f64, y: &mut f64, z: &mut f64| {
        let (sigma, rho, beta, dt) = (10.0_f64, 28.0_f64, 8.0 / 3.0, 0.02_f64);
        let deriv = |xx: f64, yy: f64, zz: f64| (sigma * (yy - xx), xx * (rho - zz) - yy, xx * yy - beta * zz);
        let (k1x, k1y, k1z) = deriv(*x, *y, *z);
        let (k2x, k2y, k2z) = deriv(*x + 0.5 * dt * k1x, *y + 0.5 * dt * k1y, *z + 0.5 * dt * k1z);
        let (k3x, k3y, k3z) = deriv(*x + 0.5 * dt * k2x, *y + 0.5 * dt * k2y, *z + 0.5 * dt * k2z);
        let (k4x, k4y, k4z) = deriv(*x + dt * k3x, *y + dt * k3y, *z + dt * k3z);
        *x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        *y += (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
        *z += (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    };
    let (mut lx, mut ly, mut lz) = (1.35_f64, -1.05_f64, 21.0_f64);
    for _ in 0..400 { rk4_step(&mut lx, &mut ly, &mut lz); }
    let total_n = 6000usize;
    let mut xs = Vec::with_capacity(total_n + 10);
    for _ in 0..(total_n + 10) {
        xs.push(lx as f32);
        for _ in 0..3 { rk4_step(&mut lx, &mut ly, &mut lz); }
    }
    let mut y_orig = Vec::with_capacity(total_n);
    for t in 0..total_n {
        let inv = t >= 2500 && t < 5000;
        let y = if inv { usize::from(xs[t + 8] < 0.0) } else { usize::from(xs[t + 8] >= 0.0) };
        y_orig.push(y);
    }
    let mut y_shuf = y_orig.clone();
    let mut rng_s = splitmix64(999);
    for i in (1..y_shuf.len()).rev() {
        rng_s = splitmix64(rng_s);
        let j = (rng_s as usize) % (i + 1);
        y_shuf.swap(i, j);
    }
    for (label, ys) in [("Original", &y_orig), ("Shuffled y", &y_shuf)] {
        let is_orig = label == "Original";
        let mut cfg = MicroCoreConfig::numeric_stream(2, 1);
        cfg.half_sectors = 0; cfg.rtrl_sectors = 10; cfg.rtrl_slow_sectors = 10;
        cfg.rtrl_max_in = 1; cfg.rtrl_lr = 0.012; cfg.rtrl_grid_mode = 1;
        cfg.rtrl_use_skip_x = true; cfg.standardize_vec = false; cfg.linear_lr = 0.06;
        cfg.ph_delta = 0.04; cfg.ph_threshold = 2.2; cfg.ph_warmup = 4; cfg.max_regimes = 4;
        let mut core = MicroContinualCore::new(cfg);
        core.encoder.freeze_to_i8(); core.start_episode();
        let mut deep = DeepRtrrlLruPhBaseline::new(1, 16, 14, 2, 4, 0.04, 2.2, 4, 1);
        deep.set_probe_accept_threshold(0.60);
        let (mut hc, mut hd, mut wf_n, mut wf_mean, mut wf_m2) = (0usize, 0usize, 0usize, 0.0_f64, 0.0_f64);
        let mut obs_hist: Vec<[f32; 1]> = Vec::with_capacity(total_n);
        let mut orig_preds_c = if is_orig { Vec::with_capacity(total_n) } else { Vec::new() };
        let mut orig_preds_d = if is_orig { Vec::with_capacity(total_n) } else { Vec::new() };
        for t in 0..total_n {
            wf_n += 1;
            let x_cur = xs[t] as f64;
            let d1 = x_cur - wf_mean; wf_mean += d1 / wf_n as f64;
            let d2 = x_cur - wf_mean; wf_m2 += d1 * d2;
            let std_c = if wf_n >= 2 { (wf_m2 / (wf_n - 1) as f64).sqrt().max(0.25) } else { 8.0 };
            obs_hist.push([((x_cur - wf_mean) / std_c).clamp(-4.0, 4.0) as f32]);
            if t >= 8 {
                let s = t - 8;
                let y_s = if is_orig {
                    if s >= 2500 && s < 5000 { usize::from(xs[t] < 0.0) } else { usize::from(xs[t] >= 0.0) }
                } else {
                    ys[s]
                };
                let p_s = core.decide_vec(&obs_hist[s], true, false, false).get().0;
                core.end_supervised_step(p_s, y_s);
                let _ = deep.step_supervised(&obs_hist[s], y_s, 0.04, 0.01, true, 1);
            }
            let win_start = if t >= 8 { t - 7 } else { 0 };
            let w_len = t + 1 - win_start;
            let mut win_refs: [&[f32]; 8] = [&[]; 8];
            for (wi, idx) in (win_start..=t).enumerate() {
                win_refs[wi] = &obs_hist[idx];
            }
            let pc = core.predict_vec_k_ahead_readonly(&win_refs[..w_len]);
            let pd = deep.predict_k_ahead_readonly(&win_refs[..w_len]);
            if is_orig {
                orig_preds_c.push(pc);
                orig_preds_d.push(pd);
            }
            if pc == ys[t] { hc += 1; }
            if pd == ys[t] { hd += 1; }
        }
        println!(
            "  {:<12}: MicroCore(60D)={:5.2}% | Deep 2L LRU(60D)={:5.2}%",
            label,
            hc as f64 * 100.0 / total_n as f64,
            hd as f64 * 100.0 / total_n as f64
        );
        if is_orig {
            // Causal Prefix Invariance check: mask future xs[t+1..] = f32::NAN at every t
            let mut cfg_m = MicroCoreConfig::numeric_stream(2, 1);
            cfg_m.half_sectors = 0; cfg_m.rtrl_sectors = 10; cfg_m.rtrl_slow_sectors = 10;
            cfg_m.rtrl_max_in = 1; cfg_m.rtrl_lr = 0.012; cfg_m.rtrl_grid_mode = 1;
            cfg_m.rtrl_use_skip_x = true; cfg_m.standardize_vec = false; cfg_m.linear_lr = 0.06;
            cfg_m.ph_delta = 0.04; cfg_m.ph_threshold = 2.2; cfg_m.ph_warmup = 4; cfg_m.max_regimes = 4;
            let mut core_m = MicroContinualCore::new(cfg_m);
            core_m.encoder.freeze_to_i8(); core_m.start_episode();
            let mut deep_m = DeepRtrrlLruPhBaseline::new(1, 16, 14, 2, 4, 0.04, 2.2, 4, 1);
            deep_m.set_probe_accept_threshold(0.60);
            let (mut mn_n, mut mn_mean, mut mn_m2) = (0usize, 0.0_f64, 0.0_f64);
            let mut masked_hist: Vec<[f32; 1]> = Vec::with_capacity(total_n);
            let (mut mis_c, mut mis_d) = (0usize, 0usize);
            for t in 0..total_n {
                let mut xs_masked = xs.clone();
                xs_masked[(t + 1)..].fill(f32::NAN);
                mn_n += 1;
                let x_cur = xs_masked[t] as f64;
                let d1 = x_cur - mn_mean; mn_mean += d1 / mn_n as f64;
                let d2 = x_cur - mn_mean; mn_m2 += d1 * d2;
                let std_c = if mn_n >= 2 { (mn_m2 / (mn_n - 1) as f64).sqrt().max(0.25) } else { 8.0 };
                masked_hist.push([((x_cur - mn_mean) / std_c).clamp(-4.0, 4.0) as f32]);
                if t >= 8 {
                    let s = t - 8;
                    let y_s = if s >= 2500 && s < 5000 { usize::from(xs_masked[t] < 0.0) } else { usize::from(xs_masked[t] >= 0.0) };
                    let p_s = core_m.decide_vec(&masked_hist[s], true, false, false).get().0;
                    core_m.end_supervised_step(p_s, y_s);
                    let _ = deep_m.step_supervised(&masked_hist[s], y_s, 0.04, 0.01, true, 1);
                }
                let win_start = if t >= 8 { t - 7 } else { 0 };
                let w_len = t + 1 - win_start;
                let mut win_refs: [&[f32]; 8] = [&[]; 8];
                for (wi, idx) in (win_start..=t).enumerate() {
                    win_refs[wi] = &masked_hist[idx];
                }
                let pc_m = core_m.predict_vec_k_ahead_readonly(&win_refs[..w_len]);
                let pd_m = deep_m.predict_k_ahead_readonly(&win_refs[..w_len]);
                if pc_m != orig_preds_c[t] { mis_c += 1; }
                if pd_m != orig_preds_d[t] { mis_d += 1; }
            }
            assert_eq!(mis_c, 0, "MicroCore violated causal prefix invariance!");
            assert_eq!(mis_d, 0, "Deep 2L LRU violated causal prefix invariance!");
            println!(
                "  Causal Prefix Invariance (xs[t+1..]=NaN): MicroCore={}/{} | Deep 2L LRU={}/{}",
                total_n - mis_c, total_n, total_n - mis_d, total_n
            );
        }
    }

    // [A3] Lorenz-96 (N=16 and N=256) Original vs Shuffled y
    println!("\n[A3] Lorenz-96 Spatiotemporal Chaos (2500 steps, K=4, eval t>=1000):");
    for &d_in in &[16usize, 256usize] {
        let traj = generate_lorenz96(d_in, 2500, 4, 1);
        let y_o: Vec<usize> = (0..2500).map(|t| usize::from(traj[t + 4][0] >= 0.0)).collect();
        let mut y_s = y_o.clone();
        let mut rng_l = splitmix64(12345 + d_in as u64);
        for i in (1..y_s.len()).rev() {
            rng_l = splitmix64(rng_l);
            let j = (rng_l as usize) % (i + 1);
            y_s.swap(i, j);
        }
        for (lbl, ys) in [("Original", &y_o), ("Shuffled y", &y_s)] {
            let is_orig = lbl == "Original";
            let mut params = PlasticCayleyParams::new_with_grid(16, d_in, 777, 1);
            let mut state = PlasticCayleyState::new(16, d_in, 0.65);
            let d = params.dim();
            let mut out_w = vec![0.0_f32; 2 * d];
            let mut v_w = vec![0.04_f32; 2 * d];
            let mut out_b = [0.0_f32; 2];
            let mut v_b = [0.04_f32; 2];
            let mut grad_b = vec![0.0_f32; d];
            let mut probs = [0.0_f32; 2];
            let mut psis = [1.0_f32; 2];
            let mut gru = GruPerStepBank::new_supervised(d_in, 32, 2, 4, 777);
            let (mut hc, mut hg, mut n_eval) = (0usize, 0usize, 0usize);
            let mut obs_hist: Vec<Vec<f32>> = Vec::with_capacity(2500);
            for t in 0..2500usize {
                let x: Vec<f32> = traj[t].iter().map(|&v| v.clamp(-3.0, 3.0)).collect();
                obs_hist.push(x);
                if t >= 4 {
                    let s = t - 4;
                    let y_s_lbl = if is_orig { usize::from(traj[t][0] >= 0.0) } else { ys[s] };
                    let xs_s = &obs_hist[s];
                    state.step_forward(&params, xs_s);
                    let l0 = out_b[0] + fast_dot(&out_w[0..d], &state.b);
                    let l1 = out_b[1] + fast_dot(&out_w[d..2 * d], &state.b);
                    rational_probs_and_psi_into(&[l0, l1], &mut probs, &mut psis);
                    grad_b.fill(0.0);
                    for c in 0..2usize {
                        let tgt = if c == y_s_lbl { 1.0 } else { 0.0 };
                        let err_c = (psis[c] * (tgt - probs[c])).clamp(-2.0, 2.0);
                        let off = c * d;
                        fast_saxpy(&mut grad_b, &out_w[off..off + d], err_c);
                        for i in 0..d {
                            let g = err_c * state.b[i];
                            v_w[off + i] = 0.98 * v_w[off + i] + 0.02 * g * g;
                            let st = (g / (v_w[off + i] + 0.0025).sqrt()).clamp(-3.0, 3.0);
                            out_w[off + i] = (out_w[off + i] * 0.99995 + 0.06 * st).clamp(-8.0, 8.0);
                        }
                        v_b[c] = 0.98 * v_b[c] + 0.02 * err_c * err_c;
                        out_b[c] = (out_b[c] + 0.25 * 0.06 * (err_c / (v_b[c] + 0.0025).sqrt()).clamp(-3.0, 3.0)).clamp(-4.0, 4.0);
                    }
                    state.accumulate_traces(&grad_b, 0.2, state.max_in);
                    state.apply_td_update(&mut params, 1.0, 0.012, state.max_in);
                    let _ = gru.step_and_learn_supervised(xs_s, Some(y_s_lbl));
                }
                let win_start = if t >= 4 { t - 3 } else { 0 };
                let w_len = t + 1 - win_start;
                let mut win_refs: [&[f32]; 4] = [&[]; 4];
                for (wi, idx) in (win_start..=t).enumerate() {
                    win_refs[wi] = &obs_hist[idx];
                }
                let mut u_ro = state.u.clone();
                let mut b_ro = state.b.clone();
                for &w in &win_refs[..w_len] {
                    PlasticCayleyState::step_state_readonly(&params, state.kappa, state.max_in, w, &mut u_ro, &mut b_ro);
                }
                let l0 = out_b[0] + fast_dot(&out_w[0..d], &b_ro);
                let l1 = out_b[1] + fast_dot(&out_w[d..2 * d], &b_ro);
                let pc = usize::from(l1 > l0);
                let pg = gru.predict_k_ahead_readonly(&win_refs[..w_len]);
                if t >= 1000 {
                    n_eval += 1;
                    if pc == ys[t] { hc += 1; }
                    if pg == ys[t] { hg += 1; }
                }
            }
            println!(
                "  N={:<3} {:<10}: Cayley(S=16)={:5.2}% | Sup-GRU(H=32)={:5.2}%",
                d_in, lbl, hc as f64 * 100.0 / n_eval as f64, hg as f64 * 100.0 / n_eval as f64
            );
        }
    }
}

// =============================================================================
// PART B: FAILURE ROOT-CAUSE DIAGNOSTICS
// =============================================================================
fn run_diag1_axis3b_repeat_prev() {
    println!("\n==========================================================================================");
    println!("DIAG 1: Why CayleyBlockSupervised fails Axis 3B RepeatPrevious (K=4, K=8) vs Selective SSM");
    println!("==========================================================================================");
    for &k_lag in &[4usize, 8usize] {
        let ep_len = (k_lag + 28).max(40);
        println!("\n--- Delay K={} (5 seeds, 250 train + 60 eval eps, 4 classes) ---", k_lag);
        let configs = [
            ("Original Axis3B: 1L S=16 band=2 (slow) alpha_sc=0.38", 16usize, 2u8, 0.38_f32, false),
            ("Abl A:           1L S=16 band=1 (fast) alpha_sc=0.38", 16usize, 1u8, 0.38_f32, false),
            ("Abl B:           1L S=16 band=1 (fast) alpha_sc=1.00", 16usize, 1u8, 1.00_f32, false),
            ("Abl C:           1L S=16 band=0 (grid) alpha_sc=1.00", 16usize, 0u8, 1.00_f32, false),
            ("Abl D:           1L S=16 band=1 + skip_x + alpha=1.0", 16usize, 1u8, 1.00_f32, true),
        ];
        for (name, sectors, band_mode, alpha_scale, use_skip) in configs {
            let mut accs = Vec::new();
            for seed in 1..=5u64 {
                let mut params = PlasticCayleyParams::new_with_grid(sectors, 4, seed ^ 0xCA71, band_mode);
                for a in &mut params.alpha { *a = (*a * alpha_scale).clamp(0.03, 1.8); }
                let mut state = PlasticCayleyState::new(sectors, 4, 0.65);
                let d = params.dim();
                let fdim = d + if use_skip { 4 } else { 0 };
                let mut out_w = vec![0.0_f32; 4 * fdim];
                let mut v_w = vec![0.04_f32; 4 * fdim];
                let mut out_b = [0.0_f32; 4];
                let mut v_b = [0.04_f32; 4];
                let mut grad_b = vec![0.0_f32; d];
                let mut feat = vec![0.0_f32; fdim];
                let mut probs = [0.0_f32; 4];
                let mut psis = [1.0_f32; 4];
                let mut rng = splitmix64(seed ^ 0x3B_1100 + k_lag as u64);
                let (mut hits, mut cnt) = (0usize, 0usize);
                for ep in 0..310usize {
                    let train = ep < 250;
                    state.reset();
                    let mut hist = Vec::with_capacity(ep_len);
                    for t in 0..ep_len {
                        rng = splitmix64(rng);
                        let sym = ((rng >> 16) as usize) % 4;
                        hist.push(sym);
                        let mut x = [0.0_f32; 4];
                        x[sym] = 1.0;
                        state.step_forward(&params, &x);
                        feat[..d].copy_from_slice(&state.b);
                        if use_skip { feat[d..].copy_from_slice(&x); }
                        let mut logits = [0.0_f32; 4];
                        let (mut best_c, mut best_l) = (0usize, f32::NEG_INFINITY);
                        for c in 0..4usize {
                            logits[c] = out_b[c] + fast_dot(&out_w[c * fdim..(c + 1) * fdim], &feat);
                            if logits[c] > best_l { best_l = logits[c]; best_c = c; }
                        }
                        if train && t >= k_lag {
                            let y = hist[t - k_lag];
                            rational_probs_and_psi_into(&logits, &mut probs, &mut psis);
                            grad_b.fill(0.0);
                            for c in 0..4usize {
                                let tgt = if c == y { 1.0 } else { 0.0 };
                                let err_c = (psis[c] * (tgt - probs[c])).clamp(-2.0, 2.0);
                                let off = c * fdim;
                                fast_saxpy(&mut grad_b, &out_w[off..off + d], err_c);
                                for i in 0..fdim {
                                    let g = err_c * feat[i];
                                    v_w[off + i] = 0.98 * v_w[off + i] + 0.02 * g * g;
                                    out_w[off + i] = (out_w[off + i] * 0.99995 + 0.10 * (g / (v_w[off + i] + 0.0025).sqrt()).clamp(-3.0, 3.0)).clamp(-8.0, 8.0);
                                }
                                v_b[c] = 0.98 * v_b[c] + 0.02 * err_c * err_c;
                                out_b[c] = (out_b[c] + 0.025 * (err_c / (v_b[c] + 0.0025).sqrt()).clamp(-3.0, 3.0)).clamp(-4.0, 4.0);
                            }
                            state.accumulate_traces(&grad_b, 0.0, 4);
                            state.apply_td_update(&mut params, 1.0, 0.015, 4);
                        }
                        if !train && t >= k_lag {
                            cnt += 1;
                            if best_c == hist[t - k_lag] { hits += 1; }
                        }
                    }
                }
                accs.push(hits as f64 * 100.0 / cnt as f64);
            }
            let mean = accs.iter().sum::<f64>() / accs.len() as f64;
            println!("  {:<56}: {:5.1}%", name, mean);
        }

        // Compare against Selective SSM (36D) & GRU (H=32, BPTT = K+2 vs BPTT=4)
        let mut ssm_accs = Vec::new();
        let mut gru_matched_accs = Vec::new();
        for seed in 1..=5u64 {
            let mut ssm = SelectiveSsmBank::new_supervised(4, 18, 4, seed ^ 0x55AA);
            let mut gru_m = GruPerStepBank::new_supervised(4, 32, 4, k_lag + 2, seed ^ 0x6400);
            gru_m.gru.lr = 0.006;
            let mut rng_s = splitmix64(seed ^ 0x3B_1100 + k_lag as u64);
            let mut rng_g = rng_s;
            let (mut hs, mut cs, mut hg, mut cg) = (0usize, 0usize, 0usize, 0usize);
            for ep in 0..310usize {
                let train = ep < 250;
                ssm.start_episode(); gru_m.start_episode();
                let mut hs_vec = Vec::with_capacity(ep_len);
                let mut hg_vec = Vec::with_capacity(ep_len);
                for t in 0..ep_len {
                    rng_s = splitmix64(rng_s);
                    let ss = ((rng_s >> 16) as usize) % 4;
                    hs_vec.push(ss);
                    let mut xs = [0.0_f32; 4]; xs[ss] = 1.0;
                    let tgt = if t >= k_lag { Some(hs_vec[t - k_lag]) } else { None };
                    let ps = ssm.step_and_learn_supervised(&xs, if train { tgt } else { None }, 0.05, 0.012);
                    if !train { if let Some(y) = tgt { cs += 1; if ps == y { hs += 1; } } }

                    rng_g = splitmix64(rng_g);
                    let sg = ((rng_g >> 16) as usize) % 4;
                    hg_vec.push(sg);
                    let mut xg = [0.0_f32; 4]; xg[sg] = 1.0;
                    let tg = if t >= k_lag { Some(hg_vec[t - k_lag]) } else { None };
                    let pg = gru_m.step_and_learn_supervised(&xg, if train { tg } else { None });
                    if !train { if let Some(y) = tg { cg += 1; if pg == y { hg += 1; } } }
                }
            }
            ssm_accs.push(hs as f64 * 100.0 / cs as f64);
            gru_matched_accs.push(hg as f64 * 100.0 / cg as f64);
        }
        println!("  {:<56}: {:5.1}%", "Ref: Selective SSM (18 complex = 36D, input-gated dt)", ssm_accs.iter().sum::<f64>() / 5.0);
        println!("  {:<56}: {:5.1}%", format!("Ref: Sup-GRU (H=32, BPTT=K+2={})", k_lag + 2), gru_matched_accs.iter().sum::<f64>() / 5.0);
    }
}

fn run_diag3_tmaze_snr() {
    println!("\n==========================================================================================");
    println!("DIAG 3: T-Maze L=15 (noise=±0.35) — Cue Cosine SNR Survival with vs without novelty_gate");
    println!("==========================================================================================");
    for &ng in &[0.0_f32, 0.28_f32] {
        let mut cos_vals = Vec::new();
        for seed in 1..=20u64 {
            let mut cfg = MicroCoreConfig::numeric_stream(2, seed);
            cfg.half_sectors = 32; cfg.belief_lambda = 0.96; cfg.belief_gamma = 0.45;
            cfg.rotor_norm = 0.08; cfg.novelty_gate = ng; cfg.rtrl_sectors = 0;
            let mut c_pos = MicroContinualCore::new(cfg);
            let mut c_neg = MicroContinualCore::new(cfg);
            c_pos.start_episode(); c_neg.start_episode();
            let mut rng = splitmix64(seed ^ 0x7A1E);
            // Step 0: opposite cues +1 vs -1
            let _ = c_pos.decide_vec(&[1.0, 0.0, 0.0], false, false, false);
            let _ = c_neg.decide_vec(&[-1.0, 0.0, 0.0], false, false, false);
            // Steps 1..15: identical corridor noise ±0.35
            for _ in 1..15usize {
                let nz = rand_signed_f32(&mut rng) * 0.35;
                let _ = c_pos.decide_vec(&[0.0, 0.0, nz], false, false, false);
                let _ = c_neg.decide_vec(&[0.0, 0.0, nz], false, false, false);
            }
            // Junction step
            let _ = c_pos.decide_vec(&[0.0, 1.0, 0.0], false, false, false);
            let _ = c_neg.decide_vec(&[0.0, 1.0, 0.0], false, false, false);
            // Distance between belief states for +1 vs -1 cue (1 - cos; 0 = collapsed/indistinguishable!)
            let cos = fast_dot(&c_pos.belief.state, &c_neg.belief.state);
            cos_vals.push(cos as f64);
        }
        let mean_cos = cos_vals.iter().sum::<f64>() / cos_vals.len() as f64;
        println!(
            "  novelty_gate = {:.2}: mean cos(b(+1), b(-1)) at L=15 junction = {:.4}  (separation 1-cos = {:.4})",
            ng, mean_cos, 1.0 - mean_cos
        );
    }
}

fn run_diag4_a5_relations() {
    println!("\n==========================================================================================");
    println!("DIAG 4: Why NonAbelianCayleyRtrl fails A_5 extrapolation at L=18..30 (SO(3) Presentation Error)");
    println!("==========================================================================================");
    // Train NonAbelianCayleyRtrl on A_5 L<=8 (same as Suite 1), then measure algebraic presentation errors:
    // ||R(a)^3 - I||_F, ||R(b)^5 - I||_F, ||(R(a) R(b))^2 - I||_F!
    let mut nac = NonAbelianCayleyRtrl::new(12, 3, 60, 0.25, 1);
    nac.enable_so3_frames_end_to_end();
    // Build A_5 multiplication table
    let mut perms: Vec<[u8; 5]> = vec![[0, 1, 2, 3, 4]];
    let compose = |p: [u8; 5], q: [u8; 5]| -> [u8; 5] { [p[q[0] as usize], p[q[1] as usize], p[q[2] as usize], p[q[3] as usize], p[q[4] as usize]] };
    let gens = [[1, 2, 0, 3, 4], [1, 2, 3, 4, 0], [0, 3, 4, 1, 2]]; // a (order 3), b (order 5), c (order 2)
    let mut idx = 0usize;
    while idx < perms.len() {
        let p = perms[idx];
        for g in gens {
            let np = compose(p, g);
            if !perms.contains(&np) { perms.push(np); }
        }
        idx += 1;
    }
    let mut mul = vec![[0usize; 3]; perms.len()];
    for (i, &p) in perms.iter().enumerate() {
        for (gi, &g) in gens.iter().enumerate() {
            let np = compose(p, g);
            mul[i][gi] = perms.iter().position(|&q| q == np).unwrap();
        }
    }
    let mut rng = splitmix64(0xA5_0001);
    for _ in 0..3500usize {
        rng = splitmix64(rng);
        let len = 1 + ((rng >> 8) as usize) % 8;
        nac.reset();
        let mut st = 0usize;
        for _ in 0..len {
            rng = splitmix64(rng);
            let gi = ((rng >> 16) as usize) % 3;
            st = mul[st][gi];
            let mut x = [0.0_f32; 3]; x[gi] = 1.0;
            let _ = nac.step_and_learn_supervised(&x, Some(st), 0.08, 0.015);
        }
    }
    // Extract exact 3x3 SO(3) double-Cayley rotation matrix R_2(g) for sector s and input x_gi
    let mat_of_gen = |n: &NonAbelianCayleyRtrl, s: usize, gi: usize| -> [[f32; 3]; 3] {
        let off = s * 3 * n.in_dim;
        let gx = n.g0[s * 3] + n.wg[off + gi];
        let gy = n.g0[s * 3 + 1] + n.wg[off + n.in_dim + gi];
        let gz = n.g0[s * 3 + 2] + n.wg[off + 2 * n.in_dim + gi];
        let mut r = [[0.0_f32; 3]; 3];
        for col in 0..3 {
            let mut e = [0.0_f32; 3]; e[col] = 1.0;
            let v = micro_continual_core::cayley_rot3_double_vec(gx, gy, gz, e[0], e[1], e[2]);
            r[0][col] = v.0; r[1][col] = v.1; r[2][col] = v.2;
        }
        r
    };
    let mm = |a: [[f32; 3]; 3], b: [[f32; 3]; 3]| -> [[f32; 3]; 3] {
        let mut c = [[0.0_f32; 3]; 3];
        for i in 0..3 { for j in 0..3 { for k in 0..3 { c[i][j] += a[i][k] * b[k][j]; } } }
        c
    };
    let frob_id = |a: [[f32; 3]; 3]| -> f32 {
        let mut s = 0.0_f32;
        for i in 0..3 { for j in 0..3 { let d = a[i][j] - if i == j { 1.0 } else { 0.0 }; s += d * d; } }
        s.sqrt()
    };
    let (mut err_a3, mut err_b5, mut err_c2, mut err_orth) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    for s in 0..12usize {
        let ra = mat_of_gen(&nac, s, 0);
        let rb = mat_of_gen(&nac, s, 1);
        let rc = mat_of_gen(&nac, s, 2);
        let ra3 = mm(mm(ra, ra), ra);
        let rb5 = mm(mm(mm(mm(rb, rb), rb), rb), rb);
        let rc2 = mm(rc, rc);
        let mut rat = [[0.0_f32; 3]; 3];
        for i in 0..3 { for j in 0..3 { rat[i][j] = ra[j][i]; } }
        err_orth += frob_id(mm(rat, ra));
        err_a3 += frob_id(ra3);
        err_b5 += frob_id(rb5);
        err_c2 += frob_id(rc2);
    }
    println!("  [Linear Readout v0.4.0]:");
    println!("    Mean orthogonality error ||R(a)^T R(a) - I||_F  = {:.6} (exact SO(3)!)", err_orth / 12.0);
    println!("    Mean relation error ||R(a)^3 - I||_F (true=0)   = {:.4}", err_a3 / 12.0);
    println!("    Mean relation error ||R(b)^5 - I||_F (true=0)   = {:.4}", err_b5 / 12.0);
    println!("    Mean relation error ||R(c)^2 - I||_F (true=0)   = {:.4}", err_c2 / 12.0);

    // Compare against task-agnostic Proto-Metric SO(3) (zero group oracle!)
    let mut nac_p = NonAbelianCayleyRtrl::new(12, 3, 60, 0.25, 1);
    nac_p.enable_so3_proto_metric(10.0, 1.2, 0.08);
    let mut rng_p = splitmix64(0xA5_0001);
    for ep in 0..4000usize {
        rng_p = splitmix64(rng_p);
        let len = 1 + ((rng_p >> 8) as usize) % 8;
        nac_p.reset();
        let mut st = 0usize;
        let dec = 1.0_f32 / (1.0 + 6.0 * (ep as f32) / 4000.0);
        for _ in 0..len {
            rng_p = splitmix64(rng_p);
            let gi = ((rng_p >> 16) as usize) % 3;
            st = mul[st][gi];
            let mut x = [0.0_f32; 3]; x[gi] = 1.0;
            let _ = nac_p.step_and_learn_supervised(&x, Some(st), 0.02 * dec, 0.020 * dec);
        }
    }
    let (mut pa3, mut pb5, mut pc2, mut porth) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    for s in 0..12usize {
        let ra = mat_of_gen(&nac_p, s, 0);
        let rb = mat_of_gen(&nac_p, s, 1);
        let rc = mat_of_gen(&nac_p, s, 2);
        let mut rat = [[0.0_f32; 3]; 3];
        for i in 0..3 { for j in 0..3 { rat[i][j] = ra[j][i]; } }
        porth += frob_id(mm(rat, ra));
        pa3 += frob_id(mm(mm(ra, ra), ra));
        pb5 += frob_id(mm(mm(mm(mm(rb, rb), rb), rb), rb));
        pc2 += frob_id(mm(rc, rc));
    }
    println!("  [Proto-Metric SO(3), 0% exp, 0 group oracle]:");
    println!("    Mean orthogonality error ||R(a)^T R(a) - I||_F  = {:.6}", porth / 12.0);
    println!("    Mean relation error ||R(a)^3 - I||_F (true=0)   = {:.4}", pa3 / 12.0);
    println!("    Mean relation error ||R(b)^5 - I||_F (true=0)   = {:.4}", pb5 / 12.0);
    println!("    Mean relation error ||R(c)^2 - I||_F (true=0)   = {:.4}", pc2 / 12.0);

    // Shuffled Labels A_5 Sanity Check + S_5 (120 elems, NOT <= SO(3)^M) Horizon Contrast Check
    let mut shuf_m = NonAbelianCayleyRtrl::new(12, 3, 60, 0.25, 1);
    shuf_m.enable_so3_proto_metric(10.0, 1.2, 0.08);
    let mut rng_sh = splitmix64(0xDEAD_BEEF);
    for ep in 0..4000usize {
        let len = 1 + (ep % 8);
        shuf_m.reset();
        let dec = 1.0_f32 / (1.0 + 6.0 * (ep as f32) / 4000.0);
        for _ in 0..len {
            rng_sh = splitmix64(rng_sh);
            let gi = ((rng_sh >> 16) as usize) % 3;
            rng_sh = splitmix64(rng_sh);
            let ry = ((rng_sh >> 16) as usize) % 60;
            let mut x = [0.0_f32; 3]; x[gi] = 1.0;
            let _ = shuf_m.step_and_learn_supervised(&x, Some(ry), 0.02 * dec, 0.020 * dec);
        }
    }
    let mut sh_hits = 0usize;
    for _ in 0..200usize {
        shuf_m.reset();
        let mut pred = 0usize;
        for _ in 0..6 {
            rng_sh = splitmix64(rng_sh);
            let gi = ((rng_sh >> 16) as usize) % 3;
            let mut x = [0.0_f32; 3]; x[gi] = 1.0;
            pred = shuf_m.step_and_learn_supervised(&x, None, 0.0, 0.0);
        }
        rng_sh = splitmix64(rng_sh);
        if pred == ((rng_sh >> 16) as usize) % 60 { sh_hits += 1; }
    }
    println!("  [A_5 Label-Shuffle Check] Proto-Metric SO(3) L=6 = {:.2}% (chance = 1.67%)", sh_hits as f64 * 100.0 / 200.0);

    // S_5 (120 elements): non-involution generators a=(0 1)(2 3 4) order 6 (odd!) and b=(0 1 2 3 4) order 5 (even) + b^-1
    // Wait: use standard S_5 generators (0 1), (0 1 2 3 4), (0 4 3 2 1) and test L=6, 12, 18, 30, 50!
    let s5_gens = [[1u8, 0, 2, 3, 4], [1, 2, 3, 4, 0], [4, 0, 1, 2, 3]];
    let mut s5_perms: Vec<[u8; 5]> = vec![[0, 1, 2, 3, 4]];
    let mut s5_idx = 0usize;
    while s5_idx < s5_perms.len() {
        let p = s5_perms[s5_idx];
        for g in s5_gens {
            let np = compose(p, g);
            if !s5_perms.contains(&np) { s5_perms.push(np); }
        }
        s5_idx += 1;
    }
    let mut s5_mul = vec![[0usize; 3]; s5_perms.len()];
    for (i, &p) in s5_perms.iter().enumerate() {
        for (gi, &g) in s5_gens.iter().enumerate() {
            let np = compose(p, g);
            s5_mul[i][gi] = s5_perms.iter().position(|&q| q == np).unwrap();
        }
    }
    let mut s5_m = NonAbelianCayleyRtrl::new(12, 3, s5_perms.len(), 0.25, 1);
    s5_m.enable_so3_proto_metric(10.0, 1.2, 0.08);
    let mut rng_s5 = splitmix64(0x5555_0001);
    for ep in 0..6000usize {
        let len = 1 + (ep % 8);
        s5_m.reset();
        let mut st = 0usize;
        let dec = 1.0_f32 / (1.0 + 6.0 * (ep as f32) / 6000.0);
        for _ in 0..len {
            rng_s5 = splitmix64(rng_s5);
            let gi = ((rng_s5 >> 16) as usize) % 3;
            st = s5_mul[st][gi];
            let mut x = [0.0_f32; 3]; x[gi] = 1.0;
            let _ = s5_m.step_and_learn_supervised(&x, Some(st), 0.02 * dec, 0.020 * dec);
        }
    }
    print!("  [Group-Theory Check: S_5 (120 elems, NOT <= SO(3)^M), Train L<=8]:");
    for &ll in &[6usize, 12, 18, 30, 50] {
        let mut h = 0usize;
        for _ in 0..200usize {
            s5_m.reset();
            let mut st = 0usize;
            let mut pred = 0usize;
            for _ in 0..ll {
                rng_s5 = splitmix64(rng_s5);
                let gi = ((rng_s5 >> 16) as usize) % 3;
                st = s5_mul[st][gi];
                let mut x = [0.0_f32; 3]; x[gi] = 1.0;
                pred = s5_m.step_and_learn_supervised(&x, None, 0.0, 0.0);
            }
            if pred == st { h += 1; }
        }
        print!(" L={}={:.1}%", ll, h as f64 * 100.0 / 200.0);
    }
    println!();
}

fn run_diag5_gru_rl_ablation() {
    println!("\n==========================================================================================");
    println!("DIAG 5: Online Per-Step GRU + PH in RL — bptt_len & trace_lambda Ablation (RepeatPrev K=4 & CartPole)");
    println!("==========================================================================================");
    // Compare bptt_len in {1, 4, 6} x trace_lambda in {0.80, 0.0} on RepeatPrevious K=4 (3 seeds) & CartPole +dx (3 seeds)
    for &(blen, tlam, lr) in &[
        (1usize, 0.80_f32, 0.005_f32),
        (4usize, 0.80_f32, 0.002_f32),
        (6usize, 0.00_f32, 0.003_f32),
        (8usize, 0.00_f32, 0.002_f32),
    ] {
        // RepeatPrevious K=4 (140 eps, 3 seeds)
        let mut rp_accs = Vec::new();
        for seed in 1..=3u64 {
            let mut gru = OnlineGruPerStepPhBaseline::new(4, 32, 4, 2, 0.14, 2.2, 25, seed);
            for b in &mut gru.banks { b.set_bptt_len(blen); }
            gru.set_hyperparams(0.95, tlam, lr);
            let mut rng = splitmix64(seed ^ 0x52_2026);
            let mut tail_hits = 0usize;
            let mut tail_n = 0usize;
            for ep in 0..140usize {
                gru.start_episode();
                let mut hist = Vec::with_capacity(32);
                let (mut h, mut n) = (0usize, 0usize);
                for t in 0..32usize {
                    rng = splitmix64(rng);
                    let sym = ((rng >> 16) as usize) % 4;
                    hist.push(sym);
                    let mut obs = [0.0_f32; 4]; obs[sym] = 1.0;
                    let p = gru.decide_vec(&obs, true, false, false);
                    if t >= 4 {
                        let ok = p == hist[t - 4];
                        if ok { h += 1; }
                        n += 1;
                        gru.observe_step_reward(if ok { 1.0 } else { -1.0 });
                    } else {
                        gru.observe_step_reward(0.0);
                    }
                }
                gru.end_episode_with_score(h as f32 / n as f32, false, true);
                if ep >= 100 { tail_hits += h; tail_n += n; }
            }
            rp_accs.push(tail_hits as f64 * 100.0 / tail_n as f64);
        }
        // CartPole +dx (140 eps normal, 3 seeds)
        let mut cp_steps = Vec::new();
        for seed in 1..=3u64 {
            let mut gru = OnlineGruPerStepPhBaseline::new(4, 32, 2, 2, 0.12, 1.8, 28, seed);
            for b in &mut gru.banks { b.set_bptt_len(blen); }
            gru.set_hyperparams(0.95, tlam, lr);
            let mut rng = splitmix64(seed ^ 0xCA27_901E);
            let mut tail_s = 0.0_f64;
            for ep in 0..140usize {
                gru.start_episode();
                let mut x = rand_signed_f32(&mut rng) * 0.05;
                let mut x_dot = rand_signed_f32(&mut rng) * 0.05;
                let mut theta = rand_signed_f32(&mut rng) * 0.05;
                let mut theta_dot = rand_signed_f32(&mut rng) * 0.05;
                let mut prev = [0.0_f32; 2];
                let mut has_prev = false;
                let mut steps = 0usize;
                loop {
                    let o = [x / 2.4, theta / 0.2094];
                    let dx0 = if has_prev { (o[0] - prev[0]) * 10.0 } else { 0.0 };
                    let dx1 = if has_prev { (o[1] - prev[1]) * 10.0 } else { 0.0 };
                    prev = o; has_prev = true;
                    let feat = [o[0], o[1], dx0.clamp(-4.0, 4.0), dx1.clamp(-4.0, 4.0)];
                    let act = gru.decide_vec(&feat, true, false, false);
                    let force = if act == 1 { 10.0_f32 } else { -10.0_f32 };
                    let ct = theta.cos(); let st = theta.sin();
                    let temp = (force + 0.05 * theta_dot * theta_dot * st) / 1.1;
                    let th_acc = (9.8 * st - ct * temp) / (0.5 * (4.0 / 3.0 - 0.1 * ct * ct / 1.1));
                    let x_acc = temp - 0.05 * th_acc * ct / 1.1;
                    x += 0.02 * x_dot; x_dot += 0.02 * x_acc;
                    theta += 0.02 * theta_dot; theta_dot += 0.02 * th_acc;
                    steps += 1;
                    let done = x.abs() > 2.4 || theta.abs() > 0.2094 || steps >= 200;
                    gru.observe_step_reward(if done && steps < 200 { -1.0 } else { 0.06 });
                    if done { break; }
                }
                gru.end_episode_with_score((steps as f32 / 70.0).clamp(0.0, 1.0), steps >= 200, true);
                if ep >= 100 { tail_s += steps as f64; }
            }
            cp_steps.push(tail_s / 40.0);
        }
        println!(
            "  bptt_len={}, trace_lambda={:.2}, lr={:.4} -> RepeatPrev K=4: {:5.1}% | CartPole+dx: {:5.1} steps",
            blen, tlam, lr,
            rp_accs.iter().sum::<f64>() / 3.0,
            cp_steps.iter().sum::<f64>() / 3.0
        );
    }
}

fn main() {
    run_part_a_label_shuffle();
    run_diag1_axis3b_repeat_prev();
    run_diag3_tmaze_snr();
    run_diag4_a5_relations();
    run_diag5_gru_rl_ablation();
}
