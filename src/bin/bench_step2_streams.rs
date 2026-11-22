// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Бенчмарк Шага 2 (v0.4.0): Мировые стандарты Concept Drift (MOA / River / IJCAI / SIGKDD)
//! 1-проходный Prequential (Test-then-Train), 10 сидов, таблица с Kappa_temp (Bifet et al. 2013):
//! 1. SEA Concepts (Recurrent Abrupt Drift: theta = 8 -> 9 -> 7 -> 9.5 -> 8, +10% шум)
//! 2. STAGGER (Recurrent Abrupt Drift: A -> B -> C -> A -> B)
//! 3. Rotating Hyperplane 10D (Gradual Drift + 10% шум)
//! 4. Uncorrelated Multi-Class RBF Drift (10D, 4 класса, нулевая автокорреляция меток, 8k)
//! 5. Electricity (ELEC2) Real Stream (45,312 строк):
//!    - честный No-Change (Persistent) бейзлайн (~85.33%) и Kappa_temp,
//!    - исправленный Online MLP (с онлайн-стандартизацией),
//!    - MicroCore (i.i.d.) vs MicroCore (Temporal Cayley) vs MicroCore (Temp + causal y_{t-1}) vs MLP (+y_{t-1}).

use std::path::Path;
use std::time::Instant;

use micro_continual_core::math::{rand_signed_f32, rand_unit_f32, splitmix64};
use micro_continual_core::rtrl::AlgebraicSelectiveCayley;
use micro_continual_core::stream_baselines::{
    NoChangePersistent, OnlineGaussianNaiveBayes, OnlineLruStream, OnlineMlpAdam, TabularBinnedPH,
};
use micro_continual_core::{MicroContinualCore, MicroCoreConfig};

#[derive(Default, Clone)]
struct StreamMetrics {
    prequential_acc: f64,
    retention_acc: f64,
    recovery_steps: f64,
    ram_kb: f64,
    us_per_step: f64,
    seu_safe: bool,
}

fn mean_sd(vals: &[f64]) -> (f64, f64) {
    let n = vals.len().max(1) as f64;
    let m = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n;
    (m, var.sqrt())
}

fn recovery_after_drift(hits: &[bool], drift_idx: usize, target_acc: f64) -> f64 {
    let win = 50usize;
    if drift_idx + win > hits.len() {
        return 0.0;
    }
    let lim = (drift_idx + 600).min(hits.len());
    for i in drift_idx..=(lim.saturating_sub(win)) {
        let c = hits[i..i + win].iter().filter(|&&b| b).count() as f64 / (win as f64);
        if c >= target_acc {
            return (i - drift_idx + win) as f64;
        }
    }
    (lim - drift_idx) as f64
}

fn kappa_temp(acc_pct: f64, persist_pct: f64) -> f64 {
    let denom = 100.0 - persist_pct;
    if denom.abs() < 1e-6 {
        0.0
    } else {
        (acc_pct - persist_pct) / denom
    }
}

// -----------------------------------------------------------------------------
// Генераторы стандартных потоков Concept Drift
// -----------------------------------------------------------------------------

fn gen_sea(seed: u64) -> (Vec<(Vec<f32>, usize)>, Vec<usize>, (usize, usize)) {
    let mut rng = splitmix64(seed ^ 0x5EA0_0001);
    let thetas = [8.0_f32, 9.0, 7.0, 9.5, 8.0];
    let block = 1000usize;
    let mut data = Vec::with_capacity(thetas.len() * block);
    let mut drifts = Vec::new();
    for (bi, &th) in thetas.iter().enumerate() {
        if bi > 0 {
            drifts.push(bi * block);
        }
        for _ in 0..block {
            let x1 = rand_unit_f32(&mut rng) * 10.0;
            let x2 = rand_unit_f32(&mut rng) * 10.0;
            let x3 = rand_unit_f32(&mut rng) * 10.0;
            let mut y = if x1 + x2 <= th { 1usize } else { 0usize };
            if rand_unit_f32(&mut rng) < 0.10 {
                y = 1 - y;
            }
            data.push((vec![x1, x2, x3], y));
        }
    }
    (data, drifts, (4 * block, 5 * block))
}

fn gen_stagger(seed: u64) -> (Vec<(Vec<f32>, usize)>, Vec<usize>, (usize, usize)) {
    let mut rng = splitmix64(seed ^ 0x57A6_6E20);
    let rules = [0usize, 1, 2, 0, 1];
    let block = 600usize;
    let mut data = Vec::with_capacity(rules.len() * block);
    let mut drifts = Vec::new();
    for (bi, &rule) in rules.iter().enumerate() {
        if bi > 0 {
            drifts.push(bi * block);
        }
        for _ in 0..block {
            rng = splitmix64(rng);
            let size = (rng as usize) % 3;
            let color = ((rng >> 8) as usize) % 3;
            let shape = ((rng >> 16) as usize) % 3;
            let y = match rule {
                0 => usize::from(size == 0 && color == 0),
                1 => usize::from(color == 1 || shape == 0),
                _ => usize::from(size == 1 || size == 2),
            };
            let mut f = vec![0.0_f32; 9];
            f[size] = 1.0;
            f[3 + color] = 1.0;
            f[6 + shape] = 1.0;
            data.push((f, y));
        }
    }
    (data, drifts, (3 * block, 5 * block))
}

fn gen_rotating_hyperplane(seed: u64) -> (Vec<(Vec<f32>, usize)>, Vec<usize>, (usize, usize)) {
    let mut rng = splitmix64(seed ^ 0x47DE_291A);
    let dim = 10usize;
    let n = 5000usize;
    let mut w = vec![0.0_f32; dim];
    let mut dir = vec![1.0_f32; dim];
    for j in 0..dim {
        w[j] = rand_unit_f32(&mut rng);
        dir[j] = if (j & 1) == 0 { 1.0 } else { -1.0 };
    }
    let mut data = Vec::with_capacity(n);
    for t in 0..n {
        let mut x = vec![0.0_f32; dim];
        let mut dot = 0.0_f32;
        let mut wsum = 0.0_f32;
        for j in 0..dim {
            x[j] = rand_unit_f32(&mut rng);
            dot += w[j] * x[j];
            wsum += w[j];
        }
        let mut y = if dot >= 0.5 * wsum { 1usize } else { 0usize };
        if rand_unit_f32(&mut rng) < 0.10 {
            y = 1 - y;
        }
        data.push((x, y));

        for j in 0..5 {
            w[j] += 0.0012 * dir[j];
            if w[j] > 1.0 || w[j] < 0.0 {
                dir[j] = -dir[j];
                w[j] = w[j].clamp(0.0, 1.0);
            }
        }
        if t == 2500 {
            for j in 5..dim {
                dir[j] = -dir[j];
            }
        }
    }
    (data, vec![2500], (4000, 5000))
}

/// 4-классовый поток Multi-Class RBF Drift (10D, 8000 строк, рекуррентный дрейф A -> B -> C -> A):
/// - строго i.i.d. сэмплирование класса (нулевая автокорреляция -> No-Change ~ 25%!),
/// - 8% шума меток, нелинейно-разделимые центроиды + плавный дрейф радиуса.
fn gen_multiclass_rbf_drift(seed: u64) -> (Vec<(Vec<f32>, usize)>, Vec<usize>, (usize, usize)) {
    let mut rng = splitmix64(seed ^ 0x2BF4_C1A5);
    let dim = 10usize;
    let n_classes = 4usize;
    let block = 2000usize;
    // 3 разных набора центроидов (режимы A, B, C), возврат A -> B -> C -> A
    let mut regimes = vec![vec![vec![0.0_f32; dim]; n_classes]; 3];
    for r in 0..3 {
        for c in 0..n_classes {
            for j in 0..dim {
                regimes[r][c][j] = rand_signed_f32(&mut rng) * 1.3;
            }
        }
    }
    let schedule = [0usize, 1, 2, 0];
    let mut data = Vec::with_capacity(schedule.len() * block);
    let mut drifts = Vec::new();
    for (bi, &reg_id) in schedule.iter().enumerate() {
        if bi > 0 {
            drifts.push(bi * block);
        }
        for _ in 0..block {
            rng = splitmix64(rng);
            let true_c = ((rng >> 12) as usize) % n_classes;
            let mut x = vec![0.0_f32; dim];
            for j in 0..dim {
                // Треугольный шум ~ N(0, 0.45)
                let noise = (rand_signed_f32(&mut rng) + rand_signed_f32(&mut rng)) * 0.45;
                x[j] = regimes[reg_id][true_c][j] + noise;
            }
            let y = if rand_unit_f32(&mut rng) < 0.08 {
                rng = splitmix64(rng);
                ((rng >> 16) as usize) % n_classes
            } else {
                true_c
            };
            data.push((x, y));
        }
    }
    (data, drifts, (3 * block, 4 * block))
}

fn try_load_electricity_csv(path: &Path) -> Option<Vec<(Vec<f32>, usize)>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut out = Vec::new();
    let is_arff = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("arff"))
        .unwrap_or(false)
        || text.contains("@relation");
    let mut in_data = !is_arff;

    for (li, line) in text.lines().enumerate() {
        let tr = line.trim();
        if tr.is_empty() || tr.starts_with('%') {
            continue;
        }
        if is_arff {
            if !in_data {
                if tr.eq_ignore_ascii_case("@data") {
                    in_data = true;
                }
                continue;
            }
        } else if li == 0 && tr.chars().any(|c| c.is_alphabetic()) {
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
        let lbl_str = cols[cols.len() - 1]
            .trim()
            .trim_matches('\'')
            .trim_matches('"')
            .to_ascii_lowercase();
        let y = if lbl_str == "up" || lbl_str == "1" || lbl_str == "true" { 1 } else { 0 };
        out.push((feats, y));
    }
    if out.len() >= 1000 { Some(out) } else { None }
}

/// Добавляет строго каузальную предыдущую метку y_{t-1} ∈ {-1, +1} (доступную ДО прогноза y_t).
fn augment_with_prev_label(data: &[(Vec<f32>, usize)]) -> Vec<(Vec<f32>, usize)> {
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

// -----------------------------------------------------------------------------
// Прогон моделей на одном потоке
// -----------------------------------------------------------------------------

fn finalize_metrics(
    hits: &[bool],
    drifts: &[usize],
    ret_range: (usize, usize),
    rec_target: f64,
    ram_kb: f64,
    elapsed_us: f64,
    seu_safe: bool,
) -> StreamMetrics {
    let preq = hits.iter().filter(|&&b| b).count() as f64 * 100.0 / (hits.len().max(1) as f64);
    let r_slice = &hits[ret_range.0.min(hits.len())..ret_range.1.min(hits.len())];
    let ret = r_slice.iter().filter(|&&b| b).count() as f64 * 100.0 / (r_slice.len().max(1) as f64);
    let rec = if drifts.is_empty() {
        0.0
    } else {
        drifts
            .iter()
            .map(|&d| recovery_after_drift(hits, d, rec_target))
            .sum::<f64>()
            / (drifts.len() as f64)
    };
    StreamMetrics {
        prequential_acc: preq,
        retention_acc: ret,
        recovery_steps: rec,
        ram_kb,
        us_per_step: elapsed_us / (hits.len().max(1) as f64),
        seu_safe,
    }
}

fn eval_micro(
    data: &[(Vec<f32>, usize)],
    num_classes: usize,
    drifts: &[usize],
    ret_range: (usize, usize),
    rec_target: f64,
    temporal: bool,
    seed: u64,
) -> StreamMetrics {
    let mut cfg = MicroCoreConfig::numeric_stream(num_classes, seed);
    if temporal {
        // Мягкая поворотная память CayleyBeliefState между соседними 30-минутными интервалами
        // (w_x = sqrt(1 - 0.35^2) = 0.937 сохраняет чёткость мгновенного рынка + 35% инерции тренда!)
        cfg.belief_lambda = 0.35;
        cfg.rotor_norm = 0.12;
        cfg.linear_lr = 0.09;
        cfg.linear_mix = 0.90;
    }

    let mut core = MicroContinualCore::new(cfg);
    core.encoder.freeze_to_i8();
    if temporal {
        core.start_episode();
    }

    let mut hits = Vec::with_capacity(data.len());
    let t0 = Instant::now();
    for (x, y) in data {
        if !temporal {
            core.start_episode();
        }
        let pred = core.decide_vec(x, false, false, false).get().0;
        hits.push(pred == *y);
        let _ = core.end_supervised_step(pred, *y);
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    core.start_episode();
    let p_seu = core.decide_vec(&[f32::NAN, 1e30, -1e30], false, true, true);
    let seu_safe = p_seu.get().1.is_finite() && core.belief.state.iter().all(|v| v.is_finite());

    finalize_metrics(
        &hits,
        drifts,
        ret_range,
        rec_target,
        core.memory_footprint_bytes() as f64 / 1024.0,
        elapsed_us,
        seu_safe,
    )
}

fn eval_micro_rtrl_ph(
    data: &[(Vec<f32>, usize)],
    num_classes: usize,
    drifts: &[usize],
    ret_range: (usize, usize),
    rec_target: f64,
    ph_threshold: f32,
    seed: u64,
) -> StreamMetrics {
    let in_dim = data.first().map(|d| d.0.len()).unwrap_or(4).min(12);
    let mut cfg = MicroCoreConfig::numeric_stream(num_classes, seed);
    cfg.half_sectors = 12;
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
    cfg.ph_threshold = ph_threshold;

    let mut core = MicroContinualCore::new(cfg);
    core.encoder.freeze_to_i8();
    core.start_episode();

    let mut hits = Vec::with_capacity(data.len());
    let t0 = Instant::now();
    for (x, y) in data {
        let pred = core.decide_vec(x, false, false, false).get().0;
        hits.push(pred == *y);
        let _ = core.end_supervised_step(pred, *y);
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    core.start_episode();
    let p_seu = core.decide_vec(&[f32::NAN, 1e30, -1e30], false, true, true);
    let seu_safe = p_seu.get().1.is_finite()
        && core.belief.state.iter().all(|v| v.is_finite())
        && core.rtrl_state.b.iter().all(|v| v.is_finite())
        && core.rtrl_slow_state.b.iter().all(|v| v.is_finite());

    finalize_metrics(
        &hits,
        drifts,
        ret_range,
        rec_target,
        core.memory_footprint_bytes() as f64 / 1024.0,
        elapsed_us,
        seu_safe,
    )
}

fn eval_alg_cayley_stream(
    data: &[(Vec<f32>, usize)],
    num_classes: usize,
    drifts: &[usize],
    ret_range: (usize, usize),
    rec_target: f64,
    selective: bool,
    seed: u64,
) -> StreamMetrics {
    let in_dim = data.first().map(|d| d.0.len()).unwrap_or(1);
    let mut m = AlgebraicSelectiveCayley::new(in_dim, 32, num_classes, selective, seed)
        .with_standardize(true);
    let mut hits = Vec::with_capacity(data.len());
    let t0 = Instant::now();
    for (x, y) in data {
        let pred = m.step_and_learn_supervised(x, Some(*y), 0.015, 0.005);
        hits.push(pred == *y);
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let mut probe = AlgebraicSelectiveCayley::new(in_dim, 32, num_classes, selective, seed)
        .with_standardize(true);
    let mut bad_x = vec![0.0_f32; in_dim];
    if !bad_x.is_empty() { bad_x[0] = f32::NAN; }
    if bad_x.len() > 1 { bad_x[1] = 1e28; }
    probe.inject_register_bit_flip(0, 30);
    probe.inject_weight_bit_flip(0, 30);
    let _ = probe.step_and_learn_supervised(&bad_x, None, 0.0, 0.0);
    let seu_safe = probe.h_re.iter().all(|v| v.is_finite())
        && probe.h_im.iter().all(|v| v.is_finite())
        && probe.out_w.iter().all(|v| v.is_finite())
        && probe.total_seu_faults() >= 2;

    finalize_metrics(
        &hits,
        drifts,
        ret_range,
        rec_target,
        m.memory_footprint_bytes() as f64 / 1024.0,
        elapsed_us,
        seu_safe,
    )
}

fn eval_micro_rtrl(
    data: &[(Vec<f32>, usize)],
    num_classes: usize,
    drifts: &[usize],
    ret_range: (usize, usize),
    rec_target: f64,
    seed: u64,
) -> StreamMetrics {
    eval_micro_rtrl_ph(data, num_classes, drifts, ret_range, rec_target, 3.6, seed)
}

fn eval_mlp(
    data: &[(Vec<f32>, usize)],
    num_classes: usize,
    drifts: &[usize],
    ret_range: (usize, usize),
    rec_target: f64,
    seed: u64,
) -> StreamMetrics {
    let in_dim = data.first().map(|d| d.0.len()).unwrap_or(1);
    let mut mlp = OnlineMlpAdam::new(in_dim, 32, num_classes, seed);
    let mut hits = Vec::with_capacity(data.len());
    let t0 = Instant::now();
    for (x, y) in data {
        let pred = mlp.predict_and_train(x, *y, true);
        hits.push(pred == *y);
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let mut probe = OnlineMlpAdam::new(in_dim, 32, num_classes, seed);
    let mut bad_x = vec![0.0_f32; in_dim];
    bad_x[0] = f32::NAN;
    let _ = probe.predict_and_train(&bad_x, 0, false);

    finalize_metrics(
        &hits,
        drifts,
        ret_range,
        rec_target,
        mlp.memory_footprint_bytes() as f64 / 1024.0,
        elapsed_us,
        !probe.had_nan,
    )
}

fn eval_nb(
    data: &[(Vec<f32>, usize)],
    num_classes: usize,
    drifts: &[usize],
    ret_range: (usize, usize),
    rec_target: f64,
) -> StreamMetrics {
    let in_dim = data.first().map(|d| d.0.len()).unwrap_or(1);
    let mut nb = OnlineGaussianNaiveBayes::new(in_dim, num_classes, 0.992);
    let mut hits = Vec::with_capacity(data.len());
    let t0 = Instant::now();
    for (x, y) in data {
        let pred = nb.predict_and_train(x, *y, true);
        hits.push(pred == *y);
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let mut probe = OnlineGaussianNaiveBayes::new(in_dim, num_classes, 0.992);
    let mut bad_x = vec![0.0_f32; in_dim];
    bad_x[0] = f32::NAN;
    let _ = probe.predict_and_train(&bad_x, 0, false);

    finalize_metrics(
        &hits,
        drifts,
        ret_range,
        rec_target,
        nb.memory_footprint_bytes() as f64 / 1024.0,
        elapsed_us,
        !probe.had_nan,
    )
}

fn eval_tab(
    data: &[(Vec<f32>, usize)],
    num_classes: usize,
    drifts: &[usize],
    ret_range: (usize, usize),
    rec_target: f64,
) -> StreamMetrics {
    let mut tab = TabularBinnedPH::new(num_classes, 6);
    let mut hits = Vec::with_capacity(data.len());
    let t0 = Instant::now();
    for (x, y) in data {
        let pred = tab.predict_and_train(x, *y, true);
        hits.push(pred == *y);
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let mut probe = TabularBinnedPH::new(num_classes, 6);
    let bad_x = [f32::NAN, 1e30, -1e30];
    let p = probe.predict_and_train(&bad_x, 0, false);

    finalize_metrics(
        &hits,
        drifts,
        ret_range,
        rec_target,
        tab.memory_footprint_bytes() as f64 / 1024.0,
        elapsed_us,
        p < num_classes,
    )
}

fn eval_lru_stream(
    data: &[(Vec<f32>, usize)],
    num_classes: usize,
    drifts: &[usize],
    ret_range: (usize, usize),
    rec_target: f64,
    seed: u64,
) -> StreamMetrics {
    let in_dim = data.first().map(|d| d.0.len()).unwrap_or(1);
    let mut lru = OnlineLruStream::new(in_dim, 32, num_classes, seed);
    let mut hits = Vec::with_capacity(data.len());
    let t0 = Instant::now();
    for (x, y) in data {
        let pred = lru.predict_and_train(x, *y, true);
        hits.push(pred == *y);
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let mut probe = OnlineLruStream::new(in_dim, 32, num_classes, seed);
    let mut bad_x = vec![0.0_f32; in_dim];
    if !bad_x.is_empty() {
        bad_x[0] = f32::NAN;
    }
    if bad_x.len() > 1 {
        bad_x[1] = 1e28;
    }
    let _ = probe.predict_and_train(&bad_x, 0, false);

    finalize_metrics(
        &hits,
        drifts,
        ret_range,
        rec_target,
        lru.memory_footprint_bytes() as f64 / 1024.0,
        elapsed_us,
        !probe.had_nan,
    )
}

fn eval_persist(
    data: &[(Vec<f32>, usize)],
    drifts: &[usize],
    ret_range: (usize, usize),
    rec_target: f64,
) -> StreamMetrics {
    let mut nc = NoChangePersistent::new();
    let mut hits = Vec::with_capacity(data.len());
    let t0 = Instant::now();
    for (x, y) in data {
        let pred = nc.predict_and_train(x, *y, true);
        hits.push(pred == *y);
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
    finalize_metrics(
        &hits,
        drifts,
        ret_range,
        rec_target,
        nc.memory_footprint_bytes() as f64 / 1024.0,
        elapsed_us,
        true,
    )
}

fn print_row(name: &str, ms: &[StreamMetrics], persist_acc: f64) {
    let sc: Vec<f64> = ms.iter().map(|m| m.prequential_acc).collect();
    let rt: Vec<f64> = ms.iter().map(|m| m.retention_acc).collect();
    let rc: Vec<f64> = ms.iter().map(|m| m.recovery_steps).collect();
    let rm: Vec<f64> = ms.iter().map(|m| m.ram_kb).collect();
    let lt: Vec<f64> = ms.iter().map(|m| m.us_per_step).collect();
    let seu = ms.iter().all(|m| m.seu_safe);

    let (sm, ss) = mean_sd(&sc);
    let (rtm, _) = mean_sd(&rt);
    let (rcm, _) = mean_sd(&rc);
    let (rmm, _) = mean_sd(&rm);
    let (ltm, _) = mean_sd(&lt);
    let kt = kappa_temp(sm, persist_acc);

    println!(
        "{:<24} | {:5.1}±{:4.1}% | {:+6.3} | {:7.1}% | {:8.1} st | {:7.1} KB | {:6.2} µs | {}",
        name,
        sm,
        ss,
        kt,
        rtm,
        rcm,
        rmm,
        ltm,
        if seu { "YES" } else { "NO (NaN)" }
    );
}

fn run_synthetic<F>(title: &str, num_classes: usize, rec_target: f64, gen_fn: F)
where
    F: Fn(u64) -> (Vec<(Vec<f32>, usize)>, Vec<usize>, (usize, usize)),
{
    println!("\n=== {} (1-pass Test-then-Train, 10 seeds) ===", title);
    println!(
        "{:<24} | {:<11} | {:<6} | {:<8} | {:<11} | {:<10} | {:<9} | {}",
        "Model", "Score(m±sd)", "KappaT", "Retent.", "Recovery", "RAM (KB)", "Latency", "SEU-Safe"
    );
    println!("{}", "-".repeat(108));

    let mut r_mic = Vec::new();
    let mut r_mlp = Vec::new();
    let mut r_nb = Vec::new();
    let mut r_tab = Vec::new();
    let mut r_nc = Vec::new();

    for seed in 1..=10u64 {
        let (data, drifts, ret_r) = gen_fn(seed);
        r_mic.push(eval_micro(&data, num_classes, &drifts, ret_r, rec_target, false, seed));
        r_mlp.push(eval_mlp(&data, num_classes, &drifts, ret_r, rec_target, seed));
        r_nb.push(eval_nb(&data, num_classes, &drifts, ret_r, rec_target));
        r_tab.push(eval_tab(&data, num_classes, &drifts, ret_r, rec_target));
        r_nc.push(eval_persist(&data, &drifts, ret_r, rec_target));
    }
    let persist_acc = r_nc.iter().map(|m| m.prequential_acc).sum::<f64>() / (r_nc.len() as f64);
    print_row("MicroContinualCore", &r_mic, persist_acc);
    print_row("Online MLP (H=32)+Adam", &r_mlp, persist_acc);
    print_row("Online Gaussian NB", &r_nb, persist_acc);
    print_row("Tabular Binned + PH", &r_tab, persist_acc);
    print_row("No-Change (Persist)", &r_nc, persist_acc);
}

fn main() {
    run_synthetic("SEA Concepts (Recurrent Abrupt Drift + 10% noise, 5k)", 2, 0.78, gen_sea);
    run_synthetic("STAGGER (Recurrent Abrupt Drift A->B->C->A->B, 3k)", 2, 0.85, gen_stagger);
    run_synthetic("Rotating Hyperplane 10D (Gradual Drift + 10% noise, 5k)", 2, 0.75, gen_rotating_hyperplane);
    run_synthetic("Uncorrelated Multi-Class RBF Drift (10D, 4-cls, 8k)", 4, 0.70, gen_multiclass_rbf_drift);

    let cli_path = std::env::args().nth(1);
    let candidates: Vec<String> = if let Some(p) = cli_path {
        vec![p]
    } else {
        vec![
            "electricity.csv".to_string(),
            "elecNormNew.csv".to_string(),
            "electricity-normalized.arff".to_string(),
            "../micro_continual_core/electricity-normalized.arff".to_string(),
            "data/electricity.csv".to_string(),
            "data/electricity-normalized.arff".to_string(),
        ]
    };
    let mut found = false;
    for p in &candidates {
        if let Some(data) = try_load_electricity_csv(Path::new(p)) {
            found = true;
            let data_yprev = augment_with_prev_label(&data);
            println!("\n=== Electricity ELEC2 Real Stream ({} rows, 1-pass Test-then-Train) ===", data.len());
            println!(
                "{:<24} | {:<11} | {:<6} | {:<8} | {:<11} | {:<10} | {:<9} | {}",
                "Model", "Score(m±sd)", "KappaT", "Tail20%", "Recovery", "RAM (KB)", "Latency", "SEU-Safe"
            );
            println!("{}", "-".repeat(108));
            let tail = (data.len() * 4 / 5, data.len());
            let r_nc = vec![eval_persist(&data, &[], tail, 0.75)];
            let persist_acc = r_nc[0].prequential_acc;
            let r_mic: Vec<_> = (1..=3).map(|s| eval_micro(&data, 2, &[], tail, 0.75, false, s)).collect();
            let r_mic_t: Vec<_> = (1..=3).map(|s| eval_micro(&data, 2, &[], tail, 0.75, true, s)).collect();
            let r_mic_rtrl: Vec<_> = (1..=3).map(|s| eval_micro_rtrl(&data, 2, &[], tail, 0.75, s)).collect();
            let r_mic_rtrl_ph8: Vec<_> = (1..=3).map(|s| eval_micro_rtrl_ph(&data, 2, &[], tail, 0.75, 8.0, s)).collect();
            let r_c2d: Vec<_> = (1..=3).map(|s| eval_alg_cayley_stream(&data, 2, &[], tail, 0.75, false, s)).collect();
            let r_asc: Vec<_> = (1..=3).map(|s| eval_alg_cayley_stream(&data, 2, &[], tail, 0.75, true, s)).collect();
            let r_mic_y: Vec<_> = (1..=3).map(|s| eval_micro(&data_yprev, 2, &[], tail, 0.75, false, s)).collect();
            let r_mic_ty: Vec<_> = (1..=3).map(|s| eval_micro(&data_yprev, 2, &[], tail, 0.75, true, s)).collect();
            let r_mic_rtrl_y: Vec<_> = (1..=3).map(|s| eval_micro_rtrl(&data_yprev, 2, &[], tail, 0.75, s)).collect();
            let r_mic_rtrl_y_ph8: Vec<_> = (1..=3).map(|s| eval_micro_rtrl_ph(&data_yprev, 2, &[], tail, 0.75, 8.0, s)).collect();
            let r_c2d_y: Vec<_> = (1..=3).map(|s| eval_alg_cayley_stream(&data_yprev, 2, &[], tail, 0.75, false, s)).collect();
            let r_asc_y: Vec<_> = (1..=3).map(|s| eval_alg_cayley_stream(&data_yprev, 2, &[], tail, 0.75, true, s)).collect();
            let r_mlp: Vec<_> = (1..=3).map(|s| eval_mlp(&data, 2, &[], tail, 0.75, s)).collect();
            let r_mlp_y: Vec<_> = (1..=3).map(|s| eval_mlp(&data_yprev, 2, &[], tail, 0.75, s)).collect();
            let r_lru: Vec<_> = (1..=3).map(|s| eval_lru_stream(&data, 2, &[], tail, 0.75, s)).collect();
            let r_lru_y: Vec<_> = (1..=3).map(|s| eval_lru_stream(&data_yprev, 2, &[], tail, 0.75, s)).collect();
            let r_nb = vec![eval_nb(&data, 2, &[], tail, 0.75)];
            let r_tab = vec![eval_tab(&data, 2, &[], tail, 0.75)];
            print_row("MicroCore (i.i.d.)", &r_mic, persist_acc);
            print_row("MicroCore (Temporal)", &r_mic_t, persist_acc);
            print_row("MicroCore+2L-RTRL(Temp)", &r_mic_rtrl, persist_acc);
            print_row("MicroCore+2L-RTRL(PH=8)", &r_mic_rtrl_ph8, persist_acc);
            print_row("2D Rat. Cayley (0% exp)", &r_c2d, persist_acc);
            print_row("Alg.Sel.Cayley (0% exp)", &r_asc, persist_acc);
            print_row("MicroCore (+y_{t-1})", &r_mic_y, persist_acc);
            print_row("MicroCore (Temp+y_{t-1})", &r_mic_ty, persist_acc);
            print_row("MicroCore+RTRL (+y_{t-1})", &r_mic_rtrl_y, persist_acc);
            print_row("MicroCore+RTRL(PH8,+y-1)", &r_mic_rtrl_y_ph8, persist_acc);
            print_row("2D Rat.Cayley (+y_{t-1})", &r_c2d_y, persist_acc);
            print_row("Alg.Sel.Cayley(+y_{t-1})", &r_asc_y, persist_acc);
            print_row("Online LRU-RTRL (Temp)", &r_lru, persist_acc);
            print_row("Online LRU-RTRL (+y_{t-1})", &r_lru_y, persist_acc);
            print_row("Online MLP (H=32)+Adam", &r_mlp, persist_acc);
            print_row("Online MLP (+y_{t-1})", &r_mlp_y, persist_acc);
            print_row("Online Gaussian NB", &r_nb, persist_acc);
            print_row("Tabular Binned + PH", &r_tab, persist_acc);
            print_row("No-Change (Persist)", &r_nc, persist_acc);
            break;
        }
    }
    if !found {
        println!("\n[Info] Для реального потока Electricity (ELEC2) положите elecNormNew.csv рядом или передайте путь:");
        println!("       cargo run --release --bin bench_step2_streams -- path/to/elecNormNew.csv");
    }
}
