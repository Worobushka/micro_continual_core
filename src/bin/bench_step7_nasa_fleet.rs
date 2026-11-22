// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov & Sofia Sidorenko
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Бенчмарк Шага 7 (v0.4.0): Авиационный бенчмарк NASA C-MAPSS (`FD001–FD004`)
//! Строго аудит-стойкий, без утечек test/RUL, с полным равенством размерности состояния (60D):
//! - Блок 1: Потоковый прогноз RUL (`1-pass` и `3-pass`, 5 сидов) по `FD001..FD004` +
//!           сырой поток `FD004` (24 входа без кластеризации режимов) + `RUL-Shuffle` тест утечки.
//! - Блок 2: Высотная радиация в полёте (`Flight SEU` на `test_FD004`, 248 двигателей):
//!           `Clean / Sensor-SEU (T30=NaN/+1e30) / State-SEU (Bit-30 / 25 cycles) / Weight-SEU (Bit-30 / 25 cycles)`
//!           для `MicroCore`, `Alg.Sel.Cayley` и всех бейзлайнов в ОБOИХ режимах `(+Clamp)` и `(+FinGuard)`.
//! - Блок 3: Рой цифровых двойников авиапарка (`N = 248 -> 1,024 -> 4,096` двигателей,
//!           1-поточная задержка, суммарная RAM, 16-поточный throughput).
//! - Экспорт телеметрии в `./logs/nasa_engine_trace.csv` и `./logs/nasa_fleet_frames.csv`.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use micro_continual_core::baselines::{DeepRtrrlLruBank, GruPerStepBank, SelectiveSsmBank};
use micro_continual_core::math::splitmix64;
use micro_continual_core::rtrl::AlgebraicSelectiveCayley;
use micro_continual_core::stream_baselines::OnlineMlpAdam;
use micro_continual_core::{MicroContinualCore, MicroCoreConfig};

const RUL_MAX: f32 = 125.0;
const CRIT_RUL: f32 = 30.0;
const STATE_DIM: usize = 60; // Строгое равенство скрытого состояния: 60D для всех рекуррентных моделей

#[derive(Clone)]
struct RawRow {
    unit_id: usize,
    cycle: usize,
    ops: [f32; 3],
    sensors: [f32; 21],
}

#[derive(Clone)]
struct EngineSeq {
    unit_id: usize,
    cycles: Vec<usize>,
    regimes: Vec<usize>,
    raw_t30: Vec<f32>, // s3 (col index 7 in 26-col file -> sensors[2])
    raw_nc: Vec<f32>,  // s9 (col index 13 in 26-col file -> sensors[8])
    feats: Vec<Vec<f32>>,
    true_rul: Vec<f32>, // piecewise-linear clipped to [0, RUL_MAX]
    end_rul: f32,
}

struct CmapssDataset {
    name: &'static str,
    in_dim: usize,
    train_units: Vec<EngineSeq>,
    test_units: Vec<EngineSeq>,
}

fn parse_cmapss_file(path: &Path) -> Vec<RawRow> {
    let f = File::open(path).unwrap_or_else(|e| panic!("Cannot open {}: {e}", path.display()));
    let reader = BufReader::new(f);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let l = line.expect("I/O error reading C-MAPSS");
        let parts: Vec<&str> = l.split_whitespace().collect();
        if parts.len() < 26 {
            continue;
        }
        let unit_id: usize = parts[0].parse().expect("unit_id");
        let cycle: usize = parts[1].parse().expect("cycle");
        let mut ops = [0.0_f32; 3];
        for i in 0..3 {
            ops[i] = parts[2 + i].parse().expect("op");
        }
        let mut sensors = [0.0_f32; 21];
        for i in 0..21 {
            sensors[i] = parts[5 + i].parse().expect("sensor");
        }
        rows.push(RawRow {
            unit_id,
            cycle,
            ops,
            sensors,
        });
    }
    rows
}

fn parse_rul_file(path: &Path) -> Vec<f32> {
    let f = File::open(path).unwrap_or_else(|e| panic!("Cannot open {}: {e}", path.display()));
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines() {
        let l = line.expect("I/O error reading RUL");
        let s = l.trim();
        if s.is_empty() {
            continue;
        }
        out.push(s.parse::<f32>().expect("rul float"));
    }
    out
}

/// Детерминированный K-means (K=6) в нормированном пространстве `(op1, op2, op3)` СТРОГО по `train_rows`.
struct RegimeNormalizer {
    k: usize,
    op_mean: [f64; 3],
    op_std: [f64; 3],
    centroids: Vec<[f64; 3]>,
    s_mean: Vec<[f64; 21]>,
    s_std: Vec<[f64; 21]>,
    raw_mode: bool,
    raw_mean: [f64; 24],
    raw_std: [f64; 24],
}

impl RegimeNormalizer {
    fn fit_train_only(train_rows: &[RawRow], multi_regime: bool, raw_mode: bool) -> Self {
        let n = train_rows.len().max(1) as f64;
        if raw_mode {
            let mut mean = [0.0_f64; 24];
            for r in train_rows {
                for j in 0..3 { mean[j] += r.ops[j] as f64; }
                for j in 0..21 { mean[3 + j] += r.sensors[j] as f64; }
            }
            for m in &mut mean { *m /= n; }
            let mut var = [0.0_f64; 24];
            for r in train_rows {
                for j in 0..3 { let d = r.ops[j] as f64 - mean[j]; var[j] += d * d; }
                for j in 0..21 { let d = r.sensors[j] as f64 - mean[3 + j]; var[3 + j] += d * d; }
            }
            let mut std = [0.0_f64; 24];
            for j in 0..24 {
                std[j] = (var[j] / n).sqrt();
            }
            return Self {
                k: 1,
                op_mean: [0.0; 3],
                op_std: [1.0; 3],
                centroids: vec![[0.0; 3]],
                s_mean: vec![[0.0; 21]],
                s_std: vec![[1.0; 21]],
                raw_mode: true,
                raw_mean: mean,
                raw_std: std,
            };
        }

        if !multi_regime {
            let mut s_mean = [0.0_f64; 21];
            for r in train_rows {
                for j in 0..21 { s_mean[j] += r.sensors[j] as f64; }
            }
            for m in &mut s_mean { *m /= n; }
            let mut s_var = [0.0_f64; 21];
            for r in train_rows {
                for j in 0..21 {
                    let d = r.sensors[j] as f64 - s_mean[j];
                    s_var[j] += d * d;
                }
            }
            let mut s_std = [0.0_f64; 21];
            for j in 0..21 { s_std[j] = (s_var[j] / n).sqrt(); }
            return Self {
                k: 1,
                op_mean: [0.0; 3],
                op_std: [1.0; 3],
                centroids: vec![[0.0; 3]],
                s_mean: vec![s_mean],
                s_std: vec![s_std],
                raw_mode: false,
                raw_mean: [0.0; 24],
                raw_std: [1.0; 24],
            };
        }

        // 6-режимный C-MAPSS (FD002/FD004): нормировка op1..op3 + farthest-point K-means строго по train
        let mut op_mean = [0.0_f64; 3];
        for r in train_rows {
            for j in 0..3 { op_mean[j] += r.ops[j] as f64; }
        }
        for m in &mut op_mean { *m /= n; }
        let mut op_var = [0.0_f64; 3];
        for r in train_rows {
            for j in 0..3 {
                let d = r.ops[j] as f64 - op_mean[j];
                op_var[j] += d * d;
            }
        }
        let mut op_std = [1.0_f64; 3];
        for j in 0..3 {
            let s = (op_var[j] / n).sqrt();
            op_std[j] = if s > 1e-6 { s } else { 1.0 };
        }
        let norm_op = |ops: &[f32; 3]| -> [f64; 3] {
            [
                (ops[0] as f64 - op_mean[0]) / op_std[0],
                (ops[1] as f64 - op_mean[1]) / op_std[1],
                (ops[2] as f64 - op_mean[2]) / op_std[2],
            ]
        };
        let dist2 = |a: &[f64; 3], b: &[f64; 3]| -> f64 {
            let d0 = a[0] - b[0];
            let d1 = a[1] - b[1];
            let d2 = a[2] - b[2];
            d0 * d0 + d1 * d1 + d2 * d2
        };

        let k = 6usize;
        let ops_n: Vec<[f64; 3]> = train_rows.iter().map(|r| norm_op(&r.ops)).collect();
        // Детерминированный farthest-point seeding (гарантирует все 6 реальных режимов полёта)
        let mut centroids = Vec::with_capacity(k);
        centroids.push(ops_n[0]);
        for _ in 1..k {
            let mut best_idx = 0usize;
            let mut best_d = -1.0_f64;
            for (idx, p) in ops_n.iter().enumerate() {
                let mut md = f64::INFINITY;
                for c in &centroids {
                    let d = dist2(p, c);
                    if d < md { md = d; }
                }
                if md > best_d {
                    best_d = md;
                    best_idx = idx;
                }
            }
            centroids.push(ops_n[best_idx]);
        }
        // 25 итераций Lloyd K-means строго на train
        let mut assign = vec![0usize; ops_n.len()];
        for _ in 0..25 {
            let mut sums = vec![[0.0_f64; 3]; k];
            let mut cnts = vec![0usize; k];
            for (i, p) in ops_n.iter().enumerate() {
                let mut bc = 0usize;
                let mut bd = f64::INFINITY;
                for (ci, c) in centroids.iter().enumerate() {
                    let d = dist2(p, c);
                    if d < bd { bd = d; bc = ci; }
                }
                assign[i] = bc;
                cnts[bc] += 1;
                for j in 0..3 { sums[bc][j] += p[j]; }
            }
            for ci in 0..k {
                if cnts[ci] > 0 {
                    for j in 0..3 { centroids[ci][j] = sums[ci][j] / cnts[ci] as f64; }
                }
            }
        }
        // Сортируем центроиды по op1 для воспроизводимой нумерации режимов 0..5
        centroids.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));

        // Порежимные mean/std 21 датчика СТРОГО на train
        let mut s_sum = vec![[0.0_f64; 21]; k];
        let mut s_cnt = vec![0usize; k];
        for (i, r) in train_rows.iter().enumerate() {
            let p = ops_n[i];
            let mut bc = 0usize;
            let mut bd = f64::INFINITY;
            for (ci, c) in centroids.iter().enumerate() {
                let d = dist2(&p, c);
                if d < bd { bd = d; bc = ci; }
            }
            assign[i] = bc;
            s_cnt[bc] += 1;
            for j in 0..21 { s_sum[bc][j] += r.sensors[j] as f64; }
        }
        let mut s_mean = vec![[0.0_f64; 21]; k];
        for ci in 0..k {
            let cnt = s_cnt[ci].max(1) as f64;
            for j in 0..21 { s_mean[ci][j] = s_sum[ci][j] / cnt; }
        }
        let mut s_var = vec![[0.0_f64; 21]; k];
        for (i, r) in train_rows.iter().enumerate() {
            let ci = assign[i];
            for j in 0..21 {
                let d = r.sensors[j] as f64 - s_mean[ci][j];
                s_var[ci][j] += d * d;
            }
        }
        let mut s_std = vec![[0.0_f64; 21]; k];
        for ci in 0..k {
            let cnt = s_cnt[ci].max(1) as f64;
            for j in 0..21 { s_std[ci][j] = (s_var[ci][j] / cnt).sqrt(); }
        }
        Self {
            k,
            op_mean,
            op_std,
            centroids,
            s_mean,
            s_std,
            raw_mode: false,
            raw_mean: [0.0; 24],
            raw_std: [1.0; 24],
        }
    }

    fn transform(&self, r: &RawRow) -> (usize, Vec<f32>) {
        if self.raw_mode {
            let mut v = vec![0.0_f32; 24];
            for j in 0..3 {
                v[j] = if self.raw_std[j] < 1e-6 {
                    0.0
                } else {
                    ((r.ops[j] as f64 - self.raw_mean[j]) / self.raw_std[j]) as f32
                };
            }
            for j in 0..21 {
                v[3 + j] = if self.raw_std[3 + j] < 1e-6 {
                    0.0
                } else {
                    ((r.sensors[j] as f64 - self.raw_mean[3 + j]) / self.raw_std[3 + j]) as f32
                };
            }
            return (0, v);
        }
        let reg = if self.k == 1 {
            0
        } else {
            let p = [
                (r.ops[0] as f64 - self.op_mean[0]) / self.op_std[0],
                (r.ops[1] as f64 - self.op_mean[1]) / self.op_std[1],
                (r.ops[2] as f64 - self.op_mean[2]) / self.op_std[2],
            ];
            let mut bc = 0usize;
            let mut bd = f64::INFINITY;
            for (ci, c) in self.centroids.iter().enumerate() {
                let d0 = p[0] - c[0];
                let d1 = p[1] - c[1];
                let d2 = p[2] - c[2];
                let d = d0 * d0 + d1 * d1 + d2 * d2;
                if d < bd { bd = d; bc = ci; }
            }
            bc
        };
        let mut v = vec![0.0_f32; 21];
        for j in 0..21 {
            let st = self.s_std[reg][j];
            v[j] = if st < 1e-6 {
                0.0
            } else {
                ((r.sensors[j] as f64 - self.s_mean[reg][j]) / st) as f32
            };
        }
        (reg, v)
    }
}

fn group_units(rows: &[RawRow], norm: &RegimeNormalizer, test_ruls: Option<&[f32]>) -> Vec<EngineSeq> {
    let max_u = rows.iter().map(|r| r.unit_id).max().unwrap_or(0);
    let mut by_u: Vec<Vec<&RawRow>> = vec![Vec::new(); max_u + 1];
    for r in rows {
        by_u[r.unit_id].push(r);
    }
    let mut seqs = Vec::new();
    for u in 1..=max_u {
        if by_u[u].is_empty() {
            continue;
        }
        by_u[u].sort_by_key(|r| r.cycle);
        let max_cyc = by_u[u].last().unwrap().cycle;
        let rul_end = match test_ruls {
            Some(ruls) => ruls[u - 1],
            None => 0.0,
        };
        let mut cycles = Vec::with_capacity(by_u[u].len());
        let mut regimes = Vec::with_capacity(by_u[u].len());
        let mut raw_t30 = Vec::with_capacity(by_u[u].len());
        let mut raw_nc = Vec::with_capacity(by_u[u].len());
        let mut feats = Vec::with_capacity(by_u[u].len());
        let mut true_rul = Vec::with_capacity(by_u[u].len());
        for r in &by_u[u] {
            let (reg, fv) = norm.transform(r);
            let rul = (rul_end + (max_cyc - r.cycle) as f32).min(RUL_MAX);
            cycles.push(r.cycle);
            regimes.push(reg);
            raw_t30.push(r.sensors[2]); // s3 = T30
            raw_nc.push(r.sensors[8]);  // s9 = Nc
            feats.push(fv);
            true_rul.push(rul);
        }
        seqs.push(EngineSeq {
            unit_id: u,
            cycles,
            regimes,
            raw_t30,
            raw_nc,
            feats,
            true_rul,
            end_rul: rul_end.min(RUL_MAX),
        });
    }
    seqs
}

fn load_dataset(dir: &Path, name: &'static str, multi_regime: bool, raw_mode: bool) -> CmapssDataset {
    let tr_rows = parse_cmapss_file(&dir.join(format!("train_{name}.txt")));
    let te_rows = parse_cmapss_file(&dir.join(format!("test_{name}.txt")));
    let te_ruls = parse_rul_file(&dir.join(format!("RUL_{name}.txt")));
    // Строго каузальная нормировка: 0 бит информации из test или RUL!
    let norm = RegimeNormalizer::fit_train_only(&tr_rows, multi_regime, raw_mode);
    let train_units = group_units(&tr_rows, &norm, None);
    let test_units = group_units(&te_rows, &norm, Some(&te_ruls));
    assert_eq!(test_units.len(), te_ruls.len(), "RUL count mismatch for {name}");
    CmapssDataset {
        name,
        in_dim: if raw_mode { 24 } else { 21 },
        train_units,
        test_units,
    }
}

/// Официальные метрики C-MAPSS (строго ВНЕ ядра!): RMSE, асимметричный NASA S-Score, Early-Warning F1/Recall (RUL <= 30).
#[derive(Clone, Copy, Default)]
struct EvalMetrics {
    rmse: f64,
    s_score: f64,
    ew_recall: f64,
    ew_f1: f64,
    finite_engines: usize,
    faults: u64,
}

fn compute_metrics(preds: &[f32], trues: &[f32], faults: u64) -> EvalMetrics {
    let total = trues.len();
    let mut finite = 0usize;
    let mut se = 0.0_f64;
    let mut s_score = 0.0_f64;
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_c = 0usize;
    for (&p_norm, &y_true) in preds.iter().zip(trues.iter()) {
        if !p_norm.is_finite() {
            if y_true <= CRIT_RUL { fn_c += 1; }
            continue;
        }
        finite += 1;
        let p_rul = (p_norm.clamp(0.0, 1.0) * RUL_MAX) as f64;
        let y = y_true as f64;
        let d = p_rul - y;
        se += d * d;
        // Официальная асимметричная NASA S-Score
        let sc = if d < 0.0 {
            (-d / 13.0).exp() - 1.0
        } else {
            (d / 10.0).exp() - 1.0
        };
        s_score += sc;
        let p_crit = p_rul <= CRIT_RUL as f64;
        let y_crit = y <= CRIT_RUL as f64;
        match (p_crit, y_crit) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_c += 1,
            (false, false) => {}
        }
    }
    let recall = if tp + fn_c > 0 { tp as f64 / (tp + fn_c) as f64 } else { 0.0 };
    let prec = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let f1 = if prec + recall > 0.0 { 2.0 * prec * recall / (prec + recall) } else { 0.0 };
    if finite < total {
        EvalMetrics {
            rmse: f64::NAN,
            s_score: f64::NAN,
            ew_recall: recall * 100.0,
            ew_f1: f1 * 100.0,
            finite_engines: finite,
            faults,
        }
    } else {
        EvalMetrics {
            rmse: (se / total.max(1) as f64).sqrt(),
            s_score,
            ew_recall: recall * 100.0,
            ew_f1: f1 * 100.0,
            finite_engines: finite,
            faults,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelKind {
    MicroCorePure2L,
    AlgSelCayley,
    Rat2DCayley,
    DeepLruClamp,
    DeepLruGuard,
    SelSsmClamp,
    SelSsmGuard,
    GruBptt4Clamp,
    GruBptt4Guard,
    MlpAdamClamp,
    MlpAdamGuard,
}

impl ModelKind {
    fn label(self) -> &'static str {
        match self {
            Self::MicroCorePure2L => "MicroCore PURE 2L+Skip(60D)",
            Self::AlgSelCayley => "Alg.Sel.Cayley (60D,0%exp)",
            Self::Rat2DCayley => "2D Rat. Cayley (60D,0%exp)",
            Self::DeepLruClamp => "Deep 2L LRU (60D,+Clamp)",
            Self::DeepLruGuard => "Deep 2L LRU (60D,+FinGuard)",
            Self::SelSsmClamp => "Selective SSM (60D,+Clamp)",
            Self::SelSsmGuard => "Selective SSM (60D,+FinGuard)",
            Self::GruBptt4Clamp => "Online GRU BPTT4(60D,+Clamp)",
            Self::GruBptt4Guard => "Online GRU BPTT4(60D,+FinGrd)",
            Self::MlpAdamClamp => "Online MLP H=32 (+Clamp)",
            Self::MlpAdamGuard => "Online MLP H=32 (+FinGuard)",
        }
    }

    fn is_stateless(self) -> bool {
        matches!(self, Self::MlpAdamClamp | Self::MlpAdamGuard)
    }
}

#[derive(Clone)]
enum FleetModel {
    MicroCore(MicroContinualCore),
    AlgCayley(AlgebraicSelectiveCayley),
    DeepLru(DeepRtrrlLruBank),
    SelSsm(SelectiveSsmBank),
    Gru(GruPerStepBank),
    Mlp(OnlineMlpAdam),
}

#[inline]
fn precond_nan_bit30(v: &mut f32) {
    if v.is_finite() && (v.abs() < 1.001 || v.abs() >= 2.0) {
        *v = if *v < 0.0 { -1.25 } else { 1.25 };
    }
}

impl FleetModel {
    fn build(kind: ModelKind, in_dim: usize, seed: u64) -> Self {
        match kind {
            ModelKind::MicroCorePure2L => {
                // Строго 60D: Band 1 = 10 секторов (30D) + Band 2 = 10 секторов (30D), B0=0!
                let mut cfg = MicroCoreConfig::numeric_stream(1, seed);
                cfg.half_sectors = 0;
                cfg.rtrl_sectors = STATE_DIM / 6;      // 10 * 3 = 30D
                cfg.rtrl_slow_sectors = STATE_DIM / 6; // 10 * 3 = 30D -> total 60D
                cfg.rtrl_max_in = in_dim;
                cfg.rtrl_use_skip_x = true;
                cfg.rtrl_grid_mode = 1;
                cfg.rtrl_kappa = 0.65;
                cfg.standardize_vec = false;
                cfg.max_regimes = 1;
                cfg.linear_lr = 0.002;
                cfg.linear_wd = 0.00005;
                cfg.rtrl_lr = 0.0008;
                cfg.trace_gamma = 0.0;
                Self::MicroCore(MicroContinualCore::new(cfg))
            }
            ModelKind::AlgSelCayley => {
                // Строго 60D: 30 комплексных осцилляторов * 2 = 60D, 0% exp
                Self::AlgCayley(
                    AlgebraicSelectiveCayley::new(in_dim, STATE_DIM / 2, 1, true, seed)
                        .with_standardize(false),
                )
            }
            ModelKind::Rat2DCayley => {
                Self::AlgCayley(
                    AlgebraicSelectiveCayley::new(in_dim, STATE_DIM / 2, 1, false, seed)
                        .with_standardize(false),
                )
            }
            ModelKind::DeepLruClamp | ModelKind::DeepLruGuard => {
                // Строго 60D: 15 fast (30D) + 15 slow (30D) = 60D, 0 dead RL RAM
                let mut m = DeepRtrrlLruBank::new_supervised(in_dim, STATE_DIM / 4, STATE_DIM / 4, 1, seed);
                m.enable_input_clamp = true;
                m.fast.enable_input_clamp = true;
                m.slow.enable_input_clamp = true;
                m.set_state_weight_guard(kind == ModelKind::DeepLruGuard);
                Self::DeepLru(m)
            }
            ModelKind::SelSsmClamp | ModelKind::SelSsmGuard => {
                // Строго 60D: 30 комплексных юнитов * 2 = 60D, 0 dead RL RAM
                let mut m = SelectiveSsmBank::new_supervised(in_dim, STATE_DIM / 2, 1, seed);
                m.enable_input_clamp = true;
                m.enable_state_weight_guard = kind == ModelKind::SelSsmGuard;
                Self::SelSsm(m)
            }
            ModelKind::GruBptt4Clamp | ModelKind::GruBptt4Guard => {
                // Строго 60D hidden state, BPTT(4), AVX2+FMA, 0 dead RL RAM
                let mut m = GruPerStepBank::new_supervised(in_dim, STATE_DIM, 1, 4, seed);
                m.gru.lr = 0.0003;
                m.gru.enable_input_clamp = true;
                m.gru.enable_state_weight_guard = kind == ModelKind::GruBptt4Guard;
                Self::Gru(m)
            }
            ModelKind::MlpAdamClamp | ModelKind::MlpAdamGuard => {
                let mut m = OnlineMlpAdam::new(in_dim, 32, 1, seed);
                m.standardize = false;
                m.lr = 0.0003;
                m.enable_input_clamp = true;
                m.set_state_weight_guard(kind == ModelKind::MlpAdamGuard);
                Self::Mlp(m)
            }
        }
    }

    fn set_guard(&mut self, on: bool) {
        match self {
            Self::MicroCore(_) | Self::AlgCayley(_) => {}
            Self::DeepLru(m) => m.set_state_weight_guard(on),
            Self::SelSsm(m) => m.enable_state_weight_guard = on,
            Self::Gru(m) => m.gru.enable_state_weight_guard = on,
            Self::Mlp(m) => m.set_state_weight_guard(on),
        }
    }

    fn reset_episode(&mut self) {
        match self {
            Self::MicroCore(m) => m.start_episode(),
            Self::AlgCayley(m) => m.reset(),
            Self::DeepLru(m) => m.start_episode(),
            Self::SelSsm(m) => m.start_episode(),
            Self::Gru(m) => m.start_episode(),
            Self::Mlp(m) => { m.had_nan = false; }
        }
    }

    #[inline]
    fn step(&mut self, x: &[f32], target: Option<f32>) -> f32 {
        match self {
            Self::MicroCore(m) => {
                let p = *m.decide_reg_vec(x, target.is_some(), false, false).get();
                if let Some(y) = target {
                    let _ = m.end_reg_step(y);
                }
                p
            }
            Self::AlgCayley(m) => m.step_and_learn_regression(x, target, 0.0006, 0.0004),
            Self::DeepLru(m) => m.step_and_learn_regression(x, target, 0.002, 0.0008),
            Self::SelSsm(m) => m.step_and_learn_regression(x, target, 0.0006, 0.0004),
            Self::Gru(m) => m.step_and_learn_regression(x, target),
            Self::Mlp(m) => m.step_and_learn_regression(x, target),
        }
    }

    fn memory_kb(&self) -> f64 {
        let bytes = match self {
            Self::MicroCore(m) => m.memory_footprint_bytes(),
            Self::AlgCayley(m) => m.memory_footprint_bytes(),
            Self::DeepLru(m) => m.memory_footprint_bytes(),
            Self::SelSsm(m) => m.memory_footprint_bytes(),
            Self::Gru(m) => m.memory_footprint_bytes(),
            Self::Mlp(m) => m.memory_footprint_bytes(),
        };
        bytes as f64 / 1024.0
    }

    fn total_seu_faults(&self) -> u64 {
        match self {
            Self::MicroCore(m) => m.total_seu_faults(),
            Self::AlgCayley(m) => m.total_seu_faults(),
            Self::DeepLru(m) => m.total_seu_faults(),
            Self::SelSsm(m) => m.seu_fault_count,
            Self::Gru(m) => m.gru.seu_fault_count,
            Self::Mlp(m) => m.total_seu_faults(),
        }
    }

    fn reset_seu_counters(&mut self) {
        match self {
            Self::MicroCore(m) => {
                m.seu_fault_count = 0;
                m.rtrl_state.seu_fault_count = 0;
                m.rtrl_slow_state.seu_fault_count = 0;
            }
            Self::AlgCayley(m) => m.seu_fault_count = 0,
            Self::DeepLru(m) => {
                m.seu_fault_count = 0;
                m.fast.seu_fault_count = 0;
                m.slow.seu_fault_count = 0;
            }
            Self::SelSsm(m) => m.seu_fault_count = 0,
            Self::Gru(m) => m.gru.seu_fault_count = 0,
            Self::Mlp(m) => m.seu_fault_count = 0,
        }
    }

    /// Строго симметричная State-SEU инъекция: ±1.25 Bit-30 XOR (гарантированный NaN) в 1 живом регистре из 60.
    fn inject_state_seu_nan(&mut self, idx: usize) {
        match self {
            Self::MicroCore(m) => {
                let nf = m.rtrl_state.u.len();
                let ns = m.rtrl_slow_state.u.len();
                let tot = (nf + ns).max(1);
                let r = idx % tot;
                if r < nf {
                    precond_nan_bit30(&mut m.rtrl_state.u[r]);
                } else if r - nf < ns {
                    precond_nan_bit30(&mut m.rtrl_slow_state.u[r - nf]);
                }
                m.inject_register_bit_flip(r, 30);
            }
            Self::AlgCayley(m) => {
                let u = m.units.max(1);
                let r = idx % (2 * u);
                if r < u {
                    precond_nan_bit30(&mut m.h_re[r]);
                } else {
                    precond_nan_bit30(&mut m.h_im[r - u]);
                }
                m.inject_register_bit_flip(r, 30);
            }
            Self::DeepLru(m) => {
                let nf = 2 * m.fast.units;
                let ns = 2 * m.slow.units;
                let tot = (nf + ns).max(1);
                let r = idx % tot;
                if r < nf {
                    let u = r / 2;
                    if (r & 1) == 0 { precond_nan_bit30(&mut m.fast.h_re[u]); }
                    else { precond_nan_bit30(&mut m.fast.h_im[u]); }
                } else {
                    let sr = r - nf;
                    let u = sr / 2;
                    if (sr & 1) == 0 { precond_nan_bit30(&mut m.slow.h_re[u]); }
                    else { precond_nan_bit30(&mut m.slow.h_im[u]); }
                }
                m.inject_register_bit_flip(r, 30);
            }
            Self::SelSsm(m) => {
                let r = idx % (2 * m.units).max(1);
                let u = r / 2;
                if (r & 1) == 0 { precond_nan_bit30(&mut m.h_re[u]); }
                else { precond_nan_bit30(&mut m.h_im[u]); }
                m.inject_register_bit_flip(r, 30);
            }
            Self::Gru(m) => {
                let r = idx % m.gru.h_cur.len().max(1);
                precond_nan_bit30(&mut m.gru.h_cur[r]);
                m.inject_register_bit_flip(r, 30);
            }
            Self::Mlp(_) => {
                // Stateless feedforward MLP has 0 recurrent state registers
            }
        }
    }

    /// Строго симметричная физическая SRAM Weight-SEU инъекция (Knuth-hash, ±1.25 Bit-30 XOR).
    fn inject_weight_seu_nan(&mut self, idx: usize) {
        match self {
            Self::MicroCore(m) => m.inject_weight_bit_flip(idx, 30),
            Self::AlgCayley(m) => m.inject_weight_bit_flip(idx, 30),
            Self::DeepLru(m) => m.inject_weight_bit_flip(idx, 30),
            Self::SelSsm(m) => m.inject_weight_bit_flip(idx, 30),
            Self::Gru(m) => m.inject_weight_bit_flip(idx, 30),
            Self::Mlp(m) => m.inject_weight_bit_flip(idx, 30),
        }
    }
}

fn shuffle_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = splitmix64(seed);
    for i in (1..n).rev() {
        rng = splitmix64(rng);
        let j = (rng as usize) % (i + 1);
        idx.swap(i, j);
    }
    idx
}

fn train_model_passes(
    model: &mut FleetModel,
    ds: &CmapssDataset,
    passes: usize,
    seed: u64,
    shuffled_targets: Option<&[f32]>,
) -> f64 {
    let t0 = Instant::now();
    let mut total_cycles = 0usize;
    for pass in 0..passes {
        let order = shuffle_indices(ds.train_units.len(), seed.wrapping_mul(131) ^ (pass as u64 + 17));
        let mut flat_pos = 0usize;
        for &u_idx in &order {
            let unit = &ds.train_units[u_idx];
            model.reset_episode();
            for (ci, feat) in unit.feats.iter().enumerate() {
                let y_norm = match shuffled_targets {
                    Some(st) => {
                        let y = st[flat_pos % st.len()];
                        flat_pos += 1;
                        y
                    }
                    None => unit.true_rul[ci] / RUL_MAX,
                };
                let _ = model.step(feat, Some(y_norm));
                total_cycles += 1;
            }
        }
    }
    t0.elapsed().as_secs_f64() * 1e6 / total_cycles.max(1) as f64
}

fn eval_model_clean(model: &mut FleetModel, ds: &CmapssDataset) -> EvalMetrics {
    let mut preds = Vec::with_capacity(ds.test_units.len());
    let mut trues = Vec::with_capacity(ds.test_units.len());
    for unit in &ds.test_units {
        model.reset_episode();
        let mut last_p = 0.0_f32;
        for feat in &unit.feats {
            last_p = model.step(feat, None);
        }
        preds.push(last_p);
        trues.push(unit.end_rul);
    }
    compute_metrics(&preds, &trues, model.total_seu_faults())
}

#[derive(Clone, Copy)]
enum FlightSeuScenario {
    Clean,
    SensorSeu, // T30 (index 2) = NaN / +1e30 for 2 mid-flight cycles
    StateSeu,  // Bit-30 XOR in 1 live register every 25 cycles starting at cycle 10
    WeightSeu, // Bit-30 XOR in live SRAM weight every 25 cycles starting at cycle 10
}

fn eval_flight_seu(
    trained_base: &FleetModel,
    ds: &CmapssDataset,
    scenario: FlightSeuScenario,
    is_stateless: bool,
) -> EvalMetrics {
    let mut preds = Vec::with_capacity(ds.test_units.len());
    let mut trues = Vec::with_capacity(ds.test_units.len());
    let mut total_faults = 0u64;
    let mut x_mut = vec![0.0_f32; ds.in_dim];

    for (u_idx, unit) in ds.test_units.iter().enumerate() {
        let mut m = trained_base.clone();
        m.reset_seu_counters();
        m.reset_episode();
        let mid = unit.feats.len() / 2;
        let mut last_p = 0.0_f32;
        for (ci, feat) in unit.feats.iter().enumerate() {
            x_mut.copy_from_slice(feat);
            match scenario {
                FlightSeuScenario::Clean => {}
                FlightSeuScenario::SensorSeu => {
                    // s3 = T30 -> index 2 in 21-sensor space
                    if ci == mid {
                        x_mut[2] = f32::NAN;
                    } else if ci == mid + 1 {
                        x_mut[2] = 1e30_f32;
                    }
                }
                FlightSeuScenario::StateSeu => {
                    if !is_stateless && ci >= 9 && (ci - 9) % 25 == 0 {
                        let kidx = u_idx.wrapping_mul(31).wrapping_add(ci / 25);
                        m.inject_state_seu_nan(kidx);
                    }
                }
                FlightSeuScenario::WeightSeu => {
                    if ci >= 9 && (ci - 9) % 25 == 0 {
                        let kidx = u_idx.wrapping_mul(37).wrapping_add(ci / 25);
                        m.inject_weight_seu_nan(kidx);
                    }
                }
            }
            last_p = m.step(&x_mut, None);
        }
        total_faults = total_faults.saturating_add(m.total_seu_faults());
        preds.push(last_p);
        trues.push(unit.end_rul);
    }
    compute_metrics(&preds, &trues, total_faults)
}

fn mean_sd(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() { return (0.0, 0.0); }
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64;
    (m, v.sqrt())
}

fn fmt_metric(v: f64, width: usize, prec: usize) -> String {
    if v.is_finite() {
        format!("{v:>width$.prec$}")
    } else {
        format!("{:>width$}", "NaN")
    }
}

fn run_block1(datasets: &[CmapssDataset], fd4_raw: &CmapssDataset) -> Vec<(String, String, f64, f64, f64, f64)> {
    println!("=================================================================================================================");
    println!("BLOCK 1: NASA C-MAPSS STREAMING RUL (FD001..FD004, STRICT 60D PARITY, CAUSAL TRAIN-ONLY NORM, 5 SEEDS)");
    println!("=================================================================================================================");

    let mut summary_3pass: Vec<(String, String, f64, f64, f64, f64)> = Vec::new();
    let models = [
        ModelKind::MicroCorePure2L,
        ModelKind::AlgSelCayley,
        ModelKind::Rat2DCayley,
        ModelKind::DeepLruClamp,
        ModelKind::SelSsmClamp,
        ModelKind::GruBptt4Clamp,
        ModelKind::MlpAdamClamp,
    ];
    let seeds: [u64; 5] = [1, 2, 3, 4, 5];

    for &passes in &[1usize, 3usize] {
        println!("\n--- Mode: {passes}-pass Streaming Training (Frozen Test Eval on Final Cycle) ---");
        for ds in datasets {
            println!(
                "\n[{} | {} train / {} test units | in_dim={}]",
                ds.name, ds.train_units.len(), ds.test_units.len(), ds.in_dim
            );
            println!(
                "{:<29} | {:>13} | {:>11} | {:>7} | {:>7} | {:>8} | {:>8}",
                "Model (State=60D)", "RMSE (cyc)", "NASA S-Scr", "EW-Rec", "EW-F1", "RAM(KB)", "Lat(us)"
            );
            println!("{}", "-".repeat(102));
            for &mk in &models {
                let mut rmses = Vec::new();
                let mut scores = Vec::new();
                let mut recs = Vec::new();
                let mut f1s = Vec::new();
                let mut lats = Vec::new();
                let mut ram_kb = 0.0_f64;
                for &s in &seeds {
                    let mut m = FleetModel::build(mk, ds.in_dim, s);
                    ram_kb = m.memory_kb();
                    let lat = train_model_passes(&mut m, ds, passes, s, None);
                    let ev = eval_model_clean(&mut m, ds);
                    rmses.push(ev.rmse);
                    scores.push(ev.s_score);
                    recs.push(ev.ew_recall);
                    f1s.push(ev.ew_f1);
                    lats.push(lat);
                }
                let (rm, rs) = mean_sd(&rmses);
                let (sm, _) = mean_sd(&scores);
                let (rcm, _) = mean_sd(&recs);
                let (f1m, _) = mean_sd(&f1s);
                let (lm, _) = mean_sd(&lats);
                if passes == 3 {
                    summary_3pass.push((ds.name.to_string(), mk.label().to_string(), rm, sm, rcm, ram_kb));
                }
                println!(
                    "{:<29} | {:6.2} ± {:4.2} | {:11.1} | {:6.1}% | {:6.1}% | {:8.1} | {:8.2}",
                    mk.label(), rm, rs, sm, rcm, f1m, ram_kb, lm
                );
            }
        }
    }

    // Raw FD004 Ablation (24 raw globally standardized inputs, NO regime clustering!)
    println!("\n--- Ablation: FD004 RAW STREAM (24 inputs: op1..3 + s1..21, NO regime clustering, 3-pass, 5 seeds) ---");
    println!(
        "{:<29} | {:>13} | {:>11} | {:>7} | {:>7} | {:>8} | {:>8}",
        "Model (State=60D)", "RMSE (cyc)", "NASA S-Scr", "EW-Rec", "EW-F1", "RAM(KB)", "Lat(us)"
    );
    println!("{}", "-".repeat(102));
    for &mk in &models {
        let mut rmses = Vec::new();
        let mut scores = Vec::new();
        let mut recs = Vec::new();
        let mut f1s = Vec::new();
        let mut lats = Vec::new();
        let mut ram_kb = 0.0_f64;
        for &s in &seeds {
            let mut m = FleetModel::build(mk, fd4_raw.in_dim, s);
            ram_kb = m.memory_kb();
            let lat = train_model_passes(&mut m, fd4_raw, 3, s, None);
            let ev = eval_model_clean(&mut m, fd4_raw);
            rmses.push(ev.rmse);
            scores.push(ev.s_score);
            recs.push(ev.ew_recall);
            f1s.push(ev.ew_f1);
            lats.push(lat);
        }
        let (rm, rs) = mean_sd(&rmses);
        let (sm, _) = mean_sd(&scores);
        let (rcm, _) = mean_sd(&recs);
        let (f1m, _) = mean_sd(&f1s);
        let (lm, _) = mean_sd(&lats);
        println!(
            "{:<29} | {:6.2} ± {:4.2} | {:11.1} | {:6.1}% | {:6.1}% | {:8.1} | {:8.2}",
            mk.label(), rm, rs, sm, rcm, f1m, ram_kb, lm
        );
    }

    // Mandatory RUL-Shuffle (Label-Shuffle) Zero-Leakage Sanity Check on FD004
    let fd4 = &datasets[3];
    let mut all_y: Vec<f32> = Vec::new();
    for u in &fd4.train_units {
        for &r in &u.true_rul {
            all_y.push(r / RUL_MAX);
        }
    }
    let test_mean: f64 = fd4.test_units.iter().map(|u| u.end_rul as f64).sum::<f64>() / fd4.test_units.len() as f64;
    let test_std: f64 = (fd4.test_units.iter().map(|u| { let d = u.end_rul as f64 - test_mean; d * d }).sum::<f64>() / fd4.test_units.len() as f64).sqrt();
    let train_mean_rul: f64 = all_y.iter().map(|&y| (y * RUL_MAX) as f64).sum::<f64>() / all_y.len() as f64;
    let const_rmse = (fd4.test_units.iter().map(|u| { let d = train_mean_rul - u.end_rul as f64; d * d }).sum::<f64>() / fd4.test_units.len() as f64).sqrt();

    println!(
        "\n--- Anti-Leakage Control: RUL-SHUFFLE on FD004 (3-pass, 3 seeds; test std={test_std:.2}, const-train-mean RMSE={const_rmse:.2}) ---"
    );
    println!("{:<29} | {:>13} | {:>11} | {:>7}", "Model (RUL-Shuffled)", "RMSE (cyc)", "NASA S-Scr", "Status");
    println!("{}", "-".repeat(72));
    for &mk in &models {
        let mut rmses = Vec::new();
        let mut scores = Vec::new();
        for s in 1u64..=3 {
            let perm = shuffle_indices(all_y.len(), s.wrapping_mul(0xDEAD_BEEF));
            let shuf: Vec<f32> = perm.iter().map(|&i| all_y[i]).collect();
            let mut m = FleetModel::build(mk, fd4.in_dim, s);
            let _ = train_model_passes(&mut m, fd4, 3, s, Some(&shuf));
            let ev = eval_model_clean(&mut m, fd4);
            rmses.push(ev.rmse);
            scores.push(ev.s_score);
        }
        let (rm, rs) = mean_sd(&rmses);
        let (sm, _) = mean_sd(&scores);
        let ok = if rm >= test_std * 0.90 { "PASS (no leak)" } else { "FAIL (LEAK!)" };
        println!("{:<29} | {:6.2} ± {:4.2} | {:11.1} | {}", mk.label(), rm, rs, sm, ok);
    }
    summary_3pass
}

/// Возвращает обученные модели (seed=1, 3-pass на FD004) для Блока 2, Блока 3 и экспорта CSV.
fn run_block2_and_export(fd4: &CmapssDataset) -> Vec<(ModelKind, FleetModel)> {
    println!("\n=============================================================================================================================");
    println!("BLOCK 2: HIGH-ALTITUDE RADIATION IN FLIGHT (FLIGHT SEU on test_FD004, 248 ENGINES, 3-PASS TRAINED, 3 SEEDS)");
    println!("Protocol: Sensor-SEU (T30=NaN/+1e30 2 cyc mid-flight) | State-SEU (Bit-30 / 25 cyc) | Weight-SEU (Bit-30 SRAM / 25 cyc)");
    println!("=============================================================================================================================");

    let base_kinds = [
        ModelKind::MicroCorePure2L,
        ModelKind::AlgSelCayley,
        ModelKind::DeepLruClamp,
        ModelKind::SelSsmClamp,
        ModelKind::GruBptt4Clamp,
        ModelKind::MlpAdamClamp,
    ];

    // Train once per (base_kind, seed) -> clone for (+Clamp) and (+FinGuard) so Clean weights are IDENTICAL!
    let seeds = [1u64, 2, 3];
    let mut seed1_models: Vec<(ModelKind, FleetModel)> = Vec::new();

    println!(
        "{:<29} | {:>15} | {:>15} | {:>15} | {:>15} | {:>11} | {:>13}",
        "Model",
        "Clean RMSE/Rec",
        "Sens  RMSE/Rec",
        "State RMSE/Rec",
        "Wght  RMSE/Rec",
        "Faults(S/W)",
        "Finite(S/W)"
    );
    println!("{}", "-".repeat(130));

    for &bk in &base_kinds {
        let mut trained_per_seed: Vec<FleetModel> = Vec::new();
        for &s in &seeds {
            let mut m = FleetModel::build(bk, fd4.in_dim, s);
            let _ = train_model_passes(&mut m, fd4, 3, s, None);
            if s == 1 {
                seed1_models.push((bk, m.clone()));
            }
            trained_per_seed.push(m);
        }

        let variants: Vec<(ModelKind, bool)> = match bk {
            ModelKind::MicroCorePure2L => vec![(ModelKind::MicroCorePure2L, true)],
            ModelKind::AlgSelCayley => vec![(ModelKind::AlgSelCayley, true)],
            ModelKind::DeepLruClamp => vec![(ModelKind::DeepLruClamp, false), (ModelKind::DeepLruGuard, true)],
            ModelKind::SelSsmClamp => vec![(ModelKind::SelSsmClamp, false), (ModelKind::SelSsmGuard, true)],
            ModelKind::GruBptt4Clamp => vec![(ModelKind::GruBptt4Clamp, false), (ModelKind::GruBptt4Guard, true)],
            ModelKind::MlpAdamClamp => vec![(ModelKind::MlpAdamClamp, false), (ModelKind::MlpAdamGuard, true)],
            _ => vec![],
        };

        for (vk, guard_on) in variants {
            let mut cl_r = Vec::new(); let mut cl_rc = Vec::new();
            let mut se_r = Vec::new(); let mut se_rc = Vec::new();
            let mut st_r = Vec::new(); let mut st_rc = Vec::new();
            let mut wt_r = Vec::new(); let mut wt_rc = Vec::new();
            let mut st_f = 0u64; let mut wt_f = 0u64;
            let mut st_fin = 0usize; let mut wt_fin = 0usize;

            for tm in &trained_per_seed {
                let mut m = tm.clone();
                m.set_guard(guard_on);
                let e_cl = eval_flight_seu(&m, fd4, FlightSeuScenario::Clean, vk.is_stateless());
                let e_se = eval_flight_seu(&m, fd4, FlightSeuScenario::SensorSeu, vk.is_stateless());
                let e_st = eval_flight_seu(&m, fd4, FlightSeuScenario::StateSeu, vk.is_stateless());
                let e_wt = eval_flight_seu(&m, fd4, FlightSeuScenario::WeightSeu, vk.is_stateless());
                cl_r.push(e_cl.rmse); cl_rc.push(e_cl.ew_recall);
                se_r.push(e_se.rmse); se_rc.push(e_se.ew_recall);
                st_r.push(e_st.rmse); st_rc.push(e_st.ew_recall);
                wt_r.push(e_wt.rmse); wt_rc.push(e_wt.ew_recall);
                st_f += e_st.faults;
                wt_f += e_wt.faults;
                st_fin += e_st.finite_engines;
                wt_fin += e_wt.finite_engines;
            }
            let n_s = seeds.len() as f64;
            let m_or_nan = |v: &[f64]| -> f64 {
                if v.iter().all(|x| x.is_finite()) { mean_sd(v).0 } else { f64::NAN }
            };
            let st_str = if vk.is_stateless() {
                format!("{:>15}", "N/A (stateless)")
            } else {
                format!("{}/{:4.0}%", fmt_metric(m_or_nan(&st_r), 6, 2), mean_sd(&st_rc).0)
            };
            let fin_str = if vk.is_stateless() {
                format!("N/A /{:>3}/248", (wt_fin as f64 / n_s).round() as usize)
            } else {
                format!("{:>3}/248,{:>3}/248", (st_fin as f64 / n_s).round() as usize, (wt_fin as f64 / n_s).round() as usize)
            };
            println!(
                "{:<29} | {}/{:4.0}% | {}/{:4.0}% | {} | {}/{:4.0}% | {:5.0}/{:5.0} | {}",
                vk.label(),
                fmt_metric(m_or_nan(&cl_r), 6, 2), mean_sd(&cl_rc).0,
                fmt_metric(m_or_nan(&se_r), 6, 2), mean_sd(&se_rc).0,
                st_str,
                fmt_metric(m_or_nan(&wt_r), 6, 2), mean_sd(&wt_rc).0,
                st_f as f64 / n_s, wt_f as f64 / n_s,
                fin_str
            );
        }
    }

    // Export CSV telemetry to ./logs/
    fs::create_dir_all("logs").expect("create logs dir");
    let m_micro = seed1_models.iter().find(|(k, _)| *k == ModelKind::MicroCorePure2L).unwrap().1.clone();
    let mut m_lru_clamp = seed1_models.iter().find(|(k, _)| *k == ModelKind::DeepLruClamp).unwrap().1.clone();
    m_lru_clamp.set_guard(false);
    let mut m_lru_guard = m_lru_clamp.clone();
    m_lru_guard.set_guard(true);
    let mut m_gru_clamp = seed1_models.iter().find(|(k, _)| *k == ModelKind::GruBptt4Clamp).unwrap().1.clone();
    m_gru_clamp.set_guard(false);

    // 1. logs/nasa_engine_trace.csv: first test_FD004 engine with len>=200 and end_rul<=25 (unit_id = 1)
    let trace_unit = fd4
        .test_units
        .iter()
        .find(|u| u.feats.len() >= 200 && u.end_rul <= 25.0)
        .unwrap_or(&fd4.test_units[0]);
    let strike_ci = trace_unit.feats.len() / 2;
    {
        let mut f = BufWriter::new(File::create("logs/nasa_engine_trace.csv").expect("create trace csv"));
        writeln!(
            f,
            "cycle,op_regime,s_raw_t30,s_raw_nc,true_rul,pred_micro,pred_lru_clamp,pred_lru_guard,pred_gru_clamp,slow_envelope_norm,seu_event,faults_repaired"
        ).unwrap();
        let mut mm = m_micro.clone(); mm.reset_seu_counters(); mm.reset_episode();
        let mut mlc = m_lru_clamp.clone(); mlc.reset_seu_counters(); mlc.reset_episode();
        let mut mlg = m_lru_guard.clone(); mlg.reset_seu_counters(); mlg.reset_episode();
        let mut mgc = m_gru_clamp.clone(); mgc.reset_seu_counters(); mgc.reset_episode();
        let mut x_mut = vec![0.0_f32; fd4.in_dim];

        for (ci, feat) in trace_unit.feats.iter().enumerate() {
            x_mut.copy_from_slice(feat);
            let mut seu_ev = 0u8;
            if ci == strike_ci {
                // High-altitude strike: Bit-30 state flip + T30 sensor loss (NaN)
                mm.inject_state_seu_nan(7);
                mlc.inject_state_seu_nan(7);
                mlg.inject_state_seu_nan(7);
                mgc.inject_state_seu_nan(7);
                x_mut[2] = f32::NAN;
                seu_ev = 1;
            } else if ci == strike_ci + 1 {
                x_mut[2] = 1e30_f32;
                seu_ev = 2;
            }
            let pm = mm.step(&x_mut, None);
            let plc = mlc.step(&x_mut, None);
            let plg = mlg.step(&x_mut, None);
            let pgc = mgc.step(&x_mut, None);
            let env_norm = match &mm {
                FleetModel::MicroCore(c) => c.slow_envelope_norm(),
                _ => 0.0,
            };
            let to_rul = |p: f32| -> String {
                if p.is_finite() { format!("{:.3}", p.clamp(0.0, 1.0) * RUL_MAX) } else { "NaN".to_string() }
            };
            writeln!(
                f,
                "{},{},{:.3},{:.3},{:.3},{},{},{},{},{:.5},{},{}",
                trace_unit.cycles[ci],
                trace_unit.regimes[ci],
                trace_unit.raw_t30[ci],
                trace_unit.raw_nc[ci],
                trace_unit.true_rul[ci],
                to_rul(pm),
                to_rul(plc),
                to_rul(plg),
                to_rul(pgc),
                env_norm,
                seu_ev,
                mm.total_seu_faults()
            ).unwrap();
        }
        println!("\n[CSV] Exported logs/nasa_engine_trace.csv (FD004 Unit #{}, {} cycles, end_RUL={:.1}, SEU strike at cycle {})",
            trace_unit.unit_id, trace_unit.feats.len(), trace_unit.end_rul, trace_unit.cycles[strike_ci]);
    }

    // 2. logs/nasa_fleet_frames.csv: first 100 engines of test_FD004 with len >= 120, synchronized last 120 frames, SEU strike at frame 60
    let fleet_units: Vec<&EngineSeq> = fd4.test_units.iter().filter(|u| u.feats.len() >= 120).take(100).collect();
    let n_frames = 120usize;
    let strike_frame = 60usize;
    {
        let mut f = BufWriter::new(File::create("logs/nasa_fleet_frames.csv").expect("create fleet csv"));
        writeln!(
            f,
            "frame,engine_idx,unit_id,cycle,true_rul,pred_micro,pred_gru_clamp,pred_lru_clamp,pred_lru_guard,seu_strike,micro_faults,gru_clamp_finite,lru_guard_finite"
        ).unwrap();

        let mut micros: Vec<FleetModel> = (0..fleet_units.len()).map(|_| { let mut m = m_micro.clone(); m.reset_seu_counters(); m.reset_episode(); m }).collect();
        let mut grus: Vec<FleetModel> = (0..fleet_units.len()).map(|_| { let mut m = m_gru_clamp.clone(); m.reset_seu_counters(); m.reset_episode(); m }).collect();
        let mut lru_cs: Vec<FleetModel> = (0..fleet_units.len()).map(|_| { let mut m = m_lru_clamp.clone(); m.reset_seu_counters(); m.reset_episode(); m }).collect();
        let mut lru_gs: Vec<FleetModel> = (0..fleet_units.len()).map(|_| { let mut m = m_lru_guard.clone(); m.reset_seu_counters(); m.reset_episode(); m }).collect();

        // Warm up each engine from cycle 1 up to (len - n_frames) so full causal history is exact!
        for (ei, unit) in fleet_units.iter().enumerate() {
            let warm = unit.feats.len() - n_frames;
            for ci in 0..warm {
                let x = &unit.feats[ci];
                let _ = micros[ei].step(x, None);
                let _ = grus[ei].step(x, None);
                let _ = lru_cs[ei].step(x, None);
                let _ = lru_gs[ei].step(x, None);
            }
        }

        let mut x_mut = vec![0.0_f32; fd4.in_dim];
        for fr in 0..n_frames {
            for (ei, unit) in fleet_units.iter().enumerate() {
                let ci = (unit.feats.len() - n_frames) + fr;
                x_mut.copy_from_slice(&unit.feats[ci]);
                let mut strike = 0u8;
                if fr == strike_frame {
                    micros[ei].inject_state_seu_nan(ei * 7 + 3);
                    grus[ei].inject_state_seu_nan(ei * 7 + 3);
                    lru_cs[ei].inject_state_seu_nan(ei * 7 + 3);
                    lru_gs[ei].inject_state_seu_nan(ei * 7 + 3);
                    x_mut[2] = f32::NAN;
                    strike = 1;
                } else if fr == strike_frame + 1 {
                    x_mut[2] = 1e30_f32;
                    strike = 2;
                }
                let pm = micros[ei].step(&x_mut, None);
                let pg = grus[ei].step(&x_mut, None);
                let plc = lru_cs[ei].step(&x_mut, None);
                let plg = lru_gs[ei].step(&x_mut, None);
                let fmt_r = |p: f32| -> String {
                    if p.is_finite() { format!("{:.2}", p.clamp(0.0, 1.0) * RUL_MAX) } else { "NaN".to_string() }
                };
                writeln!(
                    f,
                    "{},{},{},{},{:.2},{},{},{},{},{},{},{},{}",
                    fr,
                    ei,
                    unit.unit_id,
                    unit.cycles[ci],
                    unit.true_rul[ci],
                    fmt_r(pm),
                    fmt_r(pg),
                    fmt_r(plc),
                    fmt_r(plg),
                    strike,
                    micros[ei].total_seu_faults(),
                    u8::from(pg.is_finite()),
                    u8::from(plg.is_finite())
                ).unwrap();
            }
        }
        println!("[CSV] Exported logs/nasa_fleet_frames.csv ({} engines x {} frames, SEU strike at frame {})",
            fleet_units.len(), n_frames, strike_frame);
    }

    seed1_models
}

fn run_block3_swarm(
    datasets: &[CmapssDataset],
    seed1_models: &[(ModelKind, FleetModel)],
) -> Vec<(usize, String, f64, f64, f64)> {
    println!("\n=================================================================================================================");
    println!("BLOCK 3: FLEET DIGITAL TWIN SWARM (N = 248 -> 1,024 -> 4,096 ENGINES, ONLINE ADAPTATION, 16 THREADS, 0-ALLOC)");
    println!("=================================================================================================================");

    let mut swarm_rows: Vec<(usize, String, f64, f64, f64)> = Vec::new();

    // Pool all real test trajectories across FD001..FD004 (707 real aircraft engine flights, all 21-dim)
    let mut pool: Vec<&EngineSeq> = Vec::new();
    for ds in datasets {
        for u in &ds.test_units {
            pool.push(u);
        }
    }

    let swarm_kinds = [
        ModelKind::MicroCorePure2L,
        ModelKind::AlgSelCayley,
        ModelKind::DeepLruClamp,
        ModelKind::SelSsmClamp,
        ModelKind::GruBptt4Clamp,
    ];
    let n_threads = 16usize;
    let cycles_per_engine = 120usize;

    for &n_engines in &[248usize, 1024usize, 4096usize] {
        println!("\n--- Fleet Scale: N = {n_engines} Personal Digital Twins ({cycles_per_engine} online-adapting cycles/engine, {n_threads} threads) ---");
        println!(
            "{:<29} | {:>10} | {:>11} | {:>14} | {:>16}",
            "Model (State=60D)", "Fleet RAM", "1T Lat(us)", "16T Thr(M/s)", "Speedup vs GRU"
        );
        println!("{}", "-".repeat(92));

        let mut results: Vec<(ModelKind, f64, f64, f64)> = Vec::new();
        for &mk in &swarm_kinds {
            let base = &seed1_models.iter().find(|(k, _)| *k == mk).unwrap().1;
            let fleet_ram_mb = (base.memory_kb() * n_engines as f64) / 1024.0;

            // 1-thread latency measure on first 64 engines x 120 cycles (online adaptation)
            let n1 = n_engines.min(64);
            let mut single_twins: Vec<FleetModel> = (0..n1).map(|_| base.clone()).collect();
            let t1 = Instant::now();
            let mut sink1 = 0.0_f64;
            for (ei, m) in single_twins.iter_mut().enumerate() {
                let u = pool[ei % pool.len()];
                let phase = (ei * 13) % u.feats.len();
                for step in 0..cycles_per_engine {
                    let ci = (phase + step) % u.feats.len();
                    let p = m.step(&u.feats[ci], Some(u.true_rul[ci] / RUL_MAX));
                    sink1 += p as f64;
                }
            }
            let lat_1t_us = t1.elapsed().as_secs_f64() * 1e6 / (n1 * cycles_per_engine) as f64;
            std::hint::black_box(sink1);

            // 16-thread full fleet swarm throughput
            let mut twins: Vec<FleetModel> = (0..n_engines).map(|_| base.clone()).collect();
            let chunk = (n_engines + n_threads - 1) / n_threads;
            let t16 = Instant::now();
            let mut total_sink = 0.0_f64;
            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(n_threads);
                for (tid, slice) in twins.chunks_mut(chunk).enumerate() {
                    let pool_ref = &pool;
                    handles.push(s.spawn(move || {
                        let mut acc = 0.0_f64;
                        let base_ei = tid * chunk;
                        for (local_i, m) in slice.iter_mut().enumerate() {
                            let ei = base_ei + local_i;
                            let u = pool_ref[ei % pool_ref.len()];
                            let phase = (ei * 17) % u.feats.len();
                            for step in 0..cycles_per_engine {
                                let ci = (phase + step) % u.feats.len();
                                let p = m.step(&u.feats[ci], Some(u.true_rul[ci] / RUL_MAX));
                                acc += p as f64;
                            }
                        }
                        acc
                    }));
                }
                for h in handles {
                    total_sink += h.join().unwrap();
                }
            });
            let elapsed = t16.elapsed().as_secs_f64();
            std::hint::black_box(total_sink);
            let m_cyc_per_s = (n_engines * cycles_per_engine) as f64 / (elapsed * 1e6);
            results.push((mk, fleet_ram_mb, lat_1t_us, m_cyc_per_s));
        }

        let gru_thr = results.iter().find(|r| r.0 == ModelKind::GruBptt4Clamp).map(|r| r.3).unwrap_or(1.0);
        for (mk, ram_mb, lat_us, thr) in &results {
            swarm_rows.push((n_engines, mk.label().to_string(), *ram_mb, *lat_us, *thr));
            println!(
                "{:<29} | {:7.2} MB | {:8.2} us | {:9.2} M/s | {:13.2}x",
                mk.label(), ram_mb, lat_us, thr, thr / gru_thr.max(1e-6)
            );
        }
    }
    swarm_rows
}

fn main() {
    let dir = Path::new("CMAPSSData");
    assert!(dir.exists(), "CMAPSSData/ directory not found!");

    let datasets = vec![
        load_dataset(dir, "FD001", false, false),
        load_dataset(dir, "FD002", true, false),
        load_dataset(dir, "FD003", false, false),
        load_dataset(dir, "FD004", true, false),
    ];
    let fd4_raw = load_dataset(dir, "FD004", false, true);

    let b1 = run_block1(&datasets, &fd4_raw);
    let seed1_models = run_block2_and_export(&datasets[3]);
    let b3 = run_block3_swarm(&datasets, &seed1_models);

    // Export machine-readable summary for plot_nasa_fleet.py (zero hand-copied numbers!)
    fs::create_dir_all("logs").expect("create logs dir");
    let mut jf = BufWriter::new(File::create("logs/bench_step7_summary.csv").expect("create summary csv"));
    writeln!(jf, "section,key1,key2,val1,val2,val3,val4").unwrap();
    for (fd, model, rmse, sscore, rec, ram_kb) in b1 {
        writeln!(jf, "block1_3pass,{fd},\"{model}\",{rmse:.4},{sscore:.2},{rec:.2},{ram_kb:.2}").unwrap();
    }
    for (n_eng, model, ram_mb, lat_us, thr) in b3 {
        writeln!(jf, "block3_swarm,{n_eng},\"{model}\",{ram_mb:.4},{lat_us:.4},{thr:.4},0").unwrap();
    }
    println!("\n[CSV] Exported logs/bench_step7_summary.csv for plotting.");
}
