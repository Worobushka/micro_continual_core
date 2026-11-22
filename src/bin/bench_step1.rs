// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Бенчмарк Шага 1 (v0.4.0):
//! Прогоняет 3 строгих честных среза на 10 сидах (legacy-костыли и утечка словаря удалены):
//! 1. STRICT HONEST — Dense (скрытая subsystem + неразличимый глитч 15% + чистый шумный Cloze без holdout)
//! 2. STRICT HONEST — Sparse Delayed (награда только в конце, glitch=15%)
//! 3. STRICT HONEST — Sparse Delayed (награда только в конце, glitch=25%)

use std::time::Instant;

use micro_continual_core::baselines::{MfecPHBaseline, OnlineGruBaseline, TabularPHBaseline};
use micro_continual_core::math::splitmix64;
use micro_continual_core::{
    unlabeled_domain_cloze_logs_noisy, HardPomdpConfig, HardPomdpV2, HardRegime,
    MicroContinualCore, MicroCoreConfig, ParaphraseSplit, NUM_ENV_ACTIONS,
};

#[derive(Clone, Copy)]
enum SuiteMode {
    StrictDense,
    StrictSparse,
}

impl SuiteMode {
    fn make_cfg(self, reg: HardRegime, glitch_pct: u8, sp: ParaphraseSplit) -> HardPomdpConfig {
        match self {
            SuiteMode::StrictDense => {
                let mut c = HardPomdpConfig::strict_honest(reg, glitch_pct, sp);
                c.sparse_reward = false;
                c
            }
            SuiteMode::StrictSparse => HardPomdpConfig::strict_honest(reg, glitch_pct, sp),
        }
    }

    fn eps_per_regime(self) -> usize {
        match self {
            SuiteMode::StrictDense => 240,
            // В чисто разреженном режиме (шанс случайного успеха 1/1728) даём 400 эпизодов на режим
            SuiteMode::StrictSparse => 400,
        }
    }
}

#[derive(Default, Clone)]
struct SeedMetrics {
    score_online: f64,
    retention_alpha: f64,
    holdout_score: f64,
    recovery_steps: f64,
    ram_kb: f64,
    us_per_step: f64,
    seu_safe: Option<bool>,
}

fn mean_sd(vals: &[f64]) -> (f64, f64) {
    let n = vals.len().max(1) as f64;
    let m = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n;
    (m, var.sqrt())
}

fn compute_recovery_eps(solved_window: &[bool]) -> f64 {
    let win = 20usize;
    if solved_window.len() < win {
        return solved_window.len() as f64;
    }
    for i in 0..=(solved_window.len() - win) {
        let hits = solved_window[i..i + win].iter().filter(|&&b| b).count();
        if hits >= 10 {
            return (i + win) as f64;
        }
    }
    solved_window.len() as f64
}

fn run_protocol_micro(seed: u64, mode: SuiteMode, glitch_pct: u8, use_cloze: bool) -> SeedMetrics {
    let mut cfg = MicroCoreConfig::default();
    cfg.seed = seed;
    let mut agent = MicroContinualCore::new(cfg);

    if use_cloze {
        let logs = unlabeled_domain_cloze_logs_noisy();
        agent.unlabeled_cloze_adapt(&logs, 35);
    }

    let ep_n = mode.eps_per_regime();
    let mut total_steps = 0u64;
    let t0 = Instant::now();

    let mut run_phase = |ag: &mut MicroContinualCore,
                         reg: HardRegime,
                         sp: ParaphraseSplit,
                         count: usize,
                         base_sd: u64,
                         train: bool,
                         seu: bool|
     -> (Vec<bool>, bool) {
        let mut solved_vec = Vec::with_capacity(count);
        let mut finite_ok = true;
        for i in 0..count {
            let mut ep = HardPomdpV2::generate(base_sd + i as u64, mode.make_cfg(reg, glitch_pct, sp));
            ag.start_episode();
            let mut obs = ep.initial_observation();
            let mut ep_ret = 0.0_f32;
            let mut ep_solved = false;
            while ep.step_idx < ep.config.max_steps {
                let p = ag.decide_text(&obs, train, seu, seu);
                if !p.get().1.is_finite() {
                    finite_ok = false;
                }
                let act = p.get().0;
                total_steps += 1;
                let sr = ep.step(act);
                let out = sr.get();
                ag.observe_step_reward(out.reward);
                ep_ret += out.reward;
                if out.done {
                    ep_solved = out.solved;
                    break;
                }
                obs = out.observation.clone();
            }
            let _ = ag.end_episode_with_solved(ep_ret, ep_solved, train);
            if !ag.belief.state.iter().all(|v| v.is_finite()) {
                finite_ok = false;
            }
            solved_vec.push(ep_solved);
        }
        (solved_vec, finite_ok)
    };

    let sb = seed * 100_000;
    let (s_a, _) = run_phase(&mut agent, HardRegime::Alpha, ParaphraseSplit::CanonicalTrain, ep_n, sb + 1, true, false);
    let (s_hold, _) = run_phase(&mut agent, HardRegime::Alpha, ParaphraseSplit::UnseenHoldout, 60, sb + 10_000, false, false);
    let (s_b, _) = run_phase(&mut agent, HardRegime::Beta, ParaphraseSplit::CanonicalTrain, ep_n, sb + 20_000, true, false);
    let (s_g, _) = run_phase(&mut agent, HardRegime::Gamma, ParaphraseSplit::CanonicalTrain, ep_n, sb + 30_000, true, false);
    let (s_ret, _) = run_phase(&mut agent, HardRegime::Alpha, ParaphraseSplit::CanonicalTrain, 100, sb + 40_000, true, false);

    let mut seu_probe = MicroContinualCore::new(cfg);
        seu_probe.start_episode();
        let p_seu = seu_probe.decide_text("node alert disk array", false, true, true);
        let ok_seu = p_seu.get().1.is_finite() && seu_probe.belief.state.iter().all(|v| v.is_finite());

    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
    let tail_acc = |v: &[bool]| -> f64 {
        let k = v.len().min(80);
        let slice = &v[v.len() - k..];
        (slice.iter().filter(|&&b| b).count() as f64) * 100.0 / (k as f64)
    };
    SeedMetrics {
        score_online: (tail_acc(&s_a) + tail_acc(&s_b) + tail_acc(&s_g)) / 3.0,
        retention_alpha: tail_acc(&s_ret),
        holdout_score: (s_hold.iter().filter(|&&b| b).count() as f64) * 100.0 / (s_hold.len() as f64),
        recovery_steps: (compute_recovery_eps(&s_b) + compute_recovery_eps(&s_g)) * 0.5,
        ram_kb: (agent.memory_footprint_bytes() as f64) / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe: Some(ok_seu),
    }
}

fn run_protocol_tabular(seed: u64, mode: SuiteMode, glitch_pct: u8) -> SeedMetrics {
    let mut agent = TabularPHBaseline::new(NUM_ENV_ACTIONS, 6, seed);
    let ep_n = mode.eps_per_regime();
    let mut total_steps = 0u64;
    let t0 = Instant::now();

    let mut run_phase = |ag: &mut TabularPHBaseline, reg: HardRegime, sp: ParaphraseSplit, count: usize, base_sd: u64, train: bool| -> Vec<bool> {
        let mut solved = Vec::with_capacity(count);
        for i in 0..count {
            let mut ep = HardPomdpV2::generate(base_sd + i as u64, mode.make_cfg(reg, glitch_pct, sp));
            ag.start_episode();
            let mut obs = ep.initial_observation();
            let mut ret = 0.0_f32;
            let mut ok = false;
            while ep.step_idx < ep.config.max_steps {
                let act = ag.decide(&obs, train);
                total_steps += 1;
                let sr = ep.step(act);
                let out = sr.get();
                ag.observe_step_reward(out.reward);
                ret += out.reward;
                if out.done {
                    ok = out.solved;
                    break;
                }
                obs = out.observation.clone();
            }
            ag.end_episode(ret, train);
            solved.push(ok);
        }
        solved
    };

    let sb = seed * 100_000;
    let s_a = run_phase(&mut agent, HardRegime::Alpha, ParaphraseSplit::CanonicalTrain, ep_n, sb + 1, true);
    let s_hold = run_phase(&mut agent, HardRegime::Alpha, ParaphraseSplit::UnseenHoldout, 60, sb + 10_000, false);
    let s_b = run_phase(&mut agent, HardRegime::Beta, ParaphraseSplit::CanonicalTrain, ep_n, sb + 20_000, true);
    let s_g = run_phase(&mut agent, HardRegime::Gamma, ParaphraseSplit::CanonicalTrain, ep_n, sb + 30_000, true);
    let s_ret = run_phase(&mut agent, HardRegime::Alpha, ParaphraseSplit::CanonicalTrain, 100, sb + 40_000, true);

    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
    let tail_acc = |v: &[bool]| -> f64 {
        let k = v.len().min(80);
        let slice = &v[v.len() - k..];
        (slice.iter().filter(|&&b| b).count() as f64) * 100.0 / (k as f64)
    };
    SeedMetrics {
        score_online: (tail_acc(&s_a) + tail_acc(&s_b) + tail_acc(&s_g)) / 3.0,
        retention_alpha: tail_acc(&s_ret),
        holdout_score: (s_hold.iter().filter(|&&b| b).count() as f64) * 100.0 / (s_hold.len() as f64),
        recovery_steps: (compute_recovery_eps(&s_b) + compute_recovery_eps(&s_g)) * 0.5,
        ram_kb: (agent.memory_footprint_bytes() as f64) / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe: None,
    }
}

fn run_protocol_gru(seed: u64, mode: SuiteMode, glitch_pct: u8) -> SeedMetrics {
    let mut agent = OnlineGruBaseline::new(64, 32, NUM_ENV_ACTIONS, seed);
    let ep_n = mode.eps_per_regime();
    let mut total_steps = 0u64;
    let t0 = Instant::now();

    let mut run_phase = |ag: &mut OnlineGruBaseline, reg: HardRegime, sp: ParaphraseSplit, count: usize, base_sd: u64, train: bool, seu: bool| -> Vec<bool> {
        let mut solved = Vec::with_capacity(count);
        for i in 0..count {
            let mut ep = HardPomdpV2::generate(base_sd + i as u64, mode.make_cfg(reg, glitch_pct, sp));
            ag.start_episode();
            let mut obs = ep.initial_observation();
            let mut ret = 0.0_f32;
            let mut ok = false;
            while ep.step_idx < ep.config.max_steps {
                let act = ag.decide(&obs, train, seu, seu);
                total_steps += 1;
                let sr = ep.step(act);
                let out = sr.get();
                ag.observe_step_reward(out.reward);
                ret += out.reward;
                if out.done {
                    ok = out.solved;
                    break;
                }
                obs = out.observation.clone();
            }
            ag.end_episode(ret, train);
            solved.push(ok);
        }
        solved
    };

    let sb = seed * 100_000;
    let s_a = run_phase(&mut agent, HardRegime::Alpha, ParaphraseSplit::CanonicalTrain, ep_n, sb + 1, true, false);
    let s_hold = run_phase(&mut agent, HardRegime::Alpha, ParaphraseSplit::UnseenHoldout, 60, sb + 10_000, false, false);
    let s_b = run_phase(&mut agent, HardRegime::Beta, ParaphraseSplit::CanonicalTrain, ep_n, sb + 20_000, true, false);
    let s_g = run_phase(&mut agent, HardRegime::Gamma, ParaphraseSplit::CanonicalTrain, ep_n, sb + 30_000, true, false);
    let s_ret = run_phase(&mut agent, HardRegime::Alpha, ParaphraseSplit::CanonicalTrain, 100, sb + 40_000, true, false);

    let mut seu_probe = OnlineGruBaseline::new(64, 32, NUM_ENV_ACTIONS, seed);
    let _ = seu_probe.decide("node alert disk array", false, true, true);

    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
    let tail_acc = |v: &[bool]| -> f64 {
        let k = v.len().min(80);
        let slice = &v[v.len() - k..];
        (slice.iter().filter(|&&b| b).count() as f64) * 100.0 / (k as f64)
    };
    SeedMetrics {
        score_online: (tail_acc(&s_a) + tail_acc(&s_b) + tail_acc(&s_g)) / 3.0,
        retention_alpha: tail_acc(&s_ret),
        holdout_score: (s_hold.iter().filter(|&&b| b).count() as f64) * 100.0 / (s_hold.len() as f64),
        recovery_steps: (compute_recovery_eps(&s_b) + compute_recovery_eps(&s_g)) * 0.5,
        ram_kb: (agent.memory_footprint_bytes() as f64) / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe: Some(!seu_probe.had_nan),
    }
}

fn run_protocol_mfec(seed: u64, mode: SuiteMode, glitch_pct: u8, fair_ema: bool) -> SeedMetrics {
    let mut agent = MfecPHBaseline::new(64, NUM_ENV_ACTIONS, 112, 6, seed);
    if fair_ema {
        // Справедливый MFEC: тот же порог склейки cos=0.78 и простая EMA-память без роторов Кэли
        agent.coalesce_cos = 0.78;
        agent.use_belief_ema = true;
    }
    let ep_n = mode.eps_per_regime();
    let mut total_steps = 0u64;
    let t0 = Instant::now();

    let mut run_phase = |ag: &mut MfecPHBaseline,
                         reg: HardRegime,
                         sp: ParaphraseSplit,
                         count: usize,
                         base_sd: u64,
                         train: bool|
     -> Vec<bool> {
        let mut solved = Vec::with_capacity(count);
        for i in 0..count {
            let mut ep = HardPomdpV2::generate(base_sd + i as u64, mode.make_cfg(reg, glitch_pct, sp));
            ag.start_episode();
            let mut obs = ep.initial_observation();
            let mut ret = 0.0_f32;
            let mut ok = false;
            while ep.step_idx < ep.config.max_steps {
                let act = ag.decide(&obs, train);
                total_steps += 1;
                let sr = ep.step(act);
                let out = sr.get();
                ag.observe_step_reward(out.reward);
                ret += out.reward;
                if out.done {
                    ok = out.solved;
                    break;
                }
                obs = out.observation.clone();
            }
            ag.end_episode_with_solved(ret, ok, train);
            solved.push(ok);
        }
        solved
    };

    let sb = seed * 100_000;
    let s_a = run_phase(&mut agent, HardRegime::Alpha, ParaphraseSplit::CanonicalTrain, ep_n, sb + 1, true);
    let s_hold = run_phase(&mut agent, HardRegime::Alpha, ParaphraseSplit::UnseenHoldout, 60, sb + 10_000, false);
    let s_b = run_phase(&mut agent, HardRegime::Beta, ParaphraseSplit::CanonicalTrain, ep_n, sb + 20_000, true);
    let s_g = run_phase(&mut agent, HardRegime::Gamma, ParaphraseSplit::CanonicalTrain, ep_n, sb + 30_000, true);
    let s_ret = run_phase(&mut agent, HardRegime::Alpha, ParaphraseSplit::CanonicalTrain, 100, sb + 40_000, true);

    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
    let tail_acc = |v: &[bool]| -> f64 {
        let k = v.len().min(80);
        let slice = &v[v.len() - k..];
        (slice.iter().filter(|&&b| b).count() as f64) * 100.0 / (k as f64)
    };
    SeedMetrics {
        score_online: (tail_acc(&s_a) + tail_acc(&s_b) + tail_acc(&s_g)) / 3.0,
        retention_alpha: tail_acc(&s_ret),
        holdout_score: (s_hold.iter().filter(|&&b| b).count() as f64) * 100.0 / (s_hold.len() as f64),
        recovery_steps: (compute_recovery_eps(&s_b) + compute_recovery_eps(&s_g)) * 0.5,
        ram_kb: (agent.memory_footprint_bytes() as f64) / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe: None,
    }
}

fn run_protocol_oracle_or_random(seed: u64, mode: SuiteMode, glitch_pct: u8, is_oracle: bool) -> SeedMetrics {
    let mut rng = splitmix64(seed ^ 0x5555_AAAA);
    let mut solved_cnt = 0usize;
    let total = 200usize;
    for i in 0..total {
        let mut ep = HardPomdpV2::generate(
            seed * 100_000 + i as u64,
            mode.make_cfg(HardRegime::Alpha, glitch_pct, ParaphraseSplit::CanonicalTrain),
        );
        while ep.step_idx < ep.config.max_steps {
            let act = if is_oracle {
                ep.oracle_action()
            } else {
                rng = splitmix64(rng);
                (rng as usize) % NUM_ENV_ACTIONS
            };
            let sr = ep.step(act);
            if sr.get().done {
                if sr.get().solved {
                    solved_cnt += 1;
                }
                break;
            }
        }
    }
    let acc = (solved_cnt as f64) * 100.0 / (total as f64);
    SeedMetrics {
        score_online: acc,
        retention_alpha: acc,
        holdout_score: acc,
        recovery_steps: if is_oracle { 0.0 } else { mode.eps_per_regime() as f64 },
        ram_kb: 0.0,
        us_per_step: 0.1,
        seu_safe: None,
    }
}

fn print_row(name: &str, ms: &[SeedMetrics]) {
    let sc: Vec<f64> = ms.iter().map(|m| m.score_online).collect();
    let ret: Vec<f64> = ms.iter().map(|m| m.retention_alpha).collect();
    let hold: Vec<f64> = ms.iter().map(|m| m.holdout_score).collect();
    let rec: Vec<f64> = ms.iter().map(|m| m.recovery_steps).collect();
    let ram: Vec<f64> = ms.iter().map(|m| m.ram_kb).collect();
    let lat: Vec<f64> = ms.iter().map(|m| m.us_per_step).collect();
    let seu_str = if ms.iter().all(|m| m.seu_safe.is_none()) {
        "N/A (str)"
    } else if ms.iter().all(|m| m.seu_safe == Some(true)) {
        "YES"
    } else {
        "NO (NaN)"
    };

    let (sm, ss) = mean_sd(&sc);
    let (rm, _) = mean_sd(&ret);
    let (hm, _) = mean_sd(&hold);
    let (rcm, _) = mean_sd(&rec);
    let (ram_m, _) = mean_sd(&ram);
    let (lat_m, _) = mean_sd(&lat);

    println!(
        "{:<26} | {:5.1}±{:4.1}% (H:{:4.1}%) | {:6.1}% | {:8.1} ep | {:7.1} KB | {:7.2} µs | {}",
        name,
        sm,
        ss,
        hm,
        rm,
        rcm,
        ram_m,
        lat_m,
        seu_str
    );
}

fn run_suite(title: &str, mode: SuiteMode, glitch_pct: u8) {
    println!("\n=== {} (glitch={}%, {} ep/regime, 10 seeds) ===", title, glitch_pct, mode.eps_per_regime());
    println!(
        "{:<26} | {:<22} | {:<7} | {:<11} | {:<10} | {:<10} | {}",
        "Model", "Score (mean±sd, Hold)", "Retent.", "Recovery", "RAM (KB)", "Latency", "SEU-Safe"
    );
    println!("{}", "-".repeat(110));

    let seeds: Vec<u64> = (1..=10).collect();
    let r_micro: Vec<SeedMetrics> = seeds.iter().map(|&s| run_protocol_micro(s, mode, glitch_pct, false)).collect();
    let r_cloze: Vec<SeedMetrics> = seeds.iter().map(|&s| run_protocol_micro(s, mode, glitch_pct, true)).collect();
    let r_mfec: Vec<SeedMetrics> = seeds.iter().map(|&s| run_protocol_mfec(s, mode, glitch_pct, false)).collect();
    let r_mfec_fair: Vec<SeedMetrics> = seeds.iter().map(|&s| run_protocol_mfec(s, mode, glitch_pct, true)).collect();
    let r_tab: Vec<SeedMetrics> = seeds.iter().map(|&s| run_protocol_tabular(s, mode, glitch_pct)).collect();
    let r_gru: Vec<SeedMetrics> = seeds.iter().map(|&s| run_protocol_gru(s, mode, glitch_pct)).collect();
    let r_ora: Vec<SeedMetrics> = seeds.iter().map(|&s| run_protocol_oracle_or_random(s, mode, glitch_pct, true)).collect();
    let r_rnd: Vec<SeedMetrics> = seeds.iter().map(|&s| run_protocol_oracle_or_random(s, mode, glitch_pct, false)).collect();

    print_row("MicroContinualCore", &r_micro);
    print_row("MicroCore (+Can. Cloze)", &r_cloze);
    print_row("MFEC + PH (orig 0.94)", &r_mfec);
    print_row("MFEC + PH (fair 0.78+EMA)", &r_mfec_fair);
    print_row("Tabular + PH", &r_tab);
    print_row("Online GRU + A2C", &r_gru);
    print_row("Oracle (ceiling)", &r_ora);
    print_row("Random (floor)", &r_rnd);
}

fn main() {
    run_suite("HardPOMDP v2 STRICT HONEST (dense)", SuiteMode::StrictDense, 15);
    run_suite("HardPOMDP v2 STRICT HONEST (sparse)", SuiteMode::StrictSparse, 15);
    run_suite("HardPOMDP v2 STRICT HONEST (sparse)", SuiteMode::StrictSparse, 25);
}