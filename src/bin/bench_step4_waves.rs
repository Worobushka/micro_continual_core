// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Бенчмарк Шага 4 (v0.3.0): Масштабирование длинной памяти и эксперимент волновой суперпозиции
//! 10 сидов, прямое сравнение на одном CPU:
//! 1. Non-Stationary T-Maze L=30 и L=50 (шум ±0.35, Drift A -> B -> A, sparse reward)
//! 2. POPGym RepeatPrevious K=6 и K=8 (4 символа, чистое RL +1/-1)
//! 3. Wave-Superposition Zero-Shot Transfer:
//!    - Обучение ТОЛЬКО на смешанном 1D-скаляре x(t) = A(t) + B(t) + C(t) + шум (чистое RL +1/-1):
//!      A = медленный контекст (смена знака каждые 20 шагов),
//!      B = среднее колебание (период 8, случайная фаза),
//!      C = быстрая несущая (период 3, случайная фаза).
//!    - Цель: предсказать знак 1-шаг-вперёд (A+B+C)(t+1).
//!    - Заморозка весов -> Zero-Shot перенос на чистые A, B, C и невиданную суперпозицию A+B!
//!    - Сравнение: Hierarchical 2L Cayley-RTRL (< 20 KB) vs 1L Cayley-RTRL vs RTRRL-LRU vs Online GRU vs Sign(x_t).

use std::time::Instant;

use micro_continual_core::baselines::{OnlineGruBaseline, RtrrlLruBaseline};
use micro_continual_core::math::{rand_signed_f32, rand_unit_f32, splitmix64};
use micro_continual_core::{HierarchicalCayleyRtrl, MicroContinualCore, MicroCoreConfig};

#[derive(Default, Clone)]
struct RlMetrics {
    score: f64,
    retention: f64,
    recovery_eps: f64,
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

fn recovery_window(vals: &[f64], target: f64, win: usize) -> f64 {
    if vals.len() < win {
        return vals.len() as f64;
    }
    for i in 0..=(vals.len() - win) {
        let avg = vals[i..i + win].iter().sum::<f64>() / (win as f64);
        if avg >= target {
            return (i + win) as f64;
        }
    }
    vals.len() as f64
}

fn print_rl_header(title: &str) {
    println!("\n=== {} (10 seeds) ===", title);
    println!(
        "{:<26} | {:<15} | {:<8} | {:<11} | {:<10} | {:<10} | {}",
        "Model", "Score(mean±sd)", "Retent.", "Recovery", "RAM (KB)", "Latency", "SEU-Safe"
    );
    println!("{}", "-".repeat(104));
}

fn print_rl_row(name: &str, unit: &str, ms: &[RlMetrics]) {
    let sc: Vec<f64> = ms.iter().map(|m| m.score).collect();
    let rt: Vec<f64> = ms.iter().map(|m| m.retention).collect();
    let rc: Vec<f64> = ms.iter().map(|m| m.recovery_eps).collect();
    let rm: Vec<f64> = ms.iter().map(|m| m.ram_kb).collect();
    let lt: Vec<f64> = ms.iter().map(|m| m.us_per_step).collect();
    let seu = ms.iter().all(|m| m.seu_safe);

    let (sm, ss) = mean_sd(&sc);
    let (rtm, _) = mean_sd(&rt);
    let (rcm, _) = mean_sd(&rc);
    let (rmm, _) = mean_sd(&rm);
    let (ltm, _) = mean_sd(&lt);

    println!(
        "{:<26} | {:6.1}±{:4.1}{:<2} | {:6.1}{:<2} | {:8.1} ep | {:7.1} KB | {:7.2} µs | {}",
        name,
        sm,
        ss,
        unit,
        rtm,
        unit,
        rcm,
        rmm,
        ltm,
        if seu { "YES" } else { "NO (NaN)" }
    );
}

// =============================================================================
// 1. Long-Horizon Non-Stationary T-Maze (L=30, L=50, noise=±0.35, Drift A->B->A)
// =============================================================================

fn run_tmaze_micro(corridor_len: usize, seed: u64) -> RlMetrics {
    let noise_amp = 0.35_f32;
    let mut cfg = MicroCoreConfig::numeric_stream(2, seed);
    cfg.belief_lambda = 0.96;
    cfg.belief_gamma = 0.65;
    cfg.rotor_norm = 0.12;
    cfg.novelty_gate = 0.28;
    cfg.linear_lr = 0.20;
    cfg.linear_mix = 0.55;
    cfg.bank_capacity = 48;
    cfg.obs_bank_capacity = 16;
    cfg.trace_capacity = corridor_len + 4;
    cfg.explore_ucb = 0.35;
    cfg.trace_gamma = 0.98;
    cfg.ph_delta = 0.18;
    cfg.ph_threshold = 2.6;
    cfg.ph_warmup = 30;

    let mut core = MicroContinualCore::new(cfg);
    core.encoder.freeze_to_i8();
    let mut rng = splitmix64(seed ^ 0x7A1E_5CA1);
    let mut steps = 0u64;
    let t0 = Instant::now();

    let mut run_regime = |c: &mut MicroContinualCore, flipped: bool, n_eps: usize| -> Vec<f64> {
        let mut res = Vec::with_capacity(n_eps);
        for _ in 0..n_eps {
            rng = splitmix64(rng);
            let cue = (rng & 1) as usize;
            let expected = if flipped { 1 - cue } else { cue };
            c.start_episode();

            let obs0 = [if cue == 0 { -1.0_f32 } else { 1.0_f32 }, 0.0, 0.0];
            let _ = c.decide_vec(&obs0, true, false, false);
            c.observe_step_reward(0.0);
            steps += 1;

            for _ in 0..corridor_len {
                let nz = rand_signed_f32(&mut rng) * noise_amp;
                let _ = c.decide_vec(&[0.0, 0.0, nz], true, false, false);
                c.observe_step_reward(0.0);
                steps += 1;
            }

            let act = c.decide_vec(&[0.0, 1.0, 0.0], true, false, false).get().0;
            steps += 1;
            let ok = act == expected;
            let r = if ok { 1.0_f32 } else { -1.0_f32 };
            c.observe_step_reward(r);
            let _ = c.end_episode(r, true);
            res.push(if ok { 1.0 } else { 0.0 });
        }
        res
    };

    let ra = run_regime(&mut core, false, 120);
    let rb = run_regime(&mut core, true, 120);
    let rret = run_regime(&mut core, false, 80);
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    core.start_episode();
    let p_seu = core.decide_vec(&[f32::NAN, 1e30, -1e30], false, true, true);
    let seu_safe = p_seu.get().1.is_finite() && core.belief.state.iter().all(|v| v.is_finite());

    let tail = |v: &[f64]| v[v.len() - 40..].iter().sum::<f64>() * 100.0 / 40.0;
    RlMetrics {
        score: (tail(&ra) + tail(&rb)) * 0.5,
        retention: tail(&rret),
        recovery_eps: recovery_window(&rb, 0.75, 16),
        ram_kb: core.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (steps.max(1) as f64),
        seu_safe,
    }
}

fn run_tmaze_lru(corridor_len: usize, seed: u64) -> RlMetrics {
    let noise_amp = 0.35_f32;
    let mut lru = RtrrlLruBaseline::new(3, 24, 2, seed);
    lru.gamma = 0.98;
    lru.trace_lambda = 0.90;
    lru.actor_lr = 0.05;
    lru.critic_lr = 0.14;
    lru.rtrl_lr = 0.006;
    let mut rng = splitmix64(seed ^ 0x7A1E_5CA1);
    let mut steps = 0u64;
    let t0 = Instant::now();

    let mut run_regime = |m: &mut RtrrlLruBaseline, flipped: bool, n_eps: usize| -> Vec<f64> {
        let mut res = Vec::with_capacity(n_eps);
        for _ in 0..n_eps {
            rng = splitmix64(rng);
            let cue = (rng & 1) as usize;
            let expected = if flipped { 1 - cue } else { cue };
            m.start_episode();
            let obs0 = [if cue == 0 { -1.0_f32 } else { 1.0_f32 }, 0.0, 0.0];
            let _ = m.decide_vec(&obs0, true, false, false);
            m.observe_step_reward(0.0);
            steps += 1;
            for _ in 0..corridor_len {
                let nz = rand_signed_f32(&mut rng) * noise_amp;
                let _ = m.decide_vec(&[0.0, 0.0, nz], true, false, false);
                m.observe_step_reward(0.0);
                steps += 1;
            }
            let act = m.decide_vec(&[0.0, 1.0, 0.0], true, false, false);
            steps += 1;
            let ok = act == expected;
            let r = if ok { 1.0_f32 } else { -1.0_f32 };
            m.observe_step_reward(r);
            m.end_episode(true);
            res.push(if ok { 1.0 } else { 0.0 });
        }
        res
    };

    let ra = run_regime(&mut lru, false, 120);
    let rb = run_regime(&mut lru, true, 120);
    let rret = run_regime(&mut lru, false, 80);
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let mut probe = RtrrlLruBaseline::new(3, 24, 2, seed);
    let _ = probe.decide_vec(&[0.0; 3], false, true, true);
    let tail = |v: &[f64]| v[v.len() - 40..].iter().sum::<f64>() * 100.0 / 40.0;
    RlMetrics {
        score: (tail(&ra) + tail(&rb)) * 0.5,
        retention: tail(&rret),
        recovery_eps: recovery_window(&rb, 0.75, 16),
        ram_kb: lru.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (steps.max(1) as f64),
        seu_safe: !probe.had_nan,
    }
}

fn run_tmaze_gru(corridor_len: usize, seed: u64) -> RlMetrics {
    let noise_amp = 0.35_f32;
    let mut gru = OnlineGruBaseline::new(3, 32, 2, seed);
    gru.lr = 0.008;
    gru.gamma = 0.98;
    let mut rng = splitmix64(seed ^ 0x7A1E_5CA1);
    let mut steps = 0u64;
    let t0 = Instant::now();

    let mut run_regime = |g: &mut OnlineGruBaseline, flipped: bool, n_eps: usize| -> Vec<f64> {
        let mut res = Vec::with_capacity(n_eps);
        for _ in 0..n_eps {
            rng = splitmix64(rng);
            let cue = (rng & 1) as usize;
            let expected = if flipped { 1 - cue } else { cue };
            g.start_episode();
            let obs0 = [if cue == 0 { -1.0_f32 } else { 1.0_f32 }, 0.0, 0.0];
            let _ = g.decide_vec(&obs0, true, false, false);
            g.observe_step_reward(0.0);
            steps += 1;
            for _ in 0..corridor_len {
                let nz = rand_signed_f32(&mut rng) * noise_amp;
                let _ = g.decide_vec(&[0.0, 0.0, nz], true, false, false);
                g.observe_step_reward(0.0);
                steps += 1;
            }
            let act = g.decide_vec(&[0.0, 1.0, 0.0], true, false, false);
            steps += 1;
            let ok = act == expected;
            let r = if ok { 1.0_f32 } else { -1.0_f32 };
            g.observe_step_reward(r);
            g.end_episode(r, true);
            res.push(if ok { 1.0 } else { 0.0 });
        }
        res
    };

    let ra = run_regime(&mut gru, false, 120);
    let rb = run_regime(&mut gru, true, 120);
    let rret = run_regime(&mut gru, false, 80);
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let mut probe = OnlineGruBaseline::new(3, 32, 2, seed);
    let _ = probe.decide_vec(&[0.0; 3], false, true, true);
    let tail = |v: &[f64]| v[v.len() - 40..].iter().sum::<f64>() * 100.0 / 40.0;
    RlMetrics {
        score: (tail(&ra) + tail(&rb)) * 0.5,
        retention: tail(&rret),
        recovery_eps: recovery_window(&rb, 0.75, 16),
        ram_kb: gru.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (steps.max(1) as f64),
        seu_safe: !probe.had_nan,
    }
}

// =============================================================================
// 2. Deep Delay POPGym RepeatPrevious (K=6, K=8, 4 symbols, pure RL +1/-1)
// =============================================================================

fn run_rp_micro(k_delay: usize, seed: u64) -> RlMetrics {
    let mut cfg = MicroCoreConfig::numeric_stream(4, seed);
    cfg.half_sectors = 24;
    cfg.standardize_vec = false;
    cfg.linear_wd = 0.0;
    cfg.belief_lambda = 0.94;
    cfg.belief_gamma = 0.0;
    cfg.rotor_norm = 0.56;
    cfg.novelty_gate = 0.0;
    cfg.linear_lr = 0.14;
    cfg.linear_mix = 1.0;
    cfg.explore_ucb = 0.10;
    cfg.ph_threshold = 99.0;

    let mut core = MicroContinualCore::new(cfg);
    core.encoder.freeze_to_i8();
    let mut rng = splitmix64(seed ^ 0x2E9E_6800);
    let n_eps = 220usize;
    let ep_len = 28usize;
    let mut ep_accs = Vec::with_capacity(n_eps);
    let mut total_steps = 0u64;
    let t0 = Instant::now();

    for ep in 0..n_eps {
        core.start_episode();
        let mut hist = Vec::with_capacity(ep_len);
        let mut hits = 0usize;
        let mut eval_n = 0usize;
        let train_flag = ep < n_eps - 30;
        for t in 0..ep_len {
            rng = splitmix64(rng);
            let sym = ((rng >> 16) as usize) % 4;
            hist.push(sym);
            let mut obs = [0.0_f32; 4];
            obs[sym] = 1.0;
            let pred = core.decide_vec(&obs, train_flag, false, false).get().0;
            total_steps += 1;
            if t >= k_delay {
                let ok = pred == hist[t - k_delay];
                if ok {
                    hits += 1;
                }
                eval_n += 1;
                if train_flag {
                    core.observe_step_bandit_reward(pred, if ok { 1.0 } else { -1.0 });
                }
            }
        }
        ep_accs.push((hits as f64) * 100.0 / (eval_n.max(1) as f64));
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
    core.start_episode();
    let p_seu = core.decide_vec(&[f32::NAN, 1e30, 0.0, 0.0], false, true, true);
    let seu_safe = p_seu.get().1.is_finite() && core.belief.state.iter().all(|v| v.is_finite());
    let tail = ep_accs[ep_accs.len() - 30..].iter().sum::<f64>() / 30.0;
    RlMetrics {
        score: tail,
        retention: tail,
        recovery_eps: recovery_window(&ep_accs, 50.0, 10),
        ram_kb: core.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe,
    }
}

fn run_rp_hier(k_delay: usize, seed: u64) -> RlMetrics {
    let mut h = HierarchicalCayleyRtrl::new(4, 16, 16, 4, seed);
    h.gamma = 0.0;
    h.trace_lambda = 0.0;
    h.actor_lr = 0.08;
    h.critic_lr = 0.16;
    h.rtrl_lr = 0.012;
    h.entropy_beta = 0.003;
    let mut rng = splitmix64(seed ^ 0x2E9E_6800);
    let n_eps = 220usize;
    let ep_len = 28usize;
    let mut ep_accs = Vec::with_capacity(n_eps);
    let mut total_steps = 0u64;
    let t0 = Instant::now();

    for ep in 0..n_eps {
        h.start_episode();
        let mut hist = Vec::with_capacity(ep_len);
        let mut hits = 0usize;
        let mut eval_n = 0usize;
        let train_flag = ep < n_eps - 30;
        for t in 0..ep_len {
            rng = splitmix64(rng);
            let sym = ((rng >> 16) as usize) % 4;
            hist.push(sym);
            let mut obs = [0.0_f32; 4];
            obs[sym] = 1.0;
            let pred = h.decide_vec(&obs, train_flag, false, false);
            total_steps += 1;
            if t >= k_delay {
                let ok = pred == hist[t - k_delay];
                if ok {
                    hits += 1;
                }
                eval_n += 1;
                h.observe_step_reward(if ok { 1.0 } else { -1.0 });
            } else {
                h.observe_step_reward(0.0);
            }
        }
        h.end_episode(train_flag);
        ep_accs.push((hits as f64) * 100.0 / (eval_n.max(1) as f64));
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
    h.start_episode();
    let _ = h.decide_vec(&[f32::NAN, 1e30, 0.0, 0.0], false, true, true);
    let seu_safe = h.fast_state.b.iter().all(|v| v.is_finite()) && h.slow_state.b.iter().all(|v| v.is_finite());
    let tail = ep_accs[ep_accs.len() - 30..].iter().sum::<f64>() / 30.0;
    RlMetrics {
        score: tail,
        retention: tail,
        recovery_eps: recovery_window(&ep_accs, 50.0, 10),
        ram_kb: h.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe,
    }
}

fn run_rp_lru(k_delay: usize, seed: u64) -> RlMetrics {
    let mut lru = RtrrlLruBaseline::new(4, 48, 4, seed);
    lru.gamma = 0.0;
    lru.trace_lambda = 0.0;
    lru.actor_lr = 0.08;
    lru.critic_lr = 0.16;
    lru.rtrl_lr = 0.012;
    lru.entropy_beta = 0.004;
    let mut rng = splitmix64(seed ^ 0x2E9E_6800);
    let n_eps = 220usize;
    let ep_len = 28usize;
    let mut ep_accs = Vec::with_capacity(n_eps);
    let mut total_steps = 0u64;
    let t0 = Instant::now();

    for ep in 0..n_eps {
        lru.start_episode();
        let mut hist = Vec::with_capacity(ep_len);
        let mut hits = 0usize;
        let mut eval_n = 0usize;
        let train_flag = ep < n_eps - 30;
        for t in 0..ep_len {
            rng = splitmix64(rng);
            let sym = ((rng >> 16) as usize) % 4;
            hist.push(sym);
            let mut obs = [0.0_f32; 4];
            obs[sym] = 1.0;
            let pred = lru.decide_vec(&obs, train_flag, false, false);
            total_steps += 1;
            if t >= k_delay {
                let ok = pred == hist[t - k_delay];
                if ok {
                    hits += 1;
                }
                eval_n += 1;
                lru.observe_step_reward(if ok { 1.0 } else { -1.0 });
            } else {
                lru.observe_step_reward(0.0);
            }
        }
        lru.end_episode(train_flag);
        ep_accs.push((hits as f64) * 100.0 / (eval_n.max(1) as f64));
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
    let mut probe = RtrrlLruBaseline::new(4, 48, 4, seed);
    let _ = probe.decide_vec(&[0.0; 4], false, true, true);
    let tail = ep_accs[ep_accs.len() - 30..].iter().sum::<f64>() / 30.0;
    RlMetrics {
        score: tail,
        retention: tail,
        recovery_eps: recovery_window(&ep_accs, 50.0, 10),
        ram_kb: lru.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe: !probe.had_nan,
    }
}

fn run_rp_gru(k_delay: usize, seed: u64) -> RlMetrics {
    let mut gru = OnlineGruBaseline::new(4, 32, 4, seed);
    gru.lr = 0.008;
    gru.gamma = 0.0;
    let mut rng = splitmix64(seed ^ 0x2E9E_6800);
    let n_eps = 220usize;
    let ep_len = 28usize;
    let mut ep_accs = Vec::with_capacity(n_eps);
    let mut total_steps = 0u64;
    let t0 = Instant::now();

    for ep in 0..n_eps {
        gru.start_episode();
        let mut hist = Vec::with_capacity(ep_len);
        let mut hits = 0usize;
        let mut eval_n = 0usize;
        let train_flag = ep < n_eps - 30;
        let mut ep_ret = 0.0_f32;
        for t in 0..ep_len {
            rng = splitmix64(rng);
            let sym = ((rng >> 16) as usize) % 4;
            hist.push(sym);
            let mut obs = [0.0_f32; 4];
            obs[sym] = 1.0;
            let pred = gru.decide_vec(&obs, train_flag, false, false);
            total_steps += 1;
            if t >= k_delay {
                let ok = pred == hist[t - k_delay];
                if ok {
                    hits += 1;
                }
                eval_n += 1;
                let r = if ok { 1.0_f32 } else { -1.0_f32 };
                gru.observe_step_reward(r);
                ep_ret += r;
            } else {
                gru.observe_step_reward(0.0);
            }
        }
        gru.end_episode(ep_ret, train_flag);
        ep_accs.push((hits as f64) * 100.0 / (eval_n.max(1) as f64));
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
    let mut probe = OnlineGruBaseline::new(4, 32, 4, seed);
    let _ = probe.decide_vec(&[0.0; 4], false, true, true);
    let tail = ep_accs[ep_accs.len() - 30..].iter().sum::<f64>() / 30.0;
    RlMetrics {
        score: tail,
        retention: tail,
        recovery_eps: recovery_window(&ep_accs, 50.0, 10),
        ram_kb: gru.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe: !probe.had_nan,
    }
}

// =============================================================================
// 3. Прямой эксперимент волновой суперпозиции и Zero-Shot переноса (A+B+C -> A, B, C, A+B)
// =============================================================================

struct WaveEp {
    obs: Vec<f32>,
    targets: Vec<usize>,
}

fn gen_wave_episode(rng: &mut u64, wa: f32, wb: f32, wc: f32, ep_len: usize) -> WaveEp {
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
    let mut targets = Vec::with_capacity(ep_len);
    for t in 0..ep_len {
        let noise = rand_signed_f32(rng) * 0.06;
        obs.push(eval_wave(t) + noise);
        let next_val = eval_wave(t + 1);
        targets.push(if next_val >= 0.0 { 1usize } else { 0usize });
    }
    WaveEp { obs, targets }
}

#[derive(Default, Clone)]
struct WaveTransferMetrics {
    abc: f64,
    only_a: f64,
    only_b: f64,
    only_c: f64,
    ab: f64,
    ram_kb: f64,
    us_per_step: f64,
    seu_safe: bool,
}

impl WaveTransferMetrics {
    fn mean_zero_shot(&self) -> f64 {
        (self.only_a + self.only_b + self.only_c + self.ab) * 0.25
    }
}

fn eval_wave_suite<FStep>(mut rng: u64, mut step_fn: FStep) -> (f64, f64, f64, f64, f64)
where
    FStep: FnMut(bool, f32) -> usize,
{
    let configs = [
        (1.0_f32, 1.0_f32, 1.0_f32), // A+B+C (in-distribution)
        (1.0, 0.0, 0.0),             // Zero-Shot Pure A (slow)
        (0.0, 1.0, 0.0),             // Zero-Shot Pure B (mid, period 8)
        (0.0, 0.0, 1.0),             // Zero-Shot Pure C (fast, period 3)
        (1.0, 1.0, 0.0),             // Zero-Shot Superset A+B
    ];
    let mut scores = [0.0_f64; 5];
    let n_eval_eps = 40usize;
    let ep_len = 48usize;
    for (ci, &(wa, wb, wc)) in configs.iter().enumerate() {
        let mut hits = 0usize;
        let mut total = 0usize;
        for _ in 0..n_eval_eps {
            let ep = gen_wave_episode(&mut rng, wa, wb, wc, ep_len);
            for t in 0..ep_len {
                let pred = step_fn(t == 0, ep.obs[t]);
                if t >= 6 {
                    if pred == ep.targets[t] {
                        hits += 1;
                    }
                    total += 1;
                }
            }
        }
        scores[ci] = (hits as f64) * 100.0 / (total.max(1) as f64);
    }
    (scores[0], scores[1], scores[2], scores[3], scores[4])
}

fn run_wave_hier(seed: u64) -> WaveTransferMetrics {
    let mut h = HierarchicalCayleyRtrl::new(1, 14, 14, 2, seed);
    h.gamma = 0.0;
    h.trace_lambda = 0.0;
    h.actor_lr = 0.075;
    h.critic_lr = 0.14;
    h.rtrl_lr = 0.010;
    h.entropy_beta = 0.003;

    let mut rng = splitmix64(seed ^ 0xBA7E_509E);
    let n_train = 220usize;
    let ep_len = 48usize;
    let mut steps = 0u64;
    let t0 = Instant::now();

    for _ in 0..n_train {
        let ep = gen_wave_episode(&mut rng, 1.0, 1.0, 1.0, ep_len);
        h.start_episode();
        for t in 0..ep_len {
            let pred = h.decide_vec(&[ep.obs[t]], true, false, false);
            steps += 1;
            let r = if t >= 6 {
                if pred == ep.targets[t] { 1.0_f32 } else { -1.0_f32 }
            } else {
                0.0_f32
            };
            h.observe_step_reward(r);
        }
        h.end_episode(true);
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    // Замороженная Zero-Shot оценка (train_mode = false!)
    let (abc, a, b, c, ab) = eval_wave_suite(splitmix64(seed ^ 0xE7A1_0001), |rst, x| {
        if rst {
            h.start_episode();
        }
        h.decide_vec(&[x], false, false, false)
    });
    h.start_episode();
    let _ = h.decide_vec(&[f32::NAN], false, true, true);
    let seu_safe = h.fast_state.b.iter().all(|v| v.is_finite()) && h.slow_state.b.iter().all(|v| v.is_finite());

    WaveTransferMetrics {
        abc,
        only_a: a,
        only_b: b,
        only_c: c,
        ab,
        ram_kb: h.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (steps.max(1) as f64),
        seu_safe,
    }
}

fn run_wave_1l_cayley(seed: u64) -> WaveTransferMetrics {
    let mut cfg = MicroCoreConfig::numeric_stream(2, seed);
    cfg.half_sectors = 12;
    cfg.standardize_vec = false;
    cfg.belief_lambda = 0.88;
    cfg.belief_gamma = 0.0;
    cfg.rotor_norm = 0.55;
    cfg.novelty_gate = 0.0;
    cfg.trace_gamma = 0.0;
    cfg.linear_lr = 0.0;
    cfg.linear_mix = 1.0;
    cfg.linear_wd = 0.0001;
    cfg.actor_critic_lr = 0.07;
    cfg.critic_lr = 0.14;
    cfg.entropy_beta = 0.003;
    cfg.rtrl_sectors = 16;
    cfg.rtrl_max_in = 1;
    cfg.rtrl_lr = 0.010;
    cfg.bank_capacity = 0;
    cfg.obs_bank_capacity = 0;
    cfg.ph_threshold = 99.0;

    let mut core = MicroContinualCore::new(cfg);
    core.encoder.freeze_to_i8();
    let mut rng = splitmix64(seed ^ 0xBA7E_509E);
    let n_train = 220usize;
    let ep_len = 48usize;
    let mut steps = 0u64;
    let t0 = Instant::now();

    for _ in 0..n_train {
        let ep = gen_wave_episode(&mut rng, 1.0, 1.0, 1.0, ep_len);
        core.start_episode();
        let mut ep_ret = 0.0_f32;
        for t in 0..ep_len {
            let pred = core.decide_vec(&[ep.obs[t]], true, false, false).get().0;
            steps += 1;
            let r = if t >= 6 {
                if pred == ep.targets[t] { 1.0_f32 } else { -1.0_f32 }
            } else {
                0.0_f32
            };
            core.observe_step_reward(r);
            ep_ret += r;
        }
        let _ = core.end_episode(ep_ret, true);
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let (abc, a, b, c, ab) = eval_wave_suite(splitmix64(seed ^ 0xE7A1_0001), |rst, x| {
        if rst {
            core.start_episode();
        }
        core.decide_vec(&[x], false, false, false).get().0
    });
    core.start_episode();
    let p_seu = core.decide_vec(&[f32::NAN], false, true, true);
    let seu_safe = p_seu.get().1.is_finite()
        && core.belief.state.iter().all(|v| v.is_finite())
        && core.rtrl_state.b.iter().all(|v| v.is_finite());

    WaveTransferMetrics {
        abc,
        only_a: a,
        only_b: b,
        only_c: c,
        ab,
        ram_kb: core.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (steps.max(1) as f64),
        seu_safe,
    }
}

fn run_wave_lru(seed: u64) -> WaveTransferMetrics {
    let mut lru = RtrrlLruBaseline::new(1, 42, 2, seed);
    lru.gamma = 0.0;
    lru.trace_lambda = 0.0;
    lru.actor_lr = 0.075;
    lru.critic_lr = 0.14;
    lru.rtrl_lr = 0.010;
    lru.entropy_beta = 0.003;

    let mut rng = splitmix64(seed ^ 0xBA7E_509E);
    let n_train = 220usize;
    let ep_len = 48usize;
    let mut steps = 0u64;
    let t0 = Instant::now();

    for _ in 0..n_train {
        let ep = gen_wave_episode(&mut rng, 1.0, 1.0, 1.0, ep_len);
        lru.start_episode();
        for t in 0..ep_len {
            let pred = lru.decide_vec(&[ep.obs[t]], true, false, false);
            steps += 1;
            let r = if t >= 6 {
                if pred == ep.targets[t] { 1.0_f32 } else { -1.0_f32 }
            } else {
                0.0_f32
            };
            lru.observe_step_reward(r);
        }
        lru.end_episode(true);
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let (abc, a, b, c, ab) = eval_wave_suite(splitmix64(seed ^ 0xE7A1_0001), |rst, x| {
        if rst {
            lru.start_episode();
        }
        lru.decide_vec(&[x], false, false, false)
    });
    let mut probe = RtrrlLruBaseline::new(1, 42, 2, seed);
    let _ = probe.decide_vec(&[0.0], false, true, true);

    WaveTransferMetrics {
        abc,
        only_a: a,
        only_b: b,
        only_c: c,
        ab,
        ram_kb: lru.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (steps.max(1) as f64),
        seu_safe: !probe.had_nan,
    }
}

fn run_wave_gru(seed: u64) -> WaveTransferMetrics {
    let mut gru = OnlineGruBaseline::new(1, 32, 2, seed);
    gru.lr = 0.008;
    gru.gamma = 0.0;

    let mut rng = splitmix64(seed ^ 0xBA7E_509E);
    let n_train = 220usize;
    let ep_len = 48usize;
    let mut steps = 0u64;
    let t0 = Instant::now();

    for _ in 0..n_train {
        let ep = gen_wave_episode(&mut rng, 1.0, 1.0, 1.0, ep_len);
        gru.start_episode();
        let mut ep_ret = 0.0_f32;
        for t in 0..ep_len {
            let pred = gru.decide_vec(&[ep.obs[t]], true, false, false);
            steps += 1;
            let r = if t >= 6 {
                if pred == ep.targets[t] { 1.0_f32 } else { -1.0_f32 }
            } else {
                0.0_f32
            };
            gru.observe_step_reward(r);
            ep_ret += r;
        }
        gru.end_episode(ep_ret, true);
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let (abc, a, b, c, ab) = eval_wave_suite(splitmix64(seed ^ 0xE7A1_0001), |rst, x| {
        if rst {
            gru.start_episode();
        }
        gru.decide_vec(&[x], false, false, false)
    });
    let mut probe = OnlineGruBaseline::new(1, 32, 2, seed);
    let _ = probe.decide_vec(&[0.0], false, true, true);

    WaveTransferMetrics {
        abc,
        only_a: a,
        only_b: b,
        only_c: c,
        ab,
        ram_kb: gru.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (steps.max(1) as f64),
        seu_safe: !probe.had_nan,
    }
}

fn run_wave_sign_heuristic(seed: u64) -> WaveTransferMetrics {
    let (abc, a, b, c, ab) = eval_wave_suite(splitmix64(seed ^ 0xE7A1_0001), |_rst, x| {
        if x >= 0.0 { 1 } else { 0 }
    });
    WaveTransferMetrics {
        abc,
        only_a: a,
        only_b: b,
        only_c: c,
        ab,
        ram_kb: 0.0,
        us_per_step: 0.01,
        seu_safe: true,
    }
}

fn print_wave_row(name: &str, ms: &[WaveTransferMetrics]) {
    let avg = |f: fn(&WaveTransferMetrics) -> f64| -> f64 {
        ms.iter().map(f).sum::<f64>() / (ms.len().max(1) as f64)
    };
    let zs_vals: Vec<f64> = ms.iter().map(|m| m.mean_zero_shot()).collect();
    let (zsm, zss) = mean_sd(&zs_vals);
    let seu = ms.iter().all(|m| m.seu_safe);
    println!(
        "{:<24} | {:5.1}% | {:5.1}% | {:5.1}% | {:5.1}% | {:5.1}% | {:5.1}±{:3.1}% | {:6.1} KB | {:5.2} µs | {}",
        name,
        avg(|m| m.abc),
        avg(|m| m.only_a),
        avg(|m| m.only_b),
        avg(|m| m.only_c),
        avg(|m| m.ab),
        zsm,
        zss,
        avg(|m| m.ram_kb),
        avg(|m| m.us_per_step),
        if seu { "YES" } else { "NO (NaN)" }
    );
}

fn main() {
    let seeds: Vec<u64> = (1..=10).collect();

    // 1. Long-Horizon T-Maze L=30 и L=50 (шум ±0.35, Drift A -> B -> A)
    for &l in &[30usize, 50usize] {
        print_rl_header(&format!(
            "Long-Horizon T-Maze (L={}, noise=±0.35, Drift A->B->A, sparse)",
            l
        ));
        let r_mic: Vec<_> = seeds.iter().map(|&s| run_tmaze_micro(l, s)).collect();
        let r_lru: Vec<_> = seeds.iter().map(|&s| run_tmaze_lru(l, s)).collect();
        let r_gru: Vec<_> = seeds.iter().map(|&s| run_tmaze_gru(l, s)).collect();
        let r_rnd = vec![RlMetrics {
            score: 50.0,
            retention: 50.0,
            recovery_eps: 120.0,
            ram_kb: 0.0,
            us_per_step: 0.05,
            seu_safe: true,
        }];
        print_rl_row("MicroContinualCore", "%", &r_mic);
        print_rl_row("RTRRL-LRU (AAAI'25)", "%", &r_lru);
        print_rl_row("Online GRU + A2C", "%", &r_gru);
        print_rl_row("Random (floor)", "%", &r_rnd);
    }

    // 2. Deep Delay POPGym RepeatPrevious K=6 и K=8 (pure RL +1/-1)
    for &k in &[6usize, 8usize] {
        print_rl_header(&format!(
            "Deep Delay POPGym RepeatPrevious (K={}, 4 symbols, pure RL +1/-1)",
            k
        ));
        let r_mic: Vec<_> = seeds.iter().map(|&s| run_rp_micro(k, s)).collect();
        let r_h: Vec<_> = seeds.iter().map(|&s| run_rp_hier(k, s)).collect();
        let r_lru: Vec<_> = seeds.iter().map(|&s| run_rp_lru(k, s)).collect();
        let r_gru: Vec<_> = seeds.iter().map(|&s| run_rp_gru(k, s)).collect();
        let r_rnd = vec![RlMetrics {
            score: 25.0,
            retention: 25.0,
            recovery_eps: 220.0,
            ram_kb: 0.0,
            us_per_step: 0.05,
            seu_safe: true,
        }];
        print_rl_row("MicroContinualCore", "%", &r_mic);
        print_rl_row("Hierarchical 2L Cayley", "%", &r_h);
        print_rl_row("RTRRL-LRU (AAAI'25)", "%", &r_lru);
        print_rl_row("Online GRU + A2C", "%", &r_gru);
        print_rl_row("Random (floor)", "%", &r_rnd);
    }

    // 3. Wave-Superposition Zero-Shot Superset Transfer (train ONLY on A+B+C -> freeze -> eval A, B, C, A+B)
    println!("\n=== Wave-Superposition Zero-Shot Transfer (Train A+B+C -> Frozen Eval, 10 seeds) ===");
    println!(
        "{:<24} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6} | {:<10} | {:<9} | {:<8} | {}",
        "Model", "A+B+C", "ZS:A", "ZS:B", "ZS:C", "ZS:A+B", "Mean ZS", "RAM (KB)", "Latency", "SEU"
    );
    println!("{}", "-".repeat(118));
    let w_h: Vec<_> = seeds.iter().map(|&s| run_wave_hier(s)).collect();
    let w_1l: Vec<_> = seeds.iter().map(|&s| run_wave_1l_cayley(s)).collect();
    let w_lru: Vec<_> = seeds.iter().map(|&s| run_wave_lru(s)).collect();
    let w_gru: Vec<_> = seeds.iter().map(|&s| run_wave_gru(s)).collect();
    let w_sgn: Vec<_> = seeds.iter().map(|&s| run_wave_sign_heuristic(s)).collect();
    print_wave_row("Hierarchical 2L Cayley", &w_h);
    print_wave_row("1-Layer Cayley-RTRL", &w_1l);
    print_wave_row("RTRRL-LRU (AAAI'25)", &w_lru);
    print_wave_row("Online GRU + A2C", &w_gru);
    print_wave_row("Sign(x_t) Heuristic", &w_sgn);
}
