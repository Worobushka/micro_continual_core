// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Бенчмарк Шага 3: Стандарты POMDP с памятью и сменой физики (POPGym / Continual RL)
//! 10 сидов, прямое сравнение на одном CPU против `Online GRU + A2C (BPTT + Adam)` и `Random`:
//! 1. Non-Stationary T-Maze (L=5 и L=15, шумный коридор, смена правил A -> B -> A, награда только в конце!)
//! 2. POPGym RepeatPrevious (K=2 и K=4, 4 символа, чистое RL по скалярной награде +1/-1!)
//! 3. POPGym Position-Only CartPole (скорости x_dot, theta_dot скрыты!) + резкая инверсия привода A -> B -> A!

use std::time::Instant;

use micro_continual_core::baselines::{OnlineGruBaseline, RtrrlLruBaseline};
use micro_continual_core::math::{rand_signed_f32, splitmix64};
use micro_continual_core::{
    AlgebraicSelectiveCayley, HierarchicalCayleyRtrl, MicroContinualCore, MicroCoreConfig,
};

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
        "{:<28} | {:6.1}±{:4.1}{:<2} | {:6.1}{:<2} | {:8.1} ep | {:7.1} KB | {:7.2} µs | {}",
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
// 1. Non-Stationary T-Maze (Bakker / BSuite Memory Length + дрейф A -> B -> A)
// =============================================================================

fn run_tmaze_micro(corridor_len: usize, novelty_gate: f32, seed: u64) -> RlMetrics {
    let noise_amp = if corridor_len >= 10 { 0.35_f32 } else { 0.06_f32 };
    let mut cfg = MicroCoreConfig::numeric_stream(2, seed);
    cfg.belief_lambda = 0.94;
    cfg.belief_gamma = 0.65;
    cfg.rotor_norm = 0.12;
    cfg.novelty_gate = novelty_gate;
    cfg.trace_capacity = 64;
    cfg.linear_lr = 0.20;
    cfg.linear_mix = 0.55;
    cfg.bank_capacity = 48;
    cfg.obs_bank_capacity = 16;
    cfg.explore_ucb = 0.35;
    cfg.trace_gamma = 0.96;
    cfg.ph_delta = 0.18;
    cfg.ph_threshold = 2.6;
    cfg.ph_warmup = 30;

    let mut core = MicroContinualCore::new(cfg);
    core.encoder.freeze_to_i8();

    let mut rng = splitmix64(seed ^ 0x7A1E_0001);
    let mut steps = 0u64;
    let t0 = Instant::now();

    let mut run_regime = |c: &mut MicroContinualCore, flipped: bool, n_eps: usize| -> Vec<f64> {
        let mut res = Vec::with_capacity(n_eps);
        for _ in 0..n_eps {
            rng = splitmix64(rng);
            let cue = (rng & 1) as usize;
            let expected = if flipped { 1 - cue } else { cue };
            c.start_episode();

            // Шаг 0: метка cue
            let obs0 = [if cue == 0 { -1.0_f32 } else { 1.0_f32 }, 0.0, 0.0];
            let _ = c.decide_vec(&obs0, true, false, false);
            c.observe_step_reward(0.0);
            steps += 1;

            // Шаги 1..L: пустой шумный коридор (reward = 0.0)
            for _ in 0..corridor_len {
                let nz = rand_signed_f32(&mut rng) * noise_amp;
                let obs_c = [0.0_f32, 0.0, nz];
                let _ = c.decide_vec(&obs_c, true, false, false);
                c.observe_step_reward(0.0);
                steps += 1;
            }

            // Финал: Т-перекрёсток -> выбор выхода (+1 / -1)
            let obs_j = [0.0_f32, 1.0, 0.0];
            let act = c.decide_vec(&obs_j, true, false, false).get().0;
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

fn run_tmaze_gru(corridor_len: usize, seed: u64) -> RlMetrics {
    let noise_amp = if corridor_len >= 10 { 0.35_f32 } else { 0.06_f32 };
    let mut gru = OnlineGruBaseline::new(3, 32, 2, seed);
    gru.lr = 0.008;
    let mut rng = splitmix64(seed ^ 0x7A1E_0001);
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
    let _ = probe.decide_vec(&[0.0, 0.0, 0.0], false, true, true);

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

fn run_tmaze_lru(corridor_len: usize, seed: u64) -> RlMetrics {
    let noise_amp = if corridor_len >= 10 { 0.35_f32 } else { 0.06_f32 };
    let mut lru = RtrrlLruBaseline::new(3, 24, 2, seed);
    lru.gamma = 0.96;
    lru.trace_lambda = 0.85;
    lru.actor_lr = 0.055;
    lru.critic_lr = 0.14;
    lru.rtrl_lr = 0.008;
    lru.entropy_beta = 0.008;
    let mut rng = splitmix64(seed ^ 0x7A1E_0001);
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
    let _ = probe.decide_vec(&[0.0, 0.0, 0.0], false, true, true);

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

fn run_tmaze_alg_sel_cayley(corridor_len: usize, seed: u64) -> RlMetrics {
    let mut m = AlgebraicSelectiveCayley::new(3, 16, 2, true, seed);
    m.enable_rl_mode();
    let mut rng = splitmix64(seed ^ 0x7A1E_0001);
    let noise_amp = if corridor_len >= 10 { 0.35_f32 } else { 0.06_f32 };
    let mut steps = 0u64;
    let t0 = Instant::now();

    let mut run_regime = |ag: &mut AlgebraicSelectiveCayley, flipped: bool, n_eps: usize| -> Vec<f64> {
        let mut res = Vec::with_capacity(n_eps);
        for _ in 0..n_eps {
            rng = splitmix64(rng);
            let cue = (rng & 1) as usize;
            let expected = if flipped { 1 - cue } else { cue };
            ag.reset();

            let cue_val = if cue == 0 { -1.0_f32 } else { 1.0_f32 };
            let _ = ag.decide_rl(&[cue_val, 0.0, 0.0], true);
            ag.observe_step_reward_rl(0.0);
            steps += 1;

            for _ in 0..corridor_len {
                let nz = rand_signed_f32(&mut rng) * noise_amp;
                let _ = ag.decide_rl(&[0.0, 0.0, nz], true);
                ag.observe_step_reward_rl(0.0);
                steps += 1;
            }

            let act = ag.decide_rl(&[0.0, 1.0, 0.0], true);
            steps += 1;
            let ok = act == expected;
            let r = if ok { 1.0_f32 } else { -1.0_f32 };
            ag.end_episode_rl(r, true);
            res.push(if ok { 1.0 } else { 0.0 });
        }
        res
    };

    let ra = run_regime(&mut m, false, 120);
    let rb = run_regime(&mut m, true, 120);
    let rret = run_regime(&mut m, false, 80);
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let mut probe = AlgebraicSelectiveCayley::new(3, 16, 2, true, seed);
    probe.enable_rl_mode();
    probe.inject_register_bit_flip(0, 30);
    probe.inject_weight_bit_flip(0, 30);
    let _ = probe.decide_rl(&[f32::NAN, 1e28, -1e28], false);
    let seu_safe = probe.h_re.iter().all(|v| v.is_finite())
        && probe.h_im.iter().all(|v| v.is_finite())
        && probe.out_w.iter().all(|v| v.is_finite());

    let tail = |v: &[f64]| v[v.len() - 40..].iter().sum::<f64>() * 100.0 / 40.0;
    RlMetrics {
        score: (tail(&ra) + tail(&rb)) * 0.5,
        retention: tail(&rret),
        recovery_eps: recovery_window(&rb, 0.75, 16),
        ram_kb: m.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (steps.max(1) as f64),
        seu_safe,
    }
}

// =============================================================================
// 2. POPGym RepeatPrevious (K = 2, 4, 4 символа, чистое RL по награде +1 / -1)
// =============================================================================

fn run_repeat_prev_micro(k_delay: usize, seed: u64) -> RlMetrics {
    let mut cfg = MicroCoreConfig::numeric_stream(4, seed);
    cfg.standardize_vec = false;
    cfg.linear_wd = 0.0;
    cfg.belief_lambda = 0.86;
    cfg.belief_gamma = 0.0;   // чистая линейная суперпозиция роторных фаз R^k x_{t-k}!
    cfg.rotor_norm = 0.58;    // угол ~60 deg/шаг: R^0..R^4 ортогональны!
    cfg.novelty_gate = 0.0;   // обновляем ротор на каждом такте!
    cfg.linear_lr = 0.16;
    cfg.linear_mix = 1.0;     // чистое Delta-Rule декодирование ротора (~2 µs/шаг)
    cfg.explore_ucb = 0.12;
    cfg.ph_threshold = 99.0;  // стационарный тест рабочей памяти

    let mut core = MicroContinualCore::new(cfg);
    core.encoder.freeze_to_i8();

    let mut rng = splitmix64(seed ^ 0x2E9E_A700);
    let n_eps = 160usize;
    let ep_len = 24usize;
    let mut ep_accs = Vec::with_capacity(n_eps);
    let mut total_steps = 0u64;
    let t0 = Instant::now();

    for ep in 0..n_eps {
        core.start_episode();
        let mut hist = Vec::with_capacity(ep_len);
        let mut hits = 0usize;
        let mut eval_n = 0usize;
        // Мягкое затухание шума исследования по мере обучения
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
                let target = hist[t - k_delay];
                let ok = pred == target;
                if ok {
                    hits += 1;
                }
                eval_n += 1;
                if train_flag {
                    // ЧИСТОЕ RL: только скалярная награда +1/-1 за выбранное действие!
                    let r = if ok { 1.0_f32 } else { -1.0_f32 };
                    core.observe_step_bandit_reward(pred, r);
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
    let rec = recovery_window(&ep_accs, 70.0, 10);
    RlMetrics {
        score: tail,
        retention: tail,
        recovery_eps: rec,
        ram_kb: core.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe,
    }
}

fn run_repeat_prev_rtrl(k_delay: usize, seed: u64) -> RlMetrics {
    let mut cfg = MicroCoreConfig::numeric_stream(4, seed);
    cfg.half_sectors = 16; // 32 замороженных сектора + 24 пластичных RTRL-секторов (~38 KB RAM)
    cfg.standardize_vec = false;
    cfg.belief_lambda = 0.93;
    cfg.belief_gamma = 0.0;
    cfg.rotor_norm = 0.65;
    cfg.novelty_gate = 0.0;
    cfg.trace_gamma = 0.0; // мгновенная пошаговая награда в RepeatPrevious!
    cfg.linear_lr = 0.0;
    cfg.linear_mix = 1.0;
    cfg.linear_wd = 0.0001;
    cfg.actor_critic_lr = 0.14;
    cfg.critic_lr = 0.22;
    cfg.entropy_beta = 0.003;
    cfg.rtrl_sectors = 24;
    cfg.rtrl_max_in = 4;
    cfg.rtrl_lr = 0.015;
    cfg.bank_capacity = 0;
    cfg.obs_bank_capacity = 0;
    cfg.explore_ucb = 0.0;
    cfg.ph_threshold = 99.0;

    let mut core = MicroContinualCore::new(cfg);
    let mut rng = splitmix64(seed ^ 0x2E9E_A700);
    let n_eps = 160usize;
    let ep_len = 24usize;
    let mut ep_accs = Vec::with_capacity(n_eps);
    let mut total_steps = 0u64;
    let t0 = Instant::now();

    for ep in 0..n_eps {
        core.start_episode();
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

            let pred = core.decide_vec(&obs, train_flag, false, false).get().0;
            total_steps += 1;
            if t >= k_delay {
                let target = hist[t - k_delay];
                let ok = pred == target;
                if ok {
                    hits += 1;
                }
                eval_n += 1;
                let r = if ok { 1.0_f32 } else { -1.0_f32 };
                core.observe_step_reward(r);
                ep_ret += r;
            } else {
                core.observe_step_reward(0.0);
            }
        }
        let _ = core.end_episode(ep_ret, train_flag);
        ep_accs.push((hits as f64) * 100.0 / (eval_n.max(1) as f64));
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    core.start_episode();
    let p_seu = core.decide_vec(&[f32::NAN, 1e30, 0.0, 0.0], false, true, true);
    let seu_safe = p_seu.get().1.is_finite()
        && core.belief.state.iter().all(|v| v.is_finite())
        && core.rtrl_state.b.iter().all(|v| v.is_finite());

    let tail = ep_accs[ep_accs.len() - 30..].iter().sum::<f64>() / 30.0;
    let rec = recovery_window(&ep_accs, 70.0, 10);
    RlMetrics {
        score: tail,
        retention: tail,
        recovery_eps: rec,
        ram_kb: core.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe,
    }
}

fn run_repeat_prev_hier(k_delay: usize, seed: u64) -> RlMetrics {
    let mut h = HierarchicalCayleyRtrl::new(4, 16, 16, 4, seed);
    h.gamma = 0.0;
    h.trace_lambda = 0.0;
    h.actor_lr = 0.085;
    h.critic_lr = 0.16;
    h.rtrl_lr = 0.012;
    h.entropy_beta = 0.003;
    let mut rng = splitmix64(seed ^ 0x2E9E_A700);
    let n_eps = 160usize;
    let ep_len = 24usize;
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
                h.observe_step_reward(if ok { 1.0_f32 } else { -1.0_f32 });
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
    let seu_safe =
        h.fast_state.b.iter().all(|v| v.is_finite()) && h.slow_state.b.iter().all(|v| v.is_finite());

    let tail = ep_accs[ep_accs.len() - 30..].iter().sum::<f64>() / 30.0;
    let rec = recovery_window(&ep_accs, 70.0, 10);
    RlMetrics {
        score: tail,
        retention: tail,
        recovery_eps: rec,
        ram_kb: h.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe,
    }
}

fn run_repeat_prev_lru(k_delay: usize, seed: u64) -> RlMetrics {
    let mut lru = RtrrlLruBaseline::new(4, 48, 4, seed);
    lru.gamma = 0.0; // мгновенная пошаговая награда в RepeatPrevious!
    lru.trace_lambda = 0.0;
    lru.actor_lr = 0.08;
    lru.critic_lr = 0.16;
    lru.rtrl_lr = 0.012;
    lru.entropy_beta = 0.004;
    let mut rng = splitmix64(seed ^ 0x2E9E_A700);
    let n_eps = 160usize;
    let ep_len = 24usize;
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
                let r = if ok { 1.0_f32 } else { -1.0_f32 };
                lru.observe_step_reward(r);
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
    let rec = recovery_window(&ep_accs, 70.0, 10);
    RlMetrics {
        score: tail,
        retention: tail,
        recovery_eps: rec,
        ram_kb: lru.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe: !probe.had_nan,
    }
}

fn run_repeat_prev_gru(k_delay: usize, seed: u64) -> RlMetrics {
    let mut gru = OnlineGruBaseline::new(4, 32, 4, seed);
    gru.lr = 0.008;
    gru.gamma = 0.0; // мгновенная пошаговая награда в RepeatPrevious!
    let mut rng = splitmix64(seed ^ 0x2E9E_A700);
    let n_eps = 160usize;
    let ep_len = 24usize;
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
    let rec = recovery_window(&ep_accs, 70.0, 10);
    RlMetrics {
        score: tail,
        retention: tail,
        recovery_eps: rec,
        ram_kb: gru.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe: !probe.had_nan,
    }
}

// =============================================================================
// 3. POPGym Position-Only CartPole (x_dot, theta_dot скрыты!) + инверсия привода A -> B -> A
// =============================================================================

struct CartPoleEnv {
    x: f32,
    x_dot: f32,
    theta: f32,
    theta_dot: f32,
    steps: usize,
    force_sign: f32,
}

impl CartPoleEnv {
    fn reset(rng: &mut u64, force_sign: f32) -> Self {
        Self {
            x: rand_signed_f32(rng) * 0.05,
            x_dot: rand_signed_f32(rng) * 0.05,
            theta: rand_signed_f32(rng) * 0.05,
            theta_dot: rand_signed_f32(rng) * 0.05,
            steps: 0,
            force_sign,
        }
    }

    /// Position-Only POMDP: агент видит ТОЛЬКО (x, theta) — скорости скрыты!
    fn obs(&self) -> [f32; 2] {
        [self.x / 2.4, self.theta / 0.2095]
    }

    fn step(&mut self, action: usize) -> bool {
        let g = 9.8_f32;
        let mass_cart = 1.0_f32;
        let mass_pole = 0.1_f32;
        let total_mass = mass_cart + mass_pole;
        let length = 0.5_f32;
        let polemass_length = mass_pole * length;
        let force_mag = 10.0_f32;
        let tau = 0.02_f32;

        let force = (if action == 1 { force_mag } else { -force_mag }) * self.force_sign;
        let costh = self.theta.cos();
        let sinth = self.theta.sin();

        let temp = (force + polemass_length * self.theta_dot * self.theta_dot * sinth) / total_mass;
        let theta_acc = (g * sinth - costh * temp)
            / (length * (4.0 / 3.0 - mass_pole * costh * costh / total_mass));
        let x_acc = temp - polemass_length * theta_acc * costh / total_mass;

        self.x += tau * self.x_dot;
        self.x_dot += tau * x_acc;
        self.theta += tau * self.theta_dot;
        self.theta_dot += tau * theta_acc;
        self.steps += 1;

        self.x.abs() > 2.4 || self.theta.abs() > 0.2095 || self.steps >= 200
    }
}

fn run_cartpole_rtrl(seed: u64, use_delta: bool) -> RlMetrics {
    let mut cfg = MicroCoreConfig::numeric_stream(2, seed);
    cfg.half_sectors = 16; // 32 замороженных (96D) + 16 пластичных RTRL (48D) -> ~24 KB RAM!
    cfg.belief_lambda = 0.60;
    cfg.belief_gamma = 0.30;
    cfg.rotor_norm = 0.14;
    cfg.novelty_gate = 0.0;
    cfg.standardize_vec = false; // ВАЖНО: сохраняем точную нулевую точку (x=0, theta=0) при долгом балансе!
    cfg.delta_obs_scale = if use_delta { 10.0 } else { 0.0 };
    cfg.trace_capacity = 0; // онлайн TD(lambda) — длинный буфер эпизода не нужен!
    cfg.trace_gamma = 0.95;
    cfg.linear_lr = 0.0;
    cfg.linear_mix = 1.0;
    cfg.linear_wd = 0.0001;
    cfg.actor_critic_lr = 0.055;
    cfg.critic_lr = 0.14;
    cfg.entropy_beta = 0.008;
    // Половина 2: 16 пластичных секторов точного блочно-диагонального Cayley-RTRL + RMS (~1.3 KB/банк!)
    cfg.rtrl_sectors = 16;
    cfg.rtrl_max_in = if use_delta { 4 } else { 2 };
    cfg.rtrl_lr = 0.006;
    cfg.bank_capacity = 0;
    cfg.obs_bank_capacity = 0;
    cfg.explore_ucb = 0.0;
    cfg.ph_delta = 0.12;
    cfg.ph_threshold = 1.8;
    cfg.ph_warmup = 28;

    let mut core = MicroContinualCore::new(cfg);
    core.encoder.freeze_to_i8();

    let mut rng = splitmix64(seed ^ 0xCA27_901E);
    let mut total_steps = 0u64;
    let t0 = Instant::now();

    let mut run_regime = |c: &mut MicroContinualCore, sign: f32, n_eps: usize| -> Vec<f64> {
        let mut surv = Vec::with_capacity(n_eps);
        for _ in 0..n_eps {
            let mut env = CartPoleEnv::reset(&mut rng, sign);
            c.start_episode();
            loop {
                let obs = env.obs();
                let act = c.decide_vec(&obs, true, false, false).get().0;
                total_steps += 1;
                let done = env.step(act);
                let step_r = if done && env.steps < 200 { -1.0_f32 } else { 0.06_f32 };
                c.observe_step_reward(step_r);
                if done {
                    break;
                }
            }
            let norm_ret = ((env.steps as f32) / 70.0).clamp(0.0, 1.0) * 2.0 - 1.0;
            let _ = c.end_episode_truncated(norm_ret, env.steps >= 200, true);
            surv.push(env.steps as f64);
        }
        surv
    };

    let sa = run_regime(&mut core, 1.0, 140);
    let sb = run_regime(&mut core, -1.0, 140);
    let sret = run_regime(&mut core, 1.0, 80);
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    core.start_episode();
    let p_seu = core.decide_vec(&[f32::NAN, 1e30], false, true, true);
    let seu_safe = p_seu.get().1.is_finite()
        && core.belief.state.iter().all(|v| v.is_finite())
        && core.rtrl_state.b.iter().all(|v| v.is_finite());

    let tail = |v: &[f64]| v[v.len() - 40..].iter().sum::<f64>() / 40.0;
    RlMetrics {
        score: (tail(&sa) + tail(&sb)) * 0.5,
        retention: tail(&sret),
        recovery_eps: recovery_window(&sb, 50.0, 15),
        ram_kb: core.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe,
    }
}

fn run_cartpole_micro(seed: u64) -> RlMetrics {
    let mut cfg = MicroCoreConfig::numeric_stream(2, seed);
    cfg.belief_lambda = 0.55;
    cfg.belief_gamma = 0.45;
    cfg.rotor_norm = 0.14;
    cfg.novelty_gate = 0.0;
    cfg.standardize_vec = false; // сохраняем нулевую точку (x=0, theta=0)!
    cfg.delta_obs_scale = 8.0;
    cfg.trace_capacity = 128;
    cfg.trace_gamma = 0.96;
    cfg.linear_lr = 0.0;
    cfg.linear_mix = 1.0;
    cfg.linear_wd = 0.0003;
    cfg.actor_critic_lr = 0.12;
    cfg.critic_lr = 0.16;
    cfg.entropy_beta = 0.012;
    cfg.bank_capacity = 32;
    cfg.obs_bank_capacity = 16;
    cfg.explore_ucb = 0.0;
    cfg.ph_delta = 0.12;
    cfg.ph_threshold = 1.8;
    cfg.ph_warmup = 28;

    let mut core = MicroContinualCore::new(cfg);
    core.encoder.freeze_to_i8();

    let mut rng = splitmix64(seed ^ 0xCA27_901E);
    let mut total_steps = 0u64;
    let t0 = Instant::now();

    let mut run_regime = |c: &mut MicroContinualCore, sign: f32, n_eps: usize| -> Vec<f64> {
        let mut surv = Vec::with_capacity(n_eps);
        for _ in 0..n_eps {
            let mut env = CartPoleEnv::reset(&mut rng, sign);
            c.start_episode();
            loop {
                let obs = env.obs();
                let act = c.decide_vec(&obs, true, false, false).get().0;
                total_steps += 1;
                let done = env.step(act);
                let step_r = if done && env.steps < 200 { -1.0_f32 } else { 0.06_f32 };
                c.observe_step_reward(step_r);
                if done {
                    break;
                }
            }
            let norm_ret = ((env.steps as f32) / 70.0).clamp(0.0, 1.0) * 2.0 - 1.0;
            let _ = c.end_episode(norm_ret, true);
            surv.push(env.steps as f64);
        }
        surv
    };

    let sa = run_regime(&mut core, 1.0, 140);
    let sb = run_regime(&mut core, -1.0, 140);
    let sret = run_regime(&mut core, 1.0, 80);
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    core.start_episode();
    let p_seu = core.decide_vec(&[f32::NAN, 1e30], false, true, true);
    let seu_safe = p_seu.get().1.is_finite() && core.belief.state.iter().all(|v| v.is_finite());

    let tail = |v: &[f64]| v[v.len() - 40..].iter().sum::<f64>() / 40.0;
    RlMetrics {
        score: (tail(&sa) + tail(&sb)) * 0.5,
        retention: tail(&sret),
        recovery_eps: recovery_window(&sb, 50.0, 15),
        ram_kb: core.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe,
    }
}

fn run_cartpole_gru(seed: u64, use_delta: bool) -> RlMetrics {
    let in_dim = if use_delta { 4 } else { 2 };
    let mut gru = OnlineGruBaseline::new(in_dim, 32, 2, seed);
    gru.lr = 0.006;
    gru.gamma = 0.95;
    let mut rng = splitmix64(seed ^ 0xCA27_901E);
    let mut total_steps = 0u64;
    let t0 = Instant::now();

    let mut run_regime = |g: &mut OnlineGruBaseline, sign: f32, n_eps: usize| -> Vec<f64> {
        let mut surv = Vec::with_capacity(n_eps);
        for _ in 0..n_eps {
            let mut env = CartPoleEnv::reset(&mut rng, sign);
            g.start_episode();
            let mut ep_r = 0.0_f32;
            let mut prev = [0.0_f32; 2];
            let mut has_prev = false;
            loop {
                let obs = env.obs();
                let act = if use_delta {
                    let dx0 = if has_prev { (obs[0] - prev[0]) * 10.0 } else { 0.0 };
                    let dx1 = if has_prev { (obs[1] - prev[1]) * 10.0 } else { 0.0 };
                    prev = obs;
                    has_prev = true;
                    g.decide_vec(&[obs[0], obs[1], dx0.clamp(-4.0, 4.0), dx1.clamp(-4.0, 4.0)], true, false, false)
                } else {
                    g.decide_vec(&obs, true, false, false)
                };
                total_steps += 1;
                let done = env.step(act);
                let step_r = if done && env.steps < 200 { -1.0_f32 } else { 0.06_f32 };
                g.observe_step_reward(step_r);
                ep_r += step_r;
                if done {
                    break;
                }
            }
            g.end_episode_truncated(ep_r, env.steps >= 200, true);
            surv.push(env.steps as f64);
        }
        surv
    };

    let sa = run_regime(&mut gru, 1.0, 140);
    let sb = run_regime(&mut gru, -1.0, 140);
    let sret = run_regime(&mut gru, 1.0, 80);
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let mut probe = OnlineGruBaseline::new(in_dim, 32, 2, seed);
    let _ = probe.decide_vec(&vec![0.0; in_dim], false, true, true);

    let tail = |v: &[f64]| v[v.len() - 40..].iter().sum::<f64>() / 40.0;
    RlMetrics {
        score: (tail(&sa) + tail(&sb)) * 0.5,
        retention: tail(&sret),
        recovery_eps: recovery_window(&sb, 50.0, 15),
        ram_kb: gru.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe: !probe.had_nan,
    }
}

fn run_cartpole_lru(seed: u64, use_delta: bool) -> RlMetrics {
    let in_dim = if use_delta { 4 } else { 2 };
    let mut lru = RtrrlLruBaseline::new(in_dim, 48, 2, seed);
    lru.gamma = 0.95;
    lru.trace_lambda = 0.80;
    lru.actor_lr = 0.055;
    lru.critic_lr = 0.14;
    lru.rtrl_lr = 0.006;
    lru.entropy_beta = 0.008;
    let mut rng = splitmix64(seed ^ 0xCA27_901E);
    let mut total_steps = 0u64;
    let t0 = Instant::now();

    let mut run_regime = |m: &mut RtrrlLruBaseline, sign: f32, n_eps: usize| -> Vec<f64> {
        let mut surv = Vec::with_capacity(n_eps);
        for _ in 0..n_eps {
            let mut env = CartPoleEnv::reset(&mut rng, sign);
            m.start_episode();
            let mut prev = [0.0_f32; 2];
            let mut has_prev = false;
            loop {
                let obs = env.obs();
                let act = if use_delta {
                    let dx0 = if has_prev { (obs[0] - prev[0]) * 10.0 } else { 0.0 };
                    let dx1 = if has_prev { (obs[1] - prev[1]) * 10.0 } else { 0.0 };
                    prev = obs;
                    has_prev = true;
                    m.decide_vec(&[obs[0], obs[1], dx0.clamp(-4.0, 4.0), dx1.clamp(-4.0, 4.0)], true, false, false)
                } else {
                    m.decide_vec(&obs, true, false, false)
                };
                total_steps += 1;
                let done = env.step(act);
                let step_r = if done && env.steps < 200 { -1.0_f32 } else { 0.06_f32 };
                m.observe_step_reward(step_r);
                if done {
                    break;
                }
            }
            m.end_episode_truncated(true, env.steps >= 200);
            surv.push(env.steps as f64);
        }
        surv
    };

    let sa = run_regime(&mut lru, 1.0, 140);
    let sb = run_regime(&mut lru, -1.0, 140);
    let sret = run_regime(&mut lru, 1.0, 80);
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let mut probe = RtrrlLruBaseline::new(in_dim, 48, 2, seed);
    let _ = probe.decide_vec(&vec![0.0; in_dim], false, true, true);

    let tail = |v: &[f64]| v[v.len() - 40..].iter().sum::<f64>() / 40.0;
    RlMetrics {
        score: (tail(&sa) + tail(&sb)) * 0.5,
        retention: tail(&sret),
        recovery_eps: recovery_window(&sb, 50.0, 15),
        ram_kb: lru.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe: !probe.had_nan,
    }
}

fn run_cartpole_random(seed: u64) -> RlMetrics {
    let mut rng = splitmix64(seed ^ 0xCA27_901E);
    let mut sum = 0.0_f64;
    for _ in 0..100 {
        let mut env = CartPoleEnv::reset(&mut rng, 1.0);
        loop {
            rng = splitmix64(rng);
            if env.step((rng & 1) as usize) {
                break;
            }
        }
        sum += env.steps as f64;
    }
    let avg = sum / 100.0;
    RlMetrics {
        score: avg,
        retention: avg,
        recovery_eps: 140.0,
        ram_kb: 0.0,
        us_per_step: 0.05,
        seu_safe: true,
    }
}

fn print_header(title: &str) {
    println!("\n=== {} (10 seeds) ===", title);
    println!(
        "{:<28} | {:<15} | {:<8} | {:<11} | {:<10} | {:<10} | {}",
        "Model", "Score(mean±sd)", "Retent.", "Recovery", "RAM (KB)", "Latency", "SEU-Safe"
    );
    println!("{}", "-".repeat(106));
}

fn main() {
    let seeds: Vec<u64> = (1..=10).collect();

    // 1. Non-Stationary T-Maze (L=5 и L=15)
    for &l in &[5usize, 15usize] {
        let noise_amp = if l >= 10 { 0.35_f32 } else { 0.06_f32 };
        print_header(&format!(
            "Non-Stationary T-Maze (L={}, noise=±{:.2}, Drift A->B->A, sparse)",
            l, noise_amp
        ));
        let r_mic: Vec<_> = seeds.iter().map(|&s| run_tmaze_micro(l, 0.0, s)).collect();
        let r_asc: Vec<_> = seeds.iter().map(|&s| run_tmaze_alg_sel_cayley(l, s)).collect();
        let r_mic_gate: Vec<_> = seeds.iter().map(|&s| run_tmaze_micro(l, 0.28, s)).collect();
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
        print_rl_row("MicroContinualCore (ng=0.0)", "%", &r_mic);
        print_rl_row("Alg.Sel.Cayley (RL, ng=0.0)", "%", &r_asc);
        print_rl_row("MicroCore (+gate=0.28 Abl.)", "%", &r_mic_gate);
        print_rl_row("RTRRL-LRU (AAAI'25)", "%", &r_lru);
        print_rl_row("Online GRU + A2C", "%", &r_gru);
        print_rl_row("Random (floor)", "%", &r_rnd);
    }
    // 2. POPGym RepeatPrevious (K=2 и K=4)
    for &k in &[2usize, 4usize] {
        print_header(&format!("POPGym RepeatPrevious (K={}, 4 symbols, pure RL +1/-1)", k));
        let r_mic: Vec<_> = seeds.iter().map(|&s| run_repeat_prev_micro(k, s)).collect();
        let r_rtrl: Vec<_> = seeds.iter().map(|&s| run_repeat_prev_rtrl(k, s)).collect();
        let r_hier: Vec<_> = seeds.iter().map(|&s| run_repeat_prev_hier(k, s)).collect();
        let r_lru: Vec<_> = seeds.iter().map(|&s| run_repeat_prev_lru(k, s)).collect();
        let r_gru: Vec<_> = seeds.iter().map(|&s| run_repeat_prev_gru(k, s)).collect();
        let r_rnd = vec![RlMetrics { score: 25.0, retention: 25.0, recovery_eps: 160.0, ram_kb: 0.0, us_per_step: 0.05, seu_safe: true }];
        print_rl_row("MicroContinualCore", "%", &r_mic);
        print_rl_row("Split-Band Cayley-RTRL", "%", &r_rtrl);
        print_rl_row("Hierarchical 2L Cayley", "%", &r_hier);
        print_rl_row("RTRRL-LRU (AAAI'25)", "%", &r_lru);
        print_rl_row("Online GRU + A2C", "%", &r_gru);
        print_rl_row("Random (floor)", "%", &r_rnd);
    }

    // 3. POPGym Position-Only CartPole + инверсия привода A->B->A
    print_header("POPGym Position-Only CartPole + Actuator Flip A->B->A (steps / 200)");
    let r_mic: Vec<_> = seeds.iter().map(|&s| run_cartpole_micro(s)).collect();
    let r_gru_pure: Vec<_> = seeds.iter().map(|&s| run_cartpole_gru(s, false)).collect();
    let r_gru_dx: Vec<_> = seeds.iter().map(|&s| run_cartpole_gru(s, true)).collect();
    let r_lru_pure: Vec<_> = seeds.iter().map(|&s| run_cartpole_lru(s, false)).collect();
    let r_lru_dx: Vec<_> = seeds.iter().map(|&s| run_cartpole_lru(s, true)).collect();
    let r_rtrl_pure: Vec<_> = seeds.iter().map(|&s| run_cartpole_rtrl(s, false)).collect();
    let r_rtrl_dx: Vec<_> = seeds.iter().map(|&s| run_cartpole_rtrl(s, true)).collect();
    let r_rnd: Vec<_> = seeds.iter().map(|&s| run_cartpole_random(s)).collect();
    print_rl_row("Cayley-RTRL (pure 2D)", "st", &r_rtrl_pure);
    print_rl_row("Cayley-RTRL (+dx)", "st", &r_rtrl_dx);
    print_rl_row("RTRRL-LRU (pure 2D)", "st", &r_lru_pure);
    print_rl_row("RTRRL-LRU (+dx)", "st", &r_lru_dx);
    print_rl_row("Online GRU (pure 2D)", "st", &r_gru_pure);
    print_rl_row("Online GRU (+dx)", "st", &r_gru_dx);
    print_rl_row("MicroCore (Branch B)", "st", &r_mic);
    print_rl_row("Random (floor)", "st", &r_rnd);
}