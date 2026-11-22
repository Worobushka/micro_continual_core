// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Честная абляция единой архитектуры `MicroContinualCore v0.4.0`:
//! Изолирует точный числовой вклад каждого нового механизма ВНУТРИ ОДНОЙ И ТОЙ ЖЕ структуры `MicroContinualCore`:
//!   [A0] Frozen Rotor Only (без RTRL: `rtrl_sectors = 0`)
//!   [A1] + 1L Cayley-RTRL (Замороженный резервуар: `rtrl_lr = 0.0`, без градиентного обучения RTRL)
//!   [A2] + 1L Cayley-RTRL (Обучаемый, сопряжённая сетка `rtrl_grid_mode = 0`)
//!   [A3] + 1L Cayley-RTRL (Обучаемый, развязанная сетка `частота x добротность Q`, `rtrl_grid_mode = 1`)
//!   [A3+Skip] + 1L Cayley-RTRL (Обучаемый, развязанная сетка + сквозной skip-x вход)
//!   [A4] Unified 2L Fast-Slow Cayley-RTRL + PH Banks (`rtrl_slow_sectors > 0`, `rtrl_use_skip_x = true`)
//!
//! Задачи:
//! 1. POPGym RepeatPrevious (K=4 и K=6, 4 символа, чистое RL +1/-1)
//! 2. POPGym Position-Only CartPole + инверсия привода A -> B -> A (чистый 2D и +dx)
//! 3. Wave-Superposition Zero-Shot Transfer (обучение на A+B+C -> замороженная проверка A, B, C, A+B)
//! 4. Continual Wave-Superposition Flip (смена знака фазы Normal -> Inverted -> Normal, проверка банков PageHinkley)
//! 5. Electricity ELEC2 Temporal (опционально, если найден файл датасета)

use std::path::Path;
use std::time::Instant;

use micro_continual_core::baselines::OnlineGruBaseline;
use micro_continual_core::math::{rand_signed_f32, rand_unit_f32, splitmix64};
use micro_continual_core::rtrl::PlasticCayleyParams;
use micro_continual_core::stream_baselines::NoChangePersistent;
use micro_continual_core::{MicroContinualCore, MicroCoreConfig};

#[derive(Clone, Copy)]
enum Stage {
    A0NoRtrl,
    A1RtrlFrozen,
    A2RtrlTrainedCpl,
    A3RtrlDecoupledFrozen,
    A3RtrlDecoupled,
    A3RtrlSkip,
    A4Unified2L,
}

impl Stage {
    fn all() -> [Stage; 7] {
        [
            Stage::A0NoRtrl,
            Stage::A1RtrlFrozen,
            Stage::A2RtrlTrainedCpl,
            Stage::A3RtrlDecoupledFrozen,
            Stage::A3RtrlDecoupled,
            Stage::A3RtrlSkip,
            Stage::A4Unified2L,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            Stage::A0NoRtrl => "[A0] Frozen Rotor Only (no RTRL)",
            Stage::A1RtrlFrozen => "[A1] +1L RTRL Frozen (rtrl_lr=0)",
            Stage::A2RtrlTrainedCpl => "[A2] +1L RTRL Trained (coupled)",
            Stage::A3RtrlDecoupledFrozen => "[A3-Frozen] +1L Frozen (dec Qxw)",
            Stage::A3RtrlDecoupled => "[A3] +1L RTRL Trained (dec Qxw)",
            Stage::A3RtrlSkip => "[A3+Skip] +1L RTRL + SkipX",
            Stage::A4Unified2L => "[A4] Unified 2L Fast-Slow + PH",
        }
    }
}

fn log_rtrl_spectrum_diff(title: &str, init_p: &PlasticCayleyParams, learned_p: &PlasticCayleyParams) {
    let s_cnt = init_p.sectors.min(learned_p.sectors);
    let m_in = init_p.max_in.min(learned_p.max_in);
    if s_cnt == 0 {
        return;
    }
    println!("\n--- Learned Spectrum & Parameter Shift: {} (S={}, in={}) ---", title, s_cnt, m_in);
    println!(
        "{:<4} | {:<18} | {:<18} | {:<15} | {:<17} | {}",
        "Sec", "alpha (0 -> trn)", "omega (0 -> trn)", "lambda (0->t)", "T_steps (0 -> t)", "||W_in||_2 (0->t)"
    );
    println!("{}", "-".repeat(100));

    let mut sum_da = 0.0_f64;
    let mut max_da = 0.0_f64;
    let mut sum_dw = 0.0_f64;
    let mut max_dw = 0.0_f64;
    let mut sum_dt = 0.0_f64;
    let mut max_dt = 0.0_f64;
    let mut dwin_f2 = 0.0_f64;

    let period = |w: f32| -> f64 {
        let th = (w.abs() as f64).atan().max(1e-3);
        std::f64::consts::PI / th
    };
    let lambda_val = |a: f32| -> f64 {
        0.995_f64 / (1.0 + (a as f64) * (a as f64))
    };

    for s in 0..s_cnt {
        let a0 = init_p.alpha[s];
        let a1 = learned_p.alpha[s];
        let w0 = init_p.omega[s];
        let w1 = learned_p.omega[s];
        let t0 = period(w0);
        let t1 = period(w1);
        let l0 = lambda_val(a0);
        let l1 = lambda_val(a1);

        let mut wn0 = 0.0_f64;
        let mut wn1 = 0.0_f64;
        for j in 0..m_in {
            let v0 = init_p.in_w[s * m_in + j] as f64;
            let v1 = learned_p.in_w[s * m_in + j] as f64;
            wn0 += v0 * v0;
            wn1 += v1 * v1;
            dwin_f2 += (v1 - v0) * (v1 - v0);
        }
        wn0 = wn0.sqrt();
        wn1 = wn1.sqrt();

        let da = (a1 as f64 - a0 as f64).abs();
        let dw = (w1 as f64 - w0 as f64).abs();
        let dt = (t1 - t0).abs();
        sum_da += da;
        if da > max_da { max_da = da; }
        sum_dw += dw;
        if dw > max_dw { max_dw = dw; }
        sum_dt += dt;
        if dt > max_dt { max_dt = dt; }

        if s < 8 || s == s_cnt - 1 {
            println!(
                "s{:<3} | {:6.3} -> {:6.3}   | {:+6.3} -> {:+6.3}   | {:.3}->{:.3}    | {:6.1} -> {:6.1}  | {:.3} -> {:.3}",
                s, a0, a1, w0, w1, l0, l1, t0.min(999.9), t1.min(999.9), wn0, wn1
            );
        }
    }
    println!(
        "Aggregate shift: mean|dAlpha|={:.4} (max {:.4}), mean|dOmega|={:.4} (max {:.4}), mean|dT|={:.2} (max {:.2}), ||dW_in||_F={:.4}",
        sum_da / (s_cnt as f64),
        max_da,
        sum_dw / (s_cnt as f64),
        max_dw,
        sum_dt / (s_cnt as f64),
        max_dt,
        dwin_f2.sqrt()
    );
}

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
        "{:<34} | {:<15} | {:<8} | {:<11} | {:<9} | {:<9} | {}",
        "Stage / Architecture", "Score(mean±sd)", "Retent.", "Recovery", "RAM (KB)", "Latency", "SEU"
    );
    println!("{}", "-".repeat(108));
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
        "{:<34} | {:6.1}±{:4.1}{:<2} | {:6.1}{:<2} | {:8.1} ep | {:6.1} KB | {:6.2} µs | {}",
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
// 1. RepeatPrevious Ablation (K = 4, 6)
// =============================================================================

fn cfg_repeat_prev(stage: Stage, seed: u64) -> MicroCoreConfig {
    let mut cfg = MicroCoreConfig::numeric_stream(4, seed);
    cfg.standardize_vec = false;
    cfg.belief_lambda = 0.90;
    cfg.belief_gamma = 0.0;
    cfg.rotor_norm = 0.60;
    cfg.novelty_gate = 0.0;
    cfg.trace_gamma = 0.0;
    cfg.linear_wd = 0.0001;
    cfg.bank_capacity = 0;
    cfg.obs_bank_capacity = 0;
    cfg.explore_ucb = 0.0;
    cfg.ph_threshold = 99.0;
    cfg.linear_mix = 1.0;

    match stage {
        Stage::A0NoRtrl => {
            cfg.half_sectors = 20;
            cfg.linear_lr = 0.14;
            cfg.explore_ucb = 0.10;
            cfg.actor_critic_lr = 0.0;
            cfg.rtrl_sectors = 0;
        }
        Stage::A1RtrlFrozen => {
            cfg.half_sectors = 12;
            cfg.linear_lr = 0.0;
            cfg.actor_critic_lr = 0.09;
            cfg.critic_lr = 0.16;
            cfg.entropy_beta = 0.003;
            cfg.rtrl_sectors = 16;
            cfg.rtrl_max_in = 4;
            cfg.rtrl_lr = 0.0;
            cfg.rtrl_grid_mode = 0;
            cfg.rtrl_kappa = 0.65;
        }
        Stage::A2RtrlTrainedCpl => {
            cfg.half_sectors = 12;
            cfg.linear_lr = 0.0;
            cfg.actor_critic_lr = 0.09;
            cfg.critic_lr = 0.16;
            cfg.entropy_beta = 0.003;
            cfg.rtrl_sectors = 16;
            cfg.rtrl_max_in = 4;
            cfg.rtrl_lr = 0.012;
            cfg.rtrl_grid_mode = 0;
            cfg.rtrl_kappa = 0.65;
        }
        Stage::A3RtrlDecoupledFrozen | Stage::A3RtrlDecoupled | Stage::A3RtrlSkip => {
            cfg.half_sectors = 12;
            cfg.linear_lr = 0.0;
            cfg.actor_critic_lr = 0.09;
            cfg.critic_lr = 0.16;
            cfg.entropy_beta = 0.003;
            cfg.rtrl_sectors = 16;
            cfg.rtrl_max_in = 4;
            cfg.rtrl_lr = if matches!(stage, Stage::A3RtrlDecoupledFrozen) { 0.0 } else { 0.012 };
            cfg.rtrl_grid_mode = 1;
            cfg.rtrl_use_skip_x = matches!(stage, Stage::A3RtrlSkip);
            cfg.rtrl_kappa = 0.65;
        }
        Stage::A4Unified2L => {
            cfg.half_sectors = 0; // чистая 2-слойная волновая память внутри RegimeBank
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
        }
    }
    cfg
}

fn eval_repeat_prev(stage: Stage, k_delay: usize, seed: u64) -> RlMetrics {
    let mut core = MicroContinualCore::new(cfg_repeat_prev(stage, seed));
    core.encoder.freeze_to_i8();
    let mut rng = splitmix64(seed ^ 0x2E9E_AB1A);
    let n_eps = if k_delay >= 6 { 220usize } else { 160usize };
    let ep_len = if k_delay >= 6 { 28usize } else { 24usize };
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
                let ok = pred == hist[t - k_delay];
                if ok {
                    hits += 1;
                }
                eval_n += 1;
                let r = if ok { 1.0_f32 } else { -1.0_f32 };
                if matches!(stage, Stage::A0NoRtrl) {
                    if train_flag {
                        core.observe_step_bandit_reward(pred, r);
                    }
                } else {
                    core.observe_step_reward(r);
                }
                ep_ret += r;
            } else if !matches!(stage, Stage::A0NoRtrl) {
                core.observe_step_reward(0.0);
            }
        }
        if !matches!(stage, Stage::A0NoRtrl) {
            let _ = core.end_episode(ep_ret, train_flag);
        }
        ep_accs.push((hits as f64) * 100.0 / (eval_n.max(1) as f64));
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    core.start_episode();
    let p_seu = core.decide_vec(&[f32::NAN, 1e30, 0.0, 0.0], false, true, true);
    let seu_safe = p_seu.get().1.is_finite()
        && core.belief.state.iter().all(|v| v.is_finite())
        && core.rtrl_state.b.iter().all(|v| v.is_finite())
        && core.rtrl_slow_state.b.iter().all(|v| v.is_finite());

    let tail = ep_accs[ep_accs.len() - 30..].iter().sum::<f64>() / 30.0;
    RlMetrics {
        score: tail,
        retention: tail,
        recovery_eps: recovery_window(&ep_accs, 70.0, 10),
        ram_kb: core.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (total_steps.max(1) as f64),
        seu_safe,
    }
}

// =============================================================================
// 2. Position-Only CartPole + Actuator Flip A -> B -> A Ablation
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

fn cfg_cartpole(stage: Stage, use_delta: bool, seed: u64) -> MicroCoreConfig {
    let mut cfg = MicroCoreConfig::numeric_stream(2, seed);
    cfg.belief_lambda = 0.60;
    cfg.belief_gamma = 0.30;
    cfg.rotor_norm = 0.14;
    cfg.novelty_gate = 0.0;
    cfg.standardize_vec = false;
    cfg.delta_obs_scale = if use_delta { 10.0 } else { 0.0 };
    cfg.trace_gamma = 0.95;
    cfg.linear_lr = 0.0;
    cfg.linear_mix = 1.0;
    cfg.linear_wd = 0.0001;
    cfg.actor_critic_lr = 0.055;
    cfg.critic_lr = 0.14;
    cfg.entropy_beta = 0.008;
    cfg.bank_capacity = 0;
    cfg.obs_bank_capacity = 0;
    cfg.explore_ucb = 0.0;
    cfg.ph_delta = 0.12;
    cfg.ph_threshold = 1.8;
    cfg.ph_warmup = 28;
    cfg.max_regimes = 4;

    match stage {
        Stage::A0NoRtrl => {
            cfg.half_sectors = 20;
            cfg.trace_capacity = 128;
            cfg.actor_critic_lr = 0.12;
            cfg.critic_lr = 0.16;
            cfg.entropy_beta = 0.012;
            cfg.rtrl_sectors = 0;
        }
        Stage::A1RtrlFrozen => {
            cfg.half_sectors = 16;
            cfg.trace_capacity = 0;
            cfg.rtrl_sectors = 16;
            cfg.rtrl_max_in = if use_delta { 4 } else { 2 };
            cfg.rtrl_lr = 0.0;
            cfg.rtrl_grid_mode = 0;
        }
        Stage::A2RtrlTrainedCpl => {
            cfg.half_sectors = 16;
            cfg.trace_capacity = 0;
            cfg.rtrl_sectors = 16;
            cfg.rtrl_max_in = if use_delta { 4 } else { 2 };
            cfg.rtrl_lr = 0.006;
            cfg.rtrl_grid_mode = 0;
        }
        Stage::A3RtrlDecoupledFrozen | Stage::A3RtrlDecoupled | Stage::A3RtrlSkip => {
            cfg.half_sectors = 16;
            cfg.trace_capacity = 0;
            cfg.rtrl_sectors = 16;
            cfg.rtrl_max_in = if use_delta { 4 } else { 2 };
            cfg.rtrl_lr = if matches!(stage, Stage::A3RtrlDecoupledFrozen) { 0.0 } else { 0.006 };
            cfg.rtrl_grid_mode = 1;
            cfg.rtrl_use_skip_x = matches!(stage, Stage::A3RtrlSkip);
        }
        Stage::A4Unified2L => {
            cfg.half_sectors = 16;
            cfg.trace_capacity = 0;
            cfg.rtrl_sectors = 10;
            cfg.rtrl_slow_sectors = 6;
            cfg.rtrl_max_in = if use_delta { 4 } else { 2 };
            cfg.rtrl_lr = 0.006;
            cfg.rtrl_grid_mode = 1;
            cfg.rtrl_use_skip_x = false;
            cfg.rtrl_kappa = 0.5;
        }
    }
    cfg
}

fn eval_cartpole(stage: Stage, use_delta: bool, seed: u64) -> RlMetrics {
    let mut core = MicroContinualCore::new(cfg_cartpole(stage, use_delta, seed));
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
        && core.rtrl_state.b.iter().all(|v| v.is_finite())
        && core.rtrl_slow_state.b.iter().all(|v| v.is_finite());

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

// =============================================================================
// 3 & 4. Wave-Superposition Zero-Shot + Continual Regime Flip Ablation
// =============================================================================

struct WaveEp {
    obs: Vec<f32>,
    targets: Vec<usize>,
}

fn gen_wave_ep(rng: &mut u64, wa: f32, wb: f32, wc: f32, ep_len: usize, sign_flip: bool) -> WaveEp {
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
        let nv = eval_wave(t + 1);
        let raw_lbl = if nv >= 0.0 { 1usize } else { 0usize };
        targets.push(if sign_flip { 1 - raw_lbl } else { raw_lbl });
    }
    WaveEp { obs, targets }
}

fn cfg_wave(stage: Stage, ph_enabled: bool, seed: u64) -> MicroCoreConfig {
    let mut cfg = MicroCoreConfig::numeric_stream(2, seed);
    cfg.standardize_vec = false;
    cfg.belief_lambda = 0.88;
    cfg.belief_gamma = 0.0;
    cfg.rotor_norm = 0.55;
    cfg.novelty_gate = 0.0;
    cfg.trace_gamma = 0.0;
    cfg.linear_lr = 0.0;
    cfg.linear_mix = 1.0;
    cfg.linear_wd = 0.0001;
    cfg.bank_capacity = 0;
    cfg.obs_bank_capacity = 0;
    cfg.explore_ucb = 0.0;
    cfg.ph_delta = 0.14;
    cfg.ph_threshold = if ph_enabled { 2.2 } else { 99.0 };
    cfg.ph_warmup = 28;
    cfg.max_regimes = if ph_enabled { 4 } else { 2 };

    match stage {
        Stage::A0NoRtrl => {
            cfg.half_sectors = 16;
            cfg.actor_critic_lr = 0.0;
            cfg.linear_lr = 0.12;
            cfg.rtrl_sectors = 0;
        }
        Stage::A1RtrlFrozen => {
            cfg.half_sectors = 12;
            cfg.actor_critic_lr = 0.075;
            cfg.critic_lr = 0.14;
            cfg.entropy_beta = 0.003;
            cfg.rtrl_sectors = 16;
            cfg.rtrl_max_in = 1;
            cfg.rtrl_lr = 0.0;
            cfg.rtrl_grid_mode = 0;
            cfg.rtrl_kappa = 0.65;
        }
        Stage::A2RtrlTrainedCpl => {
            cfg.half_sectors = 12;
            cfg.actor_critic_lr = 0.075;
            cfg.critic_lr = 0.14;
            cfg.entropy_beta = 0.003;
            cfg.rtrl_sectors = 16;
            cfg.rtrl_max_in = 1;
            cfg.rtrl_lr = 0.010;
            cfg.rtrl_grid_mode = 0;
            cfg.rtrl_kappa = 0.65;
        }
        Stage::A3RtrlDecoupledFrozen | Stage::A3RtrlDecoupled | Stage::A3RtrlSkip => {
            cfg.half_sectors = 12;
            cfg.actor_critic_lr = 0.075;
            cfg.critic_lr = 0.14;
            cfg.entropy_beta = 0.003;
            cfg.rtrl_sectors = 16;
            cfg.rtrl_max_in = 1;
            cfg.rtrl_lr = if matches!(stage, Stage::A3RtrlDecoupledFrozen) { 0.0 } else { 0.010 };
            cfg.rtrl_grid_mode = 1;
            cfg.rtrl_use_skip_x = matches!(stage, Stage::A3RtrlSkip);
            cfg.rtrl_kappa = 0.65;
        }
        Stage::A4Unified2L => {
            cfg.half_sectors = 0;
            cfg.actor_critic_lr = 0.075;
            cfg.critic_lr = 0.14;
            cfg.entropy_beta = 0.003;
            cfg.rtrl_sectors = 14;
            cfg.rtrl_slow_sectors = 14;
            cfg.rtrl_max_in = 1;
            cfg.rtrl_lr = 0.010;
            cfg.rtrl_grid_mode = 1;
            cfg.rtrl_use_skip_x = true;
            cfg.rtrl_kappa = 0.65;
        }
    }
    cfg
}

#[derive(Default, Clone)]
struct WaveMetrics {
    abc: f64,
    a: f64,
    b: f64,
    c: f64,
    ab: f64,
    zs_mean: f64,
    ram_kb: f64,
    us_per_step: f64,
    seu_safe: bool,
}

fn eval_wave_ablation_with_params(stage: Stage, seed: u64) -> (WaveMetrics, PlasticCayleyParams) {
    let mut core = MicroContinualCore::new(cfg_wave(stage, false, seed));
    core.encoder.freeze_to_i8();
    let mut rng = splitmix64(seed ^ 0xBA7E_509E);
    let n_train = 220usize;
    let ep_len = 48usize;
    let mut steps = 0u64;
    let t0 = Instant::now();

    for _ in 0..n_train {
        let ep = gen_wave_ep(&mut rng, 1.0, 1.0, 1.0, ep_len, false);
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
            if matches!(stage, Stage::A0NoRtrl) {
                if t >= 6 {
                    core.observe_step_bandit_reward(pred, r);
                }
            } else {
                core.observe_step_reward(r);
            }
            ep_ret += r;
        }
        if !matches!(stage, Stage::A0NoRtrl) {
            let _ = core.end_episode(ep_ret, true);
        }
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let configs = [
        (1.0_f32, 1.0_f32, 1.0_f32),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ];
    let mut scores = [0.0_f64; 5];
    let mut ev_rng = splitmix64(seed ^ 0xE7A1_0001);
    for (ci, &(wa, wb, wc)) in configs.iter().enumerate() {
        let mut hits = 0usize;
        let mut total = 0usize;
        for _ in 0..40usize {
            let ep = gen_wave_ep(&mut ev_rng, wa, wb, wc, ep_len, false);
            core.start_episode();
            for t in 0..ep_len {
                let pred = core.decide_vec(&[ep.obs[t]], false, false, false).get().0;
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

    core.start_episode();
    let p_seu = core.decide_vec(&[f32::NAN], false, true, true);
    let seu_safe = p_seu.get().1.is_finite()
        && core.belief.state.iter().all(|v| v.is_finite())
        && core.rtrl_state.b.iter().all(|v| v.is_finite())
        && core.rtrl_slow_state.b.iter().all(|v| v.is_finite());

    let metrics = WaveMetrics {
        abc: scores[0],
        a: scores[1],
        b: scores[2],
        c: scores[3],
        ab: scores[4],
        zs_mean: (scores[1] + scores[2] + scores[3] + scores[4]) * 0.25,
        ram_kb: core.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (steps.max(1) as f64),
        seu_safe,
    };
    let params = core.banks[core.active_regime].rtrl_params.clone();
    (metrics, params)
}

fn eval_continual_wave_flip_unified(ph_enabled: bool, seed: u64) -> RlMetrics {
    let mut core = MicroContinualCore::new(cfg_wave(Stage::A4Unified2L, ph_enabled, seed));
    let mut rng = splitmix64(seed ^ 0xC071_5CA1);
    let ep_len = 48usize;
    let mut steps = 0u64;
    let t0 = Instant::now();

    let mut run_reg = |c: &mut MicroContinualCore, flip: bool, n_eps: usize| -> Vec<f64> {
        let mut accs = Vec::with_capacity(n_eps);
        for _ in 0..n_eps {
            let ep = gen_wave_ep(&mut rng, 1.0, 1.0, 1.0, ep_len, flip);
            c.start_episode();
            let mut hits = 0usize;
            let mut eval_n = 0usize;
            for t in 0..ep_len {
                let pred = c.decide_vec(&[ep.obs[t]], true, false, false).get().0;
                steps += 1;
                let r = if t >= 6 {
                    let ok = pred == ep.targets[t];
                    if ok {
                        hits += 1;
                    }
                    eval_n += 1;
                    if ok { 1.0_f32 } else { -1.0_f32 }
                } else {
                    0.0_f32
                };
                c.observe_step_reward(r);
            }
            let acc = (hits as f64) * 100.0 / (eval_n.max(1) as f64);
            let _ = c.end_episode_with_solved((acc / 100.0 * 2.0 - 1.0) as f32, acc >= 75.0, true);
            accs.push(acc);
        }
        accs
    };

    let ra = run_reg(&mut core, false, 180);
    let rb = run_reg(&mut core, true, 180);
    // Сразу 12 первых эпизодов возврата (моментальное восстановление банка A против затирания!)
    let rret = run_reg(&mut core, false, 40);
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let tail = |v: &[f64]| v[v.len() - 40..].iter().sum::<f64>() / 40.0;
    let first12_ret = rret[..12].iter().sum::<f64>() / 12.0;

    core.start_episode();
    let _ = core.decide_vec(&[f32::NAN], false, true, true);
    let seu_safe =
        core.rtrl_state.b.iter().all(|v| v.is_finite()) && core.rtrl_slow_state.b.iter().all(|v| v.is_finite());

    RlMetrics {
        score: (tail(&ra) + tail(&rb)) * 0.5,
        retention: first12_ret,
        recovery_eps: recovery_window(&rb, 80.0, 10),
        ram_kb: core.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (steps.max(1) as f64),
        seu_safe,
    }
}

fn eval_continual_wave_flip_gru(seed: u64) -> RlMetrics {
    let mut gru = OnlineGruBaseline::new(1, 32, 2, seed);
    gru.lr = 0.008;
    gru.gamma = 0.0;
    let mut rng = splitmix64(seed ^ 0xC071_5CA1);
    let ep_len = 48usize;
    let mut steps = 0u64;
    let t0 = Instant::now();

    let mut run_reg = |g: &mut OnlineGruBaseline, flip: bool, n_eps: usize| -> Vec<f64> {
        let mut accs = Vec::with_capacity(n_eps);
        for _ in 0..n_eps {
            let ep = gen_wave_ep(&mut rng, 1.0, 1.0, 1.0, ep_len, flip);
            g.start_episode();
            let mut hits = 0usize;
            let mut eval_n = 0usize;
            let mut ep_ret = 0.0_f32;
            for t in 0..ep_len {
                let pred = g.decide_vec(&[ep.obs[t]], true, false, false);
                steps += 1;
                let r = if t >= 6 {
                    let ok = pred == ep.targets[t];
                    if ok {
                        hits += 1;
                    }
                    eval_n += 1;
                    if ok { 1.0_f32 } else { -1.0_f32 }
                } else {
                    0.0_f32
                };
                g.observe_step_reward(r);
                ep_ret += r;
            }
            g.end_episode(ep_ret, true);
            accs.push((hits as f64) * 100.0 / (eval_n.max(1) as f64));
        }
        accs
    };

    let ra = run_reg(&mut gru, false, 180);
    let rb = run_reg(&mut gru, true, 180);
    let rret = run_reg(&mut gru, false, 40);
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    let tail = |v: &[f64]| v[v.len() - 40..].iter().sum::<f64>() / 40.0;
    let first12_ret = rret[..12].iter().sum::<f64>() / 12.0;

    let mut probe = OnlineGruBaseline::new(1, 32, 2, seed);
    let _ = probe.decide_vec(&[0.0], false, true, true);

    RlMetrics {
        score: (tail(&ra) + tail(&rb)) * 0.5,
        retention: first12_ret,
        recovery_eps: recovery_window(&rb, 80.0, 10),
        ram_kb: gru.memory_footprint_bytes() as f64 / 1024.0,
        us_per_step: elapsed_us / (steps.max(1) as f64),
        seu_safe: !probe.had_nan,
    }
}

fn try_load_elec2(path: &Path) -> Option<Vec<(Vec<f32>, usize)>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut out = Vec::new();
    let mut in_data = !text.contains("@relation");
    for line in text.lines() {
        let tr = line.trim();
        if tr.is_empty() || tr.starts_with('%') {
            continue;
        }
        if !in_data {
            if tr.eq_ignore_ascii_case("@data") {
                in_data = true;
            }
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
        if !ok {
            continue;
        }
        let lbl = cols[cols.len() - 1].trim().trim_matches('\'').trim_matches('"').to_ascii_lowercase();
        let y = if lbl == "up" || lbl == "1" || lbl == "true" { 1 } else { 0 };
        out.push((feats, y));
    }
    if out.len() >= 1000 { Some(out) } else { None }
}

fn main() {
    let seeds: Vec<u64> = (1..=10).collect();

    // 1. RepeatPrevious K=4, K=6
    for &k in &[4usize, 6usize] {
        print_rl_header(&format!("Ablation 1: POPGym RepeatPrevious (K={}, 4 sym, pure RL)", k));
        for st in Stage::all() {
            let ms: Vec<_> = seeds.iter().map(|&s| eval_repeat_prev(st, k, s)).collect();
            print_rl_row(st.label(), "%", &ms);
        }
    }

    // 2. CartPole pure 2D & +dx
    for &use_dx in &[false, true] {
        print_rl_header(&format!(
            "Ablation 2: Position-Only CartPole A->B->A ({})",
            if use_dx { "+dx" } else { "pure 2D" }
        ));
        for st in Stage::all() {
            let ms: Vec<_> = seeds.iter().map(|&s| eval_cartpole(st, use_dx, s)).collect();
            print_rl_row(st.label(), "st", &ms);
        }
    }

    // 3. Wave-Superposition Zero-Shot Transfer
    println!("\n=== Ablation 3: Wave-Superposition Zero-Shot (Train A+B+C -> Frozen Eval, 10 seeds) ===");
    println!(
        "{:<34} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6} | {:<10} | {:<8} | {:<7} | {}",
        "Stage / Architecture", "A+B+C", "ZS:A", "ZS:B", "ZS:C", "ZS:A+B", "Mean ZS", "RAM(KB)", "Lat", "SEU"
    );
    println!("{}", "-".repeat(126));
    let mut wave_seed1_learned: Option<PlasticCayleyParams> = None;
    for st in Stage::all() {
        let res: Vec<_> = seeds.iter().map(|&s| eval_wave_ablation_with_params(st, s)).collect();
        let ms: Vec<_> = res.iter().map(|(m, _)| m.clone()).collect();
        if matches!(st, Stage::A3RtrlDecoupled) {
            wave_seed1_learned = Some(res[0].1.clone());
        }
        let avg = |f: fn(&WaveMetrics) -> f64| -> f64 { ms.iter().map(f).sum::<f64>() / 10.0 };
        let zsv: Vec<f64> = ms.iter().map(|m| m.zs_mean).collect();
        let (zsm, zss) = mean_sd(&zsv);
        let seu = ms.iter().all(|m| m.seu_safe);
        println!(
            "{:<34} | {:5.1}% | {:5.1}% | {:5.1}% | {:5.1}% | {:5.1}% | {:5.1}±{:3.1}% | {:5.1} KB | {:4.2} µs | {}",
            st.label(),
            avg(|m| m.abc),
            avg(|m| m.a),
            avg(|m| m.b),
            avg(|m| m.c),
            avg(|m| m.ab),
            zsm,
            zss,
            avg(|m| m.ram_kb),
            avg(|m| m.us_per_step),
            if seu { "YES" } else { "NO" }
        );
    }
    if let Some(ref learned_w) = wave_seed1_learned {
        let init_core_w = MicroContinualCore::new(cfg_wave(Stage::A3RtrlDecoupled, false, 1));
        log_rtrl_spectrum_diff(
            "Wave-Superposition [A3] (seed 1)",
            &init_core_w.banks[0].rtrl_params,
            learned_w,
        );
    }

    // 4. Continual Wave-Superposition Flip (PageHinkley Retention on first 12 episodes back to Normal)
    println!("\n=== Ablation 4: Continual Wave Flip (Normal -> Inverted -> Immediate 12-ep Return, 10 seeds) ===");
    println!(
        "{:<34} | {:<15} | {:<8} | {:<11} | {:<9} | {:<9} | {}",
        "Architecture", "Score(mean±sd)", "Ret12ep", "Recovery", "RAM (KB)", "Latency", "SEU"
    );
    println!("{}", "-".repeat(108));
    let c_ph: Vec<_> = seeds.iter().map(|&s| eval_continual_wave_flip_unified(true, s)).collect();
    let c_noph: Vec<_> = seeds.iter().map(|&s| eval_continual_wave_flip_unified(false, s)).collect();
    let c_gru: Vec<_> = seeds.iter().map(|&s| eval_continual_wave_flip_gru(s)).collect();
    print_rl_row("Unified 2L MicroCore + PH Banks", "%", &c_ph);
    print_rl_row("Unified 2L (no PH, single bank)", "%", &c_noph);
    print_rl_row("Online GRU + A2C", "%", &c_gru);

    // 5. Optional ELEC2 Temporal Check
    for p in &[
        "electricity-normalized.arff",
        "../micro_continual_core/electricity-normalized.arff",
        "elecNormNew.csv",
    ] {
        if let Some(data) = try_load_elec2(Path::new(p)) {
            let mut nc = NoChangePersistent::new();
            let mut nc_hits = 0usize;
            for (x, y) in &data {
                if nc.predict_and_train(x, *y, true) == *y {
                    nc_hits += 1;
                }
            }
            let p_acc = (nc_hits as f64) * 100.0 / (data.len() as f64);

            println!("\n=== Ablation 5: Electricity ELEC2 Temporal ({} rows, 3 seeds) ===", data.len());
            println!("{:<34} | {:<11} | {:<7} | {:<9} | {:<9} | {}", "Stage", "Acc(m±sd)", "KappaT", "RAM (KB)", "Latency", "SEU");
            println!("{}", "-".repeat(88));
            let mut elec_init_banks: Option<Vec<PlasticCayleyParams>> = None;
            let mut elec_learned_reg: usize = 0;
            let mut elec_learned_p: Option<PlasticCayleyParams> = None;
            for st in Stage::all() {
                let mut accs = Vec::new();
                let mut rams = Vec::new();
                let mut lats = Vec::new();
                let mut seu_ok = true;
                for s in 1..=3u64 {
                    let mut cfg = MicroCoreConfig::numeric_stream(2, s);
                    cfg.trace_gamma = 0.0;
                    cfg.linear_lr = 0.07;
                    cfg.linear_mix = 1.0;
                    cfg.linear_wd = 0.0005;
                    match st {
                        Stage::A0NoRtrl => {
                            cfg.rtrl_sectors = 0;
                        }
                        Stage::A1RtrlFrozen => {
                            cfg.half_sectors = 12;
                            cfg.belief_lambda = 0.25;
                            cfg.rtrl_sectors = 12;
                            cfg.rtrl_max_in = 8;
                            cfg.rtrl_lr = 0.0;
                            cfg.rtrl_kappa = 0.65;
                        }
                        Stage::A2RtrlTrainedCpl => {
                            cfg.half_sectors = 12;
                            cfg.belief_lambda = 0.25;
                            cfg.rtrl_sectors = 12;
                            cfg.rtrl_max_in = 8;
                            cfg.rtrl_lr = 0.004;
                            cfg.rtrl_grid_mode = 0;
                            cfg.rtrl_kappa = 0.65;
                        }
                        Stage::A3RtrlDecoupledFrozen | Stage::A3RtrlDecoupled | Stage::A3RtrlSkip => {
                            cfg.half_sectors = 12;
                            cfg.belief_lambda = 0.25;
                            cfg.rtrl_sectors = 12;
                            cfg.rtrl_max_in = 8;
                            cfg.rtrl_lr = if matches!(st, Stage::A3RtrlDecoupledFrozen) { 0.0 } else { 0.004 };
                            cfg.rtrl_grid_mode = 1;
                            cfg.rtrl_use_skip_x = matches!(st, Stage::A3RtrlSkip);
                            cfg.rtrl_kappa = 0.65;
                        }
                        Stage::A4Unified2L => {
                            cfg.half_sectors = 12;
                            cfg.belief_lambda = 0.25;
                            cfg.rtrl_sectors = 12;
                            cfg.rtrl_slow_sectors = 10;
                            cfg.rtrl_max_in = 8;
                            cfg.rtrl_lr = 0.004;
                            cfg.rtrl_grid_mode = 1;
                            cfg.rtrl_use_skip_x = true;
                            cfg.rtrl_kappa = 0.65;
                        }
                    }
                    let mut core = MicroContinualCore::new(cfg);
                    if s == 1 && matches!(st, Stage::A3RtrlDecoupled) {
                        elec_init_banks = Some(core.banks.iter().map(|b| b.rtrl_params.clone()).collect());
                    }
                    core.encoder.freeze_to_i8();
                    core.start_episode();
                    let mut hits = 0usize;
                    let t0 = Instant::now();
                    for (x, y) in &data {
                        let pred = core.decide_vec(x, false, false, false).get().0;
                        if pred == *y {
                            hits += 1;
                        }
                        let _ = core.end_supervised_step(pred, *y);
                    }
                    let us = t0.elapsed().as_nanos() as f64 / 1000.0;
                    if s == 1 && matches!(st, Stage::A3RtrlDecoupled) {
                        elec_learned_reg = core.active_regime;
                        elec_learned_p = Some(core.banks[core.active_regime].rtrl_params.clone());
                    }
                    core.start_episode();
                    let p_seu = core.decide_vec(&[f32::NAN, 1e30, -1e30], false, true, true);
                    if !(p_seu.get().1.is_finite()
                        && core.belief.state.iter().all(|v| v.is_finite())
                        && core.rtrl_state.b.iter().all(|v| v.is_finite())
                        && core.rtrl_slow_state.b.iter().all(|v| v.is_finite()))
                    {
                        seu_ok = false;
                    }
                    accs.push((hits as f64) * 100.0 / (data.len() as f64));
                    rams.push(core.memory_footprint_bytes() as f64 / 1024.0);
                    lats.push(us / (data.len() as f64));
                }
                let (am, asd) = mean_sd(&accs);
                let rm = rams.iter().sum::<f64>() / 3.0;
                let lm = lats.iter().sum::<f64>() / 3.0;
                let kt = (am - p_acc) / (100.0 - p_acc);
                println!(
                    "{:<34} | {:5.1}±{:4.1}% | {:+7.3} | {:6.1} KB | {:6.2} µs | {}",
                    st.label(),
                    am,
                    asd,
                    kt,
                    rm,
                    lm,
                    if seu_ok { "YES" } else { "NO (NaN)" }
                );
            }
            if let (Some(ref init_banks), Some(ref ep1)) = (elec_init_banks, elec_learned_p) {
                log_rtrl_spectrum_diff(
                    &format!("Electricity ELEC2 [A3] (seed 1, bank {})", elec_learned_reg),
                    &init_banks[elec_learned_reg],
                    ep1,
                );
            }
            break;
        }
    }
}
