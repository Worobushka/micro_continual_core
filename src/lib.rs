// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

pub mod baselines;
pub mod belief;
pub mod core;
pub mod env_hard_pomdp;
pub mod math;
pub mod memory;
pub mod stream_baselines;
pub mod rtrl;

pub use baselines::{
    DeepRtrrlLruPhBaseline, OnlineGruPerStepPhBaseline, OnlineSelectiveSsmPhBaseline,
    RtrrlLruPhBaseline, SelectiveSsmBank,
};
pub use belief::{CayleyBeliefState, EligibilityTraceRing, PageHinkleyDetector};
pub use core::{MicroContinualCore, MicroCoreConfig};
pub use env_hard_pomdp::{
    unlabeled_domain_cloze_logs_noisy, HardPomdpConfig, HardPomdpV2, HardRegime, ParaphraseSplit,
    NUM_ENV_ACTIONS,
};
pub use rtrl::{
    cayley_rot3_double_vec, cayley_rot3_double_vec_deriv, cayley_rot3_vec, cayley_rot3_vec_deriv,
    AlgebraicSelectiveCayley, HierarchicalCayleyRtrl, NonAbelianCayleyRtrl, PlasticCayleyParams,
    PlasticCayleyState,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::fast_dot;

    #[test]
    fn test_ram_under_500kb_and_seu_safety() {
        let mut core = MicroContinualCore::new(MicroCoreConfig::default());
        core.encoder.freeze_to_i8();
        let kb = (core.memory_footprint_bytes() as f64) / 1024.0;
        assert!(kb < 500.0, "MicroContinualCore must fit under 500 KB, got {:.1} KB", kb);

        core.start_episode();
        let p_txt = core.decide_text("node alert disk array", true, true, true);
        assert!(p_txt.get().1.is_finite());
        let p_vec = core.decide_vec(&[f32::NAN, 1e30, -1e30, 0.5], true, true, true);
        assert!(p_vec.get().1.is_finite());
        assert!(core.belief.state.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_cayley_belief_preserves_hidden_subsystem_after_identical_clue() {
        let enc = math::SpinorEncoder::new(32, 10);
        let mut b_a = CayleyBeliefState::new(enc.dim, enc.half_dim, 0.82, 0.72, 0.14, 42);
        let mut b_b = CayleyBeliefState::new(enc.dim, enc.half_dim, 0.82, 0.72, 0.14, 42);

        let mut x_a0 = vec![0.0_f32; enc.dim];
        let mut x_b0 = vec![0.0_f32; enc.dim];
        let mut x_clue = vec![0.0_f32; enc.dim];
        let mut tmp = vec![0.0_f32; enc.half_dim];
        let mut sl = Vec::new();

        enc.encode_text_into("node alert disk array blk stall", &mut x_a0, &mut sl, &mut tmp);
        enc.encode_text_into("node alert ledger tx outlier lock", &mut x_b0, &mut sl, &mut tmp);
        enc.encode_text_into("clue perm denied chmod", &mut x_clue, &mut sl, &mut tmp);

        let _ = b_a.update(&x_a0);
        let _ = b_b.update(&x_b0);
        let _ = b_a.update(&x_clue);
        let _ = b_b.update(&x_clue);

        let sim = fast_dot(&b_a.state, &b_b.state);
        assert!(
            sim < 0.78,
            "Belief state failed to distinguish hidden subsystem after identical clue: cos={:.3}",
            sim
        );
    }

    #[test]
    fn test_page_hinkley_ignores_20pct_noise_but_catches_real_drift_fast() {
        let mut ph = PageHinkleyDetector::new(0.14, 1.55, 14);
        let mut false_alarms = 0usize;
        for ep in 0..80 {
            let r = if ep % 5 == 0 { 0.0 } else { 1.0 };
            if ph.observe(r) {
                false_alarms += 1;
                ph.reset_cusum();
            }
        }
        assert_eq!(false_alarms, 0, "Page-Hinkley must not false-alarm on 20% glitches");

        let mut detect_steps = 0usize;
        for step in 1..=12 {
            if ph.observe(0.0) {
                detect_steps = step;
                break;
            }
        }
        assert!(
            detect_steps > 0 && detect_steps <= 6,
            "Expected drift detection within <=6 episodes, got {}",
            detect_steps
        );
    }

    #[test]
    fn test_hard_pomdp_v2_strict_honest_contract() {
        let mut ep = HardPomdpV2::generate(
            77,
            HardPomdpConfig::strict_honest(HardRegime::Alpha, 0, ParaphraseSplit::CanonicalTrain),
        );
        let exp_ins = ep.expected_inspect();
        let r1 = ep.step(exp_ins);
        assert!(!r1.get().observation.contains("cluster"));
        assert_eq!(r1.get().reward, 0.0);

        let exp_rep = ep.expected_repair();
        let r2 = ep.step(exp_rep);
        assert_eq!(r2.get().reward, 0.0);

        let r3 = ep.step(10);
        assert!(r3.get().done && r3.get().solved && r3.get().reward == 1.0);
    }

    #[test]
    fn test_cloze_canonical_context_without_holdout_leakage() {
        let mut enc = math::SpinorEncoder::new(32, 10);
        let logs = unlabeled_domain_cloze_logs_noisy();
        let forbidden_holdout = [
            "telemetry",
            "nvme",
            "sector",
            "malformed",
            "zonealpha",
            "zonebeta",
            "zonegamma",
            "zonedelta",
            "postcheck",
            "corrupt",
        ];
        for line in &logs {
            for w in &forbidden_holdout {
                assert!(
                    !line.contains(w),
                    "Unlabeled Cloze logs leaked UnseenHoldout token '{w}' in line: {line}"
                );
            }
        }
        enc.adapt_unlabeled_cloze(&logs, 45);
        enc.freeze_to_i8();

        let mut v_canon = vec![0.0_f32; enc.dim];
        let mut v_ctx = vec![0.0_f32; enc.dim];
        let mut v_other = vec![0.0_f32; enc.dim];
        let mut tmp = vec![0.0_f32; enc.half_dim];
        let mut sl = Vec::new();

        enc.encode_text_into("node alert disk array blk stall", &mut v_canon, &mut sl, &mut tmp);
        enc.encode_text_into("block storage driver on flush tier", &mut v_ctx, &mut sl, &mut tmp);
        enc.encode_text_into("clearing journal gate under variance check", &mut v_other, &mut sl, &mut tmp);

        let hd = enc.half_dim;
        let sim_ctx = fast_dot(&v_canon[0..hd], &v_ctx[0..hd]);
        let sim_diff = fast_dot(&v_canon[0..hd], &v_other[0..hd]);
        assert!(
            sim_ctx > sim_diff + 0.10,
            "Noisy Cloze must align canonical context closer than unrelated domain: ctx={:.3}, diff={:.3}",
            sim_ctx,
            sim_diff
        );
    }

    #[test]
    fn test_cayley_rtrl_exact_gradients() {
        use crate::rtrl::{PlasticCayleyParams, PlasticCayleyState};

        let params = PlasticCayleyParams::new(4, 2, 123);
        let seq = [[0.7_f32, -0.3], [-0.4, 0.8], [0.2, 0.5], [-0.9, -0.2], [0.6, 0.1]];

        let run_u = |p: &PlasticCayleyParams| -> Vec<f32> {
            let mut st = PlasticCayleyState::new(4, 2, 0.5);
            for x in &seq {
                st.step_forward(p, x);
            }
            st.u
        };

        let mut st_exact = PlasticCayleyState::new(4, 2, 0.5);
        for x in &seq {
            st_exact.step_forward(&params, x);
        }

        let eps = 1e-3_f32;
        for s in 0..4 {
            let off = s * 3;
            let mut p_plus = params.clone();
            let mut p_minus = params.clone();
            p_plus.alpha[s] += eps;
            p_minus.alpha[s] -= eps;
            let u_p = run_u(&p_plus);
            let u_m = run_u(&p_minus);
            for k in 0..3 {
                let fd = (u_p[off + k] - u_m[off + k]) / (2.0 * eps);
                let ex = st_exact.p_alpha[off + k];
                assert!((fd - ex).abs() < 1e-2, "p_alpha mismatch s={s} k={k}: fd={fd} ex={ex}");
            }

            let mut p_plus = params.clone();
            let mut p_minus = params.clone();
            p_plus.omega[s] += eps;
            p_minus.omega[s] -= eps;
            let u_p = run_u(&p_plus);
            let u_m = run_u(&p_minus);
            for k in 0..3 {
                let fd = (u_p[off + k] - u_m[off + k]) / (2.0 * eps);
                let ex = st_exact.p_omega[off + k];
                assert!((fd - ex).abs() < 1e-2, "p_omega mismatch s={s} k={k}: fd={fd} ex={ex}");
            }

            let mut p_plus = params.clone();
            let mut p_minus = params.clone();
            p_plus.in_w[s * 2] += eps;
            p_minus.in_w[s * 2] -= eps;
            let u_p = run_u(&p_plus);
            let u_m = run_u(&p_minus);
            for k in 0..3 {
                let fd = (u_p[off + k] - u_m[off + k]) / (2.0 * eps);
                let ex = st_exact.p_w[(s * 2) * 3 + k];
                assert!((fd - ex).abs() < 1e-2, "p_w mismatch s={s} k={k}: fd={fd} ex={ex}");
            }
        }
    }

    #[test]
    fn test_rational_policy_and_tsallis_exact_gradients() {
        use crate::math::{rational_circle_wave, rational_probs_and_psi_into, rational_probs_into};

        let logits = [1.4_f32, -0.8, 0.25, -2.1];
        let mut probs = [0.0_f32; 4];
        let mut psis = [0.0_f32; 4];
        rational_probs_and_psi_into(&logits, &mut probs, &mut psis);
        let sum_p: f32 = probs.iter().sum();
        assert!((sum_p - 1.0).abs() < 1e-5, "probs must sum to 1, got {sum_p}");
        assert!(probs.iter().all(|&p| p > 0.0 && p < 1.0));

        let shifted = [logits[0] + 7.5, logits[1] + 7.5, logits[2] + 7.5, logits[3] + 7.5];
        let mut probs_sh = [0.0_f32; 4];
        rational_probs_into(&shifted, &mut probs_sh);
        for a in 0..4 {
            assert!(
                (probs[a] - probs_sh[a]).abs() < 1e-5,
                "shift invariance failed a={a}: {} vs {}",
                probs[a],
                probs_sh[a]
            );
        }

        let eps = 1e-3_f32;
        let chosen = 2usize;
        let sum_p2: f32 = probs.iter().map(|&p| p * p).sum();
        let mut raw_s = [0.0_f32; 4];
        let mut raw_h = [0.0_f32; 4];
        for a in 0..4 {
            let ind = if a == chosen { 1.0_f32 } else { 0.0_f32 };
            raw_s[a] = psis[a] * (ind - probs[a]);
            raw_h[a] = probs[a] * psis[a] * (sum_p2 - probs[a]);
        }
        let mean_s: f32 = raw_s.iter().sum::<f32>() / 4.0;
        let mean_h: f32 = raw_h.iter().sum::<f32>() / 4.0;

        for a in 0..4 {
            let mut lp = logits;
            let mut lm = logits;
            lp[a] += eps;
            lm[a] -= eps;
            let mut pp = [0.0_f32; 4];
            let mut pm = [0.0_f32; 4];
            rational_probs_into(&lp, &mut pp);
            rational_probs_into(&lm, &mut pm);

            let fd_logp = (pp[chosen].ln() - pm[chosen].ln()) / (2.0 * eps);
            let ex_logp = raw_s[a] - mean_s;
            assert!((fd_logp - ex_logp).abs() < 1e-3, "score mismatch a={a}: fd={fd_logp} ex={ex_logp}");

            let h2_p = 0.5 * (1.0 - pp.iter().map(|&x| x * x).sum::<f32>());
            let h2_m = 0.5 * (1.0 - pm.iter().map(|&x| x * x).sum::<f32>());
            let fd_h2 = (h2_p - h2_m) / (2.0 * eps);
            let ex_h2 = raw_h[a] - mean_h;
            assert!((fd_h2 - ex_h2).abs() < 1e-3, "tsallis mismatch a={a}: fd={fd_h2} ex={ex_h2}");
        }

        let (s0, c0) = rational_circle_wave(0.0);
        assert!(s0.abs() < 1e-6 && (c0 - 1.0).abs() < 1e-6);
        for k in -25..=25 {
            let (s, c) = rational_circle_wave((k as f32) * 0.37);
            assert!((s * s + c * c - 1.0).abs() < 1e-5, "circle norm failed at k={k}");
        }
    }

    #[test]
    fn test_rtrrl_lru_exact_gradients() {
        use crate::baselines::RtrrlLruBaseline;

        let base = RtrrlLruBaseline::new(2, 4, 2, 777);
        let seq = [[0.7_f32, -0.3], [-0.4, 0.8], [0.2, 0.5], [-0.9, -0.2], [0.6, 0.1]];

        let run_h = |mut m: RtrrlLruBaseline| -> (Vec<f32>, Vec<f32>) {
            for x in &seq {
                m.step_lru_forward(x);
            }
            (m.h_re, m.h_im)
        };

        let mut exact = RtrrlLruBaseline::new(2, 4, 2, 777);
        for x in &seq {
            exact.step_lru_forward(x);
        }

        let eps = 1e-3_f32;
        for j in 0..4 {
            let mut mp = RtrrlLruBaseline::new(2, 4, 2, 777);
            let mut mm = RtrrlLruBaseline::new(2, 4, 2, 777);
            mp.nu[j] += eps;
            mm.nu[j] -= eps;
            let (pr, pi) = run_h(mp);
            let (mr, mi) = run_h(mm);
            assert!(((pr[j] - mr[j]) / (2.0 * eps) - exact.p_nu_re[j]).abs() < 2e-2);
            assert!(((pi[j] - mi[j]) / (2.0 * eps) - exact.p_nu_im[j]).abs() < 2e-2);

            let mut mp = RtrrlLruBaseline::new(2, 4, 2, 777);
            let mut mm = RtrrlLruBaseline::new(2, 4, 2, 777);
            mp.theta[j] += eps;
            mm.theta[j] -= eps;
            let (pr, pi) = run_h(mp);
            let (mr, mi) = run_h(mm);
            assert!(((pr[j] - mr[j]) / (2.0 * eps) - exact.p_th_re[j]).abs() < 2e-2);
            assert!(((pi[j] - mi[j]) / (2.0 * eps) - exact.p_th_im[j]).abs() < 2e-2);

            let mut mp = RtrrlLruBaseline::new(2, 4, 2, 777);
            let mut mm = RtrrlLruBaseline::new(2, 4, 2, 777);
            mp.b_re[j * 2] += eps;
            mm.b_re[j * 2] -= eps;
            let (pr, pi) = run_h(mp);
            let (mr, mi) = run_h(mm);
            assert!(((pr[j] - mr[j]) / (2.0 * eps) - exact.p_b_re[j * 2]).abs() < 2e-2);
            assert!(((pi[j] - mi[j]) / (2.0 * eps) - exact.p_b_im[j * 2]).abs() < 2e-2);

            let mut mp = RtrrlLruBaseline::new(2, 4, 2, 777);
            let mut mm = RtrrlLruBaseline::new(2, 4, 2, 777);
            mp.b_im[j * 2] += eps;
            mm.b_im[j * 2] -= eps;
            let (pr, pi) = run_h(mp);
            let (mr, mi) = run_h(mm);
            assert!(((pr[j] - mr[j]) / (2.0 * eps) - (-exact.p_b_im[j * 2])).abs() < 2e-2);
            assert!(((pi[j] - mi[j]) / (2.0 * eps) - exact.p_b_re[j * 2]).abs() < 2e-2);
        }
        let _ = base;
    }

    #[test]
    fn test_hierarchical_cayley_rtrl_ram_and_seu() {
        let mut h = HierarchicalCayleyRtrl::new(2, 12, 12, 4, 42);
        let kb = h.memory_footprint_bytes() as f64 / 1024.0;
        assert!(kb < 40.0, "HierarchicalCayleyRtrl RAM exceeded 40 KB: {kb:.2} KB");
        h.start_episode();
        let a = h.decide_vec(&[f32::NAN, 1e30], true, true, true);
        assert!(a < 4);
        assert!(h.fast_state.b.iter().all(|v| v.is_finite()));
        assert!(h.slow_state.b.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_unified_2l_microcore_architecture_ram_and_seu() {
        let mut cfg1 = MicroCoreConfig::numeric_stream(4, 99);
        cfg1.half_sectors = 0;
        cfg1.max_regimes = 3;
        cfg1.rtrl_sectors = 14;
        cfg1.rtrl_slow_sectors = 0;
        cfg1.rtrl_max_in = 4;
        cfg1.rtrl_lr = 0.01;
        cfg1.actor_critic_lr = 0.08;
        let core1 = MicroContinualCore::new(cfg1);
        assert_eq!(core1.rtrl_slow_state.sectors, 0);
        assert_eq!(core1.banks[0].rtrl_slow_actor_w.len(), 0);

        let mut cfg2 = cfg1;
        cfg2.rtrl_slow_sectors = 14;
        cfg2.rtrl_grid_mode = 1;
        cfg2.rtrl_use_skip_x = true;
        cfg2.rtrl_kappa = 0.65;
        let mut core2 = MicroContinualCore::new(cfg2);
        let kb = core2.memory_footprint_bytes() as f64 / 1024.0;
        assert!(kb < 40.0, "Unified 2L MicroContinualCore RAM exceeded 40 KB: {kb:.2} KB");
        assert_eq!(core2.encoder.memory_footprint_bytes(), 0);
        assert_eq!(core2.belief.memory_footprint_bytes(), 0);
        assert_eq!(core2.banks[0].linear_w.len(), 0);
        assert_eq!(core2.scratch.td_e_actor_f.len(), 0);

        core2.start_episode();
        let p = core2.decide_vec(&[f32::NAN, 1e30, -1e30, 0.25], true, true, true);
        assert!(p.get().0 < 4 && p.get().1.is_finite());
        assert!(core2.rtrl_state.b.iter().all(|v| v.is_finite()));
        assert!(core2.rtrl_slow_state.b.iter().all(|v| v.is_finite()));
        core2.observe_step_reward(1.0);
        let _ = core2.end_episode(1.0, true);
    }

    #[test]
    fn test_non_abelian_cayley_rtrl_exact_gradients_and_noncommutativity() {
        // 1. Строгая некоммутативность входозависимых 3D роторов: R(g1)R(g2)v != R(g2)R(g1)v
        let v0 = (0.6_f32, -0.5_f32, 0.62_f32);
        let a1 = cayley_rot3_vec(0.9, -0.3, 0.4, v0.0, v0.1, v0.2);
        let r12 = cayley_rot3_vec(-0.2, 1.1, 0.7, a1.0, a1.1, a1.2);
        let b1 = cayley_rot3_vec(-0.2, 1.1, 0.7, v0.0, v0.1, v0.2);
        let r21 = cayley_rot3_vec(0.9, -0.3, 0.4, b1.0, b1.1, b1.2);
        let noncomm = ((r12.0 - r21.0).powi(2) + (r12.1 - r21.1).powi(2) + (r12.2 - r21.2).powi(2)).sqrt();
        assert!(noncomm > 0.25, "SO(3) Cayley rotors must be strictly non-commutative: diff={noncomm}");

        // 2. Проверка аналитических чувствительностей NonAbelianCayleyRtrl против конечных разностей
        let seq = [
            [0.6_f32, -0.4, 0.2],
            [-0.3, 0.8, -0.5],
            [0.2, 0.1, 0.9],
            [-0.7, -0.2, 0.3],
        ];
        let run_u = |mut m: NonAbelianCayleyRtrl| -> Vec<f32> {
            for x in &seq {
                m.step_forward(x);
            }
            m.u
        };

        let mut exact = NonAbelianCayleyRtrl::new(2, 3, 4, 0.55, 777);
        for x in &seq {
            exact.step_forward(x);
        }

        let eps = 1e-3_f32;
        for s in 0..2usize {
            let off = s * 3;
            let mut mp = NonAbelianCayleyRtrl::new(2, 3, 4, 0.55, 777);
            let mut mm = NonAbelianCayleyRtrl::new(2, 3, 4, 0.55, 777);
            mp.alpha[s] += eps;
            mm.alpha[s] -= eps;
            let up = run_u(mp);
            let um = run_u(mm);
            for coord in 0..3 {
                let fd = (up[off + coord] - um[off + coord]) / (2.0 * eps);
                let ex = exact.p_alpha[off + coord];
                assert!((fd - ex).abs() < 5e-3, "p_alpha mismatch s={s} c={coord}: fd={fd} ex={ex}");
            }

            for k in 0..3usize {
                let mut mp = NonAbelianCayleyRtrl::new(2, 3, 4, 0.55, 777);
                let mut mm = NonAbelianCayleyRtrl::new(2, 3, 4, 0.55, 777);
                mp.g0[off + k] += eps;
                mm.g0[off + k] -= eps;
                let up = run_u(mp);
                let um = run_u(mm);
                let goff = (off + k) * 3;
                for coord in 0..3 {
                    let fd = (up[off + coord] - um[off + coord]) / (2.0 * eps);
                    let ex = exact.p_g0[goff + coord];
                    assert!((fd - ex).abs() < 5e-3, "p_g0 mismatch s={s} k={k} c={coord}: fd={fd} ex={ex}");
                }

                for j in 0..3usize {
                    let widx = (off + k) * 3 + j;
                    let mut mp = NonAbelianCayleyRtrl::new(2, 3, 4, 0.55, 777);
                    let mut mm = NonAbelianCayleyRtrl::new(2, 3, 4, 0.55, 777);
                    mp.wg[widx] += eps;
                    mm.wg[widx] -= eps;
                    let up = run_u(mp);
                    let um = run_u(mm);
                    let poff = widx * 3;
                    for coord in 0..3 {
                        let fd = (up[off + coord] - um[off + coord]) / (2.0 * eps);
                        let ex = exact.p_wg[poff + coord];
                        assert!((fd - ex).abs() < 5e-3, "p_wg mismatch s={s} k={k} j={j} c={coord}: fd={fd} ex={ex}");
                    }

                    let mut mp = NonAbelianCayleyRtrl::new(2, 3, 4, 0.55, 777);
                    let mut mm = NonAbelianCayleyRtrl::new(2, 3, 4, 0.55, 777);
                    mp.b_mat[widx] += eps;
                    mm.b_mat[widx] -= eps;
                    let up = run_u(mp);
                    let um = run_u(mm);
                    for coord in 0..3 {
                        let fd = (up[off + coord] - um[off + coord]) / (2.0 * eps);
                        let ex = exact.p_b[poff + coord];
                        assert!((fd - ex).abs() < 5e-3, "p_b mismatch s={s} k={k} j={j} c={coord}: fd={fd} ex={ex}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_register_bit_flip_self_healing_and_fault_telemetry() {
        // 1. MicroContinualCore (2L RTRL): инъекция битовых сбоев экспоненты (bit 30) прямо в регистр состояния u
        let mut cfg = MicroCoreConfig::numeric_stream(4, 123);
        cfg.half_sectors = 0;
        cfg.rtrl_sectors = 12;
        cfg.rtrl_slow_sectors = 8;
        cfg.rtrl_max_in = 4;
        cfg.actor_critic_lr = 0.0;
        cfg.linear_lr = 0.0;
        let mut core_clean = MicroContinualCore::new(cfg);
        core_clean.encoder.freeze_to_i8();
        core_clean.start_episode();

        let mut core_fault = MicroContinualCore::new(cfg);
        core_fault.encoder.freeze_to_i8();
        core_fault.start_episode();

        for step in 0..20usize {
            let t = step as f32 * 0.25;
            let x = [t.sin(), t.cos(), 0.5, -0.3];
            let _ = core_clean.decide_vec(&x, false, false, false);
            let _ = core_fault.decide_vec(&x, false, false, false);
        }
        // Устанавливаем амплитуду в [1.0, 2.0) для детерминированной проверки переворота старшего бита экспоненты (bit 30 -> NaN)
        core_clean.rtrl_state.u[0] = 1.25;
        core_fault.rtrl_state.u[0] = 1.25;
        core_fault.inject_register_bit_flip(0, 30);

        let mut max_err_after_heal = 1.0_f32;
        for step in 20..70usize {
            let t = step as f32 * 0.25;
            let x = [t.sin(), t.cos(), 0.5, -0.3];
            let p_c = core_clean.decide_vec(&x, false, false, false);
            let p_f = core_fault.decide_vec(&x, false, false, false);
            assert!(p_c.get().1.is_finite() && p_f.get().1.is_finite());
            if step == 69 {
                max_err_after_heal = core_clean
                    .rtrl_state
                    .b
                    .iter()
                    .zip(core_fault.rtrl_state.b.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
            }
        }
        assert!(core_fault.total_seu_faults() >= 1, "SEU fault telemetry must detect register bit-flip");
        assert!(core_fault.rtrl_state.b.iter().all(|v| v.is_finite()));
        assert!(
            max_err_after_heal < 0.05,
            "Self-healing failed to contract state perturbation: max_err={max_err_after_heal}"
        );

        // 2. NonAbelianCayleyRtrl: инъекция сбоя + NaN в регистр и проверка восстановления
        let mut nac_clean = NonAbelianCayleyRtrl::new(4, 3, 5, 0.5, 42);
        for a in &mut nac_clean.alpha {
            *a = 0.45; // lambda = 0.827 -> гарантированное сжатие за 40 шагов
        }
        for step in 0..10usize {
            let x = [((step % 3) as f32) * 0.4 - 0.3, 0.5, -0.2];
            nac_clean.step_forward(&x);
        }
        let mut nac_fault = nac_clean.clone();
        nac_fault.u[0] = f32::NAN;
        nac_fault.u[3] = 0.85;
        nac_fault.inject_register_bit_flip(3, 30);
        for step in 10..55usize {
            let x = [((step % 3) as f32) * 0.4 - 0.3, 0.5, -0.2];
            nac_clean.step_forward(&x);
            nac_fault.step_forward(&x);
        }
        assert!(nac_fault.seu_fault_count >= 2);
        assert!(nac_fault.b.iter().all(|v| v.is_finite()));
        let nac_err = nac_clean
            .b
            .iter()
            .zip(nac_fault.b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(nac_err < 0.02, "NonAbelianCayleyRtrl failed to self-heal: {nac_err}");

        // 3. Инъекция битовых сбоев экспоненты (bit 30) прямо в веса (alpha, omega, in_w, rtrl_actor_w)
        let mut core_wt = MicroContinualCore::new(cfg);
        core_wt.encoder.freeze_to_i8();
        core_wt.start_episode();
        let before_faults = core_wt.total_seu_faults();
        for step in 0..8usize {
            let x = [0.5_f32, -0.3, 0.4, 0.1];
            core_wt.inject_weight_bit_flip(step, 30);
            let p = core_wt.decide_vec(&x, false, false, false);
            assert!(p.get().1.is_finite());
            assert!(core_wt.rtrl_state.b.iter().all(|v| v.is_finite()));
            assert!(core_wt.rtrl_slow_state.b.iter().all(|v| v.is_finite()));
        }
        assert!(
            core_wt.total_seu_faults() >= before_faults + 8,
            "Weight-register SEU telemetry must detect all corrupted weights: before={}, after={}",
            before_faults,
            core_wt.total_seu_faults()
        );
    }

    #[test]
    fn test_double_cayley_rot3_exact_derivatives() {
        let g = [0.72_f32, -0.48_f32, 0.35_f32];
        let v = (0.52_f32, -0.61_f32, 0.598_f32);
        let z0 = cayley_rot3_double_vec(g[0], g[1], g[2], v.0, v.1, v.2);
        let norm0 = (z0.0 * z0.0 + z0.1 * z0.1 + z0.2 * z0.2).sqrt();
        let norm_v = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt();
        assert!((norm0 - norm_v).abs() < 1e-5, "Double Cayley must preserve Euclidean norm");

        let eps = 1e-3_f32;
        for k in 0..3usize {
            let mut gp = g;
            let mut gm = g;
            gp[k] += eps;
            gm[k] -= eps;
            let zp = cayley_rot3_double_vec(gp[0], gp[1], gp[2], v.0, v.1, v.2);
            let zm = cayley_rot3_double_vec(gm[0], gm[1], gm[2], v.0, v.1, v.2);
            let an = cayley_rot3_double_vec_deriv(g[0], g[1], g[2], v.0, v.1, v.2, k);
            let fdx = (zp.0 - zm.0) / (2.0 * eps);
            let fdy = (zp.1 - zm.1) / (2.0 * eps);
            let fdz = (zp.2 - zm.2) / (2.0 * eps);
            assert!(
                (fdx - an.0).abs() < 2e-3 && (fdy - an.1).abs() < 2e-3 && (fdz - an.2).abs() < 2e-3,
                "Double Cayley derivative mismatch k={k}: fd=({fdx},{fdy},{fdz}) an={an:?}"
            );
        }
    }

    #[test]
    fn test_non_abelian_so3_frames_exact_gradients() {
        let mut nac = NonAbelianCayleyRtrl::new(6, 4, 5, 0.25, 42);
        nac.enable_so3_frames_end_to_end();
        let seq = [
            [0.6_f32, -0.3, 0.2, 0.4],
            [-0.4, 0.7, -0.5, 0.1],
            [0.3, 0.2, 0.8, -0.6],
            [-0.5, -0.2, 0.3, 0.7],
        ];
        let d = nac.sectors * 3;
        let mut rng_w = math::splitmix64(999);
        let w_obj: Vec<f32> = (0..d)
            .map(|_| math::rand_signed_f32(&mut rng_w) * 0.5)
            .collect();
        let run_j = |p: &NonAbelianCayleyRtrl| -> f32 {
            let mut c = p.clone();
            c.reset();
            for x in &seq {
                c.step_forward(x);
            }
            fast_dot(&w_obj, &c.b)
        };

        let mut nac_run = nac.clone();
        nac_run.reset();
        for x in &seq {
            nac_run.step_forward(x);
        }

        let d_in = nac_run.in_dim;
        let n_frames = nac_run.sectors / 3;
        let eps = 1e-3_f32;
        let mut max_err = 0.0_f32;
        for m in 0..n_frames {
            let s0 = m * 3;
            let frame_off = s0 * 3;
            let mut fbs = [(0.0_f32, 0.0_f32, 0.0_f32); 3];
            for a in 0..3 {
                let s = s0 + a;
                let off = s * 3;
                let (gx, gy, gz) = (w_obj[off], w_obj[off + 1], w_obj[off + 2]);
                let (bx, by, bz) = (nac_run.b[off], nac_run.b[off + 1], nac_run.b[off + 2]);
                let gb = gx * bx + gy * by + gz * bz;
                let inv_rho = 1.0 / nac_run.rho[s];
                fbs[a] = ((gx - gb * bx) * inv_rho, (gy - gb * by) * inv_rho, (gz - gb * bz) * inv_rho);
            }
            for k in 0..3 {
                for j in 0..d_in {
                    let mut an = 0.0_f32;
                    for a in 0..3 {
                        let s = s0 + a;
                        let off = s * 3;
                        let (fbx, fby, fbz) = fbs[a];
                        let p_off = ((off + k) * d_in + j) * 3;
                        an += fbx * nac_run.p_wg[p_off]
                            + fby * nac_run.p_wg[p_off + 1]
                            + fbz * nac_run.p_wg[p_off + 2];
                    }
                    let mut nac_p = nac.clone();
                    let mut nac_m = nac.clone();
                    let idx = (frame_off + k) * d_in + j;
                    nac_p.wg[idx] += eps;
                    nac_m.wg[idx] -= eps;
                    let fd = (run_j(&nac_p) - run_j(&nac_m)) / (2.0 * eps);
                    max_err = max_err.max((fd - an).abs());
                }
            }
        }
        assert!(
            max_err < 1e-3,
            "NonAbelian SO(3) frame RTRL gradient mismatch: max_err={max_err}"
        );
    }

    #[test]
    fn test_rtrl_pool_w_orthonormal() {
        let mut cfg = MicroCoreConfig::numeric_stream(4, 17);
        cfg.rtrl_sectors = 16;
        cfg.rtrl_slow_sectors = 16;
        let core = MicroContinualCore::new(cfg);
        let row_len = 3 * 16;
        assert_eq!(core.rtrl_pool_w.len(), 4 * row_len);
        let mut max_err = 0.0_f32;
        for i in 0..4usize {
            for j in 0..4usize {
                let d = fast_dot(
                    &core.rtrl_pool_w[i * row_len..(i + 1) * row_len],
                    &core.rtrl_pool_w[j * row_len..(j + 1) * row_len],
                );
                let target = if i == j { 1.0_f32 } else { 0.0_f32 };
                max_err = max_err.max((d - target).abs());
            }
        }
        assert!(
            max_err < 1e-5,
            "rtrl_pool_w must have 4 orthonormal rows: max |W W^T - I_4| = {max_err}"
        );
    }

    #[test]
    fn test_symmetric_weight_seu_nan_scrub() {
        // 1. MicroContinualCore ±1.25 bit-30 Weight-SEU creates genuine NaN/Inf then scrubs cleanly
        let mut cfg = MicroCoreConfig::numeric_stream(4, 77);
        cfg.rtrl_sectors = 8;
        cfg.rtrl_slow_sectors = 8;
        cfg.rtrl_max_in = 4;
        cfg.actor_critic_lr = 0.08;
        let mut core = MicroContinualCore::new(cfg);
        core.encoder.freeze_to_i8();
        core.start_episode();
        for k in 0..8usize {
            core.inject_weight_bit_flip(k, 30);
        }
        let bank = &core.banks[core.active_regime];
        let any_bad = |s: &[f32]| s.iter().any(|v| !v.is_finite() || v.abs() > 1e6);
        let had_bad_before = any_bad(&bank.rtrl_params.alpha)
            || any_bad(&bank.rtrl_params.omega)
            || any_bad(&bank.rtrl_params.in_w)
            || any_bad(&bank.rtrl_slow_params.alpha)
            || any_bad(&bank.rtrl_slow_params.omega)
            || any_bad(&bank.rtrl_slow_params.in_w)
            || any_bad(&bank.rtrl_actor_w)
            || any_bad(&bank.rtrl_critic_w)
            || any_bad(&bank.rtrl_slow_actor_w)
            || any_bad(&bank.rtrl_slow_critic_w)
            || any_bad(&bank.rtrl_skip_actor_w)
            || any_bad(&bank.rtrl_skip_critic_w)
            || any_bad(&bank.linear_w)
            || any_bad(&bank.critic_w);
        assert!(had_bad_before, "bit-30 weight injection must create real NaN/huge corruption before scrub");
        let pr = core.decide_vec(&[0.5_f32, -0.3, 0.2, 0.1], false, false, false);
        assert!(pr.get().1.is_finite(), "MicroCore output must stay finite after weight SEU scrub");
        assert!(core.total_seu_faults() >= 8);
        let bank_after = &core.banks[core.active_regime];
        assert!(bank_after.rtrl_params.alpha.iter().all(|v| v.is_finite()));
        assert!(bank_after.rtrl_params.omega.iter().all(|v| v.is_finite()));
        assert!(bank_after.rtrl_params.in_w.iter().all(|v| v.is_finite()));
        assert!(bank_after.rtrl_slow_params.alpha.iter().all(|v| v.is_finite()));
        assert!(bank_after.rtrl_slow_params.omega.iter().all(|v| v.is_finite()));
        assert!(bank_after.rtrl_slow_params.in_w.iter().all(|v| v.is_finite()));
        assert!(bank_after.rtrl_actor_w.iter().all(|v| v.is_finite()));
        assert!(bank_after.rtrl_slow_actor_w.iter().all(|v| v.is_finite()));

        // 2. NonAbelianCayleyRtrl weight SEU scrub
        let mut nac = NonAbelianCayleyRtrl::new(6, 4, 5, 0.25, 19);
        nac.enable_so3_frames_end_to_end();
        nac.inject_weight_bit_flip(0, 30);
        assert!(nac.wg.iter().any(|v| !v.is_finite() || v.abs() > 1e6));
        nac.step_forward(&[1.0_f32, 0.0, 0.0, 0.0]);
        assert!(nac.seu_fault_count >= 1);
        assert!(nac.wg.iter().all(|v| v.is_finite()));
        assert!(nac.out_w.iter().all(|v| v.is_finite()));
        assert!(nac.u.iter().all(|v| v.is_finite()));
        assert!(nac.b.iter().all(|v| v.is_finite()));

        // 3. Guarded baselines (+FinGuard) scrub Weight-SEU cleanly
        let mut deep = DeepRtrrlLruPhBaseline::new(4, 12, 12, 4, 2, 0.14, 2.2, 25, 5);
        deep.set_state_weight_guard(true);
        deep.start_episode();
        deep.inject_weight_bit_flip(0, 30);
        let _ = deep.decide_vec(&[1.0, 0.0, 0.0, 0.0], false, false, false);
        assert!(!deep.had_nan() && deep.total_seu_faults() >= 1);

        let mut ssm = OnlineSelectiveSsmPhBaseline::new(4, 16, 4, 2, 0.14, 2.2, 25, 5);
        ssm.set_state_weight_guard(true);
        ssm.start_episode();
        ssm.inject_weight_bit_flip(0, 30);
        let _ = ssm.decide_vec(&[1.0, 0.0, 0.0, 0.0], false, false, false);
        assert!(!ssm.had_nan() && ssm.total_seu_faults() >= 1);

        let mut gru = OnlineGruPerStepPhBaseline::new(4, 16, 4, 2, 0.14, 2.2, 25, 5);
        gru.set_state_weight_guard(true);
        gru.start_episode();
        gru.inject_weight_bit_flip(0, 30);
        let _ = gru.decide_vec(&[1.0, 0.0, 0.0, 0.0], false, false, false);
        assert!(!gru.had_nan() && gru.total_seu_faults() >= 1);
    }

    #[test]
    fn test_gru_supervised_zero_rl_ram_and_matched_ph() {
        let mut sup = OnlineGruPerStepPhBaseline::new_supervised(4, 32, 4, 4, 4, 0.04, 2.2, 4, 1);
        let sup1 = OnlineGruPerStepPhBaseline::new_supervised(4, 32, 4, 4, 1, 0.04, 2.2, 4, 1);
        let rl = OnlineGruPerStepPhBaseline::new(4, 32, 4, 4, 0.04, 2.2, 4, 1);
        assert!(sup.banks[0].is_supervised && sup1.banks[0].is_supervised);
        assert!(
            sup1.memory_footprint_bytes() < sup.memory_footprint_bytes(),
            "bptt_len=1 should avoid bptt_len=4 ring overhead"
        );
        assert!(
            sup.memory_footprint_bytes() < rl.memory_footprint_bytes(),
            "supervised GRU must allocate zero RL trace RAM: sup={} vs rl={}",
            sup.memory_footprint_bytes(),
            rl.memory_footprint_bytes()
        );
        sup.set_probe_accept_threshold(0.60);
        for t in 0..20usize {
            let p = sup.step_supervised(&[0.5, -0.2, 0.1, 0.3], t % 4, true, 1);
            assert!(p < 4);
        }
        assert!(!sup.had_nan());
    }

    #[test]
    fn test_forensic_audit_fixes() {
        // 1. DeepRtrrlLruPhBaseline: zero dead RL-head RAM, proper guard propagation, no SEU double-count, had_nan sees fast/slow
        let mut deep = DeepRtrrlLruPhBaseline::new(4, 12, 12, 4, 2, 0.14, 2.2, 25, 11);
        assert_eq!(deep.banks[0].fast.num_actions, 0);
        assert_eq!(deep.banks[0].slow.num_actions, 0);
        deep.set_state_weight_guard(true);
        assert!(deep.banks[0].fast.enable_state_weight_guard);
        assert!(deep.banks[0].slow.enable_state_weight_guard);
        deep.start_episode();
        deep.banks[0].fast.h_re[0] = f32::NAN;
        let _ = deep.decide_vec(&[0.5, -0.2, 0.1, 0.3], false, false, false);
        assert_eq!(deep.total_seu_faults(), 1, "SEU fault in fast layer must be counted exactly once, not doubled");
        assert!(!deep.had_nan());

        let mut deep_un = DeepRtrrlLruPhBaseline::new(4, 12, 12, 4, 2, 0.14, 2.2, 25, 12);
        deep_un.set_state_weight_guard(false);
        deep_un.start_episode();
        deep_un.banks[0].slow.h_im[0] = f32::NAN;
        let _ = deep_un.decide_vec(&[0.5, -0.2, 0.1, 0.3], false, false, false);
        assert!(deep_un.had_nan(), "DeepRtrrlLruPhBaseline::had_nan must detect NaN in slow layer");

        // 2. MicroContinualCore 2L State-SEU hits Band 2 as well as Band 1
        let mut cfg = MicroCoreConfig::numeric_stream(4, 13);
        cfg.half_sectors = 0;
        cfg.rtrl_sectors = 4;
        cfg.rtrl_slow_sectors = 4;
        cfg.rtrl_max_in = 4;
        let mut core = MicroContinualCore::new(cfg);
        core.start_episode();
        let nf = core.rtrl_state.u.len();
        core.rtrl_slow_state.u[0] = 1.25;
        core.inject_register_bit_flip(nf, 30);
        assert!(
            !core.rtrl_slow_state.u[0].is_finite() || core.rtrl_slow_state.u[0].abs() > 1e6,
            "inject_register_bit_flip at offset nf must hit Band 2 (rtrl_slow_state.u)"
        );
        let _ = core.decide_vec(&[0.5, -0.2, 0.1, 0.3], false, false, false);
        assert!(core.rtrl_slow_state.u.iter().all(|v| v.is_finite()));

        // 3. Welford stats freeze on runtime raw NaN/spike even with seu_nan=false, seu_spikes=false
        let mut cfg_w = MicroCoreConfig::numeric_stream(2, 14);
        cfg_w.standardize_vec = true;
        let mut core_w = MicroContinualCore::new(cfg_w);
        core_w.start_episode();
        let _ = core_w.decide_vec(&[1.0, 2.0], true, false, false);
        let n_before = core_w.scratch.feat_count;
        let _ = core_w.decide_vec(&[f32::NAN, 1.0], true, false, false);
        assert_eq!(core_w.scratch.feat_count, n_before, "Welford must freeze on raw NaN input");
        let _ = core_w.decide_vec(&[1e25, 1.0], true, false, false);
        assert_eq!(core_w.scratch.feat_count, n_before, "Welford must freeze on raw spike input");

        // 4. GruPerStepBank supervised mode honestly propagates NaN when guard is OFF (no fast_dot masking)
        let mut gru_sup = OnlineGruPerStepPhBaseline::new_supervised(4, 16, 4, 2, 4, 0.04, 2.2, 4, 15);
        gru_sup.set_state_weight_guard(false);
        gru_sup.banks[0].gru.h_cur[0] = f32::NAN;
        let _ = gru_sup.step_supervised(&[0.5, -0.2, 0.1, 0.3], 1, false, 1);
        assert!(
            gru_sup.had_nan(),
            "Unguarded supervised GRU must propagate state/weight NaN honestly via fast_dot_raw"
        );
    }

    #[test]
    fn test_rational_p4_softmax_and_pos_gate_derivatives() {
        // 1. Shift invariance, positivity, sum=1, and exact log-derivative psi for rational_probs_p4_into
        let logits = [-3.2_f32, 0.5, 2.1, -0.8, 1.4];
        let shifted: Vec<f32> = logits.iter().map(|v| v + 12.5).collect();
        let (mut p1, mut psi1) = (vec![0.0_f32; 5], vec![0.0_f32; 5]);
        let (mut p2, mut psi2) = (vec![0.0_f32; 5], vec![0.0_f32; 5]);
        math::rational_probs_p4_into(&logits, &mut p1, &mut psi1);
        math::rational_probs_p4_into(&shifted, &mut p2, &mut psi2);
        let sum_p: f32 = p1.iter().sum();
        assert!((sum_p - 1.0).abs() < 1e-5, "p4 probs must sum to 1: {sum_p}");
        for i in 0..5 {
            assert!(p1[i] > 0.0 && p1[i].is_finite());
            assert!((p1[i] - p2[i]).abs() < 1e-5, "p4 must be shift-invariant");
            assert!((psi1[i] - psi2[i]).abs() < 1e-5, "p4 psi must be shift-invariant");
        }
        // Check exact finite-diff of log(w(z)) for non-max entries (z < 0)
        let eps = 1e-3_f32;
        for &z in &[-5.0_f32, -2.0, -0.7, -0.1] {
            let mut lp = [0.0_f32; 2];
            let mut lm = [0.0_f32; 2];
            let mut pp = [0.0_f32; 2];
            let mut pm = [0.0_f32; 2];
            let mut psip = [0.0_f32; 2];
            lp[0] = 0.0; lp[1] = z + eps;
            lm[0] = 0.0; lm[1] = z - eps;
            math::rational_probs_p4_into(&lp, &mut pp, &mut psip);
            math::rational_probs_p4_into(&lm, &mut pm, &mut psip);
            let mut p0 = [0.0_f32; 2];
            let mut psi0 = [0.0_f32; 2];
            math::rational_probs_p4_into(&[0.0, z], &mut p0, &mut psi0);
            // d/dz log(p_1 / p_0) == d/dz log(w(z)) == psi(z)
            let fd = ((pp[1] / pp[0]).ln() - (pm[1] / pm[0]).ln()) / (2.0 * eps);
            assert!(
                (fd - psi0[1]).abs() < 2e-3,
                "p4 log-derivative mismatch at z={z}: fd={fd} psi={}",
                psi0[1]
            );
        }

        // 2. Exact C^1 derivative of rational_pos_and_deriv
        for &z in &[-2.5_f32, -0.8, -0.1, 0.2, 0.7] {
            let (vp, _) = math::rational_pos_and_deriv(z + eps);
            let (vm, _) = math::rational_pos_and_deriv(z - eps);
            let (v0, dv) = math::rational_pos_and_deriv(z);
            let fd = (vp - vm) / (2.0 * eps);
            assert!(v0 >= 0.05 && v0 <= 2.0);
            assert!(
                (fd - dv).abs() < 2e-3,
                "rational_pos_and_deriv mismatch at z={z}: fd={fd} an={dv}"
            );
        }
    }

    #[test]
    fn test_algebraic_selective_cayley_exact_rtrl_gradients_and_seu() {
        let seq = [
            [0.7_f32, -0.3],
            [-0.4, 0.8],
            [0.2, 0.5],
            [-0.9, -0.2],
        ];
        let mut exact = AlgebraicSelectiveCayley::new(2, 4, 3, true, 123);
        for x in &seq {
            exact.step_forward(x);
        }
        let run_h = |mut m: AlgebraicSelectiveCayley| -> (Vec<f32>, Vec<f32>) {
            for x in &seq {
                m.step_forward(x);
            }
            (m.h_re, m.h_im)
        };
        let eps = 1e-3_f32;
        let mut max_err = 0.0_f32;
        for j in 0..4usize {
            let mut mp = AlgebraicSelectiveCayley::new(2, 4, 3, true, 123);
            let mut mm = AlgebraicSelectiveCayley::new(2, 4, 3, true, 123);
            mp.alpha[j] += eps; mm.alpha[j] -= eps;
            let (pr, pi) = run_h(mp); let (mr, mi) = run_h(mm);
            max_err = max_err.max(((pr[j] - mr[j]) / (2.0 * eps) - exact.p_a_re[j]).abs());
            max_err = max_err.max(((pi[j] - mi[j]) / (2.0 * eps) - exact.p_a_im[j]).abs());

            let mut mp = AlgebraicSelectiveCayley::new(2, 4, 3, true, 123);
            let mut mm = AlgebraicSelectiveCayley::new(2, 4, 3, true, 123);
            mp.omega[j] += eps; mm.omega[j] -= eps;
            let (pr, pi) = run_h(mp); let (mr, mi) = run_h(mm);
            max_err = max_err.max(((pr[j] - mr[j]) / (2.0 * eps) - exact.p_w_re[j]).abs());
            max_err = max_err.max(((pi[j] - mi[j]) / (2.0 * eps) - exact.p_w_im[j]).abs());

            let mut mp = AlgebraicSelectiveCayley::new(2, 4, 3, true, 123);
            let mut mm = AlgebraicSelectiveCayley::new(2, 4, 3, true, 123);
            mp.b_delta[j] += eps; mm.b_delta[j] -= eps;
            let (pr, pi) = run_h(mp); let (mr, mi) = run_h(mm);
            max_err = max_err.max(((pr[j] - mr[j]) / (2.0 * eps) - exact.p_bd_re[j]).abs());
            max_err = max_err.max(((pi[j] - mi[j]) / (2.0 * eps) - exact.p_bd_im[j]).abs());

            for k in 0..2usize {
                let off = j * 2 + k;
                let mut mp = AlgebraicSelectiveCayley::new(2, 4, 3, true, 123);
                let mut mm = AlgebraicSelectiveCayley::new(2, 4, 3, true, 123);
                mp.w_delta[off] += eps; mm.w_delta[off] -= eps;
                let (pr, pi) = run_h(mp); let (mr, mi) = run_h(mm);
                max_err = max_err.max(((pr[j] - mr[j]) / (2.0 * eps) - exact.p_wd_re[off]).abs());
                max_err = max_err.max(((pi[j] - mi[j]) / (2.0 * eps) - exact.p_wd_im[off]).abs());

                let mut mp = AlgebraicSelectiveCayley::new(2, 4, 3, true, 123);
                let mut mm = AlgebraicSelectiveCayley::new(2, 4, 3, true, 123);
                mp.w_bre[off] += eps; mm.w_bre[off] -= eps;
                let (pr, pi) = run_h(mp); let (mr, mi) = run_h(mm);
                max_err = max_err.max(((pr[j] - mr[j]) / (2.0 * eps) - exact.p_br_re[off]).abs());
                max_err = max_err.max(((pi[j] - mi[j]) / (2.0 * eps) - exact.p_br_im[off]).abs());

                let mut mp = AlgebraicSelectiveCayley::new(2, 4, 3, true, 123);
                let mut mm = AlgebraicSelectiveCayley::new(2, 4, 3, true, 123);
                mp.w_bim[off] += eps; mm.w_bim[off] -= eps;
                let (pr, pi) = run_h(mp); let (mr, mi) = run_h(mm);
                max_err = max_err.max(((pr[j] - mr[j]) / (2.0 * eps) - exact.p_bi_re[off]).abs());
                max_err = max_err.max(((pi[j] - mi[j]) / (2.0 * eps) - exact.p_bi_im[off]).abs());
            }
        }
        assert!(
            max_err < 1e-3,
            "AlgebraicSelectiveCayley exact RTRL gradient mismatch: max_err={max_err}"
        );

        // Bit-30 State + Weight SEU self-healing check
        let mut m = AlgebraicSelectiveCayley::new(2, 8, 3, true, 42);
        m.step_forward(&[0.5, -0.4]);
        m.h_re[0] = 1.25;
        m.inject_register_bit_flip(0, 30);
        m.inject_weight_bit_flip(0, 30);
        m.inject_weight_bit_flip(1, 30);
        let pred = m.step_and_learn_supervised(&[0.5, -0.4], Some(1), 0.01, 0.005);
        assert!(pred < 3);
        assert!(m.total_seu_faults() >= 3);
        assert!(m.h_re.iter().all(|v| v.is_finite()) && m.h_im.iter().all(|v| v.is_finite()));
        assert!(m.out_w.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_non_abelian_proto_metric_orthogonality_and_seu() {
        let mut nac = NonAbelianCayleyRtrl::new(12, 4, 10, 0.25, 7);
        nac.enable_so3_proto_metric(10.0, 1.2, 0.08);
        assert!(nac.proto_metric_mode && nac.frame_coupled);
        for step in 0..50usize {
            let mut x = [0.0_f32; 4];
            x[step % 4] = 1.0;
            let _ = nac.step_and_learn_supervised(&x, Some(step % 10), 0.02, 0.015);
        }
        // Live unnormalized orthogonality over 200 steps without renormalization
        nac.reset();
        for step in 0..200usize {
            let mut x = [0.0_f32; 4];
            x[(step * 3 + 1) % 4] = 1.0;
            let _ = nac.step_and_learn_supervised(&x, None, 0.0, 0.0);
        }
        let drift = nac.max_orth_drift();
        assert!(
            drift < 5e-4,
            "Proto-metric SO(3) live unnormalized drift too large after 200 steps: {drift}"
        );
        // Bit-30 Weight-SEU on both wg and prototypes (out_w)
        nac.inject_weight_bit_flip(0, 30);
        nac.inject_weight_bit_flip(1, 30);
        let p = nac.step_and_learn_supervised(&[1.0, 0.0, 0.0, 0.0], None, 0.0, 0.0);
        assert!(p < 10 && nac.seu_fault_count >= 2);
        assert!(nac.wg.iter().all(|v| v.is_finite()) && nac.out_w.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_parallel_bank_probe_causal_matching() {
        let mut cfg = MicroCoreConfig::numeric_stream(2, 55);
        cfg.half_sectors = 0;
        cfg.rtrl_sectors = 4;
        cfg.rtrl_slow_sectors = 0;
        cfg.rtrl_max_in = 2;
        cfg.rtrl_use_skip_x = true;
        cfg.linear_lr = 0.12;
        cfg.rtrl_lr = 0.0;
        cfg.max_regimes = 4;
        cfg.ph_delta = 0.05;
        cfg.ph_threshold = 1.5;
        cfg.ph_warmup = 4;
        cfg.parallel_bank_probe = true;
        let mut core = MicroContinualCore::new(cfg);
        core.start_episode();
        // Train bank 0 on y = (x0 >= 0), bank 1 on y = (x0 < 0), bank 2 on y = (x1 >= 0)
        for t in 0..80usize {
            let x0 = if (t & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
            let x1 = if ((t >> 1) & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
            let y = usize::from(x0 >= 0.0);
            let p = core.decide_vec(&[x0, x1], true, false, false).get().0;
            let _ = core.end_supervised_step(p, y);
        }
        for t in 0..80usize {
            let x0 = if (t & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
            let x1 = if ((t >> 1) & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
            let y = usize::from(x0 < 0.0);
            let p = core.decide_vec(&[x0, x1], true, false, false).get().0;
            let _ = core.end_supervised_step(p, y);
        }
        for t in 0..80usize {
            let x0 = if (t & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
            let x1 = if ((t >> 1) & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
            let y = usize::from(x1 >= 0.0);
            let p = core.decide_vec(&[x0, x1], true, false, false).get().0;
            let _ = core.end_supervised_step(p, y);
        }
        // Return to regime 0 (y = x0 >= 0): parallel probe must recover high accuracy within 30 steps
        let mut hits = 0usize;
        for t in 0..30usize {
            let x0 = if (t & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
            let x1 = if ((t >> 1) & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
            let y = usize::from(x0 >= 0.0);
            let p = core.decide_vec(&[x0, x1], true, false, false).get().0;
            if t >= 10 && p == y { hits += 1; }
            let _ = core.end_supervised_step(p, y);
        }
        assert!(hits >= 16, "Parallel bank probe failed to recover returning regime: {hits}/20");
    }

    #[test]
    fn test_ph_high_threshold_and_gru_truncation_bootstrap() {
        let mut ph = PageHinkleyDetector::new(0.12, 99.0, 10);
        assert!((ph.threshold - 99.0).abs() < 1e-6, "PH threshold=99 must not be silently clamped to 8.0");
        for _ in 0..20 {
            assert!(!ph.observe(1.0));
        }
        let mut tripped = false;
        for _ in 0..80 {
            if ph.observe(0.0) {
                tripped = true;
            }
        }
        assert!(!tripped, "PH with threshold=99 should stay disabled over 80 zero-return episodes");

        // Verify OnlineGruBaseline::end_episode_truncated bootstraps instead of panicking/NaNing
        let mut gru = baselines::OnlineGruBaseline::new(2, 8, 2, 7);
        gru.start_episode();
        for _ in 0..5 {
            let _ = gru.decide_vec(&[0.1, -0.1], true, false, false);
            gru.observe_step_reward(0.06);
        }
        gru.end_episode_truncated(0.30, true, true);
        assert!(!gru.had_nan && gru.h_cur.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_regression_contour_exact_gradients_and_seu() {
        let seq = [
            [0.6_f32, -0.4, 0.2],
            [-0.3, 0.8, -0.5],
            [0.5, 0.2, 0.7],
            [-0.7, -0.1, 0.3],
        ];
        let target = 0.72_f32;
        let eps = 1e-3_f32;

        // 1. MicroContinualCore PURE 2L+Skip regression (num_actions=1, max_regimes=1 -> zero dead RL/critic RAM)
        let mut cfg = MicroCoreConfig::numeric_stream(1, 314);
        cfg.half_sectors = 0;
        cfg.rtrl_sectors = 4;
        cfg.rtrl_slow_sectors = 4;
        cfg.rtrl_max_in = 3;
        cfg.rtrl_use_skip_x = true;
        cfg.rtrl_grid_mode = 1;
        cfg.standardize_vec = false;
        cfg.max_regimes = 1;
        cfg.linear_lr = 0.05;
        cfg.linear_wd = 0.0;
        cfg.rtrl_lr = 0.001;
        cfg.trace_gamma = 0.0; // instantaneous exact RTRL trace = d(-L)/dtheta
        let base = MicroContinualCore::new(cfg);
        assert_eq!(base.banks.len(), 1);
        assert!(base.banks[0].rtrl_critic_w.is_empty(), "pure regression must allocate zero dead critic_w");
        assert!(base.banks[0].rtrl_slow_critic_w.is_empty());
        assert!(base.banks[0].rtrl_skip_critic_w.is_empty());

        let eval_loss = |mut m: MicroContinualCore| -> f32 {
            m.start_episode();
            let mut p = 0.0_f32;
            for x in &seq {
                p = *m.decide_reg_vec(x, false, false, false).get();
            }
            0.5 * (p - target) * (p - target)
        };

        // Compute analytic gradient via one end_reg_step
        let mut m0 = base.clone();
        m0.start_episode();
        let mut pred0 = 0.0_f32;
        for x in &seq {
            pred0 = *m0.decide_reg_vec(x, false, false, false).get();
        }
        let w_r1_before = m0.banks[0].rtrl_actor_w[2];
        let w_r2_before = m0.banks[0].rtrl_slow_actor_w[1];
        let w_sk_before = m0.banks[0].rtrl_skip_actor_w[0];
        let _ = m0.end_reg_step(target);
        // Check readout exact gradient: delta_w = lin_lr * (-dL/dw)
        let an_r1 = (m0.banks[0].rtrl_actor_w[2] - w_r1_before) / cfg.linear_lr;
        let an_r2 = (m0.banks[0].rtrl_slow_actor_w[1] - w_r2_before) / cfg.linear_lr;
        let an_sk = (m0.banks[0].rtrl_skip_actor_w[0] - w_sk_before) / cfg.linear_lr;

        // Finite-diff for readout weights
        let mut mp = base.clone(); mp.banks[0].rtrl_actor_w[2] += eps;
        let mut mm = base.clone(); mm.banks[0].rtrl_actor_w[2] -= eps;
        let fd_r1 = -(eval_loss(mp) - eval_loss(mm)) / (2.0 * eps);
        assert!((an_r1 - fd_r1).abs() < 2e-3, "Band1 readout grad mismatch: an={an_r1} fd={fd_r1}");

        let mut mp = base.clone(); mp.banks[0].rtrl_slow_actor_w[1] += eps;
        let mut mm = base.clone(); mm.banks[0].rtrl_slow_actor_w[1] -= eps;
        let fd_r2 = -(eval_loss(mp) - eval_loss(mm)) / (2.0 * eps);
        assert!((an_r2 - fd_r2).abs() < 2e-3, "Band2 readout grad mismatch: an={an_r2} fd={fd_r2}");

        let mut mp = base.clone(); mp.banks[0].rtrl_skip_actor_w[0] += eps;
        let mut mm = base.clone(); mm.banks[0].rtrl_skip_actor_w[0] -= eps;
        let fd_sk = -(eval_loss(mp) - eval_loss(mm)) / (2.0 * eps);
        assert!((an_sk - fd_sk).abs() < 2e-3, "Skip readout grad mismatch: an={an_sk} fd={fd_sk}");

        // Check Band 1 & Band 2 RTRL exact traces (e_alpha, e_omega, e_w) vs finite differences
        // Note: Band 1 affects output directly AND through Band 2 pool; test Band 2 exact RTRL first (direct to readout)
        let an_s2_a = m0.rtrl_slow_state.e_alpha[0];
        let an_s2_w = m0.rtrl_slow_state.e_omega[0];
        let mut mp = base.clone(); mp.banks[0].rtrl_slow_params.alpha[0] += eps;
        let mut mm = base.clone(); mm.banks[0].rtrl_slow_params.alpha[0] -= eps;
        let fd_s2_a = -(eval_loss(mp) - eval_loss(mm)) / (2.0 * eps);
        assert!((an_s2_a - fd_s2_a).abs() < 3e-3, "Band2 RTRL alpha grad mismatch: an={an_s2_a} fd={fd_s2_a}");

        let mut mp = base.clone(); mp.banks[0].rtrl_slow_params.omega[0] += eps;
        let mut mm = base.clone(); mm.banks[0].rtrl_slow_params.omega[0] -= eps;
        let fd_s2_w = -(eval_loss(mp) - eval_loss(mm)) / (2.0 * eps);
        assert!((an_s2_w - fd_s2_w).abs() < 3e-3, "Band2 RTRL omega grad mismatch: an={an_s2_w} fd={fd_s2_w}");

        // Band 1 exact RTRL (with Band 2 disabled so no un-backpropped cross-band path)
        let mut cfg1 = cfg;
        cfg1.rtrl_slow_sectors = 0;
        let base1 = MicroContinualCore::new(cfg1);
        let mut m1 = base1.clone();
        m1.start_episode();
        for x in &seq { let _ = m1.decide_reg_vec(x, false, false, false); }
        let _ = m1.end_reg_step(target);
        let eval1 = |mut m: MicroContinualCore| -> f32 {
            m.start_episode();
            let mut p = 0.0_f32;
            for x in &seq { p = *m.decide_reg_vec(x, false, false, false).get(); }
            0.5 * (p - target) * (p - target)
        };
        let mut mp = base1.clone(); mp.banks[0].rtrl_params.alpha[1] += eps;
        let mut mm = base1.clone(); mm.banks[0].rtrl_params.alpha[1] -= eps;
        let fd_s1_a = -(eval1(mp) - eval1(mm)) / (2.0 * eps);
        assert!((m1.rtrl_state.e_alpha[1] - fd_s1_a).abs() < 3e-3, "Band1 RTRL alpha grad mismatch");

        let mut mp = base1.clone(); mp.banks[0].rtrl_params.omega[1] += eps;
        let mut mm = base1.clone(); mm.banks[0].rtrl_params.omega[1] -= eps;
        let fd_s1_w = -(eval1(mp) - eval1(mm)) / (2.0 * eps);
        assert!((m1.rtrl_state.e_omega[1] - fd_s1_w).abs() < 3e-3, "Band1 RTRL omega grad mismatch");

        // 2. MicroContinualCore Regression SEU Survival (Sensor + State Band1/Band2 + Weight)
        let mut m_seu = base.clone();
        m_seu.start_episode();
        let _ = m_seu.decide_reg_vec(&seq[0], false, false, false);
        let f0 = m_seu.total_seu_faults();
        // Sensor SEU
        let p_sens = *m_seu.decide_reg_vec(&[f32::NAN, 1e30, 0.2], false, false, false).get();
        assert!(p_sens.is_finite() && m_seu.total_seu_faults() > f0);
        // State SEU on Band 1 & Band 2
        let f1 = m_seu.total_seu_faults();
        m_seu.rtrl_state.u[0] = 1.25;
        m_seu.inject_register_bit_flip(0, 30);
        m_seu.rtrl_slow_state.u[0] = 1.25;
        let nf = m_seu.rtrl_state.u.len();
        m_seu.inject_register_bit_flip(nf, 30);
        let p_st = *m_seu.decide_reg_vec(&seq[1], false, false, false).get();
        assert!(p_st.is_finite() && m_seu.total_seu_faults() >= f1 + 2);
        // Weight SEU
        let f2 = m_seu.total_seu_faults();
        m_seu.inject_weight_bit_flip(7, 30);
        let p_wt = *m_seu.decide_reg_vec(&seq[2], false, false, false).get();
        assert!(p_wt.is_finite() && m_seu.total_seu_faults() > f2);
        let _ = m_seu.end_reg_step(target);
        assert!(m_seu.slow_envelope_norm().is_finite() && pred0.is_finite());

        // 3. AlgebraicSelectiveCayley Regression Convergence + SEU Survival
        let mut asc = rtrl::AlgebraicSelectiveCayley::new(3, 6, 1, true, 42);
        let mut loss_start = 0.0_f32;
        let mut loss_end = 0.0_f32;
        for ep in 0..40usize {
            asc.reset();
            let mut p = 0.0_f32;
            for x in &seq {
                p = asc.step_and_learn_regression(x, Some(target), 0.02, 0.005);
            }
            if ep == 0 { loss_start = (p - target).abs(); }
            if ep == 39 { loss_end = (p - target).abs(); }
        }
        assert!(loss_end < loss_start * 0.35 + 0.05, "AlgSelCayley regression failed to converge: {loss_start} -> {loss_end}");
        asc.h_re[0] = 1.25;
        asc.inject_register_bit_flip(0, 30);
        asc.inject_weight_bit_flip(3, 30);
        let p_asc = asc.step_and_learn_regression(&[f32::NAN, 1e30, 0.1], None, 0.0, 0.0);
        assert!(p_asc.is_finite() && asc.total_seu_faults() >= 3);

        // 4. Symmetric Baseline (+Clamp vs +FinGuard) check: no sticky had_nan under +FinGuard
        let mut lru_c = baselines::DeepRtrrlLruBank::new_supervised(3, 4, 4, 1, 7);
        lru_c.set_state_weight_guard(false);
        lru_c.fast.h_re[0] = f32::from_bits(1.25_f32.to_bits() ^ (1u32 << 30));
        let p_c = lru_c.step_and_learn_regression(&seq[0], None, 0.0, 0.0);
        assert!(!p_c.is_finite(), "Unguarded (+Clamp) Deep LRU must honestly fail on State-SEU");

        let mut lru_g = baselines::DeepRtrrlLruBank::new_supervised(3, 4, 4, 1, 7);
        lru_g.set_state_weight_guard(true);
        lru_g.fast.h_re[0] = f32::from_bits(1.25_f32.to_bits() ^ (1u32 << 30));
        lru_g.inject_weight_bit_flip(2, 30);
        let p_g = lru_g.step_and_learn_regression(&seq[0], Some(target), 0.01, 0.002);
        assert!(p_g.is_finite() && !lru_g.had_nan && lru_g.total_seu_faults() >= 2);
    }

    #[test]
    fn test_ph_unclamped_high_threshold_and_single_bank_no_freeze() {
        let mut ph_hi = PageHinkleyDetector::new(0.05, 9999.0, 4);
        assert_eq!(ph_hi.threshold, 9999.0);
        let ph_max = PageHinkleyDetector::new(0.05, 50_000.0, 4);
        assert_eq!(ph_max.threshold, 50_000.0);
        for _ in 0..50usize {
            assert!(!ph_hi.observe(0.0));
        }

        let mut cfg = MicroCoreConfig::numeric_stream(2, 11);
        cfg.max_regimes = 1;
        cfg.half_sectors = 0;
        cfg.rtrl_sectors = 6;
        cfg.rtrl_slow_sectors = 0;
        cfg.rtrl_max_in = 2;
        cfg.rtrl_lr = 0.02;
        cfg.linear_lr = 0.08;
        cfg.ph_delta = 0.01;
        cfg.ph_threshold = 0.1;
        cfg.ph_warmup = 2;
        cfg.standardize_vec = false;
        let mut core = MicroContinualCore::new(cfg);
        core.start_episode();

        let mut w_at_30 = Vec::new();
        for t in 0..45usize {
            let x = [0.6_f32 - 0.1 * ((t % 3) as f32), -0.4_f32 + 0.15 * ((t % 4) as f32)];
            let p = core.decide_vec(&x, true, false, false).get().0;
            let _ = core.end_supervised_step(p, 1 - p);
            if t == 30 {
                w_at_30 = core.banks[0].rtrl_actor_w.clone();
            }
        }
        assert_eq!(core.total_switches, 0);
        assert_eq!(core.active_regime, 0);
        assert_ne!(
            core.banks[0].rtrl_actor_w, w_at_30,
            "Single-bank MicroContinualCore must not freeze weights when max_regimes == 1"
        );
    }

    #[test]
    fn test_algebraic_selective_cayley_split_causal_regression() {
        let mut m1 = rtrl::AlgebraicSelectiveCayley::new(3, 6, 1, true, 123);
        let mut m2 = rtrl::AlgebraicSelectiveCayley::new(3, 6, 1, true, 123);
        for t in 0..25usize {
            let s = (t as f32) * 0.37;
            let x = [s.sin() * 0.7, (s * 1.9).cos() * 0.5, 0.25 - 0.03 * (t as f32)];
            let y = 0.4 * x[0] - 0.3 * x[1] + 0.15;
            let p1 = m1.step_and_learn_regression(&x, Some(y), 0.015, 0.004);
            let p2 = m2.decide_regression(&x);
            assert_eq!(p1.to_bits(), p2.to_bits(), "step {t} prediction mismatch");
            m2.learn_regression(y, 0.015, 0.004);
        }
        assert_eq!(m1.out_w, m2.out_w);
        assert_eq!(m1.h_re, m2.h_re);
    }

    #[test]
    fn test_readonly_k_ahead_rollout_exact_equivalence() {
        let train_seq: [([f32; 3], usize); 6] = [
            ([0.8, -0.2, 0.3], 0),
            ([-0.5, 0.9, 0.1], 1),
            ([0.2, 0.4, -0.7], 2),
            ([0.6, -0.6, 0.5], 0),
            ([-0.7, 0.3, 0.4], 1),
            ([0.1, -0.2, -0.8], 2),
        ];
        let win_owned: [[f32; 3]; 4] = [
            ([0.5, 0.1, -0.4]),
            ([-0.3, 0.7, 0.2]),
            ([0.4, -0.5, 0.6]),
            ([-0.6, -0.2, 0.3]),
        ];
        let window: Vec<&[f32]> = win_owned.iter().map(|v| &v[..]).collect();

        // 1. MicroContinualCore supervised (standardize_vec + delta_obs_scale + Band 0+1+2 + skip + kernels)
        let mut cfg = MicroCoreConfig::numeric_stream(3, 55);
        cfg.num_actions = 3;
        cfg.standardize_vec = true;
        cfg.delta_obs_scale = 5.0;
        cfg.half_sectors = 8;
        cfg.rtrl_sectors = 6;
        cfg.rtrl_slow_sectors = 4;
        cfg.rtrl_max_in = 6;
        cfg.rtrl_use_skip_x = true;
        cfg.linear_mix = 0.65;
        cfg.linear_lr = 0.08;
        cfg.rtrl_lr = 0.015;
        let mut core = MicroContinualCore::new(cfg);
        core.start_episode();
        for &(x, y) in &train_seq {
            let p = core.decide_vec(&x, true, false, false).get().0;
            let _ = core.end_supervised_step(p, y);
        }
        let bel_before = core.belief.state.clone();
        let r1_before = core.rtrl_state.u.clone();
        let mu_before = core.scratch.feat_mean.clone();
        let cnt_before = core.scratch.feat_count;
        let ro_core = core.predict_vec_k_ahead_readonly(&window);
        assert_eq!(core.belief.state, bel_before);
        assert_eq!(core.rtrl_state.u, r1_before);
        assert_eq!(core.scratch.feat_mean, mu_before);
        assert_eq!(core.scratch.feat_count, cnt_before);
        let mut c_clone = core.clone();
        let mut clone_core_pred = 0usize;
        for &w in &window {
            clone_core_pred = c_clone.decide_vec(w, false, false, false).get().0;
        }
        assert_eq!(ro_core, clone_core_pred);
        let ro0 = core.predict_vec_k_ahead_readonly(&[]);
        assert!(ro0 < cfg.num_actions);

        // 2. MicroContinualCore Actor-Critic (with RTRL & without RTRL)
        for rtrl_sec in [4usize, 0usize] {
            let mut ac_cfg = cfg;
            ac_cfg.actor_critic_lr = 0.05;
            ac_cfg.rtrl_sectors = rtrl_sec;
            ac_cfg.rtrl_slow_sectors = 0;
            let mut ac = MicroContinualCore::new(ac_cfg);
            ac.start_episode();
            for &(x, _) in &train_seq {
                let _ = ac.decide_vec(&x, true, false, false);
                ac.observe_step_reward(0.2);
            }
            let r_snap = ac.rtrl_state.u.clone();
            let b_snap = ac.belief.state.clone();
            let ro_ac = ac.predict_vec_k_ahead_readonly(&window);
            assert_eq!(ac.rtrl_state.u, r_snap);
            assert_eq!(ac.belief.state, b_snap);
            let mut ac_c = ac.clone();
            let mut cl_ac = 0usize;
            for &w in &window {
                cl_ac = ac_c.decide_vec(w, false, false, false).get().0;
            }
            assert_eq!(ro_ac, cl_ac);
        }

        // 3. RtrrlLruBaseline
        let mut rl_lru = baselines::RtrrlLruBaseline::new(3, 8, 3, 7);
        rl_lru.start_episode();
        for &(x, _) in &train_seq {
            let _ = rl_lru.decide_vec(&x, true, false, false);
            rl_lru.observe_step_reward(0.1);
        }
        let h_snap = rl_lru.h_re.clone();
        let ro_rl = rl_lru.predict_k_ahead_readonly(&window);
        assert_eq!(rl_lru.h_re, h_snap);
        let mut rl_c = rl_lru.clone();
        let mut cl_rl = 0usize;
        for &w in &window {
            cl_rl = rl_c.decide_vec(w, false, false, false);
        }
        assert_eq!(ro_rl, cl_rl);

        // 4. SelectiveSsmBank & OnlineSelectiveSsmPhBaseline
        let mut ssm = baselines::SelectiveSsmBank::new_supervised(3, 8, 3, 17);
        for &(x, y) in &train_seq {
            let _ = ssm.step_and_learn_supervised(&x, Some(y), 0.05, 0.01);
        }
        let s_snap = ssm.h_re.clone();
        let ro_ssm = ssm.predict_k_ahead_readonly(&window);
        assert_eq!(ssm.h_re, s_snap);
        let mut ssm_c = ssm.clone();
        let mut cl_ssm = 0usize;
        for &w in &window {
            cl_ssm = ssm_c.step_and_learn_supervised(w, None, 0.05, 0.01);
        }
        assert_eq!(ro_ssm, cl_ssm);

        let mut ssm_ph = baselines::OnlineSelectiveSsmPhBaseline::new(3, 8, 3, 3, 0.04, 2.2, 4, 27);
        for &(x, y) in &train_seq {
            let _ = ssm_ph.step_supervised(&x, y, 0.05, 0.01, true, 1);
        }
        let sp_snap = ssm_ph.banks[ssm_ph.active_regime()].h_re.clone();
        let ro_sp = ssm_ph.predict_k_ahead_readonly(&window);
        assert_eq!(ssm_ph.banks[ssm_ph.active_regime()].h_re, sp_snap);
        let mut sp_c = ssm_ph.clone();
        let reg_sp = sp_c.active_regime();
        let mut cl_sp = 0usize;
        for &w in &window {
            cl_sp = sp_c.banks[reg_sp].step_and_learn_supervised(w, None, 0.05, 0.01);
        }
        assert_eq!(ro_sp, cl_sp);

        // 5. DeepRtrrlLruBank & DeepRtrrlLruPhBaseline
        let mut dlru = baselines::DeepRtrrlLruBank::new_supervised(3, 6, 4, 3, 37);
        for &(x, y) in &train_seq {
            let _ = dlru.step_and_learn_supervised(&x, Some(y), 0.05, 0.01);
        }
        let d_snap = dlru.fast.h_re.clone();
        let ro_dl = dlru.predict_k_ahead_readonly(&window);
        assert_eq!(dlru.fast.h_re, d_snap);
        let mut dl_c = dlru.clone();
        let mut cl_dl = 0usize;
        for &w in &window {
            cl_dl = dl_c.step_and_learn_supervised(w, None, 0.05, 0.01);
        }
        assert_eq!(ro_dl, cl_dl);

        let mut dph = baselines::DeepRtrrlLruPhBaseline::new(3, 6, 4, 3, 3, 0.04, 2.2, 4, 47);
        for &(x, y) in &train_seq {
            let _ = dph.step_supervised(&x, y, 0.05, 0.01, true, 1);
        }
        let dp_snap = dph.banks[dph.active_regime()].fast.h_re.clone();
        let ro_dp = dph.predict_k_ahead_readonly(&window);
        assert_eq!(dph.banks[dph.active_regime()].fast.h_re, dp_snap);
        let mut dp_c = dph.clone();
        let reg_dp = dp_c.active_regime();
        let mut cl_dp = 0usize;
        for &w in &window {
            cl_dp = dp_c.banks[reg_dp].step_and_learn_supervised(w, None, 0.05, 0.01);
        }
        assert_eq!(ro_dp, cl_dp);

        // 6. GruPerStepBank & OnlineGruPerStepPhBaseline
        let mut gru = baselines::GruPerStepBank::new_supervised(3, 8, 3, 4, 57);
        for &(x, y) in &train_seq {
            let _ = gru.step_and_learn_supervised(&x, Some(y));
        }
        let g_snap = gru.gru.h_cur.clone();
        let ro_g = gru.predict_k_ahead_readonly(&window);
        assert_eq!(gru.gru.h_cur, g_snap);
        let mut gru_c = gru.clone();
        let mut cl_g = 0usize;
        for &w in &window {
            cl_g = gru_c.step_and_learn_supervised(w, None);
        }
        assert_eq!(ro_g, cl_g);

        let mut gph = baselines::OnlineGruPerStepPhBaseline::new_supervised(3, 8, 3, 3, 4, 0.04, 2.2, 4, 67);
        for &(x, y) in &train_seq {
            let _ = gph.step_supervised(&x, y, true, 1);
        }
        let gp_snap = gph.banks[gph.active_regime()].gru.h_cur.clone();
        let ro_gp = gph.predict_k_ahead_readonly(&window);
        assert_eq!(gph.banks[gph.active_regime()].gru.h_cur, gp_snap);
        let mut gp_c = gph.clone();
        let reg_gp = gp_c.active_regime();
        let mut cl_gp = 0usize;
        for &w in &window {
            cl_gp = gp_c.banks[reg_gp].step_and_learn_supervised(w, None);
        }
        assert_eq!(ro_gp, cl_gp);

        // 7. OnlineLruStream (standardize false and true)
        for &std_flag in &[false, true] {
            let mut lru_s = stream_baselines::OnlineLruStream::new(3, 8, 3, 77);
            lru_s.standardize = std_flag;
            for &(x, y) in &train_seq {
                let _ = lru_s.predict_and_train(&x, y, true);
            }
            let ro_ls = lru_s.predict_k_ahead_readonly(&window);
            let mut ls_c = lru_s.clone();
            let mut cl_ls = 0usize;
            for &w in &window {
                cl_ls = ls_c.predict_and_train(w, 0, false);
            }
            assert_eq!(ro_ls, cl_ls);
        }
    }
}
