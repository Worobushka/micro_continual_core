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

pub use belief::{CayleyBeliefState, EligibilityTraceRing, PageHinkleyDetector};
pub use core::{MicroContinualCore, MicroCoreConfig};
pub use env_hard_pomdp::{
    unlabeled_domain_cloze_logs_noisy, HardPomdpConfig, HardPomdpV2, HardRegime, ParaphraseSplit,
    NUM_ENV_ACTIONS,
};
pub use rtrl::{HierarchicalCayleyRtrl, PlasticCayleyParams, PlasticCayleyState};

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
    fn test_cloze_noisy_synonym_alignment() {
        let mut enc = math::SpinorEncoder::new(32, 10);
        let logs = unlabeled_domain_cloze_logs_noisy();
        enc.adapt_unlabeled_cloze(&logs, 45);
        enc.freeze_to_i8();

        let mut v_canon = vec![0.0_f32; enc.dim];
        let mut v_hold = vec![0.0_f32; enc.dim];
        let mut v_other = vec![0.0_f32; enc.dim];
        let mut tmp = vec![0.0_f32; enc.half_dim];
        let mut sl = Vec::new();

        enc.encode_text_into("node alert disk array blk stall", &mut v_canon, &mut sl, &mut tmp);
        enc.encode_text_into("telemetry nvme volume sector wait high", &mut v_hold, &mut sl, &mut tmp);
        enc.encode_text_into("node alert ledger tx outlier lock", &mut v_other, &mut sl, &mut tmp);

        let hd = enc.half_dim;
        let sim_syn = fast_dot(&v_canon[0..hd], &v_hold[0..hd]);
        let sim_diff = fast_dot(&v_canon[0..hd], &v_other[0..hd]);
        assert!(
            sim_syn > sim_diff + 0.15,
            "Noisy Cloze must align holdout synonyms closer than unrelated domains: syn={:.3}, diff={:.3}",
            sim_syn,
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
}
