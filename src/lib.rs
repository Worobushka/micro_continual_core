// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

pub mod adaptive_od;
pub mod algebra;
pub mod baselines;
pub mod belief;
pub mod core;
pub mod env_hard_pomdp;
pub mod fixed_point;
pub mod math;
pub mod memory;
pub mod stream_baselines;
pub mod rtrl;
pub mod unified;

pub use adaptive_od::*;
pub use fixed_point::*;

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
    AlgebraicSelectiveCayley, HierarchicalCayleyRtrl, NonAbelianCayleyRtrl, NonAbelianO4Rtrl,
    PlasticCayleyParams, PlasticCayleyState,
};
pub use unified::{
    BartlettPageHinkley, CausalInputConditioner, UnifiedCayleyConfig, UnifiedCayleyCore,
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

    #[test]
    fn test_inv_sqrt_newton_f32_ten_order_jump() {
        let test_vals = [
            1e-12_f32, 1e-6, 0.003, 0.5, 0.99, 1.0, 1.5, 2.0, 125.0, 1e6, 1e-6, 1e10, 1e-10, 3.7e-4,
        ];
        for &a in &test_vals {
            let y2 = math::inv_sqrt_newton_2step_f32(a);
            let y3 = math::inv_sqrt_newton_f32(a);
            let exact = 1.0_f32 / a.sqrt();
            assert!(y2.is_finite() && y2 > 0.0, "2-step newton <= 0 for a={a}: {y2}");
            assert!(y3.is_finite() && y3 > 0.0, "3-step newton <= 0 for a={a}: {y3}");
            let rel2 = ((y2 - exact) / exact).abs();
            let rel3 = ((y3 - exact) / exact).abs();
            assert!(rel2 < 1.5e-4, "2-step rel err too big for a={a}: {rel2}");
            assert!(rel3 < 1.5e-6, "3-step rel err too big for a={a}: {rel3}");
        }
        // Резкий скачок мощности при использовании y_prev: reset должен защитить от <= 0
        let y_small = math::inv_sqrt_newton_f32(1e-4);
        let y_jump = math::inv_sqrt_newton_guarded_f32(1e6, y_small);
        assert!(y_jump > 0.0 && y_jump.is_finite());
        let exact_jump = 1.0_f32 / (1e6_f32).sqrt();
        assert!(((y_jump - exact_jump) / exact_jump).abs() < 1e-5);
        assert_eq!(math::inv_sqrt_newton_f32(0.0), 0.0);
        assert_eq!(math::inv_sqrt_newton_f32(-5.0), 0.0);
        assert_eq!(math::inv_sqrt_newton_f32(f32::NAN), 0.0);
    }

    #[test]
    fn test_o4_double_rotor_and_householder_invariants() {
        let gl = (0.42_f32, -0.31, 0.25);
        let gr = (-0.19_f32, 0.54, -0.37);
        let v = (0.5_f32, -0.5, 0.5, 0.5);
        let basis = [
            (1.0_f32, 0.0, 0.0, 0.0),
            (0.0_f32, 1.0, 0.0, 0.0),
            (0.0_f32, 0.0, 1.0, 0.0),
            (0.0_f32, 0.0, 0.0, 1.0),
        ];

        // 1. beta = 0 -> SO(4), det = +1, ортонормированность
        let cols_even: Vec<(f32, f32, f32, f32)> = basis
            .iter()
            .map(|&u| math::householder4_gated(v, 0.0, 1e-10, math::so4_double_rotor_4d(gl, gr, u)))
            .collect();
        let det_even = math::det4x4_cols(cols_even[0], cols_even[1], cols_even[2], cols_even[3]);
        assert!((det_even - 1.0).abs() < 1e-5, "SO(4) det must be +1, got {det_even}");
        for i in 0..4 {
            for j in 0..4 {
                let dot = cols_even[i].0 * cols_even[j].0
                    + cols_even[i].1 * cols_even[j].1
                    + cols_even[i].2 * cols_even[j].2
                    + cols_even[i].3 * cols_even[j].3;
                let tgt = if i == j { 1.0 } else { 0.0 };
                assert!((dot - tgt).abs() < 1e-5);
            }
        }

        // 2. beta = 2 -> O(4) reflection, det = -1, инволюция H_2^2 = I_4
        let cols_odd: Vec<(f32, f32, f32, f32)> = basis
            .iter()
            .map(|&u| math::householder4_gated(v, 2.0, 1e-10, math::so4_double_rotor_4d(gl, gr, u)))
            .collect();
        let det_odd = math::det4x4_cols(cols_odd[0], cols_odd[1], cols_odd[2], cols_odd[3]);
        assert!((det_odd + 1.0).abs() < 1e-5, "O(4) reflection det must be -1, got {det_odd}");
        let r_once = math::householder4_gated(v, 2.0, 1e-10, (0.3, -0.7, 0.2, 0.6));
        let r_twice = math::householder4_gated(v, 2.0, 1e-10, r_once);
        assert!((r_twice.0 - 0.3).abs() < 1e-5 && (r_twice.1 + 0.7).abs() < 1e-5);

        // 3. BIBO контракция для всех beta in [0, 2]
        let r_in = (0.3_f32, -0.7, 0.2, 0.6);
        let r_in_norm2 = r_in.0 * r_in.0 + r_in.1 * r_in.1 + r_in.2 * r_in.2 + r_in.3 * r_in.3;
        for b_i in 0..=20 {
            let beta = (b_i as f32) * 0.1;
            let hb = math::householder4_gated(v, beta, 1e-10, r_in);
            let h2 = hb.0 * hb.0 + hb.1 * hb.1 + hb.2 * hb.2 + hb.3 * hb.3;
            assert!(h2 <= r_in_norm2 + 1e-5, "BIBO violated at beta={beta}: {h2} > {r_in_norm2}");
        }

        // 4. Вложенность SO(3) subset SO(4): при g_minus = 0 (g_R = -g_L) координата w инвариантна
        let u_sub = (0.45_f32, -0.35, 0.62, -0.18);
        let g_neg = (-gl.0, -gl.1, -gl.2);
        let u_rot = math::so4_double_rotor_4d(gl, g_neg, u_sub);
        assert!((u_rot.0 - u_sub.0).abs() < 1e-5, "SO(3) subset SO(4) must fix u.w");

        // 5. FD проверка производных q_stereo и householder4_gated_deriv_v
        let eps = 1e-3_f32;
        for c in 0..4usize {
            let mut vp = v;
            let mut vm = v;
            match c {
                0 => { vp.0 += eps; vm.0 -= eps; }
                1 => { vp.1 += eps; vm.1 -= eps; }
                2 => { vp.2 += eps; vm.2 -= eps; }
                _ => { vp.3 += eps; vm.3 -= eps; }
            }
            let hp = math::householder4_gated(vp, 1.35, 1e-10, r_in);
            let hm = math::householder4_gated(vm, 1.35, 1e-10, r_in);
            let an = math::householder4_gated_deriv_v(v, 1.35, 1e-10, r_in, c);
            let fd = ((hp.0 - hm.0) / (2.0 * eps), (hp.1 - hm.1) / (2.0 * eps), (hp.2 - hm.2) / (2.0 * eps), (hp.3 - hm.3) / (2.0 * eps));
            assert!(
                (fd.0 - an.0).abs() < 2e-3 && (fd.1 - an.1).abs() < 2e-3 && (fd.2 - an.2).abs() < 2e-3 && (fd.3 - an.3).abs() < 2e-3,
                "Householder dv deriv mismatch c={c}: fd={fd:?} an={an:?}"
            );
        }
    }

    #[test]
    fn test_nonabelian_o4_rtrl_fd_gradients() {
        let mut o4 = NonAbelianO4Rtrl::new(2, 3, 5, 101);
        // Разнообразие beta внутри (0.1, 1.9), включая проверку строго на водоразделе beta = 1.00
        for (idx, wb) in o4.w_beta.iter_mut().enumerate() {
            *wb = match idx % 3 {
                0 => 1.00,
                1 => 1.38,
                _ => 0.62,
            };
        }
        let seq = [
            [0.75_f32, 0.15, -0.10],
            [-0.10, 0.85, 0.15],
            [0.12, -0.15, 0.90],
            [0.65, 0.25, -0.05],
        ];
        let d = o4.d_state;
        let mut rng_w = math::splitmix64(777);
        let w_obj: Vec<f32> = (0..d)
            .map(|_| math::rand_signed_f32(&mut rng_w) * 0.5)
            .collect();
        let run_j = |p: &NonAbelianO4Rtrl| -> f32 {
            let mut c = p.clone();
            c.reset();
            for x in &seq {
                c.step_forward(x);
            }
            fast_dot(&w_obj, &c.u)
        };

        let mut o4_run = o4.clone();
        o4_run.reset();
        for x in &seq {
            o4_run.step_forward(x);
        }

        // Проверка сохранения нормы даже на водоразделе beta=1.0
        assert!(
            o4_run.max_sphere_drift_4d() < 1e-5 && o4_run.max_orth_drift_4d() < 2e-5,
            "Unitary parity bridge must preserve norm/orthogonality even at beta=1: sphere={} orth={}",
            o4_run.max_sphere_drift_4d(),
            o4_run.max_orth_drift_4d()
        );

        let eps = 1e-3_f32;
        let d_in = o4_run.in_dim;
        let mut max_err_wv = 0.0_f32;
        let mut max_err_wb = 0.0_f32;

        for m in 0..o4_run.frames {
            let s0 = m * 5;
            for kc in 0..16 {
                for j in 0..d_in {
                    let mut an_v = 0.0_f32;
                    for a in 0..5 {
                        let s = s0 + a;
                        let off = s * 5;
                        let pv_off = ((s * 16 + kc) * d_in + j) * 5;
                        for q in 0..5 {
                            an_v += w_obj[off + q] * o4_run.p_wv[pv_off + q];
                        }
                    }
                    let idx = (m * 16 + kc) * d_in + j;
                    let (mut vp, mut vm) = (o4.clone(), o4.clone());
                    vp.wv[idx] += eps;
                    vm.wv[idx] -= eps;
                    let fd_v = (run_j(&vp) - run_j(&vm)) / (2.0 * eps);
                    max_err_wv = max_err_wv.max((fd_v - an_v).abs());
                }
            }
        }
        for m in 0..o4_run.frames {
            let s0 = m * 5;
            for j in 0..d_in {
                let mut an_b = 0.0_f32;
                for a in 0..5 {
                    let s = s0 + a;
                    let off = s * 5;
                    let pb_off = (s * d_in + j) * 5;
                    for q in 0..5 {
                        an_b += w_obj[off + q] * o4_run.p_wbeta[pb_off + q];
                    }
                }
                let wb_idx = m * d_in + j;
                let (mut bp, mut bm) = (o4.clone(), o4.clone());
                bp.w_beta[wb_idx] += eps;
                bm.w_beta[wb_idx] -= eps;
                let fd_b = (run_j(&bp) - run_j(&bm)) / (2.0 * eps);
                max_err_wb = max_err_wb.max((fd_b - an_b).abs());
            }
        }
        assert!(
            max_err_wv < 1e-3 && max_err_wb < 1e-3,
            "O(4) RTRL FD mismatch (u): wv={max_err_wv} wb={max_err_wb}"
        );

        // Вторая проверка: FD-паритет для сферического чтения c.b через exact inv_rho * (gb - (gb.b)*b)
        let run_jb = |p: &NonAbelianO4Rtrl| -> f32 {
            let mut c = p.clone();
            c.reset();
            for x in &seq {
                c.step_forward(x);
            }
            fast_dot(&w_obj, &c.b)
        };
        let mut grad_u_b = vec![0.0_f32; d];
        for s in 0..o4_run.sectors {
            let off = s * 5;
            let mut dot_gb = 0.0_f32;
            for q in 0..5 {
                dot_gb += w_obj[off + q] * o4_run.b[off + q];
            }
            let inv_r = o4_run.inv_rho[s];
            for q in 0..5 {
                grad_u_b[off + q] = inv_r * (w_obj[off + q] - dot_gb * o4_run.b[off + q]);
            }
        }
        let mut max_err_b_wb = 0.0_f32;
        let mut max_err_b_wv = 0.0_f32;
        for j in 0..d_in {
            for m in 0..o4_run.frames {
                let s0 = m * 5;
                let mut an_b = 0.0_f32;
                for a in 0..5 {
                    let s = s0 + a;
                    let off = s * 5;
                    let pb_off = (s * d_in + j) * 5;
                    for q in 0..5 {
                        an_b += grad_u_b[off + q] * o4_run.p_wbeta[pb_off + q];
                    }
                }
                let wb_idx = m * d_in + j;
                let (mut bp, mut bm) = (o4.clone(), o4.clone());
                bp.w_beta[wb_idx] += eps;
                bm.w_beta[wb_idx] -= eps;
                let fd_b = (run_jb(&bp) - run_jb(&bm)) / (2.0 * eps);
                max_err_b_wb = max_err_b_wb.max((fd_b - an_b).abs());
            }

            let mut an_v = 0.0_f32;
            for a in 0..5 {
                let s = a;
                let off = s * 5;
                let pv_off = ((s * 16 + 13) * d_in + j) * 5;
                for q in 0..5 {
                    an_v += grad_u_b[off + q] * o4_run.p_wv[pv_off + q];
                }
            }
            let idx_v = 13 * d_in + j;
            let (mut vp, mut vm) = (o4.clone(), o4.clone());
            vp.wv[idx_v] += eps;
            vm.wv[idx_v] -= eps;
            let fd_v = (run_jb(&vp) - run_jb(&vm)) / (2.0 * eps);
            max_err_b_wv = max_err_b_wv.max((fd_v - an_v).abs());
        }
        assert!(
            max_err_b_wb < 1e-3 && max_err_b_wv < 1e-3,
            "O(4) RTRL FD mismatch (b): wb={max_err_b_wb} wv={max_err_b_wv}"
        );
    }

    #[test]
    fn test_nonabelian_o4_readonly_seu_and_s5_watershed() {
        // 1. Read-only bit equivalence + State/Weight SEU + 0-NaN
        let mut o4 = NonAbelianO4Rtrl::new(2, 4, 16, 42);
        o4.enable_o4_proto_metric(12.0, 1.4, 0.08, 0.25, 0.04);
        for t in 0..12usize {
            let mut x = [0.0_f32; 4];
            x[t % 4] = 1.0;
            let _ = o4.step_and_learn_supervised(&x, Some((t * 3 + 1) % 16), 0.02, 0.02);
        }
        let u_snap = o4.u.clone();
        let w_refs: [&[f32]; 4] = [
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 1.0],
        ];
        let ro_pred = o4.predict_k_ahead_readonly(&w_refs);
        assert_eq!(o4.u, u_snap, "predict_k_ahead_readonly must not mutate state");
        let mut o4_clone = o4.clone();
        let mut cl_pred = 0usize;
        for &wx in &w_refs {
            cl_pred = o4_clone.step_and_learn_supervised(wx, None, 0.0, 0.0);
        }
        assert_eq!(ro_pred, cl_pred);

        // State-SEU & Weight-SEU
        for k in 0..5usize {
            o4.inject_register_bit_flip(k * 7, 30);
            o4.inject_weight_bit_flip(k * 3, 30);
        }
        o4.step_forward(&[1.0, 0.0, 0.0, 0.0]);
        assert!(o4.seu_fault_count >= 5);
        assert!(o4.u.iter().all(|v| v.is_finite()));
        assert!(o4.wv.iter().all(|v| v.is_finite()));
        assert!(o4.w_beta.iter().all(|v| v.is_finite()));
        assert!(o4.out_w.iter().all(|v| v.is_finite()));

        // 2. Диагностика Мины №2 + унитарный мост чётности: S_5 без оракулов (120 классов)
        let gens: [[u8; 5]; 4] = [
            [1, 0, 2, 3, 4], // (0 1) — нечётная (odd)
            [0, 2, 1, 3, 4], // (1 2) — нечётная (odd)
            [1, 2, 3, 4, 0], // (0 1 2 3 4) — чётная (even)
            [4, 0, 1, 2, 3], // (4 3 2 1 0) — чётная (even)
        ];
        let mut perms: Vec<[u8; 5]> = Vec::new();
        for a in 0..5u8 {
            for b in 0..5u8 {
                if b == a { continue; }
                for c in 0..5u8 {
                    if c == a || c == b { continue; }
                    for d in 0..5u8 {
                        if d == a || d == b || d == c { continue; }
                        let e = 10 - (a + b + c + d);
                        perms.push([a, b, c, d, e]);
                    }
                }
            }
        }
        perms.sort();
        let ident = perms.binary_search(&[0, 1, 2, 3, 4]).unwrap();
        let mut mul = vec![vec![0usize; 120]; 4];
        for g in 0..4 {
            for p_id in 0..120 {
                let cp = perms[p_id];
                let gp = gens[g];
                let prod = [
                    cp[gp[0] as usize],
                    cp[gp[1] as usize],
                    cp[gp[2] as usize],
                    cp[gp[3] as usize],
                    cp[gp[4] as usize],
                ];
                mul[g][p_id] = perms.binary_search(&prod).unwrap();
            }
        }
        let frames = 4usize;
        let mut net = NonAbelianO4Rtrl::new(frames, 4, 120, 7);
        net.enable_o4_proto_metric(12.0, 1.4, 0.08, 0.55, 0.0);
        let mut rng = math::splitmix64(7777);
        let n_eps = 10000usize;
        for ep in 0..n_eps {
            let len = 1 + (ep % 10);
            net.reset();
            let mut cur = ident;
            let dec = 1.0_f32 / (1.0 + 3.5 * (ep as f32) / (n_eps as f32));
            for _ in 0..len {
                rng = math::splitmix64(rng);
                let g = ((rng >> 16) as usize) % 4;
                cur = mul[g][cur];
                let mut x = [0.0_f32; 4];
                x[g] = 1.0;
                let _ = net.step_and_learn_supervised(&x, Some(cur), 0.025 * dec, 0.025 * dec);
            }
        }

        let x_onehot = [
            [1.0_f32, 0.0, 0.0, 0.0],
            [0.0_f32, 1.0, 0.0, 0.0],
            [0.0_f32, 0.0, 1.0, 0.0],
            [0.0_f32, 0.0, 0.0, 1.0],
        ];
        let mut eval_at = |len: usize, eval_seed: u64| -> (f64, f32, f32) {
            let mut hits = 0usize;
            let mut max_orth = 0.0_f32;
            let mut max_sph = 0.0_f32;
            let mut rng_ev = math::splitmix64(eval_seed);
            for _ in 0..80usize {
                net.reset();
                let mut cur = ident;
                let mut refs: Vec<&[f32]> = Vec::with_capacity(len);
                for _ in 0..len {
                    rng_ev = math::splitmix64(rng_ev);
                    let g = ((rng_ev >> 16) as usize) % 4;
                    cur = mul[g][cur];
                    refs.push(&x_onehot[g]);
                }
                let pred = net.predict_k_ahead_readonly(&refs);
                if pred == cur {
                    hits += 1;
                }
                // Также проверим живой ненормализованный шаг при одном запуске
                net.reset();
                for &r in &refs {
                    net.step_forward(r);
                }
                max_orth = max_orth.max(net.max_orth_drift_4d());
                max_sph = max_sph.max(net.max_sphere_drift_4d());
            }
            (hits as f64 * 100.0 / 80.0, max_orth, max_sph)
        };

        let (acc_30, _, _) = eval_at(30, 99991);
        let (acc_100, orth_100, sph_100) = eval_at(100, 88881);

        let mut ch_betas = [[0.0_f32; 4]; 4];
        for g in 0..4usize {
            net.channel_betas_into(g, &mut ch_betas[g]);
        }
        let rqs: Vec<f32> = (0..frames).map(|m| net.frame_parity_line_rq(m)).collect();
        let parity_spread = |m: usize| -> f32 {
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for g in 0..4usize {
                let v = ch_betas[g][m];
                if v < mn { mn = v; }
                if v > mx { mx = v; }
            }
            mx - mn
        };
        let mut faithful: Vec<usize> = (0..frames)
            .filter(|&m| rqs[m] > 0.97 && net.frame_acc_ema[m] > 0.85 && parity_spread(m) > 0.5)
            .collect();
        if faithful.is_empty() {
            faithful = (0..frames)
                .filter(|&m| rqs[m] > 0.97 && net.frame_acc_ema[m] > 0.85)
                .collect();
        }
        if faithful.is_empty() {
            faithful = (0..frames)
                .filter(|&m| rqs[m] > 0.97 && net.frame_acc_ema[m] > 0.70)
                .collect();
        }
        let best_m = (0..frames)
            .max_by(|&a, &b| rqs[a].partial_cmp(&rqs[b]).unwrap())
            .unwrap_or(0);
        let use_frames: Vec<usize> = if faithful.is_empty() { vec![best_m] } else { faithful };
        let odd_mean: f32 = use_frames
            .iter()
            .map(|&m| 0.5 * (ch_betas[0][m] + ch_betas[1][m]))
            .sum::<f32>()
            / (use_frames.len() as f32);
        let even_mean: f32 = use_frames
            .iter()
            .map(|&m| 0.5 * (ch_betas[2][m] + ch_betas[3][m]))
            .sum::<f32>()
            / (use_frames.len() as f32);

        assert!(
            acc_30 >= 85.0 && acc_100 >= 80.0,
            "S_5 extrapolation too low: L=30={acc_30:.1}% L=100={acc_100:.1}%, orth={orth_100:.2e}, sph={sph_100:.2e}, rqs={rqs:?}, ch_betas={ch_betas:?}"
        );
        assert!(
            odd_mean > 1.80 && even_mean < 0.20,
            "S_5 intrinsic parity separation failed: rqs={rqs:?}, odd_mean={odd_mean:.3}, even_mean={even_mean:.3}, ch_betas={ch_betas:?}"
        );
    }

    #[test]
    fn test_unified_b1_exact_fd_gradients_and_10_order_jump() {
        let seq: [[f32; 3]; 5] = [
            [0.8, -0.4, 0.3],
            [-0.6, 1.1, -0.2],
            [0.4, 0.5, -0.9],
            [-1.0, -0.3, 0.7],
            [0.7, -0.8, 0.5],
        ];
        let init_base = |exact_sqrt: bool| {
            let mut cfg = UnifiedCayleyConfig::checkpoint_b1(3, 3, 77);
            cfg.learn_norm = false;
            cfg.exact_inv_sqrt = exact_sqrt;
            let mut core = UnifiedCayleyCore::new(cfg);
            // Ненулевые веса обеих головок для нетривиальной проверки
            for (i, w) in core.slots[0].actor_w.iter_mut().enumerate() {
                *w += 0.12 * (((i as f32) * 0.71).sin());
            }
            for (i, w) in core.slots[0].critic_w.iter_mut().enumerate() {
                *w = 0.15 * (((i as f32) * 0.53 + 0.4).cos());
            }
            core
        };

        let eval_obj = |core_tpl: &UnifiedCayleyCore, mutate: &dyn Fn(&mut UnifiedCayleyCore)| -> f64 {
            let mut c = core_tpl.clone();
            mutate(&mut c);
            c.start_episode();
            for x in &seq {
                c.forward_and_rtrl_traces(x, false, false, false);
            }
            (c.ws.probs[0] as f64).ln() + (c.ws.critic_val as f64)
        };

        for &exact_sqrt in &[true, false] {
            let base = init_base(exact_sqrt);
            let mut c = base.clone();
            c.start_episode();
            for x in &seq {
                c.forward_and_rtrl_traces(x, false, true, false);
            }
            let target_y = 0usize;
            let mut lerrs = [0.0_f32; 3];
            UnifiedCayleyCore::fill_p4_logprob_grads(
                &c.ws.logits,
                &c.ws.probs,
                &c.ws.psis,
                target_y,
                &mut lerrs,
            );
            c.compute_instant_rtrl_grads(&lerrs, 1.0);

            let tol = if exact_sqrt { 5e-4 } else { 2e-3 };
            let eps = 1e-3_f32;
            let mut max_err = 0.0_f64;
            let mut max_abs_g = 0.0_f64;

            let mut check_param = |name: &str, ana: f32, set_fn: &dyn Fn(&mut UnifiedCayleyCore, f32)| {
                let jp = eval_obj(&base, &|m| set_fn(m, eps));
                let jm = eval_obj(&base, &|m| set_fn(m, -eps));
                let fd = (jp - jm) / (2.0 * eps as f64);
                let err = (ana as f64 - fd).abs();
                max_err = max_err.max(err);
                max_abs_g = max_abs_g.max(fd.abs());
                assert!(
                    err < tol,
                    "FD mismatch on {name} (exact={exact_sqrt}): ana={ana:.6}, fd={fd:.6}, err={err:.6} >= {tol}"
                );
            };

            for b in 0..4usize {
                check_param("gate_logits", c.ws.g_gates[b], &move |m, d| {
                    m.slots[0].gate_logits[b] += d;
                });
            }
            for k in 0..4usize {
                check_param("b1_alpha", c.ws.g_b1_alpha[k], &move |m, d| {
                    m.slots[0].b1_alpha[k] += d;
                });
                check_param("b1_omega", c.ws.g_b1_omega[k], &move |m, d| {
                    m.slots[0].b1_omega[k] += d;
                });
            }
            for j in 0..12usize {
                check_param("b1_in_w", c.ws.g_b1_inw[j], &move |m, d| {
                    m.slots[0].b1_in_w[j] += d;
                });
                check_param("b2_in_w", c.ws.g_b2_inw[j], &move |m, d| {
                    m.slots[0].b2_in_w[j] += d;
                });
            }
            for s in 0..4usize {
                check_param("b2_alpha", c.ws.g_b2_alpha[s], &move |m, d| {
                    m.slots[0].b2_alpha[s] += d;
                });
                check_param("b2_omega", c.ws.g_b2_omega[s], &move |m, d| {
                    m.slots[0].b2_omega[s] += d;
                });
            }
            let fdim = c.ws.feat_dim;
            for i in 0..8usize {
                check_param("actor_w", lerrs[0] * c.ws.feat_gated[i], &move |m, d| {
                    m.slots[0].actor_w[i] += d;
                });
                check_param("critic_w", c.ws.feat_gated[fdim - 1 - i], &move |m, d| {
                    m.slots[0].critic_w[fdim - 1 - i] += d;
                });
            }
            assert!(max_abs_g > 1e-3, "trivial zero gradient in B1 check: {max_abs_g}");
        }

        // Стресс-тест 10-порядкового скачка мощности входа (1e-5 -> 1e+5) без sqrt -> y > 0 и строго конечный вывод
        let mut jump_core = UnifiedCayleyCore::new(UnifiedCayleyConfig::checkpoint_b1(2, 3, 99));
        let jumps: [[f32; 3]; 6] = [
            [1e-5, -1e-5, 2e-5],
            [1e-5, 1e-5, -1e-5],
            [1e5, -1e5, 5e4],
            [-1e5, 8e4, -1e5],
            [1e-5, 0.0, 1e-5],
            [3e4, -2e4, 1e4],
        ];
        for (idx, x) in jumps.iter().enumerate() {
            let (pred, conf) = jump_core.step_supervised_class(x, idx & 1, true, true, false);
            assert!(pred < 2 && conf.is_finite(), "non-finite conf at jump {idx}: {conf}");
            for &ni in &jump_core.ws.norm_inv {
                assert!(ni.is_finite() && ni > 0.0, "norm_inv violated positivity bound: {ni}");
            }
            assert!(jump_core.ws.critic_val.is_finite(), "non-finite critic on 10-order jump");
        }
    }

    #[test]
    fn test_unified_b2_selectivity_and_local_feedback_fd_gradients() {
        let seq: [[f32; 3]; 5] = [
            [0.9, -0.5, 0.4],
            [-0.7, 1.0, -0.3],
            [0.5, 0.6, -0.8],
            [-0.9, -0.4, 0.8],
            [0.6, -0.7, 0.5],
        ];
        let mut cfg = UnifiedCayleyConfig::checkpoint_b2(3, 3, 123);
        cfg.learn_norm = false;
        cfg.exact_inv_sqrt = true;
        let mut base = UnifiedCayleyCore::new(cfg);

        // Ненулевые beta, bdelta и qfb для нетривиальной проверки селективности и rank-1 обратной связи dg_du
        for i in 0..base.slots[0].b1_beta.len() {
            base.slots[0].b1_beta[i] = 0.35 * if (i & 1) == 0 { 1.0 } else { -1.0 };
            base.slots[0].b1_bdelta[i] = 0.15;
        }
        for (i, q) in base.slots[0].b1_qfb.iter_mut().enumerate() {
            *q = 0.25 * if (i & 1) == 0 { 1.0 } else { -0.8 };
        }
        for i in 0..base.slots[0].b2_beta.len() {
            base.slots[0].b2_beta[i] = 0.35 * if (i & 1) == 0 { 1.0 } else { -1.0 };
            base.slots[0].b2_bdelta[i] = 0.15;
        }
        for (i, q) in base.slots[0].b2_qfb.iter_mut().enumerate() {
            *q = 0.25 * if (i & 1) == 0 { 1.0 } else { -0.7 };
        }
        for (i, w) in base.slots[0].actor_w.iter_mut().enumerate() {
            *w += 0.12 * (((i as f32) * 0.71).sin());
        }
        for (i, w) in base.slots[0].critic_w.iter_mut().enumerate() {
            *w = 0.15 * (((i as f32) * 0.53 + 0.4).cos());
        }

        let eval_obj = |core_tpl: &UnifiedCayleyCore, mutate: &dyn Fn(&mut UnifiedCayleyCore)| -> f64 {
            let mut c = core_tpl.clone();
            mutate(&mut c);
            c.start_episode();
            for x in &seq {
                c.forward_and_rtrl_traces(x, false, false, false);
            }
            (c.ws.probs[0] as f64).ln() + (c.ws.critic_val as f64)
        };

        let mut c = base.clone();
        c.start_episode();
        for x in &seq {
            c.forward_and_rtrl_traces(x, false, true, false);
        }
        let mut lerrs = [0.0_f32; 3];
        UnifiedCayleyCore::fill_p4_logprob_grads(&c.ws.logits, &c.ws.probs, &c.ws.psis, 0, &mut lerrs);
        c.compute_instant_rtrl_grads(&lerrs, 1.0);

        let tol = 5e-4_f64;
        let eps = 1e-3_f32;
        let mut max_abs_g = 0.0_f64;
        let mut check = |name: &str, ana: f32, set_fn: &dyn Fn(&mut UnifiedCayleyCore, f32)| {
            let jp = eval_obj(&base, &|m| set_fn(m, eps));
            let jm = eval_obj(&base, &|m| set_fn(m, -eps));
            let fd = (jp - jm) / (2.0 * eps as f64);
            let err = (ana as f64 - fd).abs();
            max_abs_g = max_abs_g.max(fd.abs());
            assert!(
                err < tol,
                "B2 FD mismatch on {name}: ana={ana:.6}, fd={fd:.6}, err={err:.6} >= {tol}"
            );
        };

        for k in 0..4usize {
            check("b1_beta", c.ws.g_b1_beta[k], &move |m, d| { m.slots[0].b1_beta[k] += d; });
            check("b1_bdelta", c.ws.g_b1_bdelta[k], &move |m, d| { m.slots[0].b1_bdelta[k] += d; });
        }
        for i in 0..8usize {
            check("b1_qfb", c.ws.g_b1_qfb[i], &move |m, d| { m.slots[0].b1_qfb[i] += d; });
            check("b1_sel_w", c.ws.g_b1_selw[i], &move |m, d| { m.slots[0].b1_sel_w[i] += d; });
            check("b2_sel_w", c.ws.g_b2_selw[i], &move |m, d| { m.slots[0].b2_sel_w[i] += d; });
        }
        for s in 0..4usize {
            check("b2_beta", c.ws.g_b2_beta[s], &move |m, d| { m.slots[0].b2_beta[s] += d; });
            check("b2_bdelta", c.ws.g_b2_bdelta[s], &move |m, d| { m.slots[0].b2_bdelta[s] += d; });
        }
        for i in 0..9usize {
            check("b2_qfb", c.ws.g_b2_qfb[i], &move |m, d| { m.slots[0].b2_qfb[i] += d; });
        }
        assert!(max_abs_g > 1e-3, "trivial zero gradient in B2 check: {max_abs_g}");
    }

    #[test]
    fn test_unified_b3_readonly_equivalence_bartlett_ph_and_bank_recall() {
        // 1. &self predict_class_readonly битово/точно совпадает с последовательным шагом без обучения
        let mut core = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config(3, 4, 303));
        let train_seq: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for step in 0..40usize {
            if step % 4 == 0 { core.start_episode(); }
            let _ = core.step_supervised_class(&train_seq[step % 4], step % 3, true, true, false);
        }
        let query: [&[f32]; 4] = [&train_seq[0], &train_seq[1], &train_seq[2], &train_seq[3]];
        let ro_pred = core.predict_class_readonly(&query);
        let mut clone_m = core.clone();
        clone_m.start_episode();
        let mut mut_pred = 0usize;
        for q in &query {
            mut_pred = clone_m.step_supervised_class(q, 0, false, false, false).0;
        }
        assert_eq!(ro_pred, mut_pred, "readonly &self rollout diverged from forward pass");

        // 2. Bartlett PH устойчив к сильно автокоррелированному стационарному шуму (rho ~ 0.6), но ловит смену режима
        let mut ph = PageHinkleyDetector::new(0.14, 2.2, 20).with_bartlett_scaling(true);
        let mut e = 0.0_f32;
        let mut rng = crate::math::splitmix64(999);
        let mut false_triggers = 0usize;
        for _ in 0..160usize {
            rng = crate::math::splitmix64(rng);
            let w = crate::math::rand_signed_f32(&mut rng) * 0.14;
            e = 0.65 * e + w;
            if ph.observe((0.86 + e).clamp(0.0, 1.0)) {
                false_triggers += 1;
            }
        }
        assert_eq!(false_triggers, 0, "Bartlett PH false alarm under rho=0.65 noise: {false_triggers}");
        let mut caught_at = None;
        for step in 0..40usize {
            if ph.observe(0.15) {
                caught_at = Some(step);
                break;
            }
        }
        assert!(caught_at.is_some(), "Bartlett PH missed real regime drop");

        // 3. Probe-Before-Allocate банк recall A -> B -> A возвращает в Slot 0
        let mut bank_core = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config(2, 3, 505));
        for _ in 0..36usize {
            let _ = bank_core.end_episode_score(0.90, true);
        }
        assert_eq!(bank_core.active_slot, 0);
        // Дрейф к B -> переход в свежий слот 1 (без затирания слота 0!)
        for _ in 0..40usize {
            let _ = bank_core.end_episode_score(0.10, true);
            if bank_core.active_slot != 0 { break; }
        }
        assert_eq!(bank_core.active_slot, 1, "expected switch from Slot 0 to fresh Slot 1");
        for _ in 0..32usize {
            let _ = bank_core.end_episode_score(0.88, true);
        }
        // Возврат A: Slot 1 падает -> запускает probe(Slot 0), затем принимается на хороших очках A
        for _ in 0..25usize {
            let _ = bank_core.end_episode_score(0.10, true);
            if bank_core.probe_eps_left > 0 { break; }
        }
        assert!(bank_core.probe_eps_left > 0 && bank_core.active_slot == 0, "expected frozen probe on Slot 0");
        for _ in 0..6usize {
            let _ = bank_core.end_episode_score(0.90, true);
        }
        assert_eq!(bank_core.active_slot, 0, "expected Slot 0 recall after successful 6-ep probe");
        assert_eq!(bank_core.probe_eps_left, 0);
    }

    #[test]
    fn test_fixed_point_q8_24_vs_q16_16_nasa_ruling_and_determinism() {
        // 1. q8_24_inv_sqrt по 10+ порядкам (1/16000 .. 1e6, выход <= 126.5 < 128.0 потолка Q8.24)
        let scales = [1.0f32 / 16000.0, 1.0e-3, 0.25, 1.0, 125.0, 4096.0, 1.0e6];
        for &a_f in &scales {
            let a_q = Q39_24::from_f32_boundary(a_f);
            let y_q = q8_24_inv_sqrt(a_q);
            assert!(y_q.0 > 0, "q8_24_inv_sqrt({}) must stay > 0, got {}", a_f, y_q.0);
            let y_got = y_q.to_f32_boundary() as f64;
            let y_exp = 1.0f64 / (a_f as f64).sqrt();
            let rel_err = (y_got - y_exp).abs() / y_exp;
            assert!(
                rel_err < 5.0e-4,
                "inv_sqrt rel_err too high at a={}: got={}, exp={}, rel_err={}",
                a_f,
                y_got,
                y_exp,
                rel_err
            );
        }

        // 2. Q8.24 vs Q16.16: шаг lr=1e-4 * grad=0.05 живёт в Q8.24 (>0 LSB), но зануляется в Q16.16
        let lr_q24 = Q8_24::from_f32_boundary(1.0e-4);
        let grad_q24 = Q8_24::from_f32_boundary(0.05);
        let step_q24 = q24_mul(lr_q24, grad_q24);
        assert!(step_q24.0 > 0, "Q8.24 update must stay > 0 LSB, got {}", step_q24.0);

        let lr_q16 = Q16_16::from_f32_boundary(1.0e-4);
        let grad_q16 = Q16_16::from_f32_boundary(0.05);
        let step_q16 = q16_mul(lr_q16, grad_q16);
        assert_eq!(step_q16, Q16_16(0), "expected Q16.16 underflow for 1e-4 * 0.05, got {:?}", step_q16);

        // 3. PageHinkleyQ39_24 принимает NASA RUL = 125.0 без насыщения 128.0 и ловит дрейф
        let mut ph = PageHinkleyQ39_24::new(
            Q39_24::from_f32_boundary(2.0),
            Q39_24::from_f32_boundary(60.0),
            8,
        );
        for _ in 0..16 {
            assert!(!ph.observe(Q39_24::from_f32_boundary(125.0)));
        }
        assert!(
            (ph.mu_bank.to_f32_boundary() - 125.0).abs() < 0.05,
            "Q39.24 mu_bank saturated prematurely: {}",
            ph.mu_bank.to_f32_boundary()
        );
        let mut caught = false;
        for _ in 0..10 {
            if ph.observe(Q39_24::from_f32_boundary(40.0)) {
                caught = true;
                break;
            }
        }
        assert!(caught, "PageHinkleyQ39_24 failed to detect 125.0 -> 40.0 drift");

        // 4. FixedO4Rotor удерживает единичную норму 10 000 шагов без дрейфа FPU
        let mut rotor = FixedO4Rotor::new(42, Q8_24::from_f32_boundary(1.0));
        for _ in 0..10_000 {
            rotor.step_and_renorm();
        }
        let norm_sq = rotor.sphere_norm_sq_q39().to_f32_boundary();
        assert!(
            (norm_sq - 1.0).abs() < 1.0e-5,
            "FixedO4Rotor norm^2 drifted after 10k steps: {}",
            norm_sq
        );

        // 5. FixedCayleyCore детерминированный дайджест на потоке с амплитудой NASA 125.0
        let run_stream = || -> u64 {
            let mut c = FixedCayleyCore::new(4, 2, 777);
            for t in 0..120usize {
                let x = [
                    Q39_24::from_f32_boundary(125.0 + (t as f32) * 0.1),
                    Q39_24::from_f32_boundary(((t % 5) as f32 - 2.0) * 15.0),
                    Q39_24::from_f32_boundary(0.5),
                    Q39_24::from_f32_boundary(-3.0),
                ];
                let cls = t & 1;
                let _ = c.step_supervised_class_q(&x, cls, true);
                let _ = c.step_supervised_reg_q(&x, Q39_24::from_f32_boundary(125.0 - (t as f32)), true);
            }
            c.state_digest_fnv1a()
        };
        let d1 = run_stream();
        let d2 = run_stream();
        assert_eq!(d1, d2);
        assert_ne!(d1, 0);
    }

    #[test]
    fn test_adaptive_od_isometric_zero_forgetting_and_shuffle_immunity() {
        // 1. RAM для S_6 (720 классов, M=2, D_MAX=6) строго < 260 KB (<< 450 KB бюджета)
        let big = AdaptiveOrthogonalRtrl::new(2, 4, 720, 42);
        assert!(
            big.memory_footprint_bytes() < 260 * 1024,
            "AdaptiveOrthogonalRtrl S6 RAM {} >= 260 KB",
            big.memory_footprint_bytes()
        );

        // 2. Изометрический апгрейд 2->3->4->5 сохраняет вероятности < 1e-6, логиты < 1e-5 и активирует новые плоскости
        let mut net = AdaptiveOrthogonalRtrl::new(2, 4, 16, 17);
        let onehot: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for ep in 0..80usize {
            net.reset();
            for step in 0..5usize {
                let g = (ep + step) % 4;
                let y = (g * 3 + step) % 16;
                let _ = net.step_supervised(&onehot[g], true, y, 0.025, 0.025);
            }
        }

        for target_d in 3..=5usize {
            net.reset();
            for step in 0..6usize {
                net.step_forward(&onehot[(step * 3) % 4]);
            }
            let mut logits_before = vec![0.0_f32; 16];
            let mut probs_before = vec![0.0_f32; 16];
            let mut psi = vec![0.0_f32; 16];
            net.compute_combined_logits_into(&mut logits_before);
            crate::math::rational_probs_p4_into(&logits_before, &mut probs_before, &mut psi);

            assert!(net.upgrade_dimension_isometric());
            assert_eq!(net.active_dim, target_d);

            net.reset();
            for step in 0..6usize {
                net.step_forward(&onehot[(step * 3) % 4]);
            }
            let mut logits_after = vec![0.0_f32; 16];
            let mut probs_after = vec![0.0_f32; 16];
            net.compute_combined_logits_into(&mut logits_after);
            crate::math::rational_probs_p4_into(&logits_after, &mut probs_after, &mut psi);

            let max_dl = logits_before
                .iter()
                .zip(logits_after.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let max_dp = probs_before
                .iter()
                .zip(probs_after.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_dl < 1.0e-5, "dim {} logit jump too large: {}", target_d, max_dl);
            assert!(max_dp < 1.0e-6, "dim {} prob jump too large: {}", target_d, max_dp);

            let n_old = planes_for_dim(target_d - 1);
            let n_new = planes_for_dim(target_d);
            for ep in 0..120usize {
                net.reset();
                for step in 0..5usize {
                    let g = (ep * 5 + step) % 4;
                    let y = (g * 3 + step * 2) % 16;
                    let _ = net.step_supervised(&onehot[g], true, y, 0.025, 0.025);
                }
            }
            let mut max_new_t = 0.0_f32;
            for ch in &net.planes {
                for k in n_old..n_new {
                    max_new_t = max_new_t.max(ch[k].abs());
                }
            }
            assert!(
                max_new_t > 0.02,
                "newly opened planes at dim {} stayed trapped in saddle: max|t|={}",
                target_d,
                max_new_t
            );
        }

        // 3. Контроль со случайными метками (Label-Shuffle): остаётся строго в d = 2 за 2 000 эпизодов
        let mut shuf = AdaptiveOrthogonalRtrl::new(2, 3, 24, 999);
        let mut rng = crate::math::splitmix64(999);
        for ep in 0..2_000usize {
            shuf.reset();
            let g = ep % 3;
            let _ = shuf.step_supervised(&onehot[g], true, (rng as usize) % 24, 0.02, 0.02);
            rng = crate::math::splitmix64(rng);
            let pred = shuf.predict_current_readonly();
            let y_rand = ((rng >> 16) as usize) % 24;
            let _ = shuf.observe_episode_hit(pred == y_rand, y_rand);
        }
        assert_eq!(shuf.active_dim, 2, "Label-Shuffle falsely escalated to dim {}", shuf.active_dim);
    }

    #[test]
    fn test_adaptive_od_fd_gradients() {
        use crate::math::rational_probs_p4_into;
        for &target_d in &[2usize, 3, 4] {
            let mut base = AdaptiveOrthogonalRtrl::new(1, 2, 5, 77 + target_d as u64);
            while base.active_dim < target_d {
                assert!(base.upgrade_dimension_isometric());
            }
            for ch in 0..2 {
                for k in 0..planes_for_dim(target_d) {
                    base.planes[ch][k] = 0.23 * ((k + 1) as f32) - 0.17 * (ch as f32);
                }
                for a in 0..target_d {
                    base.h_vec[ch][a] = 0.4 + 0.15 * ((a + ch) as f32);
                }
                base.betas[ch] = 0.35 + 0.12 * (ch as f32);
            }
            // Населяем все активные столбцы прототипов ненулевыми значениями
            let mut rng = crate::math::splitmix64(91 + target_d as u64);
            for c in 0..5 {
                let p_off = c * ADAPTIVE_MAX_COLS;
                for col in 0..ADAPTIVE_MAX_COLS {
                    if !is_col_active(col, target_d) {
                        continue;
                    }
                    for i in 0..target_d {
                        rng = crate::math::splitmix64(rng);
                        base.proto[p_off + col][i] = ((rng >> 40) & 0xFFFF) as f32 * 3.0517578e-5_f32 - 1.0;
                    }
                    rng = crate::math::splitmix64(rng);
                    base.proto[p_off + col][ADAPTIVE_PARITY_AXIS] =
                        ((rng >> 40) & 0xFFFF) as f32 * 3.0517578e-5_f32 - 1.0;
                }
            }
            base.lambda_par = 0.0;

            let inputs: [[f32; 2]; 3] = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
            let targets = [1usize, 3, 2];
            let eff_scale = base.proto_scale * ADAPTIVE_COL_LOGIT_SCALE;

            let eval_loss = |m: &AdaptiveOrthogonalRtrl| -> f32 {
                let mut net = m.clone();
                net.reset();
                let mut total = 0.0_f32;
                let mut probs = vec![0.0_f32; 5];
                let mut psi = vec![0.0_f32; 5];
                for t in 0..3 {
                    net.step_forward(&inputs[t]);
                    let mut logits = vec![0.0_f32; 5];
                    for c in 0..5 {
                        let p_off = c * ADAPTIVE_MAX_COLS;
                        let mut sum_dot = 0.0_f32;
                        for col in 0..ADAPTIVE_MAX_COLS {
                            if !is_col_active(col, target_d) {
                                continue;
                            }
                            for i in 0..target_d {
                                sum_dot += net.state[0][col][i] * net.proto[p_off + col][i];
                            }
                            sum_dot += net.state[0][col][ADAPTIVE_PARITY_AXIS]
                                * net.proto[p_off + col][ADAPTIVE_PARITY_AXIS];
                        }
                        logits[c] = eff_scale * sum_dot;
                    }
                    rational_probs_p4_into(&logits, &mut probs, &mut psi);
                    total += probs[targets[t]].max(1e-12).ln();
                }
                total
            };

            let mut probe = base.clone();
            probe.reset();
            let mut ana_pl = vec![vec![0.0_f32; planes_for_dim(target_d)]; 2];
            let mut ana_p = vec![0.0_f32; 2];
            let scale_mod = {
                let wm = 0.25 + base.frame_acc_ema[0] * base.frame_acc_ema[0];
                (base.solo_w + (1.0 - base.solo_w) * wm).max(0.25)
            };
            for t in 0..3 {
                let mut tmp = probe.clone();
                let _ = tmp.step_supervised(&inputs[t], true, targets[t], 0.0, 0.0);
                let mut grad_cols = [[0.0_f32; ADAPTIVE_STATE_DIM]; ADAPTIVE_MAX_COLS];
                for c in 0..5 {
                    let ec = tmp.lerrs[c];
                    let p_off = c * ADAPTIVE_MAX_COLS;
                    for col in 0..ADAPTIVE_MAX_COLS {
                        if !is_col_active(col, target_d) {
                            continue;
                        }
                        for i in 0..target_d {
                            grad_cols[col][i] += scale_mod * eff_scale * ec * probe.proto[p_off + col][i];
                        }
                        grad_cols[col][ADAPTIVE_PARITY_AXIS] +=
                            scale_mod * eff_scale * ec * probe.proto[p_off + col][ADAPTIVE_PARITY_AXIS];
                    }
                }
                for gx in 0..2 {
                    for col in 0..ADAPTIVE_MAX_COLS {
                        if !is_col_active(col, target_d) {
                            continue;
                        }
                        let cb = (col * 2 + gx) * ADAPTIVE_PARAMS_PER_CH;
                        for k in 0..planes_for_dim(target_d) {
                            for i in 0..target_d {
                                ana_pl[gx][k] += grad_cols[col][i] * tmp.sens[cb + k][i];
                            }
                            ana_pl[gx][k] += grad_cols[col][ADAPTIVE_PARITY_AXIS]
                                * tmp.sens[cb + k][ADAPTIVE_PARITY_AXIS];
                        }
                        let bi = ADAPTIVE_MAX_PLANES + ADAPTIVE_D_MAX;
                        for i in 0..target_d {
                            ana_p[gx] += grad_cols[col][i] * tmp.sens[cb + bi][i];
                        }
                        ana_p[gx] += grad_cols[col][ADAPTIVE_PARITY_AXIS]
                            * tmp.sens[cb + bi][ADAPTIVE_PARITY_AXIS];
                    }
                }
                probe.state = tmp.state;
                probe.sens = tmp.sens;
            }

            let eps = 1.0e-3_f32;
            for gx in 0..2 {
                let mut mp = base.clone();
                mp.betas[gx] += eps;
                let lp = eval_loss(&mp);
                let mut mm = base.clone();
                mm.betas[gx] -= eps;
                let lm = eval_loss(&mm);
                let fd = (lp - lm) / (2.0 * eps) * scale_mod;
                let diff = (fd - ana_p[gx]).abs();
                assert!(
                    diff < 2.0e-2 || diff / fd.abs().max(1e-4) < 0.05,
                    "d={} ch={} p_rot FD mismatch: fd={} ana={}",
                    target_d,
                    gx,
                    fd,
                    ana_p[gx]
                );
                for k in 0..planes_for_dim(target_d) {
                    let mut mp2 = base.clone();
                    mp2.planes[gx][k] += eps;
                    let lp2 = eval_loss(&mp2);
                    let mut mm2 = base.clone();
                    mm2.planes[gx][k] -= eps;
                    let lm2 = eval_loss(&mm2);
                    let fd2 = (lp2 - lm2) / (2.0 * eps) * scale_mod;
                    let diff2 = (fd2 - ana_pl[gx][k]).abs();
                    assert!(
                        diff2 < 2.0e-2 || diff2 / fd2.abs().max(1e-4) < 0.05,
                        "d={} ch={} plane={} FD mismatch: fd={} ana={}",
                        target_d,
                        gx,
                        k,
                        fd2,
                        ana_pl[gx][k]
                    );
                }
            }
        }
    }

    #[test]
    fn test_q14_hypersphere_proto_export_and_shared_backbone_ram() {
        // 1. NonAbelianO4Rtrl: Q1.14 (i16) прототипы ровно в 2x компактнее и сохраняют argmax (step132)
        let mut o4 = NonAbelianO4Rtrl::new(3, 2, 24, 42);
        o4.enable_o4_proto_metric(12.0, 1.4, 0.12, 0.55, 0.05);
        assert!(
            o4.memory_footprint_bytes() < 32 * 1024,
            "NonAbelianO4Rtrl proto_metric must free v_out_w/v_out_b"
        );
        let x0 = [1.0_f32, 0.0];
        let x1 = [0.0_f32, 1.0];
        for ep in 0..80usize {
            o4.reset();
            let seq: [&[f32]; 3] = if ep % 2 == 0 {
                [&x0, &x1, &x0]
            } else {
                [&x1, &x0, &x1]
            };
            let target = ep % 2;
            for (i, &x) in seq.iter().enumerate() {
                let y = if i == 2 { Some(target) } else { None };
                let _ = o4.step_and_learn_supervised(x, y, 0.04, 0.02);
            }
        }
        let q14_o4 = o4.export_proto_q14();
        assert_eq!(q14_o4.len(), o4.out_w.len());
        assert_eq!(q14_o4.len() * 2, o4.out_w.len() * 4 / 2);
        o4.reset();
        let w_a: [&[f32]; 3] = [&x0, &x1, &x0];
        let w_b: [&[f32]; 3] = [&x1, &x0, &x1];
        assert_eq!(o4.predict_k_ahead_readonly(&w_a), o4.predict_k_ahead_q14_readonly(&w_a, &q14_o4));
        assert_eq!(o4.predict_k_ahead_readonly(&w_b), o4.predict_k_ahead_q14_readonly(&w_b, &q14_o4));

        // 2. AdaptiveOrthogonalRtrl: Q1.14 (i16) экспорт и совпадение argmax (step132)
        let mut ad = AdaptiveOrthogonalRtrl::new(3, 2, 12, 77);
        for ep in 0..60usize {
            ad.reset();
            let seq: [&[f32]; 2] = if ep % 2 == 0 { [&x0, &x1] } else { [&x1, &x0] };
            let target = ep % 2;
            for (i, &x) in seq.iter().enumerate() {
                let _ = ad.step_supervised(x, i == 1, target, 0.03, 0.05);
            }
        }
        let q14_ad = ad.export_proto_q14();
        assert_eq!(q14_ad.len(), ad.proto.len());
        let wa2: [&[f32]; 2] = [&x0, &x1];
        let wb2: [&[f32]; 2] = [&x1, &x0];
        assert_eq!(ad.predict_k_ahead_readonly(&wa2), ad.predict_k_ahead_q14_readonly(&wa2, &q14_ad));
        assert_eq!(ad.predict_k_ahead_readonly(&wb2), ad.predict_k_ahead_q14_readonly(&wb2, &q14_ad));

        // 3. UnifiedCayleyCore::one_config: shared_backbone + use_band0=false + 4-band IDBD < 45 KB на ELEC2
        let u = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config(2, 8, 1));
        assert!(
            u.memory_footprint_bytes() < 45 * 1024,
            "Optimized UnifiedCayleyCore::one_config ELEC2 must be < 45 KB, got {} B",
            u.memory_footprint_bytes()
        );
    }

    #[test]
    fn test_prime_fields_p997_and_p9973_primitive_roots_and_zero_oracle_contract() {
        let mod_pow = |mut b: u64, mut e: u64, m: u64| -> u64 {
            let mut r = 1u64;
            b %= m;
            while e > 0 {
                if (e & 1) == 1 {
                    r = (r * b) % m;
                }
                b = (b * b) % m;
                e >>= 1;
            }
            r
        };
        let find_prim = |p: u64| -> u64 {
            let phi = p - 1;
            let mut n = phi;
            let mut facs = Vec::new();
            let mut d = 2u64;
            while d * d <= n {
                if n % d == 0 {
                    facs.push(d);
                    while n % d == 0 {
                        n /= d;
                    }
                }
                d += if d == 2 { 1 } else { 2 };
            }
            if n > 1 {
                facs.push(n);
            }
            for g in 2..p {
                if facs.iter().all(|&q| mod_pow(g, phi / q, p) != 1) {
                    return g;
                }
            }
            panic!("no primitive root for {p}");
        };
        assert_eq!(find_prim(997), 7);
        let g9973 = find_prim(9973);
        for &(p, g) in &[(997u64, 7u64), (9973u64, g9973)] {
            // Проверка биекции pow <-> dlog на всей мультипликативной группе F_p^*
            let mut seen = vec![false; p as usize];
            let mut cur = 1u64;
            for k in 0..(p - 1) {
                assert!(!seen[cur as usize], "Non-primitive root g={g} mod {p} at k={k}");
                seen[cur as usize] = true;
                assert_eq!(mod_pow(g, k, p), cur);
                cur = (cur * g) % p;
            }
            assert_eq!(cur, 1);
            assert!(seen[1..].iter().all(|&s| s));
            // Проверка циклической подгруппы порядка 12
            let gs = mod_pow(g, (p - 1) / 12, p);
            assert_eq!(mod_pow(gs, 12, p), 1);
            assert_ne!(mod_pow(gs, 6, p), 1);
            assert_ne!(mod_pow(gs, 4, p), 1);
        }
    }

    #[test]
    fn test_unified_p1_j_xor_algebra_fd_gradients_and_readonly() {
        use crate::algebra::{cross_3d_pair, symplectic_j_xor_2d};
        // 1. Точная антисимметрия и зануление диагонали J-XOR и 3D cross
        let u2 = (1.35_f32, -0.82_f32);
        let v2 = (-0.64_f32, 1.19_f32);
        let om_uv = symplectic_j_xor_2d(u2, v2, true);
        let om_vu = symplectic_j_xor_2d(v2, u2, true);
        let om_uu = symplectic_j_xor_2d(u2, u2, true);
        assert!((om_uv.0 + om_vu.0).abs() < 1e-6 && (om_uv.1 + om_vu.1).abs() < 1e-6);
        assert!(om_uu.0.abs() < 1e-7 && om_uu.1.abs() < 1e-7);

        let u3 = (0.7_f32, -1.2_f32, 0.4_f32);
        let v3 = (-0.5_f32, 0.3_f32, 1.1_f32);
        let c_uv = cross_3d_pair(u3, v3);
        let c_vu = cross_3d_pair(v3, u3);
        let c_uu = cross_3d_pair(u3, u3);
        assert!((c_uv.0 + c_vu.0).abs() < 1e-6 && (c_uv.1 + c_vu.1).abs() < 1e-6 && (c_uv.2 + c_vu.2).abs() < 1e-6);
        assert!(c_uu.0.abs() < 1e-7 && c_uu.1.abs() < 1e-7 && c_uu.2.abs() < 1e-7);
        assert!((u3.0 * c_uv.0 + u3.1 * c_uv.1 + u3.2 * c_uv.2).abs() < 1e-6);

        // 2. Точная проверка центральных конечных разностей через J-XOR + Cross3D + 2ndMoment + RMSNorm + RTRL
        let seq: [[f32; 3]; 5] = [
            [0.9, -0.5, 0.4],
            [-0.7, 1.0, -0.3],
            [0.5, 0.6, -0.8],
            [-0.9, -0.4, 0.8],
            [0.6, -0.7, 0.5],
        ];
        let mut cfg = UnifiedCayleyConfig::one_config(3, 3, 2026);
        cfg.enable_j_xor = true;
        cfg.learn_norm = false;
        cfg.exact_inv_sqrt = true;
        let mut base = UnifiedCayleyCore::new(cfg);
        assert_eq!(base.ws.nl_dim, 16);
        for (i, w) in base.slots[0].actor_w.iter_mut().enumerate() {
            *w += 0.14 * (((i as f32) * 0.63).sin());
        }
        for (i, w) in base.slots[0].critic_w.iter_mut().enumerate() {
            *w = 0.15 * (((i as f32) * 0.47 + 0.3).cos());
        }
        for i in 0..base.slots[0].b1_beta.len() {
            base.slots[0].b1_beta[i] = 0.3 * if (i & 1) == 0 { 1.0 } else { -1.0 };
        }
        for i in 0..base.slots[0].b2_beta.len() {
            base.slots[0].b2_beta[i] = 0.3 * if (i & 1) == 0 { 1.0 } else { -1.0 };
        }

        let eval_obj = |core_tpl: &UnifiedCayleyCore, mutate: &dyn Fn(&mut UnifiedCayleyCore)| -> f64 {
            let mut c = core_tpl.clone();
            mutate(&mut c);
            c.start_episode();
            for x in &seq {
                c.forward_and_rtrl_traces(x, false, false, false);
            }
            (c.ws.probs[0] as f64).ln() + (c.ws.critic_val as f64)
        };

        let mut c = base.clone();
        c.start_episode();
        for x in &seq {
            c.forward_and_rtrl_traces(x, false, true, false);
        }
        let mut lerrs = [0.0_f32; 3];
        UnifiedCayleyCore::fill_p4_logprob_grads(&c.ws.logits, &c.ws.probs, &c.ws.psis, 0, &mut lerrs);
        c.compute_instant_rtrl_grads(&lerrs, 1.0);

        let tol = 6e-4_f64;
        let eps = 1e-3_f32;
        let mut max_abs_g = 0.0_f64;
        let mut check = |name: &str, ana: f32, set_fn: &dyn Fn(&mut UnifiedCayleyCore, f32)| {
            let jp = eval_obj(&base, &|m| set_fn(m, eps));
            let jm = eval_obj(&base, &|m| set_fn(m, -eps));
            let fd = (jp - jm) / (2.0 * eps as f64);
            let err = (ana as f64 - fd).abs();
            max_abs_g = max_abs_g.max(fd.abs());
            assert!(err < tol, "P1 J-XOR FD mismatch on {name}: ana={ana:.6}, fd={fd:.6}, err={err:.6}");
        };

        for b in 0..5usize {
            check("gate_logits", c.ws.g_gates[b], &move |m, d| { m.slots[0].gate_logits[b] += d; });
        }
        for k in [0usize, 1, 2, 8, 12, 14, 15] {
            check("b1_alpha", c.ws.g_b1_alpha[k], &move |m, d| { m.slots[0].b1_alpha[k] += d; });
            check("b1_omega", c.ws.g_b1_omega[k], &move |m, d| { m.slots[0].b1_omega[k] += d; });
        }
        for s in [0usize, 3, 4, 7] {
            check("b2_alpha", c.ws.g_b2_alpha[s], &move |m, d| { m.slots[0].b2_alpha[s] += d; });
            check("b2_omega", c.ws.g_b2_omega[s], &move |m, d| { m.slots[0].b2_omega[s] += d; });
        }
        for j in 0..12usize {
            check("b1_in_w", c.ws.g_b1_inw[j], &move |m, d| { m.slots[0].b1_in_w[j] += d; });
            check("b2_in_w", c.ws.g_b2_inw[j], &move |m, d| { m.slots[0].b2_in_w[j] += d; });
        }
        assert!(max_abs_g > 1e-2, "trivial zero gradient in P1 J-XOR check: {max_abs_g}");

        // 3. Read-only эквивалентность + ленивая аллокация (0 байт при false) + бюджет RAM <= 40 KB на ELEC2
        let qrefs: [&[f32]; 5] = [&seq[0], &seq[1], &seq[2], &seq[3], &seq[4]];
        let ro_pred = base.predict_class_readonly(&qrefs);
        let mut mut_core = base.clone();
        mut_core.start_episode();
        let mut mut_pred = 0usize;
        for x in &seq {
            mut_pred = mut_core.step_supervised_class(x, 0, false, false, false).0;
        }
        assert_eq!(ro_pred, mut_pred, "P1 J-XOR predict_class_readonly diverged");

        let off_core = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config_v050(2, 8, 1));
        assert_eq!(off_core.ws.nl_raw.len(), 0);
        assert_eq!(off_core.ws.grad_nl_raw.len(), 0);
        let on_core = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config(2, 8, 1));
        assert!(
            on_core.memory_footprint_bytes() <= 40 * 1024,
            "P1 ELEC2 RAM {} > 40 KB",
            on_core.memory_footprint_bytes()
        );
    }

    #[test]
    fn test_unified_p2_bank_rotor_adapter_fd_seu_and_readonly() {
        let seq: [[f32; 3]; 5] = [
            [0.8, -0.5, 0.3],
            [-0.6, 0.9, -0.4],
            [0.4, 0.7, -0.8],
            [-0.7, -0.3, 0.6],
            [0.5, -0.6, 0.4],
        ];
        let mut cfg = UnifiedCayleyConfig::one_config(3, 3, 2027);
        cfg.enable_j_xor = true;
        cfg.enable_bank_rotor_adapter = true;
        cfg.learn_norm = false;
        cfg.exact_inv_sqrt = true;
        let mut base = UnifiedCayleyCore::new(cfg);
        base.active_slot = 1; // Проверяем работу адаптера на ненулевом банке с общим хребтом slots[0]!
        for (i, w) in base.slots[1].actor_w.iter_mut().enumerate() {
            *w += 0.13 * (((i as f32) * 0.59).sin());
        }
        for (i, w) in base.slots[1].critic_w.iter_mut().enumerate() {
            *w = 0.14 * (((i as f32) * 0.41 + 0.2).cos());
        }
        for (k, t) in base.slots[1].adapter_b1_t.iter_mut().enumerate() {
            *t = 0.35 * if (k & 1) == 0 { 1.0 } else { -0.8 };
        }
        for (i, g) in base.slots[1].adapter_b2_g.iter_mut().enumerate() {
            *g = 0.28 * (((i as f32) * 0.73).cos());
        }

        let eval_obj = |core_tpl: &UnifiedCayleyCore, mutate: &dyn Fn(&mut UnifiedCayleyCore)| -> f64 {
            let mut c = core_tpl.clone();
            mutate(&mut c);
            c.start_episode();
            for x in &seq {
                c.forward_and_rtrl_traces(x, false, false, false);
            }
            (c.ws.probs[0] as f64).ln() + (c.ws.critic_val as f64)
        };

        let mut c = base.clone();
        c.start_episode();
        for x in &seq {
            c.forward_and_rtrl_traces(x, false, true, false);
        }
        let mut lerrs = [0.0_f32; 3];
        UnifiedCayleyCore::fill_p4_logprob_grads(&c.ws.logits, &c.ws.probs, &c.ws.psis, 0, &mut lerrs);
        c.compute_instant_rtrl_grads(&lerrs, 1.0);

        let tol = 6e-4_f64;
        let eps = 1e-3_f32;
        let mut max_abs_g = 0.0_f64;
        let mut check = |name: &str, ana: f32, set_fn: &dyn Fn(&mut UnifiedCayleyCore, f32)| {
            let jp = eval_obj(&base, &|m| set_fn(m, eps));
            let jm = eval_obj(&base, &|m| set_fn(m, -eps));
            let fd = (jp - jm) / (2.0 * eps as f64);
            let err = (ana as f64 - fd).abs();
            max_abs_g = max_abs_g.max(fd.abs());
            assert!(err < tol, "P2 Adapter FD mismatch on {name}: ana={ana:.6}, fd={fd:.6}, err={err:.6}");
        };

        for k in [0usize, 1, 4, 8, 12, 15] {
            check("adapter_b1_t", c.ws.g_adapter_b1_t[k], &move |m, d| { m.slots[1].adapter_b1_t[k] += d; });
            check("b1_omega_bb", c.ws.g_b1_omega[k], &move |m, d| { m.slots[0].b1_omega[k] += d; });
        }
        for i in [0usize, 1, 2, 9, 10, 11, 21, 22, 23] {
            check("adapter_b2_g", c.ws.g_adapter_b2_g[i], &move |m, d| { m.slots[1].adapter_b2_g[i] += d; });
        }
        for s in [0usize, 3, 4, 7] {
            check("b2_omega_bb", c.ws.g_b2_omega[s], &move |m, d| { m.slots[0].b2_omega[s] += d; });
        }
        assert!(max_abs_g > 1e-2, "trivial zero gradient in P2 check: {max_abs_g}");

        // Read-only эквивалентность на ненулевом банке 1
        let qrefs: [&[f32]; 5] = [&seq[0], &seq[1], &seq[2], &seq[3], &seq[4]];
        let ro_pred = base.predict_class_readonly(&qrefs);
        let mut mut_core = base.clone();
        mut_core.start_episode();
        let mut mut_pred = 0usize;
        for x in &seq {
            mut_pred = mut_core.step_supervised_class(x, 0, false, false, false).0;
        }
        assert_eq!(ro_pred, mut_pred, "P2 Adapter predict_class_readonly diverged on bank 1");

        // SEU-инъекция (включая critic_w и роторный адаптер!) и полная очистка без NaN
        let mut seu_core = base.clone();
        seu_core.slots[1].critic_w[0] = 1.25;
        seu_core.slots[1].adapter_b1_t[0] = 1.25;
        seu_core.slots[1].adapter_b2_g[0] = 1.25;
        seu_core.inject_weight_bit30_seu(1);
        let pred_r = seu_core.step_supervised_reg(&seq[0], 0.5, true, true, true);
        assert!(pred_r.is_finite(), "P2 SEU reg output non-finite");
        assert!(seu_core.seu_faults_scrubbed >= 3, "P2 SEU must scrub critic_w + adapters, got {}", seu_core.seu_faults_scrubbed);

        // RAM бюджет: ELEC2 <= 38.5 KB (<= 40 KB) и 0 байт при отключении
        let off = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config_v050(2, 8, 1));
        assert_eq!(off.slots[0].adapter_b1_t.len(), 0);
        assert_eq!(off.slots[0].adapter_b2_g.len(), 0);
        let on_elec = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config(2, 8, 1));
        assert!(
            on_elec.memory_footprint_bytes() <= 40 * 1024,
            "P2 ELEC2 RAM {} > 40 KB",
            on_elec.memory_footprint_bytes()
        );
    }

    #[test]
    fn test_unified_p3_finsler_randers_diode_fd_and_readonly() {
        // 1. Строгая положительность и необратимая асимметрия tau(v, delta) != tau(v, -delta) + нормированный контраст in (-1, 1)
        let (tp2, _, _) = algebra::finsler_randers_diode_2d((0.8, -0.5), (0.9, -0.4), 0.25, true);
        let (tm2, _, _) = algebra::finsler_randers_diode_2d((0.8, -0.5), (-0.9, 0.4), 0.25, true);
        assert!(tp2 > 0.0 && tm2 > 0.0 && tp2 > tm2 + 0.5, "2D Randers asymmetry failed: {tp2} vs {tm2}");
        let (cp2, _, _) = algebra::finsler_randers_contrast_2d((0.8, -0.5), (0.9, -0.4), 0.25, true);
        let (cm2, _, _) = algebra::finsler_randers_contrast_2d((0.8, -0.5), (-0.9, 0.4), 0.25, true);
        assert!(cp2.abs() < 1.0 && (cp2 + cm2).abs() < 1e-6 && cp2 > 0.3, "2D contrast failed: {cp2} vs {cm2}");
        assert_eq!(algebra::finsler_randers_contrast_2d((0.0, 0.0), (0.9, -0.4), 0.25, true).0, 0.0);
        let (tp3, _, _) = algebra::finsler_randers_diode_3d((0.7, -0.6, 0.5), (0.8, -0.5, 0.4), 0.25, true);
        let (tm3, _, _) = algebra::finsler_randers_diode_3d((0.7, -0.6, 0.5), (-0.8, 0.5, -0.4), 0.25, true);
        assert!(tp3 > 0.0 && tm3 > 0.0 && tp3 > tm3 + 0.5, "3D Randers asymmetry failed: {tp3} vs {tm3}");
        let (cp3, _, _) = algebra::finsler_randers_contrast_3d((0.7, -0.6, 0.5), (0.8, -0.5, 0.4), 0.25, true);
        let (cm3, _, _) = algebra::finsler_randers_contrast_3d((0.7, -0.6, 0.5), (-0.8, 0.5, -0.4), 0.25, true);
        assert!(cp3.abs() < 1.0 && (cp3 + cm3).abs() < 1e-6 && cp3 > 0.3, "3D contrast failed: {cp3} vs {cm3}");
        assert_eq!(algebra::finsler_randers_contrast_3d((0.0, 0.0, 0.0), (0.8, -0.5, 0.4), 0.25, true).0, 0.0);

        // 2. Точная FD-верификация RTRL-градиентов с включённым Finsler-Randers диодом
        let seq: [[f32; 3]; 5] = [
            [0.85, -0.45, 0.35],
            [-0.55, 0.80, -0.30],
            [0.45, 0.65, -0.75],
            [-0.65, -0.35, 0.70],
            [0.60, -0.50, 0.40],
        ];
        let mut cfg = UnifiedCayleyConfig::one_config(3, 3, 2028);
        cfg.enable_finsler_diode = true;
        cfg.learn_norm = false;
        cfg.exact_inv_sqrt = true;
        let mut base = UnifiedCayleyCore::new(cfg);
        for (i, w) in base.slots[0].actor_w.iter_mut().enumerate() {
            *w += 0.14 * (((i as f32) * 0.61).sin());
        }
        for (i, w) in base.slots[0].critic_w.iter_mut().enumerate() {
            *w = 0.13 * (((i as f32) * 0.37 + 0.3).cos());
        }
        for i in 0..base.slots[0].b1_beta.len() {
            base.slots[0].b1_beta[i] = 0.45 * if (i & 1) == 0 { 1.0 } else { -0.8 };
        }
        for i in 0..base.slots[0].b2_beta.len() {
            base.slots[0].b2_beta[i] = 0.40 * if (i & 1) == 0 { 1.0 } else { -0.7 };
        }
        for (i, v) in base.slots[0].b1_vdrift.iter_mut().enumerate() {
            *v = 0.55 * (((i as f32) * 0.83 + 0.1).sin());
        }
        for (i, v) in base.slots[0].b2_vdrift.iter_mut().enumerate() {
            *v = 0.50 * (((i as f32) * 0.71 - 0.2).cos());
        }
        for (i, q) in base.slots[0].b1_qfb.iter_mut().enumerate() {
            *q = 0.25 * (((i as f32) * 0.47).cos());
        }
        for (i, q) in base.slots[0].b2_qfb.iter_mut().enumerate() {
            *q = 0.25 * (((i as f32) * 0.53).sin());
        }

        let eval_obj = |core_tpl: &UnifiedCayleyCore, mutate: &dyn Fn(&mut UnifiedCayleyCore)| -> f64 {
            let mut c = core_tpl.clone();
            mutate(&mut c);
            c.start_episode();
            for x in &seq {
                c.forward_and_rtrl_traces(x, false, false, false);
            }
            (c.ws.probs[0] as f64).ln() + (c.ws.critic_val as f64)
        };

        let mut c = base.clone();
        c.start_episode();
        for x in &seq {
            c.forward_and_rtrl_traces(x, false, true, false);
        }
        let mut lerrs = [0.0_f32; 3];
        UnifiedCayleyCore::fill_p4_logprob_grads(&c.ws.logits, &c.ws.probs, &c.ws.psis, 0, &mut lerrs);
        c.compute_instant_rtrl_grads(&lerrs, 1.0);

        let tol = 6e-4_f64;
        let eps = 1e-3_f32;
        let mut max_abs_g = 0.0_f64;
        let mut check = |name: &str, ana: f32, set_fn: &dyn Fn(&mut UnifiedCayleyCore, f32)| {
            let jp = eval_obj(&base, &|m| set_fn(m, eps));
            let jm = eval_obj(&base, &|m| set_fn(m, -eps));
            let fd = (jp - jm) / (2.0 * eps as f64);
            let err = (ana as f64 - fd).abs();
            max_abs_g = max_abs_g.max(fd.abs());
            assert!(err < tol, "P3 Diode FD mismatch on {name}: ana={ana:.6}, fd={fd:.6}, err={err:.6}");
        };

        for idx in [0usize, 1, 2, 3, 16, 17, 30, 31] {
            check("b1_vdrift", c.ws.g_b1_vdrift[idx], &move |m, d| { m.slots[0].b1_vdrift[idx] += d; });
            check("b1_qfb", c.ws.g_b1_qfb[idx], &move |m, d| { m.slots[0].b1_qfb[idx] += d; });
        }
        for idx in [0usize, 1, 2, 9, 10, 11, 21, 22, 23] {
            check("b2_vdrift", c.ws.g_b2_vdrift[idx], &move |m, d| { m.slots[0].b2_vdrift[idx] += d; });
            check("b2_qfb", c.ws.g_b2_qfb[idx], &move |m, d| { m.slots[0].b2_qfb[idx] += d; });
        }
        for j in 0..12usize {
            check("b1_in_w", c.ws.g_b1_inw[j], &move |m, d| { m.slots[0].b1_in_w[j] += d; });
            check("b2_in_w", c.ws.g_b2_inw[j], &move |m, d| { m.slots[0].b2_in_w[j] += d; });
        }
        for k in [0usize, 1, 8, 15] {
            check("b1_beta", c.ws.g_b1_beta[k], &move |m, d| { m.slots[0].b1_beta[k] += d; });
        }
        for s in [0usize, 3, 7] {
            check("b2_beta", c.ws.g_b2_beta[s], &move |m, d| { m.slots[0].b2_beta[s] += d; });
        }
        assert!(max_abs_g > 1e-2, "trivial zero gradient in P3 diode check: {max_abs_g}");

        // 3. Read-only эквивалентность + 0 байт при отключении + строгие RAM бюджеты
        let qrefs: [&[f32]; 5] = [&seq[0], &seq[1], &seq[2], &seq[3], &seq[4]];
        let ro_pred = base.predict_class_readonly(&qrefs);
        let mut mut_core = base.clone();
        mut_core.start_episode();
        let mut mut_pred = 0usize;
        for x in &seq {
            mut_pred = mut_core.step_supervised_class(x, 0, false, false, false).0;
        }
        assert_eq!(ro_pred, mut_pred, "P3 Diode predict_class_readonly diverged");

        let off = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config_v050(2, 8, 1));
        assert_eq!(off.slots[0].b1_vdrift.len(), 0);
        assert_eq!(off.ws.s_b1_vdrift.len(), 0);
        let on_elec = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config(2, 8, 1));
        let on_rp = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config(3, 3, 1));
        assert!(
            on_elec.memory_footprint_bytes() <= 40 * 1024,
            "P3 ELEC2 RAM {} > 40 KB",
            on_elec.memory_footprint_bytes()
        );
        assert!(
            on_rp.memory_footprint_bytes() <= 32 * 1024,
            "P3 RepeatPrev RAM {} > 32 KB",
            on_rp.memory_footprint_bytes()
        );
    }

    #[test]
    fn test_unified_p4_lyapunov_lk_balance_readonly_and_ram() {
        // 1. Свойства гейта Ляпунова: (0, 1], == 1 ниже порога, строго убывает при росте всплеска
        let g_calm = algebra::lyapunov_backbone_gate(0.10, 0.25, 1.5, 2.25, 0.02);
        let g_mild = algebra::lyapunov_backbone_gate(1.00, 0.25, 1.5, 2.25, 0.02);
        let g_spike = algebra::lyapunov_backbone_gate(10.0, 0.25, 1.5, 2.25, 0.02);
        assert!((g_calm - 1.0).abs() < 1e-6, "calm gate must be 1.0, got {g_calm}");
        assert!(g_mild > 0.0 && g_mild < 1.0, "mild spike must damp gate: {g_mild}");
        assert!(g_spike > 0.0 && g_spike < g_mild, "larger spike must damp more: {g_spike} vs {g_mild}");

        // 2. Свойства рационального баланса Либеля-Кёрнера
        let mut s_hi = 0.5_f32;
        let w0 = algebra::rational_lk_balance_step(&mut s_hi, 4.0, 0.05, 0.35, 0.5, 1.8);
        let w1 = algebra::rational_lk_balance_step(&mut s_hi, 4.0, 0.05, 0.35, 0.5, 1.8);
        assert!(s_hi > 0.5 && w1 < w0 && w1 >= 0.5, "large loss must increase s and lower w: s={s_hi}, w0={w0}, w1={w1}");
        let mut s_lo = 1.0_f32;
        let wl0 = algebra::rational_lk_balance_step(&mut s_lo, 0.01, 0.05, 0.35, 0.5, 1.8);
        let wl1 = algebra::rational_lk_balance_step(&mut s_lo, 0.01, 0.05, 0.35, 0.5, 1.8);
        assert!(s_lo < 1.0 && wl1 > wl0 && wl1 <= 1.8, "small loss must lower s and raise w: s={s_lo}, wl0={wl0}, wl1={wl1}");

        // 3. Read-only эквивалентность ПОСЛЕ count >= 32 (adaptive_var_floor активен) + 10-order jump stability
        let mut core = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config(2, 4, 2029));
        let warm_seq: [[f32; 4]; 8] = [
            [0.01, 0.50, 0.02, 0.80],
            [0.02, 0.51, 0.03, 0.79],
            [0.03, 0.49, 0.01, 0.81],
            [0.04, 0.52, 0.04, 0.78],
            [0.02, 0.48, 0.02, 0.82],
            [0.05, 0.50, 0.03, 0.77],
            [0.01, 0.53, 0.02, 0.80],
            [0.03, 0.47, 0.04, 0.83],
        ];
        core.start_episode();
        for rep in 0..5usize {
            for (t, x) in warm_seq.iter().enumerate() {
                let _ = core.step_supervised_class(x, (t + rep) & 1, true, true, false);
            }
        }
        assert!(core.in_cond.count >= 40);
        let qrefs: [&[f32]; 4] = [&warm_seq[0], &warm_seq[1], &warm_seq[2], &warm_seq[3]];
        let ro_pred = core.predict_class_readonly(&qrefs);
        let mut mut_core = core.clone();
        mut_core.start_episode();
        let mut mut_pred = 0usize;
        for x in &warm_seq[..4] {
            mut_pred = mut_core.step_supervised_class(x, 0, false, false, false).0;
        }
        assert_eq!(ro_pred, mut_pred, "P4 adaptive_var_floor predict_class_readonly diverged");

        // 10-order jump without NaN/Inf
        let r_jump = core.step_supervised_reg(&[1e5, -1e5, 1e-5, 0.0], 0.5, true, true, true);
        assert!(r_jump.is_finite(), "P4 10-order jump non-finite: {r_jump}");

        // 4. Строгий RAM-бюджет на реальных конфигурациях ELEC2 (2, 8) <= 40 KB и RepeatPrev (4, 4) <= 32 KB
        let elec_core = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config(2, 8, 1));
        let rp_core = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config(4, 4, 1));
        assert!(
            elec_core.memory_footprint_bytes() <= 40 * 1024,
            "P4 ELEC2 (2,8) RAM {} > 40 KB",
            elec_core.memory_footprint_bytes()
        );
        assert!(
            rp_core.memory_footprint_bytes() <= 32 * 1024,
            "P4 RepeatPrev (4,4) RAM {} > 32 KB",
            rp_core.memory_footprint_bytes()
        );
    }

    #[test]
    fn test_adaptive_od_p5_anchor_utilization_and_plateau_gating() {
        // 1. RAM бюджет на S_6 (720 классов, M=2, D_MAX=6) строго < 260 KB с m_theta (без плацебо plane_anchor)
        let big = AdaptiveOrthogonalRtrl::new(2, 4, 720, 42);
        assert!(
            big.memory_footprint_bytes() < 260 * 1024,
            "P5 AdaptiveOrthogonalRtrl S6 RAM {} >= 260 KB",
            big.memory_footprint_bytes()
        );

        // 2. Утилизация оболочки: сразу после апгрейда 2->3 новые плоскости ~1e-9 -> active_ratio == 0 -> блокирует ложную эскалацию
        let mut net = AdaptiveOrthogonalRtrl::new(2, 3, 12, 101);
        assert_eq!(net.current_shell_utilization(), (1.0, 1.0));
        assert!(net.upgrade_dimension_isometric());
        assert_eq!(net.active_dim, 3);
        assert_eq!(net.prev_active_dim, 2);
        assert_eq!(net.prev_active_planes, 1);
        let (ar0, e0) = net.current_shell_utilization();
        assert!(ar0 < 0.01 && e0 < 1e-6, "freshly opened shell must report ~0 utilization: ({ar0}, {e0})");
        // Новые плоскости получили свежий rms_theta = 0.04 и обнулённый m_theta = 0.0
        for ch in 0..(net.frames * net.num_inputs) {
            let base = ch * ADAPTIVE_PARAMS_PER_CH;
            assert!((net.rms_theta[base + 1] - 0.04).abs() < 1e-6);
            assert!((net.rms_theta[base + 2] - 0.04).abs() < 1e-6);
            assert!(net.m_theta[base + 1].abs() < 1e-6);
        }

        // 3. Проверка 2-блокового плато в d=2: при активно растущем fast_ema > slow_ema + 0.015 эскалация запрещена!
        let mut climb = AdaptiveOrthogonalRtrl::new(2, 2, 7, 77);
        climb.eps_in_dim = 600;
        climb.fast_ema = 0.80;
        climb.slow_ema = 0.70;
        climb.prev_block_acc = 0.75;
        climb.win_hits = 400;
        climb.win_total = 600;
        climb.block_hits = 148;
        climb.block_total = 199;
        climb.seen_classes = 7;
        assert!(!climb.observe_episode_hit(true, 0), "must not escalate while fast_ema is actively climbing");
        assert_eq!(climb.active_dim, 2);
    }
}
