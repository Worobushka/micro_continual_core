# MicroContinualCore v0.4.1 — Release Notes

**Release Date:** 2026-11-24  
**Base Release:** `v0.4.0` (`206b6d5`)  
**License:** `PolyForm-Noncommercial-1.0.0` *(embedded `crates/axiom_zero`: `MIT OR Apache-2.0`)*

---

## Summary of Changes in v0.4.1

`v0.4.1` is a scientific-verification, causality, and engineering-hygiene patch on top of `v0.4.0` (which retracted `v0.3.0`; see [RELEASE_NOTES_v0.4.0.md](RELEASE_NOTES_v0.4.0.md)).

### 1. Strict Causality & Non-Mutating `&self` $K$-Step Forecasting
- Added strictly non-mutating `&self` $K$-step rollout methods (`MicroContinualCore::predict_vec_k_ahead_readonly`, `CayleyBeliefState::step_state_readonly`, `PlasticCayleyState::step_state_readonly`, and `predict_k_ahead_readonly` across all baseline structs, `OnlineLruStream`, `CayleyBlockSupervised`, and `DeepCayleyStack`), verified bit-identical to mutating step calls (`test_readonly_k_ahead_rollout_exact_equivalence`).
- Converted all multi-step chaos benchmarks (`bench_step5_v040` Suite 3B, `bench_step6_scaling` Axes `1B-3`, `2C`, `4A`, and `bench_forensic_audit` `[A2]`/`[A3]`) to strictly causal delayed-label supervision ($s = t - K$) with `&self` rollouts.
- Added explicit **Causal Prefix Invariance** verification (`xs[t+1..] = f32::NAN`): both `MicroCore` and `Deep 2L LRU` achieve **`6000/6000`** exact invariance in `bench_forensic_audit` `[A2]`.
- **Honest Correction on 1D `Lorenz-63 K=8`:** Removing horizon-overlap label timing retracts the earlier `79–90%` figures; under strict causality, all 1-pass `win_len = 1` models sit at `56.1–59.4%` (`Suite 3B`, $\kappa_T < 0$) and `53.9–55.2%` (`Axis 1B-3`) against `73.92%` sign persistence at lag 8. On causal `Lorenz-96`, `Cayley-RTRL` still cleanly beats `Online GRU` (`Axis 2C`: `64.8–66.3%` vs `59.9%`; `Axis 4A`: `70.7–78.3%` vs `66.1–74.3%` at $d_{\text{in}} \ge 16$).

### 2. Compile-Time `#[axiom_zero]` Hardening & Lean 4
- Hardened `crates/axiom_zero/axiom_zero_macros/src/lib.rs` to also block UFCS transcendental/partial paths and macro-body tokens inside `#[axiom_zero]`; added 4 new `compile_fail` doctests (total: **4 unit + 4 API + 1 doc + 18 `compile_fail` doctests** passing).
- Added `#print axioms step66_bal_not_gt_and_lt` to `proofs/MicroContinualCoreProofs.lean` (**49/49** theorem checks pass with **0 axioms**).

### 3. Core, Controller, SEU & Baseline Fixes
- Raised `PageHinkleyDetector::new` upper threshold clamp from `1000.0` to `1_000_000.0` and guarded multi-bank drift/freezing (`bank_suspect_drift`, `end_episode_inner`, `end_supervised_step`) with `self.banks.len() > 1` (`test_ph_unclamped_high_threshold_and_single_bank_no_freeze`).
- Fixed `GruPerStepBank` `bptt_len == 1` pre-update Jacobian capture order (`dh_buf`/`dr_buf` before `wo`/`un` updates) in both supervised and regression paths.
- Fixed exact RAM accounting in `SelectiveSsmBank::memory_footprint_bytes` (`* 19 -> * 22`), `OnlineGruBaseline::memory_footprint_bytes` (`+ h_cur.len() * 4`), and `CoreScratch::memory_footprint_bytes` (`+ ep_slots_unused`), and switched purely supervised benchmarks to `SelectiveSsmBank::new_supervised`.
- Preconditioned `probe.h_re[0] = 1.25` before Bit-30 flips in `bench_step2_streams` and `bench_step3_pomdp_rl` so unguarded `LRU` baselines reliably report `NO (NaN)`.
- Updated `Axis 3A` ($A_5$) to **10 seeds with `mean ± sd`** (live drift `6.62e-5` / `2.09e-4`) and `Axis 3B` to `s_cay = 12` (`36D` parity) with explicit `GRU(H32,BPTT4)` label.
- Added `--filter <substr>` CLI filtering to `bench_step5_v040` and `bench_step6_scaling`, plus standalone external verification scripts `scripts/bench_cartpole_pytorch.py` and `scripts/bench_elec2_river.py`.

### 4. Documentation & License Hygiene
- Fixed the documented $\Delta(z)$ formula in `README.md` and `PAPER.md` to match the exact piecewise-rational `rational_pos_and_deriv` implementation in `src/math.rs` (no `exp`, no `sqrt`).
- Restored 100% verbatim 201-line official Apache License 2.0 text (including `APPENDIX`) in `crates/axiom_zero/LICENSE-APACHE`.
- Synchronized all benchmark tables in `PAPER.md`, `README.md`, and `RELEASE_NOTES_v0.4.0.md` to post-edit `--release` logs (`29/29` core tests passing).
