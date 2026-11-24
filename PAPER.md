# MicroContinualCore v0.4.1: Zero-Transcendental Algebraic Online Continual RL & Streaming Architecture with Constructive Lean 4 Proofs

**Authors:** Adrian Aleksandrovich İşmuhanov & Sofia Sidorenko  
**Version:** `v0.4.1` (Post-Audit Causality, Macro Hardening & Documentation Sync Release)

---

## Abstract

We present **MicroContinualCore v0.4.0**, a lightweight (`4.9–738.2 KB` RAM depending on configuration; `4.9–454.2 KB` with frozen `i8` encoder), single-pass online continual learning and reinforcement learning architecture implemented in Rust under a strict compile-time total-arithmetic contract (`#[axiom_zero]`). Every core module (`core.rs`, `rtrl.rs`, `math.rs`, `belief.rs`, `memory.rs`, `env_hard_pomdp.rs`) operates with **0% transcendental functions** (compile-time enforced via `[E012]` in `axiom_zero_macros`: `exp, exp2, ln, log, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, powf` are rejected by the procedural macro across method calls, UFCS paths, and macro-body tokens), **0 heap allocations on the numeric hot step path** (`decide_vec`, `observe_step_reward`, `end_supervised_step`), and **total division/indexing**.

The core combines:
1. **3D Rational Cayley Orthogonal Rotors** on $\mathrm{SO}(3)^S$:
   $$R(g) = (I_3 - [g]_\times)^{-1}(I_3 + [g]_\times) = \frac{(1 - \|g\|^2) I_3 + 2 [g]_\times + 2 g g^\top}{1 + \|g\|^2}, \qquad R(g)^\top R(g) = I_3, \quad \det R(g) = +1$$
   together with its double-angle rational composition $R_2(g) = R(g)^2$ and task-agnostic **Proto-Metric $\mathrm{SO}(3)$ readout** (`enable_so3_proto_metric`) for zero-oracle non-Abelian group state tracking up to $L = 500$.
2. **2D Rational Cayley & Algebraic Selective Cayley (`AlgebraicSelectiveCayley`)**: 100% zero-transcendental complex-plane rational rotors $(c, s) = \left(\frac{1-w^2}{1+w^2}, \frac{2w}{1+w^2}\right)$ with exact $O(U \cdot d_{\text{in}})$ RTRL, piecewise-rational $C^1$-smooth input-selective step $\Delta(z) = \min(2,\; 1 + z_c)$ for $z_c \ge 0$ and $\max(0.05,\; \frac{1}{1 - z_c + 0.5 z_c^2})$ for $z_c < 0$ (strictly bounded in $[0.05, 2.0]$ without `exp` or `sqrt`), causal online Welford standardization, supervised classification & continuous regression, and online TD($\lambda$) RL.
3. **Two-Level Fast-Slow 3D Cayley-RTRL (`MicroContinualCore`)** coupled by an orthonormalized 4-row projection matrix (`rtrl_pool_w`, $\|W W^\top - I_4\|_{\max} < 10^{-5}$), optional Band-0 frozen rotor, $W_1$ Kerr kernel memory ($p = 4$), and **causal 6-step parallel read-only bank scoring (`parallel_bank_probe`)** for multi-regime continual learning ($R \in 2..32$, guarded to `banks.len() > 1`).
4. **Shift-Invariant Rational Policies (`p=2` & `p=4`), Tsallis-2 Entropy, and Hardware IEEE-754 Bit-30 SEU Scrubbing** with explicit fault telemetry.
5. **Constructive Lean 4 Formalization** (`proofs/MicroContinualCoreProofs.lean`, **49** verified `#print axioms` checks, **0 axioms** — not even `propext` or `Quot.sound`, including $\det R(g) = +1$, quartic rational softmax positivity, and selective Cayley BIBO contraction + dual derivatives).

Following a two-stage forensic audit (including external `river==0.27.0` and `PyTorch 2.14.0` verification, global Label-Shuffle & `NaN` Causal Prefix Invariance tests, and group-theoretic falsification controls) and a strictly zero-oracle Phase-2 algebraic investigation targeting every diagnosed failure mode, **v0.4.0 formally retracts all methodologically flawed, leaky, and artificially inflated benchmark results published in `v0.3.0` (`10.5281/zenodo.24424983`)** and replaces them with fully audited, causally strict reproducible results alongside every remaining negative and mixed outcome (e.g., strictly causal 1-step `Lorenz-63 K=8` at `56.1–59.4%` with $\kappa_T < 0$, $A_5$ seed variance, `T-Maze L=15` / `Majority Corridor` without hardcoded `novelty_gate`, `ELEC2 +y_{t-1}` gap to Adam `LRU`, `CP+dx` equivalence with `Spinor`-matched LRU, `Dyck-2`, `UnseenHoldout`, and $S_5 \not\le \mathrm{SO}(3)^M$).

> [!CAUTION]
> **Scientific Integrity & Formal Retraction Notice for `v0.3.0` (`10.5281/zenodo.24424983`):**  
> We explicitly acknowledge that during the publication of `v0.3.0` and pre-audit drafts, **benchmark results were published that were methodologically flawed, leaky, and artificially inflated due to unintended code defects, baseline asymmetries, and unvetted automated-coding shortcuts** — including teacher-forced $A_5$ homomorphism/Voronoi shortcuts (`enable_so3_frame_homomorphism`), a synthetic $L=500$ drift loop, holdout vocabulary leakage into `UnseenHoldout` Cloze logs, horizon-dependent T-Maze trace buffers and undisclosed `novelty_gate = 0.28`, asymmetric SEU/baseline handicaps, uncredited `Band 0` `SpinorEncoder` feature lifting on CartPole, state-dimension mismatches on Lorenz-63 (`180D` vs `60D`) and CartPole (`144D` vs `96D`), and silent configuration clamps (`PageHinkleyDetector` cap at `8.0`, `critic_lr` override to `a_lr * 3.5`). **All benchmark results from `v0.3.0` are retracted and superseded by this `v0.4.0` release.** See [RELEASE_NOTES_v0.4.0.md](RELEASE_NOTES_v0.4.0.md) for the complete item-by-item accounting.

---

## 1. Summary of Forensic Audit Remediations & Phase-2 Algebraic Additions in v0.4.0

1. **Compile-Time Transcendental Ban (`[E012]`) & Complete $\mathrm{SO}(3)$ + Selective Cayley Lean 4 Proofs:**
   - Added `[E012]` to `crates/axiom_zero/axiom_zero_macros/src/lib.rs`, rejecting all transcendental math methods (as well as UFCS paths and macro-body tokens) at compile time inside `#[axiom_zero]` modules; verified in `crates/axiom_zero` with **4 unit + 4 API + 1 doc + 18 `compile_fail` doctests**.
   - Added `step109c_cayley3d_det_one` ($\det R(g) = +1$), `step112b_rational_softmax_p4_positive` (quartic rational softmax positivity + dual derivative), `step115_selective_cayley_contraction_and_deriv` (strict BIBO selective contraction + rational cosine dual derivative), and `#print axioms step66_bal_not_gt_and_lt` to `proofs/MicroContinualCoreProofs.lean`; all **49** theorem checks verify with **0 axioms**.
   - Removed the conflicting AGPL license file from `licenses/` and synchronized `Cargo.toml` RAM metadata (`4.9-738 KB RAM; 4.9-454 KB with i8 encoder`).
2. **Honest End-to-End $\mathrm{SO}(3)$ Frame Learning, Proto-Metric Readout & Live IEEE-754 Drift (`src/rtrl.rs`, `bench_step5_v040.rs`, `bench_step6_scaling.rs`):**
   - Removed teacher-forced homomorphism shortcuts (`enable_so3_frame_homomorphism`, `ident_class`, inverse $R_2(-g)$, Voronoi snapping).
   - Replaced with task-agnostic end-to-end training via 1-step cross-entropy and exact RTRL sensitivities `p_wg`, adding a purely algebraic **Proto-Metric $\mathrm{SO}(3)$** readout (`enable_so3_proto_metric`) that reuses `out_w` as $(S^2)^S$ prototypes with `rational_probs_p4_into` (0% `exp`), task-agnostic intra-class center-pull, and symmetric Bit-30 SEU scrubbing — zero group relations, zero `ident_class`, zero evaluation snapping.
   - Removed per-step vector renormalization during normal finite execution of `frame_coupled` frames; over 10 seeds on Axis 3A, live unnormalized `f32` drift at $L = 500$ is $\max |\|u_{500}\|_2 - 1| = 6.62 \times 10^{-5}$ and $\max \|Q_{500}^\top Q_{500} - I_3\|_F = 2.09 \times 10^{-4}$.
3. **2D Rational Cayley & Algebraic Selective Cayley (`AlgebraicSelectiveCayley`, `src/rtrl.rs`, `src/math.rs`):**
   - Added a 100% `#[axiom_zero]` 2D rational oscillator engine supporting both LTI (`selective = false`) and input-selective (`selective = true`) modes via piecewise-rational $C^1$-smooth step `rational_pos_and_deriv` ($\Delta(z) = \min(2,\; 1+z_c)$ for $z_c \ge 0$ and $\max(0.05,\; \frac{1}{1 - z_c + 0.5 z_c^2})$ for $z_c < 0$, 0% `exp` and 0% `sqrt`), exact $O(U \cdot d_{\text{in}})$ RTRL, causal online Welford standardization, Bit-30 State/Weight SEU scrubbing, supervised classification & continuous regression (`decide_regression` / `learn_regression`), and online TD($\lambda$) RL.
4. **Causal 6-Step Parallel Bank Probing (`parallel_bank_probe`, `src/core.rs`), Read-Only $K$-Step Rollouts & Config Integrity (`src/belief.rs`, `src/core.rs`, `src/baselines.rs`):**
   - Added zero-allocation causal read-only parallel scoring across up to 32 archived banks on Page-Hinkley drift detection (`CoreScratch::parallel_bank_hits` / `parallel_bank_margins`, 6 steps `Par6`), delegating directly to bank switching when $\le 1$ mature candidate exists and causally predicting with the running leader on steps 2..6 before committing, eliminating $O(R)$ sequential probe waste.
   - Removed silent `PageHinkleyDetector::new` upper threshold clamp at `8.0` (raised to `1_000_000.0`, so `ph_threshold = 99.0` genuinely disables PH), guarded multi-bank drift/freezing (`bank_suspect_drift`, `end_episode_inner`, `end_supervised_step`) with `self.banks.len() > 1`, and removed silent `critic_lr` override (`a_lr * 3.5`) when `cfg.critic_lr > 0.0` in `MicroContinualCore::apply_online_rtrl_td`.
   - Eliminated horizon-overlap state mutation in multi-step forecasting via non-mutating `&self` rollout methods (`MicroContinualCore::predict_vec_k_ahead_readonly`, `CayleyBeliefState::step_state_readonly`, `PlasticCayleyState::step_state_readonly`, and `predict_k_ahead_readonly` across all baseline structs and `OnlineLruStream`) paired with strictly causal delayed-label training ($s = t - K$, verified `6000/6000` invariant to `xs[t+1..] = f32::NAN`).
5. **Symmetric Physical-SRAM Weight-SEU & 2-Layer State-SEU (`src/core.rs`, `src/rtrl.rs`, `src/baselines.rs`, `bench_step2_streams.rs`, `bench_step3_pomdp_rl.rs`, `bench_step5_v040.rs`):**
   - Replaced asymmetric modular weight targeting with uniform physical SRAM Knuth-hashed indexing (`flip_weight_in_slices`) across all live learnable weight arrays in both `MicroContinualCore` (14 slices) and all baselines, and preconditioned `probe.h_re[0] = 1.25` in stream/POMDP SEU probes so Bit-30 flips reliably trigger IEEE-754 `NaN` in unguarded LRU baselines.
   - Extended `MicroContinualCore::inject_register_bit_flip`, `HierarchicalCayleyRtrl::inject_register_bit_flip`, and `bench_step5_v040` Suite 2 preconditioning/checks to hit both Band 1 (`rtrl_state.u`) and Band 2 (`rtrl_slow_state.u`) symmetrically with `DeepRtrrlLruBank`.
   - Fixed runtime Welford normalization freeze in `MicroContinualCore::decide_vec` to detect raw non-finite/spike inputs (`!input_corrupted`) instead of checking benchmark test flags.
6. **Elimination of All Baseline Bugs, Dead RAM, and Per-Step Allocations (`src/baselines.rs`, `src/stream_baselines.rs`, `src/math.rs`):**
   - Added `fast_dot_raw` (AVX2+FMA without NaN sanitization) so baselines achieve full SIMD throughput while faithfully propagating `NaN`/`Inf` when SEU guards are disabled (fixing the `use_simd = true` bug in supervised `GruPerStepBank`).
   - Fixed loop-carried gradient clobber (`dh_prev[i] = ...` instead of `+=`, missing `dh_prev_buf.fill(0.0)`) and `bptt_len == 1` pre-update Jacobian capture order across GRU BPTT paths (`OnlineGruBaseline::end_episode`, `GruPerStepBank::step_and_learn_supervised`, `GruPerStepBank::decide_vec`), verified by finite-difference unit test `test_gru_per_step_bptt_exact_gradients`.
   - Added `OnlineGruBaseline::end_episode_truncated` (bootstrapping $V(s_T)$ on time-limit truncation instead of treating survival as death) and zero-allocation online truncated BPTT(1, 4) + safe `set_bptt_len` to RL `GruPerStepBank`.
   - Eliminated dead RL-head RAM (~25%) and dead-weight SEU hits in `DeepRtrrlLruBank` via `RtrrlLruBaseline::new_backbone` and supervised `SelectiveSsmBank::new_supervised`; fixed `SelectiveSsmBank::memory_footprint_bytes` (`* 19 -> * 22`), `OnlineGruBaseline::memory_footprint_bytes` (`+ h_cur.len() * 4`), `CoreScratch::memory_footprint_bytes` (`+ ep_slots_unused.capacity() * 16`), `DeepRtrrlLruBank::scrub_state_and_weights()` fault double-counting, `DeepRtrrlLruPhBaseline::set_state_weight_guard` / `had_nan`, and preserved `retired_had_nan` and `retired_seu_faults` on bank resets across all `*PhBaseline` wrappers.
   - Converted `OnlineMlpAdam`, `OnlineLruStream`, `RtrrlLruBaseline`, `SelectiveSsmBank`, `DeepRtrrlLruBank`, `CayleyBlockSupervised`, `DeepCayleyStack`, and `DeepNonAbelianStack` to **0 heap allocations per step**.
7. **Strict State-Dimension Parity, Apples-to-Apples CartPole Audit & Fair `HardPOMDP v2` Ablation (`bench_step1.rs`, `bench_step5_v040.rs`):**
   - Fixed hidden 3× state-dimension mismatch in Suite 3B Lorenz-63 (`MicroCore` matched to exact **60D**: `half_sectors=0, rtrl_sectors=10, rtrl_slow_sectors=10`) and 1.5× mismatch in Suite 3C CartPole (`MicroCore` matched to exact **96D**), adding explicit raw-input `MicroCore PURE 2L+Skip (B0=0, 96D)`, `SpinorEncoder`-matched `1L/Deep 2L LRU (+Spinor 48D)`, frozen deterministic evaluation (`train_mode = false`), physics sanity ceilings, and independent `PyTorch 2.14.0` controls (`scripts/bench_cartpole_pytorch.py`, `scripts/bench_elec2_river.py`).
   - Added a fair `MFEC + PH (fair 0.78+EMA)` control row and honest `N/A (str)` SEU labels in `bench_step1.rs`, and verified zero holdout leakage in `unlabeled_domain_cloze_logs_noisy()`.

---

## 2. Constructive Lean 4 Formalization (49 Checks, 0 Axioms)

All **49** `#print axioms` checks in `proofs/MicroContinualCoreProofs.lean` compile under Lean 4 (`lean proofs/MicroContinualCoreProofs.lean`, exit code 0) with **0 axioms** (`does not depend on any axioms` for every theorem):

| Theorem ID | Formal Statement & Scope | Axioms |
| :--- | :--- | :---: |
| `step01`..`step25` | Constructive Peano arithmetic (`PosNat`) and signed `Balance` ring foundations | 0 |
| `step41`..`step43` | Complex / triphase algebraic neutrality identities | 0 |
| `step62`..`step66` | Dual numbers ($\varepsilon^2 = 0$), algebraic differentiation, and constructive `Balance` strict-order trichotomy (`step66_bal_not_gt_and_lt`) | 0 |
| `step103`..`step106` | Exact 4-point and 3-point DFT orthogonality, inversion, Parseval, and shift theorems | 0 |
| `step107_rational_circle_cayley` | Rational Cayley circle identity $s_2^2 + c_2^2 = 1$ | 0 |
| `step108_su2_lie_laws` | $\mathfrak{su}(2)$ commutator bilinearity, skew-symmetry, and Jacobi identity | 0 |
| `step109_cayley3d_exact_orthogonality` | Exact 3D Cayley rotor orthogonality $R(g)^\top R(g) = I_3$ over exact rationals | 0 |
| `step109b_cayley_rot3_equiv` | Equivalence of vector cross-product form `cayley_rot3_vec` and matrix $R(g)u$ | 0 |
| `step109c_cayley3d_det_one` | Exact determinant $\det(R_{\text{num}}(g)) = (1 + \|g\|^2)^3 \implies \det R(g) = +1$ (strictly $\mathrm{SO}(3)$) | 0 |
| `step110_so3_noncommutativity` | Explicit non-commutativity $R(e_1)R(e_2) \neq R(e_2)R(e_1)$ in $\mathrm{SO}(3)$ | 0 |
| `step111_rtrl_decay_strict_contraction` | Strict contraction $0 < \lambda(\alpha) = \frac{199}{200(1+\alpha^2)} < 1$ | 0 |
| `step112_rational_softmax_positive` | Positivity of shift-invariant rational policy weights $1 - z + \frac{1}{2}z^2 = \frac{1}{2}(1 + (z-1)^2) \ge \frac{1}{2} > 0$ | 0 |
| `step112b_rational_softmax_p4_positive` | Quartic rational softmax positivity $(8 - 4z + z^2)^4 = (4 + (z-2)^2)^4 > 0$ & exact dual derivative $2z - 4$ (`rational_probs_p4_into`) | 0 |
| `step113_kerr_w1_total_denom` | Quadratic ($p = 2$) Kerr denominator totality $\varepsilon + \sum g_i^2 \ge \varepsilon > 0$ and $g_k^2 < \text{den}$ | 0 |
| `step113b_kerr_w1_p4_total_denom` | Quartic ($p = 4$, `KerrKernelBank`) denominator totality $\varepsilon + \sum g_i^4 \ge \varepsilon > 0$ and $g_k^4 < \text{den}$ | 0 |
| `step114_rtrl_exact_derivatives_dual` | Exact dual-number derivatives of $c_1(\omega) = \frac{2\omega}{1+\omega^2}$, $c_2(\omega) = \frac{2\omega^2}{1+\omega^2}$, and $\lambda(\alpha)$ (`src/rtrl.rs`) | 0 |
| `step115_selective_cayley_contraction_and_deriv` | Strict BIBO selective contraction $0 < 998 < 1000(1 + \Delta\alpha^2)$ for any $\Delta > 0$ & exact dual derivative of $c(w) = \frac{1-w^2}{1+w^2}$ (`AlgebraicSelectiveCayley`) | 0 |

*Scope Note:* Theorems in `MicroContinualCoreProofs.lean` prove exact algebraic identities over constructive integers/rationals (`Balance`). In `f32` hardware execution, orthogonal matrices experience standard IEEE-754 rounding drift ($\sim 2.09 \times 10^{-4}$ Frobenius error after 500 unnormalized steps over 10 seeds).

---

## 3. Empirical Benchmark Results (All 8 Release Suites + Independent External Audit)

All benchmarks were executed sequentially on a single machine in `--release` mode (`cargo run --release --bin ...`), with **29/29** core unit & finite-difference gradient tests passing (`cargo test --release --lib`) and **4 unit + 4 API + 1 doc + 18 `compile_fail` doctests** passing in `crates/axiom_zero`.

### 3.1 Step 1: HardPOMDP v2 Textual Continual RL (`bench_step1`)

3 sequential regimes (`Alpha -> Beta -> Gamma -> Alpha Retent.`), 10 seeds, with holdout vocabulary removed from `unlabeled_domain_cloze_logs_noisy()` and an explicit fair-threshold EMA MFEC control (`cos=0.78 + EMA`):

| Suite / Glitch | Model | Score (mean±sd, Holdout) | Retent. | Recovery | RAM (KB) | Latency | SEU-Safe |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict Dense (15% glitch, 240 ep/reg)** | `MicroContinualCore` (no Cloze) | **53.3±10.5%** (H: 0.3%) | 47.6% | **112.5 ep** | 738.2 KB | 29.32 µs | YES |
| | `MicroCore (+Can. Cloze)` | **58.7± 7.5%** (H: 0.0%) | **57.6%** | 126.5 ep | 454.2 KB | 16.74 µs | YES |
| | `MFEC + PH (orig 0.94)` | 0.6± 0.5% (H: 0.0%) | 1.0% | 240.0 ep | 40.7 KB | 8.82 µs | N/A (str) |
| | `MFEC + PH (fair 0.78+EMA)` | 2.7± 2.3% (H: 0.0%) | 3.2% | 240.0 ep | 18.9 KB | 4.55 µs | N/A (str) |
| | `Tabular + PH` | 0.0± 0.1% (H: 0.0%) | 0.2% | 240.0 ep | 70.2 KB | 0.22 µs | N/A (str) |
| | `Online GRU + A2C` | 0.0± 0.0% (H: 0.0%) | 0.0% | 240.0 ep | 114.3 KB | 16.77 µs | NO (NaN) |
| **Strict Sparse (15% glitch, 400 ep/reg)** | `MicroContinualCore` (no Cloze) | **47.2±10.0%** (H: 0.0%) | 25.2% | 235.4 ep | 738.2 KB | 30.88 µs | YES |
| | `MicroCore (+Can. Cloze)` | **54.9±12.0%** (H: 0.0%) | **40.6%** | **197.9 ep** | 454.2 KB | 17.09 µs | YES |
| | `MFEC + PH (orig 0.94)` | 1.0± 1.1% (H: 0.0%) | 0.4% | 400.0 ep | 40.7 KB | 9.15 µs | N/A (str) |
| | `MFEC + PH (fair 0.78+EMA)` | 1.5± 1.3% (H: 0.0%) | 3.0% | 400.0 ep | 17.1 KB | 3.98 µs | N/A (str) |
| | `Online GRU + A2C` | 0.9± 1.0% (H: 0.0%) | 1.9% | 400.0 ep | 114.3 KB | 16.84 µs | NO (NaN) |
| **Strict Sparse (25% glitch, 400 ep/reg)** | `MicroContinualCore` (no Cloze) | **44.7± 8.7%** (H: 0.0%) | **24.4%** | **215.2 ep** | 738.2 KB | 31.82 µs | YES |
| | `MicroCore (+Can. Cloze)` | **48.5±10.4%** (H: 0.0%) | 17.2% | 252.5 ep | 454.2 KB | 17.15 µs | YES |
| | `MFEC + PH (fair 0.78+EMA)` | 1.4± 0.9% (H: 0.3%) | 0.9% | 400.0 ep | 16.7 KB | 3.81 µs | N/A (str) |

*Methodology Note:* Adding fair numeric downweighting, `cos=0.78` coalescing, and EMA belief memory raises `MFEC + PH` from `0.6%` to `2.7%` (dense). The remaining gap to `53.3%` is driven primarily by `MicroCore`'s structured exploration stack (`explore_ucb = 0.48`, `intra_ep_fatigue`, `intrinsic_curiosity = 0.18`, and `KerrKernelBank` aliasing gate) on a 12-action lethal-trap 3-step chain (random chance $1/12^3 \approx 0.058\%$) rather than by Cayley rotors alone. Without Cloze exposure to holdout words, `UnseenHoldout` is `0.0–0.3%`.

---

### 3.2 Step 2: Concept-Drift Data Streams, ELEC2 & External `river` Verification (`bench_step2_streams`)

1-pass Test-then-Train evaluation (10 seeds on synthetic streams; 3 seeds × 45,312 rows on real Electricity `ELEC2`), including Phase-2 zero-transcendental models (`MicroCore+2L-RTRL (PH=8)`, `2D Rat. Cayley`, `Alg.Sel.Cayley`) and an independent benchmark against untouched third-party `river==0.27.0` (`scripts/bench_elec2_river.py`):

| Benchmark | `MicroContinualCore` | `MicroCore+2L-RTRL` | `Online MLP (H=32)+Adam` | `Online Gaussian NB` | `No-Change` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SEA Concepts (5k, 10% noise)** | 82.6±0.3% ($\kappa_T +0.627$) | — | **85.9±0.3%** ($\kappa_T +0.699$) | 84.4±0.3% | 53.4% |
| **STAGGER (3k, A->B->C->A->B)** | 96.1±0.4% ($\kappa_T +0.893$) | — | **97.2±0.3%** ($\kappa_T +0.923$) | 84.0±0.6% | 63.7% |
| **Rotating Hyperplane 10D (5k)** | 77.3±0.9% ($\kappa_T +0.548$) | — | **82.0±0.8%** ($\kappa_T +0.641$) | 77.8±1.1% | 49.8% |
| **Uncorrelated 4-Class RBF (8k)** | **92.8±0.8%** ($\kappa_T +0.905$) | — | 92.6±0.6% ($\kappa_T +0.902$) | 90.7±0.6% | 24.9% |

**Electricity `ELEC2` Real Stream (45,312 rows, 1-pass Test-then-Train, 3 seeds):**

| Model | Raw Acc ($\kappa_T$) | Raw Tail20% | $+y_{t-1}$ Acc ($\kappa_T$) | $+y_{t-1}$ Tail20% | RAM (raw / $+y$) | Lat. (raw / $+y$) | SEU-Safe |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MicroCore+2L-RTRL (PH=8)` (0% exp) | **91.5±0.3%** (**$+0.422$**) | **92.5%** | 91.7±0.1% ($+0.431$) | 92.9% | 27.0 / 28.1 KB | 6.96 / 7.14 µs | **YES** |
| `Alg.Sel.Cayley` (0% exp) | 90.5±0.2% ($+0.355$) | 92.3% | **92.0±0.1%** ($+0.457$) | **93.2%** | 16.0 / 17.5 KB | 11.11 / 12.19 µs | **YES** |
| `2D Rat. Cayley` (0% exp) | 89.4±0.1% ($+0.276$) | 90.4% | 91.0±0.0% ($+0.387$) | 91.5% | 11.5 / 12.5 KB | 8.00 / 8.59 µs | **YES** |
| `MicroCore+2L-RTRL (PH=3.6, orig)` | 89.5±0.3% ($+0.283$) | 91.0% | 90.7±0.2% ($+0.367$) | 92.3% | 27.0 / 28.1 KB | 6.80 / 7.09 µs | **YES** |
| `Online LRU-RTRL` (Adam + `exp`) | 89.3±0.1% ($+0.274$) | 90.5% | **93.4±0.1%** (**$+0.547$**) | **94.6%** | 10.5 / 11.3 KB | 4.09 / 4.22 µs | NO (NaN) |
| `Online MLP (H=32) + Adam` | 87.3±0.1% ($+0.136$) | 88.9% | 91.0±0.0% ($+0.389$) | 91.6% | 5.9 / 6.4 KB | 1.44 / 1.45 µs | NO (NaN) |
| `No-Change (Persist)` | 85.3±0.0% ($+0.000$) | 84.8% | 85.3±0.0% ($+0.000$) | 84.8% | 0.0 KB | 0.00 µs | YES |

**Independent Third-Party `river==0.27.0` Benchmark on Identical `electricity-normalized.arff` (45,312 rows, 1-pass Test-then-Train):**
- `River ARFClassifier (n=10, default)`: **`88.26%`** ($\kappa_T = +0.200$, `75.0 µs/step`) | With $+y_{t-1}$: **`88.65%`** ($\kappa_T = +0.226$, `74.1 µs/step`) | Shuffled $y$: **`54.19%`**
- `River HoeffdingAdaptiveTreeClassifier`: **`83.19%`** ($\kappa_T = -0.146$, `88.2 µs/step`) | With $+y_{t-1}$: **`87.15%`** ($\kappa_T = +0.124$, `82.3 µs/step`)
- `River StandardScaler + LogisticRegression`: **`83.73%`** ($\kappa_T = -0.109$, `10.9 µs/step`)
- Raising Page-Hinkley threshold to suppress volatility false alarms (`PH=8`) brings zero-transcendental `MicroCore+2L-RTRL` to **`91.5%`** raw (**+2.2 pp** over `Online LRU-RTRL` `89.3%`, **+3.2 pp** over `river ARFClassifier` `88.26%` at **10× lower latency**), while zero-transcendental `Alg.Sel.Cayley` reaches **`92.0%`** on $+y_{t-1}$ (still `-1.4 pp` behind transcendental Adam `Online LRU-RTRL` `93.4%`, which lacks SEU safety).

---

### 3.3 Steps 3 & 4: Partially Observable Continual RL & Wave Superposition (`bench_step3_pomdp_rl`, `bench_step4_waves`)

#### A. Non-Stationary Corridor T-Maze ($L \in \{5, 15, 30, 50\}$, Drift $A \to B \to A$, Fixed `trace_capacity = 64`, 10 seeds)

| Horizon / Noise | `MicroCore (ng=0.0)` | `Alg.Sel.Cayley (RL, ng=0.0)` | `MicroCore (+gate=0.28 Abl.)` | `RTRRL-LRU (AAAI'25)` | `Online GRU + A2C` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$L = 5$ ($\pm 0.06$)** | 49.0±4.5% (Ret 49.5%) | **65.1±9.8%** (Ret **66.8%**, `5.6 KB`, `3.23 µs`, SEU=YES) | **99.6±0.8%** (Ret 99.5%) | 56.6±6.4% (Ret 61.2%) | 76.8±10.5% (Ret 45.8%) |
| **$L = 15$ ($\pm 0.35$)** | 49.8±2.7% (Ret 51.0%) | **53.8±5.5%** (Ret **56.2%**, `5.6 KB`, `3.20 µs`, SEU=YES) | **93.1±3.6%** (Ret 94.8%) | 51.1±5.8% (Ret 51.0%) | 47.5±4.1% (Ret 47.0%) |
| **$L = 30$ ($\pm 0.35$)** | 54.2±6.9% (Ret 55.8%) | — | **79.1±6.6%** (Ret 80.2%) | 48.9±5.6% (Ret 55.0%) | 52.8±4.2% (Ret 47.8%) |
| **$L = 50$ ($\pm 0.35$)** | 48.6±5.0% (Ret 48.5%) | — | **58.4±10.7%** (Ret 59.0%) | 48.6±4.0% (Ret 53.2%) | 50.4±7.3% (Ret 52.0%) |

*Honest Assessment:* Without any hardcoded `novelty_gate`, `Alg.Sel.Cayley (RL, ng=0.0)` delivers a genuine **+16.1 pp** gain over `MicroCore (ng=0.0)` at $L=5$ (`65.1%` vs `49.0%`, beating `RTRRL-LRU` by `+8.5 pp` and outperforming `GRU+A2C` on continual retention by `+21.0 pp`), and posts the highest `ng=0.0` score at $L=15$ (`53.8%`, Ret `56.2%`), **but does not solve high-noise $L=15$ sparse-reward T-Maze** (nor `Majority Corridor L=20` in Suite 3A: `48.9±3.9%`). Solving noisy sparse corridors $L \ge 15$ without domain-tuned gating remains an open problem for all pure-RL online models tested.

#### B. POPGym RepeatPrevious ($K \in \{2, 4, 6, 8\}$, 4 Symbols, Pure Bandit/RL $+1/-1$, `novelty_gate = 0.0`)

| Delay $K$ | `Hierarchical 2L Cayley` | `Split-Band 1L Cayley` | `MicroContinualCore (Band-0)` | `RTRRL-LRU (AAAI'25)` | `Online GRU + A2C` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$K = 2$** | **99.5±0.3%** (13.2 KB) | 83.6±3.3% (29.6 KB) | 67.2±3.1% (13.3 KB) | 90.8±8.0% (13.6 KB) | 52.6±11.1% (43.7 KB) |
| **$K = 4$** | **85.7±4.1%** (13.2 KB) | 64.7±8.8% (29.6 KB) | 53.3±7.6% (13.3 KB) | 63.4±6.0% (13.6 KB) | 29.6±4.3% (43.7 KB) |
| **$K = 6$** | **74.3±4.8%** (13.2 KB) | — | 54.0±2.8% (15.7 KB) | 59.3±6.0% (13.6 KB) | 29.0±3.1% (43.7 KB) |
| **$K = 8$** | **62.7±4.0%** (13.2 KB) | — | 47.3±6.9% (15.7 KB) | 44.6±3.6% (13.6 KB) | 27.4±2.5% (43.7 KB) |

#### C. POPGym Position-Only CartPole ($A \to B \to A$ Actuator Inversion) & Wave-Superposition Zero-Shot Transfer

- **Position-Only CartPole (`bench_step3_pomdp_rl`):**
  - Pure 2D $(x, \theta)$: `Cayley-RTRL (pure 2D)` reaches **77.5±11.8 steps** (Retent 109.7 st) vs `Online GRU` 57.3±15.6 st and `RTRRL-LRU` 32.7±2.6 st.
  - With first-difference velocity proxy $(+\Delta x)$: `Cayley-RTRL (+dx)` reaches **183.4±6.0 steps** (Retent 183.1 st, 4.19 µs) vs `MicroCore (Branch B)` 140.9±17.7 st, `Online GRU (+dx)` 93.4±8.2 st (Retent 174.8 st, 42.9 KB), and `RTRRL-LRU (+dx)` 91.0±20.3 st.
- **Wave-Superposition Zero-Shot Transfer (`bench_step4_waves`):**
  - `Hierarchical 2L Cayley`: **93.0%** on $A+B+C$ and **85.0±2.3%** mean zero-shot sub-wave transfer (`8.2 KB`, `3.57 µs`), outperforming `RTRRL-LRU` (`83.2±2.9%`), `Sign(x_t)` heuristic (`75.0±0.1%`), `1L Cayley-RTRL` (`64.7±0.7%`), and `Online GRU + A2C` (`63.7±2.8%`, `39.5 KB`).

---

### 3.4 Unified 7-Stage Internal Ablation (`bench_unified_ablation`)

All stages `[A0]..[A4]` run inside the single `MicroContinualCore` struct with constant `max_regimes` and explicit `[A3+Skip]` separation:

| Stage / Architecture | RepeatPrev $K=4$ | RepeatPrev $K=6$ | CartPole 2D | CartPole $+\Delta x$ | Wave ZS Mean | ELEC2 Acc ($\kappa_T$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `[A0] Frozen Rotor Only (no RTRL)` | 58.0±3.9% | 50.1±7.9% | 27.4±4.9 st | 162.4±18.7 st | 63.8±6.1% | 80.6±0.9% ($-0.321$) |
| `[A1] +1L RTRL Frozen (rtrl_lr=0)` | 56.8±8.9% | 51.0±7.6% | 55.8±9.6 st | 172.4±6.4 st | 65.7±2.0% | 85.3±0.2% ($-0.003$) |
| `[A2] +1L RTRL Trained (coupled)` | 58.7±9.1% | 51.6±7.8% | 77.5±11.8 st | 183.4±6.0 st | 64.7±1.0% | 88.1±0.5% ($+0.189$) |
| `[A3-Frozen] +1L Frozen (dec Qxw)` | 71.2±14.7% | 62.8±14.9% | 35.2±7.7 st | 173.5±4.9 st | **91.0±1.1%** | 85.1±0.1% ($-0.014$) |
| `[A3] +1L RTRL Trained (dec Qxw)` | 76.7±10.0% | 66.3±11.1% | 50.3±5.5 st | 180.9±5.5 st | 88.8±1.8% | 88.3±0.6% ($+0.204$) |
| `[A3+Skip] +1L RTRL + SkipX` | 74.5±8.5% | 65.9±10.3% | 48.0±8.7 st | 180.8±4.2 st | 86.5±3.1% | 89.2±0.2% ($+0.261$) |
| `[A4] Unified 2L Fast-Slow + PH` | **85.8±1.7%** | **77.5±3.4%** | **125.6±22.9 st** | **187.7±4.3 st** | 82.7±3.2% | **89.9±0.2%** (**$+0.314$**) |

---

### 3.5 Step 5: Critique-Response, Matched-Dimension Controls & Symmetric Hardware SEU (`bench_step5_v040`)

#### Suite 1: Non-Abelian Group $A_5 \le \mathrm{SO}(3)$ Word Tracking (End-to-End Supervised, 0% `exp`, Zero Group Oracle)

**Exact Formulation of Task-Agnostic Proto-Metric $\mathrm{SO}(3)$ (`enable_so3_proto_metric`, `src/rtrl.rs`):**
- *Frame & Prototype Initialization ($W_c$):* The state consists of $M$ orthogonal 3D frames ($S = 3M$ coupled unit vectors $u_{t,s} \in S^2$ evolved by $u_{t,s} = R_2(g_{t,m}) u_{t-1,s}$ with $g_{t,m} = (W_g)_m x_t$ and initial generator scale $0.35 + 0.18 m$). Class prototypes $W_c = (w_{c,1}, \dots, w_{c,S}) \in (S^2)^S$ are stored in-place inside `out_w` (normalized onto $(S^2)^S$ at init). On the **first training encounter** of any label $y$ (`proto_seen[y] == 0`), $W_y$ is lazily initialized to the current normalized state $w_{y,s} \leftarrow u_{t,s}/\|u_{t,s}\|_2$ — using **no** hardcoded `ident_class`, **no** group multiplication table, and **no** inverse rotation $R_2(-g)$.
- *Zero-Transcendental Readout & Evaluation:* Logits and quartic rational probabilities (`rational_probs_p4_into`, 0% `exp`, Lean 4 `step112b`) are:
  $$\ell_c(u_t) = \frac{\tau}{S} \sum_{s=1}^{S} w_{c,s}^\top u_{t,s}, \qquad p_c = \frac{\phi_4(\ell_c - \ell_{\max})}{\sum_k \phi_4(\ell_k - \ell_{\max})}, \qquad \phi_4(z) = \left(1 - \frac{z}{2} + \frac{z^2}{8}\right)^{-4}$$
  with shift-invariant rational derivative factor $\psi_c = \frac{4 - 2(\ell_c - \ell_{\max})}{8 - 4(\ell_c - \ell_{\max}) + (\ell_c - \ell_{\max})^2}$. Evaluation uses purely $\hat{y} = \arg\max_c \ell_c(u_t)$ with **zero Voronoi snapping** and **zero state projection/renormalization**.
- *Composite Loss & Prototype / Generator Updates ($W_c, g_t$):* During training (`target = Some(y)`), combining rational cross-entropy with task-agnostic intra-class center-pull $\frac{\beta}{2}\|W_y - u_t\|_2^2$ gives per-class error $e_c = \mathrm{clip}_{[-3,3]}(\psi_c(\mathbf{1}_{c=y} - p_c))$ and state ascent gradient:
  $$\delta_{u_t} = \frac{\tau}{S}\sum_{c=1}^{C} e_c W_c + \beta (W_y - u_t)$$
  Prototypes update by gradient ascent + target-class EMA + $(S^2)^S$ re-projection (training only):
  $$W_c \leftarrow W_c + \eta_{\text{ro}} \frac{\tau}{S} e_c u_t, \qquad W_y \leftarrow (1 - \alpha_{\text{ema}}) W_y + \alpha_{\text{ema}} u_t, \qquad w_{c,s} \leftarrow \frac{w_{c,s}}{\max(\|w_{c,s}\|_2, 10^{-6})}$$
  Generator weights $(W_g)_{m,k,j}$ (determining $g_{t,m} = (W_g)_m x_t$) update via tangent-space projection $\tilde{\delta}_{u_{t,s}} = \delta_{u_{t,s}} - (\delta_{u_{t,s}}^\top u_{t,s}) u_{t,s}$ contracted with exact RTRL sensitivities $\frac{\partial u_{t,s}}{\partial (W_g)_{m,k,j}}$ across the 3 frame columns $a \in \{0,1,2\}$:
  $$\Delta (W_g)_{m,k,j} = \sum_{a=0}^{2} \tilde{\delta}_{u_{t,3m+a}}^\top \frac{\partial u_{t,3m+a}}{\partial (W_g)_{m,k,j}}, \qquad m \leftarrow 0.85\, m + 0.15\,\Delta, \quad v \leftarrow 0.99\, v + 0.01\,\Delta^2$$
  $$(W_g)_{m,k,j} \leftarrow \mathrm{clip}_{[-3.5, 3.5]}\!\left((W_g)_{m,k,j} + \eta_{\text{rtrl}} \,\mathrm{clip}_{[-2.5, 2.5]}\!\left(\frac{m}{\sqrt{v + 10^{-4}}}\right)\right)$$

Trained strictly on word lengths $L \in 1..8$ (4,000 episodes, 10 seeds), evaluated on 60-class exact state / 5-point orbit $p_t(0) \in \{0..4\}$ without any group relations, identity anchor, or Voronoi snapping:

| Model (9D–36D / H=32) | $L = 6$ (60 / 5) | $L = 12$ (60 / 5) | $L = 18$ (60 / 5) | $L = 30$ (60 / 5) | RAM | Latency | Bit-30 SEU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `NonAbelian Cayley SO(3) Proto M=4` | **97.5% / 97.5%** | **88.6% / 89.3%** | **81.8% / 82.7%** | **74.3% / 75.8%** | 26.1 KB | 12.28 µs | **YES** |
| `NonAbelian Cayley SO(3) Proto M=1` | 78.4% / 81.9% | 73.3% / 78.5% | 70.2% / 75.2% | 64.7% / 69.9% | **7.6 KB** | **4.32 µs** | **YES** |
| `NonAbelian 3D Cayley (SO(3) Lin.)` | 27.4% / 38.9% | 8.1% / 24.1% | 5.0% / 22.0% | 3.0% / 18.8% | 25.3 KB | 17.79 µs | **YES** |
| `Fixed-Axis Cayley-RTRL (v0.3.0)` | 5.5% / 21.4% | 2.5% / 18.1% | 1.9% / 20.0% | 1.6% / 18.3% | 20.5 KB | 8.93 µs | NO |
| `Online Selective SSM (Mamba 36D)` | 39.2% / 50.2% | 6.9% / 23.0% | 2.0% / 18.6% | 2.8% / 20.1% | 25.5 KB | 12.48 µs | NO |
| `Online LRU-RTRL (Diagonal 36D)` | 12.2% / 29.4% | 2.5% / 19.2% | 1.6% / 20.1% | 1.7% / 20.7% | 41.9 KB | 7.72 µs | NO |
| `Online GRU + BPTT (H=32)` | 48.8% / 55.7% | 10.1% / 25.1% | 3.2% / 21.0% | 1.6% / 20.4% | 64.8 KB | 13.03 µs | NO |
| `Random Floor (1/60 / 1/5)` | 1.7% / 20.0% | 1.7% / 20.0% | 1.7% / 20.0% | 1.7% / 20.0% | 0.0 KB | 0.01 µs | YES |

*Variance & Purity Note:* At $L = 30$ (Exact-60), `Proto M=4` achieves **`74.3 ± 28.3%`** (`max = 100.0%`, `4/10` seeds $\ge 95\%$) and `Proto M=1` (`9D`, `7.6 KB`) achieves **`64.7 ± 41.4%`** (`max = 100.0%`, `4/10` seeds $\ge 95\%$), while every baseline (`Lin.`, `SSM`, `LRU`, `GRU+BPTT`) collapses to chance (`1.6–3.0%`). The high inter-seed standard deviation reflects genuine non-convex local-minima sensitivity of 1-step online optimization on $\mathrm{SO}(3)$ without a group oracle. See Section 4.1 for Label-Shuffle (`3.50%`) and $S_5 \not\le \mathrm{SO}(3)^M$ contrast verification.

#### Suite 2A & 2B: Symmetric Physical-SRAM Bit-30 State & Weight SEU Injection (`+Clamp` vs `+FinGuard`)

Bit 30 flipped every 25 steps across all live state registers (both Band 1 & Band 2) and uniformly across all live SRAM weights (exact `153.0/153.0` faults in 2A and `345.0/345.0` faults in 2B for every guarded model, 0 double-counting, 0 dead-weight hits):

| Model | Suite 2A: Clean $K=4$ | 2A: State-SEU | 2A: Weight-SEU | Suite 2B: Clean(A,B,C) | 2B: State-SEU | 2B: Weight-SEU | 2B: Ret-C (Cl/St/Wt) | Finite Seeds (St/Wt) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MicroContinualCore [A4]` | **84.6±4.2%** | **83.4±6.0%** | **79.7±5.8%** | **59.6±3.7%** | **60.8±2.4%** | **55.7±2.9%** | **61.1% / 64.3% / 53.1%** | **10/10 / 10/10** |
| `Deep 2L RTRRL-LRU + PH (+Clamp)` | 71.3±2.2% | 33.6±3.6% | 24.9±1.4% | 52.8±2.4% | 33.9±2.3% | 25.9±1.1% | 40.0% / 36.5% / 25.9% | 0/10 / 0/10 |
| `Deep 2L RTRRL-LRU + PH (+FinGuard)` | 71.3±2.2% | 72.2±4.2% | 69.2±5.0% | 52.8±2.4% | 51.7±1.4% | 50.2±1.4% | 40.0% / 38.7% / 35.3% | **10/10 / 10/10** |
| `Online Selective SSM + PH (+Clamp)` | 34.1±5.6% | 25.5±1.6% | 24.9±1.4% | 35.9±1.9% | 25.9±1.0% | 25.9±1.1% | 40.6% / 25.9% / 25.9% | 0/10 / 0/10 |
| `Online Selective SSM + PH (+FinGuard)` | 34.1±5.6% | 34.4±5.9% | 33.0±6.6% | 35.9±1.9% | 35.5±1.6% | 35.5±2.4% | 40.6% / 40.0% / 39.1% | **10/10 / 10/10** |
| `Online Per-Step GRU + PH (+Clamp)` | 25.3±1.0% | 24.6±1.4% | 24.9±1.4% | 24.9±0.7% | 25.8±1.5% | 25.9±1.1% | 25.6% / 26.0% / 25.9% | 0/10 / 0/10 |
| `Online Per-Step GRU + PH (+FinGuard)` | 25.3±1.0% | 25.9±2.1% | 26.0±1.5% | 24.9±0.7% | 25.2±1.2% | 25.3±1.0% | 25.6% / 25.7% / 25.1% | **10/10 / 10/10** |
| `1L RTRRL-LRU Single (+Clamp)` | 62.1±8.1% | 31.7±3.6% | 24.9±1.9% | 52.5±1.7% | 30.9±1.1% | 24.5±0.8% | 56.8% / 34.2% / 24.2% | 0/10 / 0/10 |
| `1L RTRRL-LRU Single (+FinGuard)` | 62.1±8.1% | 62.6±8.7% | 60.3±8.4% | 52.5±1.7% | 52.2±1.8% | 49.9±1.6% | 56.8% / 56.2% / 53.0% | **10/10 / 10/10** |

#### Suite 3A, 3B, 3C: Hard Continual RL, Chaotic Lorenz-63 & Independently Audited Position-Only CartPole

- **Suite 3A — Multi-Cue Balanced Majority Corridor $L = 20$ ($A \to B \to A$, noise $\pm 0.35$, matched `probe_accept_threshold = 0.45`):**
  - `MicroCore (Cayley-RTRL, gate=0.0)`: `51.4±7.8%` (Ret `45.8±7.0%`, `27.2 KB`)
  - `Alg. Sel. Cayley (0% exp, gate=0.0)`: `48.9±3.9%` (Ret `51.0±6.1%`, `5.6 KB`) — *honest negative: chance level (`50%`), confirming sparse terminal RL over 20 noisy steps remains unsolved without explicit gating*
  - `MicroCore [Band-0 + gate=0.28 Abl.]`: **`62.2±8.4%`** (Ret **`73.0±8.2%`**, `75.7 KB`)
  - `Online Selective SSM + PH + Clamp`: `53.2±4.4%` (Ret `50.5±7.5%`, `43.8 KB`)
  - `Online Per-Step GRU + PH + Clamp`: `50.4±6.4%` (Ret `45.8±10.6%`, `291.2 KB`)
  - `1L RTRRL-LRU + Matched PH + Clamp`: `49.0±2.9%` (Ret `51.2±7.5%`, `21.6 KB`)
  - `Deep 2L RTRRL-LRU + PH + Clamp`: `48.5±4.8%` (Ret `48.8±3.8%`, `38.0 KB`)
- **Suite 3B — Chaotic RK4 Lorenz-63 1D-Projection Stream ($\text{sign}(x_{t+8})$, $\Delta t = 0.06$, canonical params, shared causal Welford std, `win_len = 1`, strictly causal delayed-label $s = t-8$ + read-only `&self` rollout, strictly matched 60D state across all models):**
  - `MicroContinualCore (+PH, 60D)`: `56.1±2.0%` Stream Acc, `56.8±5.6%` Return Acc ($\kappa_T = -0.568$, **`12.4 KB`** RAM)
  - `Deep 2L RTRRL-LRU + PH + Clamp (60D)`: `56.2±1.6%` Stream Acc, `56.5±4.1%` Return Acc ($\kappa_T = -0.568$, `28.3 KB` RAM)
  - `Online Selective SSM + PH (60D)`: `56.2±2.2%` Stream Acc, `56.6±5.3%` Return Acc ($\kappa_T = -0.567$, `26.8 KB` RAM)
  - `Online Per-Step GRU + PH (60D, BPTT4)`: `57.1±1.3%` Stream Acc, `57.3±3.8%` Return Acc ($\kappa_T = -0.535$, `729.8 KB` RAM)
  - `Online LRU-RTRL Single-Bank (60D)`: **`59.4±1.3%`** Stream Acc, `53.1±4.3%` Return Acc ($\kappa_T = -0.452$, `4.7 KB` RAM)
  *(Honest Finding: Once horizon-overlap label timing is removed and evaluation uses non-mutating `&self` rollouts `predict_vec_k_ahead_readonly` / `predict_k_ahead_readonly`, Lorenz-63 sign persistence over its average 30.7-step lobes [`73.92%` at lag 8] makes `win_len = 1` 1-pass causal forecasting challenging for every tested online model [`56.1–59.4%`, $\kappa_T < 0$], superseding the retracted `79.3%` horizon-overlap values.)*
- **Suite 3C — Position-Only CartPole $A \to B \to A$ (Matched 96D + Raw/Spinor Feature-Lifting Controls + Frozen Deterministic Eval + Independent PyTorch Verification, 10 seeds; standalone reproduction script `scripts/bench_cartpole_pytorch.py`):**
  - *Physics Sanity Ceilings (500 eps):* `Random = 22.3 st`, memoryless `sign(theta) = 42.5 st`, memoryless `PyTorch MLP (H=32, 2D Phase A) = 50.8±4.0 st`, Oracle 4D-PD (with hidden velocities) `= 200.0 st`.

| Model | 2D Train Sc | 2D Train Ret | 2D Frozen Ev (S/R) | $+dx$ Train Sc | $+dx$ Train Ret | $+dx$ Frozen Ev (S/R) | RAM (4D) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MicroCore (B0+2L, +PH, 96D)` | **117.6±25.9** | **135.0±37.8** | **168.7 / 198.7** | **187.4±3.2** | **190.0±5.4** | 196.2 / 194.5 | **15.9 KB** |
| `MicroCore PURE 2L+Skip (B0=0, 96D)` | 73.4±7.3 | 81.6±16.9 | 149.7 / 151.4 | 127.1±15.0 | 103.7±35.0 | 178.6 / 171.8 | 23.9 KB |
| `Deep 2L RTRRL-LRU + PH (96D)` | 63.3±10.4 | 77.3±14.5 | 137.4 / 159.0 | 138.1±9.6 | 137.8±11.4 | 193.7 / 194.2 | 63.7 KB |
| `Deep 2L LRU + PH (+Spinor 48D)` | 64.8±6.5 | 110.8±12.2 | 94.5 / 169.6 | 184.4±5.4 | 187.7±4.3 | 196.6 / **200.0** | 333.9 KB |
| `1L RTRRL-LRU + PH (96D)` | 50.8±4.4 | 67.4±10.9 | 61.4 / 105.5 | 139.4±5.8 | 146.0±9.0 | 192.2 / 193.9 | 48.3 KB |
| `1L LRU + PH (+Spinor 48D)` | 49.0±3.3 | 60.7±5.1 | 45.3 / 57.6 | 182.9±4.3 | 183.0±6.8 | **200.0** / 198.9 | 317.8 KB |
| `Online Selective SSM + PH (96D)` | 31.3±4.8 | 25.5±6.4 | 33.3 / 30.7 | 115.8±7.0 | 131.4±8.5 | 188.8 / 189.3 | 76.4 KB |
| `Online Episodic GRU + A2C (32D)` | 54.9±16.1 | 108.8±51.6 | 87.4 / 115.9 | 92.8±8.7 | 161.8±52.1 | 109.4 / 171.4 | 42.9 KB |
| `Online Per-Step GRU + PH (32D)` | 43.1±17.1 | 43.5±16.3 | 44.1 / 65.7 | 67.4±21.6 | 71.2±46.1 | 89.0 / 90.9 | 298.8 KB |
| `1L RTRRL-LRU Single-Bank (96D)` | 36.6±3.8 | 37.4±13.6 | 36.2 / 51.9 | 81.0±12.1 | 107.5±59.6 | 122.1 / 161.7 | 12.0 KB |
| *Ext. Control: PyTorch `GRU+A2C+GAE+PH` (H=32, 5 seeds)* | *74.8±26.5* | *76.4±64.2* | *130.7 / 98.9* | *168.6±14.6* | *179.2±18.1* | *174.2 / 194.7* | *PyTorch* |
| *Ext. Control: PyTorch `GRU+A2C+GAE+PH` (H=96, 5 seeds)* | *44.1±16.8* | *44.6±19.8* | *77.0 / 80.5* | *158.5±10.4* | *182.2±13.2* | *184.1 / 187.8* | *PyTorch* |

*Forensic Attribution of Suite 3C:*
1. On **pure position-only `CP-2D`** (memoryless ceiling `42.5–50.8 st`), `MicroCore (B0+2L)` genuinely leads (`117.6` Train / `168.7 / 198.7` Frozen Eval in `15.9 KB`), driven by the synergy between Band-0 (`SpinorEncoder` + fast `CayleyBeliefState` $\lambda = 0.60$) and Band 1+2 Cayley-RTRL — outperforming `MicroCore PURE 2L+Skip` (`73.4` / Ev `149.7/151.4`), `Deep 2L LRU` (`63.3` / Ev `137.4/159.0`), `Deep 2L LRU (+Spinor 48D)` (`64.8` / Ev `94.5/169.6`, `333.9 KB`), and independent `PyTorch GRU+A2C+GAE+PH` (`74.8` / Ev `130.7/98.9`).
2. On **`CP+dx`**, the apparent `+48 st` stochastic-training margin over raw `1L/Deep 2L LRU` (`187.4` vs `138.1–139.4`) is **attributable to `Band 0`'s `SpinorEncoder` + training exploration noise**: on Frozen Eval, raw `Deep 2L LRU` (`193.7/194.2`) and `1L LRU` (`192.2/193.9`) nearly match `MicroCore` (`196.2/194.5`), and `Spinor`-matched `1L/Deep 2L LRU` tie `MicroCore` on both Train (`182.9–184.4`) and Frozen Eval (`196.6–200.0`) — while `MicroCore` uses **20–21× less RAM** (`15.9 KB` vs `317.8–333.9 KB`). Without `Band 0`, `MicroCore PURE 2L+Skip` scores `127.1` Train / `178.6/171.8` Eval on `CP+dx`, slightly trailing raw `Deep 2L LRU`.

---

### 3.6 Step 6: 5-Axis Architectural & Systemic Scalability Suite (`bench_step6_scaling`)

1. **Axis 1A — Width Compute & Memory Scaling ($S \in 4..256$, $D = 3S \in 12..768$, 0 heap allocs/step across all models):**
   - At $D = 768$ ($S = 256$):
     - `Cayley-RTRL`: **`100.1 KB`** RAM, `51.79 µs` step latency
     - `Online LRU-RTRL`: `146.2 KB` RAM, **`51.58 µs`** step latency
     - `Selective SSM`: `240.5 KB` RAM (`4.2, 8.0, 15.5, 30.5, 60.5, 120.5, 240.5 KB` across $D=12..768$), `91.00 µs` step latency
     - `Supervised GRU (BPTT1)`: `21,042.1 KB` (`20.5 MB`) RAM, `1,590.91 µs` step latency (**$210\times$ larger RAM, $30.7\times$ slower** than Cayley-RTRL).
   - Empirical power-law exponents ($D = 12 \to 768$):
     - RAM: `Cayley` $D^{0.99}$, `SSM` $D^{0.97}$, `LRU` $D^{0.94}$ vs `GRU` $D^{1.85}$.
     - Latency: `Cayley` $D^{0.96}$, `SSM` $D^{0.97}$, `LRU` $D^{0.95}$ vs `GRU` $D^{1.56}$.
2. **Axis 1B — Accuracy vs Width (5 seeds on 1B-1/2; 10 seeds on 1B-3; Equal Step Budgets):**
   - **1B-1 ($A_5 \le \mathrm{SO}(3)$ at $L = 30$, Train $L \le 8$, Linear vs Proto-$\mathrm{SO}(3)$):**
     - `S=3 (M=1, 7.6 KB)`: `Lin 1.6±0.3%` $\to$ **`Proto 98.4±2.3%`**
     - `S=6 (M=2, 13.8 KB)`: `Lin 1.5±1.2%` $\to$ `Proto 60.2±33.1%`
     - `S=12 (M=4, 26.1 KB)`: `Lin 3.5±1.7%` $\to$ **`Proto 87.9±20.7%`**
     - `S=24 (M=8, 50.7 KB)`: `Lin 2.8±1.3%` $\to$ `Proto 70.8±26.1%`
     - `S=48 (M=16, 99.9 KB)`: `Lin 5.4±1.9%` $\to$ **`Proto 96.5±6.7%`**
   - **1B-2 & 1B-3 ($S \in 4..64$):** `RepeatPrevious K=8` scales from `34.3±4.7%` ($S=4$) to **`50.6±4.7%`** ($S=64, 19.1\text{ KB}$); strictly causal delayed-label `Chaotic RK4 Lorenz-63 K=8` (`&self` rollout) saturates near chance/persistence across widths: `54.5±2.7%` (`1.3 KB`), `55.0±2.4%` (`2.5 KB`), `55.2±2.2%` (`4.8 KB`), `55.0±2.7%` (`9.6 KB`), and `53.9±1.9%` (`19.1 KB`, $S=64$).
3. **Axis 2 — Depth Without BPTT (`1L..4L` Local RTRL):**
   - On strictly causal delayed-label **Spatiotemporal Lorenz-96 ($N=16, F=8.0, K=6$, eval $t \ge 1500$, `&self` rollout)**: `1L..4L Local-RTRL` achieves **`64.8±1.4% / 64.9±2.2% / 63.4±0.9% / 65.0±2.5%`** (`4.3–23.5 KB`, `5.03–25.42 µs`) and width-matched `1L (S=36)` reaches **`66.3±3.4%`** (`12.5 KB`, `14.08 µs`), cleanly outperforming fixed-BPTT `Online Per-Step GRU (H=48, BPTT4)` at **`59.9±1.9%`** (`134.8 KB`, `36.26 µs`).
   - On **Hierarchical Dyck-2 Bracket Matching**, `Online Per-Step GRU (H=48, BPTT4)` achieves **`91.5±1.7%`** (`95.6%` at depth $\ge 3$), outperforming local-RTRL stacks (`72.2–72.9%`).
4. **Axis 3A — $A_5 \le \mathrm{SO}(3)$ Ultra-Long Horizon Extrapolation (Train strictly $L \le 8 \to$ Eval $L = 6..500$, 10 seeds, 60-elem Exact `mean±sd`, 0% `exp`, 0 Oracle):**

   | Model | $L=6$ | $L=12$ | $L=18$ | $L=30$ | $L=50$ | $L=100$ | $L=200$ | $L=500$ |
   | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
   | `NonAbelian Cayley SO(3) Proto M=4` | **99.2±1.2%** | **96.8±5.7%** | **94.3±9.9%** | **93.3±10.4%** | **92.1±12.4%** | **86.5±16.1%** | **79.3±22.0%** | 49.0±21.8% |
   | `NonAbelian Cayley SO(3) Proto M=8` | 97.5±3.6% | 92.6±9.6% | 89.6±13.0% | 87.9±14.0% | 87.0±17.7% | 82.4±22.5% | 75.7±28.8% | **56.8±31.5%** |
   | `NonAbelian Cayley SO(3) Proto M=1` | 76.7±36.8% | 71.5±43.0% | 70.1±44.0% | 66.5±43.1% | 57.3±38.3% | 38.5±29.1% | 22.3±14.5% | 7.5±6.1% |
   | `NonAbelian Cayley SO(3) (Lin.)` | 30.0±10.5% | 9.1±5.2% | 4.8±2.2% | 3.1±1.9% | 2.6±1.7% | 1.7±2.0% | 1.7±1.3% | 2.5±2.6% |
   | `Online Selective SSM (36D)` | 36.2±9.6% | 4.4±2.1% | 2.6±1.6% | 1.0±1.3% | 2.4±1.2% | 2.0±0.8% | 2.2±1.7% | 1.3±1.2% |
   | `Online GRU + BPTT (H=32)` | 52.9±6.6% | 9.6±2.5% | 3.3±2.0% | 1.6±1.2% | 2.4±1.5% | 1.5±1.0% | 1.8±2.0% | 1.8±1.6% |

   *Live Unnormalized `f32` Drift at $L = 500$ (10 seeds):* $\max |\|u_{500}\|_2 - 1| = 6.62 \times 10^{-5}$, $\max \|Q_{500}^\top Q_{500} - I_3\|_F = 2.09 \times 10^{-4}$.

5. **Axis 3B — RepeatPrevious Deep Delay Horizon ($K \in 2..24$, 5 seeds, Strictly Matched 36D incl. `s_cay=12`, 0% `exp` vs Transcendental):**

   | Delay $K$ | `2D Rat. Cayley (36D, 0% exp)` | `Alg.Sel.Cayley (36D, 0% exp)` | `3D Cayley b=1 (36D)` | `Selective SSM (36D, exp)` | `GRU (H32, BPTT4)` | `2D` RAM |
   | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
   | **$K = 2$** | **100.0±0.0%** | **100.0±0.0%** | 89.4±2.3% | **100.0±0.0%** | **100.0±0.0%** | 4.9 KB |
   | **$K = 4$** | **99.7±0.2%** | 98.9±0.5% | 73.7±2.8% | 96.8±1.5% | 33.2±5.0% | 4.9 KB |
   | **$K = 8$** | **85.1±2.7%** | 71.8±5.8% | 66.3±2.7% | 69.9±2.6% | 25.2±1.8% | 4.9 KB |
   | **$K = 12$** | **66.2±1.6%** | 55.6±3.8% | 29.2±4.0% | 53.2±4.0% | 25.2±1.3% | 4.9 KB |
   | **$K = 16$** | **53.1±4.4%** | 42.8±3.8% | 27.6±1.8% | 39.3±4.0% | 25.4±1.0% | 4.9 KB |
   | **$K = 24$** | **36.6±2.5%** | 33.5±3.5% | 25.4±1.9% | 33.1±2.5% | 24.6±2.2% | 4.9 KB |

   Both zero-transcendental 2D rational models (`2D Rat. Cayley` and `Alg.Sel.Cayley`, **`4.9 KB`** RAM) **outperform transcendental `Selective SSM (36D)` at every delay $K = 4..24$**, with `2D Rat. Cayley` leading by up to **+15.2 pp** ($K=8$).

6. **Axis 4A — True $N = d_{\text{in}}$ Chaotic Lorenz-96 Input Dimensionality Scaling ($d_{\text{in}} \in 4..256$, Strictly Causal Delayed-Label $K=4$ & `&self` Rollout):**
   - `Direct Cayley S=16`: `86.7%` (`4.0 KB, 4.2 µs` at $d=4$), **`78.3%`** (`8.5 KB, 7.4 µs` at $d=16$), **`76.1%`** (`14.5 KB, 12.0 µs` at $d=32$), **`73.5%`** (`26.5 KB, 21.9 µs` at $d=64$), **`70.7%`** (`50.5 KB, 40.0 µs` at $d=128$), and **`72.1%`** (`98.5 KB, 76.1 µs` at $d=256$).
   - `Count-Sketch Proj(d->8) Cayley S=16`: **`87.1%`** (`5.6 KB, 5.3 µs` at $d=4$) to **`63.5%`** at $d_{\text{in}} = 256$ in **`9.5 KB`** RAM and **`5.6 µs`** latency (**$13.6\times$ faster** than direct $d=256$).
   - `Online GRU (H=32, BPTT4)`: **`91.0%`** (`59.9 KB, 18.4 µs` at $d=4$), then `74.3%` ($16$), `73.2%` ($32$), `68.0%` ($64$), `68.4%` ($128$), and `66.1%` (`442.8 KB, 47.6 µs` at $d_{\text{in}} = 256$) — `Direct Cayley S=16` outperforms `Online GRU` at every dimension $d_{\text{in}} \ge 16$ (`70.7–78.3%` vs `66.1–74.3%`, **4.5× smaller RAM** at $d=256$).

7. **Axis 4B — Continual Regime Count Scaling ($R \in 2..32$, Cyclical Regimes, 5 seeds — Sequential vs Causal Parallel `Par6` Probing):**

   | Regimes $R$ | `Core(Seq) Stream` | `Core(Seq) Return` | `Core(Par6) Return` | `1-Bank Return` | Core RAM | Latency |
   | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
   | **$R = 2$** | 91.6±5.8% | **86.0±3.3%** | **86.0±3.3%** | 83.8±5.7% | 17.7 KB | 5.57 µs |
   | **$R = 4$** | 89.5±4.3% | 82.5±5.1% | 83.5±5.5% | **85.1±5.1%** | 27.6 KB | 5.14 µs |
   | **$R = 8$** | 86.7±2.6% | 75.0±4.5% | 84.5±3.7% | **85.5±4.0%** | 47.4 KB | 5.05 µs |
   | **$R = 16$** | 87.0±3.4% | 77.2±5.6% | 85.5±2.2% | **88.0±3.1%** | 86.9 KB | 5.12 µs |
   | **$R = 32$** | 87.0±2.6% | 75.8±5.1% | 84.1±5.4% | **88.0±4.0%** | 166.0 KB | 5.12 µs |

   Causal 6-step parallel read-only bank scoring (`Par6`) recovers **+8.3 to +9.5 pp** over sequential probing at $R \in \{8, 16, 32\}$ while preserving `O(1)` step latency (`5.05–5.57 µs`), beating a true 1-bank learner (with PH genuinely disabled) at $R = 2$ and narrowing the $R = 4..32$ gap to `-1.0..-3.9 pp`.

8. **Axis 5 — Systemic Swarm Scaling ($N = 1..10,000$ Agents, $D = 48$) & 16-Thread Throughput (All 0-Alloc/Step):**
   - `Selective SSM` scales linearly at `15.5 KB/agent` (`61.94 MB @ 4,096`, `151.21 MB @ 10,000`, `6.90 µs`).
   - 16-Thread Multi-Core Throughput ($N = 4,096$ agents): `Cayley-RTRL` reaches **`2.33 M steps/s`** (**`25.3 MB`**), `Online LRU` reaches `2.33 M steps/s` (`39.2 MB`, 1.55× higher RAM), `Selective SSM` reaches `1.37 M steps/s` (`61.9 MB`), and `Supervised GRU` reaches `0.18 M steps/s` (`400.9 MB`) — a **$12.9\times$ throughput advantage** (`15.8×` smaller RAM) for `Cayley-RTRL` over `Supervised GRU`.

---

### 3.7 Step 7: NASA C-MAPSS Aviation Fleet Prognostics, Flight-SEU & Digital Twin Swarm (`bench_step7_nasa_fleet`, `scripts/bench_nasa_external.py`)

Official NASA C-MAPSS turbofan degradation datasets (`FD001–FD004`, piecewise-linear $\mathrm{RUL}_{\max}=125$, strictly causal train-only 6-regime farthest-point K-means normalization with zero test/RUL bits, strict **60D** state parity across all recurrent models, zero dead RL/critic weights, bias-symmetric regression without intercept weight decay, 5 seeds Rust / 3 seeds external PyTorch):

#### Block 1: Streaming RUL (`3-pass`, Frozen Final-Cycle Evaluation) & Anti-Leakage Controls

| Model (`State = 60D`) | `FD001` RMSE / S-Scr / Rec | `FD002` RMSE / S-Scr / Rec | `FD003` RMSE / S-Scr / Rec | `FD004` RMSE / S-Scr / Rec | `FD004` Raw 24D | `FD004` RUL-Shuf | RAM | Lat (µs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MicroCore PURE 2L+Skip (B0=0, 0% exp)` | **19.51±1.22** / 1,479 / 49.6% | **20.89±0.88** / 3,516 / 60.7% | 20.26±1.40 / 1,491 / 54.0% | 23.33±0.62 / 5,255 / 63.0% | 28.16±1.63 | 46.82 (`PASS`) | **14.6 KB** | 9.6–10.4 |
| `Alg.Sel.Cayley (60D, 0% exp)` | 22.28±2.55 / 2,443 / 40.0% | 24.46±4.19 / 8,517 / 57.0% | 23.94±2.96 / 3,920 / 53.0% | 25.33±2.01 / 5,998 / 66.4% | 33.21±8.11 | 45.22 (`PASS`) | 33.0 KB | 24.6–26.6 |
| `2D Rat. Cayley (60D, 0% exp)` | 23.96±2.66 / 3,123 / 45.6% | 25.32±5.42 / 16,724 / 61.6% | 23.03±2.57 / 2,631 / 56.0% | 25.18±1.81 / 6,248 / 60.8% | **27.91±1.55** | 46.40 (`PASS`) | 22.7 KB | 16.6–17.8 |
| `Deep 2L RTRRL-LRU (60D)` | 20.72±3.81 / 2,686 / 52.0% | 22.58±3.30 / 5,868 / 59.0% | **19.85±0.62** / 1,327 / 62.0% | **23.27±0.69** / 4,115 / 70.2% | 28.03±2.10 | 46.87 (`PASS`) | 25.0 KB | **5.2–5.9** |
| `Selective SSM (60D)` | 22.66±4.46 / 1,723 / 29.6% | 25.74±5.36 / 38,920 / 51.1% | 25.42±5.12 / 26,415 / 53.0% | 24.43±0.87 / 5,111 / 50.9% | 34.98±3.18 | 46.45 (`PASS`) | 40.7 KB | 11.6–12.5 |
| `Online GRU BPTT4 (60D)` | 36.19±11.0 / 51,171 / 64.8% | 31.22±7.15 / 31,616 / 66.2% | 32.61±10.1 / 8,103 / 63.0% | 30.79±4.08 / 23,955 / 49.8% | 36.53±6.94 | 46.51 (`PASS`) | 238.1 KB | 31.6–33.9 |
| `Online MLP H=32 + Adam` | 25.54±3.30 / 5,301 / 72.0% | 23.61±3.90 / 12,360 / 76.4% | 24.99±5.93 / 2,116 / 72.0% | 26.62±6.22 / 78,168 / 59.6% | 29.74±3.61 | 47.00 (`PASS`) | 12.0 KB | 2.0–2.1 |
| *External `PyTorch 2.14.0 GRU-60` (full-seq BPTT)* | *17.29±1.74 / 565 / 90.7%* | *15.57±1.02 / 961 / 95.6%* | *18.98±1.56 / 918 / 76.7%* | *17.95±0.40 / 1,584 / 86.2%* | *25.86±1.14* | *44.94 (`PASS`)* | *Python* | *153* |
| *External `river==0.27.0 ARFRegressor`* | *25.70 / 1,855 / 68.0%* | *28.66 / 45,022 / 11.5%* | *23.23 / 1,432 / 5.0%* | *33.39 / 9,414 / 37.7%* | — | — | *Python* | *915–1188* |
| *External `river==0.27.0 HATR`* | *35.10 / 4,177 / 60.0%* | *30.33 / 5,493 / 91.8%* | *24.45 / 2,335 / 80.0%* | *28.40 / 6,559 / 81.1%* | — | — | *Python* | *332–390* |

*Honest Attribution Note:* `MicroCore PURE 2L+Skip (14.6 KB, 0% exp)` ranks #1 in RMSE among all per-cycle online Rust models on `FD001` (`19.51`) and `FD002` (`20.89`), is statistically tied with `Deep 2L LRU` on `FD003` (`20.26` vs `19.85`) and `FD004` (`23.33` vs `23.27`; beats LRU in 1-pass `23.78` vs `24.02`) at **42% less RAM**, and beats external `river ARF/HATR` at 30–100× lower latency; `RUL-Shuffle` passes for all 7 Rust models + external `PyTorch GRU-60` (`44.94–47.00` RMSE); however, all 1-step online streaming models trail offline-style full-trajectory `PyTorch GRU-60` (`15.57–18.98` RMSE at `153 µs/cyc`), which backpropagates through entire 200+ cycle flights per optimizer step.

#### Block 2 (`Flight SEU`, `test_FD004`, 248 Engines, 3 Seeds) & Block 3 (`Fleet Swarm`, $N = 248 \to 4,096$, 16 Threads)
- **Flight SEU (`Sensor-SEU T30=NaN/+1e30` / `State-SEU Bit-30 / 25 cyc` / `Weight-SEU SRAM / 25 cyc`):**
  - `MicroCore PURE 2L+Skip (60D)`: `Clean 23.01 / Sens 22.99 / State 23.00 / Weight 22.98` RMSE, **`248/248`** finite (`1,692 / 1,692` faults = 1,683 injections + 9 real $>8\sigma$ sensor outliers). `Alg.Sel.Cayley (60D, 0% exp)`: `24.05 / 24.05 / 24.04 / 24.04` RMSE, **`248/248`** finite (`1,683 / 1,683` faults).
  - Every `(+Clamp)` baseline crashes to **`0/248`** finite (`NaN`) under State-SEU & Weight-SEU (`Deep 2L LRU`, `Selective SSM`, `Online GRU BPTT4`; stateless `Online MLP` is `N/A` on State-SEU and `0/248` on Weight-SEU). Every `(+FinGuard)` baseline survives **`248/248`** (`1,683 / 1,683` exact faults; e.g., `Deep 2L LRU +FinGuard`: `22.82 / 22.81 / 22.79 / 22.85` RMSE).
- **Fleet Digital Twin Swarm ($N = 4,096$ online-adapting personal twins, 16 threads, 0-alloc):**
  - `MicroCore PURE 2L+Skip (60D)`: **`58.56 MB`** RAM (**16.3× smaller** than `Online GRU BPTT4`'s `952.36 MB`, **1.7× smaller** than `Deep 2L LRU`'s `99.89 MB`), **`0.88 M cycles/s`** (**3.49×** faster than GRU's `0.25 M/s`; `Selective SSM` uses **`162.64 MB`** at `0.83 M/s`; slower in raw throughput than `Deep 2L LRU`'s `1.81 M/s` due to 3D vs 2D rotor FLOPs). See `assets/nasa_cmapss_4panel.png` and `assets/nasa_fleet_mission_control.gif`.

---

## 4. Independent Leakage Verification, Phase-2 Diagnostic Resolutions & Honest Remaining Limitations (`bench_forensic_audit`)

### 4.1 Zero Future-Leakage & Group-Theory Contrast Verification

Shuffling labels globally (preserving class proportions while destroying feature-label and temporal alignment) collapses every evaluated model — including every Phase-2 model — to chance / majority-class floor:

| Dataset & Model | Original Acc | Shuffled-$y$ Acc | Expected Chance / Majority Floor |
| :--- | :---: | :---: | :---: |
| **ELEC2 (45,312 rows):** `MicroCore+2L-RTRL (raw)` | 89.16% | **52.65%** | 57.55% majority (DOWN) |
| **ELEC2 ($+y_{t-1}$):** `MicroCore+2L-RTRL (+y_{t-1})` | 90.50% | **52.32%** | 57.55% majority |
| **ELEC2 (raw):** `2D Rat. Cayley (0% exp)` | 89.45% | **53.22%** | 57.55% majority |
| **ELEC2 (raw / $+y_{t-1}$):** `Alg.Sel.Cayley (0% exp)` | 90.37% / 91.93% | **53.21% / 53.12%** | 57.55% majority |
| **ELEC2 (raw / $+y_{t-1}$):** `Online LRU-RTRL` | 89.31% / 93.29% | **55.09% / 54.64%** | 57.55% majority |
| **ELEC2 (raw):** `River ARFClassifier (v0.27.0)` | 88.26% | **54.19%** | 57.55% majority |
| **ELEC2:** `No-Change Persistent` | 85.33% | **51.17%** | 51.14% ($p^2 + (1-p)^2$) |
| **Causal Chaos `Lorenz-63` (`60D`, 3-regime, $K=8$):** `MicroCore` / `Deep 2L LRU` | 58.35% / 58.05% | **49.43% / 50.25%** | 50.0% (Causal Prefix Invariance: **6000/6000**) |
| **Causal `Lorenz-96` ($N=16 / 256$, $K=4$):** `Cayley (S=16)` / `Sup-GRU` | 71.67% / 62.00% | **52.60% / 48.80%** | 50.0% (`Sup-GRU` Shuf: 52.67% / 50.07%) |
| **$A_5 \le \mathrm{SO}(3)$ ($L=6$, 60-class):** `Proto-Metric SO(3)` | 97.5% | **3.50%** | 1.67% ($1/60$, 200 eps) |

**Group-Theory Falsification Control ($S_5 \not\le \mathrm{SO}(3)^M$, `DIAG 4`):** Running the identical zero-oracle `Proto-Metric SO(3)` model on the 120-element symmetric group $S_5$ (which contains odd permutations and has **no** faithful homomorphism into $\mathrm{SO}(3)^M$, trained on $L \le 8$) yields `L=6: 96.5% -> L=12: 81.0% -> L=18: 60.0% -> L=30: 27.5% -> L=50: 8.0%`. This confirms that $A_5$'s `92–94%` extrapolation at $L = 30..50$ (`49–57%` at $L = 500$) arises genuinely from discovering the faithful $\mathrm{SO}(3)$ representation — and fails exactly where group theory predicts it must fail.

### 4.2 Phase-2 Root-Cause Resolutions vs Remaining Open Limitations

| Diagnosed Failure Axis | Phase-1 Baseline | Phase-2 Zero-Oracle Algebraic Fix | Measured Outcome & Remaining Limitation |
| :--- | :--- | :--- | :--- |
| **1. $A_5 \le \mathrm{SO}(3)$ Extrapolation ($L=30..500$)** | Linear readout: `3.0%` at $L=30$, $\|R(b)^5 - I\|_F = 0.996$ | Task-agnostic **Proto-Metric $\mathrm{SO}(3)$** (`rational_probs_p4_into` + intra-class center-pull, 0% `exp`, 0 group oracle) | **Resolved (with variance):** $\|R(b)^5 - I\|_F$ drops **5.7×** (`0.996 -> 0.175`); Suite 1 `L=30` jumps `3.0% -> 74.3±28.3%` (`M=4`, `4/10` seeds $\ge 95\%$, max `100%`) and Axis 3A (10 seeds) sustains `93.3±10.4%` ($L=30$), `79.3±22.0%` ($L=200$), `49.0–56.8%` ($L=500$). *Open:* non-convex seed variance (`±10–44%`) and non-embeddability of $S_5 \not\le \mathrm{SO}(3)^M$. |
| **2. Axis 3B `RepeatPrevious` ($K=2..24$)** | 3D slow `band=2`: `60.9%` ($K=4$), `26.7%` ($K=8$) vs `Sel-SSM 96.8% / 69.9%` | `band=1` fix + **`AlgebraicSelectiveCayley` (2D Rational & Selective, 36D, `4.9 KB`, 0% `exp`)** | **Fully Resolved:** `2D Rat. Cayley` reaches `99.7% / 85.1% / 66.2% / 53.1% / 36.6%` ($K=4..24$), beating transcendental `Selective SSM` at every delay by up to **+15.2 pp**; `Alg.Sel.Cayley` also beats `Sel-SSM` at every delay (`98.9%..33.5%`). |
| **3. Real `ELEC2` (Raw & $+y_{t-1}$)** | `MicroCore (PH=3.6)`: `89.5%` raw / `90.7%` $+y_{t-1}$ | Volatility-calibrated `PH=8` + **`Alg.Sel.Cayley (0% exp)`** | **Mostly Resolved:** `MicroCore (PH=8)` reaches **`91.5%`** raw (**+2.2 pp** over `LRU` `89.3%`, **+3.2 pp** over `river ARF` `88.26%`, SEU-Safe = YES); `Alg.Sel.Cayley` reaches **`92.0%`** on $+y_{t-1}$ (SEU-Safe = YES). *Open:* `-1.4 pp` behind transcendental Adam `Online LRU-RTRL` (`93.4%`, SEU-Safe = NO) on $+y_{t-1}$. |
| **4. Axis 4B Continual Regimes ($R=2..32$)** | `Core(Seq)`: `75.0–77.2%` at $R=8..32$ vs `1-Bank 85.5–88.0%` | **Causal 6-step parallel read-only bank scoring (`Par6`)** | **Mostly Resolved:** Recovers **+8.3–9.5 pp** (`84.5% / 85.5% / 84.1%` at $R=8/16/32$) in `O(1)` step latency (`5.05–5.57 µs`), beating `1-Bank` at $R=2$ (`86.0%` vs `83.8%`). *Open:* `-1.0` to `-3.9 pp` below true 1-Bank on fast linear hyperplanes at $R=4..32$. |
| **5. `T-Maze` & `Majority Corridor` without `novelty_gate`** | `MicroCore (ng=0.0)`: `49.0%` ($L=5$), `49.8%` ($L=15$), `48.0%` (Maj. $L=20$) | **`Alg.Sel.Cayley` (RL, `ng=0.0`, 0% `exp`, `5.6 KB`)** | **Partial / Honest Mixed:** Gains **+16.1 pp** at T-Maze $L=5$ (`65.1%`, Ret `66.8%`, beating `RTRRL-LRU` `56.6%` and `GRU` Ret `45.8%`) and leads all `ng=0.0` models at $L=15$ (`53.8%`), **but does not solve high-noise $L=15$ T-Maze or Majority Corridor $L=20$ (`48.9%`)**. |

**Additional Unchanged Architectural Boundaries:**
- **Strictly Causal 1-Step `Lorenz-63 K=8` (`56.1–59.4%` in Suite 3B; `53.9–55.2%` in Axis 1B-3; $\kappa_T < 0$):** Under strictly causal delayed-label supervision ($s = t - 8$, `6000/6000` invariant to future `NaN`) and non-mutating `&self` rollouts (`predict_vec_k_ahead_readonly`), Lorenz-63's long lobe sign persistence (`73.92%` at lag 8) causes single-pass `win_len=1` online models (`MicroCore`, `Deep 2L LRU`, `Sel-SSM`, `Per-Step GRU`, `1L LRU`) to trail persistence (`KappaT -0.452` to `-0.568`; prior horizon-overlap `79–90%` figures are retracted). On multi-dimensional causal `Lorenz-96` (`Axis 2C` & `Axis 4A`), `Cayley-RTRL` cleanly beats `Online GRU` (`64.8–66.3%` vs `59.9%` in `Axis 2C`; `70.7–78.3%` vs `66.1–74.3%` for $d_{\text{in}} \ge 16$ in `Axis 4A`).
- **Zero-Shot `UnseenHoldout` (`0.0–0.3%`, `bench_step1`):** FNV-1a token hashing maps disjoint unseen words to orthogonal random hash slots; without lexical overlap or shared co-occurrence, zero-shot semantic transfer is information-theoretically impossible.
- **i.i.d. Static Tabular Drift (`SEA`, `STAGGER`, `Rotating Hyperplane`):** Fixed random `SpinorEncoder` + linear Delta-Rule (`rtrl_sectors = 0`) still trails a trainable 2-layer ReLU `OnlineMlpAdam` (`-1.1` to `-4.7 pp`).
- **Hierarchical `Dyck-2` Bracket Stack (`72.9%` vs `Sup-GRU` `91.5%`):** Multiplicative state-gated stack push/pop with BPTT remains superior to LTI local-DFA stacks.
- **SEU Resilience Attribution:** Hardware bit-flip survival in floating-point recurrent networks requires explicit non-finite/bound scrubbing (`scrub_state_seu`/`scrub_weight_seu` in `MicroContinualCore`/`AlgebraicSelectiveCayley`, or `+FinGuard` on baselines) alongside bounded rational dynamics.
