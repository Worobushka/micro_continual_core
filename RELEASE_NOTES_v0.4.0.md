# MicroContinualCore v0.4.0 — Release Notes & Formal Scientific Correction of v0.3.0

**Release Date:** 2026-11-22  
**Supersedes:** `v0.3.0` (`10.5281/zenodo.24424983`) — **RETRACTED / SUPERSEDED**  
**License:** PolyForm-Noncommercial-1.0.0

---

## 1. Scientific Integrity & Retraction Notice for v0.3.0

**We explicitly and transparently disclose that during the publication of `v0.3.0` (Zenodo `10.5281/zenodo.24424983`) and earlier pre-audit drafts, benchmark results were published that were methodologically flawed, leaky, and artificially inflated due to unintended code defects, baseline asymmetries, and unvetted automated-coding shortcuts in pre-audit drafts.**

Those results must **NOT** be cited or relied upon. Every benchmark claim from `v0.3.0` is hereby **retracted and superseded by `v0.4.0`**, which underwent a comprehensive multi-stage forensic code audit, complete removal of all shortcuts/leaks, external third-party verification (`river==0.27.0`, `PyTorch 2.14.0`), global Label-Shuffle leakage tests, and group-theoretic falsification controls.

### Complete Accounting of Defects, Shortcuts, and Asymmetries Removed from v0.3.0 / Pre-Audit Drafts

1. **Teacher-Forced $A_5 \le \mathrm{SO}(3)$ Homomorphism Oracle & Voronoi Snapping (`src/rtrl.rs`, `bench_step5`, `bench_step6`):**
   - Pre-audit code contained `enable_so3_frame_homomorphism`, which used ground-truth transition pairs $(y_{t-1}, y_t)$, hardcoded identity-class anchoring (`ident_class`), inverse rotations $R_2(-g)$, and evaluation-time Voronoi snapping to prototypes, alongside a synthetic standalone rotation loop for $L = 500$ drift.
   - **Remediation in `v0.4.0`:** Completely deleted. Replaced with strictly zero-oracle end-to-end training (`enable_so3_frames_end_to_end` and task-agnostic `enable_so3_proto_metric`, 0% `exp`, zero group relations, zero evaluation snapping) and live unnormalized `f32` drift tracking.
2. **Holdout Vocabulary Leakage in `HardPOMDP v2` Unlabeled Cloze Logs (`src/env_hard_pomdp.rs`, `bench_step1`):**
   - `unlabeled_domain_cloze_logs_noisy()` previously contained holdout test vocabulary (`telemetry`, `nvme`, `sector`, `malformed`, `zonealpha..zonedelta`, `postcheck`, `corrupt`), artificially fabricating zero-shot `UnseenHoldout` accuracy.
   - **Remediation in `v0.4.0`:** Stripped all holdout tokens (enforced by unit test `test_cloze_canonical_context_without_holdout_leakage`). Honest zero-shot `UnseenHoldout` accuracy is **`0.0–0.3%`**.
3. **Horizon-Dependent Trace Buffer Leakage & Undisclosed Cue-Gating on T-Maze (`bench_step3`, `bench_step4`, `bench_step5`):**
   - T-Maze previously scaled `trace_capacity` with `corridor_len + 4` and relied on hardcoded `novelty_gate = 0.28` without isolating ungated behavior.
   - **Remediation in `v0.4.0`:** Fixed `trace_capacity = 64` across all horizons and separated primary `novelty_gate = 0.0` runs (`~49%` chance on `MicroCore`) from explicitly labeled `+gate=0.28` ablations.
4. **Asymmetric SEU Injection, Baseline Bugs, State-Dimension Mismatches & Uncredited Feature Lifting (`src/core.rs`, `src/baselines.rs`, `bench_step1`, `bench_step5`):**
   - Pre-audit weight-SEU did not symmetrically pre-condition `MicroCore` weights to trigger IEEE-754 `NaN` on Bit-30 XOR and hit only Band-1 state; baselines suffered from loop-carried GRU BPTT gradient clobber bugs, ~25% dead RL-head RAM and dead-weight SEU hits in `DeepRtrrlLruBank`, double-counted SEU faults, missing `fast_dot_raw` NaN propagation, per-step heap allocations, missing truncation bootstrapping in `OnlineGruBaseline`, and uncalibrated `MFEC` thresholds.
   - `MicroCore` previously had 3× larger state dimension on Lorenz-63 (`180D` vs `60D`), 1.5× larger state dimension on CartPole (`144D` vs `96D`), and uncredited `Band 0` `SpinorEncoder` nonlinear lifting vs raw-input baselines on CartPole `CP+dx` (evaluated only on stochastic training tails without frozen deterministic eval).
   - **Remediation in `v0.4.0`:** Fixed every baseline bug (verified by finite-difference test `test_gru_per_step_bptt_exact_gradients`), unified symmetric physical-SRAM Knuth-hashed Weight-SEU and 2-layer State-SEU (`153.0/153.0` and `345.0/345.0` exact faults), enforced strict `60D` Lorenz-63 and `96D` CartPole parity, and added raw-input `MicroCore PURE 2L+Skip (B0=0)`, `SpinorEncoder`-matched `1L/Deep 2L LRU`, frozen deterministic evaluation, and independent `PyTorch 2.14.0` controls to Suite 3C.
5. **Unenforced Zero-Transcendental Contract, Missing $\det R = +1$ Proof & Silent Config Clamps (`axiom_zero_macros`, `proofs/`, `src/belief.rs`, `src/core.rs`):**
   - `#[axiom_zero]` previously did not reject `.exp()/.sin()/.cos()` at compile time; Lean 4 proved $R^\top R = I$ without $\det R = +1$; `PageHinkleyDetector::new` silently clamped `threshold` at `8.0` (breaking `ph_threshold = 99.0`); `MicroContinualCore` silently overrode `critic_lr` to `actor_critic_lr * 3.5`.
   - **Remediation in `v0.4.0`:** Added compile-time ban `[E012]`, proved $\det R(g) = +1$ (`step109c`) + quartic rational softmax (`step112b`) + selective Cayley BIBO contraction (`step115`) in Lean 4 (**48/48 checks, 0 axioms**), raised PH threshold upper clamp to `1000.0`, and respected explicit `critic_lr > 0.0`.

---

## 2. What Is Genuinely Verified in v0.4.0

- **Formal Verification & Contract Purity:**
  - **Lean 4:** `lean proofs\MicroContinualCoreProofs.lean` passes **48/48** `#print axioms` checks with **0 axioms**.
  - **Rust Compile-Time Contract:** 0% transcendentals inside `#[axiom_zero]` (`[E012]`), 0 heap allocations on numeric hot paths, **26/26** unit & exact finite-difference gradient tests passing (`cargo test --release --lib`), including continuous regression contour (`decide_reg_vec` / `end_reg_step`) and GRU BPTT(4) regression exact gradients.
  - **Zero Future-Leakage Proof:** Global Label-Shuffle collapses every model to chance/majority floor on `ELEC2` (`52.3–53.2%` vs `57.55%` floor), `Lorenz-63` (`50.05%`), `Lorenz-96` (`48.3–49.7%`), and $A_5$ (`3.50%` vs `1.67%`).
  - **Group-Theory Falsification Control:** Identical zero-oracle `Proto-Metric SO(3)` model collapses on $S_5 \not\le \mathrm{SO}(3)^M$ (`96.5%` at $L=6$ $\to$ `27.5%` at $L=30$ $\to$ `8.0%` at $L=50$) exactly as predicted by group theory.
- **Core Verified Engineering Results (`--release`):**
  - **$A_5 \le \mathrm{SO}(3)$ Zero-Oracle Extrapolation (`Train L ≤ 8`, 0% `exp`):** `Proto M=4` (`26.1 KB`, `13.49 µs`, SEU=YES) achieves **`74.3 ± 28.3%`** at $L=30$ (10 seeds, `4/10` $\ge 95\%$, max `100%`) and sustains **`94.8%`** ($L=30$), **`79.3%`** ($L=200$), **`46.0%`** ($L=500$; `M=8`: `51.7%`) on Axis 3A (5 seeds), vs `1.6–3.0%` for `Lin.`, `Selective SSM`, `LRU`, and `GRU+BPTT`. Live unnormalized `f32` drift at $L=500$: `4.50e-5` norm, `1.43e-4` Frobenius.
  - **Axis 3B `RepeatPrevious` ($K=2..24$, Matched `36D`, `4.9 KB`, 0% `exp`):** `2D Rat. Cayley` (`99.7% / 85.1% / 66.2% / 53.1% / 36.6%`) and `Alg.Sel.Cayley` (`98.9% / 71.8% / 55.6% / 42.8% / 33.5%`) both beat transcendental `Selective SSM (36D)` (`96.8% / 69.9% / 53.2% / 39.3% / 33.1%`) at every delay $K=4..24$.
  - **Real `ELEC2` (45,312 rows, 1-pass Test-then-Train):** `MicroCore+2L-RTRL (PH=8)` reaches **`91.5 ± 0.3%`** raw (SEU=YES, `7.61 µs`), beating `Online LRU-RTRL` (`89.3%`, SEU=NO) and third-party `river==0.27.0 ARFClassifier` (`88.26%`, `75 µs`); `Alg.Sel.Cayley (0% exp)` reaches **`92.0 ± 0.1%`** on $+y_{t-1}$ (SEU=YES).
  - **Suite 3C Position-Only CartPole (`CP-2D` & `CP+dx`, 10 seeds, Frozen Eval + Raw/Spinor + PyTorch Controls):**
    - Physics floor: Random `22.3 st`, memoryless `sign(theta)` `42.5 st`, memoryless `PyTorch MLP` `50.8 st`, Oracle 4D-PD `200.0 st`.
    - On **pure 2D (`CP-2D`)**: `MicroCore (B0+2L, 96D, 15.9 KB)` reaches **`117.6 ± 25.9`** Train / **`168.7 / 198.7`** Frozen Eval, beating raw `MicroCore PURE 2L+Skip (B0=0)` (`73.4` / Ev `149.7/151.4`), `Deep 2L LRU` (`63.3` / Ev `137.4/159.0`), `Deep 2L LRU (+Spinor 48D)` (`64.8` / Ev `94.5/169.6`, `333.9 KB`), and independent `PyTorch GRU+A2C+GAE+PH` (`74.8` / Ev `130.7/98.9`).
    - On **`CP+dx`**: `MicroCore (B0+2L)` (`187.4` / Ev `196.2/194.5`, `15.9 KB`) is statistically tied on Frozen Eval with raw `Deep 2L LRU` (`193.7/194.2`), raw `1L LRU` (`192.2/193.9`), `Spinor`-matched `1L/Deep 2L LRU` (`182.9–184.4` / Ev `196.6–200.0`, `318–334 KB`), and `PyTorch GRU+A2C+GAE+PH` (`168.6` / Ev `174.2/194.7`), proving the `CP+dx` stochastic-training gap was driven by `SpinorEncoder` rather than rotors alone, while retaining a **20× RAM advantage**.
  - **Symmetric Bit-30 SEU (Suite 2A/2B):** `MicroContinualCore [A4]` achieves **`84.6% / 83.4% / 79.7%`** (2A Clean/State-SEU/Weight-SEU) and **`59.6% / 60.8% / 55.7%`** (2B) with `10/10` finite seeds (`153.0/153.0` and `345.0/345.0` exact faults).
  - **Axis 4B Continual Regimes ($R=2..32$):** Causal 6-step parallel read-only probing (`Par6`) recovers **+8.3–9.5 pp** over sequential probing (`84.1–86.0%` Return Acc in `5.13–5.34 µs`), beating `1-Bank` at $R=2$ and narrowing $R=4..32$ gaps to `-1.0..-3.9 pp`.
  - **Axis 1A & Axis 5 Scaling:** Linear $O(D)$ RAM/latency ($\text{RAM} \propto D^{0.99}$, `100.1 KB` at $D=768$ vs `21.0 MB` for GRU) and **`2.29 M steps/s`** on 16 threads ($N=4,096$, `25.3 MB`, **12.7×** faster than GRU).
  - **Step 7 — NASA C-MAPSS Aviation Fleet Prognostics, Flight-SEU & Swarm (`bench_step7_nasa_fleet`, `FD001–FD004`, Strict `60D` Parity):**
    - `MicroCore PURE 2L+Skip (14.6 KB, 0% exp)` reaches **`19.51 / 20.89 / 20.26 / 23.33`** 3-pass RMSE (leading Rust streaming models on `FD001–FD002`, tied with `Deep 2L LRU` `20.72 / 22.58 / 19.85 / 23.27` at 41% lower RAM, beating `river==0.27.0 ARF` `25.70–33.39`, while trailing offline full-trajectory `PyTorch 2.14.0 GRU-60` `15.57–18.98`). `RUL-Shuffle` passes for all 8 models (`44.94–47.00` RMSE).
    - **Flight SEU (`test_FD004`, 248 engines):** `MicroCore` (`23.01 -> 23.00 / 22.98` RMSE) and `Alg.Sel.Cayley` (`24.05 -> 24.04`) keep **`248/248`** engines finite, all `(+Clamp)` baselines fail **`0/248`**, and all `(+FinGuard)` baselines survive **`248/248`**.
    - **Fleet Swarm ($N = 4,096$ personal twins, 16 threads):** `MicroCore PURE 2L` uses **`58.56 MB`** (16.3× smaller than GRU `952.36 MB`) at **`0.85 M cycles/s`** (3.70× faster than GRU, vs `Deep 2L LRU` `99.89 MB` / `1.64 M/s`).

---

## 3. Honest Remaining Limitations (Explicitly Unsolved in v0.4.0)

1. **`UnseenHoldout` zero-shot lexical transfer (`0.0–0.3%`):** Disjoint unseen FNV-1a hashed tokens cannot generalize zero-shot.
2. **High-noise sparse corridors without hardcoded gating:** `T-Maze L=15` (`53.8%` `Alg.Sel.Cayley`, `49.8%` `MicroCore`) and `Majority Corridor L=20` (`48.9–51.4%`) remain near chance for all ungated pure-RL models tested (though `Alg.Sel.Cayley` gains `+16.1 pp` on `T-Maze L=5` -> `65.1%`).
3. **$A_5$ inter-seed variance & $S_5$ non-embeddability:** `Proto M=4` has `±28.3%` inter-seed std on 10 seeds (`4/10` $\ge 95\%$) due to non-convex online local minima; $S_5 \not\le \mathrm{SO}(3)^M$ cannot extrapolate past $L \approx 20$.
4. **`ELEC2 +y_{t-1}` (`92.0%` vs `Online LRU-RTRL` `93.4%`):** Zero-transcendental models still trail Adam+Softmax `Online LRU-RTRL` (SEU-unsafe) by `-1.4 pp`.
5. **Static i.i.d. tabular drift (`SEA`, `STAGGER`, `Rotating Hyperplane`):** Fixed random `SpinorEncoder` + Delta-Rule trails trainable 2-layer ReLU `OnlineMlpAdam` (`-1.1` to `-4.7 pp`).
6. **Hierarchical `Dyck-2` (`72.9%` vs `Sup-GRU` `91.5%`):** Multiplicative state-gated stack push/pop with BPTT outperforms local-DFA LTI stacks.
7. **SEU Resilience Attribution:** Survival under IEEE-754 bit flips requires explicit non-finite/bound scrubbing (`scrub_state_seu` / `scrub_weight_seu` or `+FinGuard` on baselines) alongside bounded rational dynamics.

---

## 4. Upgrading an Existing `v0.3.0` Repository

When syncing `v0.4.0` onto an existing `v0.3.0` Git clone of `Worobushka/micro_continual_core`:
1. **Remove obsolete conflicting license file:** Run `git rm -r licenses` to delete `licenses/LICENSE-AGPL-3.0-only.txt` so GitHub and downstream scanners see a single root license (`PolyForm-Noncommercial-1.0.0`).
2. **Exclude local scratch/build/data files:** Ensure `context.txt`, `target/`, `CMAPSSData/`, `*.zip`, and `logs/` are not committed (all are listed in `.gitignore`).
3. **Add new `v0.4.0` files:** Verify `RELEASE_NOTES_v0.4.0.md`, `src/bin/bench_step5_v040.rs`, `src/bin/bench_step6_scaling.rs`, `src/bin/bench_step7_nasa_fleet.rs`, `src/bin/bench_forensic_audit.rs`, `scripts/bench_nasa_external.py`, `scripts/plot_nasa_fleet.py`, and `assets/nasa_cmapss_4panel.png` / `assets/nasa_fleet_mission_control.gif` (and optional `.mp4`) are staged.
4. **Update `v0.3.0` release notice:** After tagging `v0.4.0` (which mints a new Zenodo version DOI under concept DOI `10.5281/zenodo.24424982`), append a retraction notice pointing to `v0.4.0` on the existing `v0.3.0` GitHub Release and Zenodo record `10.5281/zenodo.24424983`.

---

## 5. License & Authorship

- **Authors:** Adrian Aleksandrovich İşmuhanov (`0009-0002-4761-1952`) & Sofia Sidorenko (`0009-0006-5368-062X`).
- **Root crate, benchmarks & Lean 4 proofs:** `PolyForm-Noncommercial-1.0.0` ([LICENSE](LICENSE)).
- **Embedded `crates/axiom_zero` & `axiom_zero_macros`:** `MIT OR Apache-2.0` ([LICENSE-MIT](crates/axiom_zero/LICENSE-MIT), [LICENSE-APACHE](crates/axiom_zero/LICENSE-APACHE) — full verbatim Apache-2.0 text restored in `v0.4.0`).
