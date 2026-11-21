# MicroContinualCore v0.3.0

**Authors:** Adrian Aleksandrovich İşmuhanov ([ORCID `0009-0002-4761-1952`](https://orcid.org/0009-0002-4761-1952)), Sofia Sidorenko ([ORCID `0009-0006-5368-062X`](https://orcid.org/0009-0006-5368-062X), [GitHub `@serjliasereev-KasaneTeto`](https://github.com/serjliasereev-KasaneTeto))  
**Repository:** [https://github.com/Worobushka/micro_continual_core](https://github.com/Worobushka/micro_continual_core)  
**Full Paper:** [`PAPER.md`](PAPER.md)  
**Lean 4 Formalization (0 Axioms):** [`proofs/MicroContinualCoreProofs.lean`](proofs/MicroContinualCoreProofs.lean)  
**License:** `PolyForm-Noncommercial-1.0.0` ([`LICENSE`](LICENSE))  

A unified, cache-resident (`8–34 KB` in streaming/RTRL configurations; `< 500 KB` with full `KerrKernelBank` episodic memory), **Axiom-Zero** continual learning architecture in pure Rust (`crates/axiom_zero` vendored locally) with formal **Lean 4 (0-axiom)** algebraic foundations.

No external ML frameworks, zero heap allocation on the numeric Branch-A / supervised hot paths (and streaming text tokenization), **0 transcendental library calls** (`.sin`, `.cos`, `.tan`, `.exp`, `.ln`, `.log`, `.powf`, `.powi`, `.tanh`, `.atan2` count = `0` across core modules; only algebraic `.sqrt` where needed for $L_2$/RMS normalization and algebraic squash), and **44 `#[axiom_zero]`-enforced** blocks/functions across `{core, math, belief, memory, rtrl}.rs` guaranteeing bounded, NaN-free execution under simulated Single-Event Upsets (SEU).

---

## 1. Unified 3-Band Architecture (`src/core.rs`)

`MicroContinualCore` (`src/core.rs`) unifies all working bands inside each `PageHinkleyDetector`-gated `RegimeBank`, enabling continual adaptation across abrupt concept drift, partial observability (POMDP), delayed credit assignment, and non-stationary multi-frequency streams:

- **Band 0 — Frozen `SO(3)` Spinor Rotor + `KerrKernelBank` (`half_sectors > 0`)**
  - Fixed orthogonal `SpinorEncoder` + `CayleyBeliefState` + `KerrKernelBank` associative memory + eligibility-traced linear / actor-critic readout.
  - Zero-byte lazy allocation when `half_sectors == 0`.
- **Band 1 — Plastic Fast/Mid 3D `Cayley-RTRL` (`rtrl_sectors > 0`)**
  - Online $O(S)$ Real-Time Recurrent Learning (`PlasticCayleyParams`, `PlasticCayleyState` in `src/rtrl.rs`) updating decay $\alpha_s$, frequency $\omega_s$, and input weights $w_{s,j}$ on exact Cayley $SO(3)$ rotors with rational normalization $\mathrm{Norm}_\kappa(u)$ (`rtrl_kappa`) and algebraic gradient RMS.
  - Supports coupled grids (`rtrl_grid_mode = 0`) and decoupled frequency $\times$ quality-factor $(Q \times \omega)$ filter grids (`rtrl_grid_mode = 1`).
- **Band 2 — Hierarchical Slow Envelope `Cayley-RTRL` (`rtrl_slow_sectors > 0`) + Skip Channel (`rtrl_use_skip_x`)**
  - Slow envelope $SO(3)$ oscillators receiving `[raw_x, pooled_fast(Band 1)]` via orthogonal 4-channel pooling (`rtrl_pool_w`), plus an optional direct skip channel `0.5 * raw_x` (`rtrl_use_skip_x = true`).
  - Both Band 1, Band 2, and skip actor/critic weights reside *inside* each `RegimeBank`, so `PageHinkleyDetector` preserves slow and fast skills across regime switches (`A -> B -> A`).
  - When disabled (`rtrl_slow_sectors == 0`), slow weights allocate `0` bytes (`test_unified_2l_microcore_architecture_ram_and_seu`).
- **Unified Supervised & RL Paths**
  - `end_supervised_step` trains Band 0 linear heads and Band 1/2 `Cayley-RTRL` parameters + readout jointly online in 1-pass test-then-train streams.
  - `end_episode` / `observe_step_reward` train actor-critic / eligibility-traced policies in POMDP and continual RL tasks.

---

## 2. Lean 4 Formal Verification (0 Axioms)

File: [`proofs/MicroContinualCoreProofs.lean`](proofs/MicroContinualCoreProofs.lean) (cleanly extracted constructive core, free of unrelated non-ML experiments).

Checked cleanly with Lean 4 (no `Mathlib`, no `Real`):

```powershell
lean proofs\MicroContinualCoreProofs.lean
```

**Observed result:** exit code `0`; all **`44`** `#print axioms` checks report `does not depend on any axioms` (`NonZeroAxiom: 0`).

| Step | Theorem(s) | Algebraic Result Verified with 0 Axioms |
|---|---|---|
| **103** | `step103_dft4_orthogonality` | 4-point DFT wave orthogonality over Gaussian `Balance` ($\mathbb{Z}[i]$) |
| **104** | `step104_dft4_inversion`, `step104b_dft3_inversion` | Exact 4-point (`GaussBalance`) and 3-phase (`EisBalance`, $1+\omega+\omega^2 \sim 0$) DFT inversion |
| **105** | `step105_dft4_parseval` | Exact Parseval energy conservation across 4-point DFT modes |
| **106** | `step106_dft4_shift_is_phase_rotation` | Cyclic time shift equals per-mode phase rotation $i^k$ |
| **107** | `step107_rational_circle_cayley` | $1 + \omega^2 > 0$ (no division by zero) and exact rational circle $(1-\omega^2)^2 + (2\omega)^2 \sim (1+\omega^2)^2$ |
| **108** | `step108_su2_lie_laws` | $\mathfrak{su}(2)$ cross-product laws ($a \times a \sim 0$, antisymmetry, Jacobi identity) over `Vec3` |
| **109** | `step109_cayley3d_exact_orthogonality`, `step109b_cayley_rot3_equiv` | Exact integer-form norm preservation of 3D Cayley rotation and equivalence to unit-axis `cayley_rot3` |
| **110** | `step110_so3_noncommutativity` | Strict non-commutativity of orthogonal 3D Cayley rotors ($R_x R_y v \not\sim R_y R_x v$) |
| **111** | `step111_rtrl_decay_strict_contraction` | Strict BIBO contraction $199 > 0$, $200(1+\alpha^2) > 0$, and $199 < 200(1+\alpha^2)$, proving $0 < \frac{0.995}{1+\alpha^2} < 1$ (matches `lambda = 0.995 / (1 + alpha^2)` in `src/rtrl.rs`) |
| **112** | `step112_rational_softmax_positive` | Strict positivity of rational quadratic softmax numerator/denominator over `Balance` |
| **113** | `step113_kerr_w1_total_denom` | Total non-vanishing denominator for `KerrKernelBank` similarity weighting |
| **114** | `step114_rtrl_exact_derivatives_dual` | Exact Cayley-RTRL parameter derivatives via dual numbers ($\varepsilon^2 \sim 0$) |

---

## 3. Repository Layout

```text
micro_continual_core_v030/
├── Cargo.toml                         # Self-contained manifest using vendored crates/axiom_zero
├── CITATION.cff                       # Authors metadata + ORCIDs (0009-0002-4761-1952, 0009-0006-5368-062X)
├── LICENSE                            # Complete PolyForm-Noncommercial-1.0.0 text + Required Notice
├── PAPER.md                           # Full scientific paper (bilingual abstract, derivations, results)
├── README.md                          # Overview, results, and reproduction instructions
├── crates/
│   └── axiom_zero/                    # Vendored #[axiom_zero] macro + totality runtime
├── licenses/
│   └── LICENSE-AGPL-3.0-only.txt      # Optional OSI copyleft license text
├── proofs/
│   └── MicroContinualCoreProofs.lean  # Clean 0-axiom Lean 4 formalization (Steps 1–114)
└── src/
    ├── lib.rs                         # Public API + 10 release unit tests
    ├── core.rs                        # Unified 3-band MicroContinualCore + RegimeBank
    ├── math.rs                        # SpinorEncoder, algebraic squash/log, rational softmax
    ├── belief.rs                      # CayleyBeliefState, EligibilityTraceRing, PageHinkleyDetector
    ├── memory.rs                      # KerrKernelBank associative memory
    ├── rtrl.rs                        # PlasticCayleyParams, PlasticCayleyState, HierarchicalCayleyRtrl
    ├── env_hard_pomdp.rs              # HardPOMDP v2 strict honest benchmark environment
    ├── baselines.rs                   # RL baselines (Online GRU+A2C, RTRRL-LRU, MFEC+PH, Tabular+PH)
    ├── stream_baselines.rs            # Stream baselines (Online MLP+Adam, Gaussian NB, Binned+PH, Persist)
    └── bin/
        ├── bench_unified_ablation.rs  # [A0]..[A4] progressive ablation across all 5 suites
        ├── bench_step1.rs             # HardPOMDP v2 (Strict Honest dense/sparse 15%/25%)
        ├── bench_step2_streams.rs     # SEA, STAGGER, Hyperplane 10D, RBF 4-cls, ELEC2 (45,312 rows)
        ├── bench_step3_pomdp_rl.rs    # T-Maze L=5/15, RepeatPrevious K=2/4, CartPole 2D/+dx
        └── bench_step4_waves.rs       # T-Maze L=30/50, RepeatPrevious K=6/8, Wave-Superposition Zero-Shot
```

---

## 4. Verified Experimental Results (`--release`)

Unit tests: **`cargo test --release` — 10 passed, 0 failed**.

### A. Unified Progressive Ablation (`bench_unified_ablation`)

`[A0] Frozen Rotor Only (no RTRL)` $\to$ `[A1] +1L RTRL Frozen (rtrl_lr=0)` $\to$ `[A2] +1L RTRL Trained (coupled)` $\to$ `[A3] +1L RTRL Trained (dec Qxw)` $\to$ **`[A4] Unified 2L Fast-Slow + PH`**.

#### Ablation 1: POPGym `RepeatPrevious` (`4 symbols`, pure RL `+1/-1`, 10 seeds)

| Stage / Architecture | `K=4` Score | `K=6` Score | RAM (`K=4`/`K=6`) | Latency | SEU |
|---|---:|---:|---:|---:|---|
| `[A0]` Frozen Rotor Only (no RTRL) | `58.0 ± 3.9%` | `50.1 ± 7.9%` | `13.3 KB` | `1.56 / 1.66 µs` | YES |
| `[A1]` +1L RTRL Frozen (`rtrl_lr=0`) | `55.4 ± 10.4%` | `49.0 ± 11.1%` | `21.0 KB` | `3.28 / 3.26 µs` | YES |
| `[A2]` +1L RTRL Trained (coupled) | `55.8 ± 8.9%` | `50.8 ± 9.5%` | `21.0 KB` | `3.61 / 3.68 µs` | YES |
| `[A3]` +1L RTRL Trained (`dec Qxw`) | `70.4 ± 10.3%` | `63.5 ± 11.4%` | `21.0 KB` | `3.58 / 3.75 µs` | YES |
| **`[A4]` Unified 2L Fast-Slow + PH** | **`83.8 ± 5.0%`** | **`74.7 ± 5.0%`** | **`18.3 KB`** | `5.49 / 5.70 µs` | **YES** |

#### Ablation 2: Position-Only CartPole `A -> B -> A` Actuator Flip (`steps / 200`, 10 seeds)

| Stage / Architecture | Pure 2D (`Score` / `Retent.` / `Rec.`) | `+dx` (`Score` / `Retent.` / `Rec.`) | RAM (`2D` / `+dx`) | SEU |
|---|---|---|---|---|
| `[A0]` Frozen Rotor Only | `27.4 ± 4.9` / `30.1` / `140.0 ep` | `162.4 ± 18.7` / `148.6` / `25.1 ep` | `74.4 / 74.4 KB` | YES |
| `[A1]` +1L RTRL Frozen | `59.7 ± 10.2` / `65.9` / `77.9 ep` | `173.7 ± 5.1` / `176.4` / `21.1 ep` | `16.8 / 18.3 KB` | YES |
| `[A2]` +1L RTRL Trained (cpl) | `62.9 ± 12.0` / `88.1` / `84.9 ep` | `182.6 ± 3.8` / `183.6` / `21.7 ep` | `16.8 / 18.3 KB` | YES |
| `[A3]` +1L RTRL Trained (`Qxw`) | `51.9 ± 7.1` / `71.0` / `92.0 ep` | `181.5 ± 4.4` / `185.5` / `22.5 ep` | `16.8 / 18.3 KB` | YES |
| **`[A4]` Unified 2L Fast-Slow + PH** | **`101.3 ± 12.6` / `119.8` / `62.6 ep`** | **`184.1 ± 7.1` / `184.2` / `21.5 ep`** | **`18.4 / 19.9 KB`** | **YES** |

#### Ablation 3: Wave-Superposition Zero-Shot Transfer (`Train A+B+C -> Frozen Eval`, 10 seeds)

| Stage / Architecture | `A+B+C` | `ZS:A` | `ZS:B` | `ZS:C` | `ZS:A+B` | Mean ZS | RAM | Lat | SEU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `[A0]` Frozen Rotor Only | `73.1%` | `81.8%` | `52.7%` | `41.6%` | `79.2%` | `63.8 ± 6.1%` | `6.1 KB` | `1.22 µs` | YES |
| `[A1]` +1L RTRL Frozen | `77.9%` | `95.2%` | `46.2%` | `26.7%` | `94.6%` | `65.7 ± 1.9%` | `9.6 KB` | `2.45 µs` | YES |
| `[A2]` +1L RTRL Trained (cpl) | `78.2%` | `95.2%` | `50.8%` | `17.4%` | `95.0%` | `64.6 ± 1.4%` | `9.6 KB` | `2.80 µs` | YES |
| **`[A3]` +1L RTRL Trained (`dec Qxw`)** | `90.6%` | `94.7%` | `90.3%` | `86.4%` | `85.4%` | **`89.2 ± 2.3%`** | **`9.6 KB`** | `2.76 µs` | **YES** |
| `[A4]` Unified 2L Fast-Slow + PH | **`92.9%`** | `89.1%` | `79.3%` | `85.9%` | `82.6%` | `84.2 ± 3.6%` | `11.3 KB` | `3.95 µs` | YES |

#### Ablation 4: Continual Wave Flip (`Normal -> Inverted -> Immediate 12-ep Return`, 10 seeds)

| Architecture | Score | `Ret12ep` | Recovery | RAM | Latency | SEU |
|---|---:|---:|---:|---:|---:|---|
| **Unified 2L MicroCore + PH Banks** | **`88.3 ± 0.3%`** | **`66.9%`** | **`22.7 ep`** | **`16.9 KB`** | `3.97 µs` | **YES** |
| Unified 2L (no PH, single bank) | `88.2 ± 0.3%` | `21.7%` | `32.8 ep` | `11.3 KB` | `3.82 µs` | YES |
| Online GRU + A2C | `67.0 ± 1.9%` | `44.1%` | `180.0 ep` | `39.4 KB` | `8.03 µs` | NO (`NaN`) |

#### Ablation 5: Electricity `ELEC2` Real Stream Temporal (`45,312` rows, 1-pass Test-then-Train, without `y_{t-1}`, 3 seeds)

| Stage | Acc (`mean±sd`) | `KappaT` ($\kappa_T$) | RAM | Latency | SEU |
|---|---:|---:|---:|---:|---|
| `[A0]` Frozen Rotor Only (no RTRL) | `86.2 ± 0.0%` | `+0.059` | `9.5 KB` | `1.17 µs` | YES |
| `[A1]` +1L RTRL Frozen (`rtrl_lr=0`) | `85.3 ± 0.2%` | `-0.003` | `16.3 KB` | `2.80 µs` | YES |
| `[A2]` +1L RTRL Trained (coupled) | `88.1 ± 0.5%` | `+0.189` | `16.3 KB` | `3.62 µs` | YES |
| `[A3]` +1L RTRL Trained (`dec Qxw`) | `88.3 ± 0.6%` | `+0.204` | `16.3 KB` | `3.56 µs` | YES |
| **`[A4]` Unified 2L Fast-Slow + PH** | **`90.0 ± 0.1%`** | **`+0.322`** | **`27.0 KB`** | `6.20 µs` | **YES** |

> *Causality & variance note:* Each row `t` runs strictly test-then-train (`decide_vec(&x_t)` using only the 8 numeric feature columns, never `class` $\to$ score `pred == y_t` $\to$ `end_supervised_step(pred, y_t)`). Across the same 3 seeds (`1..=3`), `[A4]` reaches `90.0 ± 0.1%` ($\kappa_T = +0.322$) in `bench_unified_ablation` vs. `89.2 ± 0.3%` ($\kappa_T = +0.263$) in `bench_step2_streams` solely due to `linear_lr = 0.07` (`bench_unified_ablation`) vs. `0.06` (`bench_step2_streams`).

### B. Full Benchmark Suites & Head-to-Head vs Baselines (`bench_step1` .. `bench_step4_waves`)

#### Head-to-Head vs Baselines (`--release`, 10 seeds; `ELEC2`: 3 seeds)

| Task / Suite | `MicroCore / 2L Cayley` | `RTRRL-LRU` (AAAI'25) | `Online GRU + A2C` | Other Baselines / Floor |
|---|---|---|---|---|
| `T-Maze L=5` (noise ±0.06) | **`100.0 ± 0.0%`** (`51.2 KB`) | `56.6 ± 6.4%` | `74.2 ± 7.8%` | `50.0%` Random |
| `T-Maze L=15` (noise ±0.35) | **`98.8 ± 1.2%`** (`51.2 KB`) | `51.1 ± 5.8%` | `47.8 ± 4.3%` | `50.0%` Random |
| `T-Maze L=30` (noise ±0.35) | **`79.1 ± 6.6%`** (`79.9 KB`) | `48.9 ± 5.6%` | `52.1 ± 3.7%` | `50.0%` Random |
| `T-Maze L=50` (noise ±0.35) | **`58.4 ± 10.7%`** (`99.1 KB`) | `48.6 ± 4.0%` | `50.5 ± 7.5%` | `50.0%` Random |
| `RepeatPrevious K=2` (4 sym) | **`99.5 ± 0.3%`** (`13.2 KB`) | `90.8 ± 8.0%` (`12.8 KB`) | `51.6 ± 7.7%` | `25.0%` Random |
| `RepeatPrevious K=4` (4 sym) | **`85.7 ± 4.1%`** (`13.2 KB`) | `63.4 ± 6.0%` (`12.8 KB`) | `32.4 ± 4.5%` | `25.0%` Random |
| `RepeatPrevious K=6` (4 sym) | **`74.3 ± 4.8%`** (`13.2 KB`) | `59.3 ± 6.0%` (`12.8 KB`) | `29.6 ± 3.4%` | `25.0%` Random |
| `RepeatPrevious K=8` (4 sym) | **`62.7 ± 4.0%`** (`13.2 KB`) | `44.6 ± 3.6%` (`12.8 KB`) | `25.5 ± 2.3%` | `25.0%` Random |
| `CartPole A->B->A` (pure 2D) | **`62.9 ± 12.0 st`** (`16.8 KB`; `[A4]: 101.3 ± 12.6 st`) | `32.7 ± 2.6 st` (`8.2 KB`) | `54.7 ± 14.0 st` (`40.5 KB`) | `22.8 st` Random |
| `CartPole A->B->A` (`+dx`) | **`182.6 ± 3.8 st`** (`18.3 KB`; `[A4]: 184.1 ± 7.1 st`) | `84.7 ± 16.1 st` (`11.2 KB`) | `95.6 ± 9.1 st` (`42.8 KB`) | `22.8 st` Random |
| `Wave-Superposition ZS` | **`93.0% / 85.0 ± 2.3% ZS`** (`8.2 KB`) | `92.0% / 83.2 ± 2.7% ZS` (`5.8 KB`) | `78.7% / 63.4 ± 2.6% ZS` | `68.1% / 75.0% ZS` `Sign(x_t)` |
| `HardPOMDP v2 Strict` (Dense 15%) | **`63.3 ± 7.7%`** (`454.0 KB`) | `—` | `0.0 ± 0.1%` | `0.6%` `MFEC+PH`, `0.0%` `Tab+PH` |
| `HardPOMDP v2 Strict` (Sparse 15%) | **`62.0 ± 5.3%`** (`454.0 KB`) | `—` | `0.9 ± 1.1%` | `1.0%` `MFEC+PH`, `0.2%` `Tab+PH` |
| `HardPOMDP v2 Strict` (Sparse 25%) | **`49.6 ± 7.8%`** (`454.0 KB`) | `—` | `0.1 ± 0.2%` | `0.3%` `MFEC+PH`, `0.1%` `Tab+PH` |
| `ELEC2` (`45,312` rows, no `y_{t-1}`) | `89.2 ± 0.3%` ($\kappa_T=+0.263$; **`[A4]: 90.0 ± 0.1%`**, $+0.322$) | **`89.3 ± 0.1%`** ($\kappa_T=+0.271$, `Online LRU-RTRL`) | `—` | `87.2%` `MLP+Adam`, `85.3%` `No-Change` |
| `ELEC2` (`45,312` rows, `+y_{t-1}`) | `90.9 ± 0.1%` ($\kappa_T=+0.382$, Tail20 `92.6%`) | **`93.3 ± 0.0%`** ($\kappa_T=+0.544$, `Online LRU-RTRL`) | `—` | `91.0%` `MLP+Adam`, `85.3%` `No-Change` |

> *Fairness & SEU notes:* `MicroCore / 2L Cayley` is **`SEU-Safe = YES`** on every suite; `RTRRL-LRU`, `Online LRU-RTRL`, `Online GRU + A2C`, and `Online MLP + Adam` are `NO (NaN)` under `NaN`/`±1e30` input probes. `RTRRL-LRU` uses decoupled Orvieto ring initialization and matched state dimensionality (`48` complex units = `96D` on `RepeatPrevious` / `CartPole`, `42` units = `84D` on `Wave-Superposition`, `32` units on `ELEC2`); `Online GRU + A2C` updates once per episode via BPTT while `RTRRL-LRU` and `Cayley-RTRL` update online per step; and `MicroContinualCore` additionally uses multi-bank `PageHinkley` switching (plus Cloze/UCB/curiosity on `HardPOMDP v2`).

Full methodology and consolidated tables (including all synthetic streams) are in [`PAPER.md`](PAPER.md) (Section 5.2). Highlights:
- **`bench_step2_streams`:** `SEA 82.6 ± 0.3%` (`κ_T +0.627`), `STAGGER 96.1 ± 0.4%` (`κ_T +0.893`), `Hyperplane 10D 77.3 ± 0.9%` (`κ_T +0.548`), `RBF 4-cls 92.8 ± 0.8%` (`κ_T +0.905`), `ELEC2`: `MicroCore+2L-RTRL(Temp) 89.2 ± 0.3%` (`κ_T +0.263`), `Online LRU-RTRL (Temp) 89.3 ± 0.1%` (`κ_T +0.271`), `MicroCore+RTRL (+y_{t-1}) 90.9 ± 0.1%` (`κ_T +0.382`, `Tail20% 92.6%`), `Online LRU-RTRL (+y_{t-1}) 93.3 ± 0.0%` (`κ_T +0.544`).
- **`bench_step3_pomdp_rl`:** `T-Maze L=5: 100.0%`, `T-Maze L=15: 98.8%`, `RepeatPrevious K=2: 99.5%`, `RepeatPrevious K=4: 85.7%`, `CartPole (+dx): 182.6 ± 3.8 st` (`MicroCore Branch B: 140.9 ± 17.7 st`).
- **`bench_step4_waves`:** `T-Maze L=30 / L=50: 79.1% / 58.4%`, `RepeatPrevious K=6 / K=8: 74.3% / 62.7%`, `Wave-Superposition ZS: 93.0% / 85.0 ± 2.3% ZS` (`8.2 KB`).
- **`bench_step1` (`HardPOMDP v2 Strict Honest`, 3 suites):** Dense `15%`: `63.3 ± 7.7%`; Sparse `15%`: `62.0 ± 5.3%`; Sparse `25%`: `49.6 ± 7.8%` (vs `≤ 1.0%` for `MFEC+PH`, `Tabular+PH`, and `Online GRU+A2C`).

---

## 5. Build & Reproduction Commands

Download `electricity-normalized.arff` from OpenML (or place it at `../micro_continual_core/electricity-normalized.arff`) before running `bench_unified_ablation` or `bench_step2_streams`:

```powershell
# 0. Optional: download the ELEC2 real stream dataset (45,312 rows)
Invoke-WebRequest -Uri "https://www.openml.org/data/download/2419/electricity-normalized.arff" -OutFile "electricity-normalized.arff"

# 1. Unit tests (10 tests)
cargo test --release

# 2. Unified progressive ablation ([A0]..[A4] across all 5 domains)
cargo run --release --bin bench_unified_ablation

# 3. Benchmark suites
cargo run --release --bin bench_step1
cargo run --release --bin bench_step2_streams
cargo run --release --bin bench_step3_pomdp_rl
cargo run --release --bin bench_step4_waves

# 4. Lean 4 formal verification (0 axioms)
lean proofs\MicroContinualCoreProofs.lean
```

---

## 6. Release & Zenodo DOI Workflow

1. [`LICENSE`](LICENSE) contains the full verbatim `PolyForm-Noncommercial-1.0.0` license text and Required Notice; [`CITATION.cff`](CITATION.cff) carries both authors (`Adrian Aleksandrovich İşmuhanov`, ORCID `0009-0002-4761-1952`; `Sofia Sidorenko`, ORCID `0009-0006-5368-062X`), and repository URL (`https://github.com/Worobushka/micro_continual_core`).
2. Push this standalone directory to `https://github.com/Worobushka/micro_continual_core` (commit cleanly; **no `git push` has been performed automatically**).
3. Link `Worobushka/micro_continual_core` in your Zenodo account (`https://zenodo.org`), publish release tag `v0.3.0` on GitHub, and add `doi: "10.5281/zenodo.<id>"` to [`CITATION.cff`](CITATION.cff) in a follow-up commit once Zenodo mints the DOI (and optionally upload a PDF export of [`PAPER.md`](PAPER.md) to the Zenodo record).

---

## 7. Licensing & Commercial Use

- **Source-Available Noncommercial / Academic Research:** `MicroContinualCore` and `proofs/MicroContinualCoreProofs.lean` are licensed under **`PolyForm-Noncommercial-1.0.0`** ([`LICENSE`](LICENSE)). Optional **`GNU AGPL-3.0-only`** text is preserved in [`licenses/LICENSE-AGPL-3.0-only.txt`](licenses/LICENSE-AGPL-3.0-only.txt).
- **Commercial Licensing:** Commercial integration, internal enterprise deployment, or proprietary licensing inquiries must obtain a separate Commercial License directly from Adrian Aleksandrovich İşmuhanov ([mrwerdin@gmail.com](mailto:mrwerdin@gmail.com), [ORCID `0009-0002-4761-1952`](https://orcid.org/0009-0002-4761-1952)).
