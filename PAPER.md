# MicroContinualCore: A Unified Cache-Resident Axiom-Zero Architecture with Multi-Timescale SO(3) Cayley-RTRL Rotors and 0-Axiom Lean 4 Foundations

**Authors:** Adrian Aleksandrovich İşmuhanov ([ORCID `0009-0002-4761-1952`](https://orcid.org/0009-0002-4761-1952)), Sofia Sidorenko ([ORCID `0009-0006-5368-062X`](https://orcid.org/0009-0006-5368-062X), [GitHub `@serjliasereev-KasaneTeto`](https://github.com/serjliasereev-KasaneTeto))  
**Repository:** [https://github.com/Worobushka/micro_continual_core](https://github.com/Worobushka/micro_continual_core)  
**Version:** 0.3.0 (`2026-11-21`)  
**DOI (Zenodo v0.3.0):** [`10.5281/zenodo.24424983`](https://doi.org/10.5281/zenodo.24424983) (Concept DOI: [`10.5281/zenodo.24424982`](https://doi.org/10.5281/zenodo.24424982))  
**License:** PolyForm Noncommercial License 1.0.0 ([`LICENSE`](LICENSE))  

---

## Abstract

We introduce **MicroContinualCore (`v0.3.0`)**, a unified, cache-resident continual learning and control architecture in pure Rust (`8–34 KB` RAM in streaming/RTRL configurations; `< 500 KB` with associative episodic kernel memory) backed by a formal **0-axiom** constructive verification suite in **Lean 4** without `Mathlib` or axiomatic real numbers $\mathbb{R}$. Continual learning on edge, aerospace, and embedded hardware simultaneously demands immunity to catastrophic forgetting under abrupt regime shifts (`A -> B -> A`), tolerance to Single-Event Upsets (SEU) causing `NaN`/`±Inf` bit flips, zero heap allocation in hot loops, and expressive multi-timescale temporal credit assignment without Backpropagation Through Time (BPTT). Standard recurrent neural networks (`GRU+A2C`) and complex linear recurrent units (`RTRRL-LRU`) rely on unbounded transcendental primitives (`exp`, `tanh`, `sin`, `cos`) that fail SEU bit-flip safety and struggle on deep delays or continual regime reversals.

MicroContinualCore resolves these bottlenecks by uniting three complementary algebraic processing bands inside each `PageHinkleyDetector`-gated `RegimeBank`:
1. **Band 0 (Frozen Structural Memory):** orthogonal `SO(3)` spinor encoding (`SpinorEncoder`), non-commutative Lie-bracket belief updates (`CayleyBeliefState`), and quantized `i8` kernel memory (`KerrKernelBank`) with algebraic W1-Kerr competition (`p = 4`).
2. **Band 1 (Plastic Fast/Mid `Cayley-RTRL`):** exact $O(S)$ online Real-Time Recurrent Learning over rational 3D Cayley $SO(3)$ rotors with learnable decays $\alpha_s$, angular frequencies $\omega_s$, and input couplings $w_{s,j}$, supporting decoupled $(Q \times \omega)$ filter grids (`rtrl_grid_mode = 1`) and rational normalization $\mathrm{Norm}_\kappa(u)$.
3. **Band 2 (Hierarchical Slow Envelope `Cayley-RTRL` & Skip Channel):** slow high-$Q$ $SO(3)$ oscillators driven by raw inputs and orthogonal 4-channel pooled Band-1 features, plus an optional skip channel, stored **inside** each regime bank.

We replace all `exp`/`softmax`/`tanh`/`sin`/`cos`/`ln` calls in core modules with strictly algebraic rational equivalents (`0` transcendental calls across all 5 core modules; `44` compile-time `#[axiom_zero]`-validated blocks/functions). In Lean 4 (`proofs/MicroContinualCoreProofs.lean`), `44` exported theorems—covering 4-point and 3-phase algebraic DFT, rational Cayley circles, $\mathfrak{su}(2)$ Lie identities, exact 3D Cayley orthogonality and non-commutativity, strict RTRL BIBO contraction, rational quadratic softmax positivity, non-vanishing Kerr denominators, and exact dual-number RTRL derivatives—pass with `does not depend on any axioms`. Empirically across progressive ablations (`[A0]..[A4]`) and 5 benchmark suites (`--release`), the unified architecture reaches **`83.8 ± 5.0%`** on POPGym `RepeatPrevious (K=4)` and **`74.7 ± 5.0%`** at `K=6`, **`101.3 ± 12.6`** steps on pure-2D Position-Only CartPole `A -> B -> A` (`184.1 ± 7.1` with `+dx`), **`88.3 ± 0.3%`** (`66.9%` immediate 12-ep retention) on Continual Wave Flip, **`90.0 ± 0.1%`** ($\kappa_T = +0.322$) on 45,312-row `ELEC2 Temporal` without `y_{t-1}` leakage, and **`63.3 ± 7.7%`** on `HardPOMDP v2 Strict Honest` where `GRU+A2C` and `MFEC+PH` stay below `1%`.

---

## Аннотация (Russian Summary)

Представлена единая кэш-резидентная архитектура непрерывного обучения **MicroContinualCore (`v0.3.0`)** (`8–34 КБ` RAM в потоковых и RTRL-конфигурациях, `< 500 КБ` с полной эпизодической памятью) на чистом Rust с формальной **0-аксиомной верификацией на Lean 4** (без `Mathlib` и без вещественных чисел $\mathbb{R}$). Внутри каждого банка режимов `RegimeBank`, переключаемого детектором `PageHinkleyDetector`, объединены три алгебраические полосы: замороженный спинорный ротор и кернел-память Керра (`Полоса 0`), пластичный быстрый/средний `O(S)` онлайн-RTRL на рациональных 3D роторах Кэли $SO(3)$ с развязанной частотно-добротной сеткой `(Q × ω)` (`Полоса 1`) и медленная иерархическая огибающая Кэли поверх 4-канального пулинга Полосы 1 (`Полоса 2`) + skip-канал. Ядро полностью свободно от трансцендентных вызовов (`exp`/`ln`/`sin`/`cos`/`tanh` = `0`), защищено compile-time валидатором `#[axiom_zero]` (`44` аннотации), устойчиво к сбоям SEU (`NaN`/`±Inf`) и подтверждено `44` проверками `#print axioms` в Lean 4 с нулём внешних аксиом. На объединённых абляциях `[A0]..[A4]` и 5 наборах бенчмарков показано превосходство единой архитектуры над `Online GRU + A2C`, `RTRRL-LRU` и потоковыми базовыми методами.

---

## 1. Introduction & Motivation

Continual learning agents deployed on resource-constrained microcontrollers, aerospace flight computers, or radiation-exposed embedded controllers face four coupled constraints that standard deep sequence models fail simultaneously:

1. **Tight On-Chip Memory Budgets:** Many embedded targets (e.g., microcontroller L1/SRAM banks) cannot afford multi-megabyte replay buffers, quadratic attention matrices, or large Adam optimizer states.
2. **Non-Stationarity Without Catastrophic Forgetting:** Under cyclical regime switches ($A \to B \to C \to A$), single-bank online gradient descent rapidly overwrites prior dynamical attractors unless slow and fast recurrent parameters are encapsulated per regime and gated against spurious drift.
3. **Radiation / Single-Event Upset (SEU) Safety:** Standard floating-point `GRU` and exponential recurrent units amplify bit-flips (`NaN`, `±Inf`, or $10^{28}$ spikes) through unguarded divisions, exponential gates, or panicking indexers.
4. **Temporal Delays and Partially Observable Structure:** Memoryless classifiers and tabular `Page-Hinkley` detectors fail whenever observations require retaining hidden sub-identities across long corridors (`HardPOMDP v2`, `T-Maze L=30, 50`) or superposing multi-frequency wave harmonics (`Wave-Superposition Zero-Shot`, `ELEC2 Temporal`).

MicroContinualCore addresses these constraints from first algebraic principles: every recurrent state transition is a provably contractive rational operator on $SO(3)$, every similarity weighting and probabilistic policy head avoids `exp`/`ln` and division by zero by construction, and all 0-axiom algebraic invariants are machine-checked in Lean 4 alongside compile-time totality enforcement via `#[axiom_zero]`.

---

## 2. Constructive 0-Axiom Foundations in Lean 4

Formal artifact: [`proofs/MicroContinualCoreProofs.lean`](proofs/MicroContinualCoreProofs.lean).

### 2.1 Zero-Free Constructive Arithmetic Without `Mathlib` or `Real`

Rather than importing classical set-theoretic axioms or axiomatic reals (`propext`, `Classical.choice`, `Quot.sound`), the proof suite builds a self-contained constructive tower over the equality/no-self-negation foundation:
- **Positive Naturals (`PosNat`, Stages I–II):** generated purely by `one` and `succ` so zero is structurally absent from denominators.
- **Equilibrium Integers (`Balance`, Stage V):** directed pairs `(fwd, bwd : PosNat)` with equivalence `BalanceEquiv x y : x.fwd + y.bwd = x.bwd + y.fwd`, where `bzero = ⟨one, one⟩` models dynamic equilibrium rather than emptiness.
- **Complex (`GaussBalance` $\mathbb{Z}[i]$) & 3-Phase (`EisBalance` $\mathbb{Z}[\omega]$) Balances (Stage X & Block I):** Gaussian pairs verifying `step41_i_squared_is_neg_one` ($i^2 \sim -1$) and Eisenstein triples/pairs modeling $1 + \omega + \omega^2 \sim 0$ (`step42_triphase_neutrality`, `step43_triphase_pair_not_neutral`).
- **Dual Balances (`DualBalance`, Stage XVI):** algebraic infinitesimals `a + b * ε` with $\varepsilon \not\sim 0$ and $\varepsilon^2 \sim 0$ (`step62_dual_eps_ne_zero`, `step62_dual_eps_squared_is_zero`) enabling limit-free exact differentiation (`step63_dual_exact_derivative`).
- **0-Axiom Polynomial Reflection Solver (`BExpr`):** a verified multivariate ring normalizer over `Balance` allowing arbitrary-variable polynomial identities (`bal_poly_eq`) to be discharged by kernel reduction (`rfl`) with zero external axioms.

### 2.2 Formal Theorem Map (Steps 103–114 & Prerequisites)

Executing `lean proofs\MicroContinualCoreProofs.lean` checks **44 `#print axioms` statements** and exits with code `0` (`NonZeroAxiom: 0`). Table 1 maps the 12 core continual-learning & signal-processing theorem blocks (`step103`–`step114`) directly to Rust runtime components:

| Lean Block / Step | Theorem Name(s) | Exact Algebraic Property Proven (0 Axioms) | Runtime Component |
|---|---|---|---|
| **Block I (DFT-4)** | `step103_dft4_orthogonality` | $\sum_{n=0}^3 i^{n(k-m)} \sim 4\,\delta_{km}$ over `GaussBalance` | Orthogonal phase banks |
| **Block I (Inversion)** | `step104_dft4_inversion`, `step104b_dft3_inversion` | Exact 4-point $\mathrm{IDFT}_4(\mathrm{DFT}_4(x)) \sim 4x$ and 3-phase $\mathrm{IDFT}_3(\mathrm{DFT}_3(x)) \sim 3x$ without loss | Multi-frequency wave basis |
| **Block I (Energy)** | `step105_dft4_parseval` | $\sum_k |X_k|^2 \sim 4\sum_n |x_n|^2$ (energy conservation) | Unitary spectral norms |
| **Block I (Shift)** | `step106_dft4_shift_is_phase_rotation` | Cyclic shift $(x_3,x_0,x_1,x_2)$ equals $k$-th mode phase multiplication by $i^k$ | Delay-line `RepeatPrevious` |
| **Block I (Circle)** | `step107_rational_circle_cayley` | $\mathrm{BalGtZero}(1 + \omega^2)$ and $(1-\omega^2)^2 + (2\omega)^2 \sim (1+\omega^2)^2$ | `rational_circle_wave` (`src/math.rs`) |
| **Block II (Lie $\mathfrak{su}(2)$)** | `step108_su2_lie_laws` | $a \times a \sim 0$, $a \times b \sim -(b \times a)$, and Jacobi identity over `Vec3` | `su2_lie_bracket` (`src/math.rs`) |
| **Block II (3D Cayley)** | `step109_cayley3d_exact_orthogonality`, `step109b_cayley_rot3_equiv` | Strict positivity of $1 + \|g\|^2$, exact norm preservation $\|w\|^2 \sim (1+\|g\|^2)^2 \|v\|^2$, and equivalence to unit-axis `cayley_rot3` | `cayley_rot3` (`src/rtrl.rs`), `CayleyBeliefState` |
| **Block II (Non-Comm.)** | `step110_so3_noncommutativity` | Counterexample $R_x(R_y(e_x)) \not\sim R_y(R_x(e_x))$ proving order sensitivity | Non-abelian temporal state (`T-Maze` / `HardPOMDP`) |
| **Block II (BIBO)** | `step111_rtrl_decay_strict_contraction` | $\mathrm{BalGtZero}(1+\alpha^2)$ and strict decay inequality $199 < 200(1+\alpha^2)$ matching rational contraction factor $0.995 / (1+\alpha^2) < 1$ | `lambda = 0.995 / (1 + alpha^2)` (`src/rtrl.rs`) |
| **Block III (Softmax)** | `step112_rational_softmax_positive` | Strictly positive numerators & denominators in rational quadratic policy weights | `rational_probs_and_psi_into` (`src/math.rs`) |
| **Block III (Kerr $W_1$)** | `step113_kerr_w1_total_denom` | Total strictly positive denominator $\varepsilon + \sum_i g_i^2 \ge \varepsilon > 0$ (extended to quartic $\varepsilon + \sum_i g_i^4$ in `KerrKernelBank`) | `kerr_competition_into`, `KerrKernelBank` |
| **Block III (Dual RTRL)** | `step114_rtrl_exact_derivatives_dual` | Exact rational derivatives of Cayley rotation and contraction factors in `DualBalance` | Exact $O(S)$ RTRL sensitivities (`src/rtrl.rs`) |

---

## 3. Unified 3-Band Architecture (`src/core.rs`, `src/rtrl.rs`)

`MicroContinualCore` (configured by `MicroCoreConfig`) eliminates fragmented specialized binaries by hosting all three bands inside each `RegimeBank`, gated by `PageHinkleyDetector`. Every unused band allocates **0 bytes**.

```text
x_t (text or f32 slice) ──► [Standardize / Delta x_t] ──► SEU Sanitization (sanitize_bounded_f32)
   │
   ├──► [Band 0: half_sectors > 0] SpinorEncoder ─► CayleyBeliefState ─► KerrKernelBank + Linear/AC
   │
   ├──► [Band 1: rtrl_sectors > 0] PlasticCayleyState (Fast/Mid SO(3)^S, O(S) RTRL sensitivities)
   │          │
   │          └─► 4-ch Orthogonal Pooling rtrl_pool_w ──┐
   │                                                    ▼
   └──► [Band 2: rtrl_slow_sectors > 0] PlasticCayleyState (Slow Envelope SO(3)^S2 on [x_t, pool])
   │
   └──► [Skip: rtrl_use_skip_x] 0.5 * x_t ──────────► Joint Per-Bank Readout + PageHinkleyDetector
```

### 3.1 Band 0: Frozen Spinor Rotor & `KerrKernelBank` (`src/math.rs`, `src/belief.rs`, `src/memory.rs`)

For textual and symbolic POMDPs, `SpinorEncoder` maps hashed subword trigrams and parsed numeric tokens onto rational circular phases `(s2, c2)` generated via `rational_circle_wave` (double Cayley map with $s_2^2 + c_2^2 = 1$). Unlabeled noisy domain logs (`unlabeled_cloze_adapt`) can optionally align holdout synonyms offline via normalized Hebbian projection before freezing integer-quantized `i8` tables (`freeze_to_i8()`).

Given normalized observation encoding $x_t$, `CayleyBeliefState::update` measures cosine familiarity against an EMA prototype `obs_ema`, updates $u_{\text{rot}} = R(g_{\text{gauge}})\,b_{t-1}$ via `cayley_rotate_sectors`, and computes the $\mathfrak{su}(2)$ Lie bracket $c_t = \widehat{b_{t-1} \times x_t}$ (`su2_lie_bracket`):
$$b_t = \mathrm{Norm}_1\!\Big(\lambda\, R(g_{\text{gauge}})\,b_{t-1} + w_x\, x_t + \gamma\,\nu_t\,\widehat{b_{t-1} \times x_t}\Big)$$
where $\nu_t \in [0, 1]$ is observation novelty (gated by `novelty_gate` so distractor glitches with $\nu_t < \nu_{\text{gate}}$ do not overwrite memory) and $\lambda^2 + w_x^2 = 1$ when gating is off. Associative `KerrKernelBank` stores `i8`-quantized episodic keys alongside both running values `q_vals` and optimistic maxima `q_max` (detecting perceptual aliasing), weighting neighbors through the quartic W1 Kerr law `kerr_competition_into` with `p = 4`:
$$w_k = \frac{\max(0, s_k - \tau)^4}{\varepsilon + \sum_j \max(0, s_j - \tau)^4}.$$

### 3.2 Band 1: Exact $O(S)$ Plastic `Cayley-RTRL` (`src/rtrl.rs`)

Each sector $s \in \{0, \dots, S-1\}$ maintains a 3D recurrent state $u_{t,s} \in \mathbb{R}^3$, a unit rotation axis $e_s$, an orthogonal input axis $d_s \perp e_s$ (ensuring input injections are actively rotated by $e_s$), learnable damping parameter $\alpha_s \in [0.08, 3.0]$, frequency $\omega_s \in [-3.5, 3.5]$, and input weights $w_{s,j} \in [-3.0, 3.0]$:
$$\lambda_s = \frac{0.995}{1 + \alpha_s^2} \in (0, 0.995], \qquad R(\omega_s e_s) v = v + \frac{2\omega_s}{1+\omega_s^2}(e_s \times v) + \frac{2\omega_s^2}{1+\omega_s^2}\big(e_s \times (e_s \times v)\big).$$
Letting $\mathrm{drive}_{t,s} = \sum_j w_{s,j}\, x_{t,j}$, the recurrent transition and rational readout normalization $\mathrm{Norm}_\kappa$ (`kappa = rtrl_kappa`) are:
$$u_{t,s} = \lambda_s R(\omega_s e_s) u_{t-1,s} + \mathrm{drive}_{t,s}\, d_s, \qquad b_{t,s} = \frac{u_{t,s}}{\rho_{t,s}}, \qquad \rho_{t,s} = \sqrt{\|u_{t,s}\|^2 + \kappa^2}.$$
Using the dual-number calculus verified in `step114_rtrl_exact_derivatives_dual`, forward sensitivities $p_{\alpha,s} = \partial u_{t,s}/\partial \alpha_s$, $p_{\omega,s} = \partial u_{t,s}/\partial \omega_s$, and $p_{w,s,j} = \partial u_{t,s}/\partial w_{s,j}$ evolve online in exact $O(S \cdot d_{\text{in}})$ time per step (zero BPTT truncation):
$$p_{\alpha,s}^{(t)} = \lambda_s R(\omega_s e_s) p_{\alpha,s}^{(t-1)} + \frac{-1.99\,\alpha_s}{(1+\alpha_s^2)^2} R(\omega_s e_s) u_{t-1,s},$$
$$p_{\omega,s}^{(t)} = \lambda_s \Big(R(\omega_s e_s) p_{\omega,s}^{(t-1)} + \frac{2(1-\omega_s^2)}{(1+\omega_s^2)^2}(e_s \times u_{t-1,s}) + \frac{4\omega_s}{(1+\omega_s^2)^2}\big(e_s \times (e_s \times u_{t-1,s})\big)\Big),$$
$$p_{w,s,j}^{(t)} = \lambda_s R(\omega_s e_s) p_{w,s,j}^{(t-1)} + x_{t,j}\, d_s.$$
At readout, the exact rank-1 Jacobian projection of $\mathrm{Norm}_\kappa(u_{t,s})$ turns upstream readout gradient $g_s$ into tangent force $f_{b,s} = \big(g_s - (g_s \cdot b_{t,s}) b_{t,s}\big) / \rho_{t,s}$, accumulating eligibility traces `(e_alpha, e_omega, e_w)` and applying bounded algebraic gradient-RMS updates (`v <- 0.98 * v + 0.02 * g^2`, step $\propto g / \sqrt{v + 0.0025}$).

Crucially, `PlasticCayleyParams::new_with_grid` supports:
- **`band_mode = 0` (Coupled):** monotonic quadratic damping coupled to frequency, useful when fast differentiation is tied to low-$Q$ response.
- **`band_mode = 1` (Decoupled $Q \times \omega$):** 3 orthogonal quality bands `q_band = s mod 3` crossed with linear frequencies, allowing high-$Q$ slow oscillators and damped fast oscillators across the spectrum.

### 3.3 Band 2: Hierarchical Slow Envelope & Unified `RegimeBank` Switching (`src/core.rs`)

When `rtrl_slow_sectors > 0`, a second `PlasticCayleyParams` bank (`band_mode = 2`, slower frequencies $\omega_0 \in [0.03, 0.65]$ and higher $Q$) operates on `[raw_x, pooled_fast(Band 1)]`, where 4 pooled summary channels come from matrix multiplication by an orthonormalized/scaled random projection `rtrl_pool_w` over `rtrl_state.b` (Section `970..989` of `src/core.rs`). An optional skip branch `rtrl_use_skip_x` appends `0.5 * raw_x`. All fast, slow, and skip actor/critic weights plus `rtrl_params` and `rtrl_slow_params` live **inside each `RegimeBank`** alongside `PageHinkleyDetector` (CUSUM over normalized episodic return $R_e \in [0, 1]$). When a concept drift triggers CUSUM threshold exceedance, `MicroContinualCore` switches or allocates another `RegimeBank`, protecting Band 1/2 parameters of previously solved regimes from catastrophic interference.

Online training unifies both paradigms:
- **RL / POMDP (`end_episode`):** uses rational shift-invariant policy `rational_probs_and_psi_into` with `psi_a = (1 + |z_a|) / (1 + |z_a| + 0.5 * z_a^2)`, Tsallis entropy regularization, TD/return advantage normalization, and eligibility trace updates across Band 0/1/2.
- **1-Pass Supervised Streams (`end_supervised_step`):** updates linear and split-band logits directly toward `±1` targets, backpropagating exact logit residuals into `rtrl_fb_buf` / `rtrl_slow_fb_buf` to drive 1-step online RTRL adaptation—skipping weight writes automatically when `cusum > 0.55 * threshold` to avoid polluting a learned bank right as regime drift begins.

---

## 4. Compile-Time Totality (`#[axiom_zero]`), SEU Safety & Memory Footprint

`MicroContinualCore` depends on a single internal crate, `crates/axiom_zero`, whose procedural macro `#[axiom_zero]` enforces rules `[E001]`–`[E011]` on annotated modules/functions at compile time: forbidding `panic!`, `.unwrap()`, `.expect()`, `None`, `assert!`, terminating calls, raw slice indexing (`arr[i]`), and raw division/modulo (`/`, `%`). All array indexing passes through `total_get_copy`/`total_set` and all divisions use `total_div_f32` or `total_div_unit_shifted_f32` ($n / (1 + d)$, never zero when $d \ge 0$).

Verified static and runtime checks across core modules (`src/{core,math,belief,memory,rtrl}.rs`):
- **Transcendental library calls (`.sin`, `.cos`, `.tan`, `.exp`, `.ln`, `.log`, `.powf`, `.powi`, `.tanh`, `.atan2`):** **`0`**.
- **Functions / `impl` blocks compiled under `#[axiom_zero]`:** **`44`**.
- **Zero-byte disabled bands & zero-heap-alloc hot paths:** disabling Band 0 (`half_sectors == 0` in `SpinorEncoder`, `dim == 0` in `CayleyBeliefState` / `KerrKernelBank`) allocates strictly **`0` bytes**. Streaming text tokenization (`encode_text_into` via stack buffer `[0u8; 64]`, `parse_numeric_bytes`, `token_slot_and_mass_bytes`, `encode_token_bytes_into`), `KerrKernelBank::query_obs_gated_into`, `RegimeBank::clear_all` (`reset_with_grid`), and `end_episode_inner` operate without heap allocations in the numeric Branch-A / supervised hot paths.
- **SEU input-injection resilience:** unit tests and every benchmark (`test_ram_under_500kb_and_seu_safety`, `test_hierarchical_cayley_rtrl_ram_and_seu`, `test_unified_2l_microcore_architecture_ram_and_seu`, including AVX2+FMA / AVX2 `saxpy`/`scale` NaN/Inf parity) inject adversarial corrupted inputs (`f32::NAN`, `1e30`, `-1e30`) into `decide_vec`/`decide_text` and verify finite outputs and finite internal state, whereas `Online GRU + A2C`, `RTRRL-LRU`, `Online LRU-RTRL`, `Online MLP + Adam`, and `Online Gaussian NB` propagate `NaN` (`SEU-Safe = NO (NaN)`). (Hardware TMR against silent weight-RAM bit-flips remains complementary.)
- **Exact memory footprint:** lazy allocation guarantees `0` bytes for disabled bands (`8.2–33.7 KB` across learned wave/stream/RTRL configurations, `6.1 KB` for Band-0 rotor alone; `51.2–99.1 KB` on T-Maze; `454.0 KB` when full `SpinorEncoder` + `KerrKernelBank` episodic capacity is enabled for `HardPOMDP v2`).

---

## 5. Empirical Evaluation & Progressive Ablations

All Rust binaries were compiled and executed with `--release`. Unit tests pass **`10 / 10`** (`cargo test --release`).

### 5.1 Unified Progressive Ablation (`bench_unified_ablation`)

We evaluate five strict ablation stages inside the same `MicroContinualCore` codebase:
- **`[A0]` Frozen Rotor Only (no RTRL):** Band 0 only (`rtrl_sectors = 0`).
- **`[A1]` +1L RTRL Frozen (`rtrl_lr = 0`):** adds untrainable Band 1 Cayley reservoirs.
- **`[A2]` +1L RTRL Trained (coupled):** enables online Band 1 `Cayley-RTRL` (`rtrl_grid_mode = 0`).
- **`[A3]` +1L RTRL Trained (`dec Qxw`):** switches Band 1 to decoupled `(Q × ω)` grid (`rtrl_grid_mode = 1`).
- **`[A4]` Unified 2L Fast-Slow + PH:** adds Band 2 Slow Envelope + orthogonal pooling (and skip channel where beneficial) inside `PageHinkleyDetector` regime banks.

#### Ablation 1: POPGym `RepeatPrevious` (`4 symbols`, pure RL `+1/-1`, 10 seeds)

| Stage / Architecture | `K=4` Score | `K=6` Score | RAM (`K=4`/`K=6`) | Latency (`K=4`/`K=6`) | SEU |
|---|---:|---:|---:|---:|---|
| `[A0]` Frozen Rotor Only (no RTRL) | `58.0 ± 3.9%` | `50.1 ± 7.9%` | `13.3 KB` | `1.56 / 1.66 µs` | YES |
| `[A1]` +1L RTRL Frozen (`rtrl_lr=0`) | `55.4 ± 10.4%` | `49.0 ± 11.1%` | `21.0 KB` | `3.28 / 3.26 µs` | YES |
| `[A2]` +1L RTRL Trained (coupled) | `55.8 ± 8.9%` | `50.8 ± 9.5%` | `21.0 KB` | `3.61 / 3.68 µs` | YES |
| `[A3]` +1L RTRL Trained (`dec Qxw`) | `70.4 ± 10.3%` | `63.5 ± 11.4%` | `21.0 KB` | `3.58 / 3.75 µs` | YES |
| **`[A4]` Unified 2L Fast-Slow + PH** | **`83.8 ± 5.0%`** | **`74.7 ± 5.0%`** | **`18.3 KB`** | `5.49 / 5.70 µs` | **YES** |

#### Ablation 2: Position-Only CartPole `A -> B -> A` Actuator Flip (`steps / 200`, 10 seeds)

| Stage / Architecture | Pure 2D (`Score` / `Retent.` / `Rec.`) | `+dx` (`Score` / `Retent.` / `Rec.`) | RAM (`2D` / `+dx`) | Latency (`2D` / `+dx`) | SEU |
|---|---|---|---|---|---|
| `[A0]` Frozen Rotor Only | `27.4 ± 4.9` / `30.1` / `140.0 ep` | `162.4 ± 18.7` / `148.6` / `25.1 ep` | `74.4 / 74.4 KB` | `1.28 / 3.56 µs` | YES |
| `[A1]` +1L RTRL Frozen | `59.7 ± 10.2` / `65.9` / `77.9 ep` | `173.7 ± 5.1` / `176.4` / `21.1 ep` | `16.8 / 18.3 KB` | `2.82 / 3.28 µs` | YES |
| `[A2]` +1L RTRL Trained (cpl) | `62.9 ± 12.0` / `88.1` / `84.9 ep` | `182.6 ± 3.8` / `183.6` / `21.7 ep` | `16.8 / 18.3 KB` | `3.08 / 3.69 µs` | YES |
| `[A3]` +1L RTRL Trained (`Qxw`) | `51.9 ± 7.1` / `71.0` / `92.0 ep` | `181.5 ± 4.4` / `185.5` / `22.5 ep` | `16.8 / 18.3 KB` | `3.14 / 3.70 µs` | YES |
| **`[A4]` Unified 2L Fast-Slow + PH** | **`101.3 ± 12.6` / `119.8` / `62.6 ep`** | **`184.1 ± 7.1` / `184.2` / `21.5 ep`** | **`18.4 / 19.9 KB`** | `3.75 / 4.54 µs` | **YES** |

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

**Mechanistic Findings from `[A0]..[A4]`:**
1. *Why `(Q × ω)` + Band 2 matter on delayed/wave tasks:* On `RepeatPrevious K=4,6` and `Wave Zero-Shot`, coupled RTRL (`[A2]`) improves minimally over frozen reservoirs (`55.8%` vs `55.4%`, `64.6%` ZS vs `65.7%`), because high-frequency modes are over-damped. Decoupling frequency and damping (`[A3]`) lifts `RepeatPrevious K=4` to `70.4%` and `Wave Zero-Shot` to `89.2%`. Adding Band 2 Slow Envelope (`[A4]`) pushes `RepeatPrevious K=4` to `83.8%` (`K=6`: `74.7%`, `18.3 KB` RAM) and raises `ELEC2 Temporal` without `y_{t-1}` from `86.2%` (`κ_T=+0.059`) to **`90.0%` (`κ_T=+0.322`)** while keeping `27.0 KB` RAM (`[A4]` unified config; `89.2%`, `κ_T=+0.263` in `bench_step2_streams`).
2. *Why combined Band 0 + Band 1/2 helps pure-2D CartPole:* Retaining Band 0 (`half_sectors = 16`) alongside trained 2L RTRL (`[A4]`) lifts pure-2D CartPole `A->B->A` from `27.4` (`[A0]`) and `62.9` (`[A2]`) to **`101.3 ± 12.6` steps** (`119.8` retention, `62.6 ep` recovery), while matching `+dx` saturating performance at **`184.1 ± 7.1` steps** (`21.5 ep` recovery).
3. *Why `PageHinkleyDetector` encapsulation is essential:* Without per-bank switching (`Ablation 4`), immediate 12-episode retention after returning from Inverted to Normal drops from **`66.9%`** down to `21.7%`.

### 5.2 Non-Stationary POMDP, Deep Delay & Real-Stream Benchmarks

#### A. Non-Stationary Streams (`bench_step2_streams`, 1-pass Test-then-Train)

| Benchmark | Model | Score (`m±sd`) | $\kappa_T$ | Retent. / Tail20% | Recovery | RAM | Latency | SEU |
|---|---|---|---|---|---|---|---|---|
| `SEA Concepts` (10s) | `MicroContinualCore` | `82.6 ± 0.3%` | `+0.627` | `82.8%` | `54.1 st` | `27.3 KB` | `6.16 µs` | YES |
| `SEA Concepts` (10s) | `Online MLP (H=32)+Adam` | `85.9 ± 0.3%` | `+0.699` | `86.4%` | `52.3 st` | `2.3 KB` | `1.10 µs` | NO (`NaN`) |
| `SEA Concepts` (10s) | `Online Gaussian NB` | `84.4 ± 0.3%` | `+0.666` | `86.1%` | `55.3 st` | `0.1 KB` | `0.05 µs` | NO (`NaN`) |
| `SEA Concepts` (10s) | `Tabular Binned + PH` | `62.4 ± 0.7%` | `+0.193` | `64.5%` | `443.6 st` | `112.1 KB` | `0.15 µs` | YES |
| `SEA Concepts` (10s) | `No-Change (Persist)` | `53.4 ± 0.5%` | `+0.000` | `53.4%` | `560.5 st` | `0.0 KB` | `0.00 µs` | YES |
| `STAGGER` (10s) | `MicroContinualCore` | `96.1 ± 0.4%` | `+0.893` | `96.7%` | `60.5 st` | `27.3 KB` | `5.97 µs` | YES |
| `STAGGER` (10s) | `Online MLP (H=32)+Adam` | `97.2 ± 0.3%` | `+0.923` | `96.6%` | `66.5 st` | `4.6 KB` | `1.29 µs` | NO (`NaN`) |
| `STAGGER` (10s) | `Online Gaussian NB` | `84.0 ± 0.6%` | `+0.559` | `80.4%` | `230.8 st` | `0.1 KB` | `0.10 µs` | NO (`NaN`) |
| `STAGGER` (10s) | `Tabular Binned + PH` | `97.6 ± 0.0%` | `+0.935` | `97.8%` | `66.9 st` | `3.4 KB` | `0.11 µs` | YES |
| `STAGGER` (10s) | `No-Change (Persist)` | `63.7 ± 0.9%` | `+0.000` | `65.6%` | `471.3 st` | `0.0 KB` | `0.00 µs` | YES |
| `Rotating Hyperplane 10D` | `MicroContinualCore` | `77.3 ± 0.9%` | `+0.548` | `79.8%` | `53.5 st` | `27.3 KB` | `6.56 µs` | YES |
| `Rotating Hyperplane 10D` | `Online MLP (H=32)+Adam` | `82.0 ± 0.8%` | `+0.641` | `83.8%` | `52.3 st` | `5.0 KB` | `1.44 µs` | NO (`NaN`) |
| `Rotating Hyperplane 10D` | `Online Gaussian NB` | `77.8 ± 1.1%` | `+0.558` | `81.5%` | `55.2 st` | `0.2 KB` | `0.10 µs` | NO (`NaN`) |
| `Rotating Hyperplane 10D` | `Tabular Binned + PH` | `49.9 ± 0.5%` | `+0.003` | `49.8%` | `600.0 st` | `114.8 KB` | `0.23 µs` | YES |
| `Rotating Hyperplane 10D` | `No-Change (Persist)` | `49.8 ± 0.7%` | `+0.000` | `49.1%` | `600.0 st` | `0.0 KB` | `0.00 µs` | YES |
| `Uncorrelated RBF 10D 4-cls` | **`MicroContinualCore`** | **`92.8 ± 0.8%`** | **`+0.905`** | `92.8%` | `58.5 st` | `33.7 KB` | `6.58 µs` | **YES** |
| `Uncorrelated RBF 10D 4-cls` | `Online MLP (H=32)+Adam` | `92.6 ± 0.6%` | `+0.902` | `92.5%` | `56.1 st` | `5.8 KB` | `1.53 µs` | NO (`NaN`) |
| `Uncorrelated RBF 10D 4-cls` | `Online Gaussian NB` | `90.7 ± 0.6%` | `+0.876` | `90.3%` | `123.0 st` | `0.3 KB` | `0.18 µs` | NO (`NaN`) |
| `Uncorrelated RBF 10D 4-cls` | `Tabular Binned + PH` | `25.1 ± 0.4%` | `+0.003` | `25.2%` | `600.0 st` | `249.0 KB` | `0.21 µs` | YES |
| `Uncorrelated RBF 10D 4-cls` | `No-Change (Persist)` | `24.9 ± 0.4%` | `+0.000` | `25.0%` | `600.0 st` | `0.0 KB` | `0.00 µs` | YES |
| `ELEC2` (`45,312` rows, no `y_{t-1}`) | `MicroCore (i.i.d.)` | `85.0 ± 0.0%` | `-0.022` | `87.1%` (Tail20) | `0.0 st` | `27.3 KB` | `5.64 µs` | YES |
| `ELEC2` (`45,312` rows, no `y_{t-1}`) | `MicroCore (Temporal)` | `83.1 ± 0.1%` | `-0.154` | `84.7%` (Tail20) | `0.0 st` | `27.3 KB` | `5.95 µs` | YES |
| `ELEC2` (`45,312` rows, no `y_{t-1}`) | **`MicroCore+2L-RTRL(Temp)`** | **`89.2 ± 0.3%`** | **`+0.263`** | **`91.2%`** (Tail20) | `0.0 st` | `27.0 KB` | `6.34 µs` | **YES** |
| `ELEC2` (`45,312` rows, no `y_{t-1}`) | `Online LRU-RTRL (Temp)` | `89.3 ± 0.1%` | `+0.271` | `90.5%` (Tail20) | `0.0 st` | `9.4 KB` | `4.27 µs` | NO (`NaN`) |
| `ELEC2` (`45,312` rows, no `y_{t-1}`) | `Online MLP (H=32)+Adam` | `87.2 ± 0.3%` | `+0.130` | `88.8%` (Tail20) | `0.0 st` | `4.2 KB` | `1.64 µs` | NO (`NaN`) |
| `ELEC2` (`45,312` rows, `+y_{t-1}`) | `MicroCore (+y_{t-1})` | `87.2 ± 0.0%` | `+0.129` | `87.2%` (Tail20) | `0.0 st` | `27.3 KB` | `6.09 µs` | YES |
| `ELEC2` (`45,312` rows, `+y_{t-1}`) | `MicroCore (Temp+y_{t-1})` | `85.8 ± 0.1%` | `+0.033` | `86.1%` (Tail20) | `0.0 st` | `27.3 KB` | `6.65 µs` | YES |
| `ELEC2` (`45,312` rows, `+y_{t-1}`) | **`MicroCore+RTRL (+y_{t-1})`** | **`90.9 ± 0.1%`** | **`+0.382`** | **`92.6%`** (Tail20) | `0.0 st` | `28.1 KB` | `6.78 µs` | **YES** |
| `ELEC2` (`45,312` rows, `+y_{t-1}`) | `Online LRU-RTRL (+y_{t-1})` | `93.3 ± 0.0%` | `+0.544` | `94.3%` (Tail20) | `0.0 st` | `10.2 KB` | `4.45 µs` | NO (`NaN`) |
| `ELEC2` (`45,312` rows, `+y_{t-1}`) | `Online MLP (+y_{t-1})` | `91.0 ± 0.1%` | `+0.387` | `91.6%` (Tail20) | `0.0 st` | `4.6 KB` | `1.55 µs` | NO (`NaN`) |
| `ELEC2` (`45,312` rows, no `y_{t-1}`) | `Online Gaussian NB` | `71.8 ± 0.0%` | `-0.923` | `71.1%` (Tail20) | `0.0 st` | `0.1 KB` | `0.10 µs` | NO (`NaN`) |
| `ELEC2` (`45,312` rows, no `y_{t-1}`) | `Tabular Binned + PH` | `71.3 ± 0.0%` | `-0.957` | `69.5%` (Tail20) | `0.0 st` | `40.4 KB` | `0.10 µs` | YES |
| `ELEC2` (`45,312` rows) | `No-Change (Persist)` | `85.3 ± 0.0%` | `+0.000` | `84.8%` (Tail20) | `0.0 st` | `0.0 KB` | `0.00 µs` | YES |

On real `ELEC2` without autoregressive label leakage (`y_{t-1}`), non-recurrent `MicroCore (i.i.d.)` performs below `No-Change` (`κ_T = -0.022`), whereas unified online `Cayley-RTRL` learns genuine temporal structure to reach **`89.2%`** (`κ_T = +0.263`, `Tail20% 91.2%`; **`90.0%`, `κ_T = +0.322`** in `[A4]`), outperforming `Online MLP (H=32)+Adam` (`87.2%`) and matching recurrent `Online LRU-RTRL (Temp)` (`89.3%`, `κ_T = +0.271`, `NO (NaN)`) while remaining 100% `#[axiom_zero]` SEU-safe. When lagged label `y_{t-1}` is supplied, `MicroCore+RTRL (+y_{t-1})` (`90.9%`, `Tail20% 92.6%`) matches `Online MLP (+y_{t-1})` (`91.0%`, `Tail20% 91.6%`), while unconstrained complex `Online LRU-RTRL (+y_{t-1})` hits `93.3%` (`κ_T = +0.544`, `NO (NaN)`).

#### B. Partial Observability & Continual RL (`bench_step3_pomdp_rl`, `bench_step4_waves`, `bench_step1`, 10 seeds)

| Suite / Task | MicroCore / 2L Cayley | `RTRRL-LRU` (AAAI'25) | `Online GRU + A2C` | Random / Oracle |
|---|---|---|---|---|
| `T-Maze L=5` (glitch ±0.06) | **`100.0 ± 0.0%`** (`51.2 KB`, `2.58 µs`) | `56.6 ± 6.4%` (`4.9 KB`) | `74.2 ± 7.8%` (`41.7 KB`) | `50.0%` floor |
| `T-Maze L=15` (glitch ±0.35) | **`98.8 ± 1.2%`** (`51.2 KB`, `2.91 µs`) | `51.1 ± 5.8%` (`4.9 KB`) | `47.8 ± 4.3%` (`41.7 KB`) | `50.0%` floor |
| `T-Maze L=30` (glitch ±0.35) | **`79.1 ± 6.6%`** (`79.9 KB`, `6.56 µs`) | `48.9 ± 5.6%` (`4.9 KB`) | `52.1 ± 3.7%` (`41.7 KB`) | `50.0%` floor |
| `T-Maze L=50` (glitch ±0.35) | **`58.4 ± 10.7%`** (`99.1 KB`, `7.72 µs`) | `48.6 ± 4.0%` (`4.9 KB`) | `50.5 ± 7.5%` (`41.7 KB`) | `50.0%` floor |
| `RepeatPrevious K=2` | **`99.5 ± 0.3%`** (`13.2 KB`, `5.06 µs`) | `90.8 ± 8.0%` (`12.8 KB`) | `51.6 ± 7.7%` (`43.6 KB`) | `25.0%` floor |
| `RepeatPrevious K=4` | **`85.7 ± 4.1%`** (`13.2 KB`, `5.14 µs`) | `63.4 ± 6.0%` (`12.8 KB`) | `32.4 ± 4.5%` (`43.6 KB`) | `25.0%` floor |
| `RepeatPrevious K=6` | **`74.3 ± 4.8%`** (`13.2 KB`, `5.35 µs`) | `59.3 ± 6.0%` (`12.8 KB`) | `29.6 ± 3.4%` (`43.6 KB`) | `25.0%` floor |
| `RepeatPrevious K=8` | **`62.7 ± 4.0%`** (`13.2 KB`, `5.17 µs`) | `44.6 ± 3.6%` (`12.8 KB`) | `25.5 ± 2.3%` (`43.6 KB`) | `25.0%` floor |
| `CartPole (pure 2D, A->B->A)` | **`101.3 ± 12.6 st`** (`[A4]`, `18.4 KB`) / `62.9 ± 12.0 st` (`1L`) | `32.7 ± 2.6 st` (`8.2 KB`) | `54.7 ± 14.0 st` (`40.5 KB`) | `22.8 st` floor |
| `CartPole (+dx, A->B->A)` | **`184.1 ± 7.1 st`** (`[A4]`, `19.9 KB`) / **`182.6 ± 3.8 st`** (`1L`) | `84.7 ± 16.1 st` (`11.2 KB`) | `95.6 ± 9.1 st` (`42.8 KB`) | `22.8 st` floor |
| `Wave-Superposition ZS` | **`93.0% / 85.0 ± 2.3% ZS`** (`8.2 KB`, `3.57 µs`) | `92.0% / 83.2 ± 2.7% ZS` (`5.8 KB`) | `78.7% / 63.4 ± 2.6% ZS` (`39.4 KB`) | `Sign(x_t) 75.0% ZS` |
| `HardPOMDP v2 Strict (dense, 15%)` | **`63.3 ± 7.7%`** (`H:15.3%`, `454.0 KB`) | — | `0.0 ± 0.1%` (`114.2 KB`) | `99.2%` ceil / `0.2%` floor |
| `HardPOMDP v2 Strict (sparse, 15%)` | **`62.0 ± 5.3%`** (`H:12.7%`, `454.0 KB`) | — | `0.9 ± 1.1%` (`114.2 KB`) | `99.2%` ceil / `0.2%` floor |
| `HardPOMDP v2 Strict (sparse, 25%)` | **`49.6 ± 7.8%`** (`H:10.2%`, `454.0 KB`) | — | `0.1 ± 0.2%` (`114.2 KB`) | `96.1%` ceil / `0.1%` floor |

*(Note: on `HardPOMDP v2 Strict Honest`, `MFEC + Page-Hinkley` achieves `0.6%` / `1.0%` / `0.3%` and `Tabular + PH` achieves `0.0%` / `0.2%` / `0.1%` across the three configurations respectively, because hidden sub-identity aliasing and 1/1728 sparse random odds defeat non-spinor memoryless/tabular policies.)*

**Methodological & Fairness Notes:**
1. *Single-Bank Baselines vs. `PageHinkley` Banks:* On continual regime-flip tasks (`T-Maze A->B->A`, `CartPole A->B->A`), `MicroContinualCore` maintains multi-regime banks switched via `PageHinkley` (`Ablation 4` isolates this directly: multi-bank `PH` lifts `Ret12ep` from `21.7%` to `66.9%`), whereas `RTRRL-LRU` and `Online GRU + A2C` maintain a single parameter bank.
2. *Per-Step Eligibility vs. Episodic BPTT:* Both `Cayley-RTRL` and `RTRRL-LRU` compute online per-step TD($\lambda$) eligibility updates with truncation-aware `end_episode_truncated` handling, whereas `Online GRU + A2C` caches the trajectory and executes episodic A2C + BPTT at episode end.
3. *`HardPOMDP v2` Unlabeled Cloze Pre-Adaptation:* Prior to the `HardPOMDP v2` RL regimes (`bench_step1`), `MicroContinualCore` runs `unlabeled_cloze_adapt(&logs, 35)` on unlabeled domain logs to prime Band-0 spinor token geometry, whereas `MFEC + PH`, `Tabular + PH`, and `Online GRU + A2C` start strictly from scratch on the RL episodes.

---

## 6. Limitations & Future Directions

1. **Exact Integer vs. `f32` Realization:** Lean 4 theorems verify algebraic invariants over `PosNat`/`Balance`/`GaussBalance`/`EisBalance`/`DualBalance`, whereas the Rust runtime operates in bounded IEEE-754 `f32` guarded by `sanitize_f32` and `#[axiom_zero]`. Extending proofs to fixed-point (`Q16.16`/`i32`) arithmetic is an immediate next step for bare-metal targets lacking FPUs.
2. **Continuous Spatial Drift without Discrete Switch Signals:** On smooth gradual drift in high-dimensional linear/MLP spaces (`Rotating Hyperplane 10D`: `77.3%` vs `Online MLP+Adam 82.0%`), dense neural layers remain faster per step (`1.44 µs` vs `6.56 µs`) when SEU guarantees are not required.
3. **Holdout Symbolic Generalization Under Sparse Delayed RL:** While Band 0 Cloze adaptation enables strong canonical transfer (`63.3%` on Strict Dense, `62.0%` on Strict Sparse), zero-shot unseen holdout generalization (`H:15.3%` / `H:12.7%`) remains open for future symbolic-relational consolidation work.

---

## 7. Reproducibility, Licensing & Citation

Every result in Section 5 can be reproduced locally from [`Cargo.toml`](Cargo.toml) (`axiom_zero` vendored at [`crates/axiom_zero`](crates/axiom_zero)) and [`proofs/MicroContinualCoreProofs.lean`](proofs/MicroContinualCoreProofs.lean):

```powershell
# Download Electricity (ELEC2) dataset from OpenML (needed once for ELEC2 tables):
Invoke-WebRequest -Uri "https://www.openml.org/data/download/2419/electricity-normalized.arff" -OutFile "electricity-normalized.arff"

cargo test --release
cargo run --release --bin bench_unified_ablation
cargo run --release --bin bench_step1
cargo run --release --bin bench_step2_streams
cargo run --release --bin bench_step3_pomdp_rl
cargo run --release --bin bench_step4_waves
lean proofs\MicroContinualCoreProofs.lean
```

Licensed under **`PolyForm-Noncommercial-1.0.0`** ([`LICENSE`](LICENSE); alternative OSI copyleft text provided in [`licenses/LICENSE-AGPL-3.0-only.txt`](licenses/LICENSE-AGPL-3.0-only.txt)). Commercial integration or closed corporate deployment inquiries require a separate commercial license from Adrian Aleksandrovich İşmuhanov ([mrwerdin@gmail.com](mailto:mrwerdin@gmail.com), [ORCID `0009-0002-4761-1952`](https://orcid.org/0009-0002-4761-1952)). Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
