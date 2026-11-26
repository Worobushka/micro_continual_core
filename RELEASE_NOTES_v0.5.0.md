# MicroContinualCore v0.5.0 Release Notes

**Release Date:** 2026-11-26  
**Authors:** Adrian Aleksandrovich İşmuhanov & Sofia Sidorenko  
**Zenodo Concept DOI:** [`10.5281/zenodo.24424982`](https://doi.org/10.5281/zenodo.24424982)  
**License:** `PolyForm-Noncommercial-1.0.0` (embedded `crates/axiom_zero`: `MIT OR Apache-2.0`)

---

## Executive Summary

`MicroContinualCore v0.5.0` completes six major algebraic, formal, optimization, and systems milestones across the zero-transcendental & zero-`sqrt` (`0% exp/ln/sin/cos/tanh`, `0% .sqrt()` via compile-time `[E013]`, `0` heap allocations on hot paths, enforced via `#[axiom_zero]`) streaming engine:

1. **68/68 Constructive Lean 4 Proofs with 0 Axioms (`proofs/MicroContinualCoreProofs.lean`) & Compile-Time `.sqrt()` Ban (`[E013]`):**
   - Added `[E013]` in `crates/axiom_zero` blocking `.sqrt()` and UFCS `sqrt` inside `#[axiom_zero]` (**20/20 `compile_fail` doctests**), tightened `#[cfg(test)]` macro-skip handling, replaced every `.sqrt()` across core `src/` with `inv_sqrt_newton_f32` / `sqrt_newton_f32`, and consolidated shared `#[axiom_zero]` algebraic primitives into `src/algebra.rs`.
   - Extended constructive `proofs/MicroContinualCoreProofs.lean` from 49 to **68 `#print axioms` checks** (`step01`–`step134`) with **0 axioms** (not even `propext` or `Quot.sound`) over exact constructive integers (`Balance`): $C^1$ selective step bounds (`step116`), rational unit quaternions & $\mathrm{SO}(4)$ double rotor (`step117`), gated Householder $\mathrm{O}(4)$ BIBO/parity & two-reflection $S_5$ witness (`step118`–`step119`), convex Master-Equation BIBO invariance (`step120`), dual derivatives (`step121`), Newton RMSNorm & Bartlett PH totality (`step122`), bounded polynomial repulsion (`step123`), `Q8.24`/`Q39.24` bounds (`step124`), Householder cascade embedding (`step125`), adaptive update algebra (`step126`), nested $\mathrm{SO}(2)\subset\mathrm{SO}(3)\subset\mathrm{SO}(4)\subset\mathrm{O}(4)$ parity & derivatives (`step127`), saddle escape & `Q8.24` product bounds (`step128`), Newton `inv_sqrt`/`sqrt` bracket & contraction (`step129`–`step130`), bank-head swap & 4-band IDBD stability (`step131`), `Q1.14` hypersphere prototype argmax margin preservation (`step132`), 3D Givens-parity matrix fusion exactness (`step133`), and non-degenerate `Q8.24` 4-HH $\mathrm{SO}(5)$ watershed isometry (`step134`).

2. **Suite A — $\mathrm{O}(4) \hookrightarrow \mathrm{SO}(5)$ Parity Bridge (`NonAbelianO4Rtrl`, `src/rtrl.rs`, `bench_step8_v050 --suite a`):**
   Four-Householder $\mathrm{SO}(5)$ cascades ($H_3(u_3)\cdots H_0(u_0)$ with non-degenerate watershed axis $u_3 = ((2-\beta)v_3, \beta)$, initialized at $\beta = 1.00$), with prototype-mode memory layout (no dead `v_out_w`/`v_out_b`), lazy dirty-class reprojection, fused `5×5` sensitivity transport, and `Q1.14` (`i16`) export/inference (`step132`):
   - **$S_5$ `NonAbelianO4Rtrl M=3` (`62.7 KB`; `45.1 KB` with `Q1.14` prototypes):** `99.5±1.4%` ($L=6$) $\to$ `96.8±8.1%` ($L=12$) $\to$ `87.5±25.5%` ($L=30$) $\to$ `77.2±33.2%` ($L=100$) $\to$ `67.9±34.7%` ($L=250$) $\to$ `54.0±39.7%` ($L=500$); Track 2 Unitary $S_5$: **`99.3% → 86.3% → 78.6% → 62.8%`**. Parity separation: `odd_beta=1.788±0.617`, `even_beta=0.162±0.502`; `Bit-30 State+Weight SEU @L=100`: `71.9±32.7%` (`any_nan=false`); `Label-Shuffle Control`: `0.7–1.4%`.
   - **$A_5$ `NonAbelianO4Rtrl M=3` (`43.2 KB`):** **`100.0%` ($L=6$) $\to$ `99.9%` ($L=30$) $\to$ `98.0%` ($L=100$) $\to$ `92.8±8.7%` ($L=500$)**; `SO(3)+Pillar2 v0.5.0 M=4` (`26.8 KB`): `100.0% → 99.3% → 91.0% → 59.1±38.2%` (vs `v0.4.0` `23.6%`).

3. **Suite B — `UnifiedCayleyCore` (`src/unified.rs`, Ablation Ladder `B1 -> B2 -> B3 One-Config`, Track 2 Unitary):**
   - Uses `shared_backbone` (`slots[0]` owns `B1+B2`, banks `1..3` store lightweight heads/gates/IDBD/PH), `use_band0 = false` in `B3 One-Config` (`true` in `B1`/`B2`), and 4-scalar per-band rational IDBD, keeping `One-Config` RAM at **`34.6 KB`** on `ELEC2` and **`26.2 KB`** on `RepeatPrev`, with `6000/6000` Causal Prefix Invariance and `0` heap allocations (`tests/zero_alloc_unified.rs`):
   - **Track 1 `One-Config` (`B3`, one hyperparameter set across all 8 suites):**
     - `ELEC2` (`45,312` rows, 5 seeds, `34.6 KB`): `84.96%` raw (`85.24±0.47%` `+y_{t-1}`, `Label-Shuffle 55.61±0.06%`).
     - `NASA C-MAPSS` (3-pass): `FD001` **`23.89`** RMSE (`RUL-Shuffle 40.83±1.01`), `FD004` **`26.94`** RMSE (`Bit-30 SEU 26.15±2.50`, `EW-Rec=53.0%`, `any_nan=false`; note `inject_weight_bit30_seu` flips `b1_alpha`/`b2_in_w`/`actor_w` rather than `critic_w`).
     - `CartPole-2D` / `CartPole +dx` (Train / Ev S / Ev R): `64.2 (79.1 / 86.3)` and `91.0 (124.0 / 89.0)`.
     - `RepeatPrevious`: $K=4$ `76.8%`, $K=8$ `58.0%`, $K=16$ `33.1±5.8%`, $K=24$ `27.4±3.0%` (`26.2 KB`).
     - `Dyck-2` (depth $3..5$): **`86.9%`** (`Depth>=3: 87.8%`, `Shuffled: 33.2%` vs `33.3%` chance; memoryless 0-step floor is `71.8%` overall / `77.8%` on `Depth>=3` since push targets equal current token).
     - `Multi-Scale Waves`: `ABC=90.0%`, `ZeroShot=89.7%`, `Flip=89.2%`, `Ret12=72.3%`.
     - `Non-Stationary T-Maze` (**zero `novelty_gate`**; $L=5$ `noise=±0.06`, $L=15$ `noise=±0.35`): $L=5$ **`Score=98.1% / Ret=98.5%`**, $L=15$ **`Score=94.9% / Ret=97.2%`** (`B1 Base` solves $L=15$ at `94.2%` via decoupled Master-Eq gain `g_in=2/3` + uncapped decay + Band RMSNorm; `B3` Bartlett PH bank switching lifts $A\to B\to A$ retention to `97.2%–98.5%`).
     - *Honest Price of Universality & Limitations:* Single `One-Config` still trails specialized `v0.4.1` configs on `ELEC2` (`84.96%` vs `91.5%`), `NASA` (`23.89/26.94` vs `19.51/23.33`), `CartPole` (`64.2/91.0` vs `117.6/187.4`), and `RepeatPrev K=8` (`58.0%` vs `85.1%`); without an episodic/Kerr associative memory bank, pure continuous A2C leaves `HardPOMDP v2` (24D Cloze) near chance (`Dense Online=0.2%`, `Sparse Online=0.9%`).
   - **Track 2 Unitary Limit (`sigma=0`):** `RepeatPrev K=24`: `25.9±1.7%` (`8.8 KB`), `A_5`: `100.0% -> 59.1%` (`26.8 KB`), `S_5`: `99.3% -> 62.8%` (`62.7 KB`).

4. **Suite C — Zero-FPU Fixed-Point Core `Q8.24` / `Q39.24` (`src/fixed_point.rs`):**
   - Pure integer (`i32`/`i64`/`i128`) core with a non-degenerate 4-Householder $\mathrm{SO}(5)$ `FixedO4Rotor` in `Q8.24` (`step134`) and precomputed `lc_cache`/`ls_cache`:
     - `q8_24_inv_sqrt [1/16000..1e6]` (3 Newton steps in `i128`): `max_rel_err = 2.021e-4`.
     - `FixedO4Rotor (4-HH SO(5), beta=1.0)` 10k-step `|norm^2 - 1|`: `unnorm = 4.687e-3 | renorm = 0.000e0`.
     - Small-step resolution (`lr=1e-4 * grad=0.05`): `Q8.24 raw=84 (5.007e-6)` vs `Q16.16 raw=0` (underflow).
     - `PageHinkleyQ39_24 @ RUL=125.0`: `mu_bank=125.000`, drift alarm in 1 step on drop to `40.0`.
     - **Deterministic 64-bit FNV-1a integer digests** verified across default release and `--features strict_deterministic` builds on Windows `x86_64` *(note `--features strict_deterministic` disables `x86_64` AVX2+FMA on the `f32` path, while `Q8.24` itself is pure integer; cross-architecture verification via `--target wasm32-wasip1` under `wasmtime` against these same 4 digests is planned next)*:
       - `inv_sqrt`: `0x38030f12b5509866`
       - `ELEC2` (`74.65±0.22%`, `2.17–2.89 us/step`): `0xe89e5db12b6719d2`
       - `NASA FD001` (`RMSE=27.20±4.65 cyc`, `1.68–2.05 us/step`): `0x614d2e87b5a2d524`
       - `Amplitude Stress (125..175)`: `0x285502bbee2b296d`

5. **Suite D — `AdaptiveOrthogonalRtrl` Dimensional Ladder through $S_6$ (`720` Classes, `src/adaptive_od.rs`):**
   - Zero heap allocations on hot paths including `predict_k_ahead_readonly` (`tests/zero_alloc_unified.rs`), fused `GivensParityOp` (`step133`), `Q1.14` (`i16`) 2× prototype deployment (`step132`), and 2-timescale shuffle-immune plateau escalation (`fast_ema`/`slow_ema` + 5-sigma $Z^2$):
     - `Label-Shuffle Control`: **`d = 2.00±0.00 [2..2]`** (`0.5–1.2%` chance, `0` false escalations across 10 seeds).
     - `C_5 (5 cls, 2.5k eps)`: `d = 2.20±0.63 [2..4]`, `15.3 KB`, **`97.2%` ($L=6$) $\to$ `88.8%` ($L=500$)**.
     - `A_5 (60 cls, 6k eps)`: `d = 4.80±1.32 [3..6]`, `44.8 KB`, `78.8%` ($L=6$) $\to$ `57.6%` ($L=12$) $\to$ `40.7%` ($L=30$) $\to$ `21.4%` ($L=500$).
     - `S_5 (120 cls, 12k eps)`: `d = 5.80±0.42 [5..6]`, `63.1 KB`, **`81.3%`** ($L=6$) $\to$ `38.1%` ($L=12$) $\to$ `11.8%` ($L=30$) $\to$ `2.8%` ($L=500$).
     - `S_6 (720 cls, 48k eps)`: **`d = 6.00±0.00 [6..6]`**, `245.9 KB` (`Q1.14` proto deploy: `~147 KB`), **`77.2±18.0%`** ($L=6$, vs `0.14%` chance), `25.8±19.7%` ($L=12$), collapsing at long horizons (`0.7%` @ $L=30$ $\to$ `0.1%` @ $L=500$); visit coverage across all 720 classes: `end_ep(min=3, med=38, mean=66.7)`, `total_step(min=33)`.
     - *Honest Caveats:* $A_5$ (`4.80`) and $S_5$ (`5.80`) slightly over-escalate above minimal $d=3,4$, and on long-horizon extrapolation ($L=30..500$), the adaptive frame notably trails fixed `NonAbelianO4Rtrl` (`S_5 @ L=500`: `2.8%` vs `54.0–62.8%`; `S_6 @ L=30..500`: `0.7% -> 0.1%`).

6. **Step 9 — Prime-Field Modular Arithmetic ($\mathbb{F}_{997}$ & $\mathbb{F}_{9973}$, `bench_step9_prime_arith`, `v0.4.0` vs `v0.4.1` vs `v0.5.0`):**
   - Strictly zero-oracle, disjoint 80/20 holdout evaluation across all 4 arithmetic operations (`Add`, `Mul`, `PowG`/`PowAB`, `DLog`) on verified prime fields $\mathbb{F}_{997}$ ($g=7$, order 996) and $\mathbb{F}_{9973}$ ($g=11$, order 9972):
     - **Bit-Serial `Add (a+b) mod p` (`LSB -> MSB`, 6D step features incl. normalized `(a/p, b/p)` alongside current bits):** `v0.4.1 NonAbelian SO(3)` (`10.7 KB`) leads at **`74.39±1.05%`** ($\mathbb{F}_{997}$, `Bit-0 72.87%`, `Shuf 49.1%`) and **`71.35±1.31%`** ($\mathbb{F}_{9973}$, `Bit-0 75.60%`, `Shuf 53.2%`), beating both the zero-carry rule (`69.3% / 65.5%`) and stateless `OnlineMlpAdam H=32` (`54.23% / 54.65%`), followed by `v0.4.1 MicroCore PURE 2L` (`56.79% / 59.98%`), `v0.5.0 NonAbelian O(4) M=1` (`56.03% / 53.12%`, `Shuf 48.3% / 52.7%`), and `v0.4.0 Alg.Sel.Cayley 2D` (`6.6 KB`, `52.33% / 55.57%`), while additive-input linear-readout cores (`Unified`, `Fixed Q8.24`) sit at the bit-majority floor (`50.2–53.5%`; note $\mathbb{F}_{9973}$ has a `53.87%` per-bit majority floor since bit 13 is `82.14%` zero).
     - **Honest Cryptographic Wall (`Mul`, `PowG`, `PowAB`, `DLog`):** All 8 models across `v0.4.0–v0.5.0` and `OnlineMlpAdam` score at the chance / majority / `Label-Shuffle` floor (`49.0–52.0%` on $\mathbb{F}_{997}$, `50.8–54.2%` on $\mathbb{F}_{9973}$; `0.2865–0.3595` direct residue RMSE vs uniform $\sigma = 1/\sqrt{12} = 0.2887$; Bit-30 SEU `any_nan=false`), confirming zero oracle leakage.
     - **Dynamic Order-12 Multiplicative Subgroup ($\langle 906 \rangle \le \mathbb{F}_{997}^\times$ & $\langle 1942 \rangle \le \mathbb{F}_{9973}^\times$, isomorphic to $C_{12}$ under `[1,0]/[0,1]` generator actions, distinct per-prime seeds, train $L \le 8$):** `v0.4.0 Alg.Sel.Cayley 2D` (`7.1 KB`: `69.6% → 9.6%` / `57.1% → 7.5%` at $L=6 \to 30$) $\to$ `v0.5.0 SO(3)+Pillar2 M=3` (`6.5 KB`: `96.7% → 39.6%` / `97.5% → 45.4%`) $\to$ **`v0.5.0 NonAbelianO4Rtrl M=2` (`11.1 KB`: `100.0% → 68.3±34.5%` on $\mathbb{F}_{997}$ seeds; `99.6% → 90.0±12.9% → 36.7±30.7%` on $\mathbb{F}_{9973}$ seeds)** with exact `Q1.14` (`i16`) inference parity (`9.9 KB`, `Label-Shuffle = 7.1–13.3%`).
