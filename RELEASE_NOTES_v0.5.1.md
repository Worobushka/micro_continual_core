# MicroContinualCore v0.5.1 Release Notes

**Release Date:** 2026-11-28  
**Authors:** Adrian Aleksandrovich İşmuhanov & Sofia Sidorenko  
**Zenodo Concept DOI:** [`10.5281/zenodo.24424982`](https://doi.org/10.5281/zenodo.24424982)  
**License:** `PolyForm-Noncommercial-1.0.0` (embedded `crates/axiom_zero`: `MIT OR Apache-2.0`)

---

## Executive Summary & Forensic Sanitization Statement

`MicroContinualCore v0.5.1` upgrades `UnifiedCayleyCore` and `AdaptiveOrthogonalRtrl`, adds **4 new 0-axiom constructive Lean 4 proofs** (`step135`–`step138`, bringing the total to **72/72**, `0 axioms`), verifies **4/4 bit-identical `Suite C` (`Q8.24`/`Q39.24`) FNV-1a hashes across `x86_64`, `x86_64 --features strict_deterministic`, and `wasm32-wasip1`**, and adds **Table 1A** (Epistemic Status) plus 3 synchronized figures to `PAPER.md`.

Crucially, **every mechanism in `v0.5.1` underwent a strict post-implementation forensic audit and 5-point sanitization** before release:
1. **Complete flag decoupling in `UnifiedCayleyConfig`:** Split every auxiliary behavior out of `enable_lyap_lk` into explicit independent flags (`use_band0`, `enable_smooth_var_norm`, `enable_shared_actor_rms`, `enable_idbd_binary_fix`, `enable_j_xor`, `enable_bank_rotor_adapter`, `enable_finsler_diode`, `enable_lyap_lk`) and removed an ad-hoc `b == 4` regression scaling branch.
2. **Honest leave-one-out (LOO) pruning of `P2` and `P3`:** Under strict decoupled LOO ablation, `P2` (`160 B/bank` geodesic rotor adapter, `step136`) had `0.00` effect on single-bank tasks (`ELEC2`, `RepeatPrev`, `Add mod 997`) and pushed `RepeatPrev` over its `32 KB` RAM ceiling (`32.9 KB`), while `P3` (normalized Finsler-Randers contrast diode, `step137`) degraded `RepeatPrev K=8` by `-3.7 pp` (`68.0% → 64.3%`) and also exceeded `32 KB` (`33.1 KB`). Both remain compiled, Lean-proven, FD-tested, SEU-tested, and 0-alloc-verified **optional primitives**, and are **disabled by default in `UnifiedCayleyConfig::one_config()`**.
3. **Transparent attribution of `Band 0` and `smooth_var_norm`:** `var + 0.0035` is explicitly attributed as a smooth sub-unit Welford variance floor (preventing `v0.5.0`'s `var + 1.0` from compressing `[0, 1]` sensors like `ELEC2` by `~10×`, driving `+3.55 pp` on `ELEC2` in LOO) rather than hidden under `P2`/`P4`.
4. **Removal of placebo `lambda_delta_rank` in `AdaptiveOrthogonalRtrl (P5)`:** Ablation showed `lambda_delta_rank` had `0.0` effect on `A_5` and slightly hurt `S_5`. It was removed completely (`S_6` RAM `247.1 → 246.6 KB`), replaced with honest mechanism naming (**3-Tier Anchor/Warmup LR + Shell Utilization + 2-Block Plateau/Climbing Gating + Smooth `center_ramp` + CE/Center-Pull + `m_theta`**), and audited on **4 unseen blind OOD groups** (`D_5`, `A_4`, `S_4`, `C_7`) plus a **`24k`-episode `S_5` clock-runout check**.
5. **100% Lean 4 / Rust / Figure / Paper synchronization:** Aligned `step137` (`tau_+ + tau_- = 2*D*rho > 0`, `|c(v, delta)| < 1` matching `finsler_randers_contrast`), `step138` (exact `5/4 = 1.25 / (1 + s^2)` weight and dual gradient matching `src/algebra.rs:825-830`), benchmark labels in `bench_step9_prime_arith.rs`, and all 3 publication figures in `assets/`.

---

## 1. Formal, Systems & Cross-Architecture Verification

- **Lean 4 (`lean proofs/MicroContinualCoreProofs.lean`):** **72/72** `#print axioms` checks (`step01`–`step138`), **0 axioms** (not even `propext` or `Quot.sound`):
  - `step135` (`P1`): Symplectic `J`-XOR antisymmetry $\Omega(u,v) + \Omega(v,u) = 0$, diagonal vanishing $\Omega(u,u) = 0$, $J^2 = -I_2$, rational saturation denominator totality.
  - `step136` (`P2`, optional): Orthogonal composition isometry & $\det = +1$ for per-bank 2D/3D Cayley geodesic rotor adapters.
  - `step137` (`P3`, optional): Finsler-Randers Cauchy–Schwarz positivity $\tau(\delta) > 0$, directional asymmetry $\tau(+\delta) - \tau(-\delta) = 2\langle\beta, \delta\rangle$, and contrast totality $\tau(+\delta) + \tau(-\delta) = 2\rho > 0 \implies |c(v,\delta)| < 1$.
  - `step138` (`P4`): Lyapunov backbone gate bounds $0 < g_{\text{Lyap}} \le 1$, monotonic surge damping, exact `5/4 = 1.25 / (1 + s^2)` rational Liebel–Koerner weight positivity, cleared-denominator objective $5L + 2\lambda s^2(1+s^2) \ge 0$, and exact `DualBalance` derivative numerator.
- **`#[axiom_zero]` & Unit / Alloc Tests:**
  - `python scripts/audit_v050_invariants.py`: **10/10** core modules clean (`0` forbidden math/panic calls).
  - `crates/axiom_zero`: **4 unit + 4 API + 1 doc + 20 `compile_fail`** (`[E012]` + `[E013]`) PASS.
  - `cargo test --release --lib` & `--features strict_deterministic --lib`: **46/46** PASS (both builds).
  - `cargo test --release --test zero_alloc_unified`: **2/2** PASS (**0 heap allocations** on `one_config()`, `one_config_v050()`, optional `P2+P3+shared_actor_rms`, and `AdaptiveOrthogonalRtrl`).
  - `bench_forensic_audit`: **6000/6000** Causal Prefix Invariance; all Label-Shuffle controls at chance.
- **Cross-Architecture `wasm32-wasip1` Determinism (`Suite C`, `scripts/run_wasm_suite_c.mjs`):**
  Fixed a genuine 32/64-bit platform bug in `bench_step8_v050`'s Fisher–Yates shuffle (`(rng as usize) % n` → `(rng % (n as u64)) as usize`). All **4/4** 64-bit FNV-1a integer digests now match bit-for-bit across `x86_64` default release, `x86_64 --features strict_deterministic`, and `wasm32-wasip1` (Node.js WASI):
  - `[C-1]` `q8_24_inv_sqrt`: `0x38030f12b5509866` (`max_rel_err = 2.021e-4`)
  - `[C-2]` `ELEC2 Q8.24` (`74.65±0.22%`): `0xe89e5db12b6719d2`
  - `[C-3]` `NASA FD001 Q8.24` (`RMSE 27.20±4.65`): `0x614d2e87b5a2d524`
  - `[C-4]` `Amplitude Stress (125..175)`: `0x285502bbee2b296d`

---

## 2. Clean `UnifiedCayleyCore One-Config` Results & Decoupled LOO Ablation

Active default preset: `Band0(4)` + `smooth_var_norm` + `idbd_binary_fix` + `P1 (J-XOR)` + `P4 (Lyap/LK)` (`P2 = OFF`, `P3 = OFF`, `shared_actor_rms = OFF`; Bit-30 Weight SEU now includes `critic_w`):

| Benchmark (ONE Single Preset!) | `v0.5.0 B3` | **`Clean v0.5.1`** | Delta | Anti-Leakage / SEU / RAM |
| :--- | :---: | :---: | :---: | :--- |
| **`ELEC2` Raw / `+y_{t-1}` (5 seeds)** | `84.96% / 85.24%` | **`89.36% / 89.75±0.50%`** | **`+4.40 / +4.51 pp`** | `Shuf = 54.48±0.17%`, `RAM = 35.6 / 38.2 KB ≤ 40 KB` |
| **`CartPole 2D` Train / Ev(S/R)** | `64.2 (79.1 / 86.3)` | **`135.4 (176.0 / 161.2)`** | **`+71.2` (`+96.9/+74.9`)** | 10 seeds, A→B→A flip (`P1 J-XOR` driven) |
| **`CartPole +dx` Train / Ev(S/R)** | `91.0 (124.0 / 89.0)` | **`123.8 (163.7 / 94.4)`** | **`+32.8` (`+39.7/+5.4`)** | 10 seeds, A→B→A flip |
| **`RepeatPrev K=4/8/16/24` (RL)** | `76.8 / 58.0 / 33.1 / 27.4%` | **`77.8 / 64.6 / 40.3 / 31.0%`** | **`+6.6 / +7.2 / +3.6 pp`** | **`RAM = 31.6 KB ≤ 32 KB`**, 10 seeds |
| **`Dyck-2` All / `Depth >= 3`** | `86.9% / 87.8%` | **`87.8% / 88.5%`** | `+0.9 / +0.7 pp` | `Shuf = 33.1%` (chance `33.3%`) |
| **`Waves` ABC / ZS / Flip / Ret12** | `90.0 / 89.7 / 89.2 / 72.3%` | **`90.5 / 90.8 / 89.7 / 75.7%`** | **`+0.5 / +1.1 / +0.5 / +3.4 pp`** | 10 seeds, A→B→A flip |
| **`T-Maze` `L=15` (0 gate) / `L=5`** | `94.9/97.2%` / `98.1/98.5%` | **`93.6/93.5%`** / **`97.8/99.2%`** | `-1.3 pp` | **0 `novelty_gate`**, sparse reward |
| **Bit-Serial `Add mod 997 / 9973`** | `53.0% / 53.5%` | **`65.67±3.12% / 57.05±4.42%`** | **`+12.7 / +3.6 pp`** | `Shuf = 48.5% / 52.4%`, `RAM = 30.3 KB` (`Mul/Pow/DLog` at chance) |
| **`NASA C-MAPSS` `FD001 / FD004`** | `23.89 / 26.94` | **`24.46 / 28.90`** | `+0.57 / +1.96 cyc` | `FD004 SEU = 27.26±1.35` (`EW-Rec 49.6%, any_nan=false`), `RUL-Shuf = 41.27` |
| **Step 2 Streams (`STAG/SEA/Hyp/RBF`)** | `91.6 / 81.4 / 77.6 / 91.4%` | **`93.4 / 80.8 / 77.2 / 91.1%`** | **`+1.8`** / `-0.6..-0.3 pp` | 10 seeds, 1-pass Test-then-Train, SEU=YES |

*Decoupled LOO Highlights (3–5 seeds, see `PAPER.md` Table 6.1A):* `-smooth_var_norm` drops `ELEC2` by `-3.55 pp` (`89.27% → 85.72%`); `-P1` drops `CartPole 2D` by `-61.6 st` (`131.1 → 69.5`) and `Add mod 997` by `-9.80 pp` (`61.53% → 51.73%`); `-P4` drops `RepeatPrev K=8` by `-4.8 pp` (`68.0% → 63.2%`); `+P2` gives `0.00` delta on `ELEC2`/`RPrev`/`Add997` (`32.9 KB > 32 KB`); `+P3` drops `RPrev K=8` by `-3.7 pp` (`33.1 KB > 32 KB`).  
*Honest Gaps:* Single `One-Config` still trails specialized `v0.4.1 PURE 2L` (`19.51 / 23.33`) on `NASA C-MAPSS`, `OnlineMlpAdam` on `SEA`/`Hyperplane`, and requires an episodic `KerrKernelBank` for `HardPOMDP v2` Cloze (`0.0–1.0%`).

---

## 3. Placebo-Free `AdaptiveOrthogonalRtrl (P5)` — Suite D, Blind OOD Groups & `S_5 24k`

- **Suite D Core Ladder (10 seeds, 0 heap allocs):**
  - `Label-Shuffle`: **`d = 2.00±0.00`** (`0` false esc, `0.2–1.1%`, `63.8 KB`).
  - `C_5 (5 cls)`: **`d = 2.00±0.00`**, **`99.6 | 98.5 | 97.3 | 98.5 | 99.1 | 99.0±2.0%`** (**`+10.2 pp`** @ `L=500`, `15.7 KB`).
  - `A_5 (60 cls)`: **`d = 3.10±0.32`** (`9/10` exact `d=3`, down from `4.80`), **`94.9 | 90.8 | 85.2 (+44.5 pp) | 63.2 (+38.0 pp) | 38.4 | 20.6%`** (`45.6 KB`).
  - `S_5 (120 cls, 12k eps)`: **`d = 4.20±0.42`** (**`8/10` exact `d=4`!** down from `5.80`), **`82.2 | 46.9 (+8.8 pp) | 8.8 | 1.3 | 1.4 | 0.8%`** (`63.8 KB`).
  - `S_6 (720 cls, 48k eps)`: **`d = 5.50±0.53`** (`5/10` exact `d=5`), **`93.0 (+15.8 pp) | 42.7 (+16.9 pp) | 4.5 (6.4×) | 0.6 | 0.1 | 0.2%`** (`246.6 KB`).
  - `Step 9 C_12` (`26.4 KB`, `d=2.8`): `61.7±38.7%` ($\mathbb{F}_{997}^\times$) and `83.3±15.8%` ($\mathbb{F}_{9973}^\times$) @ `L=6` (`Shuf = 5.4–14.6%`).
- **Blind OOD Finite-Group Verification (5 seeds, zero tuning exposure) & `S_5 24k` Clock-Runout:**
  - `D_5 (ord=10, O(2))`: `d = 2.20 [2..3]`, **`100.0 | 100.0 | 100.0 | 100.0 | 99.1%`** @ `[6, 12, 30, 100, 500]`.
  - `A_4 (ord=12, SO(3))`: **`d = 3.00 [3..3]` (`5/5` exact!)**, **`100.0 | 100.0 | 100.0 | 99.4 | 86.0%`**.
  - `S_4 (ord=24, O(3))`: `d = 2.80 [2..3]`, **`96.6 | 90.3 | 82.9 | 76.3 | 46.9%`**.
  - `C_7 (2-gen, ord=7, SO(2))` — **Honest Limit:** `d = 2.40 [2..3]`, `100.0 | 96.7 | 56.5 | 57.6 | 0.0%` (tight `51.4°` spacing + inverse-angle random-walk drift $\theta(+1) + \theta(-1) \neq 0$ by `L=500`).
  - `S_5 (24k eps clock-runout)` — **Honest Check:** `d = 5.00 [4..6]` (confirming `min_dwell` partially caps `d` at `12k`), while extrapolation climbs to **`92.3 | 58.3 | 40.8 | 39.2 | 22.4%`** @ `L=500`.

---

## 4. New Artifacts & Documentation

- [PAPER.md](PAPER.md): Updated Abstract, Section 2 (`step135`–`step138`), **Table 1A** (Epistemic Status), and **Section 6** (Tables 6.1A, 6.1B, 6.2A, 6.2B, and 6.3).
- [ROADMAP_v0.5.1.md](ROADMAP_v0.5.1.md) & [ROADMAP_v0.6.0.md](ROADMAP_v0.6.0.md): Completed `v0.5.1` plan and next-generation modular + fractional-dimension roadmap for `v0.6.0`.
- Synchronized Publication Figures (`scripts/plot_v051_figures.py`): [assets/v051_o4_orbits_and_watershed.png](assets/v051_o4_orbits_and_watershed.png), [assets/v051_unified_ablation_ladder.png](assets/v051_unified_ablation_ladder.png), and [assets/v051_adaptive_od_trajectories.png](assets/v051_adaptive_od_trajectories.png) (plus print PDF and dark variants under `assets/figures/`).
