# `axiom-zero`

[![GitHub](https://img.shields.io/badge/github-Worobushka%2Faxiom--zero-blue.svg)](https://github.com/Worobushka/axiom-zero)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![Build Status](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

> **Compile-time totality validation and homeostatic runtime for zero-crash Rust systems.**

`axiom-zero` is a high-assurance Rust toolkit designed for mission-critical software, autonomous agent loops, and distributed systems where process termination, thread panics, and unhandled empty states are unacceptable.

It enforces a strict architectural rule: **Constructive Totality (Non-Self-Negation)**. Any operation capable of collapsing a thread into non-existence (`panic!`, `.unwrap()`, `Option::None`, `unreachable!`) is statically intercepted at compile time and replaced with total presence containers and homeostatic state machines.

---

## The Engineering Problem

Production outages across modern backends, autonomous agents, and embedded devices overwhelmingly stem from three patterns:

1. **The Vacuum Trap (`Option::None` / Null):** Representing missing records as an uninhabited empty state that cascades into downstream unwrap failures.
2. **Crash-On-Error (`panic!`, `.unwrap()`, `.expect()`):** Terminating threads when inputs deviate from happy-path assumptions, triggering cascading failures (`CrashLoopBackOff` in orchestrators).
3. **Partial Control Flows (`todo!`, `unimplemented!`, non-exhaustive paths):** Leaving execution gaps that pass compiler checks but fail catastrophically at runtime under edge cases.

---

## How `axiom-zero` Solves This

`axiom-zero` introduces two coordinated layers:

| Layer | Mechanism | Engineering Guarantee |
| :--- | :--- | :--- |
| **Static Compiler Macro** (`#[axiom_zero]`) | AST-level validation pass (`[E001]`–`[E012]`) using `syn` | Statically prevents compilation of `panic!`, `assert!`/`debug_assert!`, `.unwrap*()`/`.expect*()`, `Option::None`, `todo!`, `unreachable!`, `exit`/`abort`, raw indexing `arr[i]`, raw `/` / `%`, and transcendental float methods (`exp`/`ln`/`sin`/`cos`/`tanh`/`powf`/etc.). |
| **Total Presence Container** (`Presence<T>`) | Replacement for `Option<T>` | Eliminates unhandled absence; queries resolve to either a live payload or an explicit boundary fallback token. |
| **Non-Empty Attractor** (`Attractor<S>`) | Structurally non-empty limit cycle `(head, tail)` | Period `>= 1` guaranteed by construction (`NonZeroUsize`) with zero-panic phase indexing and $\tau = 1$ perturbation recovery. |
| **Total Slice & Arithmetic Guards** (`total_get`, `total_div_*`, `sanitize_bounded_f32`) | Bounds-checked & zero-safe runtime primitives | Eliminates out-of-bounds panics, integer/float division-by-zero traps, `NaN`/`Inf`, and extreme amplitude spikes. |

---

## Quickstart

Add `axiom_zero` to your `Cargo.toml`:

```toml
[dependencies]
axiom_zero = "0.1.0"
```

### 1. Compile-Time Rejection of Destructive Primitives

Annotate functions or modules with `#[axiom_zero]`. The compiler statically enforces totality:

```rust
use axiom_zero::axiom_zero;

#[axiom_zero]
fn vulnerable_handler(input: Option<i32>) {
    // COMPILATION ERROR: [E007] Direct reference to `Option::None` attempts to instantiate void.
    let fallback = None;

    // COMPILATION ERROR: [E005] Method `.unwrap()` permits thread collapse on None/Err.
    let value = input.unwrap();

    // COMPILATION ERROR: [E001] `panic!` invokes state annihilation.
    if value < 0 {
        panic!("Invalid value received");
    }
}
```

#### Compiler Output:
```text
error: [E007] AxiomZero Violation: Direct reference to `Option::None` attempts to instantiate void (Ø). 
       Use `Presence::boundary(default)` or define a concrete fallback token.

       [E005] AxiomZero Violation: Method `.unwrap()` permits thread collapse on None/Err. 
       Use Total Presence (`Presence<T>`) or handle boundary values via `.resolve()`.

       [E001] AxiomZero Violation: `panic!` invokes state annihilation. 
       Replace with orbital relaxation (`Attractor::absorb_perturbation`) or return a boundary-guarded `Presence<T>`.
  --> src/main.rs:4:1
   |
 4 | fn vulnerable_handler(input: Option<i32>) {
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

---

### 2. Total Presence (`Presence<T>`)

Instead of returning `Option<T>` (which admits unhandled absence), return `Presence<T>`:

```rust
use axiom_zero::{axiom_zero, Presence};

#[derive(Debug, PartialEq, Eq)]
pub struct UserSession {
    pub user_id: u64,
    pub permissions: u8,
}

#[axiom_zero]
pub fn fetch_session(user_id: u64) -> Presence<UserSession> {
    if user_id == 42 {
        Presence::new(UserSession {
            user_id: 42,
            permissions: 0xFF,
        })
    } else {
        // Safe boundary token: guarantees a valid, non-null fallback state
        Presence::boundary(UserSession {
            user_id: 0,
            permissions: 0x00,
        })
    }
}

fn main() {
    let session = fetch_session(999);

    // Resolve safely without unwrap or panic risk
    let role = session.resolve(
        |user| format!("Authenticated User #{}", user.user_id),
        |_guest| "Guest/Boundary session active".to_string(),
    );

    println!("Status: {}", role);
}
```

---

### 3. Homeostatic Attractor Engine (`Attractor<S>`)

For services, autonomous loops, or event pipelines, model state evolution as a closed orbital cycle:

```rust
use axiom_zero::{axiom_zero, Attractor};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelinePhase {
    Ingest,
    Verify,
    Commit,
    Broadcast,
}

#[axiom_zero]
fn main() {
    let mut service = Attractor::new(
        PipelinePhase::Ingest,
        vec![
            PipelinePhase::Verify,
            PipelinePhase::Commit,
            PipelinePhase::Broadcast,
        ],
    );

    // Normal forward execution
    let _ = service.step();
    let _ = service.step();

    // SIMULATED ANOMALOUS HARDWARE SPIKE / CORRUPTED INPUT
    // Instead of panicking or restarting, the attractor absorbs perturbation in tau = 1:
    let recovered_phase = service.absorb_perturbation(1);

    println!("Recovered Phase: {:?}", recovered_phase);
    // Pipeline remains on its invariant limit-cycle without dropping frames or crashing!
}
```

---

## Comparison: Standard Rust vs `axiom-zero`

| Aspect | Standard Idiomatic Rust | Rust with `axiom-zero` |
| :--- | :--- | :--- |
| **Missing Data** | `Option::None` (risks unhandled cascades) | `Presence<T>` (enforces boundary tokens) |
| **Error Handling** | `panic!`, `.unwrap()`, `assert!` | Compile-time rejection; homeostatic relaxation |
| **Process Model** | `Crash-and-restart` (supervised pods) | `Homeostatic-orbital` (self-healing in $\tau = 1$) |
| **Memory Policy** | Unconditional deallocation (`Drop`) | Resource assimilation (`Absorb`) |
| **Service Downtime** | `CrashLoopBackOff` on unexpected edge cases | **0 downtime**: state space topologically closed |

---

## Zero-Overhead Performance Guarantees

* **Zero Allocation:** `Presence<T>` uses standard struct alignment and compile-time inlining.
* **Predictable Branching:** `Attractor<S>` is implemented as a contiguous vector with index modulo arithmetic ($\mathcal{O}(1)$ step, zero heap operations during execution).
* **Compile-Time Only Overhead:** The static AST validator runs exclusively during `cargo check` and `cargo build`, leaving no runtime tracing footprint.

---

## Running Tests

Run the unit and macro validation suite in `tests/api_tests.rs`:

```bash
cargo test --manifest-path crates/axiom_zero/Cargo.toml
```

---

## Validation Error Codes (`[E001]`–`[E012]`)

Inside any `#[axiom_zero]` function, module, or `impl` block (`#[test]` and `#[cfg(test)]` are exempted):

| Code | Rejected Construct | Total Replacement |
| :--- | :--- | :--- |
| `[E001]` | `panic!` | Return `Presence::boundary(...)` or absorb via `Attractor` |
| `[E002]`–`[E004]` | `todo!`, `unimplemented!`, `unreachable!` | Total match arms and complete transition logic |
| `[E005]`–`[E006]` | `.unwrap*()`, `.expect*()` | `.resolve(...)`, `.unwrap_or(...)`, `Presence::from_option` |
| `[E007]` | `None` / `Option::None` | `Presence::boundary(default)` or wildcard match fallback |
| `[E008]` | `assert*!`, `debug_assert*!` | `RelationalIdentity::measure` or `Presence<T>` |
| `[E009]` | `exit`, `abort`, `panic_any` | Bounded recovery / boundary tokens |
| `[E010]` | Raw indexing `arr[i]` | `total_get`, `total_get_copy`, `total_set`, `mask_get`, `TotalSliceExt` |
| `[E011]` | Raw division `/` and modulo `%` | `total_div_usize`, `total_rem_nonzero`, `total_div_f32`, `total_div_unit_shifted_f32` |
| `[E012]` | Transcendental float methods (`exp`, `exp2`, `exp_m1`, `ln`, `ln_1p`, `log*`, `sin*`, `cos*`, `tan*`, `asin*`, `acos*`, `atan*`, `sinh`, `cosh`, `tanh`, `powf`) | Exact rational/algebraic operators (Cayley rotors, Kerr competition, rational softmax) |

## Finite Stabilization & Slice Access

```rust
use axiom_zero::{
    axiom_zero, mask_get, sanitize_bounded_f32, total_div_unit_shifted_f32, total_get_copy,
    Presence,
};

#[axiom_zero]
fn process_reading(samples: &[f32], valid_mask: &[bool], idx: usize) -> Presence<f32> {
    if !mask_get(valid_mask, idx) {
        return Presence::boundary(0.0);
    }
    let raw = total_get_copy(samples, idx, 0.0);
    let safe = sanitize_bounded_f32(raw, 100.0);
    Presence::new(total_div_unit_shifted_f32(safe, safe * safe))
}
```

---

## License

Dual-licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
