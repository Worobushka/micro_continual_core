//! # Axiom-Zero (`axiom_zero`)
//!
//! A compile-time totality validator and homeostatic runtime toolkit for zero-crash Rust.
//!
//! Modern software often relies on partial operations that can collapse at runtime: `Option::None`,
//! `panic!`, `.unwrap()`, out-of-bounds indexing (`arr[i]`), and unchecked division or modulo
//! (`/` and `%`).
//!
//! `axiom_zero` replaces those fragile patterns with a constructive, non-annihilating model:
//!
//! * **Compile-Time Static Validation (`#[axiom_zero]`):** AST validation pass (`[E001]`–`[E011]`)
//!   that rejects panics, assertions, `todo!`/`unimplemented!`/`unreachable!`, `.unwrap*()`/`.expect*()`,
//!   direct `None` references, terminating calls (`exit`/`abort`), raw indexing (`arr[i]`), and raw
//!   division/modulo (`/` and `%`) at compile time.
//! * **Total Presence (`Presence<T>`):** Algebraic replacement for `Option<T>` that eliminates empty
//!   null states by resolving every query to either a `Live(T)` payload or an explicit `Boundary(T)`
//!   fallback token.
//! * **Non-Empty Homeostatic Attractor (`Attractor<S>`):** Inductively non-empty cyclical state
//!   machine (`(head, tail)` with `period >= 1`) using zero-free modular phase indexing and total
//!   slice access.
//! * **Total Slice, Arithmetic & Amplitude Guard Helpers:** Bounds-safe `total_get`/`mask_get`,
//!   zero-safe integer/float division (`total_div_*`, `total_rem_nonzero`,
//!   `total_div_unit_shifted_f32`), and coercive finite clamping (`sanitize_f32`,
//!   `sanitize_bounded_f32`).
//!
//! ## Compile-Time Rule Verification (Doctests)
//!
//! `#[axiom_zero]` rejects partial primitives at compile time:
//!
//! ```compile_fail
//! use axiom_zero::axiom_zero;
//!
//! #[axiom_zero]
//! fn bad_panic() {
//!     panic!("boom");
//! }
//! ```
//!
//! ```compile_fail
//! use axiom_zero::axiom_zero;
//!
//! #[axiom_zero]
//! fn bad_unwrap(x: Option<i32>) -> i32 {
//!     x.unwrap()
//! }
//! ```
//!
//! ```compile_fail
//! use axiom_zero::axiom_zero;
//!
//! #[axiom_zero]
//! fn bad_none() -> Option<i32> {
//!     None
//! }
//! ```
//!
//! ```compile_fail
//! use axiom_zero::axiom_zero;
//!
//! #[axiom_zero]
//! fn bad_debug_assert(x: i32) {
//!     debug_assert!(x > 0);
//! }
//! ```
//!
//! ```compile_fail
//! use axiom_zero::axiom_zero;
//!
//! #[axiom_zero]
//! fn bad_abort() {
//!     std::process::abort();
//! }
//! ```
//!
//! ```compile_fail
//! use axiom_zero::axiom_zero;
//!
//! #[axiom_zero]
//! fn bad_index(xs: &[i32]) -> i32 {
//!     xs[0]
//! }
//! ```
//!
//! ```compile_fail
//! use axiom_zero::axiom_zero;
//!
//! #[axiom_zero]
//! fn bad_div(a: usize, b: usize) -> usize {
//!     a / b
//! }
//! ```
//!
//! ```compile_fail
//! use axiom_zero::axiom_zero;
//!
//! #[axiom_zero]
//! fn bad_rem(a: usize, b: usize) -> usize {
//!     a % b
//! }
//! ```
//!
//! ```compile_fail
//! use axiom_zero::axiom_zero;
//!
//! #[axiom_zero]
//! fn bad_div_assign(mut a: usize, b: usize) -> usize {
//!     a /= b;
//!     a
//! }
//! ```
//!
//! ```compile_fail
//! use axiom_zero::axiom_zero;
//!
//! #[axiom_zero]
//! fn bad_index_write(xs: &mut [i32]) {
//!     xs[0] = 1;
//! }
//! ```
//!
//! ```compile_fail
//! use axiom_zero::axiom_zero;
//!
//! #[axiom_zero]
//! fn bad_expect(x: Option<i32>) -> i32 {
//!     x.expect("missing")
//! }
//! ```
//!
//! ```compile_fail
//! use axiom_zero::axiom_zero;
//!
//! #[axiom_zero]
//! fn bad_match_none(x: Option<i32>) -> i32 {
//!     match x {
//!         Some(v) => v,
//!         None => 0,
//!     }
//! }
//! ```
//!
//! ```compile_fail
//! use axiom_zero::axiom_zero;
//!
//! #[axiom_zero]
//! fn bad_exit() {
//!     std::process::exit(1);
//! }
//! ```
//!
//! ```compile_fail
//! use axiom_zero::axiom_zero;
//!
//! #[axiom_zero]
//! fn bad_incomplete(flag: bool) {
//!     if flag {
//!         todo!();
//!     } else {
//!         unreachable!();
//!     }
//! }
//! ```
//!
//! Total code compiles cleanly:
//!
//! ```rust
//! use axiom_zero::{
//!     axiom_zero, sanitize_bounded_f32, total_div_unit_shifted_f32, total_get_copy, Attractor,
//!     Presence,
//! };
//!
//! #[axiom_zero]
//! fn total_pipeline(xs: &[f32], idx: usize) -> Presence<f32> {
//!     let raw = total_get_copy(xs, idx, 0.0);
//!     let safe = sanitize_bounded_f32(raw, 10.0);
//!     let normalized = total_div_unit_shifted_f32(safe, safe * safe);
//!     if raw.is_finite() && raw.abs() <= 10.0 {
//!         Presence::new(normalized)
//!     } else {
//!         Presence::boundary(0.0)
//!     }
//! }
//!
//! let mut orbit = Attractor::new("IDLE", vec!["READ", "PROCESS", "EMIT"]);
//! assert_eq!(*orbit.step(), "READ");
//! assert!(matches!(total_pipeline(&[3.0], 0), Presence::Live(_)));
//! assert!(matches!(total_pipeline(&[1e25], 0), Presence::Boundary(_)));
//! ```

pub use axiom_zero_macros::axiom_zero;

use std::fmt::Debug;
use std::num::NonZeroUsize;

/// Internal primitive arithmetic kernels.
///
/// External and `#[axiom_zero]`-validated code calls the total public wrappers below,
/// which enforce finiteness and non-zero lower bounds prior to IEEE-754 division.
mod checked_ops {
    #[inline]
    pub(crate) fn div_f32_checked(num: f32, den: f32) -> f32 {
        num / den
    }

    #[inline]
    pub(crate) fn div_f64_checked(num: f64, den: f64) -> f64 {
        num / den
    }
}

/// Total integer division on `usize`, returning `fallback` when `den == 0`.
#[axiom_zero]
#[inline]
pub fn total_div_usize(num: usize, den: usize, fallback: usize) -> usize {
    num.checked_div(den).unwrap_or(fallback)
}

/// Total integer remainder on `usize`, returning `fallback` when `den == 0`.
#[axiom_zero]
#[inline]
pub fn total_rem_usize(num: usize, den: usize, fallback: usize) -> usize {
    num.checked_rem(den).unwrap_or(fallback)
}

/// Total remainder modulo a statically guaranteed non-zero `NonZeroUsize`.
#[axiom_zero]
#[inline]
pub fn total_rem_nonzero(idx: usize, modulus: NonZeroUsize) -> usize {
    idx.checked_rem(modulus.get()).unwrap_or(0)
}

/// Total integer division by a statically guaranteed non-zero `NonZeroUsize`.
#[axiom_zero]
#[inline]
pub fn total_div_nonzero_usize(num: usize, den: NonZeroUsize) -> usize {
    num.checked_div(den.get()).unwrap_or(0)
}

/// Total signed integer division on `i64`, catching both `den == 0` and `i64::MIN / -1` overflow.
#[axiom_zero]
#[inline]
pub fn total_div_i64(num: i64, den: i64, fallback: i64) -> i64 {
    num.checked_div(den).unwrap_or(fallback)
}

/// Total signed integer remainder on `i64`, catching both `den == 0` and `i64::MIN % -1` overflow.
#[axiom_zero]
#[inline]
pub fn total_rem_i64(num: i64, den: i64, fallback: i64) -> i64 {
    num.checked_rem(den).unwrap_or(fallback)
}

/// Total IEEE-754 `f32` division.
///
/// Returns `fallback` (or `0.0` if `fallback` itself is non-finite) whenever inputs are non-finite,
/// `|den| <= 1e-12`, or the quotient fails finiteness checks.
#[axiom_zero]
#[inline]
pub fn total_div_f32(num: f32, den: f32, fallback: f32) -> f32 {
    let safe_fallback = sanitize_f32(fallback);
    if !num.is_finite() || !den.is_finite() || den.abs() <= 1e-12 {
        return safe_fallback;
    }
    let q = checked_ops::div_f32_checked(num, den);
    if q.is_finite() {
        q
    } else {
        safe_fallback
    }
}

/// Total IEEE-754 `f64` division.
///
/// Returns `fallback` (or `0.0` if `fallback` itself is non-finite) whenever inputs are non-finite,
/// `|den| <= 1e-15`, or the quotient fails finiteness checks.
#[axiom_zero]
#[inline]
pub fn total_div_f64(num: f64, den: f64, fallback: f64) -> f64 {
    let safe_fallback = sanitize_f64(fallback);
    if !num.is_finite() || !den.is_finite() || den.abs() <= 1e-15 {
        return safe_fallback;
    }
    let q = checked_ops::div_f64_checked(num, den);
    if q.is_finite() {
        q
    } else {
        safe_fallback
    }
}

/// Total rational normalization `num / (1.0 + norm_sq.abs())` with denominator guaranteed `>= 1.0`.
///
/// Returns `0.0` if either input is non-finite (`NaN`/`Inf`). Because the denominator is strictly
/// bounded below by `1.0`, division by zero and quotient overflow cannot occur for finite inputs.
#[axiom_zero]
#[inline]
pub fn total_div_unit_shifted_f32(num: f32, norm_sq: f32) -> f32 {
    if !num.is_finite() || !norm_sq.is_finite() {
        return 0.0;
    }
    let den = 1.0_f32 + norm_sq.abs();
    if !den.is_finite() || den < 1.0 {
        return 0.0;
    }
    let q = checked_ops::div_f32_checked(num, den);
    if q.is_finite() {
        q
    } else {
        0.0
    }
}

/// Total `f64` rational normalization `num / (1.0 + norm_sq.abs())` with denominator `>= 1.0`.
#[axiom_zero]
#[inline]
pub fn total_div_unit_shifted_f64(num: f64, norm_sq: f64) -> f64 {
    if !num.is_finite() || !norm_sq.is_finite() {
        return 0.0;
    }
    let den = 1.0_f64 + norm_sq.abs();
    if !den.is_finite() || den < 1.0 {
        return 0.0;
    }
    let q = checked_ops::div_f64_checked(num, den);
    if q.is_finite() {
        q
    } else {
        0.0
    }
}

/// Sanitizes an `f32` value by projecting `NaN` and `±Inf` onto `0.0`.
#[axiom_zero]
#[inline]
pub fn sanitize_f32(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Sanitizes an `f64` value by projecting `NaN` and `±Inf` onto `0.0`.
#[axiom_zero]
#[inline]
pub fn sanitize_f64(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Coercive finite amplitude guard for `f32`.
///
/// Absorbs non-finite values (`NaN`, `±Inf`) and single-event extreme finite spikes
/// (`|x| > max_abs`, or invalid `max_abs <= 0`) to `0.0`, preserving uncorrupted values in
/// `[-max_abs, max_abs]` untouched.
#[axiom_zero]
#[inline]
pub fn sanitize_bounded_f32(x: f32, max_abs: f32) -> f32 {
    if !x.is_finite() || !max_abs.is_finite() || max_abs <= 0.0 || x.abs() > max_abs {
        0.0
    } else {
        x
    }
}

/// Coercive finite amplitude guard for `f64`.
#[axiom_zero]
#[inline]
pub fn sanitize_bounded_f64(x: f64, max_abs: f64) -> f64 {
    if !x.is_finite() || !max_abs.is_finite() || max_abs <= 0.0 || x.abs() > max_abs {
        0.0
    } else {
        x
    }
}

/// Applies [`sanitize_bounded_f32`] element-wise across a mutable slice without indexing.
#[axiom_zero]
#[inline]
pub fn sanitize_bounded_slice_f32(data: &mut [f32], max_abs: f32) {
    for val in data.iter_mut() {
        *val = sanitize_bounded_f32(*val, max_abs);
    }
}

/// Bounds-safe slice read with explicit fallback, eliminating out-of-bounds panics.
#[axiom_zero]
#[inline]
pub fn total_get<'a, T>(slice: &'a [T], idx: usize, fallback: &'a T) -> &'a T {
    slice.get(idx).unwrap_or(fallback)
}

/// Bounds-safe slice read by copy with explicit fallback.
#[axiom_zero]
#[inline]
pub fn total_get_copy<T: Copy>(slice: &[T], idx: usize, fallback: T) -> T {
    slice.get(idx).copied().unwrap_or(fallback)
}

/// Bounds-safe slice write without panicking. Returns `true` when `idx < slice.len()`.
#[axiom_zero]
#[inline]
pub fn total_set<T>(slice: &mut [T], idx: usize, value: T) -> bool {
    match slice.get_mut(idx) {
        Some(slot) => {
            *slot = value;
            true
        }
        _ => false,
    }
}

/// Total boolean mask lookup, returning `false` on out-of-bounds access.
#[axiom_zero]
#[inline]
pub fn mask_get(mask: &[bool], idx: usize) -> bool {
    mask.get(idx).copied().unwrap_or(false)
}

/// Total byte-mask lookup (`!= 0`), returning `false` on out-of-bounds access.
#[axiom_zero]
#[inline]
pub fn mask_byte_get(mask: &[u8], idx: usize) -> bool {
    mask.get(idx).copied().unwrap_or(0) != 0
}

/// Extension trait adding total bounds-safe accessors to slices (`[T]`).
pub trait TotalSliceExt<T> {
    /// Returns the element at `idx`, or `fallback` if out of bounds.
    fn total_get<'a>(&'a self, idx: usize, fallback: &'a T) -> &'a T;

    /// Returns `Presence::Live` when `idx` is in bounds, or `Presence::Boundary(boundary)` otherwise.
    fn presence_get(&self, idx: usize, boundary: T) -> Presence<T>
    where
        T: Clone;

    /// Writes `value` into `idx` if in bounds, returning `true` if written and `false` otherwise.
    fn total_set(&mut self, idx: usize, value: T) -> bool;
}

#[axiom_zero]
impl<T> TotalSliceExt<T> for [T] {
    #[inline]
    fn total_get<'a>(&'a self, idx: usize, fallback: &'a T) -> &'a T {
        total_get(self, idx, fallback)
    }

    #[inline]
    fn presence_get(&self, idx: usize, boundary: T) -> Presence<T>
    where
        T: Clone,
    {
        match self.get(idx) {
            Some(v) => Presence::new(v.clone()),
            _ => Presence::boundary(boundary),
        }
    }

    #[inline]
    fn total_set(&mut self, idx: usize, value: T) -> bool {
        total_set(self, idx, value)
    }
}

/// Total Presence container replacing nullable `Option<T>`.
///
/// Guarantees a value is never structurally absent; if a primary state fails validation or hits
/// an edge limit, it transitions into an explicit `Boundary(T)` fallback token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Presence<T> {
    /// Live valid state payload.
    Live(T),
    /// Boundary token preventing null cascades and crashes.
    Boundary(T),
}

#[axiom_zero]
impl<T> Presence<T> {
    /// Wraps a valid payload into `Presence::Live`.
    #[inline]
    pub fn new(value: T) -> Self {
        Presence::Live(value)
    }

    /// Wraps an edge-case fallback into `Presence::Boundary`.
    #[inline]
    pub fn boundary(default_val: T) -> Self {
        Presence::Boundary(default_val)
    }

    /// Converts an external `Option<T>` cleanly into `Presence<T>` without panicking.
    #[inline]
    pub fn from_option(opt: Option<T>, fallback: T) -> Self {
        match opt {
            Some(v) => Presence::Live(v),
            _ => Presence::Boundary(fallback),
        }
    }

    /// Resolves total presence without branching into failure or unwrapping panics.
    #[inline]
    pub fn resolve<F, B, R>(self, live_op: F, boundary_op: B) -> R
    where
        F: FnOnce(T) -> R,
        B: FnOnce(T) -> R,
    {
        match self {
            Presence::Live(val) => live_op(val),
            Presence::Boundary(val) => boundary_op(val),
        }
    }

    /// Transforms an inner state preserving its boundary topological status.
    #[inline]
    pub fn map<F, U>(self, op: F) -> Presence<U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Presence::Live(val) => Presence::Live(op(val)),
            Presence::Boundary(val) => Presence::Boundary(op(val)),
        }
    }

    /// Returns `true` if this container holds a `Live` payload.
    #[inline]
    pub fn is_live(&self) -> bool {
        matches!(self, Presence::Live(_))
    }

    /// Returns `true` if this container holds a `Boundary` fallback token.
    #[inline]
    pub fn is_boundary(&self) -> bool {
        matches!(self, Presence::Boundary(_))
    }

    /// Borrows the contained payload (`Live` or `Boundary`) without panicking.
    #[inline]
    pub fn get(&self) -> &T {
        match self {
            Presence::Live(val) | Presence::Boundary(val) => val,
        }
    }

    /// Mutably borrows the contained payload (`Live` or `Boundary`) without panicking.
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        match self {
            Presence::Live(val) | Presence::Boundary(val) => val,
        }
    }

    /// Extracts the underlying payload (whether `Live` or `Boundary`).
    #[inline]
    pub fn into_inner(self) -> T {
        match self {
            Presence::Live(val) | Presence::Boundary(val) => val,
        }
    }
}

/// Constructive Resource Assimilation trait.
///
/// Instead of dropping resources into empty nothingness, objects reintegrate resources
/// explicitly back into an active pool or parent frame.
pub trait Absorb<Target> {
    /// Assimilate `self` safely into `target`.
    fn absorb(self, target: &mut Target);
}

/// Cyclic Orbital Homeostasis Engine.
///
/// Encapsulates state transitions as a closed cyclical phase orbit guaranteed non-empty by
/// construction (`head` + `tail`), ensuring `period >= 1` (`NonZeroUsize`) without `assert!`,
/// unchecked slice indexing, or raw modulo `%`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attractor<S> {
    head: S,
    tail: Vec<S>,
    current_phase: usize,
    period: NonZeroUsize,
}

#[axiom_zero]
impl<S> Attractor<S> {
    /// Initializes an attractor with a guaranteed non-empty limit-cycle orbit (`head :: tail`).
    ///
    /// Because `head` is always present, `period = 1 + tail.len() >= 1` holds statically with zero
    /// assertions.
    #[inline]
    pub fn new(head: S, tail: Vec<S>) -> Self {
        let len = tail.len().saturating_add(1);
        let period = NonZeroUsize::new(len).unwrap_or(NonZeroUsize::MIN);
        Self {
            head,
            tail,
            current_phase: 0,
            period,
        }
    }

    /// Initializes a 1-state fixed-point attractor orbit.
    #[inline]
    pub fn singleton(head: S) -> Self {
        Self::new(head, Vec::new())
    }

    /// Constructively converts a dynamically sized `Vec<S>` into a non-empty `Attractor<S>`,
    /// falling back to `singleton(fallback)` if `orbit` is empty.
    #[inline]
    pub fn from_vec_or(orbit: Vec<S>, fallback: S) -> Self {
        let mut iter = orbit.into_iter();
        match iter.next() {
            Some(head) => Self::new(head, iter.collect()),
            _ => Self::singleton(fallback),
        }
    }

    /// Constructively converts a dynamically sized slice `&[S]` into a non-empty `Attractor<S>`,
    /// falling back to `singleton(fallback)` if `orbit` is empty.
    #[inline]
    pub fn from_slice_or(orbit: &[S], fallback: S) -> Self
    where
        S: Clone,
    {
        match orbit.split_first() {
            Some((head, tail)) => Self::new(head.clone(), tail.to_vec()),
            _ => Self::singleton(fallback),
        }
    }

    /// Returns the orbital period (`>= 1`).
    #[inline]
    pub fn period(&self) -> usize {
        self.period.get()
    }

    /// Returns the statically guaranteed non-zero orbital period.
    #[inline]
    pub fn period_nonzero(&self) -> NonZeroUsize {
        self.period
    }

    /// Returns the current phase offset within `0..self.period()`.
    #[inline]
    pub fn current_phase(&self) -> usize {
        self.current_phase
    }

    /// Returns the anchor `head` state (phase 0).
    #[inline]
    pub fn head(&self) -> &S {
        &self.head
    }

    /// Returns the remaining orbital tail states (`phases 1..period`).
    #[inline]
    pub fn tail(&self) -> &[S] {
        &self.tail
    }

    /// Totally reads the orbital state at any index via modulo `period >= 1` and bounds-safe access.
    #[inline]
    pub fn get_phase_state(&self, idx: usize) -> &S {
        let rem = total_rem_nonzero(idx, self.period);
        if rem == 0 {
            &self.head
        } else {
            total_get(&self.tail, rem.saturating_sub(1), &self.head)
        }
    }

    /// Returns a reference to the current phase state on the invariant limit cycle.
    #[inline]
    pub fn current_state(&self) -> &S {
        self.get_phase_state(self.current_phase)
    }

    /// Steps cleanly to the next cyclic phase along the attractor orbit.
    #[inline]
    pub fn step(&mut self) -> &S {
        self.current_phase =
            total_rem_nonzero(self.current_phase.saturating_add(1), self.period);
        self.current_state()
    }

    /// Absorbs external system disturbances or invalid perturbations into an orbital phase shift,
    /// maintaining invariant stability in tau = 1 step without thread panics.
    #[inline]
    pub fn absorb_perturbation(&mut self, distortion_shift: usize) -> &S {
        self.current_phase =
            total_rem_nonzero(self.current_phase.wrapping_add(distortion_shift), self.period);
        self.current_state()
    }
}

/// Relational Identity Operator measuring equality (`= 0`) vs distinction (`≠ 0`).
pub struct RelationalIdentity<T>(std::marker::PhantomData<T>);

#[axiom_zero]
impl<T: PartialEq> RelationalIdentity<T> {
    /// Evaluates two entities: returns `Presence::Live(a)` if equal (Zero-Difference),
    /// or `Presence::Boundary(a)` containing an explicit relational separation measure.
    #[inline]
    pub fn measure(a: T, b: T) -> Presence<T> {
        if a == b {
            Presence::new(a)
        } else {
            Presence::boundary(a)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_presence_totality() {
        let live: Presence<i32> = Presence::new(42);
        let bound: Presence<i32> = Presence::boundary(0);

        assert!(live.is_live());
        assert!(!bound.is_live());
        assert_eq!(live.resolve(|x| x + 1, |_| -1), 43);
        assert_eq!(bound.resolve(|x| x + 1, |b| b), 0);
        assert_eq!(Presence::from_option(None, 7), Presence::Boundary(7));
    }

    #[test]
    fn test_attractor_homeostasis_and_nonempty_orbit() {
        let mut attractor = Attractor::new("PHASE_A", vec!["PHASE_B", "PHASE_C"]);
        assert_eq!(attractor.period(), 3);
        assert_eq!(*attractor.current_state(), "PHASE_A");
        assert_eq!(*attractor.step(), "PHASE_B");
        assert_eq!(*attractor.absorb_perturbation(100), "PHASE_C");
        assert_eq!(
            *attractor.get_phase_state(usize::MAX),
            *attractor.get_phase_state(usize::MAX % 3)
        );

        let empty_fallback = Attractor::from_vec_or(Vec::<i32>::new(), 99);
        assert_eq!(empty_fallback.period(), 1);
        assert_eq!(*empty_fallback.current_state(), 99);
    }

    #[test]
    fn test_total_indexing_and_division() {
        let data = [10, 20, 30];
        let fallback = -1;
        assert_eq!(*data.total_get(1, &fallback), 20);
        assert_eq!(*data.total_get(99, &fallback), -1);
        assert_eq!(total_get_copy(&data, 99, 0), 0);

        let mut mut_data = [1, 2, 3];
        assert!(mut_data.total_set(1, 22));
        assert!(!mut_data.total_set(10, 99));
        assert_eq!(mut_data, [1, 22, 3]);

        let mask = [true, false];
        assert!(mask_get(&mask, 0));
        assert!(!mask_get(&mask, 1));
        assert!(!mask_get(&mask, 5));

        assert_eq!(total_div_usize(10, 0, 0), 0);
        assert_eq!(total_div_i64(i64::MIN, -1, 0), 0);
        assert_eq!(total_div_f32(4.0, 0.0, 1.0), 1.0);
        assert_eq!(total_div_f32(f32::NAN, 2.0, -5.0), -5.0);
        assert!((total_div_unit_shifted_f32(3.0, 2.0) - 1.0).abs() < 1e-6);
        assert_eq!(total_div_unit_shifted_f32(f32::INFINITY, 1.0), 0.0);
    }

    #[test]
    fn test_sanitize_bounded_seu_guard() {
        assert_eq!(sanitize_bounded_f32(0.5, 1.0), 0.5);
        assert_eq!(sanitize_bounded_f32(1e25, 1.0), 0.0);
        assert_eq!(sanitize_bounded_f32(-1e30, 1.0), 0.0);
        assert_eq!(sanitize_bounded_f32(f32::MAX, 10.0), 0.0);
        assert_eq!(sanitize_bounded_f32(f32::NAN, 10.0), 0.0);
        assert_eq!(sanitize_bounded_f32(f32::INFINITY, 10.0), 0.0);

        let mut vec = [0.25_f32, 1e25_f32, -0.5_f32, -1e30_f32];
        sanitize_bounded_slice_f32(&mut vec, 1.0);
        assert_eq!(vec, [0.25, 0.0, -0.5, 0.0]);
    }
}
