use axiom_zero::{
    axiom_zero, mask_get, sanitize_bounded_f32, total_div_f32, total_div_unit_shifted_f32,
    total_div_usize, total_get_copy, total_rem_nonzero, Attractor, Presence, RelationalIdentity,
    TotalSliceExt,
};
use std::num::NonZeroUsize;

#[test]
fn test_presence_resolution() {
    let active: Presence<i32> = Presence::new(100);
    let boundary: Presence<i32> = Presence::boundary(-1);

    assert_eq!(active.resolve(|v| v * 2, |_| 0), 200);
    assert_eq!(boundary.resolve(|v| v * 2, |b| b), -1);
}

#[test]
fn test_attractor_orbital_homeostasis() {
    let mut fsm = Attractor::new(10, vec![20, 30, 40]);

    assert_eq!(fsm.period(), 4);
    assert_eq!(*fsm.current_state(), 10);
    assert_eq!(*fsm.step(), 20);
    assert_eq!(*fsm.step(), 30);

    // Absorb arbitrary perturbation index safely without panicking
    let recovered = fsm.absorb_perturbation(1);
    assert_eq!(*recovered, 40);

    // Constructive fallback when constructing from a dynamic empty Vec
    let fallback_fsm = Attractor::from_vec_or(Vec::<i32>::new(), 77);
    assert_eq!(fallback_fsm.period(), 1);
    assert_eq!(*fallback_fsm.current_state(), 77);
}

#[test]
fn test_relational_identity() {
    let same = RelationalIdentity::measure(55, 55);
    let diff = RelationalIdentity::measure(55, 60);

    assert!(same.is_live());
    assert!(!diff.is_live());
}

#[axiom_zero]
fn validated_pipeline_step(values: &[f32], mask: &[bool], idx: usize) -> Presence<f32> {
    if !mask_get(mask, idx) {
        return Presence::boundary(0.0);
    }
    let raw = total_get_copy(values, idx, 0.0);
    let guarded = sanitize_bounded_f32(raw, 100.0);
    let scaled = total_div_unit_shifted_f32(guarded, guarded * guarded);
    let fallback_slice = 0.0_f32;
    let check = *values.total_get(idx, &fallback_slice);
    if check.is_finite() && check.abs() <= 100.0 {
        Presence::new(scaled)
    } else {
        Presence::boundary(0.0)
    }
}

#[test]
fn test_total_helpers_under_strict_macro() {
    let values = [2.0_f32, 1e25_f32, f32::NAN, f32::MAX];
    let mask = [true, true, true, true];

    assert!(validated_pipeline_step(&values, &mask, 0).is_live());
    assert_eq!(validated_pipeline_step(&values, &mask, 1), Presence::Boundary(0.0));
    assert_eq!(validated_pipeline_step(&values, &mask, 2), Presence::Boundary(0.0));
    assert_eq!(validated_pipeline_step(&values, &mask, 3), Presence::Boundary(0.0));
    assert_eq!(validated_pipeline_step(&values, &mask, 99), Presence::Boundary(0.0));

    let nz = NonZeroUsize::new(4).unwrap();
    assert_eq!(total_rem_nonzero(11, nz), 3);
    assert_eq!(total_div_usize(42, 0, 999), 999);
    assert_eq!(total_div_f32(10.0, 0.0, -1.0), -1.0);
}
