// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Жёсткий аудит нуля аллокаций на куче для `UnifiedCayleyCore` (`GlobalAlloc`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use micro_continual_core::{AdaptiveOrthogonalRtrl, UnifiedCayleyConfig, UnifiedCayleyCore};

struct CountingAlloc;

static TEST_LOCK: Mutex<()> = Mutex::new(());
static TRACKING: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[test]
fn test_unified_zero_heap_allocations_on_hot_paths() {
    let _guard = TEST_LOCK.lock().unwrap();
    let mut cls = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config(4, 6, 42));
    let mut cls_v050 = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config_v050(4, 6, 45));
    let mut opt_cfg = UnifiedCayleyConfig::one_config(4, 6, 46);
    opt_cfg.enable_bank_rotor_adapter = true;
    opt_cfg.enable_finsler_diode = true;
    opt_cfg.enable_shared_actor_rms = true;
    let mut cls_opt = UnifiedCayleyCore::new(opt_cfg);
    cls_opt.active_slot = 1;
    let mut reg = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config(2, 6, 43));
    let mut rl = UnifiedCayleyCore::new(UnifiedCayleyConfig::one_config(3, 4, 44));

    let x6 = [0.4_f32, -0.3, 0.8, -0.1, 0.2, 0.5];
    let x4 = [0.2_f32, -0.7, 0.1, 0.9];
    let seq: [&[f32]; 4] = [&x6, &x6, &x6, &x6];

    // Прогрев
    cls.start_episode();
    cls_v050.start_episode();
    cls_opt.start_episode();
    reg.start_episode();
    rl.start_episode();
    let _ = cls.step_supervised_class(&x6, 1, true, true, false);
    let _ = cls_v050.step_supervised_class(&x6, 1, true, true, false);
    let _ = cls_opt.step_supervised_class(&x6, 1, true, true, false);
    let _ = reg.step_supervised_reg(&x6, 0.5, true, true, false);
    let _ = rl.decide_rl(&x4, true, false, false);
    rl.observe_rl_reward(0.25, &[], false, true);
    let _ = cls.predict_class_readonly(&seq);
    let _ = cls_v050.predict_class_readonly(&seq);
    let _ = cls_opt.predict_class_readonly(&seq);

    ALLOC_COUNT.store(0, Ordering::SeqCst);
    TRACKING.store(true, Ordering::SeqCst);

    for step in 0..1000usize {
        if step % 50 == 0 {
            cls.start_episode();
            cls_v050.start_episode();
            cls_opt.start_episode();
            reg.start_episode();
            rl.start_episode();
        }
        let _ = cls.step_supervised_class(&x6, step % 4, true, true, true);
        let _ = cls_v050.step_supervised_class(&x6, step % 4, true, true, true);
        let _ = cls_opt.step_supervised_class(&x6, step % 4, true, true, true);
        let _ = reg.step_supervised_reg(&x6, 0.4, true, true, true);
        let _ = rl.decide_rl(&x4, true, false, true);
        rl.observe_rl_reward(0.1, &[], (step % 50) == 49, true);
        let _ = cls.predict_class_readonly(&seq);
        let _ = cls_v050.predict_class_readonly(&seq);
        let _ = cls_opt.predict_class_readonly(&seq);
        if (step % 50) == 49 {
            let _ = rl.end_episode_score(0.8, true);
        }
    }

    TRACKING.store(false, Ordering::SeqCst);
    let count = ALLOC_COUNT.load(Ordering::SeqCst);
    assert_eq!(count, 0, "Expected 0 heap allocations in hot path, got {count}");
}

#[test]
fn test_adaptive_od_zero_heap_allocations_on_hot_paths() {
    let _guard = TEST_LOCK.lock().unwrap();
    let mut model = AdaptiveOrthogonalRtrl::new(2, 2, 120, 42);
    let _ = model.upgrade_dimension_isometric();
    let _ = model.upgrade_dimension_isometric();
    let x0 = [1.0_f32, 0.0];
    let x1 = [0.0_f32, 1.0];
    let seq: [&[f32]; 6] = [&x0, &x1, &x0, &x1, &x0, &x1];

    // Прогрев
    model.reset();
    let _ = model.step_supervised(&x0, true, 3, 0.03, 0.05);
    let _ = model.predict_k_ahead_readonly(&seq);

    ALLOC_COUNT.store(0, Ordering::SeqCst);
    TRACKING.store(true, Ordering::SeqCst);

    for ep in 0..40usize {
        model.reset();
        for t in 0..6usize {
            let inp: &[f32] = if ((ep + t) & 1) == 0 { &x0 } else { &x1 };
            let _ = model.step_supervised(inp, t == 5, ep % 120, 0.03, 0.05);
        }
        let _ = model.predict_k_ahead_readonly(&seq);
        let _ = model.predict_current_readonly();
    }

    TRACKING.store(false, Ordering::SeqCst);
    let count = ALLOC_COUNT.load(Ordering::SeqCst);
    assert_eq!(
        count, 0,
        "Expected 0 heap allocations in AdaptiveOrthogonalRtrl hot path, got {count}"
    );
}
