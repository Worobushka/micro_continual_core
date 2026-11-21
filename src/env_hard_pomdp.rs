// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

//! Честная среда HardPOMDP v2 (`HardPomdpConfig::strict_honest()`):
//!   1. Глитч возвращает ту же награду и точно ту же строку `"action_failed no_state_change"`, что и промах.
//!   2. `render_hidden_clue()` НЕ раскрывает `sub_tag` — агент обязан помнить `subsystem` с шага 0.
//!   3. `sparse_reward: true` — на промежуточных шагах reward = 0.0, награда (+1.0/-1.0) только в конце.
//!   4. `unlabeled_domain_cloze_logs_noisy()` — перекрывающиеся рамки, шумные токены, дисбаланс 3:1.

use axiom_zero::{axiom_zero, total_rem_usize, Presence};

use crate::math::splitmix64;

pub const NUM_ENV_ACTIONS: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardRegime {
    Alpha = 0,
    Beta = 1,
    Gamma = 2,
    Delta = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParaphraseSplit {
    CanonicalTrain = 0,
    UnseenHoldout = 1,
}

#[derive(Clone, Copy, Debug)]
pub struct HardPomdpConfig {
    pub max_steps: usize,
    pub glitch_pct: u8,
    pub paraphrase_split: ParaphraseSplit,
    pub regime: HardRegime,
    /// Fix 1: глитчи неотличимы от реального промаха по тексту и награде.
    pub indistinguishable_glitch: bool,
    /// Fix 2: убрать sub_tag из render_hidden_clue (истинно скрытая подсистема).
    pub hide_subsystem_after_inspect: bool,
    /// Fix 3: отложенная разреженная награда (0.0 на шагах, +1.0/-1.0 в конце эпизода).
    pub sparse_reward: bool,
}

impl HardPomdpConfig {
    pub fn strict_honest(regime: HardRegime, glitch_pct: u8, split: ParaphraseSplit) -> Self {
        Self {
            max_steps: 6,
            glitch_pct,
            paraphrase_split: split,
            regime,
            indistinguishable_glitch: true,
            hide_subsystem_after_inspect: true,
            sparse_reward: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepOutcome {
    pub observation: String,
    pub reward: f32,
    pub done: bool,
    pub solved: bool,
    pub fatal: bool,
}

#[derive(Clone, Debug)]
pub struct HardPomdpV2 {
    pub seed: u64,
    pub config: HardPomdpConfig,
    pub subsystem: usize, // 0..4
    pub subfault: usize,  // 0..4 (скрыт до Inspect)
    pub step_idx: usize,
    pub inspected: bool,
    pub repaired: bool,
    pub verified: bool,
    pub dirty_rollback: bool,
    pub fatal_trapped: bool,
}

/// Fix 4: зашумлённые неразмеченные Cloze-логи:
/// - общие перекрывающиеся шаблоны между РАЗНЫМИ подсистемами;
/// - случайные мусорные токены;
/// - несбалансированная частота 3:1 (канонических примеров в 3 раза больше, чем holdout).
pub fn unlabeled_domain_cloze_logs_noisy() -> Vec<String> {
    let mut out = Vec::new();
    let shared_frames = [
        ("service daemon noticed", "during production cycle"),
        ("monitor watcher logged", "under cluster load"),
    ];
    let noise_pool = ["xtag", "m_901", "vbuf", "tmpnode", "k_44", "sync_q", "ev_72", "bus_p"];

    let stage1: [(&str, &str, &str, &str); 4] = [
        ("disk array blk stall", "nvme volume sector wait", "block storage driver", "on flush tier"),
        ("compiler rustc type err", "build pipeline ast fault", "crate checker pass", "in trait solver"),
        ("webhook json parse drop", "http ingest malformed body", "gateway proxy filter", "on socket stream"),
        ("ledger tx outlier lock", "audit queue balance freeze", "clearing journal gate", "under variance check"),
    ];

    let mut rng = 0x9911_2233_4455_6677u64;
    let next_noise = |r: &mut u64| -> &'static str {
        *r = splitmix64(*r);
        let idx = ((*r >> 16) as usize) % noise_pool.len();
        noise_pool[idx]
    };

    for (canon, hold, pre_dom, post_dom) in stage1 {
        // Дисбаланс 3:1 (канон x3, holdout x1) + шум + перекрывающиеся общие рамки
        for _ in 0..3 {
            let n1 = next_noise(&mut rng);
            let n2 = next_noise(&mut rng);
            out.push(format!("{} {} {} {} {}", pre_dom, n1, canon, post_dom, n2));
            out.push(format!("{} {} {} {}", shared_frames[0].0, canon, n1, shared_frames[0].1));
        }
        let nh = next_noise(&mut rng);
        out.push(format!("{} {} {} {}", pre_dom, hold, post_dom, nh));
        out.push(format!("{} {} {} {}", shared_frames[1].0, hold, nh, shared_frames[1].1));
    }

    let stage2: [(&str, &str, &str, &str); 4] = [
        ("clue perm denied chmod", "diag access revoked uid", "posix capability gate", "for file descriptor"),
        ("clue mismatch schema field", "diag bad contract payload", "protobuf wire check", "on required attribute"),
        ("clue pool deadlock mutex", "diag thread starve lock", "worker semaphore wait", "across queue"),
        ("clue auth breach signature", "diag spoof token alert", "crypto cert verifier", "from remote session"),
    ];

    for (canon, hold, pre_dom, post_dom) in stage2 {
        for _ in 0..3 {
            let n1 = next_noise(&mut rng);
            out.push(format!("{} {} {} {}", pre_dom, canon, n1, post_dom));
            out.push(format!("{} {} {}", shared_frames[0].0, canon, shared_frames[1].1));
        }
        let nh = next_noise(&mut rng);
        out.push(format!("{} {} {} {}", pre_dom, nh, hold, post_dom));
        out.push(format!("{} {} {}", shared_frames[1].0, hold, shared_frames[0].1));
    }

    // Общие контрольные фразы с шумом и 3:1 дисбалансом
    let ctrl = [
        ("state dirty undo rollback needed", "corrupt buffer revert previous action", "journal unwind", "before retry"),
        ("repair staged run verify suite", "patch committed execute postcheck", "hotfix ready", "for validation"),
    ];
    for (canon, hold, pre_dom, post_dom) in ctrl {
        for _ in 0..3 {
            let n1 = next_noise(&mut rng);
            out.push(format!("{} {} {} {}", pre_dom, canon, n1, post_dom));
        }
        out.push(format!("{} {} {}", pre_dom, hold, post_dom));
    }
    out
}

#[axiom_zero]
impl HardPomdpV2 {
    pub fn generate(seed: u64, config: HardPomdpConfig) -> Self {
        let h1 = splitmix64(seed ^ 0xA11C_E001_55AA_9933);
        let h2 = splitmix64(h1.wrapping_add(0x9E37_79B9_7F4A_7C15));
        let subsystem = total_rem_usize(h1 as usize, 4, 0);
        let subfault = total_rem_usize((h2 >> 8) as usize, 4, 0);
        Self {
            seed,
            config,
            subsystem,
            subfault,
            step_idx: 0,
            inspected: false,
            repaired: false,
            verified: false,
            dirty_rollback: false,
            fatal_trapped: false,
        }
    }

    fn noise_tag(&self, salt: u64) -> String {
        let h = splitmix64(self.seed ^ salt.wrapping_mul(0x517c_c1b7_2722_0a95));
        let n1 = total_rem_usize(h as usize, 899, 0) + 100;
        let tag = match total_rem_usize((h >> 16) as usize, 4, 0) {
            0 => "rack",
            1 => "shard",
            2 => "lane",
            _ => "port",
        };
        format!("{}_{}", tag, n1)
    }

    pub fn initial_observation(&self) -> String {
        let base = match (self.config.paraphrase_split, self.subsystem) {
            (ParaphraseSplit::CanonicalTrain, 0) => "node alert disk array blk stall",
            (ParaphraseSplit::CanonicalTrain, 1) => "node alert compiler rustc type err",
            (ParaphraseSplit::CanonicalTrain, 2) => "node alert webhook json parse drop",
            (ParaphraseSplit::CanonicalTrain, _) => "node alert ledger tx outlier lock",
            (ParaphraseSplit::UnseenHoldout, 0) => "telemetry nvme volume sector wait high",
            (ParaphraseSplit::UnseenHoldout, 1) => "telemetry build pipeline ast fault trigger",
            (ParaphraseSplit::UnseenHoldout, 2) => "telemetry http ingest malformed body warn",
            (ParaphraseSplit::UnseenHoldout, _) => "telemetry audit queue balance freeze hold",
        };
        format!("{} {}", base, self.noise_tag(17))
    }

    /// Fix 2: при `hide_subsystem_after_inspect == true` НЕ выдаём `sub_tag`!
    pub fn render_hidden_clue(&self) -> String {
        let clue = match (self.config.paraphrase_split, self.subfault) {
            (ParaphraseSplit::CanonicalTrain, 0) => "clue perm denied chmod",
            (ParaphraseSplit::CanonicalTrain, 1) => "clue mismatch schema field",
            (ParaphraseSplit::CanonicalTrain, 2) => "clue pool deadlock mutex",
            (ParaphraseSplit::CanonicalTrain, _) => "clue auth breach signature",
            (ParaphraseSplit::UnseenHoldout, 0) => "diag access revoked uid",
            (ParaphraseSplit::UnseenHoldout, 1) => "diag bad contract payload",
            (ParaphraseSplit::UnseenHoldout, 2) => "diag thread starve lock",
            (ParaphraseSplit::UnseenHoldout, _) => "diag spoof token alert",
        };
        if self.config.hide_subsystem_after_inspect {
            format!("{} {}", clue, self.noise_tag(41))
        } else {
            let sub_tag = match (self.config.paraphrase_split, self.subsystem) {
                (ParaphraseSplit::CanonicalTrain, 0) => "clusternorth",
                (ParaphraseSplit::CanonicalTrain, 1) => "clustereast",
                (ParaphraseSplit::CanonicalTrain, 2) => "clustersouth",
                (ParaphraseSplit::CanonicalTrain, _) => "clusterwest",
                (ParaphraseSplit::UnseenHoldout, 0) => "zonealpha",
                (ParaphraseSplit::UnseenHoldout, 1) => "zonebeta",
                (ParaphraseSplit::UnseenHoldout, 2) => "zonegamma",
                (ParaphraseSplit::UnseenHoldout, _) => "zonedelta",
            };
            format!("{} {} {}", clue, sub_tag, self.noise_tag(41))
        }
    }

    fn render_dirty_notice(&self) -> String {
        match self.config.paraphrase_split {
            ParaphraseSplit::CanonicalTrain => "state dirty undo rollback needed".to_string(),
            ParaphraseSplit::UnseenHoldout => {
                format!("corrupt buffer revert previous action {}", self.noise_tag(91))
            }
        }
    }

    fn render_repaired_notice(&self) -> String {
        match self.config.paraphrase_split {
            ParaphraseSplit::CanonicalTrain => "repair staged run verify suite".to_string(),
            ParaphraseSplit::UnseenHoldout => {
                format!("patch committed execute postcheck {}", self.noise_tag(107))
            }
        }
    }

    pub fn expected_inspect(&self) -> usize {
        match (self.config.regime, self.subsystem) {
            (HardRegime::Alpha, s) => total_rem_usize(s, 5, 0),
            (HardRegime::Beta, s) => total_rem_usize(s + 2, 5, 0),
            (HardRegime::Gamma, s) => {
                total_rem_usize(4usize.saturating_sub(total_rem_usize(s, 4, 0)), 5, 0)
            }
            (HardRegime::Delta, s) => total_rem_usize(s * 2 + 1, 5, 0),
        }
    }

    pub fn expected_repair(&self) -> usize {
        let reg_shift = match self.config.regime {
            HardRegime::Alpha => 0usize,
            HardRegime::Beta => 1usize,
            HardRegime::Gamma => 3usize,
            HardRegime::Delta => 2usize,
        };
        5 + total_rem_usize(self.subsystem + self.subfault * 2 + reg_shift, 5, 0)
    }

    pub fn lethal_trap_repair(&self) -> usize {
        let exp = self.expected_repair().saturating_sub(5);
        let trap_off = 1 + total_rem_usize(self.subsystem + self.subfault, 3, 0);
        5 + total_rem_usize(exp + trap_off, 5, 0)
    }

    /// Используется ТОЛЬКО для Oracle-бейзлайна (никогда не передаётся агенту!).
    pub fn oracle_action(&self) -> usize {
        if self.dirty_rollback {
            11 // Undo
        } else if !self.inspected {
            self.expected_inspect()
        } else if !self.repaired {
            self.expected_repair()
        } else {
            10 // Verify
        }
    }

    fn is_glitch_step(&self, action: usize) -> bool {
        if self.config.glitch_pct == 0 || action == 11 {
            return false;
        }
        let h = splitmix64(
            self.seed
                ^ ((self.step_idx as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03))
                ^ ((action as u64).wrapping_add(11)),
        );
        total_rem_usize(h as usize, 100, 0) < (self.config.glitch_pct as usize)
    }

    #[inline]
    fn shape_reward(&self, dense_r: f32, done: bool, solved: bool) -> f32 {
        if !self.config.sparse_reward {
            return dense_r;
        }
        if !done {
            0.0
        } else if solved {
            1.0
        } else {
            -1.0
        }
    }

    pub fn step(&mut self, action: usize) -> Presence<StepOutcome> {
        if self.fatal_trapped || self.verified || self.step_idx >= self.config.max_steps {
            return Presence::boundary(StepOutcome {
                observation: "episode terminated".to_string(),
                reward: self.shape_reward(-0.5, true, self.verified),
                done: true,
                solved: self.verified,
                fatal: self.fatal_trapped,
            });
        }
        self.step_idx += 1;
        let max_hit = self.step_idx >= self.config.max_steps;

        // Dirty rollback: агент сам обязан выдать Undo (11)
        if self.dirty_rollback {
            if action == 11 {
                self.dirty_rollback = false;
                let obs = if !self.inspected {
                    self.initial_observation()
                } else {
                    self.render_hidden_clue()
                };
                return Presence::new(StepOutcome {
                    observation: obs,
                    reward: self.shape_reward(0.15, max_hit, false),
                    done: max_hit,
                    solved: false,
                    fatal: false,
                });
            }
            return Presence::boundary(StepOutcome {
                observation: self.render_dirty_notice(),
                reward: self.shape_reward(-0.5, max_hit, false),
                done: max_hit,
                solved: false,
                fatal: false,
            });
        }

        // Fix 1: честный неразличимый глитч!
        if self.is_glitch_step(action) {
            if self.config.indistinguishable_glitch {
                let dense_pen = if action >= 5 && action < 10 { -0.55 } else { -0.25 };
                return Presence::new(StepOutcome {
                    observation: "action_failed no_state_change".to_string(),
                    reward: self.shape_reward(dense_pen, max_hit, false),
                    done: max_hit,
                    solved: false,
                    fatal: false,
                });
            } else {
                return Presence::new(StepOutcome {
                    observation: format!(
                        "transient_bus_glitch packet_drop retry {}",
                        self.noise_tag(self.step_idx as u64 + 5)
                    ),
                    reward: self.shape_reward(-0.15, max_hit, false),
                    done: max_hit,
                    solved: false,
                    fatal: false,
                });
            }
        }

        // Фаза 1: Inspect (0..5)
        if action < 5 {
            if !self.inspected && action == self.expected_inspect() {
                self.inspected = true;
                return Presence::new(StepOutcome {
                    observation: self.render_hidden_clue(),
                    reward: self.shape_reward(0.35, max_hit, false),
                    done: max_hit,
                    solved: false,
                    fatal: false,
                });
            }
            let obs = if self.config.indistinguishable_glitch {
                "action_failed no_state_change".to_string()
            } else {
                format!("inspect_mismatch empty_probe {}", self.initial_observation())
            };
            return Presence::new(StepOutcome {
                observation: obs,
                reward: self.shape_reward(-0.25, max_hit, false),
                done: max_hit,
                solved: false,
                fatal: false,
            });
        }

        // Фаза 2: Repair (5..10)
        if action < 10 {
            if !self.inspected {
                if (self.subfault & 1) == 1 {
                    self.fatal_trapped = true;
                    return Presence::boundary(StepOutcome {
                        observation: "fatal_corruption episode_abort".to_string(),
                        reward: self.shape_reward(-1.5, true, false),
                        done: true,
                        solved: false,
                        fatal: true,
                    });
                }
                self.dirty_rollback = true;
                return Presence::boundary(StepOutcome {
                    observation: self.render_dirty_notice(),
                    reward: self.shape_reward(-0.55, max_hit, false),
                    done: max_hit,
                    solved: false,
                    fatal: false,
                });
            }
            if !self.repaired && action == self.expected_repair() {
                self.repaired = true;
                return Presence::new(StepOutcome {
                    observation: self.render_repaired_notice(),
                    reward: self.shape_reward(0.45, max_hit, false),
                    done: max_hit,
                    solved: false,
                    fatal: false,
                });
            }
            if !self.repaired && action == self.lethal_trap_repair() {
                self.fatal_trapped = true;
                return Presence::boundary(StepOutcome {
                    observation: "fatal_trap irrecoverable_state episode_abort".to_string(),
                    reward: self.shape_reward(-1.5, true, false),
                    done: true,
                    solved: false,
                    fatal: true,
                });
            }
            self.dirty_rollback = true;
            let obs = if self.config.indistinguishable_glitch {
                "action_failed no_state_change".to_string()
            } else {
                self.render_dirty_notice()
            };
            return Presence::boundary(StepOutcome {
                observation: obs,
                reward: self.shape_reward(-0.55, max_hit, false),
                done: max_hit,
                solved: false,
                fatal: false,
            });
        }

        // Фаза 3: Verify (10)
        if action == 10 {
            if self.inspected && self.repaired {
                self.verified = true;
                return Presence::new(StepOutcome {
                    observation: "verification_passed_complete".to_string(),
                    reward: self.shape_reward(1.0, true, true),
                    done: true,
                    solved: true,
                    fatal: false,
                });
            }
            let obs = if self.config.indistinguishable_glitch {
                "action_failed no_state_change".to_string()
            } else {
                "verify_failed_precondition".to_string()
            };
            return Presence::boundary(StepOutcome {
                observation: obs,
                reward: self.shape_reward(-0.25, max_hit, false),
                done: max_hit,
                solved: false,
                fatal: false,
            });
        }

        // Undo (11) в чистом состоянии
        Presence::new(StepOutcome {
            observation: "action_failed no_state_change".to_string(),
            reward: self.shape_reward(-0.25, max_hit, false),
            done: max_hit,
            solved: false,
            fatal: false,
        })
    }
}
