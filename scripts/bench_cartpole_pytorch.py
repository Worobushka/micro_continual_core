#!/usr/bin/env python3
# Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov & Sofia Sidorenko
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
External PyTorch RL Verification for Position-Only CartPole A->B->A
==================================================================
Exact physics & protocol mirror of `src/bin/bench_step3_pomdp_rl.rs` (`CartPoleEnv`):
  - Position-only POMDP: agent observes only (x / 2.4, theta / 0.2095) [2D],
    or with causal scaled finite difference (+dx -> 4D).
  - 3 continual phases with sudden actuator inversion:
      Phase A: 140 episodes (force_sign = +1.0)
      Phase B: 140 episodes (force_sign = -1.0)
      Phase C:  80 episodes (force_sign = +1.0)  [Retention]
  - Compares external PyTorch optimizers (torch.optim.Adam):
      1. Online GRU (H=32) + Actor-Critic (episodic BPTT + GAE + entropy)
      2. Online GRU-DQN (H=32) + Experience Replay Buffer (cap=2000)

Usage:
  python scripts/bench_cartpole_pytorch.py          # default 10 seeds (pure 2D & +dx)
  python scripts/bench_cartpole_pytorch.py --quick  # 2-seed smoke test
"""

import argparse
import json
import math
import random
import statistics
import time
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(f"PyTorch is required to run bench_cartpole_pytorch.py: {exc}")


class CartPoleEnv:
    """Exact mirror of Rust CartPoleEnv in src/bin/bench_step3_pomdp_rl.rs."""

    def __init__(self, rng: random.Random, force_sign: float):
        self.x = rng.uniform(-0.05, 0.05)
        self.x_dot = rng.uniform(-0.05, 0.05)
        self.theta = rng.uniform(-0.05, 0.05)
        self.theta_dot = rng.uniform(-0.05, 0.05)
        self.steps = 0
        self.force_sign = force_sign

    def obs(self):
        return (self.x / 2.4, self.theta / 0.2095)

    def step(self, action: int) -> bool:
        g = 9.8
        mass_cart = 1.0
        mass_pole = 0.1
        total_mass = mass_cart + mass_pole
        length = 0.5
        polemass_length = mass_pole * length
        force_mag = 10.0
        tau = 0.02

        force = (force_mag if action == 1 else -force_mag) * self.force_sign
        costh = math.cos(self.theta)
        sinth = math.sin(self.theta)

        temp = (force + polemass_length * self.theta_dot * self.theta_dot * sinth) / total_mass
        theta_acc = (g * sinth - costh * temp) / (
            length * (4.0 / 3.0 - mass_pole * costh * costh / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * costh / total_mass

        self.x += tau * self.x_dot
        self.x_dot += tau * x_acc
        self.theta += tau * self.theta_dot
        self.theta_dot += tau * theta_acc
        self.steps += 1

        return abs(self.x) > 2.4 or abs(self.theta) > 0.2095 or self.steps >= 200


class GruActorCritic(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)

    def step(self, x_t: torch.Tensor, h: torch.Tensor):
        out, h_next = self.gru(x_t.view(1, 1, -1), h)
        feat = out.squeeze(0).squeeze(0)
        return self.actor(feat), self.critic(feat).squeeze(-1), h_next

    def forward_seq(self, seq: torch.Tensor):
        out, _ = self.gru(seq.unsqueeze(0))
        feat = out.squeeze(0)
        return self.actor(feat), self.critic(feat).squeeze(-1)


class WindowGruDqn(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.q_head = nn.Linear(hidden_dim, 2)

    def forward(self, seq_batch: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(seq_batch)
        return self.q_head(out[:, -1, :])


def make_feat(obs2, prev2, use_dx: bool):
    if not use_dx:
        return [obs2[0], obs2[1]], obs2
    if prev2 is None:
        return [obs2[0], obs2[1], 0.0, 0.0], obs2
    dx0 = max(-4.0, min(4.0, (obs2[0] - prev2[0]) * 10.0))
    dx1 = max(-4.0, min(4.0, (obs2[1] - prev2[1]) * 10.0))
    return [obs2[0], obs2[1], dx0, dx1], obs2


def tail_40(vals):
    k = min(40, len(vals))
    return sum(vals[-k:]) / max(1, k)


def recovery_window(vals, target=50.0, win=15):
    if len(vals) < win:
        return float(len(vals))
    for i in range(len(vals) - win + 1):
        if sum(vals[i : i + win]) / win >= target:
            return float(i + win)
    return float(len(vals))


def run_gru_ac_seed(seed: int, use_dx: bool, phases=(140, 140, 80)):
    torch.manual_seed(seed)
    rng = random.Random(seed ^ 0xCA27901E)
    in_dim = 4 if use_dx else 2
    net = GruActorCritic(in_dim, hidden_dim=32)
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    gamma = 0.95
    gae_lam = 0.90

    phase_scores = []
    t0 = time.perf_counter()
    total_steps = 0

    for sign, n_eps in zip((1.0, -1.0, 1.0), phases):
        surv = []
        for _ in range(n_eps):
            env = CartPoleEnv(rng, sign)
            prev = None
            h = torch.zeros(1, 1, 32)
            feats, actions, rewards = [], [], []
            while True:
                o = env.obs()
                f, prev = make_feat(o, prev, use_dx)
                x_t = torch.tensor(f, dtype=torch.float32)
                with torch.no_grad():
                    logits, _, h = net.step(x_t, h)
                    probs = F.softmax(logits, dim=-1)
                    act = int(torch.multinomial(probs, 1).item())
                done = env.step(act)
                step_r = -1.0 if (done and env.steps < 200) else 0.06
                feats.append(f)
                actions.append(act)
                rewards.append(step_r)
                total_steps += 1
                if done:
                    break

            seq = torch.tensor(feats, dtype=torch.float32)
            logits_seq, values_seq = net.forward_seq(seq)
            with torch.no_grad():
                v_np = values_seq.tolist()
                next_v = v_np[-1] if env.steps >= 200 else 0.0
                gae = 0.0
                advs = [0.0] * len(rewards)
                rets = [0.0] * len(rewards)
                for idx in range(len(rewards) - 1, -1, -1):
                    nv = next_v if idx == len(rewards) - 1 else v_np[idx + 1]
                    delta = rewards[idx] + gamma * nv - v_np[idx]
                    gae = delta + gamma * gae_lam * gae
                    advs[idx] = gae
                    rets[idx] = v_np[idx] + gae
                adv_t = torch.tensor(advs, dtype=torch.float32)
                if adv_t.numel() > 1:
                    adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-6)
                ret_t = torch.tensor(rets, dtype=torch.float32)

            log_probs = F.log_softmax(logits_seq, dim=-1)
            probs = log_probs.exp()
            act_t = torch.tensor(actions, dtype=torch.long)
            chosen_lp = log_probs.gather(1, act_t.unsqueeze(1)).squeeze(1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            policy_loss = -(chosen_lp * adv_t).mean()
            value_loss = 0.5 * F.mse_loss(values_seq, ret_t)
            loss = policy_loss + value_loss - 0.01 * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            surv.append(float(env.steps))
        phase_scores.append(surv)

    elapsed = time.perf_counter() - t0
    sa, sb, sc = phase_scores
    params = sum(p.numel() for p in net.parameters())
    return {
        "score": 0.5 * (tail_40(sa) + tail_40(sb)),
        "retention": tail_40(sc),
        "recovery": recovery_window(sb, 50.0, 15),
        "params": params,
        "us_per_step": (elapsed * 1e6) / max(1, total_steps),
    }


def run_gru_dqn_replay_seed(seed: int, use_dx: bool, phases=(140, 140, 80)):
    torch.manual_seed(seed)
    rng = random.Random(seed ^ 0xD0011001)
    in_dim = 4 if use_dx else 2
    q_net = WindowGruDqn(in_dim, hidden_dim=32)
    t_net = WindowGruDqn(in_dim, hidden_dim=32)
    t_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=0.002)
    replay = deque(maxlen=2000)

    win_len = 8
    gamma = 0.95
    batch_size = 32
    eps = 0.15
    total_steps = 0
    phase_scores = []
    t0 = time.perf_counter()

    def pad_window(hist):
        sub = hist[-win_len:]
        if len(sub) < win_len:
            pad = [[0.0] * in_dim for _ in range(win_len - len(sub))]
            return pad + sub
        return list(sub)

    for sign, n_eps in zip((1.0, -1.0, 1.0), phases):
        surv = []
        for _ in range(n_eps):
            env = CartPoleEnv(rng, sign)
            prev = None
            hist = []
            o = env.obs()
            f, prev = make_feat(o, prev, use_dx)
            hist.append(f)
            win = pad_window(hist)
            while True:
                if rng.random() < eps:
                    act = rng.randrange(2)
                else:
                    with torch.no_grad():
                        qs = q_net(torch.tensor([win], dtype=torch.float32))
                        act = int(qs.argmax(dim=-1).item())
                done = env.step(act)
                step_r = -1.0 if (done and env.steps < 200) else 0.06
                total_steps += 1
                if done:
                    next_win = win
                else:
                    no = env.obs()
                    nf, prev = make_feat(no, prev, use_dx)
                    hist.append(nf)
                    next_win = pad_window(hist)
                replay.append((win, act, step_r, next_win, 1.0 if (done and env.steps < 200) else 0.0))
                win = next_win

                if len(replay) >= batch_size and (total_steps % 4 == 0):
                    batch = rng.sample(replay, batch_size)
                    s_b = torch.tensor([b[0] for b in batch], dtype=torch.float32)
                    a_b = torch.tensor([b[1] for b in batch], dtype=torch.long)
                    r_b = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                    ns_b = torch.tensor([b[3] for b in batch], dtype=torch.float32)
                    d_b = torch.tensor([b[4] for b in batch], dtype=torch.float32)
                    q_pred = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        q_next = t_net(ns_b).max(dim=-1).values
                        q_tgt = r_b + gamma * (1.0 - d_b) * q_next
                    loss = F.smooth_l1_loss(q_pred, q_tgt)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                    opt.step()

                if total_steps % 200 == 0:
                    t_net.load_state_dict(q_net.state_dict())
                if done:
                    break
            surv.append(float(env.steps))
        phase_scores.append(surv)

    elapsed = time.perf_counter() - t0
    sa, sb, sc = phase_scores
    params = sum(p.numel() for p in q_net.parameters())
    return {
        "score": 0.5 * (tail_40(sa) + tail_40(sb)),
        "retention": tail_40(sc),
        "recovery": recovery_window(sb, 50.0, 15),
        "params": params,
        "us_per_step": (elapsed * 1e6) / max(1, total_steps),
    }


def summarize(rows):
    sc = [r["score"] for r in rows]
    rt = [r["retention"] for r in rows]
    rc = [r["recovery"] for r in rows]
    lt = [r["us_per_step"] for r in rows]
    sd = statistics.pstdev(sc) if len(sc) > 1 else 0.0
    rtsd = statistics.pstdev(rt) if len(rt) > 1 else 0.0
    return {
        "score_mean": statistics.mean(sc),
        "score_sd": sd,
        "ret_mean": statistics.mean(rt),
        "ret_sd": rtsd,
        "rec_mean": statistics.mean(rc),
        "us_mean": statistics.mean(lt),
        "params": rows[0]["params"],
    }


def main():
    parser = argparse.ArgumentParser(description="PyTorch CartPole A->B->A POMDP Benchmark")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds (default: 10)")
    parser.add_argument("--quick", action="store_true", help="Run 2-seed smoke test")
    parser.add_argument("--json", type=str, default="", help="Optional JSON output path")
    args = parser.parse_args()

    n_seeds = 2 if args.quick else max(1, args.seeds)
    seeds = list(range(1, n_seeds + 1))

    torch.set_num_threads(1)
    print(f"=== PyTorch External Verification: Position-Only CartPole A->B->A ({n_seeds} seeds, torch {torch.__version__}) ===")
    print(f"{'Model':<32} | {'Score (mean±sd)':<17} | {'Retent. (mean±sd)':<17} | {'Recov.':<9} | {'Params':<7} | {'Step Lat'}")
    print("-" * 105)

    results = {}
    configs = [
        ("PyTorch GRU A2C (pure 2D, H=32)", False, run_gru_ac_seed),
        ("PyTorch GRU A2C (+dx 4D, H=32)", True, run_gru_ac_seed),
        ("PyTorch DQN+Replay (pure 2D)", False, run_gru_dqn_replay_seed),
        ("PyTorch DQN+Replay (+dx 4D)", True, run_gru_dqn_replay_seed),
    ]
    for label, use_dx, fn in configs:
        rows = [fn(s, use_dx) for s in seeds]
        agg = summarize(rows)
        results[label] = agg
        print(
            f"{label:<32} | {agg['score_mean']:6.1f}±{agg['score_sd']:4.1f} st     | "
            f"{agg['ret_mean']:6.1f}±{agg['ret_sd']:4.1f} st     | "
            f"{agg['rec_mean']:6.1f} ep | {agg['params']:<7d} | {agg['us_mean']:7.1f} µs"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
