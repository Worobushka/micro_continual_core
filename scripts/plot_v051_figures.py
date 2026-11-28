#!/usr/bin/env python3
# Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov & Sofia Sidorenko
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
MicroContinualCore v0.5.1 — Publication Figure Suite (Clean Sanitized Release).

Generates the 3 publication-grade figures for v0.5.1 directly in `assets/`
(plus print-ready vector PDF in `assets/figures/paper/` and dark copies in
`assets/figures/dark/`):
  1. `assets/v051_o4_orbits_and_watershed.png`
     - Exact rational Cayley A_5 orbits on S^2, O(4)->SO(5) 4-HH parity watershed
       (||u_3(beta)||^2 & det_{4x4}) with empirical S_5 beta discovery, optional P3
       Finsler-Randers normalized contrast c(v, delta) in (-1, 1) (step137), and P4
       Lyapunov gate + rational Liebel-Koerner channel weights w(s) = 1.25/(1+s^2) (step138).
  2. `assets/v051_unified_ablation_ladder.png`
     - UnifiedCayleyCore v0.5.0 B3 vs Clean v0.5.1 One-Config (Band0(4) + smooth_var_norm
       + idbd_binary_fix + P1 J-XOR + P4 Lyap/LK; P2 & P3 LOO-filtered OFF) across all 8
       Suite B tasks + bit-serial Add mod 997/9973 + CartPole 2D/+dx + NASA C-MAPSS + RAM.
  3. `assets/v051_adaptive_od_trajectories.png`
     - Live 10-seed AdaptiveOrthogonalRtrl (placebo-free P5) d(t) escalation trajectories
       (Label-Shuffle, C_5, A_5, S_5, S_6), v0.5.0 vs Clean v0.5.1 P5 final d vs d_min,
       and L=6..500 extrapolation curves + blind OOD summary.

All empirical values and 10-seed escalation timestamps match live `--release`
outputs from `bench_step8_v050 --all --dump-traj` and `bench_step9_prime_arith`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
OUT_PAPER = ASSETS_DIR / "figures" / "paper"
OUT_DARK = ASSETS_DIR / "figures" / "dark"

# Exact 10-seed Suite D escalation events exported by clean sanitized v0.5.1:
# `cargo run --release --bin bench_step8_v050 -- --suite d --dump-traj <path>`
DEFAULT_SUITE_D_TRAJ = {
    "C_5": {
        "total_eps": 2500,
        "seeds": [{"final_dim": 2, "events": []} for _ in range(10)],
    },
    "A_5": {
        "total_eps": 6000,
        "seeds": [
            {"final_dim": 3, "events": [[1400, 3]]},
            {"final_dim": 3, "events": [[1600, 3]]},
            {"final_dim": 3, "events": [[2200, 3]]},
            {"final_dim": 3, "events": [[1200, 3]]},
            {"final_dim": 3, "events": [[1200, 3]]},
            {"final_dim": 3, "events": [[1200, 3]]},
            {"final_dim": 3, "events": [[1200, 3]]},
            {"final_dim": 4, "events": [[1200, 3], [3400, 4]]},
            {"final_dim": 3, "events": [[1000, 3]]},
            {"final_dim": 3, "events": [[1400, 3]]},
        ],
    },
    "S_5": {
        "total_eps": 12000,
        "seeds": [
            {"final_dim": 5, "events": [[2910, 3], [4692, 4], [10452, 5]]},
            {"final_dim": 4, "events": [[1530, 3], [6570, 4]]},
            {"final_dim": 4, "events": [[1524, 3], [4758, 4]]},
            {"final_dim": 4, "events": [[2985, 3], [7665, 4]]},
            {"final_dim": 4, "events": [[1890, 3], [5844, 4]]},
            {"final_dim": 5, "events": [[3266, 3], [5057, 4], [10097, 5]]},
            {"final_dim": 4, "events": [[4683, 3], [6480, 4]]},
            {"final_dim": 4, "events": [[2658, 3], [7335, 4]]},
            {"final_dim": 4, "events": [[2229, 3], [5091, 4]]},
            {"final_dim": 4, "events": [[1488, 3], [3999, 4]]},
        ],
    },
    "S_6": {
        "total_eps": 48000,
        "seeds": [
            {"final_dim": 5, "events": [[3000, 3], [36000, 4], [41000, 5]]},
            {"final_dim": 5, "events": [[3000, 3], [20000, 4], [25000, 5]]},
            {"final_dim": 5, "events": [[7000, 3], [12000, 4], [23000, 5]]},
            {"final_dim": 5, "events": [[3000, 3], [32000, 4], [47000, 5]]},
            {"final_dim": 6, "events": [[10000, 3], [20000, 4], [25000, 5], [39000, 6]]},
            {"final_dim": 6, "events": [[4000, 3], [13000, 4], [18000, 5], [24000, 6]]},
            {"final_dim": 6, "events": [[3000, 3], [12000, 4], [20000, 5], [29000, 6]]},
            {"final_dim": 5, "events": [[7000, 3], [13000, 4], [21000, 5]]},
            {"final_dim": 6, "events": [[3000, 3], [11000, 4], [19000, 5], [29000, 6]]},
            {"final_dim": 6, "events": [[7000, 3], [18000, 4], [29000, 5], [38000, 6]]},
        ],
    },
    "Label_Shuffle": {
        "total_eps": 12000,
        "seeds": [{"final_dim": 2, "events": []} for _ in range(10)],
    },
}

THEMES = {
    "paper": {
        "name": "paper",
        "bg": "#FFFFFF",
        "card": "#F8FAFC",
        "border": "#CBD5E1",
        "grid": "#E2E8F0",
        "text": "#0F172A",
        "muted": "#475569",
        "primary": "#0284C7",
        "secondary": "#059669",
        "amber": "#D97706",
        "crimson": "#DC2626",
        "purple": "#7C3AED",
        "sky": "#0891B2",
        "slate": "#64748B",
        "orange": "#EA580C",
    },
    "dark": {
        "name": "dark",
        "bg": "#0B1220",
        "card": "#111A2E",
        "border": "#263759",
        "grid": "#233252",
        "text": "#E6EDF7",
        "muted": "#93A7C7",
        "primary": "#00E5FF",
        "secondary": "#10B981",
        "amber": "#FFB300",
        "crimson": "#FF3B5C",
        "purple": "#A855F7",
        "sky": "#38BDF8",
        "slate": "#8EA3C3",
        "orange": "#FB923C",
    },
}


def apply_theme(t: dict):
    plt.rcParams.update({
        "figure.facecolor": t["bg"],
        "axes.facecolor": t["card"],
        "axes.edgecolor": t["border"],
        "axes.linewidth": 0.95,
        "axes.labelcolor": t["text"],
        "axes.titlecolor": t["text"],
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "text.color": t["text"],
        "grid.color": t["grid"],
        "grid.alpha": 0.72 if t["name"] == "paper" else 0.50,
        "grid.linewidth": 0.75,
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.2,
        "legend.fontsize": 8.0,
        "legend.frameon": True,
        "legend.facecolor": t["card"],
        "legend.edgecolor": t["border"],
        "legend.framealpha": 0.94,
        "mathtext.fontset": "dejavusans",
        "savefig.dpi": 300,
    })


def style_ax(ax, t: dict, grid_axis: str = "both"):
    ax.set_facecolor(t["card"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(t["border"])
        ax.spines[sp].set_linewidth(0.95)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle="--", alpha=0.65 if t["name"] == "paper" else 0.48)
        ax.set_axisbelow(True)


def save_v051_fig(fig: plt.Figure, stem: str, theme_name: str):
    if theme_name == "paper":
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        OUT_PAPER.mkdir(parents=True, exist_ok=True)
        fig.savefig(ASSETS_DIR / f"{stem}.png", dpi=300, facecolor=fig.get_facecolor())
        fig.savefig(OUT_PAPER / f"{stem}.png", dpi=300, facecolor=fig.get_facecolor())
        fig.savefig(OUT_PAPER / f"{stem}.pdf", facecolor=fig.get_facecolor())
    else:
        OUT_DARK.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_DARK / f"{stem}.png", dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def cayley_rot3_matrix(g: np.ndarray) -> np.ndarray:
    """Exact 3D rational Cayley orthogonal rotor R(g) in SO(3) (step109/109c, 0% sin/cos)."""
    gx, gy, gz = float(g[0]), float(g[1]), float(g[2])
    n2 = gx * gx + gy * gy + gz * gz
    inv_d = 1.0 / (1.0 + n2)
    K = np.array([[0.0, -gz, gy], [gz, 0.0, -gx], [-gy, gx, 0.0]], dtype=np.float64)
    return ((1.0 - n2) * np.eye(3) + 2.0 * K + 2.0 * np.outer(g, g)) * inv_d


def build_a5_icosahedral_orbit() -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    """Generate the exact 60-element A_5 icosahedral orbit on S^2 using rational Cayley SO(3) rotors."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    # Order-3 rotation (120 deg -> tan(60 deg) = sqrt(3)) about (1, 1, 1)/sqrt(3): g_a = (1, 1, 1)
    R_a = cayley_rot3_matrix(np.array([1.0, 1.0, 1.0]))
    # Order-5 rotation (72 deg -> tan(36 deg) = sqrt(5 - 2*sqrt(5))) about icosahedral vertex (0, 1, phi)
    ax_b = np.array([0.0, 1.0, phi])
    ax_b = ax_b / np.linalg.norm(ax_b)
    tan36 = math.sqrt(5.0 - 2.0 * math.sqrt(5.0))
    R_b = cayley_rot3_matrix(tan36 * ax_b)

    # BFS to enumerate all 60 distinct elements of A_5 <= SO(3)
    mats = [np.eye(3)]
    edges_a: list[tuple[int, int]] = []
    edges_b: list[tuple[int, int]] = []

    def find_mat(M: np.ndarray) -> int:
        for idx, X in enumerate(mats):
            if np.max(np.abs(X - M)) < 1e-4:
                return idx
        return -1

    queue = [0]
    while queue:
        i = queue.pop(0)
        for gen_idx, G in enumerate((R_a, R_b)):
            M = G @ mats[i]
            j = find_mat(M)
            if j == -1:
                j = len(mats)
                mats.append(M)
                queue.append(j)
            if gen_idx == 0:
                edges_a.append((i, j))
            else:
                edges_b.append((i, j))

    # Generic basepoint v0 in S^2 with trivial stabilizer under A_5 -> 60 distinct orbit points
    v0 = np.array([0.29, 0.53, 0.797], dtype=np.float64)
    v0 /= np.linalg.norm(v0)
    pts = np.array([M @ v0 for M in mats])
    return pts, edges_a, edges_b


# ==============================================================================
# FIGURE 1: A_5 Orbit on S^2, O(4)->SO(5) Watershed, Finsler Diode & Lyap/LK
# ==============================================================================

def make_fig1_orbits_and_watershed(t: dict):
    apply_theme(t)
    fig = plt.figure(figsize=(15.2, 10.2))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.06, right=0.96, top=0.885, bottom=0.075,
        wspace=0.28, hspace=0.33,
    )

    fig.text(
        0.5, 0.965,
        "MicroContinualCore v0.5.1 — Non-Abelian Orbits, O(4)→SO(5) Watershed & Rational Algebraic Primitives",
        ha="center", va="top", fontsize=13.2, fontweight="bold", color=t["text"],
    )
    fig.text(
        0.5, 0.930,
        "All primitives verified 0% transcendentals [E012], 0% .sqrt() [E013], and 0-axiom Lean 4 (steps 109–138)",
        ha="center", va="top", fontsize=9.3, color=t["muted"],
    )

    # --- Panel A: Exact 60-element A_5 orbit on S^2 ---
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    ax_a.set_facecolor(t["card"])
    u_s = np.linspace(0, 2 * np.pi, 36)
    v_s = np.linspace(0, np.pi, 18)
    xs = np.outer(np.cos(u_s), np.sin(v_s))
    ys = np.outer(np.sin(u_s), np.sin(v_s))
    zs = np.outer(np.ones_like(u_s), np.cos(v_s))
    ax_a.plot_wireframe(xs, ys, zs, color=t["border"], linewidth=0.45, alpha=0.38)

    pts, edges_a, edges_b = build_a5_icosahedral_orbit()
    # Draw generator-b (order-5 pentagons) and generator-a (order-3 triangles)
    for (i, j) in edges_b:
        ax_a.plot(
            [pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], [pts[i, 2], pts[j, 2]],
            color=t["primary"], alpha=0.55, linewidth=0.95,
        )
    for (i, j) in edges_a:
        ax_a.plot(
            [pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], [pts[i, 2], pts[j, 2]],
            color=t["amber"], alpha=0.40, linewidth=0.70,
        )
    sc = ax_a.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=pts[:, 2], cmap="viridis" if t["name"] == "dark" else "coolwarm",
        s=28, edgecolors=t["text"], linewidths=0.4, depthshade=True, zorder=5,
    )
    ax_a.scatter([pts[0, 0]], [pts[0, 1]], [pts[0, 2]], color=t["crimson"], s=75, marker="*", zorder=10, label="Identity $e$")
    ax_a.set_title("A. Exact 60-State $A_5 \\leq \\mathrm{SO}(3)$ Cayley Orbit on $S^2$", fontweight="bold", pad=6)
    ax_a.set_xlim(-1.08, 1.08)
    ax_a.set_ylim(-1.08, 1.08)
    ax_a.set_zlim(-1.08, 1.08)
    ax_a.set_xticks([-1, 0, 1])
    ax_a.set_yticks([-1, 0, 1])
    ax_a.set_zticks([-1, 0, 1])
    ax_a.tick_params(labelsize=7.5, pad=1)
    ax_a.view_init(elev=22, azim=34)
    ax_a.xaxis.pane.fill = False
    ax_a.yaxis.pane.fill = False
    ax_a.zaxis.pane.fill = False
    ax_a.text2D(
        0.03, 0.03,
        "Blue: order-5 gen ($R_b^5=I$) | Amber: order-3 gen ($R_a^3=I$)\n$A_5$ Suite A2: 100.0% @ L=6 → 92.8% @ L=500 (O(4) M=3)",
        transform=ax_a.transAxes, fontsize=7.6, color=t["muted"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor=t["bg"], edgecolor=t["border"], alpha=0.90),
    )

    # --- Panel B: O(4)->SO(5) 4-HH Parity Bridge Watershed (step134) ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, t)
    beta = np.linspace(0.0, 2.0, 300)
    u3_norm2 = 2.0 * (beta - 1.0) ** 2 + 2.0
    det4 = -2.0 * (beta - 1.0) / ((beta - 1.0) ** 2 + 1.0)

    # Shaded empirical 10-seed discovered S_5 parity regions (Suite A1)
    ax_b.axvspan(0.0, 0.162 + 0.502, color=t["secondary"], alpha=0.10)
    ax_b.axvspan(1.788 - 0.617, 2.0, color=t["crimson"], alpha=0.10)
    ax_b.axvline(1.0, color=t["amber"], linestyle=":", linewidth=1.6, zorder=2)

    l1, = ax_b.plot(beta, det4, color=t["primary"], linewidth=2.4, label=r"4×4 Parity $\det_{4\times 4}(H_3\cdots H_0)$")
    ax_b.axhline(0.0, color=t["border"], linewidth=0.9, linestyle="-")
    ax_b.set_xlabel(r"Parity Gate $\beta \in [0, 2]$ ($u_3 = ((2-\beta)v_3,\; \beta) \in \mathbb{R}^5$)")
    ax_b.set_ylabel(r"Effective 4D Determinant $\det_{4\times 4} \in [-1, +1]$", color=t["primary"])
    ax_b.tick_params(axis="y", labelcolor=t["primary"])
    ax_b.set_xlim(-0.04, 2.04)
    ax_b.set_ylim(-1.28, 1.32)

    # Empirical mean markers on the det4 curve
    ev_b, odd_b = 0.162, 1.788
    ev_det = -2.0 * (ev_b - 1.0) / ((ev_b - 1.0) ** 2 + 1.0)
    odd_det = -2.0 * (odd_b - 1.0) / ((odd_b - 1.0) ** 2 + 1.0)
    ax_b.scatter([ev_b], [ev_det], color=t["secondary"], s=70, zorder=6, edgecolor=t["bg"], linewidth=1.2)
    ax_b.scatter([odd_b], [odd_det], color=t["crimson"], s=70, zorder=6, edgecolor=t["bg"], linewidth=1.2)
    ax_b.annotate(
        f"$S_5$ Even gens\n$\\beta = 0.162$",
        xy=(ev_b, ev_det), xytext=(0.24, 0.50),
        fontsize=7.8, fontweight="bold", color=t["secondary"],
        arrowprops=dict(arrowstyle="->", color=t["secondary"], lw=1.1),
    )
    ax_b.annotate(
        f"$S_5$ Odd gens\n$\\beta = 1.788$",
        xy=(odd_b, odd_det), xytext=(1.15, -0.35),
        fontsize=7.8, fontweight="bold", color=t["crimson"],
        arrowprops=dict(arrowstyle="->", color=t["crimson"], lw=1.1),
    )
    ax_b.text(
        1.02, 1.06, "Watershed $\\beta=1$\n$\\|u_3\\|^2_{\\min}=2>0$",
        fontsize=7.7, color=t["amber"], fontweight="bold", ha="left", va="top",
    )

    ax_b2 = ax_b.twinx()
    ax_b2.spines["top"].set_visible(False)
    ax_b2.spines["right"].set_color(t["purple"])
    l2, = ax_b2.plot(beta, u3_norm2, color=t["purple"], linewidth=2.0, linestyle="--",
                     label=r"Householder norm $\|u_3(\beta)\|^2 = 2(\beta-1)^2+2$")
    ax_b2.set_ylabel(r"Squared Norm $\|u_3(\beta)\|^2 \geq 2$ (non-degenerate)", color=t["purple"])
    ax_b2.tick_params(axis="y", labelcolor=t["purple"])
    ax_b2.set_ylim(1.4, 4.7)

    ax_b.set_title("B. $\\mathrm{O}(4) \\to \\mathrm{SO}(5)$ Parity Watershed & Oracle-Free $S_5$ Discovery", fontweight="bold", pad=8)
    ax_b.legend(handles=[l1, l2], loc="lower left", fontsize=7.6)

    # --- Panel C: Optional P3 Rational Finsler-Randers Normalized Contrast (src/algebra.rs:717, step137) ---
    ax_c = fig.add_subplot(gs[1, 0])
    style_ax(ax_c, t)
    delta = np.linspace(-2.5, 2.5, 300)
    mu0 = 0.25
    configs = [
        (0.0, t["slate"], "--", "$v = 0.0$ ($\\beta = 0$, zero drift)"),
        (1.0, t["primary"], "-", "$v = +1.0$ ($\\beta = 0.707$, mild diode)"),
        (2.5, t["crimson"], "-", "$v = +2.5$ ($\\beta = 0.928$, strong diode)"),
    ]
    for v_val, col, ls, lbl in configs:
        b_val = v_val / math.sqrt(1.0 + v_val * v_val)
        # Exact Rust `finsler_randers_contrast_2d`: c(v, delta) = (tau(+d)-tau(-d))/(tau(+d)+tau(-d))
        contrast = (b_val * delta) / np.sqrt(delta * delta + mu0 * mu0)
        ax_c.plot(delta, contrast, color=col, linestyle=ls, linewidth=2.2, label=lbl)

    ax_c.axvline(0.0, color=t["border"], linewidth=0.9)
    ax_c.axhline(0.0, color=t["border"], linewidth=0.9)
    ax_c.axhline(1.0, color=t["amber"], linestyle=":", linewidth=1.35)
    ax_c.axhline(-1.0, color=t["amber"], linestyle=":", linewidth=1.35)
    ax_c.text(
        2.44, 1.05, r"Strict bound $|c(v,\delta)| < 1$ ($\tau_+ + \tau_- = 2\rho > 0$, step137)",
        fontsize=7.3, color=t["amber"], fontweight="bold", ha="right", va="bottom",
    )
    ax_c.annotate(
        "Reverse shift ($\\langle\\beta,\\delta\\rangle < 0$):\n$c \\to -|\\beta|$ (memory hold)",
        xy=(-1.75, -0.92), xytext=(-2.42, -0.45),
        fontsize=7.5, color=t["text"],
        arrowprops=dict(arrowstyle="->", color=t["muted"], lw=1.0),
        bbox=dict(boxstyle="round,pad=0.28", facecolor=t["bg"], edgecolor=t["border"], alpha=0.92),
    )
    ax_c.annotate(
        "Aligned shift ($\\langle\\beta,\\delta\\rangle > 0$):\n$c \\to +|\\beta|$ (fast write)",
        xy=(1.75, 0.92), xytext=(0.62, 0.28),
        fontsize=7.5, color=t["crimson"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=t["crimson"], lw=1.0),
    )
    ax_c.text(
        0.15, -0.22,
        "Exact code formula (finsler_randers_contrast):\nOdd: $c(v,-\\delta) = -c(v,\\delta)$, $c(0,\\delta)=0$\nVerified optional primitive (LOO-filtered OFF)",
        fontsize=7.3, color=t["muted"], va="top",
        bbox=dict(boxstyle="round,pad=0.30", facecolor=t["bg"], edgecolor=t["border"], alpha=0.92),
    )
    ax_c.set_xlim(-2.5, 2.5)
    ax_c.set_ylim(-1.25, 1.32)
    ax_c.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax_c.set_xlabel(r"Directional Shift $\delta_k = x_{\mathrm{in},k} - \hat{u}_{t-1,k}$")
    ax_c.set_ylabel(r"Normalized Contrast $c(v,\delta) = \frac{\langle\beta(v),\delta\rangle}{\sqrt{\delta^2+\mu_0^2}} \in (-1,1)$")
    ax_c.set_title("C. Optional P3 Finsler-Randers Contrast ($\\mu_0 = 0.25$, OFF in One-Config)", fontweight="bold", pad=8)
    ax_c.legend(loc="upper left", fontsize=7.4)

    # --- Panel D: P4 Lyapunov Backbone Gate & Rational LK Balance (src/algebra.rs:810, step138) ---
    ax_d = fig.add_subplot(gs[1, 1])
    style_ax(ax_d, t)
    ratio = np.linspace(0.0, 10.0, 300)
    excess = np.maximum(0.0, ratio - 3.0)
    g_lyap = 1.0 / (1.0 + 0.60 * excess)

    s_vals = np.linspace(0.0, 3.0, 300)
    w_lk = 1.25 / (1.0 + s_vals * s_vals)

    ax_d.plot(ratio, g_lyap, color=t["secondary"], linewidth=2.4,
              label=r"Lyapunov gate $g_{\mathrm{Lyap}}(r) = \frac{1}{1 + 0.6\max(0, r - 3)}$")
    ax_d.plot(s_vals * (10.0 / 3.0), w_lk, color=t["purple"], linewidth=2.2, linestyle="--",
              label=r"Rational LK weight $w(s) = \frac{1.25}{1 + s^2}$ (top axis: $s \in [0,3]$)")
    ax_d.axvline(3.0, color=t["amber"], linestyle=":", linewidth=1.5)
    ax_d.axhline(1.25, color=t["purple"], linestyle=":", linewidth=1.0, alpha=0.55)
    ax_d.fill_between(ratio, g_lyap, 1.0, where=(ratio >= 3.0), color=t["secondary"], alpha=0.11)
    ax_d.text(
        3.22, 1.20, "Drift/spike zone ($r > \\tau_{\\mathrm{lyap}}=3.0$):\ndamps shared backbone slots[0]\nwhile Bartlett PH switches bank\n(synced 5/4 LK weight, step138)",
        fontsize=7.4, color=t["text"], va="top",
        bbox=dict(boxstyle="round,pad=0.32", facecolor=t["bg"], edgecolor=t["border"], alpha=0.92),
    )
    ax_d.set_xlim(0.0, 10.0)
    ax_d.set_ylim(0.0, 1.38)
    ax_d.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    ax_d.set_xlabel(r"Normalized Instantaneous Error Surge $r = e_t^2 / (\sigma_{\mathrm{ema}}^2 + \varepsilon)$")
    ax_d.set_ylabel(r"Gate $g_{\mathrm{Lyap}} \in (0, 1]$ / LK Weight $w(s) \in (0, 1.25]$")
    ax_d.set_title("D. P4 Lyapunov Backbone Gate & Rational Liebel-Koerner Weights (step138)", fontweight="bold", pad=8)
    ax_d.legend(loc="lower left", fontsize=7.4)

    ax_d_top = ax_d.twiny()
    ax_d_top.spines["right"].set_visible(False)
    ax_d_top.spines["top"].set_color(t["purple"])
    ax_d_top.set_xlim(0.0, 3.0)
    ax_d_top.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax_d_top.tick_params(axis="x", colors=t["purple"], labelsize=7.8)
    ax_d_top.set_xlabel(r"LK Channel Parameter $|s_t|$", color=t["purple"], fontsize=8.2, labelpad=3)

    save_v051_fig(fig, "v051_o4_orbits_and_watershed", t["name"])


# ==============================================================================
# FIGURE 2: UnifiedCayleyCore One-Config Ablation Ladder (v0.5.0 vs v0.5.1 P1-P4)
# ==============================================================================

def make_fig2_unified_ablation(t: dict):
    apply_theme(t)
    fig = plt.figure(figsize=(15.8, 9.6))
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.15, 1.0],
        left=0.055, right=0.975, top=0.885, bottom=0.085,
        wspace=0.25, hspace=0.38,
    )

    fig.text(
        0.5, 0.965,
        "UnifiedCayleyCore One-Config: v0.5.0 B3 vs Clean v0.5.1 (Band0(4) + smooth_var + P1 J-XOR + P4 Lyap/LK)",
        ha="center", va="top", fontsize=12.4, fontweight="bold", color=t["text"],
    )
    fig.text(
        0.5, 0.930,
        "Single preset across all 8 Suite B tasks + Add mod p | P2 & P3 LOO-filtered OFF | 6000/6000 Causal Prefix | 0 Heap Allocs",
        ha="center", va="top", fontsize=9.1, color=t["muted"],
    )

    # --- Panel A (Top, spans all 3 cols): Percentage benchmarks ---
    ax_a = fig.add_subplot(gs[0, :])
    style_ax(ax_a, t, grid_axis="y")

    tasks = [
        "ELEC2\nRaw",
        "ELEC2\n$+y_{t-1}$",
        "RepeatPrev\nK=8",
        "RepeatPrev\nK=16",
        "Dyck-2\nAll",
        "Dyck-2\nDepth≥3",
        "Waves\nZeroShot",
        "Waves\nFlip",
        "T-Maze\nL=15 (0-gate)",
        "Add mod 997\n(Bit-Serial)",
        "Add mod 9973\n(Bit-Serial)",
    ]
    v050 = [84.96, 85.24, 58.0, 33.1, 86.9, 87.8, 89.7, 89.2, 94.9, 53.0, 53.5]
    v051 = [89.36, 89.75, 64.6, 40.3, 87.8, 88.5, 90.8, 89.7, 93.6, 65.67, 57.05]
    shuf = [54.48, 54.48, 25.0, 25.0, 33.1, 33.1, 50.0, 50.0, 50.0, 48.5, 52.4]

    x = np.arange(len(tasks))
    w = 0.34
    b0 = ax_a.bar(x - w / 2, v050, width=w, color=t["slate"], alpha=0.80, edgecolor=t["border"], label="v0.5.0 B3 One-Config")
    b1 = ax_a.bar(x + w / 2, v051, width=w, color=t["primary"], alpha=0.94, edgecolor=t["text"], linewidth=0.6, label="Clean v0.5.1 One-Config (P2/P3 OFF)")
    ax_a.scatter(x, shuf, color=t["crimson"], marker="_", s=260, linewidths=2.2, zorder=5, label="Label-Shuffle / Chance Floor")

    for i in range(len(tasks)):
        diff = v051[i] - v050[i]
        top_y = max(v050[i], v051[i])
        ax_a.text(x[i] - w / 2, v050[i] + 1.0, f"{v050[i]:.1f}", ha="center", va="bottom", fontsize=7.0, color=t["muted"])
        ax_a.text(x[i] + w / 2, v051[i] + 1.0, f"{v051[i]:.1f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=t["primary"])
        col_d = t["secondary"] if diff >= 0.4 else (t["muted"] if diff >= -1.5 else t["crimson"])
        sign = "+" if diff >= 0 else ""
        ax_a.text(x[i], top_y + 5.6, f"{sign}{diff:.1f} pp", ha="center", va="bottom",
                  fontsize=7.4, fontweight="bold", color=col_d)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(tasks, fontsize=8.2)
    ax_a.set_ylim(0, 120)
    ax_a.set_ylabel("Accuracy (%)")
    ax_a.set_title("A. Suite B & Bit-Serial Modular Addition — Clean Sanitized One-Config (5–10 Seeds)", fontweight="bold", pad=8)
    ax_a.legend(loc="upper center", ncol=3, fontsize=8.0)

    # --- Panel B (Bottom-Left): CartPole 2D & +dx Breakthrough ---
    ax_b = fig.add_subplot(gs[1, 0])
    style_ax(ax_b, t, grid_axis="y")
    cp_lbls = ["CP-2D\nTrain", "CP-2D\nEval Sc", "CP-2D\nEval Ret", "CP+dx\nTrain", "CP+dx\nEval Sc", "CP+dx\nEval Ret"]
    cp_050 = [64.2, 79.1, 86.3, 91.0, 124.0, 89.0]
    cp_051 = [135.4, 176.0, 161.2, 123.8, 163.7, 94.4]
    xc = np.arange(len(cp_lbls))
    wc = 0.36
    ax_b.bar(xc - wc / 2, cp_050, width=wc, color=t["slate"], alpha=0.80, label="v0.5.0 B3")
    ax_b.bar(xc + wc / 2, cp_051, width=wc, color=t["secondary"], alpha=0.92, edgecolor=t["text"], linewidth=0.5, label="Clean v0.5.1 (P1+P4)")
    ax_b.axhline(200.0, color=t["amber"], linestyle="--", linewidth=1.2, label="Ceiling (200)")
    for i in range(len(cp_lbls)):
        ax_b.text(xc[i] + wc / 2, cp_051[i] + 3.5, f"{cp_051[i]:.0f}", ha="center", va="bottom",
                  fontsize=7.4, fontweight="bold", color=t["secondary"])
    ax_b.set_xticks(xc)
    ax_b.set_xticklabels(cp_lbls, fontsize=7.6)
    ax_b.set_ylim(0, 245)
    ax_b.set_yticks([0, 50, 100, 150, 200])
    ax_b.set_ylabel("Survival Steps (max 200)")
    ax_b.set_title("B. Position-Only CartPole 2D & +dx (P1 J-XOR Driven)", fontweight="bold", pad=8)
    ax_b.legend(loc="upper center", ncol=3, fontsize=6.9, columnspacing=0.8, handlelength=1.4)

    # --- Panel C (Bottom-Mid): NASA C-MAPSS RMSE + SEU + Anti-Leakage ---
    ax_c = fig.add_subplot(gs[1, 1])
    style_ax(ax_c, t, grid_axis="y")
    nasa_lbls = ["FD001\n3-pass", "FD004\n3-pass", "FD004\nBit-30 SEU", "FD001\nRUL-Shuf"]
    n_050 = [23.89, 26.94, 26.15, 40.83]
    n_051 = [24.46, 28.90, 27.26, 41.27]
    xn = np.arange(len(nasa_lbls))
    ax_c.bar(xn - wc / 2, n_050, width=wc, color=t["slate"], alpha=0.80, label="v0.5.0 B3")
    ax_c.bar(xn + wc / 2, n_051, width=wc, color=t["amber"], alpha=0.92, edgecolor=t["text"], linewidth=0.5, label="Clean v0.5.1 (SEU incl. critic_w)")
    for i in range(len(nasa_lbls)):
        ax_c.text(xn[i] - wc / 2, n_050[i] + 0.7, f"{n_050[i]:.1f}", ha="center", va="bottom", fontsize=7.0, color=t["muted"])
        ax_c.text(xn[i] + wc / 2, n_051[i] + 0.7, f"{n_051[i]:.1f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color=t["amber"])
    ax_c.hlines([21.5, 24.5], [-0.45, 0.55], [0.45, 1.45], colors=t["crimson"], linestyles=":", linewidths=1.5, label="Stretch target (<21.5 / 24.5)")
    ax_c.text(2.0, 33.2, "SEU: any_nan=false\nEW-Rec=49.6%", ha="center", fontsize=7.4, color=t["secondary"], fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.25", facecolor=t["bg"], edgecolor=t["border"], alpha=0.9))
    ax_c.set_xticks(xn)
    ax_c.set_xticklabels(nasa_lbls, fontsize=7.8)
    ax_c.set_ylim(0, 48.0)
    ax_c.set_ylabel("RUL RMSE (cycles, ↓ better)")
    ax_c.set_title("C. NASA C-MAPSS 3-Pass & Honest Gap", fontweight="bold", pad=8)
    ax_c.legend(loc="upper left", fontsize=7.0)

    # --- Panel D (Bottom-Right): Compact RAM Audit ---
    ax_d = fig.add_subplot(gs[1, 2])
    style_ax(ax_d, t, grid_axis="x")
    ram_tasks = ["ELEC2\n($+y_{t-1}$)", "Add mod p\n(Bit-Serial)", "RepeatPrev\nK=4..24", "AdaptiveO(d)\n$S_6$ (720 cls)"]
    ram_050 = [34.6, 26.8, 26.2, 245.9]
    ram_051 = [38.2, 30.3, 31.6, 246.6]
    ram_cap = [40.0, 40.0, 32.0, 260.0]

    # Normalize to % of contract RAM cap for clean side-by-side comparison, with exact KB labels!
    pct_050 = [100.0 * r / c for r, c in zip(ram_050, ram_cap)]
    pct_051 = [100.0 * r / c for r, c in zip(ram_051, ram_cap)]
    yr = np.arange(len(ram_tasks))
    hr = 0.34
    ax_d.barh(yr + hr / 2, pct_050, height=hr, color=t["slate"], alpha=0.75, label="v0.5.0 RAM")
    ax_d.barh(yr - hr / 2, pct_051, height=hr, color=t["purple"], alpha=0.90, edgecolor=t["text"], linewidth=0.5, label="Clean v0.5.1 RAM")
    ax_d.axvline(100.0, color=t["crimson"], linestyle="--", linewidth=1.4, label="v0.5.1 Contract Ceiling (100%)")
    for i in range(len(ram_tasks)):
        ax_d.text(102.5, yr[i] - hr / 2, f"{ram_051[i]:.1f} / {ram_cap[i]:.0f} KB",
                  va="center", ha="left", fontsize=7.4, fontweight="bold", color=t["purple"])
    ax_d.set_yticks(yr)
    ax_d.set_yticklabels(ram_tasks, fontsize=7.9)
    ax_d.set_xlim(0, 158)
    ax_d.set_ylim(-0.6, 4.45)
    ax_d.set_xlabel("% of v0.5.1 RAM Contract Ceiling (0 Heap Allocs)")
    ax_d.set_title("D. Memory Footprint vs Contract Ceilings", fontweight="bold", pad=8)
    ax_d.legend(loc="upper center", ncol=3, fontsize=6.7, columnspacing=0.7, handlelength=1.4)

    save_v051_fig(fig, "v051_unified_ablation_ladder", t["name"])


# ==============================================================================
# FIGURE 3: AdaptiveOrthogonalRtrl Live d(t) Trajectories & P5 Extrapolation
# ==============================================================================

def build_step_curve(total_eps: int, events: list[list[int]], n_grid: int = 240) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, n_grid)
    ys = np.full(n_grid, 2.0, dtype=np.float64)
    for ep, new_d in events:
        frac = float(ep) / float(total_eps)
        ys[xs >= frac] = float(new_d)
    return xs, ys


def make_fig3_adaptive_od(t: dict, traj_data: dict):
    apply_theme(t)
    fig = plt.figure(figsize=(16.2, 5.8))
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.052, right=0.978, top=0.835, bottom=0.135,
        wspace=0.25,
    )

    fig.text(
        0.5, 0.962,
        "AdaptiveOrthogonalRtrl Clean v0.5.1 (P5): Live 10-Seed d(t) Escalation, Rank Calibration & Extrapolation",
        ha="center", va="top", fontsize=12.4, fontweight="bold", color=t["text"],
    )
    fig.text(
        0.5, 0.908,
        r"3-Tier Anchor/Warmup LR + Shell Utilization + Plateau Gating + CE/Center-Pull + $m_\theta$ (No Delta-Rank Placebo) | 0 False Esc on Shuffle",
        ha="center", va="top", fontsize=8.9, color=t["muted"],
    )

    # --- Panel A: Live 10-seed d(t) escalation trajectories ---
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, t)
    specs = [
        ("S_6", "$S_6$ (720 cls, $d_{\\min}=5$)", t["purple"], "-"),
        ("S_5", "$S_5$ (120 cls, $d_{\\min}=4$)", t["crimson"], "-"),
        ("A_5", "$A_5$ (60 cls, $d_{\\min}=3$)", t["primary"], "-"),
        ("C_5", "$C_5$ (5 cls, $d_{\\min}=2$)", t["secondary"], "-"),
        ("Label_Shuffle", "Label-Shuffle ($d=2.00$, 0 esc)", t["slate"], "--"),
    ]
    offsets = {"C_5": 0.04, "Label_Shuffle": -0.04}
    for key, lbl, col, ls in specs:
        entry = traj_data[key]
        tot = entry["total_eps"]
        seed_curves = []
        for s_obj in entry["seeds"]:
            xs, ys = build_step_curve(tot, s_obj["events"])
            seed_curves.append(ys)
            if key in ("A_5", "S_5", "S_6"):
                ax_a.step(xs * 100.0, ys, where="post", color=col, alpha=0.14, linewidth=0.9)
        mean_y = np.mean(seed_curves, axis=0) + offsets.get(key, 0.0)
        ax_a.plot(xs * 100.0, mean_y, color=col, linestyle=ls, linewidth=2.4, label=lbl, zorder=5)

    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(1.75, 6.35)
    ax_a.set_yticks([2, 3, 4, 5, 6])
    ax_a.set_xlabel("Training Progress (% of episodes)")
    ax_a.set_ylabel("Active Orthogonal Dimension $d(t)$")
    ax_a.set_title("A. Live 10-Seed $d(t)$ Trajectories (Thin: Seeds, Bold: Mean)", fontweight="bold", pad=8)
    ax_a.legend(loc="upper left", fontsize=7.5)

    # --- Panel B: Final Dimension v0.5.0 vs Clean v0.5.1 P5 vs d_min ---
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, t, grid_axis="y")
    g_lbls = ["Label-\nShuffle", "$C_5$\n(5 cls)", "$A_5$\n(60 cls)", "$S_5$\n(120 cls)", "$S_6$\n(720 cls)"]
    d_050 = [2.00, 2.20, 4.80, 5.80, 6.00]
    sd_050 = [0.00, 0.63, 1.32, 0.42, 0.00]
    d_051 = [2.00, 2.00, 3.10, 4.20, 5.50]
    sd_051 = [0.00, 0.00, 0.32, 0.42, 0.53]
    d_min = [2.0, 2.0, 3.0, 4.0, 5.0]

    xb = np.arange(len(g_lbls))
    wb = 0.34
    ax_b.bar(xb - wb / 2, d_050, yerr=sd_050, capsize=3, width=wb, color=t["slate"], alpha=0.78, label="v0.5.0 Final $d$")
    ax_b.bar(xb + wb / 2, d_051, yerr=sd_051, capsize=3, width=wb, color=t["primary"], alpha=0.92, edgecolor=t["text"], linewidth=0.5, label="Clean v0.5.1 P5 Final $d$")
    ax_b.scatter(xb, d_min, color=t["crimson"], marker="D", s=46, zorder=6, label="Minimal Faithful $d_{\\min}$")

    for i in range(len(g_lbls)):
        ax_b.text(xb[i] + wb / 2, d_051[i] + sd_051[i] + 0.14, f"{d_051[i]:.2f}",
                  ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=t["primary"])
        if d_050[i] > d_051[i] + 0.05:
            ax_b.text(xb[i] - wb / 2, d_050[i] + sd_050[i] + 0.14, f"{d_050[i]:.2f}",
                      ha="center", va="bottom", fontsize=7.2, color=t["muted"])

    ax_b.text(
        -0.38, 4.55,
        "Exact $d_{\\min}$ hits:\n• $A_5$: 9/10 ($d=3$)\n• $S_5$: 8/10 ($d=4$)\n• Blind OOD $A_4$: 5/5 ($d=3$)",
        fontsize=7.2, fontweight="bold", color=t["secondary"], va="top",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=t["bg"], edgecolor=t["border"], alpha=0.92),
    )
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(g_lbls, fontsize=8.2)
    ax_b.set_ylim(0, 7.5)
    ax_b.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax_b.set_ylabel("Final Active Dimension $d_{\\mathrm{final}}$ (mean ± sd)")
    ax_b.set_title("B. Over-Escalation Reduction vs Minimal $d_{\\min}$", fontweight="bold", pad=8)
    ax_b.legend(loc="upper left", fontsize=7.3)

    # --- Panel C: Long-Horizon Extrapolation L=6..500 ---
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, t)
    horizons = [6, 12, 30, 100, 250, 500]
    xh = np.arange(len(horizons))

    curves = [
        ("C_5", [97.2, 95.3, 93.4, 92.2, 91.0, 88.8], [99.6, 98.5, 97.3, 98.5, 99.1, 99.0], t["secondary"], "o"),
        ("A_5", [78.8, 57.6, 40.7, 25.2, 21.7, 21.4], [94.9, 90.8, 85.2, 63.2, 38.4, 20.6], t["primary"], "s"),
        ("S_6", [77.2, 25.8, 0.7, 0.2, 0.1, 0.1], [93.0, 42.7, 4.5, 0.6, 0.1, 0.2], t["purple"], "^"),
        ("S_5", [81.3, 38.1, 11.8, 3.8, 2.9, 2.8], [82.2, 46.9, 8.8, 1.3, 1.4, 0.8], t["crimson"], "d"),
    ]
    for name, c050, c051, col, mk in curves:
        ax_c.plot(xh, c050, color=col, linestyle="--", linewidth=1.35, alpha=0.52)
        ax_c.plot(xh, c051, color=col, linestyle="-", marker=mk, markersize=4.8, linewidth=2.2, label=f"${name}$ v0.5.1")

    ax_c.plot([], [], color=t["muted"], linestyle="--", linewidth=1.4, label="v0.5.0 base")
    ax_c.annotate("+44.5 pp @ L=30", xy=(2, 85.2), xytext=(2.45, 75.5),
                  fontsize=7.3, fontweight="bold", color=t["primary"],
                  arrowprops=dict(arrowstyle="->", color=t["primary"], lw=1.0))
    ax_c.annotate("+16.9 pp @ L=12", xy=(1, 42.7), xytext=(1.45, 51.5),
                  fontsize=7.3, fontweight="bold", color=t["purple"],
                  arrowprops=dict(arrowstyle="->", color=t["purple"], lw=1.0))
    ax_c.text(
        2.34, 4.0,
        "Blind OOD & Honest Limits:\n• $D_5/A_4/S_4$ @L=500: 99.1/86.0/46.9%\n• $S_5$ 12k: $V_4$ 3D hyst. (24k: 22.4%)\n• $C_7$ 2-gen: 0% @L=500 (angle drift)",
        fontsize=6.5, color=t["crimson"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor=t["bg"], edgecolor=t["border"], alpha=0.92),
    )
    ax_c.set_xticks(xh)
    ax_c.set_xticklabels([f"L={h}" for h in horizons], fontsize=8.1)
    ax_c.set_ylim(-4, 124)
    ax_c.set_yticks([0, 20, 40, 60, 80, 100])
    ax_c.set_ylabel("Exact Group-State Accuracy (%)")
    ax_c.set_xlabel("Evaluation Word Length $L$ (Train $L \\leq 10$, $S_6$: $L \\leq 24$)")
    ax_c.set_title("C. Extrapolation: v0.5.0 (Dashed) vs Clean v0.5.1 (Solid)", fontweight="bold", pad=8)
    ax_c.legend(loc="upper center", fontsize=6.9, ncol=3, columnspacing=0.8, handlelength=1.4)

    save_v051_fig(fig, "v051_adaptive_od_trajectories", t["name"])


def main():
    parser = argparse.ArgumentParser(description="Generate v0.5.1 publication figures.")
    parser.add_argument("--traj", type=str, default=None, help="Optional JSON path from bench_step8_v050 --suite d --dump-traj")
    args = parser.parse_args()

    traj_data = DEFAULT_SUITE_D_TRAJ
    if args.traj:
        p = Path(args.traj)
        if p.exists():
            traj_data = json.loads(p.read_text(encoding="utf-8"))

    for theme_name, theme in THEMES.items():
        make_fig1_orbits_and_watershed(theme)
        make_fig2_unified_ablation(theme)
        make_fig3_adaptive_od(theme, traj_data)

    print("Generated 3 v0.5.1 publication figures in:")
    print(f"  - {ASSETS_DIR / 'v051_o4_orbits_and_watershed.png'}")
    print(f"  - {ASSETS_DIR / 'v051_unified_ablation_ladder.png'}")
    print(f"  - {ASSETS_DIR / 'v051_adaptive_od_trajectories.png'}")
    print(f"  + paper PDF/PNG in {OUT_PAPER} and dark PNG in {OUT_DARK}")


if __name__ == "__main__":
    main()
