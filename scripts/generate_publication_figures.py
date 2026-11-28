#!/usr/bin/env python3
# Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov & Sofia Sidorenko
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
MicroContinualCore v0.5.0 — Complete Publication Figure & Architecture Visualization Suite.

Generates 8 publication-grade multi-panel figures covering both the complete system/mathematical
architecture and every audited empirical benchmark (v0.4.0–v0.5.0), in two cohesive visual themes:
  1. `assets/figures/paper/` : Print/LaTeX-ready white theme (300 DPI PNG + vector PDF + vector SVG)
  2. `assets/figures/dark/`  : High-contrast dark aerospace theme matching `assets/nasa_cmapss_4panel.png`
                               (300 DPI PNG, ideal for GitHub Releases, README, and presentations)

All empirical values match verified live benchmark outputs in `PAPER.md`, `logs/bench_step8_v050_opt.txt`,
`logs/bench_step7_nasa_v050.txt`, live `bench_forensic_audit`, live `bench_step6_scaling --filter width`,
and post-fix `bench_step9_prime_arith`. Mathematical primitive curves in Fig. 2 & Fig. 3 are computed
directly from the exact Rust algorithms (`src/math.rs`, `src/fixed_point.rs`, `src/rtrl.rs`).
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PAPER = REPO_ROOT / "assets" / "figures" / "paper"
OUT_DARK = REPO_ROOT / "assets" / "figures" / "dark"


# ==============================================================================
# Theme Definitions (Print-Ready Light vs GitHub/Aerospace Dark)
# ==============================================================================

THEMES = {
    "paper": {
        "name": "paper",
        "bg": "#FFFFFF",
        "card": "#F8FAFC",
        "card_alt": "#F1F5F9",
        "border": "#CBD5E1",
        "grid": "#E2E8F0",
        "text": "#0F172A",
        "muted": "#475569",
        "primary": "#0284C7",
        "primary_soft": "#E0F2FE",
        "secondary": "#059669",
        "secondary_soft": "#D1FAE5",
        "amber": "#D97706",
        "amber_soft": "#FEF3C7",
        "crimson": "#DC2626",
        "crimson_soft": "#FEE2E2",
        "purple": "#7C3AED",
        "purple_soft": "#EDE9FE",
        "sky": "#0891B2",
        "sky_soft": "#CFFAFE",
        "slate": "#64748B",
        "slate_soft": "#E2E8F0",
        "orange": "#EA580C",
        "heat_cmap": "YlGnBu",
    },
    "dark": {
        "name": "dark",
        "bg": "#0B1220",
        "card": "#111A2E",
        "card_alt": "#16223A",
        "border": "#263759",
        "grid": "#233252",
        "text": "#E6EDF7",
        "muted": "#93A7C7",
        "primary": "#00E5FF",
        "primary_soft": "#0B2E42",
        "secondary": "#10B981",
        "secondary_soft": "#0B2E28",
        "amber": "#FFB300",
        "amber_soft": "#33260C",
        "crimson": "#FF3B5C",
        "crimson_soft": "#381322",
        "purple": "#A855F7",
        "purple_soft": "#28173F",
        "sky": "#38BDF8",
        "sky_soft": "#0F2940",
        "slate": "#8EA3C3",
        "slate_soft": "#1C2942",
        "orange": "#FB923C",
        "heat_cmap": "viridis",
    },
}


def apply_theme(t: dict):
    plt.rcParams.update({
        "figure.facecolor": t["bg"],
        "axes.facecolor": t["card"],
        "axes.edgecolor": t["border"],
        "axes.linewidth": 0.9,
        "axes.labelcolor": t["text"],
        "axes.titlecolor": t["text"],
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "text.color": t["text"],
        "grid.color": t["grid"],
        "grid.alpha": 0.75 if t["name"] == "paper" else 0.55,
        "grid.linewidth": 0.7,
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.titlesize": 10.4,
        "axes.labelsize": 9.2,
        "legend.fontsize": 7.9,
        "legend.frameon": True,
        "legend.facecolor": t["card"],
        "legend.edgecolor": t["border"],
        "legend.framealpha": 0.95,
        "mathtext.fontset": "dejavusans",
        "savefig.dpi": 300,
    })


def save_figure(fig: plt.Figure, stem: str, theme_name: str):
    if theme_name == "paper":
        OUT_PAPER.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PAPER / f"{stem}.png", dpi=300, facecolor=fig.get_facecolor())
        fig.savefig(OUT_PAPER / f"{stem}.pdf", facecolor=fig.get_facecolor())
        fig.savefig(OUT_PAPER / f"{stem}.svg", facecolor=fig.get_facecolor())
    else:
        OUT_DARK.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_DARK / f"{stem}.png", dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def style_ax(ax, t: dict, grid_axis: str = "both"):
    ax.set_facecolor(t["card"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(t["border"])
        ax.spines[sp].set_linewidth(0.95)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle="--", alpha=0.55 if t["name"] == "dark" else 0.7)
        ax.set_axisbelow(True)


def add_header(fig: plt.Figure, title: str, subtitle: str, t: dict, y_title: float = 0.972, y_sub: float = 0.938):
    fig.text(0.5, y_title, title, ha="center", va="top", fontsize=13.4, fontweight="bold", color=t["text"])
    fig.text(0.5, y_sub, subtitle, ha="center", va="top", fontsize=9.1, color=t["muted"])


def draw_box(
    ax, x: float, y: float, w: float, h: float,
    title: str, body: str, fill: str, edge: str, t: dict,
    title_color: str | None = None, title_size: float = 8.9, body_size: float = 7.4,
    lw: float = 1.4, radius: float = 0.015, zorder: int = 3, title_y: float | None = None,
    body_y: float | None = None
):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=zorder
    )
    ax.add_patch(box)
    tc = title_color if title_color else edge
    if body:
        ty = title_y if title_y is not None else (0.78 if "\n" in body else 0.68)
        by = body_y if body_y is not None else (0.35 if "\n" in body else 0.30)
        ax.text(
            x + w * 0.5, y + h * ty,
            title, ha="center", va="center", fontsize=title_size,
            fontweight="bold", color=tc, zorder=zorder + 1
        )
        ax.text(
            x + w * 0.5, y + h * by,
            body, ha="center", va="center", fontsize=body_size,
            color=t["text"], linespacing=1.30, zorder=zorder + 1
        )
    else:
        ax.text(
            x + w * 0.5, y + h * 0.5,
            title, ha="center", va="center", fontsize=title_size,
            fontweight="bold", color=tc, zorder=zorder + 1
        )


def draw_arrow(ax, x0: float, y0: float, x1: float, y1: float, color: str, lw: float = 1.6,
               rad: float = 0.0, ls: str = "-", zorder: int = 4):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=5.5,head_width=3.2",
        color=color, linewidth=lw, linestyle=ls, zorder=zorder
    )
    ax.add_patch(arr)


def draw_polyline_arrow(ax, pts: list[tuple[float, float]], color: str, lw: float = 1.3,
                        ls: str = "--", zorder: int = 4):
    xs = [p[0] for p in pts[:-1]]
    ys = [p[1] for p in pts[:-1]]
    if len(pts) >= 2:
        ax.plot([p[0] for p in pts[:-1]] + [pts[-1][0]],
                [p[1] for p in pts[:-1]] + [pts[-1][1]],
                color=color, lw=lw, ls=ls, zorder=zorder)
        draw_arrow(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1],
                   color=color, lw=lw, ls=ls, zorder=zorder)


# ==============================================================================
# Exact Rust Math Replicas for Fig. 3 (bit-faithful to src/math.rs & fixed_point.rs)
# ==============================================================================

def rust_inv_sqrt_newton_f32(a: float, steps: int = 2) -> float:
    if not math.isfinite(a) or a <= 1e-30:
        return 0.0
    ac = np.float32(min(a, 1e30))
    bits = struct.unpack("<I", struct.pack("<f", float(ac)))[0]
    exp_b = (bits >> 23) & 0xFF
    e = exp_b - 127
    m_bits = (bits & 0x007F_FFFF) | 0x3F80_0000
    m = np.float32(struct.unpack("<f", struct.pack("<I", m_bits))[0])
    k = e >> 1
    r = e & 1
    pow2_bits = (((127 - k) & 0xFF) << 23)
    pow2_neg_k = np.float32(struct.unpack("<f", struct.pack("<I", pow2_bits))[0])
    odd_scale = np.float32(0.70703125 if r != 0 else 1.0)
    y = np.float32(0.25) * (np.float32(5.0) - m) * odd_scale * pow2_neg_k
    for _ in range(steps):
        y = np.float32(0.5) * y * (np.float32(3.0) - ac * y * y)
    return max(0.0, float(y))


def rust_q8_24_inv_sqrt(a: float) -> float:
    raw = int(round(a * (1 << 24)))
    if raw <= 0:
        return 0.0
    msb = raw.bit_length() - 1
    e = msb - 24
    r = e & 1
    k = e >> 1
    m_q24 = (raw >> (msb - 24)) if msb >= 24 else (raw << (24 - msb))
    seed = ((5 << 24) - m_q24) >> 2
    if r != 0:
        seed = (seed * 181 + 128) >> 8
    m_eff = (m_q24 << 1) if r != 0 else m_q24
    three_q24 = 3 << 24
    y = max(seed, 1)
    for _ in range(3):
        y2 = (y * y + (1 << 23)) >> 24
        my2 = (m_eff * y2 + (1 << 23)) >> 24
        term = three_q24 - my2
        yn = (y * term + (1 << 24)) >> 25
        if yn > 0:
            y = yn
    if k > 0:
        sh = min(k, 62)
        y_sh = (y + (1 << (sh - 1))) >> sh
    elif k < 0:
        sh = min(-k, 38)
        y_sh = y << sh
    else:
        y_sh = y
    out = max(1, min(y_sh, (1 << 31) - 1))
    return out / float(1 << 24)


def rational_pos_and_deriv(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -4.0, 4.0)
    pos = np.minimum(2.0, 1.0 + zc)
    neg = np.maximum(0.05, 1.0 / (1.0 - zc + 0.5 * zc * zc))
    return np.where(zc >= 0.0, pos, neg)


# ==============================================================================
# FIGURE 1: Full System Architecture & #[axiom_zero] Assurance Stack
# ==============================================================================

def make_fig01(t: dict):
    fig = plt.figure(figsize=(15.6, 9.3), facecolor=t["bg"])
    add_header(
        fig,
        "MicroContinualCore v0.5.0 — Unified Zero-Transcendental Architecture & Assurance Stack",
        "Single-pass online continual RL & streaming engine | 2.3–738 KB RAM | 0% exp/ln/sin/cos/tanh/sqrt | 0 heap allocs/step | 68 0-axiom Lean 4 proofs",
        t, y_title=0.976, y_sub=0.943
    )

    ax = fig.add_axes([0.018, 0.025, 0.964, 0.895])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background container for Panel A
    main_bg = FancyBboxPatch(
        (0.008, 0.270), 0.984, 0.715,
        boxstyle="round,pad=0.005,rounding_size=0.018",
        facecolor=t["card"], edgecolor=t["border"], linewidth=1.2, zorder=1
    )
    ax.add_patch(main_bg)
    ax.text(
        0.022, 0.965,
        "A.  Online Continual Signal & Gradient Architecture (UnifiedCayleyCore + Specialized Non-Abelian & Fixed-Point Kernels)",
        fontsize=10.1, fontweight="bold", color=t["primary"], va="top", zorder=2
    )

    # Column 1: Input & Causal Conditioner
    draw_box(
        ax, 0.022, 0.625, 0.145, 0.275,
        "1. Causal Input\nConditioner",
        "Raw stream $x_t$ (RL / Stream)\n"
        "Causal Welford norm:\n"
        "$(x_i - \\mu_i) / (\\sigma_i^2 + 1)^{1/2}$\n"
        "(Newton inv_sqrt, 0 sqrt)\n"
        "+ Algebraic velocity $\\Delta x_t$\n"
        "+ NaN / spike freeze guard",
        t["sky_soft"], t["sky"], t, title_size=8.8, body_size=7.3, title_y=0.80, body_y=0.35
    )

    # Skip-X bypass box below Col 1
    draw_box(
        ax, 0.022, 0.470, 0.145, 0.115,
        "Direct Skip Channel",
        "Optional Band-0 (use_band0)\n"
        "Skip $x_t + \\Delta x_t \\rightarrow$ RMSNorm",
        t["slate_soft"], t["slate"], t, title_size=8.3, body_size=7.2
    )

    # Column 2: Shared Multi-Band Orthogonal Backbone
    bb_bg = FancyBboxPatch(
        (0.198, 0.475), 0.312, 0.460,
        boxstyle="round,pad=0.004,rounding_size=0.014",
        facecolor=t["card_alt"], edgecolor=t["primary"], linewidth=1.3, linestyle="--", zorder=2
    )
    ax.add_patch(bb_bg)
    ax.text(
        0.354, 0.922,
        "2. Shared Orthogonal Backbone (slots[0], Exact O(D) Online RTRL)",
        ha="center", va="center", fontsize=8.6, fontweight="bold", color=t["primary"], zorder=3
    )

    # Band 1 (Fast 16x2D)
    draw_box(
        ax, 0.212, 0.720, 0.284, 0.180,
        "Band 1 — Fast 16×2D Rational Cayley Rotors (32D)",
        "$R_2(\\omega) = (\\frac{1-\\omega^2}{1+\\omega^2}, \\frac{2\\omega}{1+\\omega^2})$,  $\\lambda(\\alpha) = \\frac{199}{200(1+\\Delta\\alpha^2)}$\n"
        "Piecewise-rational $C^1$ step $\\Delta(z) \\in [0.05, 2.0]$ (step116)\n"
        "Master-Eq gain $g_{\\mathrm{in}} = \\Delta / (1 + 0.5\\Delta)$ + local feedback $q_{\\mathrm{fb}} u_{t-1}$",
        t["primary_soft"], t["primary"], t, title_size=8.3, body_size=7.1, zorder=3
    )

    # Band 2 (Slow 8x3D SO(3))
    draw_box(
        ax, 0.212, 0.500, 0.284, 0.185,
        "Band 2 — Slow 8×3D SO(3) Cayley Rotors (24D)",
        "$R(g) = (I_3 - [g]_\\times)^{-1}(I_3 + [g]_\\times),\\;\\; R^\\top R = I_3,\\; \\det R = +1$\n"
        "Double-angle $R_2(g) = R(g)^2$, decoupled $(Q \\times \\omega)$ spectrum\n"
        "Exact $O(S)$ online RTRL sensitivities $\\partial u_t / \\partial (\\alpha, \\omega, W_{\\mathrm{in}})$",
        t["secondary_soft"], t["secondary"], t, title_size=8.3, body_size=7.1, zorder=3
    )

    # Specialized v0.5.0 Extensions strip
    draw_box(
        ax, 0.022, 0.290, 0.488, 0.130,
        "Specialized v0.5.0 Algebraic & Hardware-Target Cores (Same #[axiom_zero] Foundation)",
        "• NonAbelianO4Rtrl (Suite A): 4-Householder $O(4)\\hookrightarrow SO(5)$ Parity Bridge + $(S^4)^{5M}$ Proto-Metric + Q1.14 (i16) deploy (62.7 → 45.1 KB)\n"
        "• AdaptiveOrthogonalRtrl (Suite D): Self-escalating $O(d)\\hookrightarrow SO(7)$ Givens-Parity tower ($d=2..6$, <1e-6 isometric upgrade, 720-class $S_6$)\n"
        "• FixedCayleyCore (Suite C): 100% Zero-FPU integer Q8.24 / Q39.24 engine (2.3 KB, 1.68 µs/step, deterministic FNV-1a digests)",
        t["purple_soft"], t["purple"], t, title_size=8.1, body_size=7.0, zorder=3, title_y=0.82, body_y=0.35
    )

    # Column 3: Per-Band RMSNorm + Multi-Regime Bank Controller
    draw_box(
        ax, 0.548, 0.720, 0.205, 0.185,
        "3. Per-Band Newton RMSNorm\n& 4-Scalar Rational IDBD",
        "2-step Newton inv_sqrt per band\n"
        "(strictly positive, 0 .sqrt())\n"
        "Learnable gates $[g_{\\mathrm{skip}}, g_{B0}, g_{B1}, g_{B2}]$\n"
        "Meta-adapted via 4-scalar IDBD (step131)",
        t["amber_soft"], t["amber"], t, title_size=8.3, body_size=7.1
    )

    draw_box(
        ax, 0.548, 0.485, 0.205, 0.205,
        "4. Multi-Regime Continual\nBank Controller (4–32 Slots)",
        "BartlettPageHinkley drift detector\n"
        "(variance-frozen, lag-1 autocorrelation)\n"
        "Probe-Before-Allocate / Par6 causal\n"
        "6-step parallel read-only scoring\n"
        "Lightweight heads in slots[1..3]",
        t["amber_soft"], t["amber"], t, title_size=8.3, body_size=7.1
    )

    # Column 4: Output Heads & Hardware SEU Scrubber
    draw_box(
        ax, 0.785, 0.675, 0.193, 0.230,
        "5. Universal Output Heads\n& Online Learning Rules",
        "• Classification: Shift-invariant\n"
        "  quartic softmax $\\phi_4(z)$ (0% exp)\n"
        "• RL Control: Online TD($\\lambda$) Actor-Critic\n"
        "  + Tsallis-2 entropy regularization\n"
        "• Regression: Damped RMSProp / NLMS\n"
        "• Non-mutating &self $K$-step rollout",
        t["primary_soft"], t["primary"], t, title_size=8.3, body_size=7.1
    )

    draw_box(
        ax, 0.785, 0.475, 0.193, 0.170,
        "6. Hardware IEEE-754\nBit-30 SEU Scrubber",
        "Detects NaN / Inf / $|x| > B$ across\n"
        "state registers & SRAM weights\n"
        "(scrub_state_seu, scrub_weight_seu)\n"
        "Unit-sphere projection + telemetry\n"
        "(248/248 NASA & 10/10 SEU survival)",
        t["crimson_soft"], t["crimson"], t, title_size=8.2, body_size=7.0
    )

    # KPI Banner bottom-right of Panel A
    draw_box(
        ax, 0.548, 0.290, 0.430, 0.130,
        "Measured Single-Config & Swarm Envelope (--release, 0 Heap Allocs/Step)",
        "• Unified One-Config: 26.2–34.6 KB RAM across 8 domains  •  6000/6000 Causal Prefix Invariance\n"
        "• Width Scaling ($D = 12 \\rightarrow 768$): RAM $\\propto D^{0.99}$ (100 KB vs GRU 21 MB), Lat $\\propto D^{0.97}$ (21–31× faster)\n"
        "• 16-Thread Swarm ($N = 4,096$): 2.33 M steps/s in 25.3 MB (NASA twins: 58.6 MB, 0.91 M cyc/s)",
        t["secondary_soft"], t["secondary"], t, title_size=8.1, body_size=7.0, title_y=0.82, body_y=0.35
    )

    # Clean non-crossing arrows!
    draw_arrow(ax, 0.169, 0.800, 0.210, 0.805, t["primary"], lw=1.8)
    draw_arrow(ax, 0.169, 0.700, 0.210, 0.605, t["secondary"], lw=1.8)
    draw_arrow(ax, 0.095, 0.623, 0.095, 0.588, t["slate"], lw=1.4)
    # Clean orthogonal Skip bypass routing cleanly in the corridor BELOW Backbone (y=0.453) and UP the gap (x=0.528) to RMSNorm!
    draw_polyline_arrow(
        ax,
        [(0.169, 0.495), (0.184, 0.495), (0.184, 0.452), (0.528, 0.452), (0.528, 0.765), (0.546, 0.765)],
        color=t["slate"], lw=1.35, ls="--", zorder=3
    )
    ax.text(0.354, 0.458, "Skip-X residual bypass (zero interference with Band 1/2)",
            ha="center", va="bottom", fontsize=6.7, color=t["muted"], style="italic")

    # Band 1 & Band 2 to RMSNorm
    draw_arrow(ax, 0.498, 0.810, 0.546, 0.828, t["primary"], lw=1.8)
    draw_arrow(ax, 0.498, 0.610, 0.546, 0.790, t["secondary"], lw=1.8)
    # RMSNorm to Bank Controller & Output Heads
    draw_arrow(ax, 0.650, 0.718, 0.650, 0.692, t["amber"], lw=1.6)
    draw_arrow(ax, 0.755, 0.815, 0.783, 0.815, t["primary"], lw=1.8)
    draw_arrow(ax, 0.755, 0.600, 0.783, 0.735, t["amber"], lw=1.6)
    # Loss / PH feedback arrow cleanly in the corridor x=0.768
    draw_arrow(ax, 0.783, 0.700, 0.755, 0.545, t["amber"], lw=1.3, rad=-0.22, ls=":")
    # SEU scrubber guard UP to Output Heads and cleanly routed BELOW Box 4 (y=0.440) to Backbone!
    draw_arrow(ax, 0.882, 0.647, 0.882, 0.673, t["crimson"], lw=1.6)
    draw_polyline_arrow(
        ax,
        [(0.830, 0.473), (0.830, 0.438), (0.460, 0.438), (0.460, 0.474)],
        color=t["crimson"], lw=1.25, ls=":", zorder=3
    )
    ax.text(0.655, 0.443, "Bit-30 State & SRAM Weight SEU scrub path",
            ha="center", va="bottom", fontsize=6.7, color=t["crimson"], style="italic")

    # ==========================================================================
    # Panel B: Bottom 3-Pillar Formal & Compile-Time Assurance Stack
    # ==========================================================================
    b_bg = FancyBboxPatch(
        (0.008, 0.015), 0.984, 0.236,
        boxstyle="round,pad=0.005,rounding_size=0.018",
        facecolor=t["card"], edgecolor=t["border"], linewidth=1.2, zorder=1
    )
    ax.add_patch(b_bg)
    ax.text(
        0.022, 0.236,
        "B.  Three-Layer Formal, Static & Runtime Assurance Stack (crates/axiom_zero + tests/ + proofs/MicroContinualCoreProofs.lean)",
        fontsize=10.1, fontweight="bold", color=t["secondary"], va="top", zorder=2
    )

    draw_box(
        ax, 0.022, 0.032, 0.305, 0.168,
        "Pillar 1 — Compile-Time #[axiom_zero] AST Gate",
        "• 10/10 core Rust modules enforced at build time ([E001]–[E013])\n"
        "• Bans panic!, unwrap, None, arr[i], raw / & %, exit / abort\n"
        "• Bans ALL transcendentals ([E012]) & .sqrt() ([E013])\n"
        "• Verified by 20/20 compile_fail doctests + audit_v050_invariants.py",
        t["primary_soft"], t["primary"], t, title_size=8.4, body_size=7.2, title_y=0.81, body_y=0.35
    )

    draw_box(
        ax, 0.347, 0.032, 0.305, 0.168,
        "Pillar 2 — Zero-Alloc & Causal Runtime Integrity",
        "• 0 heap allocations on hot paths (tests/zero_alloc_unified.rs)\n"
        "• 41/41 unit & finite-difference tests (default + strict_deterministic)\n"
        "• 6000/6000 bit-identical Causal Prefix Invariance under future NaN\n"
        "• Global Label-Shuffle & RUL-Shuffle controls collapse to chance",
        t["amber_soft"], t["amber"], t, title_size=8.4, body_size=7.2, title_y=0.81, body_y=0.35
    )

    draw_box(
        ax, 0.672, 0.032, 0.305, 0.168,
        "Pillar 3 — Constructive Lean 4 Formal Proofs",
        "• 68/68 #print axioms checks (step01–step134), 0 axioms\n"
        "• Zero sorry, zero propext, zero Quot.sound (over exact Balance)\n"
        "• Covers SO(3) det=+1, p4 softmax, selective BIBO, Newton inv_sqrt,\n"
        "  Q1.14 margin (step132), 4-HH Q8.24 SO(5) watershed (step134)",
        t["secondary_soft"], t["secondary"], t, title_size=8.4, body_size=7.2, title_y=0.81, body_y=0.35
    )

    draw_arrow(ax, 0.329, 0.116, 0.345, 0.116, t["border"], lw=1.8)
    draw_arrow(ax, 0.654, 0.116, 0.670, 0.116, t["border"], lw=1.8)

    save_figure(fig, "fig01_architecture_overview", t["name"])


# ==============================================================================
# FIGURE 2: O(4) -> SO(5) Parity Bridge & O(d) -> SO(7) Adaptive Geometry
# ==============================================================================

def make_fig02(t: dict):
    fig = plt.figure(figsize=(15.8, 6.0), facecolor=t["bg"])
    add_header(
        fig,
        "Mathematical Geometry: Non-Degenerate O(4) ↪ SO(5) Parity Bridge & Isometric O(d) ↪ SO(7) Ladder",
        "Continuous orientation switching across det(Q) = ±1 without singular collapse (Suite A) and zero-forgetting rank escalation (Suite D)",
        t, y_title=0.972, y_sub=0.922
    )

    gs = gridspec.GridSpec(1, 3, left=0.052, right=0.985, top=0.84, bottom=0.11, wspace=0.22)

    # --- Panel A: Watershed Curve ||u3(beta)||^2 & Empirical Parity Separation ---
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, t)
    beta = np.linspace(-0.08, 2.08, 400)
    u3_sq = (2.0 - beta) ** 2 + beta ** 2

    ax1_r = ax1.twinx()
    ax1_r.set_ylim(0, 2.2)
    ax1_r.set_yticks([])
    for sp in ("top", "right", "left", "bottom"):
        ax1_r.spines[sp].set_visible(False)

    even_mu, even_sd = 0.162, 0.502
    odd_mu, odd_sd = 1.788, 0.617
    g_even = np.exp(-0.5 * ((beta - even_mu) / even_sd) ** 2)
    g_odd = np.exp(-0.5 * ((beta - odd_mu) / odd_sd) ** 2)
    ax1_r.fill_between(beta, 0, g_even * 0.85, color=t["secondary"], alpha=0.22,
                       label="Measured $S_5$ even $\\beta$ ($0.162\\pm 0.502$)")
    ax1_r.fill_between(beta, 0, g_odd * 0.85, color=t["crimson"], alpha=0.22,
                       label="Measured $S_5$ odd $\\beta$ ($1.788\\pm 0.617$)")
    ax1_r.axvline(even_mu, color=t["secondary"], ls=":", lw=1.4)
    ax1_r.axvline(odd_mu, color=t["crimson"], ls=":", lw=1.4)

    ax1.plot(beta, u3_sq, color=t["primary"], lw=2.6, label=r"$\|u_3(\beta)\|^2 = 2(\beta-1)^2 + 2 \geq 2$")
    ax1.axhline(2.0, color=t["amber"], ls="--", lw=1.2, alpha=0.85)
    ax1.scatter([0.0, 1.0, 2.0], [4.0, 2.0, 4.0], color=[t["secondary"], t["amber"], t["crimson"]], s=65, zorder=6)

    ax1.annotate("Even pole ($\\beta=0$)\n$\\det(Q)=+1$ ($SO(4)$)", xy=(0.0, 4.0), xytext=(0.12, 4.35),
                 fontsize=7.6, color=t["secondary"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=t["secondary"], lw=1.1))
    ax1.annotate("Symmetric Watershed\n$\\beta=1,\\; \\|u_3\\|^2=2>0$", xy=(1.0, 2.0), xytext=(0.56, 2.78),
                 fontsize=7.6, color=t["amber"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=t["amber"], lw=1.1))
    ax1.annotate("Odd pole ($\\beta=2$)\n$\\det(Q)=-1$ (Reflection)", xy=(2.0, 4.0), xytext=(1.15, 4.35),
                 fontsize=7.6, color=t["crimson"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=t["crimson"], lw=1.1))

    ax1.set_xlim(-0.08, 2.08)
    ax1.set_ylim(1.2, 5.05)
    ax1.set_xlabel(r"Learned Parity Parameter $\beta \in [0, 2]$ (init $\beta_0 = 1.0$)")
    ax1.set_ylabel(r"Watershed Axis Norm Squared $\|u_3(\beta)\|^2$")
    ax1.set_title("A. Non-Degenerate Watershed & Oracle-Free $S_5$ Parity Split", fontweight="bold", pad=8)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=7.2, framealpha=0.93)

    # --- Panel B: 4-Householder SO(5) Cascade & Q1.14 Proto-Metric ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("B. 4-Householder $SO(5)$ Cascade & $(S^4)^{5M}$ Proto-Metric", fontweight="bold", pad=8, color=t["text"])

    p_bg = FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.01,rounding_size=0.025",
                          facecolor=t["card"], edgecolor=t["border"], lw=1.0)
    ax2.add_patch(p_bg)

    draw_box(ax2, 0.055, 0.81, 0.89, 0.145,
             "Input Action $x_t \\rightarrow$ Generator Weights $(W_v, W_\\beta)$",
             "Produces 4 axes $v_0..v_3 \\in \\mathbb{R}^4$ + scalar parity $\\beta_t \\in [0, 2]$",
             t["sky_soft"], t["sky"], t, title_size=8.2, body_size=7.3)

    draw_box(ax2, 0.055, 0.555, 0.89, 0.195,
             "Unitary 4-Householder Cascade in $SO(5)$ (step134)",
             "$Q_t = H_3(u_3(\\beta_t))\\, H_2(v_2,0)\\, H_1(v_1,0)\\, H_0(v_0,0)\\, Q_{t-1}$\n"
             "$u_3(\\beta) = ((2-\\beta)\\hat{v}_3,\\; \\beta) \\in \\mathbb{R}^5,\\;\\; \\|Q_t^\\top Q_t - I_5\\|_F \\approx 1.1\\times 10^{-4}$\n"
             "Cartan–Dieudonné embedding $\\iota(Q) = \\mathrm{diag}(Q, \\det Q) \\in SO(5)$",
             t["primary_soft"], t["primary"], t, title_size=8.1, body_size=7.1)

    draw_box(ax2, 0.055, 0.295, 0.89, 0.200,
             "Zero-Oracle Hypersphere Proto-Metric on $(S^4)^{5M}$",
             "Logits: $\\ell_c = \\frac{\\tau}{5M}\\sum_s w_{c,s}^\\top q_{t,s}$,  Probs: quartic $\\phi_4(\\ell_c - \\ell_{\\max})$\n"
             "Lazy first-encounter init + Pillar-2 polynomial repulsion (step123)\n"
             "Delayed parity regularizer $\\lambda_{\\mathrm{par}} \\beta(2-\\beta)$ (0 group oracle)",
             t["secondary_soft"], t["secondary"], t, title_size=8.1, body_size=7.1)

    draw_box(ax2, 0.055, 0.055, 0.89, 0.180,
             "Exact Q1.14 (i16) Hypersphere Deployment (step132)",
             "2× prototype compression ($S_5$ M=3: 62.7 KB → 45.1 KB)\n"
             "Argmax margin preserved whenever $\\Delta\\ell > 2\\delta_q$ (100% parity in Step 9)",
             t["amber_soft"], t["amber"], t, title_size=8.0, body_size=7.1)

    draw_arrow(ax2, 0.50, 0.808, 0.50, 0.753, t["primary"], lw=1.7)
    draw_arrow(ax2, 0.50, 0.553, 0.50, 0.498, t["secondary"], lw=1.7)
    draw_arrow(ax2, 0.50, 0.293, 0.50, 0.238, t["amber"], lw=1.7)

    # --- Panel C: Isometric O(d) -> SO(7) Block-Matrix Embedding ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.set_title("C. Isometric $O(d)\\hookrightarrow SO(7)$ Embedding (AdaptiveO(d))", fontweight="bold", pad=8, color=t["text"])

    c_bg = FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.01,rounding_size=0.025",
                          facecolor=t["card"], edgecolor=t["border"], lw=1.0)
    ax3.add_patch(c_bg)

    # Shifted slightly left so both callouts sit in completely open right column!
    gx0, gy0, cell = 0.095, 0.37, 0.076
    n_dim = 7
    active_d = 4
    for r in range(n_dim):
        for c in range(n_dim):
            x = gx0 + c * cell
            y = gy0 + (n_dim - 1 - r) * cell
            if r == 6 and c == 6:
                fc, ec, txt = t["crimson_soft"], t["crimson"], "P"
            elif (r == 6 and c < active_d) or (c == 6 and r < active_d):
                fc, ec, txt = t["amber_soft"], t["amber"], r"$R_p$"
            elif r < active_d and c < active_d:
                fc, ec, txt = t["primary_soft"], t["primary"], r"$G$"
            elif r == c and active_d <= r < 6:
                fc, ec, txt = t["secondary_soft"], t["secondary"], "1"
            else:
                fc, ec, txt = t["card_alt"], t["border"], "0"
            rect = Rectangle((x, y), cell * 0.90, cell * 0.90, facecolor=fc, edgecolor=ec, lw=1.0)
            ax3.add_patch(rect)
            ax3.text(x + cell * 0.45, y + cell * 0.45, txt, ha="center", va="center",
                     fontsize=7.6, fontweight="bold" if txt != "0" else "normal",
                     color=ec if txt != "0" else t["muted"])

    axis_lbls = ["0", "1", "2", "3", "4", "5", "P=6"]
    for i, lbl in enumerate(axis_lbls):
        ax3.text(gx0 + i * cell + cell * 0.45, gy0 + n_dim * cell + 0.022, lbl,
                 ha="center", va="center", fontsize=7.2, fontweight="bold",
                 color=t["crimson"] if i == 6 else (t["primary"] if i < active_d else t["muted"]))
        ax3.text(gx0 - 0.020, gy0 + (n_dim - 1 - i) * cell + cell * 0.45, lbl,
                 ha="right", va="center", fontsize=7.2, fontweight="bold",
                 color=t["crimson"] if i == 6 else (t["primary"] if i < active_d else t["muted"]))

    # Right-column callouts completely outside the 7x7 grid (grid ends at x = 0.095 + 7*0.076 = 0.627!)
    ax3.annotate(
        "Active $O(d)$ block\n($d=4$ shown;\n15 Givens planes)",
        xy=(gx0 + 3.92 * cell, gy0 + 5.45 * cell),
        xytext=(0.815, 0.80),
        fontsize=7.0, fontweight="bold", color=t["primary"], ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc=t["primary_soft"], ec=t["primary"], lw=0.7),
        arrowprops=dict(arrowstyle="->", color=t["primary"], lw=1.2)
    )
    ax3.annotate(
        "Isometric $d \\to d+1$\n(unlocks axis 4,\n$<10^{-6}$ prob jump)",
        xy=(gx0 + 4.90 * cell, gy0 + 2.45 * cell),
        xytext=(0.815, 0.59),
        fontsize=7.0, fontweight="bold", color=t["secondary"], ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc=t["secondary_soft"], ec=t["secondary"], lw=0.7),
        arrowprops=dict(arrowstyle="->", color=t["secondary"], lw=1.2)
    )
    ax3.annotate(
        "Fixed Parity Axis\nP = 6  ($R_v(p)^2$)",
        xy=(gx0 + 6.90 * cell, gy0 + 0.45 * cell),
        xytext=(0.815, 0.40),
        fontsize=7.0, fontweight="bold", color=t["crimson"], ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc=t["crimson_soft"], ec=t["crimson"], lw=0.7),
        arrowprops=dict(arrowstyle="->", color=t["crimson"], lw=1.2)
    )

    # Clean wrapped footer box inside Panel C (zero overflow!)
    draw_box(
        ax3, 0.045, 0.042, 0.91, 0.275,
        "2-Timescale Shuffle-Immune Escalation (fast/slow EMA + $5\\sigma\\, Z^2$)",
        "• Starts at $d=2$; upgrades $d \\to d+1$ when accuracy stalls\n"
        "  AND beats chance by $>5\\sigma$ (0 false triggers on Shuffle)\n"
        "• Fused GivensParityOp (step133): 15 planar Givens + parity\n"
        "  rotation ($\\eta_{\\mathrm{par}} = 0.35\\,\\eta_{\\mathrm{rot}}$), scales up to $S_6$ (720 classes)",
        t["purple_soft"], t["purple"], t, title_size=7.6, body_size=6.8, title_y=0.83, body_y=0.38
    )

    save_figure(fig, "fig02_o4_parity_and_adaptive_geometry", t["name"])


# ==============================================================================
# FIGURE 3: Exact Zero-Transcendental & Zero-sqrt Algebraic Primitives
# ==============================================================================

def make_fig03(t: dict):
    fig = plt.figure(figsize=(15.2, 9.4), facecolor=t["bg"])
    add_header(
        fig,
        "Zero-Transcendental & Zero-sqrt Algebraic Building Blocks (Compile-Time Enforced [E012] & [E013])",
        "Exact mathematical behavior computed directly from src/math.rs, src/fixed_point.rs, and Lean 4 theorems step111–step134",
        t, y_title=0.975, y_sub=0.941
    )

    gs = gridspec.GridSpec(2, 2, left=0.07, right=0.975, top=0.885, bottom=0.075, wspace=0.23, hspace=0.31)

    # --- Panel A ---
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, t)
    z = np.linspace(-4.0, 2.5, 500)
    delta_rat = rational_pos_and_deriv(z)
    softplus = np.log1p(np.exp(z)) / math.log(2.0)
    exp_z = np.exp(z)

    ax.plot(z, delta_rat, color=t["primary"], lw=2.6, label=r"Rational $\Delta(z)$ (step116, 0% exp, $\in [0.05, 2.0]$)")
    ax.plot(z, softplus, color=t["purple"], lw=1.7, ls="--", label=r"Scaled $\mathrm{softplus}(z)/\ln 2$ (transcendental)")
    ax.plot(z, np.minimum(exp_z, 3.2), color=t["crimson"], lw=1.4, ls=":", label=r"$\exp(z)$ (unbounded)")
    ax.scatter([0.0], [1.0], color=t["amber"], s=60, zorder=6)
    ax.axhline(2.0, color=t["slate"], ls=":", lw=1.0)
    ax.axhline(0.05, color=t["slate"], ls=":", lw=1.0)
    ax.annotate("$C^1$ Splice at $z=0$\n$\\Delta(0)=1,\\; \\Delta'(0)=1$", xy=(0.0, 1.0), xytext=(0.45, 0.65),
                fontsize=7.8, fontweight="bold", color=t["amber"],
                arrowprops=dict(arrowstyle="->", color=t["amber"], lw=1.2))
    ax.text(-3.85, 2.07, "BIBO ceiling = 2.0", fontsize=7.5, color=t["muted"])
    ax.text(-3.85, 0.13, "Strict floor = 0.05", fontsize=7.5, color=t["muted"])
    ax.set_xlim(-4.0, 2.5)
    ax.set_ylim(-0.05, 2.65)
    ax.set_xlabel("Selector Pre-Activation $z$")
    ax.set_ylabel(r"Step Scale $\Delta(z)$")
    ax.set_title(r"A. $C^1$ Piecewise-Rational Selective Step (rational_pos_and_deriv)", fontweight="bold", pad=7)
    ax.legend(loc="upper left")

    # --- Panel B ---
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, t)
    zs = np.linspace(-8.0, 0.0, 500)
    phi4 = (1.0 - 0.5 * zs + (zs * zs) / 8.0) ** (-4)
    phi2 = (1.0 - zs + 0.5 * zs * zs) ** (-1)
    exp_s = np.exp(zs)

    ax.plot(zs, phi4, color=t["primary"], lw=2.6, label=r"Quartic $\phi_4(z) = (1 - z/2 + z^2/8)^{-4}$ (step112b)")
    ax.plot(zs, exp_s, color=t["crimson"], lw=1.9, ls="--", label=r"Exact $\exp(z)$ (transcendental baseline)")
    ax.plot(zs, phi2, color=t["amber"], lw=1.7, ls="-.", label=r"Quadratic $\phi_2(z) = (1 - z + z^2/2)^{-1}$ (step112)")
    ax.fill_between(zs, np.minimum(phi4, exp_s), np.maximum(phi4, exp_s), color=t["primary"], alpha=0.14)
    ax.set_xlim(-8.0, 0.15)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel(r"Max-Shifted Logit $z = \ell_c - \ell_{\max} \leq 0$")
    ax.set_ylabel("Unnormalized Weight $w(z)$")
    ax.set_title(r"B. Shift-Invariant Rational Softmax Kernel (0% exp, Strictly $> 0$)", fontweight="bold", pad=7)
    ax.legend(loc="upper left")
    ax.text(-7.6, 0.42, "Matches $\\exp(z)$ tightly on $[-3, 0]$\nwith algebraic tail & exact rational\nderivative $\\psi(z) = \\frac{4-2z}{8-4z+z^2}$",
            fontsize=7.8, color=t["muted"], bbox=dict(boxstyle="round,pad=0.35", fc=t["card_alt"], ec=t["border"], lw=0.8))

    # --- Panel C ---
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, t)
    grid_a = np.logspace(-4.2, 6.0, 260)
    err_f32_2, err_f32_3, err_q24 = [], [], []
    for a in grid_a:
        exact = 1.0 / math.sqrt(float(a))
        err_f32_2.append(max(abs(rust_inv_sqrt_newton_f32(float(a), 2) - exact) / exact, 1e-8))
        err_f32_3.append(max(abs(rust_inv_sqrt_newton_f32(float(a), 3) - exact) / exact, 1e-8))
        err_q24.append(max(abs(rust_q8_24_inv_sqrt(float(a)) - exact) / exact, 1e-8))

    ax.loglog(grid_a, err_f32_2, color=t["amber"], lw=1.7, label="f32 2-step Newton (inv_sqrt_newton_2step_f32, $<1.3\\times 10^{-4}$)")
    ax.loglog(grid_a, err_q24, color=t["purple"], lw=1.7, ls="-.", label="Zero-FPU q8_24_inv_sqrt (3-step i128, Suite C max $2.02\\times 10^{-4}$)")
    ax.loglog(grid_a, err_f32_3, color=t["primary"], lw=2.0, label="f32 3-step Newton (inv_sqrt_newton_f32, $<10^{-6}$)")
    ax.axhline(1.3e-4, color=t["amber"], ls=":", lw=1.0, alpha=0.8)
    ax.axhline(1e-6, color=t["primary"], ls=":", lw=1.0, alpha=0.8)
    ax.set_xlim(1e-4, 1e6)
    ax.set_ylim(5e-9, 2.5e-2)
    ax.set_xlabel(r"Input Magnitude $a$ (10+ Orders of Magnitude, 0 .sqrt())")
    ax.set_ylabel("Relative Error $|\\hat{y} - a^{-1/2}| / a^{-1/2}$")
    ax.set_title("C. Exponent-Bracketed Newton inv_sqrt Accuracy (step129–step130)", fontweight="bold", pad=7)
    ax.legend(loc="lower left", fontsize=7.2)

    ax.text(
        1.2e1, 3.2e-3,
        "Small-step control (lr = 1e-4 × g = 0.05):\n"
        "Q8.24: raw = 84 ($5.01\\times 10^{-6}$)\n"
        "Q16.16: raw = 0 (underflow!)",
        fontsize=7.3, color=t["text"],
        bbox=dict(boxstyle="round,pad=0.35", fc=t["purple_soft"], ec=t["purple"], lw=0.9)
    )

    # --- Panel D ---
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, t)
    alpha = np.linspace(-3.0, 3.0, 400)
    lam_1 = 0.995 / (1.0 + alpha * alpha)
    lam_fast = 0.998 / (1.0 + 2.0 * alpha * alpha)
    lam_slow = 0.998 / (1.0 + 0.05 * alpha * alpha)
    ax.plot(alpha, lam_slow, color=t["secondary"], lw=2.4, label=r"Selective Slow ($\Delta=0.05$): long memory $\lambda \to 0.998$")
    ax.plot(alpha, lam_1, color=t["primary"], lw=2.4, label=r"Nominal ($\Delta=1.0$, step111): $\lambda(\alpha)=\frac{199}{200(1+\alpha^2)}$")
    ax.plot(alpha, lam_fast, color=t["amber"], lw=2.2, ls="--", label=r"Selective Fast ($\Delta=2.0$): rapid reset / gating")
    ax.axhline(1.0, color=t["crimson"], ls=":", lw=1.4, label=r"BIBO Stability Boundary ($\lambda = 1$, never crossed)")
    ax.fill_between(alpha, 0, lam_slow, color=t["secondary"], alpha=0.07)

    # Place drift box cleanly in open lower-center pocket (y in [0.06, 0.34], x centered)
    ax.text(
        0.0, 0.24,
        "Measured Unnormalized Drift (L=500 / 10k steps):\n"
        "• SO(3) L=500 (f32, 10 seeds): $\\|Q^\\top Q - I_3\\|_F = 2.09\\times 10^{-4}$\n"
        "• O(4)↪SO(5) L=500 (f32): $\\|Q^\\top Q - I_5\\|_F = 1.12\\times 10^{-4}$\n"
        "• FixedO4Rotor 10k (Q8.24): 4.69e-3 unnorm / 0.00 renorm",
        fontsize=7.2, color=t["text"], ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc=t["card_alt"], ec=t["border"], lw=0.9, alpha=0.96)
    )

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(0.0, 1.12)
    ax.set_xlabel(r"Learnable Damping Parameter $\alpha$")
    ax.set_ylabel(r"Spectral Contraction Radius $\lambda(\alpha, \Delta)$")
    ax.set_title(r"D. Exact Rational BIBO Contraction (step111 / step115) & Live Drift", fontweight="bold", pad=7)
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.72), fontsize=7.1)

    save_figure(fig, "fig03_zero_transcendental_primitives", t["name"])


# ==============================================================================
# FIGURE 4: Non-Abelian Group Extrapolation (Suites A & D)
# ==============================================================================

def make_fig04(t: dict):
    fig = plt.figure(figsize=(15.5, 9.5), facecolor=t["bg"])
    add_header(
        fig,
        "Non-Abelian Group Extrapolation & Self-Escalating Dimensional Ladder (Suites A & D)",
        "Zero-oracle word tracking from short training horizons (L ≤ 8–10) to L = 500 across C₅, A₅ (60), S₅ (120), and S₆ (720 classes) — 10 seeds",
        t, y_title=0.975, y_sub=0.941
    )

    gs = gridspec.GridSpec(2, 2, left=0.065, right=0.975, top=0.885, bottom=0.075, wspace=0.22, hspace=0.31)

    # --- Panel A: Suite A1 S_5 ---
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, t)
    Ls = np.array([6, 12, 30, 100, 250, 500])
    s5_o4_m3 = np.array([99.5, 96.8, 87.5, 77.2, 67.9, 54.0])
    s5_o4_m3_sd = np.array([1.4, 8.1, 25.5, 33.2, 34.7, 39.7])
    s5_unitary_L = np.array([6, 30, 100, 500])
    s5_unitary = np.array([99.3, 86.3, 78.6, 62.8])
    s5_so3_v04 = np.array([88.4, 54.6, 14.5, 1.6, 1.6, 1.5])
    s5_sel2d = np.array([20.8, 2.5, 1.1, 0.5, 0.8, 0.8])
    s5_gru = np.array([80.2, 26.4, 1.4, 0.7, 1.1, 1.1])
    s5_shuf = np.array([0.9, 0.9, 1.4, 0.7, 0.8, 0.8])

    ax.axvspan(1, 10, color=t["secondary"], alpha=0.09, label="Train horizon ($L \\leq 10$)")
    ax.plot(Ls, s5_o4_m3, "-o", color=t["primary"], lw=2.6, ms=6, label="NonAbelianO4Rtrl M=3 (62.7 KB)")
    ax.fill_between(Ls, np.clip(s5_o4_m3 - s5_o4_m3_sd, 0, 100), np.clip(s5_o4_m3 + s5_o4_m3_sd, 0, 100),
                    color=t["primary"], alpha=0.14)
    ax.plot(s5_unitary_L, s5_unitary, "--D", color=t["secondary"], lw=2.2, ms=5.5, label="Track 2 Unitary $O(4)\\hookrightarrow SO(5)$ (62.7 KB)")
    ax.plot(Ls, s5_so3_v04, "-s", color=t["amber"], lw=1.8, ms=5, label="SO(3) v0.4.0 M=4 (45.8 KB, $S_5 \\not\\leq SO(3)^M$)")
    ax.plot(Ls, s5_gru, "-^", color=t["crimson"], lw=1.7, ms=5, label="GRU-BPTT H=64 (246.7 KB)")
    ax.plot(Ls, s5_sel2d, ":v", color=t["purple"], lw=1.6, ms=4.5, label="Selective 2D U=38 (87.7 KB)")
    ax.plot(Ls, s5_shuf, "--", color=t["slate"], lw=1.4, label="Label-Shuffle Control (0.7–1.4%)")

    ax.annotate("Bit-30 SEU @L=100:\n71.9% (any_nan = false)", xy=(100, 77.2), xytext=(125, 86),
                fontsize=7.3, fontweight="bold", color=t["primary"],
                arrowprops=dict(arrowstyle="->", color=t["primary"], lw=1.1))
    ax.annotate("SO(3) Orientation Barrier\n(collapses to ~1.5% chance)", xy=(30, 14.5), xytext=(45, 29),
                fontsize=7.4, fontweight="bold", color=t["amber"],
                arrowprops=dict(arrowstyle="->", color=t["amber"], lw=1.1))

    ax.set_xscale("log")
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(x) for x in Ls])
    ax.set_xlim(5, 580)
    ax.set_ylim(-2, 105)
    ax.set_xlabel("Evaluation Word Length $L$ (log scale)")
    ax.set_ylabel("Exact 120-Class Accuracy (%)")
    ax.set_title("A. Suite A1: $S_5$ (120 Classes) Breaking the $SO(3)$ Parity Barrier", fontweight="bold", pad=7)
    ax.legend(loc="lower left", fontsize=7.1)

    # --- Panel B: Suite A2 A_5 ---
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, t)
    La = np.array([6, 30, 100, 500])
    a5_o4 = np.array([100.0, 99.9, 98.0, 92.8])
    a5_o4_sd = np.array([0.0, 0.2, 4.0, 8.7])
    a5_so3_v05 = np.array([100.0, 99.3, 91.0, 59.1])
    a5_so3_v05_sd = np.array([0.0, 2.2, 17.4, 38.2])
    a5_so3_v04 = np.array([97.4, 85.7, 71.4, 23.6])
    La_full = np.array([6, 12, 18, 30, 50, 100, 200, 500])
    a5_lin = np.array([30.0, 9.1, 4.8, 3.1, 2.6, 1.7, 1.7, 2.5])
    a5_gru = np.array([52.9, 9.6, 3.3, 1.6, 2.4, 1.5, 1.8, 1.8])
    a5_ssm = np.array([36.2, 4.4, 2.6, 1.0, 2.4, 2.0, 2.2, 1.3])

    ax.axvspan(1, 8, color=t["secondary"], alpha=0.09, label="Train horizon ($L \\leq 8$)")
    ax.plot(La, a5_o4, "-o", color=t["primary"], lw=2.7, ms=6.5, label="NonAbelianO4Rtrl M=3 (43.2 KB) — 92.8% @ L=500")
    ax.fill_between(La, np.clip(a5_o4 - a5_o4_sd, 0, 100), np.clip(a5_o4 + a5_o4_sd, 0, 100), color=t["primary"], alpha=0.16)
    ax.plot(La, a5_so3_v05, "-s", color=t["secondary"], lw=2.2, ms=6, label="SO(3)+Pillar2 v0.5.0 M=4 (26.8 KB)")
    ax.fill_between(La, np.clip(a5_so3_v05 - a5_so3_v05_sd, 0, 100), np.clip(a5_so3_v05 + a5_so3_v05_sd, 0, 100), color=t["secondary"], alpha=0.10)
    ax.plot(La, a5_so3_v04, "--^", color=t["amber"], lw=1.9, ms=5.5, label="SO(3) v0.4.0 M=4 (26.8 KB)")
    ax.plot(La_full, a5_gru, "-v", color=t["crimson"], lw=1.5, ms=4.5, label="GRU-BPTT H=32 (64.8 KB)")
    ax.plot(La_full, a5_ssm, ":D", color=t["purple"], lw=1.5, ms=4.2, label="Selective SSM 36D (25.5 KB)")
    ax.plot(La_full, a5_lin, "-.", color=t["slate"], lw=1.4, label="SO(3) Linear Readout (no Proto-Metric)")

    for lx, ly in zip(La, a5_o4):
        ax.text(lx, ly + 2.5 if lx < 500 else ly + 3.4, f"{ly:.1f}%", ha="center", fontsize=7.6, fontweight="bold", color=t["primary"])

    ax.set_xscale("log")
    ax.set_xticks([6, 12, 30, 100, 200, 500])
    ax.set_xticklabels(["6", "12", "30", "100", "200", "500"])
    ax.set_xlim(5, 580)
    ax.set_ylim(-2, 109)
    ax.set_xlabel("Evaluation Word Length $L$ (log scale)")
    ax.set_ylabel("Exact 60-Class Accuracy (%)")
    ax.set_title("B. Suite A2: $A_5$ (60 Classes) Long-Horizon Extrapolation ($L \\to 500$)", fontweight="bold", pad=7)
    ax.legend(loc="lower left", fontsize=7.1)

    # --- Panel C: Suite D Autonomous Dimensional Escalation (Zero Collisions!) ---
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, t, grid_axis="y")
    tasks_d = ["Label-Shuffle\n(Null Control)", "$C_5$\n(5 cls)", "$A_5$\n(60 cls)", "$S_5$\n(120 cls)", "$S_6$\n(720 cls)"]
    d_mean = np.array([2.00, 2.20, 4.80, 5.80, 6.00])
    d_sd = np.array([0.00, 0.63, 1.32, 0.42, 0.00])
    d_min_theory = [2.0, 2.0, 3.0, 4.0, 5.0]
    colors_d = [t["slate"], t["sky"], t["secondary"], t["amber"], t["primary"]]

    x_pos = np.arange(len(tasks_d))
    ax.bar(x_pos, d_mean, width=0.55, color=colors_d, edgecolor=t["border"], lw=1.0, zorder=3)
    ax.errorbar(x_pos, d_mean, yerr=d_sd, fmt="none", ecolor=t["text"], capsize=4.5, elinewidth=1.4, zorder=4)
    ax.scatter(x_pos, d_min_theory, marker="D", s=65, color=t["crimson"], zorder=5,
               label="Minimal faithful $O(d)$ dimension ($d_{\\min} = 2, 2, 3, 4, 5$)")

    for i, (m, sd) in enumerate(zip(d_mean, d_sd)):
        ax.text(i, m + sd + 0.22, f"$d={m:.2f}$", ha="center", va="bottom",
                fontsize=8.2, fontweight="bold", color=t["text"])

    ax.annotate("0 false escalations\nacross 10/10 seeds", xy=(0, 2.1), xytext=(0.45, 3.85),
                fontsize=7.5, fontweight="bold", color=t["slate"], ha="center",
                arrowprops=dict(arrowstyle="->", color=t["slate"], lw=1.2))
    ax.annotate("Honest note: slight\nover-escalation above $d_{\\min}$",
                xy=(1.95, 5.05), xytext=(0.65, 5.65),
                fontsize=7.3, fontweight="bold", color=t["crimson"], ha="center",
                bbox=dict(boxstyle="round,pad=0.25", fc=t["crimson_soft"], ec=t["crimson"], lw=0.7),
                arrowprops=dict(arrowstyle="->", color=t["crimson"], lw=1.1))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(tasks_d, fontsize=8.3)
    ax.set_ylim(0, 7.65)
    ax.set_yticks([2, 3, 4, 5, 6])
    ax.set_ylabel("Autonomously Discovered Active Dimension $d$")
    ax.set_title("C. Suite D: Shuffle-Immune Autonomous Rank Escalation (AdaptiveO(d))", fontweight="bold", pad=7)
    ax.legend(loc="upper left", fontsize=7.4)

    # --- Panel D: Suite D Accuracy vs Horizon ---
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, t)
    d_c5 = np.array([97.2, 92.2, 87.8, 84.7, 86.2, 88.8])
    d_a5 = np.array([78.8, 57.6, 40.7, 25.2, 21.7, 21.4])
    d_s5 = np.array([81.3, 38.1, 11.8, 8.2, 4.3, 2.8])
    d_s6 = np.array([77.2, 25.8, 0.7, 0.2, 0.1, 0.1])
    d_sh = np.array([0.5, 0.8, 0.8, 0.8, 0.6, 1.2])

    ax.plot(Ls, d_c5, "-o", color=t["sky"], lw=2.3, ms=5.5, label="$C_5$ (5 cls, 15.3 KB, $d=2.20$)")
    ax.plot(Ls, d_a5, "-s", color=t["secondary"], lw=2.2, ms=5.5, label="$A_5$ (60 cls, 44.8 KB, $d=4.80$)")
    ax.plot(Ls, d_s5, "-^", color=t["amber"], lw=2.2, ms=5.5, label="$S_5$ (120 cls, 63.1 KB, $d=5.80$)")
    ax.plot(Ls, d_s6, "-D", color=t["primary"], lw=2.5, ms=6, label="$S_6$ (720 cls, 245.9 KB, $d=6.00$; chance 0.14%)")
    ax.plot(Ls, d_sh, "--", color=t["slate"], lw=1.4, label="Label-Shuffle Control (0.5–1.2%)")

    ax.annotate("$S_6$ 720-class capture:\n77.2% @L=6 (550× chance!)\nHonest decay: 0.7% @L=30",
                xy=(6, 77.2), xytext=(16, 57),
                fontsize=7.4, fontweight="bold", color=t["primary"],
                bbox=dict(boxstyle="round,pad=0.3", fc=t["primary_soft"], ec=t["primary"], lw=0.8),
                arrowprops=dict(arrowstyle="->", color=t["primary"], lw=1.2))

    ax.set_xscale("log")
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(x) for x in Ls])
    ax.set_xlim(5, 580)
    ax.set_ylim(-2, 105)
    ax.set_xlabel("Evaluation Word Length $L$ (log scale)")
    ax.set_ylabel("Exact Group-Element Accuracy (%)")
    ax.set_title("D. Suite D: Adaptive Frame Accuracy & Long-Horizon Tradeoff", fontweight="bold", pad=7)
    ax.legend(loc="upper right", fontsize=7.2)

    save_figure(fig, "fig04_nonabelian_groups_and_ladder", t["name"])


# ==============================================================================
# FIGURE 5: Unified One-Config Challenge (Suite B) & 7-Stage Internal Ablation
# ==============================================================================

def make_fig05(t: dict):
    fig = plt.figure(figsize=(15.6, 9.5), facecolor=t["bg"])
    add_header(
        fig,
        "Unified One-Config Universality Challenge (Suite B) & 7-Stage Internal Architecture Ablation",
        "One fixed hyperparameter preset (26.2–34.6 KB, 6000/6000 Causal Prefix Invariance) across 8 tasks vs specialized v0.4.1 ceilings",
        t, y_title=0.975, y_sub=0.941
    )

    gs = gridspec.GridSpec(2, 2, left=0.132, right=0.975, top=0.885, bottom=0.075,
                           wspace=0.23, hspace=0.34, height_ratios=[1.12, 1.0])

    # --- Panel A ---
    ax = fig.add_subplot(gs[0, :])
    style_ax(ax, t, grid_axis="y")

    tasks = [
        "ELEC2\nRaw (%)",
        "NASA FD001\n(RMSE cyc)",
        "NASA FD004\n(RMSE cyc)",
        "CartPole 2D\n(Train st)",
        "CartPole +dx\n(Train st)",
        "RepeatPrev\nK=4 (%)",
        "RepeatPrev\nK=8 (%)",
        "Dyck-2\nAll (%)",
        "Waves\nZeroShot (%)",
        "T-Maze L=15\nRet (ng=0)",
    ]
    b1_norm = [85.45, (1 - 24.05/50)*100, (1 - 28.24/50)*100, 44.9/2, 91.2/2, 85.4, 66.4, 87.2, 90.0, 84.2]
    b2_norm = [85.35, (1 - 22.69/50)*100, (1 - 26.48/50)*100, 50.3/2, 89.0/2, 83.5, 68.3, 86.1, 89.0, 92.2]
    b3_norm = [84.96, (1 - 23.89/50)*100, (1 - 26.94/50)*100, 64.2/2, 91.0/2, 76.8, 58.0, 86.9, 89.7, 97.2]
    spec_norm = [91.50, (1 - 19.51/50)*100, (1 - 23.33/50)*100, 117.6/2, 187.4/2, 99.7, 85.1, 72.9, 85.0, 56.2]

    b3_labels = ["85.0%", "23.9", "26.9", "64", "91", "76.8%", "58.0%", "86.9%", "89.7%", "97.2%"]
    spec_labels = ["91.5%", "19.5", "23.3", "118", "187", "99.7%", "85.1%", "72.9%", "85.0%", "56.2%"]

    x = np.arange(len(tasks))
    w = 0.20
    ax.bar(x - 1.5*w, b1_norm, w, color=t["sky"], alpha=0.80, label="B1 Base (Master-Eq + Newton RMSNorm)")
    ax.bar(x - 0.5*w, b2_norm, w, color=t["amber"], alpha=0.85, label="B2 (+Selectivity $\\beta_k$ + Local Feedback)")
    ax.bar(x + 0.5*w, b3_norm, w, color=t["primary"], edgecolor=t["text"], lw=0.7,
           label="B3 One-Config (Shared Backbone + Bartlett PH, 26–35 KB)")
    ax.bar(x + 1.5*w, spec_norm, w, color=t["secondary"], alpha=0.72, hatch="//", edgecolor=t["border"],
           label="Best Specialized v0.4.1 Config (per-task ceiling)")

    for i in range(len(tasks)):
        y_b3 = b3_norm[i] + 1.8
        y_sp = spec_norm[i] + 1.8
        if abs(y_b3 - y_sp) < 5.5 and i != 7:
            y_b3 = max(b3_norm[i], b2_norm[i]) + 2.2
        ax.text(x[i] + 0.42*w, y_b3, b3_labels[i], ha="center", va="bottom",
                fontsize=6.6, fontweight="bold", color=t["primary"])
        ax.text(x[i] + 1.65*w, y_sp, spec_labels[i], ha="center", va="bottom",
                fontsize=6.5, fontweight="bold", color=t["secondary"])

    # Clean pill badge for Dyck-2 memoryless 0-step floor (71.8%)
    ax.plot([7 - 2.0*w, 7 + 2.0*w], [71.8, 71.8], color=t["crimson"], ls="--", lw=1.7, zorder=5)
    ax.text(7.0, 58.0, "0-step floor\n71.8%", ha="center", va="center", fontsize=6.5, fontweight="bold",
            color=t["crimson"], bbox=dict(boxstyle="round,pad=0.22", fc=t["card"], ec=t["crimson"], lw=0.8, alpha=0.95), zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=8.0)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Normalized Score (0–100%)\n[NASA: $100(1-\\mathrm{RMSE}/50)$, CP: $\\mathrm{st}/2$]")
    ax.set_title("A. Suite B Cross-Domain One-Config Profile (B1 → B2 → B3) vs Specialized v0.4.1 — Honest Price of Universality",
                 fontweight="bold", pad=7)
    ax.legend(loc="upper center", ncol=4, fontsize=7.4, bbox_to_anchor=(0.5, 1.005))

    # --- Panel B: 7-Stage Internal Ablation (Zero Clipping!) ---
    ax = fig.add_subplot(gs[1, 0])
    stages = [
        "[A0] Frozen Rotor",
        "[A1] +1L Frozen",
        "[A2] +1L Trained",
        "[A3-Fr] Decoupled (Fr)",
        "[A3] Decoupled (Tr)",
        "[A3+Sk] +1L + SkipX",
        "[A4] Unified 2L + PH",
    ]
    cols = ["RPrev\nK=4", "RPrev\nK=6", "CP-2D\n(st)", "CP+dx\n(st)", "Wave\nZS", "ELEC2\nAcc"]
    raw_mat = np.array([
        [58.0, 50.1,  27.4, 162.4, 63.8, 80.6],
        [56.8, 51.0,  55.8, 172.4, 65.7, 85.3],
        [58.7, 51.6,  77.5, 183.4, 64.7, 88.1],
        [71.2, 62.8,  35.2, 173.5, 91.0, 85.1],
        [76.7, 66.3,  50.3, 180.9, 88.8, 88.3],
        [74.5, 65.9,  48.0, 180.8, 86.5, 89.2],
        [85.8, 77.5, 125.6, 187.7, 82.7, 89.9],
    ])
    norm_mat = (raw_mat - raw_mat.min(axis=0)) / np.maximum(raw_mat.max(axis=0) - raw_mat.min(axis=0), 1e-6)
    ax.imshow(norm_mat, cmap=t["heat_cmap"], aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=8.0)
    ax.set_yticks(np.arange(len(stages)))
    ax.set_yticklabels(stages, fontsize=8.0)
    for r in range(raw_mat.shape[0]):
        for c in range(raw_mat.shape[1]):
            v = raw_mat[r, c]
            txt = f"{v:.1f}" if c in (2, 3) else f"{v:.1f}%"
            nv = norm_mat[r, c]
            tc = ("#FFFFFF" if nv > 0.58 else "#0F172A") if t["name"] == "paper" else ("#0B1220" if nv > 0.62 else "#E6EDF7")
            ax.text(c, r, txt, ha="center", va="center", fontsize=7.8,
                    fontweight="bold" if r == 6 or nv > 0.95 else "normal", color=tc)
    for sp in ax.spines.values():
        sp.set_color(t["border"])
    ax.set_title("B. 7-Stage Internal Ablation (bench_unified_ablation)", fontweight="bold", pad=7)

    # --- Panel C: Zero-Gate T-Maze Breakthrough (Clean Labels!) ---
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, t, grid_axis="y")
    tm_models = [
        "MicroCore v0.4\n(ng=0)",
        "RTRRL-LRU\n(AAAI'25)",
        "Online GRU\n+A2C",
        "Alg.Sel.Cayley\nv0.4 (ng=0)",
        "Unified B3\nv0.5 (ng=0)",
    ]
    l5_sc = [49.0, 56.6, 76.8, 65.1, 98.1]
    l15_sc = [49.8, 51.1, 47.5, 53.8, 94.9]
    l15_ret = [51.0, 51.0, 47.0, 56.2, 97.2]

    xt = np.arange(len(tm_models))
    wt = 0.26
    ax.bar(xt - wt, l5_sc, wt, color=t["sky"], label="$L=5$ ($\\pm 0.06$) Score")
    ax.bar(xt, l15_sc, wt, color=t["primary"], label="$L=15$ ($\\pm 0.35$) Score")
    ax.bar(xt + wt, l15_ret, wt, color=t["secondary"], label="$L=15$ $A\\to B\\to A$ Retention")
    ax.axhline(50.0, color=t["crimson"], ls="--", lw=1.3, label="Random Chance (50%)")

    for i in range(len(tm_models)):
        # Annotate L=15 Score and Retention cleanly staggered without touching
        ax.text(xt[i] - 0.02, l15_sc[i] + 2.2, f"{l15_sc[i]:.0f}%", ha="center", va="bottom",
                fontsize=7.2, fontweight="bold" if i == 4 else "normal", color=t["primary"])
        ax.text(xt[i] + wt + 0.04, l15_ret[i] + 5.8 if i < 4 else l15_ret[i] + 2.2,
                f"{l15_ret[i]:.1f}%", ha="center", va="bottom",
                fontsize=7.0, fontweight="bold" if i == 4 else "normal", color=t["secondary"])

    ax.annotate("0 novelty_gate!\nMaster-Eq $g_{\\mathrm{in}} = 2/3$\n+ RMSNorm + Bartlett PH",
                xy=(3.98, 95.5), xytext=(1.45, 92.5),
                fontsize=7.4, fontweight="bold", color=t["primary"],
                bbox=dict(boxstyle="round,pad=0.3", fc=t["primary_soft"], ec=t["primary"], lw=0.8),
                arrowprops=dict(arrowstyle="->", color=t["primary"], lw=1.2))

    ax.set_xticks(xt)
    ax.set_xticklabels(tm_models, fontsize=7.8)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Episode Success / Retention (%)")
    ax.set_title("C. Non-Stationary T-Maze Without Hardcoded Gating (ng = 0.0)", fontweight="bold", pad=7)
    ax.legend(loc="upper left", fontsize=7.0)

    save_figure(fig, "fig05_unified_one_config_and_ablations", t["name"])


# ==============================================================================
# FIGURE 6: Scaling Power Laws, Deep Working Memory, Continual Regimes & SEU
# ==============================================================================

def make_fig06(t: dict):
    fig = plt.figure(figsize=(15.5, 9.5), facecolor=t["bg"])
    add_header(
        fig,
        "Systemic Scaling Laws, Deep Working Memory, Continual Bank Probing & Hardware SEU Audit",
        "Live Axis 1A width scaling (D = 12 → 768), Axis 3B RepeatPrevious (36D, 4.9 KB), Axis 4B Regimes (R = 2 → 32), and Symmetric Bit-30 SEU",
        t, y_title=0.975, y_sub=0.941
    )

    gs = gridspec.GridSpec(2, 2, left=0.068, right=0.975, top=0.885, bottom=0.075, wspace=0.24, hspace=0.31)

    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.30)
    dims = np.array([12, 24, 48, 96, 192, 384, 768])
    ram_cay = np.array([1.6, 3.2, 6.3, 12.6, 25.1, 50.1, 100.1])
    ram_lru = np.array([3.0, 5.3, 9.8, 18.9, 37.1, 73.5, 146.2])
    ram_ssm = np.array([4.2, 8.0, 15.5, 30.5, 60.5, 120.5, 240.5])
    ram_gru = np.array([10.0, 29.9, 100.2, 362.4, 1372.6, 5337.1, 21042.1])

    lat_cay = np.array([1.31, 2.41, 4.60, 9.02, 18.56, 35.49, 72.58])
    lat_lru = np.array([1.08, 1.91, 3.43, 6.77, 13.25, 26.17, 53.41])
    lat_ssm = np.array([1.76, 3.17, 6.05, 11.64, 23.60, 45.40, 93.84])
    lat_gru = np.array([2.58, 5.76, 13.88, 40.67, 121.78, 452.64, 1517.84])

    ax_r = fig.add_subplot(gs_a[0, 0])
    style_ax(ax_r, t)
    ax_r.loglog(dims, ram_gru, "-^", color=t["crimson"], lw=2.2, ms=5, label="GRU ($D^{1.85}$)")
    ax_r.loglog(dims, ram_ssm, "-D", color=t["purple"], lw=1.8, ms=4.5, label="SSM ($D^{0.97}$)")
    ax_r.loglog(dims, ram_lru, "-s", color=t["amber"], lw=1.8, ms=4.5, label="LRU ($D^{0.94}$)")
    ax_r.loglog(dims, ram_cay, "-o", color=t["primary"], lw=2.5, ms=5.5, label="Cayley ($D^{0.99}$)")
    ax_r.annotate("21.0 MB\n(210×!)", xy=(768, 21042), xytext=(95, 8500),
                  fontsize=7.3, fontweight="bold", color=t["crimson"],
                  arrowprops=dict(arrowstyle="->", color=t["crimson"], lw=1.1))
    ax_r.annotate("100.1 KB", xy=(768, 100.1), xytext=(145, 14),
                  fontsize=7.5, fontweight="bold", color=t["primary"],
                  arrowprops=dict(arrowstyle="->", color=t["primary"], lw=1.1))
    ax_r.set_xticks([12, 48, 192, 768])
    ax_r.set_xticklabels(["12", "48", "192", "768"])
    ax_r.set_xlabel("State Dim $D$")
    ax_r.set_ylabel("RAM Footprint (KB, log)")
    ax_r.set_title("A1. Width RAM Law ($O(D)$ vs $O(D^2)$)", fontweight="bold", fontsize=9.5, pad=6)
    ax_r.legend(loc="upper left", fontsize=7.0)

    ax_l = fig.add_subplot(gs_a[0, 1])
    style_ax(ax_l, t)
    ax_l.loglog(dims, lat_gru, "-^", color=t["crimson"], lw=2.2, ms=5, label="GRU ($D^{1.55}$)")
    ax_l.loglog(dims, lat_ssm, "-D", color=t["purple"], lw=1.8, ms=4.5, label="SSM ($D^{0.96}$)")
    ax_l.loglog(dims, lat_cay, "-o", color=t["primary"], lw=2.5, ms=5.5, label="Cayley ($D^{0.97}$)")
    ax_l.loglog(dims, lat_lru, "-s", color=t["amber"], lw=1.8, ms=4.5, label="LRU ($D^{0.94}$)")
    ax_l.annotate("1,518 µs\n(21× slower)", xy=(768, 1518), xytext=(68, 560),
                  fontsize=7.2, fontweight="bold", color=t["crimson"],
                  arrowprops=dict(arrowstyle="->", color=t["crimson"], lw=1.1))
    ax_l.set_xticks([12, 48, 192, 768])
    ax_l.set_xticklabels(["12", "48", "192", "768"])
    ax_l.set_xlabel("State Dim $D$")
    ax_l.set_ylabel("Step Latency (µs, log)")
    ax_l.set_title("A2. Step Latency Law (0 alloc/step)", fontweight="bold", fontsize=9.5, pad=6)
    ax_l.legend(loc="upper left", fontsize=7.0)

    # --- Panel B ---
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, t)
    Ks = np.array([2, 4, 8, 12, 16, 24])
    rp_2d = np.array([100.0, 99.7, 85.1, 66.2, 53.1, 36.6])
    rp_2d_sd = np.array([0.0, 0.2, 2.7, 1.6, 4.4, 2.5])
    rp_sel = np.array([100.0, 98.9, 71.8, 55.6, 42.8, 33.5])
    rp_ssm = np.array([100.0, 96.8, 69.9, 53.2, 39.3, 33.1])
    rp_3d = np.array([89.4, 73.7, 66.3, 29.2, 27.6, 25.4])
    rp_gru = np.array([100.0, 33.2, 25.2, 25.2, 25.4, 24.6])

    ax.plot(Ks, rp_2d, "-o", color=t["primary"], lw=2.6, ms=6, label="2D Rat. Cayley (36D, 4.9 KB, 0% exp)")
    ax.fill_between(Ks, rp_2d - rp_2d_sd, rp_2d + rp_2d_sd, color=t["primary"], alpha=0.14)
    ax.plot(Ks, rp_sel, "-s", color=t["secondary"], lw=2.2, ms=5.5, label="Alg.Sel.Cayley (36D, 4.9 KB, 0% exp)")
    ax.plot(Ks, rp_ssm, "--D", color=t["purple"], lw=2.0, ms=5.5, label="Selective SSM (36D, transcendental exp)")
    ax.plot(Ks, rp_3d, "-.^", color=t["amber"], lw=1.7, ms=5, label="3D Cayley b=1 (36D)")
    ax.plot(Ks, rp_gru, ":v", color=t["crimson"], lw=1.7, ms=5, label="GRU H=32 BPTT4 (truncated @ K ≥ 4)")
    ax.axhline(25.0, color=t["slate"], ls=":", lw=1.2, label="Random Chance (25%, 4 symbols)")

    ax.annotate("+15.2 pp over\nSelective SSM!", xy=(8, 85.1), xytext=(11.5, 86.5),
                fontsize=7.7, fontweight="bold", color=t["primary"],
                arrowprops=dict(arrowstyle="->", color=t["primary"], lw=1.2))
    ax.set_xticks(Ks)
    ax.set_xlim(1, 25.5)
    ax.set_ylim(18, 105)
    ax.set_xlabel("Delay Horizon $K$ (Steps)")
    ax.set_ylabel("Symbol Recall Accuracy (%)")
    ax.set_title("B. Axis 3B: Deep Working Memory (RepeatPrevious, Matched 36D)", fontweight="bold", pad=7)
    ax.legend(loc="upper right", fontsize=7.1)

    # --- Panel C ---
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, t)
    Rs = np.array([2, 4, 8, 16, 32])
    r_seq = np.array([86.0, 82.5, 75.0, 77.2, 75.8])
    r_par6 = np.array([86.0, 83.5, 84.5, 85.5, 84.1])
    r_par6_sd = np.array([3.3, 5.5, 3.7, 2.2, 5.4])
    r_1b = np.array([83.8, 85.1, 85.5, 88.0, 88.0])

    ax.plot(Rs, r_par6, "-o", color=t["primary"], lw=2.6, ms=6.5, label="Core + Causal Par6 Parallel Probing (5.05–5.57 µs, O(1))")
    ax.fill_between(Rs, r_par6 - r_par6_sd, r_par6 + r_par6_sd, color=t["primary"], alpha=0.14)
    ax.plot(Rs, r_1b, "--s", color=t["secondary"], lw=2.0, ms=5.5, label="True 1-Bank Learner (PH disabled)")
    ax.plot(Rs, r_seq, "-^", color=t["amber"], lw=2.1, ms=5.5, label="Core Sequential Probing (baseline)")

    for r_val, ys, yp in [(8, 75.0, 84.5), (16, 77.2, 85.5), (32, 75.8, 84.1)]:
        ax.annotate("", xy=(r_val, yp - 0.6), xytext=(r_val, ys + 0.6),
                    arrowprops=dict(arrowstyle="<->", color=t["primary"], lw=1.3))
        ax.text(r_val * 1.07, 0.5 * (ys + yp), f"+{yp - ys:.1f} pp", fontsize=7.4, fontweight="bold", color=t["primary"])

    ax.set_xscale("log", base=2)
    ax.set_xticks(Rs)
    ax.set_xticklabels([f"R={r}" for r in Rs])
    ax.set_xlim(1.75, 48.0)
    ax.set_ylim(68, 95)
    ax.set_xlabel("Number of Cyclical Regimes $R$ (log2 scale)")
    ax.set_ylabel("Regime Return Accuracy (%)")
    ax.set_title("C. Axis 4B: Multi-Regime Continual Retention (Seq vs Par6)", fontweight="bold", pad=7)
    ax.legend(loc="lower left", fontsize=7.3)

    # --- Panel D ---
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, t, grid_axis="y")
    seu_m = [
        "MicroCore\n[A4]",
        "Deep 2L LRU\n(+Clamp)",
        "Deep 2L LRU\n(+FinGuard)",
        "1L LRU\n(+Clamp)",
        "1L LRU\n(+FinGuard)",
        "Sel. SSM\n(+Clamp)",
        "Sel. SSM\n(+FinGuard)",
    ]
    s_clean = [84.6, 71.3, 71.3, 62.1, 62.1, 34.1, 34.1]
    s_state = [83.4, 33.6, 72.2, 31.7, 62.6, 25.5, 34.4]
    s_weight = [79.7, 24.9, 69.2, 24.9, 60.3, 24.9, 33.0]
    finite_tag = ["10/10", "0/10\nNaN", "10/10", "0/10\nNaN", "10/10", "0/10\nNaN", "10/10"]

    xs = np.arange(len(seu_m))
    ws = 0.25
    ax.bar(xs - ws, s_clean, ws, color=t["sky"], label="Clean $K=4$")
    ax.bar(xs, s_state, ws, color=t["primary"], label="Bit-30 State-SEU (/25 st)")
    ax.bar(xs + ws, s_weight, ws, color=t["amber"], label="Bit-30 SRAM Weight-SEU")
    ax.axhline(25.0, color=t["slate"], ls=":", lw=1.2, label="Random Chance (25%)")

    for i, tag in enumerate(finite_tag):
        is_dead = "NaN" in tag
        top_y = max(s_clean[i], s_state[i], s_weight[i]) + 2.5
        ax.text(xs[i], top_y, tag, ha="center", va="bottom", fontsize=6.9, fontweight="bold",
                color=t["crimson"] if is_dead else t["secondary"])

    ax.set_xticks(xs)
    ax.set_xticklabels(seu_m, fontsize=7.3)
    ax.set_ylim(0, 102)
    ax.set_ylabel("RepeatPrevious $K=4$ Accuracy (%)")
    ax.set_title("D. Suite 2A: Physical-SRAM Bit-30 SEU Audit (Finite Seeds Annotated)", fontweight="bold", pad=7)
    ax.legend(loc="upper right", fontsize=7.1)

    save_figure(fig, "fig06_scaling_memory_and_seu", t["name"])


# ==============================================================================
# FIGURE 7: Real Streams (ELEC2), CartPole 3C, Causal Lorenz-96 & Swarm Pareto
# ==============================================================================

def make_fig07(t: dict):
    fig = plt.figure(figsize=(15.6, 9.6), facecolor=t["bg"])
    add_header(
        fig,
        "Real-World Data Streams (ELEC2), Position-Only CartPole, Causal Chaos (Lorenz-96) & Swarm Pareto",
        "Independent external controls (river 0.27.0, PyTorch 2.14.0), strict dimension parity, and 16-thread 4,096-agent swarm throughput",
        t, y_title=0.975, y_sub=0.941
    )

    gs = gridspec.GridSpec(2, 2, left=0.068, right=0.975, top=0.885, bottom=0.075, wspace=0.23, hspace=0.32)

    # --- Panel A: ELEC2 (Zero label collisions!) ---
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, t)
    elec_pts = [
        # (name, lat, acc, ram, col, mk, tx, ty, ha)
        ("MicroCore+2L (PH=8, raw)\n91.5% | SEU=YES", 6.96, 91.5, 27.0, t["primary"], "o", 1.15, 92.2, "left"),
        ("Alg.Sel.Cayley (+y, 0% exp)\n92.0% | SEU=YES", 12.19, 92.0, 17.5, t["secondary"], "s", 22.0, 92.6, "left"),
        ("Online LRU (+y, exp)\n93.4% | SEU=NO", 4.22, 93.4, 11.3, t["amber"], "D", 6.2, 95.0, "left"),
        ("Online LRU (raw)\n89.3%", 4.09, 89.3, 10.5, t["amber"], "d", 1.85, 88.8, "left"),
        ("2D Rat. Cayley (raw)\n89.4%", 8.00, 89.4, 11.5, t["sky"], "p", 16.0, 88.8, "left"),
        ("Online MLP H=32\n87.3%", 1.44, 87.3, 5.9, t["crimson"], "^", 1.18, 83.2, "left"),
        ("Unified One-Config\n85.0% (1 preset)", 14.0, 84.96, 34.6, t["purple"], "P", 17.5, 82.0, "left"),
        ("river ARF (v0.27, raw)\n88.3% | 75 µs", 75.0, 88.26, 40.0, t["slate"], "X", 34.0, 86.2, "left"),
        ("Fixed Q8.24 (0-FPU)\n74.7% | 2.2 µs", 2.17, 74.65, 2.3, t["sky"], "*", 3.3, 74.8, "left"),
    ]
    for name, lat, acc, ram, col, mk, tx, ty, ha in elec_pts:
        sz = 45 + ram * 3.2
        ax.scatter([lat], [acc], s=sz, color=col, marker=mk, edgecolors=t["text"], linewidths=0.7, zorder=5)
        ax.annotate(name, xy=(lat, acc), xytext=(tx, ty), ha=ha, va="center",
                    fontsize=6.8, fontweight="bold" if col in (t["primary"], t["secondary"]) else "normal",
                    color=col, arrowprops=dict(arrowstyle="-", color=col, lw=0.75, alpha=0.65))

    ax.axhline(85.33, color=t["slate"], ls="--", lw=1.2, label="No-Change Persistence (85.33%)")
    ax.axhspan(52.0, 57.55, color=t["crimson"], alpha=0.09, label="Label-Shuffle / Majority Floor (52–57.6%)")
    ax.set_xscale("log")
    ax.set_xlim(1.0, 125.0)
    ax.set_ylim(71.5, 97.4)
    ax.set_xlabel("Per-Step Latency (µs, log scale; marker area $\\propto$ RAM)")
    ax.set_ylabel("1-Pass Test-then-Train Accuracy (%)")
    ax.set_title("A. Real ELEC2 Stream (45,312 Rows): Accuracy vs Latency", fontweight="bold", pad=7)
    ax.legend(loc="lower right", fontsize=7.1)

    # --- Panel B: Suite 3C CartPole (Legend cleanly ABOVE bars!) ---
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, t, grid_axis="y")
    cp_models = [
        "MicroCore\nB0+2L\n(15.9 KB)",
        "MicroCore\nPURE 2L\n(23.9 KB)",
        "Deep 2L\nLRU\n(63.7 KB)",
        "Deep 2L\n+Spinor\n(334 KB)",
        "PyTorch\nGRU H=32\n(ext)",
        "Sel. SSM\n96D\n(76.4 KB)",
    ]
    cp2d_tr = [117.6, 73.4, 63.3, 64.8, 74.8, 31.3]
    cp2d_ev = [168.7, 149.7, 137.4, 94.5, 130.7, 33.3]
    cpdx_ev = [196.2, 178.6, 193.7, 196.6, 174.2, 188.8]

    xc = np.arange(len(cp_models))
    wc = 0.25
    ax.bar(xc - wc, cp2d_tr, wc, color=t["primary"], label="CP-2D Train Score")
    ax.bar(xc, cp2d_ev, wc, color=t["secondary"], label="CP-2D Frozen Eval (S)")
    ax.bar(xc + wc, cpdx_ev, wc, color=t["amber"], alpha=0.85, label="CP+dx Frozen Eval (S)")

    ax.axhline(50.8, color=t["crimson"], ls="--", lw=1.3, label="Memoryless MLP (50.8 st)")
    ax.axhline(200.0, color=t["slate"], ls=":", lw=1.1, label="Oracle 4D-PD (200 st)")

    for i in range(len(cp_models)):
        ax.text(xc[i] - wc, cp2d_tr[i] + 3.5, f"{cp2d_tr[i]:.0f}", ha="center", va="bottom", fontsize=6.9,
                fontweight="bold" if i == 0 else "normal", color=t["primary"])
        ax.text(xc[i], cp2d_ev[i] + 3.5, f"{cp2d_ev[i]:.0f}", ha="center", va="bottom", fontsize=6.9,
                fontweight="bold" if i == 0 else "normal", color=t["secondary"])

    ax.set_xticks(xc)
    ax.set_xticklabels(cp_models, fontsize=7.5)
    ax.set_ylim(0, 250)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_ylabel("Episode Length (Steps, max = 200)")
    ax.set_title("B. Suite 3C: Position-Only CartPole (Matched 96D + Frozen Eval)", fontweight="bold", pad=7)
    ax.legend(loc="upper center", ncol=3, fontsize=6.9, bbox_to_anchor=(0.5, 1.005))

    # --- Panel C: Axis 4A Causal Lorenz-96 ---
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, t)
    d_in = np.array([4, 16, 32, 64, 128, 256])
    l96_cay = np.array([86.7, 78.3, 76.1, 73.5, 70.7, 72.1])
    l96_gru = np.array([91.0, 74.3, 73.2, 68.0, 68.4, 66.1])

    ax.plot(d_in, l96_cay, "-o", color=t["primary"], lw=2.6, ms=6.5, label="Direct Cayley S=16 (98.5 KB @ d=256, 0% exp)")
    ax.plot(d_in, l96_gru, "-^", color=t["crimson"], lw=2.1, ms=6, label="Online GRU H=32 BPTT4 (442.8 KB @ d=256, 4.5× RAM)")
    ax.scatter([4, 256], [87.1, 63.5], color=t["secondary"], marker="D", s=65, zorder=6,
               label="Count-Sketch Proj(d→8) Cayley (9.5 KB, 5.6 µs, 13.6× faster)")
    ax.plot([4, 256], [87.1, 63.5], ":", color=t["secondary"], lw=1.4, alpha=0.8)

    for xv, yc, yg in zip(d_in, l96_cay, l96_gru):
        if xv >= 16:
            ax.text(xv, yc + 2.0, f"{yc:.1f}%", ha="center", fontsize=7.3, fontweight="bold", color=t["primary"])
            if xv == 256:
                ax.text(xv * 0.84, yg + 1.1, f"{yg:.1f}%", ha="right", fontsize=7.0, color=t["crimson"])
            else:
                ax.text(xv, yg - 2.8, f"{yg:.1f}%", ha="center", fontsize=7.0, color=t["crimson"])

    ax.text(
        18, 52.3,
        "Honest Negative (1D Lorenz-63 K=8, strictly causal):\n"
        "All 1-pass models sit at 56.1–59.4% ($\\kappa_T < 0$, vs 73.9% sign persist.);\n"
        "On multi-dim Lorenz-96, Cayley beats GRU for every $d_{\\mathrm{in}} \\geq 16$.",
        fontsize=7.2, color=t["text"],
        bbox=dict(boxstyle="round,pad=0.35", fc=t["card_alt"], ec=t["border"], lw=0.85)
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(d_in)
    ax.set_xticklabels([str(x) for x in d_in])
    ax.set_ylim(50, 96.5)
    ax.set_xlabel("True Chaotic System Dimension $N = d_{\\mathrm{in}}$ (log2 scale)")
    ax.set_ylabel("Causal $K=4$ Sign Forecast Accuracy (%)")
    ax.set_title("C. Axis 4A: Strictly Causal Chaotic Lorenz-96 ($d_{\\mathrm{in}} = 4 \\to 256$)", fontweight="bold", pad=7)
    ax.legend(loc="upper right", fontsize=7.0)

    # --- Panel D: 16-Thread Swarm Pareto (Zero Overlaps!) ---
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, t)

    swarm_pts = [
        # (lbl, ram, thr, col, mk, sz, tx, ty, ha)
        ("Cayley-RTRL (Axis 5, 48D)\n25.3 MB | 2.33 M/s", 25.3, 2.33, t["primary"], "o", 115, 14.5, 1.82, "left"),
        ("Online LRU (Axis 5, 48D)\n39.2 MB | 2.33 M/s", 39.2, 2.33, t["amber"], "s", 110, 62.0, 2.52, "left"),
        ("Deep 2L LRU (NASA 60D)\n99.9 MB | 1.84 M/s", 99.89, 1.84, t["sky"], "s", 95, 155.0, 1.98, "left"),
        ("Selective SSM (Axis 5, 48D)\n61.9 MB | 1.37 M/s", 61.9, 1.37, t["purple"], "D", 105, 105.0, 1.35, "left"),
        ("MicroCore 2L (NASA 60D)\n58.6 MB | 0.91 M/s", 58.56, 0.91, t["secondary"], "o", 105, 14.5, 0.72, "left"),
        ("Selective SSM (NASA 60D)\n162.6 MB | 0.83 M/s", 162.64, 0.83, t["purple"], "d", 95, 250.0, 1.05, "left"),
        ("Supervised GRU (Axis 5, 48D)\n400.9 MB | 0.18 M/s", 400.9, 0.18, t["crimson"], "^", 105, 82.0, 0.25, "left"),
        ("Online GRU BPTT4 (NASA 60D)\n952.4 MB | 0.26 M/s", 952.36, 0.26, t["crimson"], "v", 95, 395.0, 0.63, "left"),
    ]

    for lbl, ram, thr, col, mk, sz, tx, ty, ha in swarm_pts:
        ax.scatter([ram], [thr], s=sz, color=col, marker=mk, edgecolors=t["text"], lw=0.85, zorder=6)
        ax.annotate(lbl, xy=(ram, thr), xytext=(tx, ty), ha=ha, va="center",
                    fontsize=6.8, fontweight="bold" if col in (t["primary"], t["secondary"]) else "normal",
                    color=col, arrowprops=dict(arrowstyle="-", color=col, lw=0.75, alpha=0.75))

    ax.axvline(32.0, color=t["amber"], ls="--", lw=1.2, alpha=0.8)
    ax.text(34.5, 1.25, "32 MB L3 Cache", rotation=90, ha="left", va="center",
            fontsize=7.3, fontweight="bold", color=t["amber"])
    ax.text(14.5, 2.74, "★ Pareto Ideal (Low RAM, High M/s)", fontsize=7.6, fontweight="bold",
            color=t["secondary"], ha="left", va="top")

    ax.set_xscale("log")
    ax.set_xlim(12.5, 1550)
    ax.set_ylim(0.0, 2.86)
    ax.set_xlabel("Total 4,096-Agent Swarm RAM (MB, log scale)")
    ax.set_ylabel("16-Thread Throughput (Million steps/s)")
    ax.set_title("D. 16-Thread Swarm Pareto ($N=4,096$ Personal Twins, 0-Alloc)", fontweight="bold", pad=7)

    save_figure(fig, "fig07_streams_rl_chaos_and_swarm", t["name"])


# ==============================================================================
# FIGURE 8: Prime-Field Modular Arithmetic (Step 9) & Global Forensic Audit
# ==============================================================================

def make_fig08(t: dict):
    fig = plt.figure(figsize=(15.6, 9.8), facecolor=t["bg"])
    add_header(
        fig,
        "Prime-Field Modular Arithmetic (Step 9, $\\mathbb{F}_{997}$ & $\\mathbb{F}_{9973}$) & Global Anti-Leakage Forensic Audit",
        "Post-fix disjoint holdout results (0% Fourier/mod-p oracles), honest cryptographic folding wall, and live Label-Shuffle verification",
        t, y_title=0.976, y_sub=0.943
    )

    gs = gridspec.GridSpec(2, 2, left=0.088, right=0.975, top=0.875, bottom=0.072, wspace=0.25, hspace=0.38)

    # --- Panel A ---
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, t, grid_axis="y")
    s9_models = [
        "SO(3)\nv0.4.1\n(10.7 KB)",
        "PURE 2L\nv0.4.1\n(7.8 KB)",
        "O(4) M=1\nv0.5.0\n(12.1 KB)",
        "MLP H=32\nAdam\n(4.9 KB)",
        "AlgSel2D\nv0.4.0\n(6.6 KB)",
        "Unified\nv0.5.0\n(26.8 KB)",
        "Fixed Q24\nv0.5.0\n(2.3 KB)",
        "MicroCore\nv0.4.0\n(11.4 KB)",
    ]
    add_997 = [74.39, 56.79, 56.03, 54.23, 52.33, 50.23, 50.19, 49.98]
    add_9973 = [71.35, 59.98, 53.12, 54.65, 55.57, 53.53, 53.20, 53.87]

    x9 = np.arange(len(s9_models))
    w9 = 0.35
    ax.bar(x9 - w9/2, add_997, w9, color=t["primary"], label=r"$\mathbb{F}_{997}$ (Add (a+b) mod 997, 10-bit)")
    ax.bar(x9 + w9/2, add_9973, w9, color=t["secondary"], label=r"$\mathbb{F}_{9973}$ (Add (a+b) mod 9973, 14-bit)")
    ax.axhline(69.3, color=t["amber"], ls="--", lw=1.3, label=r"Zero-Carry Floor ($\mathbb{F}_{997}$: 69.3%, $\mathbb{F}_{9973}$: 65.5%)")
    ax.axhline(53.87, color=t["crimson"], ls=":", lw=1.2, label=r"Bit-Majority Floor ($\mathbb{F}_{9973}$: 53.87%; $\mathbb{F}_{997}$: 50.75%)")

    ax.text(-0.22, 75.5, "74.4%", ha="center", fontsize=7.5, fontweight="bold", color=t["primary"])
    ax.text(0.32, 72.4, "71.4%", ha="center", fontsize=7.5, fontweight="bold", color=t["secondary"])
    ax.annotate("SO(3) Shuf Ctrl:\n49.1% / 53.2%",
                xy=(0.18, 53.2), xytext=(2.1, 72.5),
                fontsize=7.3, fontweight="bold", color=t["text"],
                bbox=dict(boxstyle="round,pad=0.28", fc=t["card_alt"], ec=t["border"], lw=0.8),
                arrowprops=dict(arrowstyle="->", color=t["muted"], lw=1.0))

    ax.set_xticks(x9)
    ax.set_xticklabels(s9_models, fontsize=7.1)
    ax.set_ylim(44, 83.5)
    ax.set_ylabel("Disjoint Holdout All-Bit Accuracy (%)")
    ax.set_title(r"A. Bit-Serial Modular Add (a+b) mod p (Includes $(a/p, b/p)$ Features)", fontweight="bold", pad=8)
    ax.legend(loc="upper right", fontsize=6.7)

    # --- Panel B: Cryptographic Folding Wall Heatmap (Clean Title Spacing!) ---
    ax = fig.add_subplot(gs[0, 1])
    ops = ["Add\n(a+b)", "Mul\n(a·b)", "PowG\n(g^k)", "PowAB\n(a^b)", "DLog\n(log_g x)"]
    hm_models = [
        "SO(3) v0.4.1",
        "PURE 2L v0.4.1",
        "O(4) M=1 v0.5.0",
        "MLP H=32 Adam",
        "AlgSel2D v0.4.0",
        "Unified v0.5.0",
        "Fixed Q24 v0.5.0",
        "MicroCore v0.4.0",
    ]
    hm_data = np.array([
        [74.39, 50.40, 50.05, 50.24, 50.28],
        [56.79, 50.46, 50.18, 50.19, 49.00],
        [56.03, 50.09, 50.47, 50.45, 50.65],
        [54.23, 50.00, 51.28, 51.45, 50.96],
        [52.33, 50.56, 49.19, 50.40, 51.09],
        [50.23, 50.23, 49.47, 50.63, 51.80],
        [50.19, 50.55, 50.12, 50.73, 51.43],
        [49.98, 50.27, 48.97, 50.03, 51.99],
    ])
    ax.imshow(hm_data, cmap=t["heat_cmap"], aspect="auto", vmin=48.5, vmax=75.5)
    ax.set_xticks(np.arange(len(ops)))
    ax.set_xticklabels(ops, fontsize=8.1)
    ax.set_yticks(np.arange(len(hm_models)))
    ax.set_yticklabels(hm_models, fontsize=7.8)
    for r in range(hm_data.shape[0]):
        for c in range(hm_data.shape[1]):
            v = hm_data[r, c]
            tc = ("#FFFFFF" if v > 62.0 else "#0F172A") if t["name"] == "paper" else ("#0B1220" if v > 64.0 else "#E6EDF7")
            ax.text(c, r, f"{v:.1f}%", ha="center", va="center", fontsize=7.7,
                    fontweight="bold" if v > 56.0 else "normal", color=tc)
    wall_rect = Rectangle((0.5, -0.5), 4.0, 8.0, fill=False, edgecolor=t["crimson"], lw=2.0, ls="--")
    ax.add_patch(wall_rect)
    ax.text(2.5, -0.62, "Honest Cryptographic / Modular-Folding Wall (~50% chance, 0 leaks)",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=t["crimson"])
    for sp in ax.spines.values():
        sp.set_color(t["border"])
    ax.set_title(r"B. $\mathbb{F}_{997}$ Operation Matrix: Add Carry vs Cryptographic Wall", fontweight="bold", pad=22)

    # --- Panel C: Order-12 Cyclic Subgroup (Zero Legend Overlap!) ---
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, t, grid_axis="y")
    sub_models = [
        "v0.4.0\nAlgSel 2D\n(7.1 KB)",
        "v0.5.0\nSO(3)+P2\n(6.5 KB)",
        "v0.5.0\nO(4) M=2\n(11.1 KB)",
        "v0.5.0\nO(4) Q1.14\n(9.9 KB)",
        "Control\nLabel-Shuf\n(chance)",
    ]
    f997_l6 = [69.6, 96.7, 100.0, 100.0, 12.1]
    f997_l30 = [9.6, 39.6, 68.3, 68.3, 7.5]
    f9973_l6 = [57.1, 97.5, 99.6, 99.6, 10.8]
    f9973_l30 = [7.5, 45.4, 90.0, 90.0, 13.3]

    x3 = np.arange(len(sub_models))
    w3 = 0.19
    ax.bar(x3 - 1.5*w3, f997_l6, w3, color=t["sky"], label=r"$\langle 906\rangle \leq \mathbb{F}_{997}^\times$ ($L=6$)")
    ax.bar(x3 - 0.5*w3, f997_l30, w3, color=t["primary"], label=r"$\langle 906\rangle \leq \mathbb{F}_{997}^\times$ ($L=30$)")
    ax.bar(x3 + 0.5*w3, f9973_l6, w3, color=t["secondary"], alpha=0.65, label=r"$\langle 1942\rangle \leq \mathbb{F}_{9973}^\times$ ($L=6$)")
    ax.bar(x3 + 1.5*w3, f9973_l30, w3, color=t["secondary"], label=r"$\langle 1942\rangle \leq \mathbb{F}_{9973}^\times$ ($L=30$)")
    ax.axhline(8.33, color=t["slate"], ls=":", lw=1.2, label="Random Chance (1/12 = 8.3%)")

    ax.annotate("100% Exact Q1.14 (i16)\nDeployment Parity!", xy=(3.0, 101.0), xytext=(0.45, 114.5),
                fontsize=7.4, fontweight="bold", color=t["secondary"], ha="center",
                bbox=dict(boxstyle="round,pad=0.25", fc=t["secondary_soft"], ec=t["secondary"], lw=0.7),
                arrowprops=dict(arrowstyle="->", color=t["secondary"], lw=1.2))
    for i in [0, 1, 2, 3]:
        ax.text(x3[i] - 0.5*w3, f997_l30[i] + 2.0, f"{f997_l30[i]:.0f}%", ha="center", fontsize=6.9, fontweight="bold", color=t["primary"])
        ax.text(x3[i] + 1.5*w3, f9973_l30[i] + 2.0, f"{f9973_l30[i]:.0f}%", ha="center", fontsize=6.9, fontweight="bold", color=t["secondary"])

    ax.set_xticks(x3)
    ax.set_xticklabels(sub_models, fontsize=7.6)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Subgroup State Accuracy (%)")
    ax.set_title(r"C. Order-12 Subgroup ($\cong C_{12}$, Post-Fix Per-Prime Seeds, Train $L\leq 8$)", fontweight="bold", pad=8)
    ax.legend(loc="upper right", fontsize=6.7)

    # --- Panel D: Global Forensic Integrity Audit (Legend Above, Zero Collisions!) ---
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, t, grid_axis="x")
    audit_rows = [
        ("ELEC2: AlgSel (+y)", 92.10, 53.05, 57.55),
        ("ELEC2: MicroCore (raw)", 88.24, 52.29, 57.55),
        ("ELEC2: Unified One-Cfg", 84.96, 55.61, 57.55),
        ("Dyck-2: Unified One-Cfg", 86.90, 33.20, 33.33),
        ("Lorenz-96 (N=16): Cayley", 71.73, 50.33, 50.00),
        ("Lorenz-96 (N=256): Cayley", 62.47, 48.07, 50.00),
        ("Lorenz-63 (60D): MicroCore", 57.72, 48.67, 50.00),
        ("A_5 (L=6): Proto SO(3)", 97.50, 2.50, 1.67),
        ("S_5 (L=6): O(4) M=3", 99.50, 0.90, 0.83),
        ("F_997 Add: SO(3) v0.4.1", 74.39, 49.10, 50.75),
    ]
    y_idx = np.arange(len(audit_rows))[::-1]
    for yi, (lbl, orig, shuf, flr) in zip(y_idx, audit_rows):
        ax.plot([shuf, orig], [yi, yi], color=t["border"], lw=2.4, zorder=2)
        ax.scatter([orig], [yi], color=t["primary"], s=62, zorder=4,
                   label="Original Stream" if yi == len(audit_rows) - 1 else None)
        ax.scatter([shuf], [yi], color=t["crimson"], marker="X", s=65, zorder=4,
                   label="Global Label-Shuffle" if yi == len(audit_rows) - 1 else None)
        ax.scatter([flr], [yi], color=t["slate"], marker="|", s=95, linewidths=1.8, zorder=3,
                   label="Chance / Majority Floor" if yi == len(audit_rows) - 1 else None)
        ax.text(orig + 2.0, yi, f"{orig:.1f}%", va="center", ha="left", fontsize=7.2, fontweight="bold", color=t["primary"])
        if shuf < 15.0:
            ax.text(shuf + 3.2, yi, f"{shuf:.1f}%", va="center", ha="left", fontsize=7.0, fontweight="bold", color=t["crimson"])
        else:
            ax.text(shuf - 2.2, yi, f"{shuf:.1f}%", va="center", ha="right", fontsize=7.0, color=t["crimson"])

    ax.set_yticks(y_idx)
    ax.set_yticklabels([r[0] for r in audit_rows], fontsize=7.6)
    ax.set_xlim(-3, 115)
    ax.set_xlabel("Accuracy (%) — Live bench_forensic_audit & Suite A/B/Step 9 Verification")
    ax.set_title("D. Anti-Leakage Audit: Original vs Shuffled-$y$ (6000/6000 Causal)", fontweight="bold", pad=22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, fontsize=6.9)

    save_figure(fig, "fig08_prime_field_and_forensic_audit", t["name"])


# ==============================================================================
# Main Runner
# ==============================================================================

BUILDERS = [
    ("fig01_architecture_overview", make_fig01),
    ("fig02_o4_parity_and_adaptive_geometry", make_fig02),
    ("fig03_zero_transcendental_primitives", make_fig03),
    ("fig04_nonabelian_groups_and_ladder", make_fig04),
    ("fig05_unified_one_config_and_ablations", make_fig05),
    ("fig06_scaling_memory_and_seu", make_fig06),
    ("fig07_streams_rl_chaos_and_swarm", make_fig07),
    ("fig08_prime_field_and_forensic_audit", make_fig08),
]


def main():
    for theme_name, theme in THEMES.items():
        apply_theme(theme)
        for stem, fn in BUILDERS:
            fn(theme)
            print(f"[OK] {theme_name:5s} -> {stem}")
    print(f"\nGenerated {len(BUILDERS)} figures in:")
    print(f"  - Print/LaTeX (PNG 300 DPI + PDF + SVG): {OUT_PAPER}")
    print(f"  - Dark GitHub/Slides (PNG 300 DPI):      {OUT_DARK}")


if __name__ == "__main__":
    main()
