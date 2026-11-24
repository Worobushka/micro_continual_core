#!/usr/bin/env python3
# Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov & Sofia Sidorenko
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
External River Verification Benchmark on Electricity (ELEC2) Stream
==================================================================
Strictly causal 1-pass prequential (test-then-train) evaluation matching `bench_forensic_audit.rs`:
  - ARFClassifier (Adaptive Random Forest, 10 trees)
  - HoeffdingAdaptiveTreeClassifier
  - StandardScaler | LogisticRegression (+ ADWIN reset control)
  - Persistent / No-Change control baseline (y_hat_t = y_{t-1})

Usage:
  python scripts/bench_elec2_river.py [--data path/to/electricity-normalized.arff] [--max-rows 45312]
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

try:
    from river import drift, forest, linear_model, preprocessing, tree
    RIVER_IMPORT_ERR = None
except ImportError as exc:
    RIVER_IMPORT_ERR = exc


DEFAULT_CANDIDATES = [
    "electricity-normalized.arff",
    "../micro_continual_core/electricity-normalized.arff",
    "electricity.csv",
    "elecNormNew.csv",
    "data/electricity-normalized.arff",
]


def find_dataset(cli_path: str):
    if cli_path:
        p = Path(cli_path)
        return p if p.is_file() else None
    script_dir = Path(__file__).resolve().parent.parent
    for rel in DEFAULT_CANDIDATES:
        for base in (Path.cwd(), script_dir):
            cand = (base / rel).resolve()
            if cand.is_file():
                return cand
    return None


def load_elec2(path: Path, max_rows: int = 0):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    is_arff = path.suffix.lower() == ".arff" or "@relation" in text
    in_data = not is_arff
    for li, line in enumerate(text.splitlines()):
        tr = line.strip()
        if not tr or tr.startswith("%"):
            continue
        if is_arff and not in_data:
            if tr.lower() == "@data":
                in_data = True
            continue
        if (not is_arff) and li == 0 and any(ch.isalpha() for ch in tr):
            continue
        cols = tr.split(",")
        if len(cols) < 4:
            continue
        feats = {}
        ok = True
        for i, c in enumerate(cols[:-1]):
            clean = c.strip().strip("'\"")
            try:
                feats[f"f{i}"] = float(clean)
            except ValueError:
                ok = False
                break
        if not ok or not feats:
            continue
        lbl = cols[-1].strip().strip("'\"").lower()
        y = 1 if lbl in ("up", "1", "true") else 0
        rows.append((feats, y))
        if max_rows > 0 and len(rows) >= max_rows:
            break
    return rows


def eval_river_models(rows):
    n = len(rows)
    # 1. Persistent (No-Change) sanity check
    p_hits = 0
    prev_y = 1
    for _, y in rows:
        if prev_y == y:
            p_hits += 1
        prev_y = y
    p_acc = p_hits / max(1, n)

    models = [
        ("River ARFClassifier (10 trees)", lambda: forest.ARFClassifier(n_models=10, grace_period=50, seed=42)),
        ("River HoeffdingAdaptiveTree", lambda: tree.HoeffdingAdaptiveTreeClassifier(grace_period=50, seed=42)),
        (
            "River LogReg + StandardScaler",
            lambda: preprocessing.StandardScaler() | linear_model.LogisticRegression(),
        ),
    ]

    print(f"ELEC2 rows={n}, Persistent(y_{{t-1}})={p_acc*100:.2f}%")
    print(f"{'Model':<34} | {'Acc %':<8} | {'KappaT':<8} | {'Time s':<8}")
    print("-" * 70)

    for name, ctor in models:
        clf = ctor()
        hits = 0
        t0 = time.perf_counter()
        for x, y in rows:
            pred = clf.predict_one(x)
            pred_i = 1 if pred in (1, True, "1", "up") else 0
            if pred_i == y:
                hits += 1
            clf.learn_one(x, y)
        dt = time.perf_counter() - t0
        acc = hits / max(1, n)
        kappa_t = (acc - p_acc) / max(1e-6, 1.0 - p_acc)
        print(f"{name:<34} | {acc*100:6.2f}% | {kappa_t:+7.3f} | {dt:7.2f}s")

    # Adaptive LogReg + ADWIN reset control
    adwin = drift.ADWIN()
    clf = preprocessing.StandardScaler() | linear_model.LogisticRegression()
    hits = 0
    t0 = time.perf_counter()
    for x, y in rows:
        pred = clf.predict_one(x)
        pred_i = 1 if pred in (1, True, "1", "up") else 0
        correct = 1 if pred_i == y else 0
        hits += correct
        adwin.update(1 - correct)
        if adwin.drift_detected:
            clf = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        clf.learn_one(x, y)
    dt = time.perf_counter() - t0
    acc = hits / max(1, n)
    kappa_t = (acc - p_acc) / max(1e-6, 1.0 - p_acc)
    print(f"{'River LogReg + ADWIN Reset':<34} | {acc*100:6.2f}% | {kappa_t:+7.3f} | {dt:7.2f}s")


def main():
    parser = argparse.ArgumentParser(description="External River prequential benchmark on ELEC2")
    parser.add_argument("--data", type=str, default="", help="Path to electricity-normalized.arff or CSV")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row limit (0 = all rows)")
    args = parser.parse_args()

    if RIVER_IMPORT_ERR is not None:
        print(
            f"[ERROR] The 'river' package is not installed ({RIVER_IMPORT_ERR}). "
            f"Install with `pip install river` to run external ELEC2 verification.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    data_path = find_dataset(args.data)
    if data_path is None:
        print(
            "[ERROR] ELEC2 dataset file not found. Pass `--data path/to/electricity-normalized.arff`.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    rows = load_elec2(data_path, args.max_rows)
    if not rows:
        print(f"[ERROR] Failed to parse any rows from {data_path}", file=sys.stderr)
        raise SystemExit(1)

    eval_river_models(rows)


if __name__ == "__main__":
    main()
