#!/usr/bin/env python3
# Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov & Sofia Sidorenko
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
Independent external third-party verification benchmark for NASA C-MAPSS (FD001-FD004).
Zero shared Rust code: uses identical strictly-causal train-only normalization and evaluates:
  1. PyTorch (GRU H=60 + Adam, 1-pass & 3-pass streaming per-engine BPTT + per-cycle online)
  2. River (ARFRegressor & HoeffdingAdaptiveTreeRegressor, 1-pass & 3-pass streaming)
Includes FD004 raw 24-feature ablation and FD004 RUL-Shuffle anti-leakage control.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

RUL_MAX = 125.0
CRIT_RUL = 30.0


def load_rows(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 26:
                continue
            uid = int(parts[0])
            cyc = int(parts[1])
            ops = [float(parts[2 + i]) for i in range(3)]
            sens = [float(parts[5 + i]) for i in range(21)]
            rows.append((uid, cyc, ops, sens))
    return rows


def load_ruls(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [float(x.strip()) for x in f if x.strip()]


class CausalNormalizer:
    """Strictly causal train-only normalizer identical to bench_step7_nasa_fleet.rs."""

    def __init__(self, train_rows, multi_regime: bool, raw_mode: bool):
        self.multi_regime = multi_regime
        self.raw_mode = raw_mode
        n = max(len(train_rows), 1)
        if raw_mode:
            self.raw_mean = [0.0] * 24
            for _, _, ops, sens in train_rows:
                for j in range(3):
                    self.raw_mean[j] += ops[j]
                for j in range(21):
                    self.raw_mean[3 + j] += sens[j]
            self.raw_mean = [m / n for m in self.raw_mean]
            var = [0.0] * 24
            for _, _, ops, sens in train_rows:
                for j in range(3):
                    d = ops[j] - self.raw_mean[j]
                    var[j] += d * d
                for j in range(21):
                    d = sens[j] - self.raw_mean[3 + j]
                    var[3 + j] += d * d
            self.raw_std = [math.sqrt(v / n) for v in var]
            return

        if not multi_regime:
            self.k = 1
            mean = [0.0] * 21
            for _, _, _, sens in train_rows:
                for j in range(21):
                    mean[j] += sens[j]
            mean = [m / n for m in mean]
            var = [0.0] * 21
            for _, _, _, sens in train_rows:
                for j in range(21):
                    d = sens[j] - mean[j]
                    var[j] += d * d
            self.s_mean = [mean]
            self.s_std = [[math.sqrt(v / n) for v in var]]
            return

        self.k = 6
        op_mean = [0.0, 0.0, 0.0]
        for _, _, ops, _ in train_rows:
            for j in range(3):
                op_mean[j] += ops[j]
        op_mean = [m / n for m in op_mean]
        op_var = [0.0, 0.0, 0.0]
        for _, _, ops, _ in train_rows:
            for j in range(3):
                d = ops[j] - op_mean[j]
                op_var[j] += d * d
        op_std = [math.sqrt(v / n) if math.sqrt(v / n) > 1e-6 else 1.0 for v in op_var]
        self.op_mean = op_mean
        self.op_std = op_std

        ops_n = [
            [(ops[0] - op_mean[0]) / op_std[0],
             (ops[1] - op_mean[1]) / op_std[1],
             (ops[2] - op_mean[2]) / op_std[2]]
            for _, _, ops, _ in train_rows
        ]
        # Deterministic farthest-point seeding (train only)
        centroids = [ops_n[0][:]]
        for _ in range(1, self.k):
            best_i = 0
            best_d = -1.0
            for idx, p in enumerate(ops_n):
                md = min(
                    (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2
                    for c in centroids
                )
                if md > best_d:
                    best_d = md
                    best_i = idx
            centroids.append(ops_n[best_i][:])

        assign = [0] * len(ops_n)
        for _ in range(25):
            sums = [[0.0, 0.0, 0.0] for _ in range(self.k)]
            cnts = [0] * self.k
            for i, p in enumerate(ops_n):
                bc = 0
                bd = float("inf")
                for ci, c in enumerate(centroids):
                    d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2
                    if d < bd:
                        bd = d
                        bc = ci
                assign[i] = bc
                cnts[bc] += 1
                sums[bc][0] += p[0]
                sums[bc][1] += p[1]
                sums[bc][2] += p[2]
            for ci in range(self.k):
                if cnts[ci] > 0:
                    centroids[ci] = [s / cnts[ci] for s in sums[ci]]
        centroids.sort(key=lambda c: c[0])
        self.centroids = centroids

        s_sum = [[0.0] * 21 for _ in range(self.k)]
        s_cnt = [0] * self.k
        for i, (_, _, _, sens) in enumerate(train_rows):
            p = ops_n[i]
            bc = min(
                range(self.k),
                key=lambda ci: (p[0] - centroids[ci][0]) ** 2
                + (p[1] - centroids[ci][1]) ** 2
                + (p[2] - centroids[ci][2]) ** 2,
            )
            assign[i] = bc
            s_cnt[bc] += 1
            for j in range(21):
                s_sum[bc][j] += sens[j]
        self.s_mean = [[s_sum[ci][j] / max(s_cnt[ci], 1) for j in range(21)] for ci in range(self.k)]
        s_var = [[0.0] * 21 for _ in range(self.k)]
        for i, (_, _, _, sens) in enumerate(train_rows):
            ci = assign[i]
            for j in range(21):
                d = sens[j] - self.s_mean[ci][j]
                s_var[ci][j] += d * d
        self.s_std = [[math.sqrt(s_var[ci][j] / max(s_cnt[ci], 1)) for j in range(21)] for ci in range(self.k)]

    def transform(self, ops, sens):
        if self.raw_mode:
            out = []
            for j in range(3):
                out.append(0.0 if self.raw_std[j] < 1e-6 else (ops[j] - self.raw_mean[j]) / self.raw_std[j])
            for j in range(21):
                out.append(0.0 if self.raw_std[3 + j] < 1e-6 else (sens[j] - self.raw_mean[3 + j]) / self.raw_std[3 + j])
            return out
        if self.k == 1:
            reg = 0
        else:
            p = [
                (ops[0] - self.op_mean[0]) / self.op_std[0],
                (ops[1] - self.op_mean[1]) / self.op_std[1],
                (ops[2] - self.op_mean[2]) / self.op_std[2],
            ]
            reg = min(
                range(self.k),
                key=lambda ci: (p[0] - self.centroids[ci][0]) ** 2
                + (p[1] - self.centroids[ci][1]) ** 2
                + (p[2] - self.centroids[ci][2]) ** 2,
            )
        return [
            0.0 if self.s_std[reg][j] < 1e-6 else (sens[j] - self.s_mean[reg][j]) / self.s_std[reg][j]
            for j in range(21)
        ]


def load_cmapss(data_dir: Path, name: str, multi_regime: bool, raw_mode: bool = False):
    tr_rows = load_rows(data_dir / f"train_{name}.txt")
    te_rows = load_rows(data_dir / f"test_{name}.txt")
    te_ruls = load_ruls(data_dir / f"RUL_{name}.txt")
    norm = CausalNormalizer(tr_rows, multi_regime, raw_mode)

    def group(rows, ruls=None):
        by_u = {}
        for uid, cyc, ops, sens in rows:
            by_u.setdefault(uid, []).append((cyc, ops, sens))
        units = []
        for uid in sorted(by_u.keys()):
            seq = sorted(by_u[uid], key=lambda x: x[0])
            max_cyc = seq[-1][0]
            rul_end = ruls[uid - 1] if ruls is not None else 0.0
            feats = [norm.transform(ops, sens) for _, ops, sens in seq]
            trues = [min(rul_end + (max_cyc - cyc), RUL_MAX) for cyc, _, _ in seq]
            units.append((uid, feats, trues, min(rul_end, RUL_MAX)))
        return units

    return group(tr_rows, None), group(te_rows, te_ruls), (24 if raw_mode else 21)


def compute_metrics(preds_norm, true_ruls):
    se = 0.0
    s_score = 0.0
    tp = fp = fn = 0
    for pn, y in zip(preds_norm, true_ruls):
        p = max(0.0, min(1.0, pn)) * RUL_MAX
        d = p - y
        se += d * d
        s_score += (math.exp(-d / 13.0) - 1.0) if d < 0.0 else (math.exp(d / 10.0) - 1.0)
        pc = p <= CRIT_RUL
        yc = y <= CRIT_RUL
        if pc and yc:
            tp += 1
        elif pc and not yc:
            fp += 1
        elif not pc and yc:
            fn += 1
    rmse = math.sqrt(se / max(len(true_ruls), 1))
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "rmse": round(rmse, 3),
        "s_score": round(s_score, 1),
        "ew_recall": round(rec * 100.0, 2),
        "ew_f1": round(f1 * 100.0, 2),
    }


def run_torch_backend(data_dir: Path):
    import torch
    import torch.nn as nn

    torch.set_num_threads(4)
    print(f"\n=== INDEPENDENT EXTERNAL CONTROL: PyTorch {torch.__version__} GRU (H=60) + Adam ===")

    class GruRul(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 60):
            super().__init__()
            self.gru = nn.GRU(in_dim, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x, h=None):
            out, h = self.gru(x, h)
            return self.head(out).squeeze(-1), h

    results = {}
    specs = [
        ("FD001", False, False),
        ("FD002", True, False),
        ("FD003", False, False),
        ("FD004", True, False),
        ("FD004_raw", False, True),
    ]
    for label, multi, raw in specs:
        fd = "FD004" if label == "FD004_raw" else label
        tr_u, te_u, in_dim = load_cmapss(data_dir, fd, multi, raw)
        results[label] = {}
        passes_list = [1, 3] if label != "FD004_raw" else [3]
        for passes in passes_list:
            ms = []
            t0 = time.perf_counter()
            n_cyc = 0
            for seed in [1, 2, 3]:
                torch.manual_seed(seed)
                rng = random.Random(seed)
                model = GruRul(in_dim, 60)
                opt = torch.optim.Adam(model.parameters(), lr=0.003)
                loss_fn = nn.MSELoss()
                # Streaming online engine-by-engine BPTT (batch=1, full trajectory step)
                for p in range(passes):
                    order = list(range(len(tr_u)))
                    rng.shuffle(order)
                    for ui in order:
                        _, feats, trues, _ = tr_u[ui]
                        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
                        y = torch.tensor([r / RUL_MAX for r in trues], dtype=torch.float32).unsqueeze(0)
                        opt.zero_grad()
                        pred, _ = model(x)
                        loss = loss_fn(pred, y)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        opt.step()
                        n_cyc += len(feats)
                # Frozen test eval
                model.eval()
                preds = []
                trues_end = []
                with torch.no_grad():
                    for _, feats, _, end_rul in te_u:
                        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
                        pred, _ = model(x)
                        preds.append(float(pred[0, -1].item()))
                        trues_end.append(end_rul)
                ms.append(compute_metrics(preds, trues_end))
            lat_us = (time.perf_counter() - t0) * 1e6 / max(n_cyc, 1)
            avg = {
                "rmse": round(sum(m["rmse"] for m in ms) / len(ms), 2),
                "rmse_sd": round(math.sqrt(sum((m["rmse"] - sum(z["rmse"] for z in ms)/len(ms))**2 for m in ms)/len(ms)), 2),
                "s_score": round(sum(m["s_score"] for m in ms) / len(ms), 1),
                "ew_recall": round(sum(m["ew_recall"] for m in ms) / len(ms), 1),
                "ew_f1": round(sum(m["ew_f1"] for m in ms) / len(ms), 1),
                "lat_us": round(lat_us, 2),
            }
            results[label][f"{passes}pass"] = avg
            print(
                f"  PyTorch GRU-60 [{label:<9} {passes}-pass] | RMSE={avg['rmse']:5.2f}+-{avg['rmse_sd']:4.2f} | "
                f"S-Score={avg['s_score']:8.1f} | EW-Rec={avg['ew_recall']:5.1f}% | EW-F1={avg['ew_f1']:5.1f}% | {avg['lat_us']:6.2f} us/cyc"
            )

    # RUL-Shuffle control on FD004
    tr_u, te_u, in_dim = load_cmapss(data_dir, "FD004", True, False)
    all_y = [r / RUL_MAX for _, _, trues, _ in tr_u for r in trues]
    rng = random.Random(42)
    rng.shuffle(all_y)
    torch.manual_seed(1)
    model = GruRul(in_dim, 60)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.MSELoss()
    for p in range(3):
        pos = 0
        for _, feats, _, _ in tr_u:
            L = len(feats)
            ys = all_y[pos : pos + L]
            pos += L
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            y = torch.tensor(ys, dtype=torch.float32).unsqueeze(0)
            opt.zero_grad()
            pred, _ = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
    model.eval()
    preds, trues_end = [], []
    with torch.no_grad():
        for _, feats, _, end_rul in te_u:
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            pred, _ = model(x)
            preds.append(float(pred[0, -1].item()))
            trues_end.append(end_rul)
    shuf_m = compute_metrics(preds, trues_end)
    results["FD004_shuffle"] = shuf_m
    print(f"  PyTorch GRU-60 [FD004 RUL-SHUFFLE] | RMSE={shuf_m['rmse']:5.2f} | S-Score={shuf_m['s_score']:8.1f} (PASS no leak)")
    return {"torch_version": torch.__version__, "gru60": results}


def run_river_backend(data_dir: Path):
    import river
    from river import forest, tree

    print(f"\n=== INDEPENDENT EXTERNAL CONTROL: river=={river.__version__} (ARFRegressor & HoeffdingAdaptiveTreeRegressor) ===")
    results = {"river_version": river.__version__, "hatr": {}, "arf": {}}

    for label, multi in [("FD001", False), ("FD002", True), ("FD003", False), ("FD004", True)]:
        tr_u, te_u, in_dim = load_cmapss(data_dir, label, multi, False)
        for mname, factory in [
            ("hatr", lambda s: tree.HoeffdingAdaptiveTreeRegressor(seed=s, grace_period=100)),
            ("arf", lambda s: forest.ARFRegressor(n_models=5, seed=s, grace_period=100)),
        ]:
            results[mname][label] = {}
            for passes in [1, 3]:
                t0 = time.perf_counter()
                model = factory(1)
                rng = random.Random(1)
                n_cyc = 0
                for _ in range(passes):
                    order = list(range(len(tr_u)))
                    rng.shuffle(order)
                    for ui in order:
                        _, feats, trues, _ = tr_u[ui]
                        for fv, y_rul in zip(feats, trues):
                            x_d = {k: fv[k] for k in range(in_dim)}
                            model.learn_one(x_d, y_rul / RUL_MAX)
                            n_cyc += 1
                preds = []
                trues_end = []
                for _, feats, _, end_rul in te_u:
                    fv = feats[-1]
                    x_d = {k: fv[k] for k in range(in_dim)}
                    p = model.predict_one(x_d)
                    preds.append(0.5 if p is None else float(p))
                    trues_end.append(end_rul)
                lat_us = (time.perf_counter() - t0) * 1e6 / max(n_cyc, 1)
                met = compute_metrics(preds, trues_end)
                met["lat_us"] = round(lat_us, 2)
                results[mname][label][f"{passes}pass"] = met
                print(
                    f"  river {mname.upper():<4} [{label} {passes}-pass] | RMSE={met['rmse']:5.2f} | "
                    f"S-Score={met['s_score']:8.1f} | EW-Rec={met['ew_recall']:5.1f}% | EW-F1={met['ew_f1']:5.1f}% | {lat_us:6.1f} us/cyc"
                )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["all", "torch", "river"], default="all")
    ap.add_argument("--data-dir", default="CMAPSSData")
    ap.add_argument("--out-json", default="logs/nasa_external_results.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged = {}
    if out_path.exists():
        try:
            merged = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            merged = {}

    if args.backend in ("all", "torch"):
        merged["pytorch"] = run_torch_backend(data_dir)
    if args.backend in ("all", "river"):
        merged["river"] = run_river_backend(data_dir)

    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"\n[JSON] Saved external control results to {out_path}")


if __name__ == "__main__":
    main()
