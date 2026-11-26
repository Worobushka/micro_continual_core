#!/usr/bin/env python3
# Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov & Sofia Sidorenko
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Code-invariant auditor for MicroContinualCore v0.5.0 core modules."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

CORE_FILES = (
    "src/algebra.rs",
    "src/unified.rs",
    "src/fixed_point.rs",
    "src/adaptive_od.rs",
    "src/core.rs",
    "src/rtrl.rs",
    "src/math.rs",
    "src/belief.rs",
    "src/memory.rs",
    "src/env_hard_pomdp.rs",
)

FORBIDDEN = (
    ".exp(",
    ".exp2(",
    ".ln(",
    ".log(",
    ".log2(",
    ".log10(",
    ".sin(",
    ".cos(",
    ".tan(",
    ".sinh(",
    ".cosh(",
    ".tanh(",
    ".asin(",
    ".acos(",
    ".atan(",
    ".powf(",
    ".sqrt(",
    "f32::sqrt",
    "f64::sqrt",
    "panic!",
    ".unwrap(",
    ".expect(",
)


def strip_comments_and_tests(text: str) -> str:
    idx = text.find("#[cfg(test)]")
    if idx != -1:
        text = text[:idx]
    out_lines = []
    for line in text.splitlines():
        code = line.split("//", 1)[0]
        out_lines.append(code)
    return "\n".join(out_lines)


def scan_text(label: str, code: str) -> list[str]:
    errs = []
    for tok in FORBIDDEN:
        if tok in code:
            errs.append(f"{label}: forbidden token `{tok}` found")
    return errs


def main() -> int:
    errors: list[str] = []

    for rel in CORE_FILES:
        p = ROOT / rel
        if not p.exists():
            errors.append(f"Missing required file: {p}")
            continue
        code = strip_comments_and_tests(p.read_text(encoding="utf-8"))
        e = scan_text(rel, code)
        errors.extend(e)
        print(f"[PASS] {rel}: 0 forbidden math/panic calls" if not e else f"[FAIL] {rel}: {e}")

    if errors:
        print("\nAUDIT FAILED:")
        for e in errors:
            print("  -", e)
        return 1
    print(f"\nALL {len(CORE_FILES)} CORE MODULES VERIFIED CLEAN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
