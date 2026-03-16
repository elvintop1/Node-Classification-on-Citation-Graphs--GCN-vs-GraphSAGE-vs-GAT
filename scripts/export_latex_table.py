#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export aggregate benchmark results to a LaTeX table.")
    parser.add_argument("--csv", type=str, required=True, help="Path to aggregate_results.csv")
    parser.add_argument("--out", type=str, required=True, help="Output .tex path")
    return parser.parse_args()


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    lines = []
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Model & Test Accuracy & Test Macro-F1 \\")
    lines.append(r"\midrule")
    for _, row in df.sort_values(["dataset", "model"]).iterrows():
        line = (
            f"{row['dataset']} & {row['model']} & "
            f"{fmt(row['test_accuracy_mean'], row['test_accuracy_std'])} & "
            f"{fmt(row['test_macro_f1_mean'], row['test_macro_f1_std'])} \\")
        lines.append(line)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved LaTeX table to {args.out}")


if __name__ == "__main__":
    main()
