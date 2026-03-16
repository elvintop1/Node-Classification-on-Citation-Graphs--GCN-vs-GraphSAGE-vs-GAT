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

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregate Macro-F1 summary from aggregate_results.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to aggregate_results.csv")
    parser.add_argument("--out", type=str, required=True, help="Output image path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    datasets = list(df["dataset"].drop_duplicates())
    models = list(df["model"].drop_duplicates())

    width = 0.25
    x = range(len(datasets))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("dataset")
        means = [sub.loc[d, "test_macro_f1_mean"] for d in datasets]
        stds = [sub.loc[d, "test_macro_f1_std"] for d in datasets]
        positions = [p + (i - (len(models) - 1) / 2) * width for p in x]
        ax.bar(positions, means, width=width, label=model)
        ax.errorbar(positions, means, yerr=stds, fmt="none", capsize=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Citation Graph Benchmark: Test Macro-F1")
    ax.legend()
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
