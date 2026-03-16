#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import json
from pathlib import Path

import pandas as pd

from citation_gnn_benchmark.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed experiment folders into CSV tables.")
    parser.add_argument("--runs-root", type=str, required=True, help="Directory containing run folders with metrics.json.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write summary CSV / JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_dir = ensure_dir(args.out_dir)

    records = []
    for metrics_path in sorted(runs_root.glob("**/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        cfg = payload["config"]
        records.append(
            {
                "dataset": cfg["data"]["dataset_name"],
                "model": cfg["model"]["name"],
                "seed": cfg["experiment"]["seed"],
                "best_epoch": payload["best_epoch"],
                "train_accuracy": payload["train_metrics"]["accuracy"],
                "val_accuracy": payload["val_metrics"]["accuracy"],
                "test_accuracy": payload["test_metrics"]["accuracy"],
                "train_macro_f1": payload["train_metrics"]["macro_f1"],
                "val_macro_f1": payload["val_metrics"]["macro_f1"],
                "test_macro_f1": payload["test_metrics"]["macro_f1"],
                "train_weighted_f1": payload["train_metrics"]["weighted_f1"],
                "val_weighted_f1": payload["val_metrics"]["weighted_f1"],
                "test_weighted_f1": payload["test_metrics"]["weighted_f1"],
                "output_dir": str(metrics_path.parent),
            }
        )

    if not records:
        raise FileNotFoundError(f"No metrics.json files found under {runs_root}")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "all_runs.csv", index=False)

    agg = []
    for (dataset, model), group in df.groupby(["dataset", "model"]):
        row = {"dataset": dataset, "model": model, "num_runs": int(len(group))}
        for metric in [
            "train_accuracy",
            "val_accuracy",
            "test_accuracy",
            "train_macro_f1",
            "val_macro_f1",
            "test_macro_f1",
            "train_weighted_f1",
            "val_weighted_f1",
            "test_weighted_f1",
            "best_epoch",
        ]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        agg.append(row)

    agg_df = pd.DataFrame(agg)
    agg_df.to_csv(out_dir / "aggregate_results.csv", index=False)
    (out_dir / "aggregate_results.json").write_text(
        json.dumps(agg_df.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    print(f"Saved summaries to {out_dir}")


if __name__ == "__main__":
    main()
