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
from copy import deepcopy
from pathlib import Path

import pandas as pd

from citation_gnn_benchmark.config import deep_update, load_config, save_yaml
from citation_gnn_benchmark.experiment import run_experiment
from citation_gnn_benchmark.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed benchmark across citation datasets and GNN models.")
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark YAML config.")
    return parser.parse_args()


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
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
    ]
    grouped = df.groupby(["dataset", "model"], as_index=False)
    records = []
    for (dataset, model), group in grouped:
        record = {"dataset": dataset, "model": model, "num_runs": int(len(group))}
        for col in metric_cols:
            if col in group.columns:
                record[f"{col}_mean"] = float(group[col].mean())
                record[f"{col}_std"] = float(group[col].std(ddof=0))
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    bench_cfg = load_config(args.config)

    benchmark = bench_cfg["benchmark"]
    out_dir = ensure_dir(benchmark["output_dir"])
    runs_root = ensure_dir(out_dir / "runs")
    model_templates = bench_cfg["model_templates"]
    shared_overrides = bench_cfg.get("shared_overrides", {})

    records = []
    for dataset_name in benchmark["datasets"]:
        for model_name in benchmark["models"]:
            template_path = model_templates[model_name]
            base_cfg = load_config(template_path)
            base_cfg = deep_update(base_cfg, shared_overrides)
            base_cfg["data"]["dataset_name"] = dataset_name
            base_cfg["model"]["name"] = model_name

            for seed in benchmark["seeds"]:
                cfg = deepcopy(base_cfg)
                cfg["experiment"]["seed"] = seed
                cfg["experiment"]["name"] = f"{dataset_name.lower()}_{model_name}_seed{seed}"
                cfg["experiment"]["output_dir"] = str(runs_root)

                print(f"Running {dataset_name} | {model_name} | seed={seed}")
                result = run_experiment(cfg)
                record = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": seed,
                    "output_dir": result["output_dir"],
                    "best_epoch": result["best_epoch"],
                }
                for split_name in ["train", "val", "test"]:
                    metrics = result[f"{split_name}_metrics"]
                    record[f"{split_name}_accuracy"] = metrics["accuracy"]
                    record[f"{split_name}_macro_f1"] = metrics["macro_f1"]
                    record[f"{split_name}_weighted_f1"] = metrics["weighted_f1"]
                records.append(record)

    all_runs = pd.DataFrame(records)
    all_runs_path = out_dir / "all_runs.csv"
    all_runs.to_csv(all_runs_path, index=False)

    agg = aggregate_results(all_runs)
    agg_path = out_dir / "aggregate_results.csv"
    agg.to_csv(agg_path, index=False)

    json_path = out_dir / "aggregate_results.json"
    json_path.write_text(json.dumps(agg.to_dict(orient="records"), indent=2), encoding="utf-8")

    save_yaml(bench_cfg, out_dir / "benchmark_resolved.yaml")
    print(f"Saved run-level results to: {all_runs_path}")
    print(f"Saved aggregate summary to: {agg_path}")


if __name__ == "__main__":
    main()
