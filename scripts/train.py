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

from citation_gnn_benchmark.config import apply_overrides, load_config
from citation_gnn_benchmark.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GNN for citation-graph node classification.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Override config values with key=value, e.g. data.dataset_name=PubMed or training.max_epochs=500. "
            "May be passed multiple times."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    result = run_experiment(cfg)
    out_dir = Path(result["output_dir"]).resolve()
    print(f"Finished. Outputs saved to: {out_dir}")
    print("Test metrics:")
    for k, v in result["test_metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
