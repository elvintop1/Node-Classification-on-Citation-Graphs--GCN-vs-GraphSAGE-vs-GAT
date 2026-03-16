from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from citation_gnn_benchmark.config import save_yaml
from citation_gnn_benchmark.data import load_planetoid_dataset
from citation_gnn_benchmark.models import build_model
from citation_gnn_benchmark.trainer import NodeClassificationTrainer
from citation_gnn_benchmark.utils import ensure_dir, get_device, save_json, set_seed, timestamp



def run_experiment(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    graph_bundle = load_planetoid_dataset(
        root_dir=cfg["data"].get("root_dir", "data"),
        dataset_name=cfg["data"]["dataset_name"],
        normalize_features=bool(cfg["data"].get("normalize_features", True)),
    )

    device = get_device(cfg["system"].get("device", "auto"))
    model = build_model(cfg["model"], in_dim=graph_bundle.num_features, out_dim=graph_bundle.num_classes).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 0.01)),
        weight_decay=float(cfg["training"].get("weight_decay", 5e-4)),
    )

    experiment_name = cfg["experiment"].get("name", "run")
    output_root = ensure_dir(cfg["experiment"].get("output_dir", "outputs"))
    run_dir = ensure_dir(output_root / f"{experiment_name}_{timestamp()}")

    trainer = NodeClassificationTrainer(model=model, data=graph_bundle.data, optimizer=optimizer, device=device, cfg=cfg)
    output = trainer.train()

    pd.DataFrame(output.history).to_csv(run_dir / "history.csv", index=False)
    output.predictions_df.to_csv(run_dir / "predictions.csv", index=False)
    np.save(run_dir / "embeddings.npy", output.embeddings)
    pd.DataFrame(output.test_metrics["confusion_matrix"]).to_csv(run_dir / "confusion_matrix.csv", index=False)
    save_json(output.test_metrics["classification_report"], run_dir / "classification_report.json")
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    save_yaml(cfg, run_dir / "resolved_config.yaml")

    metrics_payload = {
        "best_epoch": output.best_epoch,
        "device": str(device),
        "config": cfg,
        "train_metrics": {k: v for k, v in output.train_metrics.items() if k not in {"classification_report", "confusion_matrix"}},
        "val_metrics": {k: v for k, v in output.val_metrics.items() if k not in {"classification_report", "confusion_matrix"}},
        "test_metrics": {k: v for k, v in output.test_metrics.items() if k not in {"classification_report", "confusion_matrix"}},
    }
    save_json(metrics_payload, run_dir / "metrics.json")

    return {
        "output_dir": str(run_dir),
        "best_epoch": output.best_epoch,
        "train_metrics": metrics_payload["train_metrics"],
        "val_metrics": metrics_payload["val_metrics"],
        "test_metrics": metrics_payload["test_metrics"],
    }
