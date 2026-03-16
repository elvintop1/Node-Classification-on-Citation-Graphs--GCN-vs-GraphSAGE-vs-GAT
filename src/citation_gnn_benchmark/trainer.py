from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from citation_gnn_benchmark.metrics import compute_classification_metrics
from citation_gnn_benchmark.utils import save_json


@dataclass
class TrainerOutput:
    best_epoch: int
    history: list[dict[str, Any]]
    train_metrics: dict[str, Any]
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    predictions_df: pd.DataFrame
    embeddings: np.ndarray


class NodeClassificationTrainer:
    def __init__(self, model, data, optimizer, device, cfg: dict[str, Any]):
        self.model = model
        self.data = data.to(device)
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.training_cfg = cfg["training"]
        self.monitor = self.training_cfg.get("monitor", "val_macro_f1")
        self.monitor_mode = self.training_cfg.get("monitor_mode", "max")
        if self.monitor_mode not in {"max", "min"}:
            raise ValueError("monitor_mode must be 'max' or 'min'")

    def _forward(self, return_embeddings: bool = False):
        return self.model(self.data.x, self.data.edge_index, return_embeddings=return_embeddings)

    def _loss(self, logits: torch.Tensor) -> torch.Tensor:
        label_smoothing = float(self.training_cfg.get("label_smoothing", 0.0))
        return F.cross_entropy(
            logits[self.data.train_mask],
            self.data.y[self.data.train_mask],
            label_smoothing=label_smoothing,
        )

    @torch.no_grad()
    def evaluate_mask(self, mask_name: str) -> dict[str, Any]:
        self.model.eval()
        logits = self._forward(return_embeddings=False)
        preds = logits.argmax(dim=-1)
        mask = getattr(self.data, f"{mask_name}_mask")
        y_true = self.data.y[mask].detach().cpu().numpy()
        y_pred = preds[mask].detach().cpu().numpy()
        return compute_classification_metrics(y_true, y_pred)

    @torch.no_grad()
    def infer_all(self) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        logits, embeddings = self._forward(return_embeddings=True)
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()
        return preds, embeddings

    def _metric_value(self, history_record: dict[str, Any]) -> float:
        if self.monitor not in history_record:
            raise KeyError(f"Monitor key {self.monitor} not found in history record.")
        return float(history_record[self.monitor])

    def _is_better(self, new_value: float, best_value: float | None) -> bool:
        if best_value is None:
            return True
        if self.monitor_mode == "max":
            return new_value > best_value
        return new_value < best_value

    def train(self) -> TrainerOutput:
        best_state = None
        best_value = None
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[dict[str, Any]] = []

        max_epochs = int(self.training_cfg.get("max_epochs", 300))
        patience = int(self.training_cfg.get("early_stopping_patience", 50))
        log_every = int(self.training_cfg.get("log_every", 10))
        grad_clip_norm = self.training_cfg.get("grad_clip_norm", None)

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            logits = self._forward(return_embeddings=False)
            loss = self._loss(logits)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(grad_clip_norm))
            self.optimizer.step()

            train_metrics = self.evaluate_mask("train")
            val_metrics = self.evaluate_mask("val")

            record = {
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu().item()),
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
            }
            history.append(record)

            metric_value = self._metric_value(record)
            if self._is_better(metric_value, best_value):
                best_value = metric_value
                best_epoch = epoch
                best_state = deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:03d} | loss={record['train_loss']:.4f} | "
                    f"train_acc={record['train_accuracy']:.4f} | val_acc={record['val_accuracy']:.4f} | "
                    f"val_macro_f1={record['val_macro_f1']:.4f}"
                )

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
                break

        if best_state is None:
            raise RuntimeError("Training did not produce a best model state.")

        self.model.load_state_dict(best_state)

        train_metrics = self.evaluate_mask("train")
        val_metrics = self.evaluate_mask("val")
        test_metrics = self.evaluate_mask("test")
        preds, embeddings = self.infer_all()

        split_names = np.full(shape=(preds.shape[0],), fill_value="unlabeled", dtype=object)
        for split in ["train", "val", "test"]:
            mask = getattr(self.data, f"{split}_mask").detach().cpu().numpy().astype(bool)
            split_names[mask] = split

        predictions_df = pd.DataFrame(
            {
                "node_id": np.arange(len(preds), dtype=int),
                "split": split_names,
                "y_true": self.data.y.detach().cpu().numpy().astype(int),
                "y_pred": preds.astype(int),
            }
        )

        return TrainerOutput(
            best_epoch=best_epoch,
            history=history,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            predictions_df=predictions_df,
            embeddings=embeddings,
        )
