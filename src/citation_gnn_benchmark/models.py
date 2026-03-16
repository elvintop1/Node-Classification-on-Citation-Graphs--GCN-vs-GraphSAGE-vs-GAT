from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GCNNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.classifier = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, return_embeddings: bool = False):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embeddings = x
        logits = self.classifier(x, edge_index)
        if return_embeddings:
            return logits, embeddings
        return logits


class GraphSAGENet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.classifier = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, return_embeddings: bool = False):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embeddings = x
        logits = self.classifier(x, edge_index)
        if return_embeddings:
            return logits, embeddings
        return logits


class GATNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.6,
        heads: int = 8,
        gat_out_heads: int = 1,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        hidden_out_dim = hidden_dim * heads
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_out_dim, hidden_dim, heads=heads, dropout=dropout))
            hidden_out_dim = hidden_dim * heads
        self.classifier = GATConv(hidden_out_dim, out_dim, heads=gat_out_heads, concat=False, dropout=dropout)

    def forward(self, x, edge_index, return_embeddings: bool = False):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embeddings = x
        logits = self.classifier(x, edge_index)
        if return_embeddings:
            return logits, embeddings
        return logits



def build_model(model_cfg: dict, in_dim: int, out_dim: int) -> nn.Module:
    name = model_cfg["name"].strip().lower()
    common = {
        "in_dim": in_dim,
        "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
        "out_dim": out_dim,
        "num_layers": int(model_cfg.get("num_layers", 2)),
        "dropout": float(model_cfg.get("dropout", 0.5)),
    }
    if name == "gcn":
        return GCNNet(**common)
    if name == "graphsage":
        return GraphSAGENet(**common)
    if name == "gat":
        return GATNet(
            **common,
            heads=int(model_cfg.get("heads", 8)),
            gat_out_heads=int(model_cfg.get("gat_out_heads", 1)),
        )
    raise ValueError(f"Unsupported model name: {model_cfg['name']}")
