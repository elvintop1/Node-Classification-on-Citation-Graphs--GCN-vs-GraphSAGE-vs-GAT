from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import Compose, NormalizeFeatures


@dataclass
class GraphBundle:
    dataset_name: str
    num_features: int
    num_classes: int
    data: object
    dataset: object


PLANETOID_NAMES = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}


def canonicalize_planetoid_name(name: str) -> str:
    key = name.strip().lower()
    if key not in PLANETOID_NAMES:
        raise ValueError(f"Unsupported Planetoid dataset: {name}. Choose from {list(PLANETOID_NAMES.values())}")
    return PLANETOID_NAMES[key]



def load_planetoid_dataset(root_dir: str | Path, dataset_name: str, normalize_features: bool = True) -> GraphBundle:
    dataset_name = canonicalize_planetoid_name(dataset_name)
    root_dir = Path(root_dir)
    transforms = []
    if normalize_features:
        transforms.append(NormalizeFeatures())
    transform = Compose(transforms) if transforms else None
    dataset = Planetoid(root=str(root_dir / dataset_name), name=dataset_name, transform=transform)
    data = dataset[0]
    return GraphBundle(
        dataset_name=dataset_name,
        num_features=int(dataset.num_features),
        num_classes=int(dataset.num_classes),
        data=data,
        dataset=dataset,
    )
