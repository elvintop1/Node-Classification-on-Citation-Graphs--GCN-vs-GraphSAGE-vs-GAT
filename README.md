# Node Classification on Citation Graphs: GCN vs GraphSAGE vs GAT

Full research-style source code for an empirical study on node classification over citation graphs using **GCN**, **GraphSAGE**, and **GAT**.

The codebase is designed for:
- reproducible benchmarking on **Cora**, **CiteSeer**, and **PubMed**
- multiple random seeds
- automatic logging of **accuracy**, **macro-F1**, **weighted-F1**, per-class metrics, and confusion matrices
- export of benchmark summaries to **CSV** and **LaTeX**

## Project structure

```text
citation_gnn_benchmark_full_source/
├── configs/
│   ├── benchmark_planetoid.yaml
│   ├── gat_cora.yaml
│   ├── gcn_cora.yaml
│   └── graphsage_cora.yaml
├── scripts/
│   ├── export_latex_table.py
│   ├── plot_results.py
│   ├── run_benchmark.py
│   ├── summarize_results.py
│   └── train.py
├── src/citation_gnn_benchmark/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── experiment.py
│   ├── metrics.py
│   ├── models.py
│   ├── trainer.py
│   └── utils.py
├── pyproject.toml
└── requirements.txt
```

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

If `torch-geometric` installation fails because of your local CUDA / PyTorch build, install PyTorch first, then follow the official PyG instructions for your platform, and finally run `pip install -e .` again.

## Single-run training

### GCN on Cora
```bash
python scripts/train.py --config configs/gcn_cora.yaml
```

### GraphSAGE on Cora
```bash
python scripts/train.py --config configs/graphsage_cora.yaml
```

### GAT on Cora
```bash
python scripts/train.py --config configs/gat_cora.yaml
```

Each run will create an output folder containing:
- `metrics.json`
- `history.csv`
- `classification_report.json`
- `confusion_matrix.csv`
- `predictions.csv`
- `embeddings.npy`
- `best_model.pt`
- `resolved_config.yaml`

## Full benchmark across datasets / models / seeds

```bash
python scripts/run_benchmark.py --config configs/benchmark_planetoid.yaml
```

This will run:
- datasets: `Cora`, `CiteSeer`, `PubMed`
- models: `GCN`, `GraphSAGE`, `GAT`
- seeds: configurable in the benchmark YAML

Benchmark outputs:
- `all_runs.csv`
- `aggregate_results.csv`
- `aggregate_results.json`

## Plotting benchmark summaries

```bash
python scripts/plot_results.py \
  --csv outputs/benchmark_planetoid/aggregate_results.csv \
  --out outputs/benchmark_planetoid/benchmark_macro_f1.png
```

## Exporting a LaTeX table

```bash
python scripts/export_latex_table.py \
  --csv outputs/benchmark_planetoid/aggregate_results.csv \
  --out outputs/benchmark_planetoid/results_table.tex
```

## Reconstruct summary tables from completed runs

```bash
python scripts/summarize_results.py \
  --runs-root outputs/benchmark_planetoid/runs \
  --out-dir outputs/benchmark_planetoid/reconstructed_summary
```
