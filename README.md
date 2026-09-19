# Druglib thesis-to-conference reproduction

This repository re-runs the undergraduate thesis pipeline on the official UCI Drug Reviews (Druglib.com) dataset, ID 461, DOI 10.24432/C55G6J. It preserves the thesis models and grids (Random Forest, Logistic Regression, SVC; `cv=3`, `random_state=42`, balanced class weights) and documents reporting corrections and integrity checks.

## Reproduce locally

```powershell
.venv\Scripts\python.exe reproduce\original_pipeline.py --corrected --skip-learning
```

The official TSV files are under `data/raw/`. Outputs are written to `results/` and figures to `figures/`. See [reproduce/README.md](reproduce/README.md), [results/REPORT.md](results/REPORT.md), and [results/reconciliation.md](results/reconciliation.md).

## Kaggle evidence notebook

`kaggle/official_uci_reproduction.ipynb` reproduces the pipeline in a visible notebook, preserves the thesis-reported values, and includes corrected ROC-AUC, confusion matrices, class distributions, review-length analysis, word cloud, and top-term outputs.
