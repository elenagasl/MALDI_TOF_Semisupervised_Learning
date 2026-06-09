# Experiment 01 — Classical ML Benchmark (Paper Objective 1)

Establishes species-specific MLP and LPS baselines for MALDI-TOF AMR prediction.

## Scripts

| Script | Description |
|--------|-------------|
| `mlp_models.py` | MLP model definitions: binary (one per antibiotic), multilabel, LPS |
| `train_benchmark.py` | Optuna hyperparameter search on CPU (200 trials per species/antibiotic) |
| `train_benchmark_gpu.py` | GPU-accelerated hyperparameter search |
| `eval_fair_mlp.py` | **Main evaluation script.** 5-fold CV with binary + multilabel MLPs, directly comparable to global recommender |
| `combine_datasets.py` | Merges DRIAMS-A + MARISMa pickles into a single training file |
| `inspect_ood_dataset.py` | Diagnostic summary of the Göttingen OOD dataset (species, antibiotics, counts) |

## Key design choices

- **Fair comparison**: `eval_fair_mlp.py` uses the *same* global K-Fold splits as the global recommender (Experiment 03), enabling direct AUC comparison.
- **Masked loss for multilabel**: NaN AMR labels are handled via a masked BCE loss — no samples are discarded.
- **LPS excluded from fair comparison**: The LPS model requires complete rows and is not comparable to the recommender; it is benchmarked separately in `mlp_models.py`.

## Outputs

Results are written to `benchmark_outputs/` (not tracked by git) as CSV + JSON files after each fold.
