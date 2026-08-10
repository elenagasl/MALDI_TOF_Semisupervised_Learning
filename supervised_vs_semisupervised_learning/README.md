# Supervised vs Semi-Supervised Learning

Experiments comparing conventional supervised baselines against models trained on incomplete susceptibility profiles.

## Models

- Supervised binary MLP: one MLP per species-antibiotic pair, trained only on complete AST profiles.
- Supervised multi-output MLP: one MLP per species, trained only on complete AST profiles.
- Semi-supervised binary MLP: one MLP per species-antibiotic pair, trained on observed labels from incomplete AST profiles.
- Semi-supervised multi-output MLP: one MLP per species with masked BCE over incomplete AST profiles.
- Species-specific recommender: one recommender per species trained on observed MALDI-antibiotic interactions.

All models are evaluated on the same test fold for each species. Semi-supervised models can use more training samples, but test indices are shared across all approaches.

## Input

Default in-distribution pickle:

```text
/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl
```

Expected payload:

```python
X = payload["data"]
y_species = payload["label"]
amr = payload["amr"]
antibiotics = list(payload["antibiotics"])
```

## Training Setup

- Five-fold cross-validation.
- Batch size: 64.
- Maximum epochs: 1200.
- Early stopping patience: 15 epochs.
- Fixed MLP architecture: `input_dim -> 512 -> 256 -> 128 -> output`.
- Activation: ReLU.
- Optimizer: Adam.
- Learning rate: 1e-3.

The recommender uses the fixed architecture specified in the thesis experiment:

```text
MALDI encoder: 6000 -> 512 -> 256 -> 128 -> 64
Antibiotic embedding: 30
Interaction MLP: 94 -> 64 -> 32 -> 1
```

## Run

From this folder:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python scripts/run_supervised_vs_semisupervised.py \
  --pickle /export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl \
  --output-dir results/supervised_vs_semisupervised \
  --device cuda
```

For a short smoke test:

```bash
python scripts/run_supervised_vs_semisupervised.py \
  --output-dir results/smoke_test \
  --device cuda \
  --max-epochs 3 \
  --only-species "Escherichia coli"
```

## Outputs

- `global_metrics_by_model.csv`: global comparison by model.
  - `patient_auc_by_sample`: mean Patient-AUC across evaluable test samples.
  - `micro_auc_by_species`: mean of species-level Micro-AUC values.
  - `macro_auc_by_species_antibiotic`: mean AUC across evaluable species-antibiotic tasks.
- `species_metrics_by_model.csv`: same metrics broken down by species and model.
- `auc_by_species_antibiotic/<model>.csv`: one CSV per model with AUC for each species-antibiotic task.
- `metrics_by_fold.csv`: fold-level monitoring table with one row per model, species, and fold.
- `metrics_summary.csv`: fold-level mean and standard deviation by model.
- `predictions/<model>/<species>_fold_<k>.npz`: test indices, antibiotic indices, true labels, and predictions.
- `config.json`: exact runtime configuration.

## Notes

The complete-profile antibiotic panel is selected with a deterministic greedy procedure: antibiotics are ordered by availability and added while keeping at least `min_train_samples` complete training profiles and `min_val_samples` complete validation profiles. This keeps the logic explicit and isolated in `src/supervised_semisupervised/panels.py`.
