# GRASP

Global recommender architectures for antimicrobial susceptibility prediction
from MALDI-TOF spectra.

The models learn from sparse AST profiles and predict:

```text
spectrum + species + antibiotic -> probability of resistance
```

## Models

### Species-Conditioned GRASP

```text
MALDI encoder: input_dim -> 512 -> 256 -> 128 -> 64 
Antibiotic embedding: 32
Species embedding: 16
Interaction: 64 + 32 + 16 -> 32 -> 1
```

### Multi-Head GRASP

```text
Shared MALDI encoder: input_dim -> 512 -> 128
Species-specific MALDI heads: 128 -> 64
Antibiotic embedding: 32
Species-specific interaction MLPs: 64 + 32 -> 32 -> 1
```

### Hypernetwork GRASP

```text
MALDI encoder: input_dim -> 512 -> 256 -> 128 -> 64
Species embedding: 16
Species hypernetwork: 16 -> 64 -> 64
Corrected MALDI embedding: maldi_embedding + alpha * species_correction
alpha initialization: 0.05
Antibiotic embedding: 32
Species-specific interaction MLPs: 64 + 32 -> 32 -> 1
```

`input_dim` is inferred from the pickle (`payload["data"].shape[1]`), so the same code supports 6000-bin spectra or a reduced 600-feature representation.

The global multi-head and hypernetwork models follow the benchmark-style architecture with GELU activations, dropout `0.1`, compact MALDI/drug embeddings, and species-specific final interaction MLPs.

## Splits

The experiment uses the same fold construction as `supervised_vs_semisupervised_learning`:

- Five folds by default.
- Per-species KFold with the same `random_seed`.
- Per-species train/validation split with `random_seed + fold`.
- Global folds are formed by concatenating the per-species train, validation, and test indices.

This keeps the test partitions aligned with the supervised/semi-supervised experiment.

## Optimization

The architecture dimensions are fixed as specified above.

- Optimizer: Adam.
- Learning rate: 1e-4.
- Weight decay: 1e-5.

Training settings:

- Batch size: 64.
- Maximum epochs: 1200.
- Early stopping patience: 30 epochs.
- Early stopping metric: validation Patient-AUC by default. Use `--early-stopping-metric loss` to recover validation-loss stopping.
- GPU support through PyTorch (`--device cuda`).

## Run

From this folder:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python scripts/run_grasp.py \
  --pickle /export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl \
  --output-dir results/grasp \
  --device cuda \
  --evaluation-panel species
```

Short smoke test with one model:

```bash
python scripts/run_grasp.py \
  --output-dir results/grasp_smoke_test \
  --device cuda \
  --max-epochs 3 \
  --models species_conditioned_grasp
```

## Interaction MLP

The species-aware interaction MLP baseline is stored in:

```text
grasp/interaction_mlp/
```

Run it from that folder so its local `src/` directory is added correctly:

```bash
cd interaction_mlp
python scripts/run_grasp_mlp.py --output-dir results/grasp_mlp_ind --device cuda
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
- `config.json`: exact runtime configuration and species-code mapping.
