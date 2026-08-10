# GRASP Interaction MLP

Species-aware global MLP baseline for AMR prediction from MALDI-TOF spectra.
This folder is nested under `grasp/` because it is part of the same GRASP
experimental block.

The model is trained globally with all species but receives a trainable species embedding:

```text
MALDI spectrum -> 512 -> 256 -> 128
species_id -> Embedding(32)
concat(128 + 32) -> 128 -> 64 -> n_antibiotics
```

Missing AST labels are ignored with masked BCE.

## Experiments

- `scripts/run_grasp_mlp.py`: in-distribution 5-fold evaluation.
- `scripts/run_grasp_mlp_ood.py`: OOD zero-shot plus fine-tuning on 20% of OOD.

Both use `--evaluation-panel species` by default, so each species/fold is evaluated on the same sparse species panel used by the semi-supervised baselines.

## Default Training

- Learning rate: `1e-4`
- Dropout: `0.1`
- Weight decay: `1e-5`
- Early stopping: validation Patient-AUC
- Patience: `30`

`Candida albicans` and `Streptococcus pneumoniae` are excluded by default with robust species-name normalization.

## Commands

In-distribution:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python scripts/run_grasp_mlp.py \
  --output-dir results/grasp_mlp_ind \
  --device cuda
```

OOD:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python scripts/run_grasp_mlp_ood.py \
  --output-dir results/grasp_mlp_ood \
  --device cuda
```

## Outputs

- `metrics_by_fold.csv`
- `metrics_summary.csv`
- `global_metrics_by_model.csv`
- `species_metrics_by_model.csv`
- `auc_by_species_antibiotic/*.csv`
- `predictions/species_aware_global_mlp/*.npz`
- `config.json`
