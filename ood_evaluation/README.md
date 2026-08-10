# OOD Evaluation

External validation on the MS-UMG Gottingen cohort.

## Models

This experiment evaluates the semi-supervised models only:

- Semi-supervised binary MLP.
- Semi-supervised multi-output MLP.
- Species-specific recommender.
- Species-conditioned GRASP.
- Multi-head GRASP.
- Hypernetwork GRASP.

Supervised complete-profile baselines are intentionally excluded.

## Protocol

For each of five folds:

1. Build the same in-distribution train/validation/test partitions used by the previous experiments.
2. Train each model on the in-distribution train split and validate on the in-distribution validation split.
3. Perform zero-shot prediction on the OOD test split.
4. Fine-tune from the in-distribution model using 20% of the OOD cohort.
5. Predict again on the remaining 80% OOD test split.

The OOD test split is shared across all models within each fold.

For fine-tuning, the 20% adaptation subset is internally split into fine-tuning train/validation subsets for early stopping. The default validation fraction is `--finetune-val-size 0.25`, so the held-out 80% OOD test split is not touched.

## Inputs

Default in-distribution pickle:

```text
/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl
```

Default OOD pickle:

```text
/export/data_ml4ds/bacteria_id/MALDIAlign_Alex/MSUMG_study_full.pkl
```

Both payloads are expected to expose:

```python
X = payload["data"]
y_species = payload["label"]
amr = payload["amr"]
antibiotics = list(payload["antibiotics"])
```

The script aligns both datasets to shared species and shared antibiotics before training/evaluation.

## Run

From this folder:

```bash
python scripts/run_ood_evaluation.py \
  --in-pickle /export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl \
  --ood-pickle /export/data_ml4ds/bacteria_id/MALDIAlign_Alex/MSUMG_study_full.pkl \
  --output-dir results/ood_evaluation \
  --device cuda
```

To reproduce the older global-GRASP OOD protocol used by the standalone Gottingen scripts:

```bash
python scripts/run_ood_evaluation.py \
  --protocol legacy_global_ood \
  --output-dir results/ood_legacy_global \
  --device cuda \
  --models multihead_grasp hypernetwork_grasp \
  --legacy-source-val-size 0.15 \
  --batch-size 1024 \
  --prediction-batch-size 512 \
  --max-epochs 300 \
  --patience 15 \
  --finetune-epochs 100 \
  --finetune-patience 10 \
  --learning-rate 1e-3 \
  --dropout 0.2 \
  --weight-decay 0.0 \
  --early-stopping-metric loss
```

Short smoke test:

```bash
python scripts/run_ood_evaluation.py \
  --output-dir results/ood_smoke_test \
  --device cuda \
  --max-epochs 3 \
  --finetune-epochs 2 \
  --models species_conditioned_grasp
```

## Outputs

- `global_metrics_by_model.csv`: global metrics by `scenario` and model.
  - `scenario` is `zero_shot` or `finetuned`.
  - `patient_auc_by_sample`: mean Patient-AUC across evaluable OOD test samples.
  - `micro_auc_by_species`: mean of species-level Micro-AUC values.
  - `macro_auc_by_species_antibiotic`: mean AUC across evaluable species-antibiotic tasks.
- `species_metrics_by_model.csv`: same metrics by `scenario`, model and species.
- `auc_by_species_antibiotic/<scenario>__<model>.csv`: one CSV per scenario/model with AUC for each species-antibiotic task.
- `metrics_by_fold.csv`: fold-level monitoring table.
- `metrics_summary.csv`: fold-level mean and standard deviation by scenario/model.
- `predictions/<model>/<scenario>_<species>_fold_<k>.npz`: OOD test labels and predictions.
- `config.json`: exact runtime configuration.
