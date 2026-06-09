# Experiment 04 — Out-of-Distribution Evaluation (Paper Objective 4)

Assesses the external generalisation of all modelling strategies to an
independent clinical cohort from Universitätsmedizin Göttingen (MS-UMG),
with and without target-site adaptation via fine-tuning.

## Protocol

For each model and each of 5 independent runs:

1. **Identify overlap**: keep species and antibiotics common to the source
   training data (MARISMA + DRIAMS) and the Göttingen cohort.
2. **Train on source**: fit the model on the full source training set.
3. **Split Göttingen**: stratify Göttingen samples per species into:
   - Fine-tuning subset (20% of each species)
   - OOD test subset (remaining 80%)
4. **Zero-shot evaluation**: evaluate the source-trained model on the test set.
5. **Fine-tuning**: adapt the model for a small number of epochs on the
   fine-tuning subset (early stopping, patience 10).
6. **Post-fine-tuning evaluation**: re-evaluate on the same test set.

## Scripts

| Script | Model |
|--------|-------|
| `finetune_mlp.py` | Binary MLP baseline (Experiment 01) |
| `finetune_species_recommender.py` | Species-specific NCF (Experiment 02) |
| `finetune_global_recommender.py` | Global species-conditioned recommender (Experiment 03) |
| `finetune_multihead.py` | Multi-head global recommender (Experiment 03) |

## Metrics

All metrics are reported **before** (zero-shot) and **after** (fine-tuned) adaptation:

- Global micro AUC — AUC over all observed (sample, antibiotic) pairs
- Global macro AUC — mean per-antibiotic AUC
- Per-species micro/macro AUC
- Per-antibiotic AUC

## Key question

Which architecture is most robust to cross-site domain shift, and how much
does target-site fine-tuning improve generalisation for each model family?
