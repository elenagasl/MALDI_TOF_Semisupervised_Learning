# Experiment 02 — Species-Specific Recommender (Paper Objective 2)

Semi-supervised Neural Collaborative Filtering (NCF) for MALDI-TOF AMR prediction,
trained independently for each bacterial species.

## Architecture

Each `(spectrum, antibiotic)` pair is treated as an interaction:

```
score = drug_embedding · spectrum_embedding
```

- **Spectrum encoder**: MLP (6 000 → 512 → 256 → 128) or Transformer over m/z bins
- **Drug encoder**: configurable — ECFP MLP, SMILES GRU/CNN/Transformer, one-hot

The key advantage over binary MLPs is that the model **shares information across
antibiotics** within a species via the joint interaction space, without requiring
complete (non-NaN) rows.

## Files

| File | Description |
|------|-------------|
| `train_species_recommender.py` | 5-fold CV training loop for one species |
| `models/amr_recommender.py` | `AMRModel` and `MaldiLightningModule` base class |
| `models/spectrum_encoder.py` | MLP and Transformer spectrum encoders |
| `datasets/recommender_dataset.py` | `RecommenderDataset` — yields `(spectrum, drug)` interaction triplets |
| `datasets/drug_embedders.py` | All drug embedding modules |
| `datasets/metrics.py` | Micro/macro AUC, IC-level AUC |
| `datasets/data_utils.py` | DRIAMS AMR data module |

## Comparison baseline

Results from this experiment are directly compared against the binary and
multilabel MLP from Experiment 01 to quantify the gain from modelling
cross-antibiotic interactions.
