# Experiment 03 — Global Species-Aware Recommender (Paper Objective 3)

A single global NCF model that jointly predicts AMR for all bacterial species,
overcoming the sample-size limitations of per-species training.

## Motivation

Species-specific models (Experiment 02) are limited by the number of samples
available per species. A global model can transfer knowledge across species by
learning a shared MALDI-TOF spectrum representation while accounting for
species identity through explicit conditioning.

## Conditioning strategies compared

| Script | Strategy | Description |
|--------|----------|-------------|
| `train_global_conditioned.py` | **Species embedding** | Species ID → learnable vector concatenated to input |
| `train_global_multihead.py` | **Multi-head** | Shared encoder + one output head per species |
| `train_global_hypernetwork.py` | **Hypernetwork** | Species identity generates parameters for the final layer |
| `train_species_multihead.py` | **Multi-head (v2)** | Alternative multi-head with species-specific projection layers |

## Model (NCF + AMR context encoder)

```
maldi_emb  = MLP(spectrum)                     # 6000 → 512 → 256 → 128 → 64
drug_emb   = Embedding(antibiotic_id)          # → drug_emb_dim (16)
family_emb = Embedding(antibiotic_family_id)   # → family_emb_dim (8)
amr_emb    = MLP([amr_values ; obs_mask])      # → 64  (context encoder)
score      = CF_net([maldi; drug; family; amr])
```

The AMR context encoder receives **both** the observed resistance values and a
binary mask, so the model distinguishes true susceptibility (label 0) from
unobserved measurements (NaN → mask 0).

## Files

### Core models
| File | Description |
|------|-------------|
| `models/ncf.py` | Neural Collaborative Filtering + AMR context (main model) |
| `models/ncf_graph.py` | NCF with graph diffusion over co-resistance graph |
| `models/ncf_pred_context.py` | NCF conditioned on predicted context |
| `models/maldi_transformer.py` | Transformer-based MALDI encoder |
| `models/attention.py` | Scaled dot-product attention (math / flash / memory-efficient) |
| `models/embeddings.py` | Discrete and continuous embeddings with masking and CLS token |
| `models/positional_encoding.py` | Sinusoidal and learnable positional encodings |

### Datasets
| File | Description |
|------|-------------|
| `datasets/global_dataset.py` | Multi-species `(spectrum, species, drug)` interaction dataset with AMR context masking |
| `datasets/pred_context_dataset.py` | Prediction-context variant |
| `datasets/graph_dataset.py` | Graph-based dataset for co-resistance structure |

### Enrichment
| File | Description |
|------|-------------|
| `enrichment/amr_context_builder.py` | Builds antibiotic-family context vectors from resistance patterns |
| `enrichment/antibiotic_filter.py` | Filters antibiotics with insufficient observations |

### `exploratory/`
Contains experimental architectures developed during the research process:
bio-prior gated models, logit enrichment, two-step training, teacher–student
distillation, and hypernetwork variants. These are kept for reproducibility
but are not part of the final reported comparisons.
