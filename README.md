# MALDI-TOF AMR Prediction via Structured Deep Learning

**Antimicrobial resistance (AMR) prediction from MALDI-TOF mass spectrometry spectra using neural collaborative filtering and global species-aware recommender systems.**

---

## Overview

This repository contains the complete experimental pipeline for a progressive benchmark study on MALDI-TOF MS-based AMR prediction. The central hypothesis is that resistance prediction improves when it is modelled as a **structured interaction problem** — exploiting relationships across antibiotics and bacterial species — rather than as a collection of isolated binary classifiers.

The study proceeds through four objectives, each building on the previous:

| Objective | Model family | Description |
|-----------|-------------|-------------|
| **1** | Classical ML | Binary MLPs, multilabel MLPs and LPS MLPs as species-specific baselines |
| **2** | Species-specific recommender | Semi-supervised NCF that shares information across antibiotics within each species |
| **3** | Global species-aware recommender | Single model conditioned on species identity; learns a shared MALDI-TOF representation across all species |
| **4** | OOD evaluation | External generalisation to an independent Göttingen / MS-UMG cohort, with and without target-site fine-tuning |

---

## Repository Structure

```
.
├── dataloader/                          # Raw data loading
│   ├── spectrum_object.py               # MALDI-TOF spectrum data structure (m/z, intensity)
│   ├── driams_dataset.py                # DRIAMS multi-centre dataset manager & PyTorch Dataset
│   └── marisma_dataset.py               # MARISMa clinical dataset manager & PyTorch Dataset
│
├── preprocessing/                       # Spectrum preprocessing pipeline
│   ├── pipeline.py                      # All preprocessing transforms (see below)
│   ├── preprocess_driams.py             # DRIAMS: preprocess + save AMR-labelled pickles
│   └── preprocess_marisma.py            # MARISMa: preprocess + save AMR-labelled pickles
│
├── losses/
│   └── pike_kernel.py                   # GPU-accelerated PIKE kernel for spectrum similarity
│
├── visualization/
│   ├── visualize.py                     # Preprocessing visualisation, t-SNE, PIKE plots
│   └── plot_helpers.py                  # Colour palettes, species name dictionaries
│
├── utils/
│   └── config.py                        # Dataset paths and project-level configuration
│
└── experiments/                         # Experimental pipeline (one directory per objective)
    │
    ├── 00_preprocessing_analysis/       # Exploratory preprocessing experiments (notebooks)
    │
    ├── 01_classical_benchmark/          # ── OBJECTIVE 1: Classical ML benchmark ──
    │   ├── mlp_models.py                # MLP model definitions: binary, multilabel, LPS
    │   ├── train_benchmark.py           # Optuna hyperparameter search (CPU)
    │   ├── train_benchmark_gpu.py       # Optuna hyperparameter search (GPU)
    │   ├── eval_fair_mlp.py             # Fair 5-fold evaluation comparable to the recommender
    │   ├── combine_datasets.py          # Merge DRIAMS + MARISMa pickles into a single file
    │   ├── inspect_ood_dataset.py       # Diagnostic inspection of the Göttingen dataset
    │   ├── analysis/
    │   │   └── results_analysis.ipynb   # Interactive analysis of benchmark outputs
    │   └── preliminary/                 # Earlier exploratory experiments (MLP + RF)
    │
    ├── 02_species_recommender/          # ── OBJECTIVE 2: Species-specific recommender ──
    │   ├── train_species_recommender.py # 5-fold CV training for each species
    │   ├── models/
    │   │   ├── amr_recommender.py       # AMRModel: dot-product NCF (drug × spectrum)
    │   │   └── spectrum_encoder.py      # MLP and Transformer spectrum encoders
    │   ├── datasets/
    │   │   ├── recommender_dataset.py   # (sample, drug) interaction dataset with context masking
    │   │   ├── drug_embedders.py        # ECFP, SMILES, CNN, GRU, Transformer drug encoders
    │   │   ├── metrics.py               # AUC metrics: micro, macro, IC-level
    │   │   └── data_utils.py            # DRIAMS AMR data module
    │   └── analysis/
    │       └── auc_analysis.ipynb
    │
    ├── 03_global_recommender/           # ── OBJECTIVE 3: Global species-aware recommender ──
    │   ├── train_global_conditioned.py  # Species embedding conditioning (main variant)
    │   ├── train_global_multihead.py    # Species-specific output heads
    │   ├── train_global_hypernetwork.py # Hypernetwork-based species conditioning
    │   ├── train_species_multihead.py   # Multi-head architecture (species-specific heads + shared encoder)
    │   ├── models/
    │   │   ├── ncf.py                   # NCF with AMR context encoder (main model)
    │   │   ├── ncf_graph.py             # NCF + graph diffusion over antibiotic co-resistance graph
    │   │   ├── ncf_pred_context.py      # NCF with prediction-context conditioning
    │   │   ├── maldi_transformer.py     # Transformer-based MALDI-TOF encoder
    │   │   ├── attention.py             # Scaled dot-product attention (math/flash/memory-efficient)
    │   │   ├── embeddings.py            # Discrete and continuous embeddings with masking and CLS
    │   │   └── positional_encoding.py   # Sinusoidal and learnable positional encodings
    │   ├── datasets/
    │   │   ├── global_dataset.py        # Multi-species (spectrum, species, drug) dataset with AMR context
    │   │   ├── pred_context_dataset.py  # Prediction-context variant of the dataset
    │   │   └── graph_dataset.py         # Graph-based dataset for antibiotic interaction modelling
    │   ├── enrichment/
    │   │   ├── amr_context_builder.py   # Builds antibiotic-family context vectors
    │   │   └── antibiotic_filter.py     # Filters antibiotics by minimum observation count
    │   ├── exploratory/                 # Experimental architectures (bio-prior gating,
    │   │                                #   two-step training, teacher–student, logit enrichment…)
    │   └── analysis/
    │       ├── ncf_test.ipynb           # Unit-level model testing and sanity checks
    │       └── resistance_correlation.ipynb  # AMR resistance pattern correlation analysis
    │
    └── 04_ood_evaluation/              # ── OBJECTIVE 4: OOD generalisation ──
        ├── finetune_mlp.py             # MLP fine-tuning on Göttingen / MS-UMG
        ├── finetune_global_recommender.py  # Global recommender fine-tuning
        ├── finetune_species_recommender.py # Species recommender fine-tuning
        └── finetune_multihead.py       # Multi-head recommender fine-tuning
```

---

## Datasets

### DRIAMS (Drug Resistance in MALDI-TOF MS)
Multi-centre Swiss dataset collected at four hospital sites (DRIAMS-A, B, C, D).
Raw spectra are paired with antibiotic susceptibility test (AST) results from
EUCAST breakpoints. Each site/year is a separate CSV metadata file pointing to
raw `.txt` spectrum files.

- **Reference:** Weis et al., *Nature Medicine*, 2022.
- **Managed by:** `dataloader/driams_dataset.py`

### MARISMa (MALDI-TOF AMR — Hospital Ramón y Cajal, Madrid)
In-house clinical dataset from Hospital Universitario Ramón y Cajal (Madrid, Spain).
Organised by year, genus, species, and study identifier in a hierarchical folder
structure using Bruker `.fid` / `.acqu` format.

- **Managed by:** `dataloader/marisma_dataset.py`

### MS-UMG / Göttingen (OOD test cohort)
Independent cohort from Universitätsmedizin Göttingen used exclusively for
out-of-distribution evaluation in Experiment 04. Never seen during training.

---

## Preprocessing Pipeline

All spectra go through the same sequential pipeline before model training.
Transforms are composable via `SequentialPreprocessor` (`preprocessing/pipeline.py`):

| Step | Class | Purpose |
|------|-------|---------|
| 1 | `VarStabilizer` | Square-root or log transform to stabilise intensity variance |
| 2 | `Smoother` | Savitzky–Golay filter to reduce high-frequency noise |
| 3 | `BaselineCorrecter` | SNIP, ALS or ArPLS baseline removal |
| 4 | `Trimmer` | Restrict m/z range to 2 000–20 000 Da |
| 5 | `Binner` | Aggregate into equal-width bins (default: 3 Da, 6 000 bins) |
| 6 | `StdThresholder` | Zero out intensities below k × σ |
| 7 | `LogScaler` | Log₁₀(x + 1) scaling |

After preprocessing, spectra are serialised as pickle files containing a
NumPy array of shape `(N_samples, 6000)` alongside AMR label matrices of
shape `(N_samples, N_antibiotics)` with `{0, 1, NaN}` entries.

---

## Models

### Experiment 01 — Classical Benchmark

Three MLP families evaluated for AMR prediction:

| Model | Input | Output | Loss |
|-------|-------|--------|------|
| **Binary MLP** | Spectrum (6 000) | 1 logit per antibiotic | BCE |
| **Multilabel MLP** | Spectrum (6 000) | N logits (one per antibiotic) | Masked BCE |
| **LPS MLP** | Spectrum (6 000) | One-hot LPS pattern | Cross-entropy |

Hyperparameters are tuned with Optuna (200 trials, TPE sampler) separately per
(species, antibiotic) pair. Evaluation uses 5-fold global K-Fold.

### Experiment 02 — Species-Specific Recommender

Each species gets its own NCF model:

```
AMRModel(spectrum_embedder, drug_embedder)
    score = drug_embedding · spectrum_embedding
```

The spectrum encoder is a 3-layer MLP (6 000 → 512 → 256 → 128) or a
Transformer over m/z bins. The drug encoder accepts ECFP fingerprints,
SMILES-based GRU/CNN/Transformer representations, or a one-hot embedding.

### Experiment 03 — Global Species-Aware Recommender

A single model replaces per-species models:

```
NCF(spectrum, species_id, antibiotic_id, amr_context)
    maldi_emb = MLP(spectrum)                  # 6000 → 512 → 256 → 128 → 64
    drug_emb  = Embedding(antibiotic_id)       # → drug_emb_dim
    family_emb = Embedding(family_id)          # → family_emb_dim
    amr_emb   = MLP([amr_values; amr_mask])   # → 64  (context encoder)
    score = CF_net( [maldi_emb; drug_emb; family_emb; amr_emb] )
```

The AMR context encoder receives the known resistance labels of the sample
concatenated with a binary observation mask, so the model can distinguish
true susceptibility (0) from unobserved labels (NaN).

Three conditioning strategies are compared:
- **Species embedding** (`train_global_conditioned.py`) — species_id mapped to a
  learnable vector concatenated to the input
- **Multi-head** (`train_global_multihead.py`) — shared encoder + species-specific
  output heads
- **Hypernetwork** (`train_global_hypernetwork.py`) — species identity generates
  parameters for the final layer

### Experiment 04 — OOD Evaluation

All models from experiments 01–03 are evaluated on the Göttingen cohort:

1. **Zero-shot**: model trained on MARISMA + DRIAMS, evaluated directly on Göttingen
2. **Fine-tuned**: 20% of Göttingen samples used for target-site adaptation,
   remaining 80% used for test evaluation

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd maldi-tof-amr-prediction

# Create environment (Python 3.9+)
conda create -n maldi-amr python=3.9
conda activate maldi-amr

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning lightning torchmetrics
pip install scikit-learn pandas numpy scipy tqdm optuna
pip install matplotlib seaborn
```

---

## Reproducing the Experiments

### Step 0 — Preprocessing

Generate the preprocessed pickle files from raw spectra:

```bash
# DRIAMS
python preprocessing/preprocess_driams.py

# MARISMa
python preprocessing/preprocess_marisma.py

# Merge DRIAMS + MARISMa into a single training pickle
python experiments/01_classical_benchmark/combine_datasets.py
```

### Step 1 — Classical Benchmark (Objective 1)

```bash
# Hyperparameter search with Optuna (GPU)
python experiments/01_classical_benchmark/train_benchmark_gpu.py

# Fair 5-fold evaluation with optimised or default hyperparameters
python experiments/01_classical_benchmark/eval_fair_mlp.py \
  --pickle-path /data/COMBINED_MARISMA_DRIAMS_samples.pkl \
  --output-dir /results/benchmark \
  --device auto
```

### Step 2 — Species-Specific Recommender (Objective 2)

```bash
python experiments/02_species_recommender/train_species_recommender.py
```

### Step 3 — Global Recommender (Objective 3)

```bash
# Species-embedding conditioning (main variant)
python experiments/03_global_recommender/train_global_conditioned.py

# Multi-head variant
python experiments/03_global_recommender/train_global_multihead.py

# Hypernetwork variant
python experiments/03_global_recommender/train_global_hypernetwork.py
```

### Step 4 — OOD Evaluation (Objective 4)

```bash
# Fine-tune global recommender on Göttingen cohort
python experiments/04_ood_evaluation/finetune_global_recommender.py \
  --source-pickle /data/COMBINED_MARISMA_DRIAMS_samples.pkl \
  --ood-pickle /data/MSUMG_study_full.pkl \
  --output-dir /results/ood_global \
  --device cuda --n-runs 5 --finetune-frac 0.2

# Fine-tune MLP baseline
python experiments/04_ood_evaluation/finetune_mlp.py \
  --source-pickle /data/COMBINED_MARISMA_DRIAMS_samples.pkl \
  --ood-pickle /data/MSUMG_study_full.pkl \
  --output-dir /results/ood_mlp \
  --device cuda
```

---

## Configuration

Edit `utils/config.py` to point to your local dataset roots:

```python
DATASET_ROOT = Path("/your/data/root")   # Parent of DRIAMS and MARISMa directories
MARISMA_ROOT = DATASET_ROOT / "MARISMa"
USER_ROOT    = Path("/your/output/root")
PROJECT_ROOT = USER_ROOT / "MALDI_for_AMR_prediction"
PICKLE_OUTPUT_DIR = PROJECT_ROOT / "data"
```

---

## Metrics

All models are evaluated with:

| Metric | Definition |
|--------|-----------|
| **Micro AUC** | ROC-AUC computed over all observed (sample, antibiotic) pairs jointly |
| **Macro AUC** | Mean per-antibiotic ROC-AUC, averaged over antibiotics with sufficient observations |

Minimum thresholds for antibiotic inclusion: ≥ 50 training observations and ≥ 5 validation observations with both classes present.

---

## Project Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥ 2.0 | Deep learning framework |
| PyTorch Lightning | ≥ 2.0 | Training orchestration |
| scikit-learn | ≥ 1.3 | K-Fold, AUC metrics, Random Forest baseline |
| Optuna | ≥ 3.0 | Bayesian hyperparameter optimisation |
| pandas / numpy | — | Data manipulation |
| scipy | — | Baseline correction, binning, Savitzky–Golay filter |
| tqdm | — | Progress bars |
| matplotlib / seaborn | — | Visualisation |

---

## Repository Conventions

- All spectra are represented as fixed-length 1-D vectors of **6 000 intensity bins** covering m/z 2 000–20 000 Da (3 Da bins).
- AMR labels are stored as `float32` matrices with `0.0` (susceptible), `1.0` (resistant) and `NaN` (not tested).
- Species names follow the convention `Genus_species` (e.g., `Klebsiella_pneumoniae`).
- Random seed 42 is used throughout for reproducibility.

---

## Citation

> Garcia-Arroyo E., et al. (2026). *A Progressive Benchmark for MALDI-TOF MS-Based Antimicrobial Resistance Prediction via Structured Deep Learning*. (Manuscript in preparation)

---

## License

This project is released for academic use. Please contact the authors before using it in derivative commercial work.

---

## Contact

Elena Garcia Arroyo — `elenagarciarroyo16@gmail.com` — Universidad Carlos III de Madrid
