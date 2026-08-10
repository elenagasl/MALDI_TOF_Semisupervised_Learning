# GRASP: Antimicrobial Resistance Prediction from MALDI-TOF Spectra

This repository contains the Python code used for antimicrobial susceptibility
prediction experiments with MALDI-TOF mass spectra. The project focuses on
GRASP-style global recommender models, interaction MLP baselines, OOD
evaluation, and the final clinical ranking assessment.

## Repository Layout

```text
.
├── grasp/
│   ├── scripts/                  # GRASP training/evaluation entry point
│   ├── src/grasp_global/         # GRASP model, data, metrics, training code
│   └── interaction_mlp/          # Species-aware interaction MLP experiments
├── clinical_assessment/          # Final clinical NDCG and recommendation analysis
├── grasp_ndcg_training/          # Soft-NDCG training script for multi-head GRASP
├── ood_evaluation/               # OOD zero-shot and fine-tuning evaluation code
└── supervised_vs_semisupervised_learning/
    └──                           # Supervised and semi-supervised baselines
```

## Core Problem

The main GRASP formulation models antimicrobial resistance as an interaction:

```text
MALDI-TOF spectrum + bacterial species + antibiotic -> probability of resistance
```

This allows the model to use sparse AST panels, share information across
species and antibiotics, and evaluate antibiotics that are not observed for
every isolate.

## Main Components

- `grasp/`: global GRASP architectures, including species-conditioned,
  multi-head, and hypernetwork variants.
- `grasp/interaction_mlp/`: species-aware global MLP baseline that predicts the
  antibiotic panel jointly from spectra and species embeddings.
- `ood_evaluation/`: external validation on an OOD cohort, including zero-shot
  transfer and target-site fine-tuning.
- `clinical_assessment/`: final clinical ranking experiment for NDCG, resistant
  versus susceptible recommendation percentages, and higher-generation
  antibiotic overuse.
- `supervised_vs_semisupervised_learning/`: earlier baseline experiments

## Data Expectations

The scripts expect pickle payloads containing MALDI spectra, species labels,
antibiotic names, and sparse AST labels. Paths are passed through command-line
arguments such as `--pickle` or `--results-root`; no raw data is committed to
the repository.

## Git Policy

The `.gitignore` is configured so a normal `git add .` stages only:

- Python source files: `*.py`
- README documentation: `README.md`
- The root `.gitignore`

Everything else is treated as local or generated material.
