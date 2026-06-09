# ============================================================
# BIO-PRIOR GATED RECOMMENDERS FOR MALDI-TOF AMR PREDICTION
#
# Runs two enriched-but-controlled experiments:
#
#   1) bio_prior_gated_species_head
#      - Shared MALDI backbone
#      - Species-specific MALDI head
#      - Biology-conditioned residual correction
#      - Training prevalence as explicit logit prior
#
#   2) bio_prior_gated_implicit_hypernetwork
#      - Global MALDI encoder
#      - Biology/species/antibiotic-conditioned hypernetwork
#      - Training prevalence as explicit logit prior
#
# Core idea:
#
#   z_maldi = MALDI encoder(x)
#
#   context = [species_emb, antibiotic_emb, ATC embeddings, mechanism embedding, beta_lactamase]
#
#   delta = DeltaNet(context)
#   gate  = sigmoid(GateNet(context))
#
#   z_adapted = z_maldi + gate * delta
#
#   final_logit = recommender(z_adapted, species_emb, antibiotic_emb, bio_emb)
#                 + gamma[species, antibiotic] * prevalence_logit_prior
#
# The prevalence prior is computed ONLY on train inside each fold.
# This avoids leakage.
# ============================================================

import os
import gc
import json
import math
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from pytorch_lightning.callbacks import EarlyStopping


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl"

OUTPUT_FOLD_CSV = "bio_prior_gated_fold_results.csv"
OUTPUT_SPECIES_CSV = "bio_prior_gated_per_species.csv"
OUTPUT_ANTIBIOTIC_CSV = "bio_prior_gated_per_antibiotic.csv"
OUTPUT_SUMMARY_CSV = "bio_prior_gated_summary.csv"
OUTPUT_MAPPING_JSON = "bio_prior_gated_mappings.json"
OUTPUT_METADATA_USED_CSV = "bio_prior_gated_antibiotic_metadata_used.csv"
OUTPUT_GAMMA_PAIR_CSV = "bio_prior_gated_gamma_by_species_antibiotic.csv"

N_SPLITS = 5
RANDOM_STATE = 42

# Antibiotic filtering per fold
MIN_TRAIN_OBS_PER_ANTIBIOTIC = 50
MIN_VAL_OBS_PER_ANTIBIOTIC = 5

# Training
MAX_EPOCHS = 300
PATIENCE = 15
LR = 1e-3

# CPU/GPU optimized settings
USE_CUDA = torch.cuda.is_available()
ACCELERATOR = "gpu" if USE_CUDA else "cpu"
DEVICES = 1
PRECISION = "16-mixed" if USE_CUDA else "32-true"

if USE_CUDA:
    BATCH_SIZE = 1024
    VAL_BATCH_SIZE = 2048
    PRED_BATCH_SIZE = 512
    NUM_WORKERS = 4
else:
    BATCH_SIZE = 256
    VAL_BATCH_SIZE = 512
    PRED_BATCH_SIZE = 128
    NUM_WORKERS = 0

CPU_THREADS = min(8, os.cpu_count() or 1)
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)

# Model dimensions
MALDI_EMB_DIM = 32
DRUG_EMB_DIM = 16
SPECIES_EMB_DIM = 16

ATC2_EMB_DIM = 4
ATC3_EMB_DIM = 6
ATC4_EMB_DIM = 8
MECHANISM_EMB_DIM = 6

CONTEXT_HIDDEN_DIM = 64
RECOMMENDER_HIDDEN_DIMS = [128, 64]

# Regularization for correction vector
LAMBDA_DELTA = 1e-4

# Prevalence smoothing
PREVALENCE_SMOOTHING_ALPHA = 30.0
PREVALENCE_CLIP_MIN = 0.02
PREVALENCE_CLIP_MAX = 0.98

# Initial strength of epidemiological prior.
# gamma[species, antibiotic] = sigmoid(gamma_raw[species, antibiotic]).
# Every pair is initialized around 0.10 and then learned independently.
INITIAL_GAMMA = 0.10

MODEL_MODES = [
    "bio_prior_gated_species_head",
    "bio_prior_gated_implicit_hypernetwork",
]

pl.seed_everything(RANDOM_STATE, workers=True)

if USE_CUDA:
    print("CUDA available:", torch.cuda.get_device_name(0), flush=True)
    torch.set_float32_matmul_precision("medium")
else:
    print("WARNING: CUDA is not available. Running on CPU.", flush=True)

print("Accelerator:", ACCELERATOR, flush=True)
print("Precision:", PRECISION, flush=True)
print("Batch size:", BATCH_SIZE, flush=True)
print("Val batch size:", VAL_BATCH_SIZE, flush=True)
print("Num workers:", NUM_WORKERS, flush=True)


# ============================================================
# ANTIBIOTIC METADATA DICTIONARY HERE
# ============================================================
antibiotic_metadata = {
    "5-Fluorocytosine": {
        "ATC2": "J02",
        "ATC3": "J02A",
        "ATC4": "J02AX",
        "mechanism": "antifungal",
        "beta_lactamase_inhibitor": 0,
    },
    "Amikacin": {
        "ATC2": "J01",
        "ATC3": "J01G",
        "ATC4": "J01GB",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Amoxicillin": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Amoxicillin-Clavulanic acid": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CR",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 1,
    },
    "Amoxicillin-Clavulanic acid_uncomplicated_HWI": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CR",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 1,
    },
    "Amphotericin B": {
        "ATC2": "J02",
        "ATC3": "J02A",
        "ATC4": "J02AA",
        "mechanism": "membrane",
        "beta_lactamase_inhibitor": 0,
    },
    "Ampicillin": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Ampicillin-Sulbactam": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CR",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 1,
    },
    "Anidulafungin": {
        "ATC2": "J02",
        "ATC3": "J02A",
        "ATC4": "J02AX",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Azithromycin": {
        "ATC2": "J01",
        "ATC3": "J01F",
        "ATC4": "J01FA",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Aztreonam": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DF",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Bacitracin": {
        "ATC2": "J01",
        "ATC3": "J01X",
        "ATC4": "J01XX",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Benzylpenicillin": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CE",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Benzylpenicillin_others": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CE",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Benzylpenicillin_with_meningitis": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CE",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Benzylpenicillin_with_pneumonia": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CE",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Caspofungin": {
        "ATC2": "J02",
        "ATC3": "J02A",
        "ATC4": "J02AX",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Cefalotin-Cefazolin": {
        "ATC2": None,
        "ATC3": None,
        "ATC4": None,
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Cefazolin": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DB",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Cefepime": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DE",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Cefixime": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DD",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Cefotaxime": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DD",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Cefoxitin": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DC",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Cefoxitin_screen": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DC",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Cefpodoxime": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DD",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Ceftarolin": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DI",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Ceftazidime": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DD",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Ceftazidime-Avibactam": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DD",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 1,
    },
    "Ceftobiprole": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DI",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Ceftolozane-Tazobactam": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DI",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 1,
    },
    "Ceftriaxone": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DD",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Cefuroxime": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DC",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Cefuroxime.1": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DC",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Chloramphenicol": {
        "ATC2": "J01",
        "ATC3": "J01B",
        "ATC4": "J01BA",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Ciprofloxacin": {
        "ATC2": "J01",
        "ATC3": "J01M",
        "ATC4": "J01MA",
        "mechanism": "DNA_RNA",
        "beta_lactamase_inhibitor": 0,
    },
    "Clarithromycin": {
        "ATC2": "J01",
        "ATC3": "J01F",
        "ATC4": "J01FA",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Clindamycin": {
        "ATC2": "J01",
        "ATC3": "J01F",
        "ATC4": "J01FF",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Clindamycin_induced": {
        "ATC2": "J01",
        "ATC3": "J01F",
        "ATC4": "J01FF",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Colistin": {
        "ATC2": "J01",
        "ATC3": "J01X",
        "ATC4": "J01XB",
        "mechanism": "membrane",
        "beta_lactamase_inhibitor": 0,
    },
    "Cotrimoxazol": {
        "ATC2": "J01",
        "ATC3": "J01E",
        "ATC4": "J01EE",
        "mechanism": "metabolic",
        "beta_lactamase_inhibitor": 0,
    },
    "Cotrimoxazole": {
        "ATC2": "J01",
        "ATC3": "J01E",
        "ATC4": "J01EE",
        "mechanism": "metabolic",
        "beta_lactamase_inhibitor": 0,
    },
    "Daptomycin": {
        "ATC2": "J01",
        "ATC3": "J01X",
        "ATC4": "J01XX",
        "mechanism": "membrane",
        "beta_lactamase_inhibitor": 0,
    },
    "Doxycycline": {
        "ATC2": "J01",
        "ATC3": "J01A",
        "ATC4": "J01AA",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Ertapenem": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DH",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Erythromycin": {
        "ATC2": "J01",
        "ATC3": "J01F",
        "ATC4": "J01FA",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Ethambutol_5mg-l": {
        "ATC2": "J04",
        "ATC3": "J04A",
        "ATC4": "J04AK",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Fluconazole": {
        "ATC2": "J02",
        "ATC3": "J02A",
        "ATC4": "J02AC",
        "mechanism": "membrane",
        "beta_lactamase_inhibitor": 0,
    },
    "Fosfomycin": {
        "ATC2": "J01",
        "ATC3": "J01X",
        "ATC4": "J01XX",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Fusidic acid": {
        "ATC2": "J01",
        "ATC3": "J01X",
        "ATC4": "J01XC",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Gentamicin": {
        "ATC2": "J01",
        "ATC3": "J01G",
        "ATC4": "J01GB",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Gentamicin_high_level": {
        "ATC2": "J01",
        "ATC3": "J01G",
        "ATC4": "J01GB",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Imipenem": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DH",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Isavuconazole": {
        "ATC2": "J02",
        "ATC3": "J02A",
        "ATC4": "J02AC",
        "mechanism": "membrane",
        "beta_lactamase_inhibitor": 0,
    },
    "Isoniazid_.1mg-l": {
        "ATC2": "J04",
        "ATC3": "J04A",
        "ATC4": "J04AC",
        "mechanism": "metabolic",
        "beta_lactamase_inhibitor": 0,
    },
    "Isoniazid_.4mg-l": {
        "ATC2": "J04",
        "ATC3": "J04A",
        "ATC4": "J04AC",
        "mechanism": "metabolic",
        "beta_lactamase_inhibitor": 0,
    },
    "Itraconazole": {
        "ATC2": "J02",
        "ATC3": "J02A",
        "ATC4": "J02AC",
        "mechanism": "membrane",
        "beta_lactamase_inhibitor": 0,
    },
    "Levofloxacin": {
        "ATC2": "J01",
        "ATC3": "J01M",
        "ATC4": "J01MA",
        "mechanism": "DNA_RNA",
        "beta_lactamase_inhibitor": 0,
    },
    "Linezolid": {
        "ATC2": "J01",
        "ATC3": "J01X",
        "ATC4": "J01XX",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "MRSA": {
        "ATC2": None,
        "ATC3": None,
        "ATC4": None,
        "mechanism": None,
        "beta_lactamase_inhibitor": None,
    },
    "Meropenem": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DH",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Meropenem-Vaborbactam": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DH",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 1,
    },
    "Meropenem_with_meningitis": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DH",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Meropenem_with_pneumonia": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DH",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Meropenem_without_meningitis": {
        "ATC2": "J01",
        "ATC3": "J01D",
        "ATC4": "J01DH",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Metronidazole": {
        "ATC2": "J01",
        "ATC3": "J01X",
        "ATC4": "J01XD",
        "mechanism": "DNA_RNA",
        "beta_lactamase_inhibitor": 0,
    },
    "Micafungin": {
        "ATC2": "J02",
        "ATC3": "J02A",
        "ATC4": "J02AX",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Minocycline": {
        "ATC2": "J01",
        "ATC3": "J01A",
        "ATC4": "J01AA",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Moxifloxacin": {
        "ATC2": "J01",
        "ATC3": "J01M",
        "ATC4": "J01MA",
        "mechanism": "DNA_RNA",
        "beta_lactamase_inhibitor": 0,
    },
    "Mupirocin": {
        "ATC2": "D06",
        "ATC3": "D06A",
        "ATC4": "D06AX",
        "mechanism": "antifungal",
        "beta_lactamase_inhibitor": 0,
    },
    "Nitrofurantoin": {
        "ATC2": "J01",
        "ATC3": "J01X",
        "ATC4": "J01XE",
        "mechanism": "DNA_RNA",
        "beta_lactamase_inhibitor": 0,
    },
    "Norfloxacin": {
        "ATC2": "J01",
        "ATC3": "J01M",
        "ATC4": "J01MA",
        "mechanism": "DNA_RNA",
        "beta_lactamase_inhibitor": 0,
    },
    "Novobiocin": {
        "ATC2": None,
        "ATC3": None,
        "ATC4": None,
        "mechanism": "DNA_gyrase",
        "beta_lactamase_inhibitor": 0,
    },
    "Ofloxacin": {
        "ATC2": "J01",
        "ATC3": "J01M",
        "ATC4": "J01MA",
        "mechanism": "DNA_RNA",
        "beta_lactamase_inhibitor": 0,
    },
    "Oxacillin": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CF",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Pefloxacin": {
        "ATC2": "J01",
        "ATC3": "J01M",
        "ATC4": "J01MA",
        "mechanism": "DNA_RNA",
        "beta_lactamase_inhibitor": 0,
    },
    "Penicillin": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Penicillin_with_endokarditis": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Penicillin_with_meningitis": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Penicillin_with_other_infections": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Penicillin_with_pneumonia": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Penicillin_without_endokarditis": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Penicillin_without_meningitis": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Piperacillin": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Piperacillin-Tazobactam": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CR",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 1,
    },
    "Polymyxin B": {
        "ATC2": "J01",
        "ATC3": "J01X",
        "ATC4": "J01XB",
        "mechanism": "membrane",
        "beta_lactamase_inhibitor": 0,
    },
    "Posaconazole": {
        "ATC2": "J02",
        "ATC3": "J02A",
        "ATC4": "J02AC",
        "mechanism": "membrane",
        "beta_lactamase_inhibitor": 0,
    },
    "Pristinamycin": {
        "ATC2": "J01",
        "ATC3": "J01F",
        "ATC4": "J01FG",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Pyrazinamide": {
        "ATC2": "J04",
        "ATC3": "J04A",
        "ATC4": "J04AK",
        "mechanism": "metabolic",
        "beta_lactamase_inhibitor": 0,
    },
    "Rifampicin": {
        "ATC2": "J04",
        "ATC3": "J04A",
        "ATC4": "J04AB",
        "mechanism": "DNA_RNA",
        "beta_lactamase_inhibitor": 0,
    },
    "Rifampicin_1mg-l": {
        "ATC2": "J04",
        "ATC3": "J04A",
        "ATC4": "J04AB",
        "mechanism": "DNA_RNA",
        "beta_lactamase_inhibitor": 0,
    },
    "Sparfloxacin": {
        "ATC2": "J01",
        "ATC3": "J01M",
        "ATC4": "J01MA",
        "mechanism": "DNA_RNA",
        "beta_lactamase_inhibitor": 0,
    },
    "Strepomycin_high_level": {
        "ATC2": "J01",
        "ATC3": "J01G",
        "ATC4": "J01GA",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Streptomycin": {
        "ATC2": "J01",
        "ATC3": "J01G",
        "ATC4": "J01GA",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Teicoplanin": {
        "ATC2": "J01",
        "ATC3": "J01X",
        "ATC4": "J01XA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Teicoplanin_GRD": {
        "ATC2": "J01",
        "ATC3": "J01X",
        "ATC4": "J01XA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Telithromycin": {
        "ATC2": "J01",
        "ATC3": "J01F",
        "ATC4": "J01FA",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Tetracycline": {
        "ATC2": "J01",
        "ATC3": "J01A",
        "ATC4": "J01AA",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Ticarcillin": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Ticarcillin-Clavulan acid": {
        "ATC2": "J01",
        "ATC3": "J01C",
        "ATC4": "J01CR",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 1,
    },
    "Tigecycline": {
        "ATC2": "J01",
        "ATC3": "J01A",
        "ATC4": "J01AA",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Tobramycin": {
        "ATC2": "J01",
        "ATC3": "J01G",
        "ATC4": "J01GB",
        "mechanism": "protein_synthesis",
        "beta_lactamase_inhibitor": 0,
    },
    "Vancomycin": {
        "ATC2": "J01",
        "ATC3": "J01X",
        "ATC4": "J01XA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Vancomycin_GRD": {
        "ATC2": "J01",
        "ATC3": "J01X",
        "ATC4": "J01XA",
        "mechanism": "cell_wall",
        "beta_lactamase_inhibitor": 0,
    },
    "Voriconazole": {
        "ATC2": "J02",
        "ATC3": "J02A",
        "ATC4": "J02AC",
        "mechanism": "membrane",
        "beta_lactamase_inhibitor": 0,
    },
}


# Keep uppercase alias used by the metadata-building utility.
ANTIBIOTIC_METADATA = antibiotic_metadata


# ============================================================
# DATASET
# ============================================================

class BioPriorRecDataset(Dataset):
    """
    Each item:
        (sample, species, antibiotic, biological metadata, prevalence prior)
        -> resistance label

    Returns:
        maldi
        species_id
        drug_id
        atc2_id
        atc3_id
        atc4_id
        mechanism_id
        beta_lactamase_inhibitor
        prior_logit
        label
    """

    def __init__(
        self,
        X,
        species_ids,
        amr,
        antibiotic_feature_table,
        prior_logit_matrix
    ):
        self.X = np.asarray(X)
        self.species_ids = np.asarray(species_ids)
        self.amr = np.asarray(amr)
        self.antibiotic_feature_table = antibiotic_feature_table
        self.prior_logit_matrix = np.asarray(prior_logit_matrix, dtype=np.float32)

        valid = (~np.isnan(self.amr)) & ((self.amr == 0) | (self.amr == 1))

        sample_idx, drug_idx = np.where(valid)

        labels = self.amr[sample_idx, drug_idx].astype(np.float32)

        self.sample_idx = sample_idx.astype(np.int64)
        self.drug_idx = drug_idx.astype(np.int64)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample_idx = self.sample_idx[idx]
        drug_id = self.drug_idx[idx]
        species_id = self.species_ids[sample_idx]
        label = self.labels[idx]

        row = self.antibiotic_feature_table[drug_id]

        prior_logit = self.prior_logit_matrix[species_id, drug_id]

        return (
            torch.tensor(self.X[sample_idx]).float(),
            torch.tensor(species_id).long(),
            torch.tensor(drug_id).long(),
            torch.tensor(row["atc2_id"]).long(),
            torch.tensor(row["atc3_id"]).long(),
            torch.tensor(row["atc4_id"]).long(),
            torch.tensor(row["mechanism_id"]).long(),
            torch.tensor(row["beta_lactamase_inhibitor"]).float(),
            torch.tensor(prior_logit).float(),
            torch.tensor(label).float()
        )


# ============================================================
# MODEL COMPONENTS
# ============================================================

class GlobalMALDIEncoder(nn.Module):
    """
    Global MALDI encoder:
        MALDI spectrum -> MALDI embedding
    """

    def __init__(self, input_dim, maldi_emb_dim=32):
        super().__init__()

        self.output_dim = maldi_emb_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(64, maldi_emb_dim),
            nn.GELU()
        )

    def forward(self, maldi):
        return self.encoder(maldi.float())


class SpeciesHeadMALDIEncoder(nn.Module):
    """
    Shared backbone + one species-specific MALDI head per species.

    This corresponds to the previous species_only model.
    """

    def __init__(self, input_dim, num_species, maldi_emb_dim=32):
        super().__init__()

        self.output_dim = maldi_emb_dim
        self.num_species = num_species

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.species_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Dropout(0.2),

                    nn.Linear(64, maldi_emb_dim),
                    nn.GELU()
                )
                for _ in range(num_species)
            ]
        )

    def forward(self, maldi, species_id):
        z = self.backbone(maldi.float())

        h_species = torch.zeros(
            size=(z.shape[0], self.output_dim),
            dtype=z.dtype,
            device=z.device
        )

        unique_species = torch.unique(species_id.long())

        for sp in unique_species:
            sp_int = int(sp.item())
            mask = species_id.long() == sp
            h_species[mask] = self.species_heads[sp_int](z[mask])

        return h_species


class BioEmbeddingBlock(nn.Module):
    """
    Embeds antibiotic biological metadata:
        ATC2, ATC3, ATC4, mechanism, beta_lactamase_inhibitor
    """

    def __init__(
        self,
        num_atc2,
        num_atc3,
        num_atc4,
        num_mechanisms,
        atc2_dim=4,
        atc3_dim=6,
        atc4_dim=8,
        mechanism_dim=6
    ):
        super().__init__()

        self.atc2_embedding = nn.Embedding(num_atc2, atc2_dim)
        self.atc3_embedding = nn.Embedding(num_atc3, atc3_dim)
        self.atc4_embedding = nn.Embedding(num_atc4, atc4_dim)
        self.mechanism_embedding = nn.Embedding(num_mechanisms, mechanism_dim)

        self.output_dim = atc2_dim + atc3_dim + atc4_dim + mechanism_dim + 1

    def forward(
        self,
        atc2_id,
        atc3_id,
        atc4_id,
        mechanism_id,
        beta_lactamase_inhibitor
    ):
        atc2_emb = self.atc2_embedding(atc2_id.long())
        atc3_emb = self.atc3_embedding(atc3_id.long())
        atc4_emb = self.atc4_embedding(atc4_id.long())
        mech_emb = self.mechanism_embedding(mechanism_id.long())

        beta = beta_lactamase_inhibitor.view(-1, 1).float()

        bio_emb = torch.cat(
            [atc2_emb, atc3_emb, atc4_emb, mech_emb, beta],
            dim=-1
        )

        return bio_emb


class ContextCorrectionBlock(nn.Module):
    """
    Biology/species/antibiotic-conditioned correction.

    context -> delta
    context -> gate

    z_adapted = z_base + gate * delta
    """

    def __init__(self, context_dim, maldi_emb_dim=32, hidden_dim=64):
        super().__init__()

        self.delta_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, maldi_emb_dim)
        )

        self.gate_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, maldi_emb_dim)
        )

    def forward(self, context):
        delta = self.delta_net(context)
        gate = torch.sigmoid(self.gate_net(context))
        correction = gate * delta

        return correction, delta, gate


class BioPriorGatedRecommender(pl.LightningModule):
    """
    Unified model for two modes:

    1) bio_prior_gated_species_head:
        z_base = species-specific MALDI head(backbone(MALDI))

    2) bio_prior_gated_implicit_hypernetwork:
        z_base = global MALDI encoder(MALDI)

    In both:
        context = [species_emb, drug_emb, bio_emb]
        correction = gate(context) * delta(context)
        z_adapted = z_base + correction

        maldi_logit = MLP([z_adapted, species_emb, drug_emb, bio_emb])
        gamma_pair = gamma[species_id, drug_id]
        final_logit = maldi_logit + gamma_pair * prior_logit

    prior_logit is computed only from training fold.
    """

    def __init__(
        self,
        num_feat,
        num_items,
        num_species,
        num_atc2,
        num_atc3,
        num_atc4,
        num_mechanisms,
        model_mode,
        maldi_emb_dim=32,
        drug_emb_dim=16,
        species_emb_dim=16,
        atc2_dim=4,
        atc3_dim=6,
        atc4_dim=8,
        mechanism_dim=6,
        context_hidden_dim=64,
        recommender_hidden_dims=[128, 64],
        lr=1e-3,
        lambda_delta=1e-4,
        initial_gamma=0.10
    ):
        super().__init__()

        assert model_mode in [
            "bio_prior_gated_species_head",
            "bio_prior_gated_implicit_hypernetwork",
        ]

        self.save_hyperparameters()

        self.model_mode = model_mode
        self.lr = lr
        self.lambda_delta = lambda_delta

        if model_mode == "bio_prior_gated_species_head":
            self.maldi_encoder = SpeciesHeadMALDIEncoder(
                input_dim=num_feat,
                num_species=num_species,
                maldi_emb_dim=maldi_emb_dim
            )
        else:
            self.maldi_encoder = GlobalMALDIEncoder(
                input_dim=num_feat,
                maldi_emb_dim=maldi_emb_dim
            )

        self.species_embedding = nn.Embedding(num_species, species_emb_dim)
        self.drug_embedding = nn.Embedding(num_items, drug_emb_dim)

        self.bio_embedding = BioEmbeddingBlock(
            num_atc2=num_atc2,
            num_atc3=num_atc3,
            num_atc4=num_atc4,
            num_mechanisms=num_mechanisms,
            atc2_dim=atc2_dim,
            atc3_dim=atc3_dim,
            atc4_dim=atc4_dim,
            mechanism_dim=mechanism_dim
        )

        bio_dim = self.bio_embedding.output_dim

        context_dim = species_emb_dim + drug_emb_dim + bio_dim

        self.context_correction = ContextCorrectionBlock(
            context_dim=context_dim,
            maldi_emb_dim=maldi_emb_dim,
            hidden_dim=context_hidden_dim
        )

        fusion_input_dim = maldi_emb_dim + species_emb_dim + drug_emb_dim + bio_dim

        sizes = [fusion_input_dim] + list(recommender_hidden_dims) + [1]

        layers = []

        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        self.recommender_mlp = nn.Sequential(*layers)

        # Pair-specific epidemiological prior strength.
        # gamma_matrix[species, antibiotic] = sigmoid(gamma_raw[species, antibiotic])
        # Shape: [num_species, num_items].
        initial_gamma = float(np.clip(initial_gamma, 1e-4, 1 - 1e-4))
        gamma_raw_init = math.log(initial_gamma / (1.0 - initial_gamma))

        self.gamma_raw = nn.Parameter(
            torch.full(
                size=(num_species, num_items),
                fill_value=gamma_raw_init,
                dtype=torch.float32
            )
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.fusion_input_dim = fusion_input_dim
        self.context_dim = context_dim
        self.bio_dim = bio_dim

    def _encode_maldi(self, maldi, species_id):
        if self.model_mode == "bio_prior_gated_species_head":
            return self.maldi_encoder(maldi, species_id)
        return self.maldi_encoder(maldi)

    def forward(
        self,
        maldi,
        species_id,
        drug_id,
        atc2_id,
        atc3_id,
        atc4_id,
        mechanism_id,
        beta_lactamase_inhibitor,
        prior_logit
    ):
        z_base = self._encode_maldi(maldi, species_id)

        species_emb = self.species_embedding(species_id.long())
        drug_emb = self.drug_embedding(drug_id.long())

        bio_emb = self.bio_embedding(
            atc2_id=atc2_id,
            atc3_id=atc3_id,
            atc4_id=atc4_id,
            mechanism_id=mechanism_id,
            beta_lactamase_inhibitor=beta_lactamase_inhibitor
        )

        context = torch.cat(
            [species_emb, drug_emb, bio_emb],
            dim=-1
        )

        correction, delta, gate = self.context_correction(context)

        z_adapted = z_base + correction

        x = torch.cat(
            [z_adapted, species_emb, drug_emb, bio_emb],
            dim=-1
        )

        maldi_logit = self.recommender_mlp(x)

        gamma_matrix = torch.sigmoid(self.gamma_raw)

        gamma_pair = gamma_matrix[
            species_id.long(),
            drug_id.long()
        ].view(-1, 1)

        final_logit = maldi_logit + gamma_pair * prior_logit.view(-1, 1).float()

        return final_logit, delta, gate, gamma_pair

    def training_step(self, batch, batch_idx):
        (
            maldi,
            species_id,
            drug_id,
            atc2_id,
            atc3_id,
            atc4_id,
            mechanism_id,
            beta,
            prior_logit,
            labels
        ) = batch

        logits, delta, gate, gamma = self.forward(
            maldi=maldi,
            species_id=species_id,
            drug_id=drug_id,
            atc2_id=atc2_id,
            atc3_id=atc3_id,
            atc4_id=atc4_id,
            mechanism_id=mechanism_id,
            beta_lactamase_inhibitor=beta,
            prior_logit=prior_logit
        )

        labels = labels.view(-1, 1).float()

        bce = self.loss_fn(logits, labels)

        delta_penalty = torch.mean(delta.pow(2))
        loss = bce + self.lambda_delta * delta_penalty

        self.log("loss_tr", loss, prog_bar=True)
        self.log("bce_tr", bce, prog_bar=False)
        self.log("delta_penalty", delta_penalty, prog_bar=False)
        self.log("gamma_mean", gamma.detach().mean(), prog_bar=True)
        self.log("gamma_min", gamma.detach().min(), prog_bar=False)
        self.log("gamma_max", gamma.detach().max(), prog_bar=False)
        self.log("gate_mean", gate.detach().mean(), prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        (
            maldi,
            species_id,
            drug_id,
            atc2_id,
            atc3_id,
            atc4_id,
            mechanism_id,
            beta,
            prior_logit,
            labels
        ) = batch

        logits, delta, gate, gamma = self.forward(
            maldi=maldi,
            species_id=species_id,
            drug_id=drug_id,
            atc2_id=atc2_id,
            atc3_id=atc3_id,
            atc4_id=atc4_id,
            mechanism_id=mechanism_id,
            beta_lactamase_inhibitor=beta,
            prior_logit=prior_logit
        )

        labels = labels.view(-1, 1).float()

        bce = self.loss_fn(logits, labels)
        delta_penalty = torch.mean(delta.pow(2))
        loss = bce + self.lambda_delta * delta_penalty

        self.log("loss_val", loss, prog_bar=True)
        self.log("bce_val", bce, prog_bar=False)
        self.log("gamma_val_mean", gamma.detach().mean(), prog_bar=False)
        self.log("gamma_val_min", gamma.detach().min(), prog_bar=False)
        self.log("gamma_val_max", gamma.detach().max(), prog_bar=False)
        self.log("gate_val_mean", gate.detach().mean(), prog_bar=False)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ============================================================
# UTILS
# ============================================================

def clean_amr(amr):
    """
    Keeps only:
        0 = susceptible
        1 = resistant
        NaN = missing

    Everything else becomes NaN.
    """

    amr = amr.copy()
    valid = (amr == 0) | (amr == 1) | np.isnan(amr)
    amr[~valid] = np.nan
    return amr


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=USE_CUDA,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False
    )


def count_observed_with_two_classes(col):
    obs = col[~np.isnan(col)]
    n_obs = len(obs)
    has_two_classes = n_obs > 0 and len(np.unique(obs)) > 1
    return n_obs, has_two_classes


def select_valid_antibiotics(amr_train, amr_val):
    valid_cols = []
    n_antibiotics = amr_train.shape[1]

    for j in range(n_antibiotics):
        train_col = amr_train[:, j]
        val_col = amr_val[:, j]

        n_train, train_two_classes = count_observed_with_two_classes(train_col)
        n_val, val_two_classes = count_observed_with_two_classes(val_col)

        if n_train < MIN_TRAIN_OBS_PER_ANTIBIOTIC:
            continue

        if not train_two_classes:
            continue

        if n_val < MIN_VAL_OBS_PER_ANTIBIOTIC:
            continue

        if not val_two_classes:
            continue

        valid_cols.append(j)

    return valid_cols


def safe_auc(y, p):
    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return roc_auc_score(y, p)


def logit_np(p):
    p = np.clip(p, PREVALENCE_CLIP_MIN, PREVALENCE_CLIP_MAX)
    return np.log(p / (1.0 - p))


def get_metadata_value(meta, key, default_value):
    if meta is None:
        return default_value

    value = meta.get(key, default_value)

    if value is None:
        return default_value

    return value


def build_antibiotic_metadata_tables(selected_antibiotics):
    """
    Builds categorical IDs for ATC and mechanism among selected antibiotics.

    Unknown values are explicitly represented.
    """

    rows = []

    atc2_values = ["UNK_ATC2"]
    atc3_values = ["UNK_ATC3"]
    atc4_values = ["UNK_ATC4"]
    mechanism_values = ["unknown"]

    for antibiotic_name in selected_antibiotics:
        antibiotic_name = str(antibiotic_name)
        meta = ANTIBIOTIC_METADATA.get(antibiotic_name, None)

        atc2 = str(get_metadata_value(meta, "ATC2", "UNK_ATC2"))
        atc3 = str(get_metadata_value(meta, "ATC3", "UNK_ATC3"))
        atc4 = str(get_metadata_value(meta, "ATC4", "UNK_ATC4"))
        mechanism = str(get_metadata_value(meta, "mechanism", "unknown"))
        beta = get_metadata_value(meta, "beta_lactamase_inhibitor", 0)

        if beta is None:
            beta = 0

        beta = float(beta)

        rows.append(
            {
                "antibiotic": antibiotic_name,
                "ATC2": atc2,
                "ATC3": atc3,
                "ATC4": atc4,
                "mechanism": mechanism,
                "beta_lactamase_inhibitor": beta,
                "metadata_found": int(meta is not None),
            }
        )

        if atc2 not in atc2_values:
            atc2_values.append(atc2)
        if atc3 not in atc3_values:
            atc3_values.append(atc3)
        if atc4 not in atc4_values:
            atc4_values.append(atc4)
        if mechanism not in mechanism_values:
            mechanism_values.append(mechanism)

    atc2_to_id = {v: i for i, v in enumerate(atc2_values)}
    atc3_to_id = {v: i for i, v in enumerate(atc3_values)}
    atc4_to_id = {v: i for i, v in enumerate(atc4_values)}
    mechanism_to_id = {v: i for i, v in enumerate(mechanism_values)}

    feature_table = []

    for row in rows:
        feature_table.append(
            {
                "antibiotic": row["antibiotic"],
                "atc2_id": atc2_to_id[row["ATC2"]],
                "atc3_id": atc3_to_id[row["ATC3"]],
                "atc4_id": atc4_to_id[row["ATC4"]],
                "mechanism_id": mechanism_to_id[row["mechanism"]],
                "beta_lactamase_inhibitor": row["beta_lactamase_inhibitor"],
            }
        )

    mappings = {
        "atc2_to_id": atc2_to_id,
        "atc3_to_id": atc3_to_id,
        "atc4_to_id": atc4_to_id,
        "mechanism_to_id": mechanism_to_id,
    }

    df_meta_used = pd.DataFrame(rows)

    return feature_table, mappings, df_meta_used


def compute_train_prevalence_prior(amr_train, species_train, num_species):
    """
    Computes smoothed species-antibiotic prevalence using train only.

    For each antibiotic a:
        prev_a = resistant_a / observed_a

    For each species s and antibiotic a:
        prev_sa = (res_sa + alpha * prev_a) / (obs_sa + alpha)

    Then:
        prior_logit_sa = logit(clip(prev_sa))

    Returns:
        prior_logit_matrix [num_species, num_items]
    """

    amr_train = np.asarray(amr_train)
    species_train = np.asarray(species_train)

    num_items = amr_train.shape[1]

    observed_global = ~np.isnan(amr_train)
    resistant_global = amr_train == 1

    total_obs = np.sum(observed_global)
    total_res = np.sum(resistant_global)

    if total_obs > 0:
        dataset_prev = total_res / total_obs
    else:
        dataset_prev = 0.5

    antibiotic_prev = np.zeros(num_items, dtype=np.float32)

    for j in range(num_items):
        obs_j = observed_global[:, j]
        n_obs_j = int(np.sum(obs_j))

        if n_obs_j > 0:
            n_res_j = int(np.sum(resistant_global[:, j]))
            antibiotic_prev[j] = n_res_j / n_obs_j
        else:
            antibiotic_prev[j] = dataset_prev

    prior_logit_matrix = np.zeros((num_species, num_items), dtype=np.float32)

    alpha = PREVALENCE_SMOOTHING_ALPHA

    for sp in range(num_species):
        sp_mask = species_train == sp

        for j in range(num_items):
            col = amr_train[sp_mask, j]
            obs = ~np.isnan(col)

            n_obs = int(np.sum(obs))

            if n_obs > 0:
                n_res = int(np.sum(col[obs] == 1))
            else:
                n_res = 0

            prev_smoothed = (
                n_res + alpha * antibiotic_prev[j]
            ) / (
                n_obs + alpha
            )

            prior_logit_matrix[sp, j] = logit_np(prev_smoothed)

    return prior_logit_matrix, antibiotic_prev


def predict_all_pairs(
    model,
    X,
    species_ids,
    num_items,
    antibiotic_feature_table,
    prior_logit_matrix,
    batch_size=128
):
    """
    Predict resistance probabilities for all validation:
        samples x antibiotics

    Output:
        preds_matrix with shape (n_samples, num_items)
    """

    model.eval()
    device = model.device

    X = np.asarray(X)
    species_ids = np.asarray(species_ids)

    n_samples = X.shape[0]

    preds_matrix = np.zeros((n_samples, num_items), dtype=np.float32)

    # Pre-load antibiotic metadata tensors on device
    atc2_all = torch.tensor(
        [row["atc2_id"] for row in antibiotic_feature_table],
        dtype=torch.long,
        device=device
    )
    atc3_all = torch.tensor(
        [row["atc3_id"] for row in antibiotic_feature_table],
        dtype=torch.long,
        device=device
    )
    atc4_all = torch.tensor(
        [row["atc4_id"] for row in antibiotic_feature_table],
        dtype=torch.long,
        device=device
    )
    mech_all = torch.tensor(
        [row["mechanism_id"] for row in antibiotic_feature_table],
        dtype=torch.long,
        device=device
    )
    beta_all = torch.tensor(
        [row["beta_lactamase_inhibitor"] for row in antibiotic_feature_table],
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):

            end = min(start + batch_size, n_samples)

            maldi_batch = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=device
            )

            species_batch = torch.tensor(
                species_ids[start:end],
                dtype=torch.long,
                device=device
            )

            batch_size_real = maldi_batch.shape[0]

            drug_preds = []

            for drug_id in range(num_items):
                drug_batch = torch.full(
                    size=(batch_size_real,),
                    fill_value=drug_id,
                    dtype=torch.long,
                    device=device
                )

                atc2_batch = atc2_all[drug_id].repeat(batch_size_real)
                atc3_batch = atc3_all[drug_id].repeat(batch_size_real)
                atc4_batch = atc4_all[drug_id].repeat(batch_size_real)
                mech_batch = mech_all[drug_id].repeat(batch_size_real)
                beta_batch = beta_all[drug_id].repeat(batch_size_real)

                prior_vals = prior_logit_matrix[
                    species_ids[start:end],
                    drug_id
                ]

                prior_batch = torch.tensor(
                    prior_vals,
                    dtype=torch.float32,
                    device=device
                )

                logits, _, _, _ = model(
                    maldi=maldi_batch,
                    species_id=species_batch,
                    drug_id=drug_batch,
                    atc2_id=atc2_batch,
                    atc3_id=atc3_batch,
                    atc4_id=atc4_batch,
                    mechanism_id=mech_batch,
                    beta_lactamase_inhibitor=beta_batch,
                    prior_logit=prior_batch
                )

                probs = torch.sigmoid(logits).view(-1)
                drug_preds.append(probs)

            drug_preds = torch.stack(drug_preds, dim=1)

            preds_matrix[start:end] = drug_preds.float().cpu().numpy()

    return preds_matrix


def compute_global_metrics(y_true, preds):
    """
    Computes:
        - micro AUC over all observed sample-antibiotic pairs
        - macro AUC averaged across antibiotics
    """

    mask = ~np.isnan(y_true)

    auc_micro = np.nan
    auc_macro_antibiotic = np.nan

    if np.any(mask):
        y_flat = y_true[mask]
        p_flat = preds[mask]
        auc_micro = safe_auc(y_flat, p_flat)

    antibiotic_aucs = []

    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        valid = ~np.isnan(col)

        auc_j = safe_auc(col[valid], preds[valid, j])

        if np.isfinite(auc_j):
            antibiotic_aucs.append(auc_j)

    if len(antibiotic_aucs) > 0:
        auc_macro_antibiotic = float(np.mean(antibiotic_aucs))

    return auc_micro, auc_macro_antibiotic, antibiotic_aucs


def compute_per_species_metrics(
    model_mode,
    fold,
    y_true,
    preds,
    species_ids,
    id_to_species
):
    rows = []

    unique_species_ids = np.unique(species_ids)

    for sp_id in unique_species_ids:
        sp_mask = species_ids == sp_id

        y_sp = y_true[sp_mask]
        p_sp = preds[sp_mask]

        obs_mask = ~np.isnan(y_sp)

        n_samples = int(np.sum(sp_mask))
        n_pairs = int(np.sum(obs_mask))

        auc = np.nan

        if n_pairs > 0:
            y_flat = y_sp[obs_mask]
            p_flat = p_sp[obs_mask]
            auc = safe_auc(y_flat, p_flat)

        rows.append(
            {
                "model_mode": model_mode,
                "fold": fold,
                "species_id": int(sp_id),
                "species": id_to_species[int(sp_id)],
                "n_val_samples": n_samples,
                "n_val_pairs": n_pairs,
                "auc": auc
            }
        )

    return rows


def compute_per_antibiotic_metrics(
    model_mode,
    fold,
    y_true,
    preds,
    selected_antibiotics
):
    rows = []

    for j, antibiotic_name in enumerate(selected_antibiotics):
        col = y_true[:, j]
        valid = ~np.isnan(col)

        n_pairs = int(np.sum(valid))

        auc = safe_auc(col[valid], preds[valid, j])

        rows.append(
            {
                "model_mode": model_mode,
                "fold": fold,
                "antibiotic_id": j,
                "antibiotic": str(antibiotic_name),
                "n_val_pairs": n_pairs,
                "auc": auc
            }
        )

    return rows


def summarize_entity_results(df, group_col):
    if len(df) == 0:
        return pd.DataFrame()

    rows = []

    for (model_mode, entity), sub in df.groupby(["model_mode", group_col]):
        rows.append(
            {
                "model_mode": model_mode,
                group_col: entity,
                "mean_auc": sub["auc"].mean(),
                "std_auc": sub["auc"].std(),
                "mean_n_val_pairs": sub["n_val_pairs"].mean(),
                "n_folds": sub["fold"].nunique(),
            }
        )

    return pd.DataFrame(rows)


def extract_gamma_pair_rows(
    model,
    model_mode,
    fold,
    selected_antibiotics,
    id_to_species
):
    """
    Saves the learned epidemiological-prior weight for every
    species-antibiotic pair.

    gamma = 0 means: ignore the prevalence prior.
    gamma = 1 means: use the full prevalence prior logit.
    Intermediate values mean partial trust in the epidemiological prior.
    """

    gamma_matrix = torch.sigmoid(model.gamma_raw).detach().cpu().numpy()

    rows = []

    for sp_id in range(gamma_matrix.shape[0]):
        for drug_id in range(gamma_matrix.shape[1]):
            rows.append(
                {
                    "model_mode": model_mode,
                    "fold": fold,
                    "species_id": int(sp_id),
                    "species": id_to_species[int(sp_id)],
                    "antibiotic_id": int(drug_id),
                    "antibiotic": str(selected_antibiotics[drug_id]),
                    "gamma": float(gamma_matrix[sp_id, drug_id]),
                }
            )

    return rows


# ============================================================
# LOAD DATA
# ============================================================

print("Loading combined pickle...", flush=True)

with open(DATA_PATH, "rb") as f:
    payload = pickle.load(f)

X_all = np.asarray(payload["data"])
species_labels = np.asarray(payload["label"])
amr_all = clean_amr(np.asarray(payload["amr"]))
antibiotics_all = np.asarray(payload["antibiotics"])

if "hospital" in payload:
    hospital_all = np.asarray(payload["hospital"])
else:
    hospital_all = np.asarray(["unknown"] * X_all.shape[0])

n_samples = X_all.shape[0]
num_feat = X_all.shape[1]
num_antibiotics_total = amr_all.shape[1]

unique_species = np.unique(species_labels)

species_to_id = {
    sp: i for i, sp in enumerate(unique_species)
}

id_to_species = {
    i: sp for sp, i in species_to_id.items()
}

species_ids_all = np.array(
    [species_to_id[sp] for sp in species_labels],
    dtype=np.int64
)

num_species = len(unique_species)

print("Total samples:", n_samples, flush=True)
print("Total features:", num_feat, flush=True)
print("Total antibiotics:", num_antibiotics_total, flush=True)
print("Total species:", num_species, flush=True)

print("\nSpecies distribution:", flush=True)
unique_sp_ids, sp_counts = np.unique(species_ids_all, return_counts=True)

for sp_id, c in zip(unique_sp_ids, sp_counts):
    print(f"{id_to_species[int(sp_id)]}: {c} samples", flush=True)

print("\nHospital/source distribution:", flush=True)
unique_hospitals, hospital_counts = np.unique(hospital_all, return_counts=True)

for h, c in zip(unique_hospitals, hospital_counts):
    print(f"Hospital/source {h}: {c} samples", flush=True)


# ============================================================
# 5-FOLD TRAINING
# ============================================================

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_results = []
species_results = []
antibiotic_results = []
gamma_pair_results = []
metadata_used_rows = []

sample_indices = np.arange(n_samples)

global_mapping_payload = {
    "species_to_id": {str(k): int(v) for k, v in species_to_id.items()},
    "id_to_species": {str(k): str(v) for k, v in id_to_species.items()},
    "antibiotics_all": [str(a) for a in antibiotics_all],
    "model_modes": MODEL_MODES,
    "config": {
        "maldi_emb_dim": MALDI_EMB_DIM,
        "drug_emb_dim": DRUG_EMB_DIM,
        "species_emb_dim": SPECIES_EMB_DIM,
        "lambda_delta": LAMBDA_DELTA,
        "prevalence_smoothing_alpha": PREVALENCE_SMOOTHING_ALPHA,
        "prevalence_clip_min": PREVALENCE_CLIP_MIN,
        "prevalence_clip_max": PREVALENCE_CLIP_MAX,
        "initial_gamma": INITIAL_GAMMA,
        "gamma_mode": "learned_per_species_antibiotic_pair",
        "batch_size": BATCH_SIZE,
        "val_batch_size": VAL_BATCH_SIZE,
        "precision": PRECISION,
        "accelerator": ACCELERATOR,
    },
    "fold_mappings": {}
}

for model_mode in MODEL_MODES:

    print("\n##################################################", flush=True)
    print(f"RUNNING MODEL MODE: {model_mode}", flush=True)
    print("##################################################", flush=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(sample_indices)):

        print("\n==================================================", flush=True)
        print(f"MODE: {model_mode} | FOLD {fold + 1}/{N_SPLITS}", flush=True)
        print("==================================================", flush=True)

        X_train_source = X_all[train_idx]
        X_val_source = X_all[val_idx]

        species_train_source = species_ids_all[train_idx]
        species_val_source = species_ids_all[val_idx]

        amr_train_source = amr_all[train_idx]
        amr_val_source = amr_all[val_idx]

        print("Train samples:", X_train_source.shape[0], flush=True)
        print("Validation samples:", X_val_source.shape[0], flush=True)

        valid_cols = select_valid_antibiotics(
            amr_train=amr_train_source,
            amr_val=amr_val_source
        )

        if len(valid_cols) == 0:
            print("Skipping fold: no valid antibiotics", flush=True)
            continue

        selected_antibiotics = antibiotics_all[valid_cols]
        num_items = len(selected_antibiotics)

        print("Valid antibiotics:", num_items, flush=True)
        print("Antibiotics:", list(selected_antibiotics), flush=True)

        X_train = np.asarray(X_train_source)
        X_val = np.asarray(X_val_source)

        species_train = np.asarray(species_train_source)
        species_val = np.asarray(species_val_source)

        amr_train = amr_train_source[:, valid_cols]
        amr_val = amr_val_source[:, valid_cols]

        antibiotic_feature_table, fold_bio_mappings, df_meta_used = (
            build_antibiotic_metadata_tables(selected_antibiotics)
        )

        df_meta_used["model_mode"] = model_mode
        df_meta_used["fold"] = fold
        metadata_used_rows.append(df_meta_used)

        missing_meta = int((df_meta_used["metadata_found"] == 0).sum())
        print("Antibiotics missing metadata:", missing_meta, flush=True)

        prior_logit_matrix, antibiotic_prev = compute_train_prevalence_prior(
            amr_train=amr_train,
            species_train=species_train,
            num_species=num_species
        )

        train_dataset = BioPriorRecDataset(
            X=X_train,
            species_ids=species_train,
            amr=amr_train,
            antibiotic_feature_table=antibiotic_feature_table,
            prior_logit_matrix=prior_logit_matrix
        )

        val_dataset = BioPriorRecDataset(
            X=X_val,
            species_ids=species_val,
            amr=amr_val,
            antibiotic_feature_table=antibiotic_feature_table,
            prior_logit_matrix=prior_logit_matrix
        )

        if len(train_dataset) == 0:
            print("Skipping fold: empty train dataset", flush=True)
            continue

        if len(val_dataset) == 0:
            print("Skipping fold: empty val dataset", flush=True)
            continue

        print("Train pairs:", len(train_dataset), flush=True)
        print("Validation pairs:", len(val_dataset), flush=True)

        loader_train = make_loader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        loader_val = make_loader(
            dataset=val_dataset,
            batch_size=VAL_BATCH_SIZE,
            shuffle=False
        )

        model = BioPriorGatedRecommender(
            num_feat=num_feat,
            num_items=num_items,
            num_species=num_species,
            num_atc2=len(fold_bio_mappings["atc2_to_id"]),
            num_atc3=len(fold_bio_mappings["atc3_to_id"]),
            num_atc4=len(fold_bio_mappings["atc4_to_id"]),
            num_mechanisms=len(fold_bio_mappings["mechanism_to_id"]),
            model_mode=model_mode,
            maldi_emb_dim=MALDI_EMB_DIM,
            drug_emb_dim=DRUG_EMB_DIM,
            species_emb_dim=SPECIES_EMB_DIM,
            atc2_dim=ATC2_EMB_DIM,
            atc3_dim=ATC3_EMB_DIM,
            atc4_dim=ATC4_EMB_DIM,
            mechanism_dim=MECHANISM_EMB_DIM,
            context_hidden_dim=CONTEXT_HIDDEN_DIM,
            recommender_hidden_dims=RECOMMENDER_HIDDEN_DIMS,
            lr=LR,
            lambda_delta=LAMBDA_DELTA,
            initial_gamma=INITIAL_GAMMA
        )

        print("Fusion input dim:", model.fusion_input_dim, flush=True)
        print("Context dim:", model.context_dim, flush=True)

        initial_gamma_matrix = torch.sigmoid(model.gamma_raw).detach().cpu().numpy()

        print(
            "Initial gamma mean/std/min/max:",
            float(initial_gamma_matrix.mean()),
            float(initial_gamma_matrix.std()),
            float(initial_gamma_matrix.min()),
            float(initial_gamma_matrix.max()),
            flush=True
        )

        global_mapping_payload["fold_mappings"][f"{model_mode}_fold_{fold}"] = {
            "selected_antibiotics": [str(a) for a in selected_antibiotics],
            "valid_cols_original": [int(c) for c in valid_cols],
            "bio_mappings": fold_bio_mappings,
            "antibiotic_train_prevalence": {
                str(selected_antibiotics[j]): float(antibiotic_prev[j])
                for j in range(num_items)
            }
        }

        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator=ACCELERATOR,
            devices=DEVICES,
            precision=PRECISION,
            callbacks=[
                EarlyStopping(
                    monitor="loss_val",
                    patience=PATIENCE,
                    mode="min"
                )
            ],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            deterministic=False
        )

        trainer.fit(model, loader_train, loader_val)

        final_gamma_matrix = torch.sigmoid(model.gamma_raw).detach().cpu().numpy()

        final_gamma = float(final_gamma_matrix.mean())
        final_gamma_std = float(final_gamma_matrix.std())
        final_gamma_min = float(final_gamma_matrix.min())
        final_gamma_max = float(final_gamma_matrix.max())

        print(
            "Final gamma mean/std/min/max:",
            final_gamma,
            final_gamma_std,
            final_gamma_min,
            final_gamma_max,
            flush=True
        )

        gamma_pair_rows = extract_gamma_pair_rows(
            model=model,
            model_mode=model_mode,
            fold=fold,
            selected_antibiotics=selected_antibiotics,
            id_to_species=id_to_species
        )

        gamma_pair_results.extend(gamma_pair_rows)

        print("Predicting all validation sample-antibiotic pairs...", flush=True)

        preds_val = predict_all_pairs(
            model=model,
            X=X_val,
            species_ids=species_val,
            num_items=num_items,
            antibiotic_feature_table=antibiotic_feature_table,
            prior_logit_matrix=prior_logit_matrix,
            batch_size=PRED_BATCH_SIZE
        )

        auc_global, auc_macro_antibiotic, antibiotic_aucs = compute_global_metrics(
            y_true=amr_val,
            preds=preds_val
        )

        print("Fold global/micro AUC:", auc_global, flush=True)
        print("Fold macro antibiotic AUC:", auc_macro_antibiotic, flush=True)

        fold_results.append(
            {
                "model_mode": model_mode,
                "fold": fold,
                "n_train_samples": X_train.shape[0],
                "n_val_samples": X_val.shape[0],
                "n_antibiotics": num_items,
                "n_train_pairs": len(train_dataset),
                "n_val_pairs": len(val_dataset),
                "fusion_input_dim": model.fusion_input_dim,
                "context_dim": model.context_dim,
                "bio_dim": model.bio_dim,
                "maldi_emb_dim": MALDI_EMB_DIM,
                "species_emb_dim": SPECIES_EMB_DIM,
                "drug_emb_dim": DRUG_EMB_DIM,
                "lambda_delta": LAMBDA_DELTA,
                "prevalence_smoothing_alpha": PREVALENCE_SMOOTHING_ALPHA,
                "final_gamma": final_gamma,
                "final_gamma_std": final_gamma_std,
                "final_gamma_min": final_gamma_min,
                "final_gamma_max": final_gamma_max,
                "global_auc": auc_global,
                "macro_antibiotic_auc": auc_macro_antibiotic,
                "antibiotics": ";".join(map(str, selected_antibiotics))
            }
        )

        species_rows = compute_per_species_metrics(
            model_mode=model_mode,
            fold=fold,
            y_true=amr_val,
            preds=preds_val,
            species_ids=species_val,
            id_to_species=id_to_species
        )

        species_results.extend(species_rows)

        antibiotic_rows = compute_per_antibiotic_metrics(
            model_mode=model_mode,
            fold=fold,
            y_true=amr_val,
            preds=preds_val,
            selected_antibiotics=selected_antibiotics
        )

        antibiotic_results.extend(antibiotic_rows)

        del model
        del trainer
        del loader_train
        del loader_val
        del train_dataset
        del val_dataset

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# SAVE RESULTS
# ============================================================

df_folds = pd.DataFrame(fold_results)
df_species = pd.DataFrame(species_results)
df_antibiotics = pd.DataFrame(antibiotic_results)
df_gamma_pairs = pd.DataFrame(gamma_pair_results)

if len(metadata_used_rows) > 0:
    df_metadata_used = pd.concat(metadata_used_rows, ignore_index=True)
else:
    df_metadata_used = pd.DataFrame()

print("\n==================================================")
print("FINAL FOLD RESULTS")
print("==================================================")
print(df_folds)

if len(df_folds) > 0:
    summary_rows = []

    for model_mode in MODEL_MODES:
        df_mode = df_folds[df_folds["model_mode"] == model_mode]

        summary_rows.append(
            {
                "model_mode": model_mode,
                "mean_global_auc": df_mode["global_auc"].mean(),
                "std_global_auc": df_mode["global_auc"].std(),
                "mean_macro_antibiotic_auc": df_mode["macro_antibiotic_auc"].mean(),
                "std_macro_antibiotic_auc": df_mode["macro_antibiotic_auc"].std(),
                "mean_final_gamma": df_mode["final_gamma"].mean(),
                "std_final_gamma": df_mode["final_gamma"].std(),
                "mean_final_gamma_pair_std": df_mode["final_gamma_std"].mean(),
                "mean_final_gamma_pair_min": df_mode["final_gamma_min"].mean(),
                "mean_final_gamma_pair_max": df_mode["final_gamma_max"].mean(),
                "n_folds": len(df_mode)
            }
        )

    df_summary = pd.DataFrame(summary_rows)

    print("\n==================================================")
    print("SUMMARY")
    print("==================================================")
    print(df_summary)

else:
    df_summary = pd.DataFrame()

df_species_summary = summarize_entity_results(df_species, "species")
df_antibiotic_summary = summarize_entity_results(df_antibiotics, "antibiotic")

df_folds.to_csv(OUTPUT_FOLD_CSV, index=False)
df_species.to_csv(OUTPUT_SPECIES_CSV, index=False)
df_antibiotics.to_csv(OUTPUT_ANTIBIOTIC_CSV, index=False)
df_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
df_metadata_used.to_csv(OUTPUT_METADATA_USED_CSV, index=False)
df_gamma_pairs.to_csv(OUTPUT_GAMMA_PAIR_CSV, index=False)

df_species_summary.to_csv(
    "bio_prior_gated_per_species_summary.csv",
    index=False
)

df_antibiotic_summary.to_csv(
    "bio_prior_gated_per_antibiotic_summary.csv",
    index=False
)

with open(OUTPUT_MAPPING_JSON, "w") as f:
    json.dump(global_mapping_payload, f, indent=4)

print("\nSaved:")
print(OUTPUT_FOLD_CSV)
print(OUTPUT_SPECIES_CSV)
print(OUTPUT_ANTIBIOTIC_CSV)
print(OUTPUT_SUMMARY_CSV)
print(OUTPUT_MAPPING_JSON)
print(OUTPUT_METADATA_USED_CSV)
print(OUTPUT_GAMMA_PAIR_CSV)
print("bio_prior_gated_per_species_summary.csv")
print("bio_prior_gated_per_antibiotic_summary.csv")
