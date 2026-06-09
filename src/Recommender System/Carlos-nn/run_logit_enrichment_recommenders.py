# ============================================================
# LOGIT-ENRICHED RECOMMENDERS FOR MALDI-TOF AMR PREDICTION
#
# Runs two controlled experiments:
#
#   1) species_multihead_logit_enrichment
#      - Shared MALDI backbone
#      - Species-specific MALDI head
#      - Base recommender logit from MALDI + species + antibiotic
#      - Biological/epidemiological/clinical enrichment added as an extra logit
#
#   2) implicit_species_logit_enrichment
#      - Global MALDI encoder
#      - Base recommender logit from MALDI + species + antibiotic
#      - Biological/epidemiological/clinical enrichment added as an extra logit
#
# Core idea:
#
#   base_logit = BaseMLP([z_maldi, species_emb, drug_emb])
#
#   enrichment_emb = [
#       ATC2_emb, ATC3_emb, ATC4_emb, mechanism_emb,
#       beta_lactamase_projection,
#       genus_emb, gram_emb,
#       hospital_emb, year_projection,
#       train_prevalence_prior_projection
#   ]
#
#   enrichment_logit = EnrichmentMLP([enrichment_emb, species_emb, drug_emb])
#
#   final_logit = base_logit + enrichment_logit
#
# The epidemiological prior is computed ONLY on train inside each fold.
# No validation labels are used to build any train-derived prior.
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

OUTPUT_FOLD_CSV = "logit_enrichment_fold_results.csv"
OUTPUT_SPECIES_CSV = "logit_enrichment_per_species.csv"
OUTPUT_ANTIBIOTIC_CSV = "logit_enrichment_per_antibiotic.csv"
OUTPUT_SPECIES_ANTIBIOTIC_CSV = "logit_enrichment_per_species_antibiotic.csv"
OUTPUT_SPECIES_ANTIBIOTIC_SUMMARY_CSV = "logit_enrichment_per_species_antibiotic_summary.csv"
OUTPUT_SUMMARY_CSV = "logit_enrichment_summary.csv"
OUTPUT_SPECIES_SUMMARY_CSV = "logit_enrichment_per_species_summary.csv"
OUTPUT_ANTIBIOTIC_SUMMARY_CSV = "logit_enrichment_per_antibiotic_summary.csv"
OUTPUT_MAPPING_JSON = "logit_enrichment_mappings.json"
OUTPUT_METADATA_USED_CSV = "logit_enrichment_metadata_used.csv"

N_SPLITS = 2
RANDOM_STATE = 42

# Antibiotic filtering per fold
MIN_TRAIN_OBS_PER_ANTIBIOTIC = 50
MIN_VAL_OBS_PER_ANTIBIOTIC = 5

# Training
MAX_EPOCHS = 300
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 0.0

# GPU settings
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

# Embedding dimensions
MALDI_EMB_DIM = 32
DRUG_EMB_DIM = 16
SPECIES_EMB_DIM = 16

ATC2_EMB_DIM = 4
ATC3_EMB_DIM = 6
ATC4_EMB_DIM = 8
MECHANISM_EMB_DIM = 6
BETA_EMB_DIM = 2
GENUS_EMB_DIM = 8
GRAM_EMB_DIM = 3
HOSPITAL_EMB_DIM = 4
YEAR_EMB_DIM = 2
PRIOR_EMB_DIM = 4

BASE_HIDDEN_DIMS = [128, 64]
ENRICHMENT_HIDDEN_DIMS = [64, 32]

# Prevalence smoothing
PREVALENCE_SMOOTHING_ALPHA = 30.0
PREVALENCE_CLIP_MIN = 0.02
PREVALENCE_CLIP_MAX = 0.98

MODEL_MODES = [
    "species_multihead_logit_enrichment",
    "implicit_species_logit_enrichment",
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
print("Prediction batch size:", PRED_BATCH_SIZE, flush=True)
print("Num workers:", NUM_WORKERS, flush=True)


# ============================================================
# ANTIBIOTIC METADATA
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


# ============================================================
# BIOLOGICAL HEURISTICS
# ============================================================

GRAM_MAP_BY_GENUS = {
    "Staphylococcus": "gram_positive",
    "Streptococcus": "gram_positive",
    "Enterococcus": "gram_positive",
    "Micrococcus": "gram_positive",
    "Bacillus": "gram_positive",
    "Listeria": "gram_positive",
    "Corynebacterium": "gram_positive",
    "Cutibacterium": "gram_positive",
    "Propionibacterium": "gram_positive",

    "Escherichia": "gram_negative",
    "Klebsiella": "gram_negative",
    "Pseudomonas": "gram_negative",
    "Acinetobacter": "gram_negative",
    "Enterobacter": "gram_negative",
    "Citrobacter": "gram_negative",
    "Proteus": "gram_negative",
    "Serratia": "gram_negative",
    "Morganella": "gram_negative",
    "Providencia": "gram_negative",
    "Salmonella": "gram_negative",
    "Shigella": "gram_negative",
    "Haemophilus": "gram_negative",
    "Neisseria": "gram_negative",
    "Moraxella": "gram_negative",
    "Campylobacter": "gram_negative",
    "Helicobacter": "gram_negative",
    "Stenotrophomonas": "gram_negative",
    "Burkholderia": "gram_negative",
    "Bacteroides": "gram_negative",

    "Candida": "fungus",
    "Cryptococcus": "fungus",
    "Aspergillus": "fungus",

    "Mycobacterium": "mycobacteria",
}


def clean_name(x):
    if x is None:
        return "unknown"
    x = str(x).strip()
    if x == "" or x.lower() in ["nan", "none"]:
        return "unknown"
    return x


def get_genus(species_name):
    species_name = clean_name(species_name)
    if species_name == "unknown":
        return "unknown"
    if "_" in species_name:
        return species_name.split("_")[0]
    if " " in species_name:
        return species_name.split(" ")[0]
    return species_name


def get_gram_status(species_name):
    genus = get_genus(species_name)
    return GRAM_MAP_BY_GENUS.get(genus, "unknown")


def get_metadata_value(meta, key, default_value):
    if meta is None:
        return default_value
    value = meta.get(key, default_value)
    if value is None:
        return default_value
    return value


# ============================================================
# DATASET
# ============================================================

class LogitEnrichmentRecDataset(Dataset):
    """
    Each item:
        (MALDI sample, species, antibiotic, enrichment variables) -> AMR label
    """

    def __init__(
        self,
        X,
        species_ids,
        amr,
        antibiotic_feature_table,
        sample_feature_table,
        prior_logit_matrix
    ):
        self.X = np.asarray(X)
        self.species_ids = np.asarray(species_ids, dtype=np.int64)
        self.amr = np.asarray(amr)
        self.antibiotic_feature_table = antibiotic_feature_table
        self.sample_feature_table = sample_feature_table
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
        species_id = int(self.species_ids[sample_idx])
        label = self.labels[idx]

        drug_row = self.antibiotic_feature_table[drug_id]
        sample_row = self.sample_feature_table[sample_idx]
        prior_logit = self.prior_logit_matrix[species_id, drug_id]

        return (
            torch.tensor(self.X[sample_idx]).float(),
            torch.tensor(species_id).long(),
            torch.tensor(drug_id).long(),
            torch.tensor(drug_row["atc2_id"]).long(),
            torch.tensor(drug_row["atc3_id"]).long(),
            torch.tensor(drug_row["atc4_id"]).long(),
            torch.tensor(drug_row["mechanism_id"]).long(),
            torch.tensor(drug_row["beta_lactamase_inhibitor"]).float(),
            torch.tensor(sample_row["genus_id"]).long(),
            torch.tensor(sample_row["gram_id"]).long(),
            torch.tensor(sample_row["hospital_id"]).long(),
            torch.tensor(sample_row["year_norm"]).float(),
            torch.tensor(prior_logit).float(),
            torch.tensor(label).float(),
        )


# ============================================================
# MODEL COMPONENTS
# ============================================================

class GlobalMALDIEncoder(nn.Module):
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
            nn.GELU(),
        )

    def forward(self, maldi):
        return self.encoder(maldi.float())


class SpeciesHeadMALDIEncoder(nn.Module):
    """
    Shared MALDI backbone + one species-specific MALDI head per species.
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

        self.species_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, maldi_emb_dim),
                nn.GELU(),
            )
            for _ in range(num_species)
        ])

    def forward(self, maldi, species_id):
        z = self.backbone(maldi.float())
        out = torch.zeros(
            size=(z.shape[0], self.output_dim),
            dtype=z.dtype,
            device=z.device,
        )

        for sp in torch.unique(species_id.long()):
            sp_int = int(sp.item())
            mask = species_id.long() == sp
            out[mask] = self.species_heads[sp_int](z[mask])

        return out


class EnrichmentEmbeddingBlock(nn.Module):
    """
    Separate embedding/projection for each enrichment variable.
    Each source keeps its own dimensionality.
    """

    def __init__(
        self,
        num_atc2,
        num_atc3,
        num_atc4,
        num_mechanisms,
        num_genus,
        num_gram,
        num_hospital,
        atc2_dim=4,
        atc3_dim=6,
        atc4_dim=8,
        mechanism_dim=6,
        beta_dim=2,
        genus_dim=8,
        gram_dim=3,
        hospital_dim=4,
        year_dim=2,
        prior_dim=4,
    ):
        super().__init__()

        self.atc2_embedding = nn.Embedding(num_atc2, atc2_dim)
        self.atc3_embedding = nn.Embedding(num_atc3, atc3_dim)
        self.atc4_embedding = nn.Embedding(num_atc4, atc4_dim)
        self.mechanism_embedding = nn.Embedding(num_mechanisms, mechanism_dim)
        self.genus_embedding = nn.Embedding(num_genus, genus_dim)
        self.gram_embedding = nn.Embedding(num_gram, gram_dim)
        self.hospital_embedding = nn.Embedding(num_hospital, hospital_dim)

        self.beta_projection = nn.Sequential(
            nn.Linear(1, beta_dim),
            nn.GELU(),
        )
        self.year_projection = nn.Sequential(
            nn.Linear(1, year_dim),
            nn.GELU(),
        )
        self.prior_projection = nn.Sequential(
            nn.Linear(1, prior_dim),
            nn.GELU(),
        )

        self.output_dim = (
            atc2_dim + atc3_dim + atc4_dim + mechanism_dim
            + beta_dim + genus_dim + gram_dim + hospital_dim
            + year_dim + prior_dim
        )

    def forward(
        self,
        atc2_id,
        atc3_id,
        atc4_id,
        mechanism_id,
        beta_lactamase_inhibitor,
        genus_id,
        gram_id,
        hospital_id,
        year_norm,
        prior_logit,
    ):
        atc2_emb = self.atc2_embedding(atc2_id.long())
        atc3_emb = self.atc3_embedding(atc3_id.long())
        atc4_emb = self.atc4_embedding(atc4_id.long())
        mechanism_emb = self.mechanism_embedding(mechanism_id.long())
        genus_emb = self.genus_embedding(genus_id.long())
        gram_emb = self.gram_embedding(gram_id.long())
        hospital_emb = self.hospital_embedding(hospital_id.long())

        beta_emb = self.beta_projection(beta_lactamase_inhibitor.view(-1, 1).float())
        year_emb = self.year_projection(year_norm.view(-1, 1).float())
        prior_emb = self.prior_projection(prior_logit.view(-1, 1).float())

        return torch.cat(
            [
                atc2_emb,
                atc3_emb,
                atc4_emb,
                mechanism_emb,
                beta_emb,
                genus_emb,
                gram_emb,
                hospital_emb,
                year_emb,
                prior_emb,
            ],
            dim=-1,
        )


class LogitEnrichedRecommender(pl.LightningModule):
    def __init__(
        self,
        num_feat,
        num_items,
        num_species,
        num_atc2,
        num_atc3,
        num_atc4,
        num_mechanisms,
        num_genus,
        num_gram,
        num_hospital,
        model_mode,
        maldi_emb_dim=32,
        drug_emb_dim=16,
        species_emb_dim=16,
        base_hidden_dims=[128, 64],
        enrichment_hidden_dims=[64, 32],
        lr=1e-3,
        weight_decay=0.0,
    ):
        super().__init__()

        assert model_mode in [
            "species_multihead_logit_enrichment",
            "implicit_species_logit_enrichment",
        ]

        self.save_hyperparameters()
        self.model_mode = model_mode
        self.lr = lr
        self.weight_decay = weight_decay

        if model_mode == "species_multihead_logit_enrichment":
            self.maldi_encoder = SpeciesHeadMALDIEncoder(
                input_dim=num_feat,
                num_species=num_species,
                maldi_emb_dim=maldi_emb_dim,
            )
        else:
            self.maldi_encoder = GlobalMALDIEncoder(
                input_dim=num_feat,
                maldi_emb_dim=maldi_emb_dim,
            )

        self.species_embedding = nn.Embedding(num_species, species_emb_dim)
        self.drug_embedding = nn.Embedding(num_items, drug_emb_dim)

        self.enrichment_embedding = EnrichmentEmbeddingBlock(
            num_atc2=num_atc2,
            num_atc3=num_atc3,
            num_atc4=num_atc4,
            num_mechanisms=num_mechanisms,
            num_genus=num_genus,
            num_gram=num_gram,
            num_hospital=num_hospital,
            atc2_dim=ATC2_EMB_DIM,
            atc3_dim=ATC3_EMB_DIM,
            atc4_dim=ATC4_EMB_DIM,
            mechanism_dim=MECHANISM_EMB_DIM,
            beta_dim=BETA_EMB_DIM,
            genus_dim=GENUS_EMB_DIM,
            gram_dim=GRAM_EMB_DIM,
            hospital_dim=HOSPITAL_EMB_DIM,
            year_dim=YEAR_EMB_DIM,
            prior_dim=PRIOR_EMB_DIM,
        )

        self.enrichment_dim = self.enrichment_embedding.output_dim

        base_input_dim = maldi_emb_dim + species_emb_dim + drug_emb_dim
        enrichment_input_dim = self.enrichment_dim + species_emb_dim + drug_emb_dim

        self.base_mlp = self._make_mlp(
            input_dim=base_input_dim,
            hidden_dims=base_hidden_dims,
            output_dim=1,
            zero_last=False,
        )

        # Zero-initialize the final enrichment layer so training starts from the base recommender.
        self.enrichment_logit_mlp = self._make_mlp(
            input_dim=enrichment_input_dim,
            hidden_dims=enrichment_hidden_dims,
            output_dim=1,
            zero_last=True,
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.base_input_dim = base_input_dim
        self.enrichment_input_dim = enrichment_input_dim

    def _make_mlp(self, input_dim, hidden_dims, output_dim, zero_last=False):
        sizes = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []

        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        last = nn.Linear(sizes[-2], sizes[-1])

        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

        layers.append(last)
        return nn.Sequential(*layers)

    def _encode_maldi(self, maldi, species_id):
        if self.model_mode == "species_multihead_logit_enrichment":
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
        genus_id,
        gram_id,
        hospital_id,
        year_norm,
        prior_logit,
    ):
        z_maldi = self._encode_maldi(maldi, species_id)
        species_emb = self.species_embedding(species_id.long())
        drug_emb = self.drug_embedding(drug_id.long())

        base_x = torch.cat([z_maldi, species_emb, drug_emb], dim=-1)
        base_logit = self.base_mlp(base_x)

        enrichment_emb = self.enrichment_embedding(
            atc2_id=atc2_id,
            atc3_id=atc3_id,
            atc4_id=atc4_id,
            mechanism_id=mechanism_id,
            beta_lactamase_inhibitor=beta_lactamase_inhibitor,
            genus_id=genus_id,
            gram_id=gram_id,
            hospital_id=hospital_id,
            year_norm=year_norm,
            prior_logit=prior_logit,
        )

        enrichment_x = torch.cat([enrichment_emb, species_emb, drug_emb], dim=-1)
        enrichment_logit = self.enrichment_logit_mlp(enrichment_x)

        final_logit = base_logit + enrichment_logit

        return final_logit, base_logit, enrichment_logit

    def _shared_step(self, batch, stage):
        (
            maldi,
            species_id,
            drug_id,
            atc2_id,
            atc3_id,
            atc4_id,
            mechanism_id,
            beta,
            genus_id,
            gram_id,
            hospital_id,
            year_norm,
            prior_logit,
            labels,
        ) = batch

        logits, base_logit, enrichment_logit = self.forward(
            maldi=maldi,
            species_id=species_id,
            drug_id=drug_id,
            atc2_id=atc2_id,
            atc3_id=atc3_id,
            atc4_id=atc4_id,
            mechanism_id=mechanism_id,
            beta_lactamase_inhibitor=beta,
            genus_id=genus_id,
            gram_id=gram_id,
            hospital_id=hospital_id,
            year_norm=year_norm,
            prior_logit=prior_logit,
        )

        labels = labels.view(-1, 1).float()
        loss = self.loss_fn(logits, labels)

        self.log(f"loss_{stage}", loss, prog_bar=True)
        self.log(f"base_logit_mean_{stage}", base_logit.detach().mean(), prog_bar=False)
        self.log(f"enrichment_logit_mean_{stage}", enrichment_logit.detach().mean(), prog_bar=False)
        self.log(f"enrichment_logit_abs_{stage}", enrichment_logit.detach().abs().mean(), prog_bar=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "tr")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


# ============================================================
# UTILS
# ============================================================

def clean_amr(amr):
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
        drop_last=False,
    )


def count_observed_with_two_classes(col):
    obs = col[~np.isnan(col)]
    n_obs = len(obs)
    has_two_classes = n_obs > 0 and len(np.unique(obs)) > 1
    return n_obs, has_two_classes


def select_valid_antibiotics(amr_train, amr_val):
    valid_cols = []
    for j in range(amr_train.shape[1]):
        n_train, train_two_classes = count_observed_with_two_classes(amr_train[:, j])
        n_val, val_two_classes = count_observed_with_two_classes(amr_val[:, j])

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
    y = np.asarray(y)
    p = np.asarray(p)

    if len(y) == 0:
        return np.nan
    if len(np.unique(y)) < 2:
        return np.nan

    return roc_auc_score(y, p)


def logit_np(p):
    p = np.clip(p, PREVALENCE_CLIP_MIN, PREVALENCE_CLIP_MAX)
    return np.log(p / (1.0 - p))


def normalize_year_array(year_values, y_min=None, y_max=None):
    years = np.asarray(year_values, dtype=float)
    finite = years[np.isfinite(years)]

    if y_min is None or y_max is None:
        if len(finite) == 0:
            y_min, y_max = 0.0, 1.0
        else:
            y_min, y_max = float(np.min(finite)), float(np.max(finite))

    if y_max <= y_min:
        return np.zeros_like(years, dtype=np.float32), y_min, y_max

    out = (years - y_min) / (y_max - y_min)
    out = np.where(np.isfinite(out), out, 0.0)
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32), y_min, y_max


def build_metadata_maps(selected_antibiotics, species_labels_all, hospital_all):
    atc2_values = ["UNK_ATC2"]
    atc3_values = ["UNK_ATC3"]
    atc4_values = ["UNK_ATC4"]
    mechanism_values = ["unknown"]

    for antibiotic_name in selected_antibiotics:
        antibiotic_name = str(antibiotic_name)
        meta = antibiotic_metadata.get(antibiotic_name, None)

        atc2 = str(get_metadata_value(meta, "ATC2", "UNK_ATC2"))
        atc3 = str(get_metadata_value(meta, "ATC3", "UNK_ATC3"))
        atc4 = str(get_metadata_value(meta, "ATC4", "UNK_ATC4"))
        mechanism = str(get_metadata_value(meta, "mechanism", "unknown"))

        if atc2 not in atc2_values:
            atc2_values.append(atc2)
        if atc3 not in atc3_values:
            atc3_values.append(atc3)
        if atc4 not in atc4_values:
            atc4_values.append(atc4)
        if mechanism not in mechanism_values:
            mechanism_values.append(mechanism)

    genus_values = ["unknown"]
    gram_values = ["unknown"]

    for sp in species_labels_all:
        genus = get_genus(sp)
        gram = get_gram_status(sp)
        if genus not in genus_values:
            genus_values.append(genus)
        if gram not in gram_values:
            gram_values.append(gram)

    hospital_values = ["unknown"]
    for h in hospital_all:
        h = clean_name(h)
        if h not in hospital_values:
            hospital_values.append(h)

    maps = {
        "atc2_to_id": {v: i for i, v in enumerate(atc2_values)},
        "atc3_to_id": {v: i for i, v in enumerate(atc3_values)},
        "atc4_to_id": {v: i for i, v in enumerate(atc4_values)},
        "mechanism_to_id": {v: i for i, v in enumerate(mechanism_values)},
        "genus_to_id": {v: i for i, v in enumerate(genus_values)},
        "gram_to_id": {v: i for i, v in enumerate(gram_values)},
        "hospital_to_id": {v: i for i, v in enumerate(hospital_values)},
    }

    return maps


def get_id(value, mapping):
    value = clean_name(value)
    if value not in mapping:
        value = "unknown"
    return int(mapping.get(value, 0))


def build_antibiotic_feature_table(selected_antibiotics, maps):
    rows = []
    used_rows = []

    for antibiotic_name in selected_antibiotics:
        antibiotic_name = str(antibiotic_name)
        meta = antibiotic_metadata.get(antibiotic_name, None)

        atc2 = str(get_metadata_value(meta, "ATC2", "UNK_ATC2"))
        atc3 = str(get_metadata_value(meta, "ATC3", "UNK_ATC3"))
        atc4 = str(get_metadata_value(meta, "ATC4", "UNK_ATC4"))
        mechanism = str(get_metadata_value(meta, "mechanism", "unknown"))
        beta = get_metadata_value(meta, "beta_lactamase_inhibitor", 0)

        if beta is None:
            beta = 0

        row = {
            "antibiotic": antibiotic_name,
            "atc2_id": int(maps["atc2_to_id"].get(atc2, 0)),
            "atc3_id": int(maps["atc3_to_id"].get(atc3, 0)),
            "atc4_id": int(maps["atc4_to_id"].get(atc4, 0)),
            "mechanism_id": int(maps["mechanism_to_id"].get(mechanism, 0)),
            "beta_lactamase_inhibitor": float(beta),
        }
        rows.append(row)

        used_rows.append({
            "antibiotic": antibiotic_name,
            "ATC2": atc2,
            "ATC3": atc3,
            "ATC4": atc4,
            "mechanism": mechanism,
            "beta_lactamase_inhibitor": float(beta),
            "metadata_found": int(meta is not None),
        })

    return rows, pd.DataFrame(used_rows)


def build_sample_feature_table(species_labels_split, hospital_split, year_norm_split, maps):
    rows = []

    for sp, h, y in zip(species_labels_split, hospital_split, year_norm_split):
        genus = get_genus(sp)
        gram = get_gram_status(sp)
        hospital = clean_name(h)

        rows.append({
            "genus_id": int(maps["genus_to_id"].get(genus, 0)),
            "gram_id": int(maps["gram_to_id"].get(gram, 0)),
            "hospital_id": int(maps["hospital_to_id"].get(hospital, 0)),
            "year_norm": float(y),
        })

    return rows


def compute_train_prevalence_prior(amr_train, species_train, num_species):
    """
    Train-only smoothed species-antibiotic prevalence prior.
    Returns prior_logit_matrix[num_species, num_items] and antibiotic_prev[num_items].
    """

    amr_train = np.asarray(amr_train)
    species_train = np.asarray(species_train)
    num_items = amr_train.shape[1]

    observed = ~np.isnan(amr_train)
    resistant = amr_train == 1

    total_obs = int(np.sum(observed))
    total_res = int(np.sum(resistant))
    dataset_prev = total_res / total_obs if total_obs > 0 else 0.5

    antibiotic_prev = np.zeros(num_items, dtype=np.float32)
    antibiotic_nobs = np.zeros(num_items, dtype=np.float32)

    for j in range(num_items):
        obs_j = observed[:, j]
        n_obs_j = int(np.sum(obs_j))
        antibiotic_nobs[j] = n_obs_j

        if n_obs_j > 0:
            antibiotic_prev[j] = int(np.sum(resistant[:, j])) / n_obs_j
        else:
            antibiotic_prev[j] = dataset_prev

    prior_logit_matrix = np.zeros((num_species, num_items), dtype=np.float32)
    species_antibiotic_nobs = np.zeros((num_species, num_items), dtype=np.float32)
    alpha = PREVALENCE_SMOOTHING_ALPHA

    for sp in range(num_species):
        sp_mask = species_train == sp

        for j in range(num_items):
            col = amr_train[sp_mask, j]
            obs = ~np.isnan(col)
            n_obs = int(np.sum(obs))
            n_res = int(np.sum(col[obs] == 1)) if n_obs > 0 else 0

            species_antibiotic_nobs[sp, j] = n_obs

            prev_smoothed = (
                n_res + alpha * antibiotic_prev[j]
            ) / (
                n_obs + alpha
            )

            prior_logit_matrix[sp, j] = logit_np(prev_smoothed)

    return prior_logit_matrix, antibiotic_prev, antibiotic_nobs, species_antibiotic_nobs


def predict_all_pairs(
    model,
    X,
    species_ids,
    num_items,
    antibiotic_feature_table,
    sample_feature_table,
    prior_logit_matrix,
    batch_size=128,
):
    model.eval()
    device = model.device

    X = np.asarray(X)
    species_ids = np.asarray(species_ids, dtype=np.int64)
    n_samples = X.shape[0]
    preds_matrix = np.zeros((n_samples, num_items), dtype=np.float32)

    # Antibiotic metadata tensors on device
    atc2_all = torch.tensor([r["atc2_id"] for r in antibiotic_feature_table], dtype=torch.long, device=device)
    atc3_all = torch.tensor([r["atc3_id"] for r in antibiotic_feature_table], dtype=torch.long, device=device)
    atc4_all = torch.tensor([r["atc4_id"] for r in antibiotic_feature_table], dtype=torch.long, device=device)
    mech_all = torch.tensor([r["mechanism_id"] for r in antibiotic_feature_table], dtype=torch.long, device=device)
    beta_all = torch.tensor([r["beta_lactamase_inhibitor"] for r in antibiotic_feature_table], dtype=torch.float32, device=device)

    genus_all = np.asarray([r["genus_id"] for r in sample_feature_table], dtype=np.int64)
    gram_all = np.asarray([r["gram_id"] for r in sample_feature_table], dtype=np.int64)
    hospital_all = np.asarray([r["hospital_id"] for r in sample_feature_table], dtype=np.int64)
    year_all = np.asarray([r["year_norm"] for r in sample_feature_table], dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            bs = end - start

            maldi_batch = torch.tensor(X[start:end], dtype=torch.float32, device=device)
            species_batch = torch.tensor(species_ids[start:end], dtype=torch.long, device=device)

            genus_batch = torch.tensor(genus_all[start:end], dtype=torch.long, device=device)
            gram_batch = torch.tensor(gram_all[start:end], dtype=torch.long, device=device)
            hospital_batch = torch.tensor(hospital_all[start:end], dtype=torch.long, device=device)
            year_batch = torch.tensor(year_all[start:end], dtype=torch.float32, device=device)

            drug_preds = []

            for drug_id in range(num_items):
                drug_batch = torch.full((bs,), drug_id, dtype=torch.long, device=device)
                atc2_batch = atc2_all[drug_id].repeat(bs)
                atc3_batch = atc3_all[drug_id].repeat(bs)
                atc4_batch = atc4_all[drug_id].repeat(bs)
                mech_batch = mech_all[drug_id].repeat(bs)
                beta_batch = beta_all[drug_id].repeat(bs)

                prior_vals = prior_logit_matrix[species_ids[start:end], drug_id]
                prior_batch = torch.tensor(prior_vals, dtype=torch.float32, device=device)

                logits, _, _ = model(
                    maldi=maldi_batch,
                    species_id=species_batch,
                    drug_id=drug_batch,
                    atc2_id=atc2_batch,
                    atc3_id=atc3_batch,
                    atc4_id=atc4_batch,
                    mechanism_id=mech_batch,
                    beta_lactamase_inhibitor=beta_batch,
                    genus_id=genus_batch,
                    gram_id=gram_batch,
                    hospital_id=hospital_batch,
                    year_norm=year_batch,
                    prior_logit=prior_batch,
                )

                probs = torch.sigmoid(logits).view(-1)
                drug_preds.append(probs)

            drug_preds = torch.stack(drug_preds, dim=1)
            preds_matrix[start:end] = drug_preds.float().cpu().numpy()

    return preds_matrix


# ============================================================
# METRICS
# ============================================================

def compute_global_metrics(y_true, preds):
    mask = ~np.isnan(y_true)

    if np.any(mask):
        global_micro_auc = safe_auc(y_true[mask], preds[mask])
    else:
        global_micro_auc = np.nan

    antibiotic_aucs = []
    for j in range(y_true.shape[1]):
        valid = ~np.isnan(y_true[:, j])
        auc_j = safe_auc(y_true[valid, j], preds[valid, j])
        if np.isfinite(auc_j):
            antibiotic_aucs.append(auc_j)

    macro_antibiotic_auc = float(np.mean(antibiotic_aucs)) if len(antibiotic_aucs) > 0 else np.nan

    return global_micro_auc, macro_antibiotic_auc, antibiotic_aucs


def compute_per_species_metrics(model_mode, fold, y_true, preds, species_ids, id_to_species):
    rows = []

    for sp_id in np.unique(species_ids):
        sp_mask = species_ids == sp_id
        y_sp = y_true[sp_mask]
        p_sp = preds[sp_mask]
        obs_mask = ~np.isnan(y_sp)

        n_samples = int(np.sum(sp_mask))
        n_pairs = int(np.sum(obs_mask))

        # 1) Micro AUC by species: flatten all observed species-antibiotic pairs.
        species_micro_auc = np.nan
        if n_pairs > 0:
            species_micro_auc = safe_auc(y_sp[obs_mask], p_sp[obs_mask])

        # 2) Macro AUC by species: compute AUC per antibiotic within that species, then average.
        per_antibiotic_aucs = []
        per_antibiotic_n_pairs = []

        for j in range(y_sp.shape[1]):
            valid_j = ~np.isnan(y_sp[:, j])
            auc_j = safe_auc(y_sp[valid_j, j], p_sp[valid_j, j])

            if np.isfinite(auc_j):
                per_antibiotic_aucs.append(auc_j)
                per_antibiotic_n_pairs.append(int(np.sum(valid_j)))

        species_macro_antibiotic_auc = (
            float(np.mean(per_antibiotic_aucs))
            if len(per_antibiotic_aucs) > 0
            else np.nan
        )

        rows.append({
            "model_mode": model_mode,
            "fold": fold,
            "species_id": int(sp_id),
            "species": str(id_to_species[int(sp_id)]),
            "n_val_samples": n_samples,
            "n_val_pairs": n_pairs,
            "species_micro_auc": species_micro_auc,
            "species_macro_antibiotic_auc": species_macro_antibiotic_auc,
            "n_antibiotics_with_auc": len(per_antibiotic_aucs),
            "mean_pairs_per_antibiotic_with_auc": float(np.mean(per_antibiotic_n_pairs)) if len(per_antibiotic_n_pairs) > 0 else np.nan,
        })

    return rows


def compute_per_antibiotic_metrics(model_mode, fold, y_true, preds, selected_antibiotics):
    rows = []

    for j, antibiotic_name in enumerate(selected_antibiotics):
        valid = ~np.isnan(y_true[:, j])
        n_pairs = int(np.sum(valid))
        auc = safe_auc(y_true[valid, j], preds[valid, j])

        rows.append({
            "model_mode": model_mode,
            "fold": fold,
            "antibiotic_id": j,
            "antibiotic": str(antibiotic_name),
            "n_val_pairs": n_pairs,
            "antibiotic_auc": auc,
        })

    return rows

def compute_per_species_antibiotic_metrics(
    model_mode,
    fold,
    y_true,
    preds,
    species_ids,
    id_to_species,
    selected_antibiotics
):
    """
    Computes crossed AUC for each species-antibiotic pair.

    For each species s and antibiotic a:
        AUC(y_true samples from species s for antibiotic a,
            preds samples from species s for antibiotic a)

    AUC is NaN when there are not two classes in that pair.
    """

    rows = []

    unique_species_ids = np.unique(species_ids)

    for sp_id in unique_species_ids:
        sp_mask = species_ids == sp_id
        species_name = str(id_to_species[int(sp_id)])

        y_sp = y_true[sp_mask]
        p_sp = preds[sp_mask]

        for j, antibiotic_name in enumerate(selected_antibiotics):
            y_pair = y_sp[:, j]
            p_pair = p_sp[:, j]

            valid = ~np.isnan(y_pair)
            n_pairs = int(np.sum(valid))

            auc = safe_auc(y_pair[valid], p_pair[valid])

            n_susceptible = np.nan
            n_resistant = np.nan

            if n_pairs > 0:
                n_susceptible = int(np.sum(y_pair[valid] == 0))
                n_resistant = int(np.sum(y_pair[valid] == 1))

            rows.append({
                "model_mode": model_mode,
                "fold": fold,
                "species_id": int(sp_id),
                "species": species_name,
                "antibiotic_id": int(j),
                "antibiotic": str(antibiotic_name),
                "n_val_pairs": n_pairs,
                "n_susceptible": n_susceptible,
                "n_resistant": n_resistant,
                "species_antibiotic_auc": auc,
            })

    return rows

def summarize_species_results(df_species):
    if len(df_species) == 0:
        return pd.DataFrame()

    rows = []

    for (model_mode, species), sub in df_species.groupby(["model_mode", "species"]):
        rows.append({
            "model_mode": model_mode,
            "species": species,
            "mean_species_micro_auc": sub["species_micro_auc"].mean(),
            "std_species_micro_auc": sub["species_micro_auc"].std(),
            "mean_species_macro_antibiotic_auc": sub["species_macro_antibiotic_auc"].mean(),
            "std_species_macro_antibiotic_auc": sub["species_macro_antibiotic_auc"].std(),
            "mean_n_val_pairs": sub["n_val_pairs"].mean(),
            "mean_n_antibiotics_with_auc": sub["n_antibiotics_with_auc"].mean(),
            "n_folds": sub["fold"].nunique(),
        })

    return pd.DataFrame(rows)


def summarize_antibiotic_results(df_antibiotics):
    if len(df_antibiotics) == 0:
        return pd.DataFrame()

    rows = []

    for (model_mode, antibiotic), sub in df_antibiotics.groupby(["model_mode", "antibiotic"]):
        rows.append({
            "model_mode": model_mode,
            "antibiotic": antibiotic,
            "mean_antibiotic_auc": sub["antibiotic_auc"].mean(),
            "std_antibiotic_auc": sub["antibiotic_auc"].std(),
            "mean_n_val_pairs": sub["n_val_pairs"].mean(),
            "n_folds": sub["fold"].nunique(),
        })

    return pd.DataFrame(rows)


def summarize_species_antibiotic_results(df_species_antibiotic):
    if len(df_species_antibiotic) == 0:
        return pd.DataFrame()

    rows = []

    for (model_mode, species, antibiotic), sub in df_species_antibiotic.groupby(
        ["model_mode", "species", "antibiotic"]
    ):
        rows.append({
            "model_mode": model_mode,
            "species": species,
            "antibiotic": antibiotic,
            "mean_species_antibiotic_auc": sub["species_antibiotic_auc"].mean(),
            "std_species_antibiotic_auc": sub["species_antibiotic_auc"].std(),
            "mean_n_val_pairs": sub["n_val_pairs"].mean(),
            "mean_n_susceptible": sub["n_susceptible"].mean(),
            "mean_n_resistant": sub["n_resistant"].mean(),
            "n_folds": sub["fold"].nunique(),
            "n_folds_with_valid_auc": sub["species_antibiotic_auc"].notna().sum(),
        })

    return pd.DataFrame(rows)

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

if "year" in payload:
    year_all = np.asarray(payload["year"], dtype=float)
elif "years" in payload:
    year_all = np.asarray(payload["years"], dtype=float)
elif "date" in payload:
    year_all = pd.to_datetime(payload["date"], errors="coerce").year.to_numpy(dtype=float)
elif "dates" in payload:
    year_all = pd.to_datetime(payload["dates"], errors="coerce").year.to_numpy(dtype=float)
else:
    year_all = np.asarray([0.0] * X_all.shape[0], dtype=float)
    print("WARNING: No year/date field found in payload. year metadata set to 0.", flush=True)

n_samples = X_all.shape[0]
num_feat = X_all.shape[1]
num_antibiotics_total = amr_all.shape[1]

unique_species = np.unique(species_labels)
species_to_id = {sp: i for i, sp in enumerate(unique_species)}
id_to_species = {i: sp for sp, i in species_to_id.items()}

species_ids_all = np.asarray([species_to_id[sp] for sp in species_labels], dtype=np.int64)
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

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_results = []
species_results = []
antibiotic_results = []
species_antibiotic_results = []
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
        "atc2_emb_dim": ATC2_EMB_DIM,
        "atc3_emb_dim": ATC3_EMB_DIM,
        "atc4_emb_dim": ATC4_EMB_DIM,
        "mechanism_emb_dim": MECHANISM_EMB_DIM,
        "beta_emb_dim": BETA_EMB_DIM,
        "genus_emb_dim": GENUS_EMB_DIM,
        "gram_emb_dim": GRAM_EMB_DIM,
        "hospital_emb_dim": HOSPITAL_EMB_DIM,
        "year_emb_dim": YEAR_EMB_DIM,
        "prior_emb_dim": PRIOR_EMB_DIM,
        "prevalence_smoothing_alpha": PREVALENCE_SMOOTHING_ALPHA,
        "prevalence_clip_min": PREVALENCE_CLIP_MIN,
        "prevalence_clip_max": PREVALENCE_CLIP_MAX,
        "batch_size": BATCH_SIZE,
        "val_batch_size": VAL_BATCH_SIZE,
        "pred_batch_size": PRED_BATCH_SIZE,
        "precision": PRECISION,
        "accelerator": ACCELERATOR,
    },
    "fold_mappings": {},
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

        species_labels_train_source = species_labels[train_idx]
        species_labels_val_source = species_labels[val_idx]

        hospital_train_source = hospital_all[train_idx]
        hospital_val_source = hospital_all[val_idx]

        year_train_source = year_all[train_idx]
        year_val_source = year_all[val_idx]

        amr_train_source = amr_all[train_idx]
        amr_val_source = amr_all[val_idx]

        print("Train samples:", X_train_source.shape[0], flush=True)
        print("Validation samples:", X_val_source.shape[0], flush=True)

        valid_cols = select_valid_antibiotics(
            amr_train=amr_train_source,
            amr_val=amr_val_source,
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

        species_labels_train = np.asarray(species_labels_train_source)
        species_labels_val = np.asarray(species_labels_val_source)

        hospital_train = np.asarray(hospital_train_source)
        hospital_val = np.asarray(hospital_val_source)

        year_train = np.asarray(year_train_source, dtype=float)
        year_val = np.asarray(year_val_source, dtype=float)

        amr_train = amr_train_source[:, valid_cols]
        amr_val = amr_val_source[:, valid_cols]

        # Metadata maps are built using available labels/categories, not AMR outcomes.
        metadata_maps = build_metadata_maps(
            selected_antibiotics=selected_antibiotics,
            species_labels_all=species_labels,
            hospital_all=hospital_all,
        )

        # Normalize year using TRAIN fold only.
        year_train_norm, year_min, year_max = normalize_year_array(year_train)
        year_val_norm, _, _ = normalize_year_array(year_val, y_min=year_min, y_max=year_max)

        antibiotic_feature_table, df_meta_used = build_antibiotic_feature_table(
            selected_antibiotics=selected_antibiotics,
            maps=metadata_maps,
        )

        df_meta_used["model_mode"] = model_mode
        df_meta_used["fold"] = fold
        metadata_used_rows.append(df_meta_used)

        missing_meta = int((df_meta_used["metadata_found"] == 0).sum())
        print("Antibiotics missing metadata:", missing_meta, flush=True)

        sample_feature_train = build_sample_feature_table(
            species_labels_split=species_labels_train,
            hospital_split=hospital_train,
            year_norm_split=year_train_norm,
            maps=metadata_maps,
        )

        sample_feature_val = build_sample_feature_table(
            species_labels_split=species_labels_val,
            hospital_split=hospital_val,
            year_norm_split=year_val_norm,
            maps=metadata_maps,
        )

        prior_logit_matrix, antibiotic_prev, antibiotic_nobs, species_antibiotic_nobs = compute_train_prevalence_prior(
            amr_train=amr_train,
            species_train=species_train,
            num_species=num_species,
        )

        train_dataset = LogitEnrichmentRecDataset(
            X=X_train,
            species_ids=species_train,
            amr=amr_train,
            antibiotic_feature_table=antibiotic_feature_table,
            sample_feature_table=sample_feature_train,
            prior_logit_matrix=prior_logit_matrix,
        )

        val_dataset = LogitEnrichmentRecDataset(
            X=X_val,
            species_ids=species_val,
            amr=amr_val,
            antibiotic_feature_table=antibiotic_feature_table,
            sample_feature_table=sample_feature_val,
            prior_logit_matrix=prior_logit_matrix,
        )

        if len(train_dataset) == 0:
            print("Skipping fold: empty train dataset", flush=True)
            continue
        if len(val_dataset) == 0:
            print("Skipping fold: empty val dataset", flush=True)
            continue

        print("Train pairs:", len(train_dataset), flush=True)
        print("Validation pairs:", len(val_dataset), flush=True)

        loader_train = make_loader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        loader_val = make_loader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

        model = LogitEnrichedRecommender(
            num_feat=num_feat,
            num_items=num_items,
            num_species=num_species,
            num_atc2=len(metadata_maps["atc2_to_id"]),
            num_atc3=len(metadata_maps["atc3_to_id"]),
            num_atc4=len(metadata_maps["atc4_to_id"]),
            num_mechanisms=len(metadata_maps["mechanism_to_id"]),
            num_genus=len(metadata_maps["genus_to_id"]),
            num_gram=len(metadata_maps["gram_to_id"]),
            num_hospital=len(metadata_maps["hospital_to_id"]),
            model_mode=model_mode,
            maldi_emb_dim=MALDI_EMB_DIM,
            drug_emb_dim=DRUG_EMB_DIM,
            species_emb_dim=SPECIES_EMB_DIM,
            base_hidden_dims=BASE_HIDDEN_DIMS,
            enrichment_hidden_dims=ENRICHMENT_HIDDEN_DIMS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        print("Base input dim:", model.base_input_dim, flush=True)
        print("Enrichment dim:", model.enrichment_dim, flush=True)
        print("Enrichment input dim:", model.enrichment_input_dim, flush=True)

        global_mapping_payload["fold_mappings"][f"{model_mode}_fold_{fold}"] = {
            "selected_antibiotics": [str(a) for a in selected_antibiotics],
            "valid_cols_original": [int(c) for c in valid_cols],
            "metadata_maps": metadata_maps,
            "year_min_train": float(year_min),
            "year_max_train": float(year_max),
            "antibiotic_train_prevalence": {
                str(selected_antibiotics[j]): float(antibiotic_prev[j])
                for j in range(num_items)
            },
            "antibiotic_train_nobs": {
                str(selected_antibiotics[j]): float(antibiotic_nobs[j])
                for j in range(num_items)
            },
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
                    mode="min",
                )
            ],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            deterministic=False,
        )

        trainer.fit(model, loader_train, loader_val)

        print("Predicting all validation sample-antibiotic pairs...", flush=True)

        preds_val = predict_all_pairs(
            model=model,
            X=X_val,
            species_ids=species_val,
            num_items=num_items,
            antibiotic_feature_table=antibiotic_feature_table,
            sample_feature_table=sample_feature_val,
            prior_logit_matrix=prior_logit_matrix,
            batch_size=PRED_BATCH_SIZE,
        )

        global_micro_auc, macro_antibiotic_auc, antibiotic_aucs = compute_global_metrics(
            y_true=amr_val,
            preds=preds_val,
        )

        species_rows = compute_per_species_metrics(
            model_mode=model_mode,
            fold=fold,
            y_true=amr_val,
            preds=preds_val,
            species_ids=species_val,
            id_to_species=id_to_species,
        )
        species_results.extend(species_rows)

        species_antibiotic_rows = compute_per_species_antibiotic_metrics(
            model_mode=model_mode,
            fold=fold,
            y_true=amr_val,
            preds=preds_val,
            species_ids=species_val,
            id_to_species=id_to_species,
            selected_antibiotics=selected_antibiotics,
        )

        species_antibiotic_results.extend(species_antibiotic_rows)

        species_micro_values = [r["species_micro_auc"] for r in species_rows if np.isfinite(r["species_micro_auc"])]
        species_macro_values = [r["species_macro_antibiotic_auc"] for r in species_rows if np.isfinite(r["species_macro_antibiotic_auc"])]

        mean_species_micro_auc = float(np.mean(species_micro_values)) if len(species_micro_values) > 0 else np.nan
        mean_species_macro_antibiotic_auc = float(np.mean(species_macro_values)) if len(species_macro_values) > 0 else np.nan

        print("Fold global/micro AUC:", global_micro_auc, flush=True)
        print("Fold macro antibiotic AUC:", macro_antibiotic_auc, flush=True)
        print("Fold mean species micro AUC:", mean_species_micro_auc, flush=True)
        print("Fold mean species macro-antibiotic AUC:", mean_species_macro_antibiotic_auc, flush=True)

        fold_results.append({
            "model_mode": model_mode,
            "fold": fold,
            "n_train_samples": X_train.shape[0],
            "n_val_samples": X_val.shape[0],
            "n_antibiotics": num_items,
            "n_train_pairs": len(train_dataset),
            "n_val_pairs": len(val_dataset),
            "base_input_dim": model.base_input_dim,
            "enrichment_dim": model.enrichment_dim,
            "enrichment_input_dim": model.enrichment_input_dim,
            "maldi_emb_dim": MALDI_EMB_DIM,
            "species_emb_dim": SPECIES_EMB_DIM,
            "drug_emb_dim": DRUG_EMB_DIM,
            "global_micro_auc": global_micro_auc,
            "macro_antibiotic_auc": macro_antibiotic_auc,
            "mean_species_micro_auc": mean_species_micro_auc,
            "mean_species_macro_antibiotic_auc": mean_species_macro_antibiotic_auc,
            "antibiotics": ";".join(map(str, selected_antibiotics)),
        })

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
df_species_antibiotic = pd.DataFrame(species_antibiotic_results)

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

        summary_rows.append({
            "model_mode": model_mode,
            "mean_global_micro_auc": df_mode["global_micro_auc"].mean(),
            "std_global_micro_auc": df_mode["global_micro_auc"].std(),
            "mean_macro_antibiotic_auc": df_mode["macro_antibiotic_auc"].mean(),
            "std_macro_antibiotic_auc": df_mode["macro_antibiotic_auc"].std(),
            "mean_species_micro_auc": df_mode["mean_species_micro_auc"].mean(),
            "std_species_micro_auc": df_mode["mean_species_micro_auc"].std(),
            "mean_species_macro_antibiotic_auc": df_mode["mean_species_macro_antibiotic_auc"].mean(),
            "std_species_macro_antibiotic_auc": df_mode["mean_species_macro_antibiotic_auc"].std(),
            "n_folds": len(df_mode),
        })

    df_summary = pd.DataFrame(summary_rows)
else:
    df_summary = pd.DataFrame()

print("\n==================================================")
print("SUMMARY")
print("==================================================")
print(df_summary)

df_species_summary = summarize_species_results(df_species)
df_antibiotic_summary = summarize_antibiotic_results(df_antibiotics)
df_species_antibiotic_summary = summarize_species_antibiotic_results(df_species_antibiotic)

df_antibiotics.to_csv(OUTPUT_ANTIBIOTIC_CSV, index=False)
df_species_antibiotic.to_csv(OUTPUT_SPECIES_ANTIBIOTIC_CSV, index=False)

df_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
df_species_summary.to_csv(OUTPUT_SPECIES_SUMMARY_CSV, index=False)
df_antibiotic_summary.to_csv(OUTPUT_ANTIBIOTIC_SUMMARY_CSV, index=False)
df_species_antibiotic_summary.to_csv(
    OUTPUT_SPECIES_ANTIBIOTIC_SUMMARY_CSV,
    index=False
)

df_metadata_used.to_csv(OUTPUT_METADATA_USED_CSV, index=False)

with open(OUTPUT_MAPPING_JSON, "w") as f:
    json.dump(global_mapping_payload, f, indent=4)

print("\nSaved:")
print(OUTPUT_FOLD_CSV)
print(OUTPUT_SPECIES_CSV)
print(OUTPUT_ANTIBIOTIC_CSV)
print(OUTPUT_SUMMARY_CSV)
print(OUTPUT_SPECIES_SUMMARY_CSV)
print(OUTPUT_ANTIBIOTIC_SUMMARY_CSV)
print(OUTPUT_MAPPING_JSON)
print(OUTPUT_METADATA_USED_CSV)
