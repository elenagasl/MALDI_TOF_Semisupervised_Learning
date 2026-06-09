# ============================================================
# GLOBAL SPECIES-CONDITIONED RECOMMENDER
# + BIOLOGICAL / EPIDEMIOLOGICAL / CLINICAL METADATA ENCODER
# ============================================================

import pickle
import gc
import json
import math
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
from sklearn.cluster import AgglomerativeClustering
from pytorch_lightning.callbacks import EarlyStopping


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl"

OUTPUT_FOLD_CSV = "global_metadata_recommender_5fold_results.csv"
OUTPUT_SPECIES_CSV = "global_metadata_recommender_5fold_per_species.csv"
OUTPUT_ANTIBIOTIC_CSV = "global_metadata_recommender_5fold_per_antibiotic.csv"
OUTPUT_MAPPING_JSON = "global_metadata_recommender_mappings.json"
OUTPUT_METADATA_COLUMNS_JSON = "global_metadata_recommender_metadata_columns.json"

N_SPLITS = 5
RANDOM_STATE = 42

MIN_TRAIN_OBS_PER_ANTIBIOTIC = 50
MIN_VAL_OBS_PER_ANTIBIOTIC = 5

BATCH_SIZE = 64
VAL_BATCH_SIZE = 128
MAX_EPOCHS = 300
PATIENCE = 15
LR = 1e-3

DRUG_EMB_DIM = 16
SPECIES_EMB_DIM = 16
METADATA_EMB_DIM = 32
HIDDEN_DIMS = [128, 64]

PREVALENCE_ALPHA = 10.0
N_CORR_CLUSTERS = 8

DEVICE = "gpu"
torch.set_num_threads(8)

if torch.cuda.is_available():
    print("CUDA available:", torch.cuda.get_device_name(0), flush=True)
else:
    print("WARNING: CUDA is not available. Training will fail if accelerator='gpu'.", flush=True)

print("Using device:", DEVICE, flush=True)


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

try:
    antibiotic_metadata
except NameError:
    antibiotic_metadata = {}
    print(
        "\nWARNING: antibiotic_metadata dictionary is empty. "
        "Paste your dictionary in the marked section to use ATC/mechanism metadata.\n",
        flush=True
    )


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
    x = str(x)
    x = x.strip()
    if x == "" or x.lower() == "nan" or x.lower() == "none":
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


def is_beta_lactam_from_atc(atc3, atc4, drug_name):
    atc3 = clean_name(atc3)
    atc4 = clean_name(atc4)
    name = clean_name(drug_name).lower()

    beta_lactam_keywords = [
        "penicillin", "ampicillin", "amoxicillin", "piperacillin",
        "ticarcillin", "oxacillin", "benzylpenicillin",
        "cef", "ceft", "ceph", "cefoxitin", "cefuroxime",
        "meropenem", "imipenem", "ertapenem", "doripenem",
        "aztreonam"
    ]

    if atc3 in ["J01C", "J01D"]:
        return 1

    if any(k in name for k in beta_lactam_keywords):
        return 1

    return 0


def get_beta_lactam_subclass(drug_name, atc3, atc4):
    name = clean_name(drug_name).lower()
    atc3 = clean_name(atc3)
    atc4 = clean_name(atc4)

    if is_beta_lactam_from_atc(atc3, atc4, drug_name) == 0:
        return "not_beta_lactam"

    if "aztreonam" in name or atc4 == "J01DF":
        return "monobactam"

    if "meropenem" in name or "imipenem" in name or "ertapenem" in name or atc4 == "J01DH":
        return "carbapenem"

    if (
        "cef" in name
        or "ceft" in name
        or "ceph" in name
        or atc3 == "J01D"
    ):
        return "cephalosporin"

    if (
        "amoxicillin" in name
        or "ampicillin" in name
        or "piperacillin" in name
        or "ticarcillin" in name
        or "penicillin" in name
        or "oxacillin" in name
        or atc3 == "J01C"
    ):
        if "-" in name:
            return "penicillin_combination"
        return "penicillin"

    return "beta_lactam_other"


def get_cephalosporin_generation(drug_name, atc4):
    name = clean_name(drug_name).lower()
    atc4 = clean_name(atc4)

    if not (
        "cef" in name
        or "ceft" in name
        or "ceph" in name
        or atc4 in ["J01DB", "J01DC", "J01DD", "J01DE", "J01DI"]
    ):
        return "none"

    first_gen = ["cefazolin", "cefalotin", "cephalexin"]
    second_gen = ["cefuroxime", "cefoxitin", "cefaclor"]
    third_gen = ["ceftriaxone", "cefotaxime", "ceftazidime", "cefixime", "cefpodoxime"]
    fourth_gen = ["cefepime"]
    fifth_gen = ["ceftarolin", "ceftobiprole", "ceftolozane"]

    if any(k in name for k in first_gen) or atc4 == "J01DB":
        return "generation_1"

    if any(k in name for k in second_gen) or atc4 == "J01DC":
        return "generation_2"

    if any(k in name for k in third_gen) or atc4 == "J01DD":
        return "generation_3"

    if any(k in name for k in fourth_gen) or atc4 == "J01DE":
        return "generation_4"

    if any(k in name for k in fifth_gen) or atc4 == "J01DI":
        return "generation_5"

    return "generation_unknown"


# ============================================================
# BASIC UTILS
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
        num_workers=4,
        pin_memory=True,
		persistent_workers=True
    )


def count_observed_with_two_classes(col):
    obs = col[~np.isnan(col)]
    n_obs = len(obs)
    has_two_classes = False

    if n_obs > 0 and len(np.unique(obs)) > 1:
        has_two_classes = True

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


# ============================================================
# ONE-HOT METADATA ENCODING
# ============================================================

def build_category_maps(
    selected_antibiotics,
    species_labels_all,
    hospital_all,
    year_all
):
    atc2_values = ["unknown"]
    atc3_values = ["unknown"]
    atc4_values = ["unknown"]
    mechanism_values = ["unknown"]
    subclass_values = ["unknown"]
    generation_values = ["unknown"]

    for drug in selected_antibiotics:
        drug = str(drug)
        meta = antibiotic_metadata.get(drug, {})

        atc2 = clean_name(meta.get("ATC2", "unknown"))
        atc3 = clean_name(meta.get("ATC3", "unknown"))
        atc4 = clean_name(meta.get("ATC4", "unknown"))
        mechanism = clean_name(meta.get("mechanism", "unknown"))

        subclass = get_beta_lactam_subclass(drug, atc3, atc4)
        generation = get_cephalosporin_generation(drug, atc4)

        atc2_values.append(atc2)
        atc3_values.append(atc3)
        atc4_values.append(atc4)
        mechanism_values.append(mechanism)
        subclass_values.append(subclass)
        generation_values.append(generation)

    genus_values = ["unknown"]
    gram_values = ["unknown"]

    for sp in species_labels_all:
        genus_values.append(get_genus(sp))
        gram_values.append(get_gram_status(sp))

    hospital_values = ["unknown"]

    for h in hospital_all:
        hospital_values.append(clean_name(h))

    year_values = np.asarray(year_all)

    maps = {
        "ATC2": {v: i for i, v in enumerate(sorted(set(atc2_values)))},
        "ATC3": {v: i for i, v in enumerate(sorted(set(atc3_values)))},
        "ATC4": {v: i for i, v in enumerate(sorted(set(atc4_values)))},
        "mechanism": {v: i for i, v in enumerate(sorted(set(mechanism_values)))},
        "beta_lactam_subclass": {v: i for i, v in enumerate(sorted(set(subclass_values)))},
        "generation": {v: i for i, v in enumerate(sorted(set(generation_values)))},
        "genus": {v: i for i, v in enumerate(sorted(set(genus_values)))},
        "gram": {v: i for i, v in enumerate(sorted(set(gram_values)))},
        "hospital": {v: i for i, v in enumerate(sorted(set(hospital_values)))},
    }

    finite_years = year_values[np.isfinite(year_values)]

    if len(finite_years) == 0:
        maps["year_min"] = 0.0
        maps["year_max"] = 1.0
    else:
        maps["year_min"] = float(np.min(finite_years))
        maps["year_max"] = float(np.max(finite_years))

    return maps


def one_hot(value, mapping):
    vec = np.zeros(len(mapping), dtype=np.float32)
    value = clean_name(value)

    if value not in mapping:
        value = "unknown"

    idx = mapping.get(value, mapping.get("unknown", 0))
    vec[idx] = 1.0

    return vec


def get_antibiotic_static_metadata(drug_name, category_maps):
    drug_name = str(drug_name)
    meta = antibiotic_metadata.get(drug_name, {})

    atc2 = clean_name(meta.get("ATC2", "unknown"))
    atc3 = clean_name(meta.get("ATC3", "unknown"))
    atc4 = clean_name(meta.get("ATC4", "unknown"))
    mechanism = clean_name(meta.get("mechanism", "unknown"))

    beta_lactamase_inhibitor = meta.get("beta_lactamase_inhibitor", 0)

    if beta_lactamase_inhibitor is None:
        beta_lactamase_inhibitor = 0

    beta_lactamase_inhibitor = float(beta_lactamase_inhibitor)

    is_beta_lactam = float(is_beta_lactam_from_atc(atc3, atc4, drug_name))
    subclass = get_beta_lactam_subclass(drug_name, atc3, atc4)
    generation = get_cephalosporin_generation(drug_name, atc4)

    parts = [
        one_hot(atc2, category_maps["ATC2"]),
        one_hot(atc3, category_maps["ATC3"]),
        one_hot(atc4, category_maps["ATC4"]),
        one_hot(mechanism, category_maps["mechanism"]),
        np.asarray([beta_lactamase_inhibitor], dtype=np.float32),
        np.asarray([is_beta_lactam], dtype=np.float32),
        one_hot(subclass, category_maps["beta_lactam_subclass"]),
        one_hot(generation, category_maps["generation"]),
    ]

    return np.concatenate(parts).astype(np.float32)


def get_species_static_metadata(species_label, category_maps):
    genus = get_genus(species_label)
    gram = get_gram_status(species_label)

    parts = [
        one_hot(genus, category_maps["genus"]),
        one_hot(gram, category_maps["gram"]),
    ]

    return np.concatenate(parts).astype(np.float32)


def normalize_year(year_value, category_maps):
    if year_value is None:
        return 0.0

    try:
        y = float(year_value)
    except Exception:
        return 0.0

    if not np.isfinite(y):
        return 0.0

    y_min = category_maps["year_min"]
    y_max = category_maps["year_max"]

    if y_max <= y_min:
        return 0.0

    return float((y - y_min) / (y_max - y_min))


def get_clinical_static_metadata(hospital_value, year_value, category_maps):
    h = clean_name(hospital_value)

    parts = [
        one_hot(h, category_maps["hospital"]),
        np.asarray([normalize_year(year_value, category_maps)], dtype=np.float32),
    ]

    return np.concatenate(parts).astype(np.float32)


# ============================================================
# EPIDEMIOLOGICAL METADATA FROM TRAIN ONLY
# ============================================================

def compute_train_prevalence_features(amr_train, species_train, num_species, num_items):
    observed = ~np.isnan(amr_train)

    resistant = (amr_train == 1)

    n_obs_global = int(np.sum(observed))
    n_res_global = int(np.sum(resistant))

    if n_obs_global > 0:
        global_prev = n_res_global / n_obs_global
    else:
        global_prev = 0.5

    antibiotic_prev = np.zeros(num_items, dtype=np.float32)
    antibiotic_nobs = np.zeros(num_items, dtype=np.float32)

    for j in range(num_items):
        obs_j = observed[:, j]
        n_j = int(np.sum(obs_j))
        r_j = int(np.sum(resistant[:, j]))

        antibiotic_nobs[j] = n_j

        if n_j > 0:
            antibiotic_prev[j] = r_j / n_j
        else:
            antibiotic_prev[j] = global_prev

    species_prev = np.zeros(num_species, dtype=np.float32)
    species_nobs = np.zeros(num_species, dtype=np.float32)

    for sp in range(num_species):
        sp_mask = species_train == sp

        obs_sp = observed[sp_mask]
        res_sp = resistant[sp_mask]

        n_sp = int(np.sum(obs_sp))
        r_sp = int(np.sum(res_sp))

        species_nobs[sp] = n_sp

        if n_sp > 0:
            species_prev[sp] = r_sp / n_sp
        else:
            species_prev[sp] = global_prev

    species_antibiotic_prev = np.zeros((num_species, num_items), dtype=np.float32)
    species_antibiotic_nobs = np.zeros((num_species, num_items), dtype=np.float32)

    for sp in range(num_species):
        sp_mask = species_train == sp

        for j in range(num_items):
            obs = observed[sp_mask, j]
            res = resistant[sp_mask, j]

            n = int(np.sum(obs))
            r = int(np.sum(res))

            species_antibiotic_nobs[sp, j] = n

            prior = antibiotic_prev[j]

            smoothed = (r + PREVALENCE_ALPHA * prior) / (n + PREVALENCE_ALPHA)
            species_antibiotic_prev[sp, j] = smoothed

    stats = {
        "global_prev": float(global_prev),
        "antibiotic_prev": antibiotic_prev,
        "antibiotic_nobs": antibiotic_nobs,
        "species_prev": species_prev,
        "species_nobs": species_nobs,
        "species_antibiotic_prev": species_antibiotic_prev,
        "species_antibiotic_nobs": species_antibiotic_nobs,
    }

    return stats


def get_epi_metadata(species_id, drug_id, epi_stats):
    global_prev = epi_stats["global_prev"]

    antibiotic_prev = epi_stats["antibiotic_prev"][drug_id]
    antibiotic_nobs = epi_stats["antibiotic_nobs"][drug_id]

    species_prev = epi_stats["species_prev"][species_id]
    species_nobs = epi_stats["species_nobs"][species_id]

    sp_ab_prev = epi_stats["species_antibiotic_prev"][species_id, drug_id]
    sp_ab_nobs = epi_stats["species_antibiotic_nobs"][species_id, drug_id]

    vec = np.asarray(
        [
            global_prev,
            antibiotic_prev,
            species_prev,
            sp_ab_prev,
            np.log1p(antibiotic_nobs),
            np.log1p(species_nobs),
            np.log1p(sp_ab_nobs),
        ],
        dtype=np.float32
    )

    return vec


# ============================================================
# CORRELATION CLUSTER FROM TRAIN ONLY
# ============================================================

def compute_antibiotic_correlation_clusters(amr_train, n_clusters):
    num_items = amr_train.shape[1]

    if num_items <= 1:
        return np.zeros(num_items, dtype=np.int64)

    filled = amr_train.copy()

    col_means = np.nanmean(filled, axis=0)
    global_mean = np.nanmean(filled)

    if not np.isfinite(global_mean):
        global_mean = 0.5

    col_means = np.where(np.isfinite(col_means), col_means, global_mean)

    inds = np.where(np.isnan(filled))
    filled[inds] = np.take(col_means, inds[1])

    corr = np.corrcoef(filled.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    distance = 1.0 - np.abs(corr)
    np.fill_diagonal(distance, 0.0)

    k = min(n_clusters, num_items)

    if k <= 1:
        return np.zeros(num_items, dtype=np.int64)

    try:
        model = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average"
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=k,
            affinity="precomputed",
            linkage="average"
        )

    labels = model.fit_predict(distance)

    return labels.astype(np.int64)


def one_hot_cluster(cluster_id, n_clusters):
    vec = np.zeros(n_clusters, dtype=np.float32)

    cluster_id = int(cluster_id)

    if cluster_id < 0 or cluster_id >= n_clusters:
        cluster_id = 0

    vec[cluster_id] = 1.0

    return vec


# ============================================================
# METADATA MATRIX BUILDER
# ============================================================

def build_metadata_matrix(
    selected_antibiotics,
    species_labels,
    species_ids,
    hospital,
    year,
    category_maps,
    epi_stats,
    corr_clusters,
    n_corr_clusters
):
    n_samples = len(species_ids)
    num_items = len(selected_antibiotics)

    antibiotic_static = []

    for drug in selected_antibiotics:
        antibiotic_static.append(
            get_antibiotic_static_metadata(drug, category_maps)
        )

    antibiotic_static = np.asarray(antibiotic_static, dtype=np.float32)

    species_static_by_id = {}

    for i in range(n_samples):
        sp_id = int(species_ids[i])

        if sp_id not in species_static_by_id:
            species_static_by_id[sp_id] = get_species_static_metadata(
                species_labels[i],
                category_maps
            )

    metadata_rows = []

    for i in range(n_samples):
        sp_id = int(species_ids[i])

        species_vec = species_static_by_id[sp_id]

        clinical_vec = get_clinical_static_metadata(
            hospital[i],
            year[i],
            category_maps
        )

        for j in range(num_items):
            ab_vec = antibiotic_static[j]

            epi_vec = get_epi_metadata(
                species_id=sp_id,
                drug_id=j,
                epi_stats=epi_stats
            )

            cluster_vec = one_hot_cluster(
                corr_clusters[j],
                n_corr_clusters
            )

            row = np.concatenate(
                [
                    ab_vec,
                    species_vec,
                    epi_vec,
                    cluster_vec,
                    clinical_vec,
                ]
            ).astype(np.float32)

            metadata_rows.append(row)

    metadata_matrix = np.asarray(metadata_rows, dtype=np.float32)

    metadata_matrix = np.nan_to_num(
        metadata_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return metadata_matrix


def make_metadata_columns(category_maps, n_corr_clusters):
    cols = []

    for block in ["ATC2", "ATC3", "ATC4", "mechanism"]:
        for value in category_maps[block]:
            cols.append(f"drug_{block}_{value}")

    cols.append("drug_beta_lactamase_inhibitor")
    cols.append("drug_is_beta_lactam")

    for value in category_maps["beta_lactam_subclass"]:
        cols.append(f"drug_beta_lactam_subclass_{value}")

    for value in category_maps["generation"]:
        cols.append(f"drug_generation_{value}")

    for value in category_maps["genus"]:
        cols.append(f"species_genus_{value}")

    for value in category_maps["gram"]:
        cols.append(f"species_gram_{value}")

    cols.extend(
        [
            "epi_global_prev_train",
            "epi_antibiotic_prev_train",
            "epi_species_prev_train",
            "epi_species_antibiotic_prev_train_smoothed",
            "epi_log1p_antibiotic_nobs_train",
            "epi_log1p_species_nobs_train",
            "epi_log1p_species_antibiotic_nobs_train",
        ]
    )

    for k in range(n_corr_clusters):
        cols.append(f"epi_corr_cluster_{k}")

    for value in category_maps["hospital"]:
        cols.append(f"clinical_hospital_{value}")

    cols.append("clinical_year_norm")

    return cols


# ============================================================
# DATASET
# ============================================================

class GlobalMetadataRecDataset(Dataset):
    """
    Each item:

        (sample, species, antibiotic, metadata) -> resistance

    The metadata vector is indexed by:
        metadata_index = sample_idx * num_items + drug_id
    """

    def __init__(self, X, species_ids, amr, metadata_matrix):
        self.X = np.asarray(X)
        self.species_ids = np.asarray(species_ids)
        self.amr = np.asarray(amr)
        self.metadata_matrix = np.asarray(metadata_matrix)

        self.num_items = self.amr.shape[1]

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
        label = self.labels[idx]

        maldi = self.X[sample_idx]
        species_id = self.species_ids[sample_idx]

        metadata_idx = sample_idx * self.num_items + drug_id
        metadata_vec = self.metadata_matrix[metadata_idx]

        return (
            torch.tensor(maldi).float(),
            torch.tensor(species_id).long(),
            torch.tensor(drug_id).long(),
            torch.tensor(metadata_vec).float(),
            torch.tensor(label).float()
        )


# ============================================================
# MODEL
# ============================================================

class MALDIEncoder(nn.Module):
    """
    MALDI spectrum -> 512 -> 128 -> 64 -> 32
    """

    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x.float())


class MetadataEncoder(nn.Module):
    """
    Metadata vector -> compact metadata embedding
    """

    def __init__(self, metadata_dim, metadata_emb_dim=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, metadata_emb_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x.float())


class GlobalMetadataSpeciesNCF(pl.LightningModule):
    """
    Global species-conditioned recommender enriched with metadata.

    Inputs:
        - MALDI spectrum
        - species_id
        - drug_id
        - metadata vector

    Prediction:
        P(resistant)
    """

    def __init__(
        self,
        num_feat,
        num_items,
        num_species,
        metadata_dim,
        drug_emb_dim=16,
        species_emb_dim=16,
        metadata_emb_dim=32,
        hidden_dims=[128, 64],
        lr=1e-3
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr

        self.maldi_encoder = MALDIEncoder(num_feat)

        self.drug_embedding = nn.Embedding(num_items, drug_emb_dim)

        self.species_embedding = nn.Embedding(num_species, species_emb_dim)

        self.metadata_encoder = MetadataEncoder(
            metadata_dim=metadata_dim,
            metadata_emb_dim=metadata_emb_dim
        )

        fusion_input_dim = (
            32
            + drug_emb_dim
            + species_emb_dim
            + metadata_emb_dim
        )

        sizes = [fusion_input_dim] + list(hidden_dims) + [1]

        layers = []

        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        self.mlp = nn.Sequential(*layers)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, maldi, species_id, drug_id, metadata_vec):
        maldi_emb = self.maldi_encoder(maldi)

        species_emb = self.species_embedding(species_id.long())

        drug_emb = self.drug_embedding(drug_id.long())

        metadata_emb = self.metadata_encoder(metadata_vec)

        x = torch.cat(
            [
                maldi_emb,
                species_emb,
                drug_emb,
                metadata_emb
            ],
            dim=-1
        )

        logits = self.mlp(x)

        return logits

    def training_step(self, batch, batch_idx):
        maldi, species_id, drug_id, metadata_vec, labels = batch

        logits = self.forward(
            maldi,
            species_id,
            drug_id,
            metadata_vec
        )

        labels = labels.view(-1, 1).float()

        loss = self.loss_fn(logits, labels)

        self.log("loss_tr", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        maldi, species_id, drug_id, metadata_vec, labels = batch

        logits = self.forward(
            maldi,
            species_id,
            drug_id,
            metadata_vec
        )

        labels = labels.view(-1, 1).float()

        loss = self.loss_fn(logits, labels)

        self.log("loss_val", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ============================================================
# PREDICTION AND METRICS
# ============================================================

def predict_all_pairs(
    model,
    X,
    species_ids,
    metadata_matrix,
    num_items,
    batch_size=128
):
    model.eval()

    device = model.device

    X = np.asarray(X)
    species_ids = np.asarray(species_ids)
    metadata_matrix = np.asarray(metadata_matrix)

    n_samples = X.shape[0]

    preds_matrix = np.zeros((n_samples, num_items), dtype=np.float32)

    with torch.no_grad():

        for start in range(0, n_samples, batch_size):

            end = min(start + batch_size, n_samples)

            maldi_batch = torch.tensor(X[start:end]).float().to(device)

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

                meta_indices = np.arange(start, end) * num_items + drug_id

                metadata_batch = torch.tensor(
                    metadata_matrix[meta_indices],
                    dtype=torch.float32,
                    device=device
                )

                logits = model(
                    maldi_batch,
                    species_batch,
                    drug_batch,
                    metadata_batch
                )

                probs = torch.sigmoid(logits).view(-1)

                drug_preds.append(probs)

            drug_preds = torch.stack(drug_preds, dim=1)

            preds_matrix[start:end] = drug_preds.cpu().numpy()

    return preds_matrix


def compute_global_metrics(y_true, preds):
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
                "fold": fold,
                "antibiotic_id": j,
                "antibiotic": str(antibiotic_name),
                "n_val_pairs": n_pairs,
                "auc": auc
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

print("\nHospital/source distribution:")
unique_hospitals, hospital_counts = np.unique(hospital_all, return_counts=True)

for h, c in zip(unique_hospitals, hospital_counts):
    print(f"Hospital/source {h}: {c} samples", flush=True)

print("\nSpecies distribution:")
unique_sp_ids, sp_counts = np.unique(species_ids_all, return_counts=True)

for sp_id, c in zip(unique_sp_ids, sp_counts):
    print(f"{id_to_species[int(sp_id)]}: {c} samples", flush=True)

mapping_payload = {
    "species_to_id": {str(k): int(v) for k, v in species_to_id.items()},
    "id_to_species": {str(k): str(v) for k, v in id_to_species.items()},
    "antibiotics": [str(a) for a in antibiotics_all]
}

with open(OUTPUT_MAPPING_JSON, "w") as f:
    json.dump(mapping_payload, f, indent=4)

print("\nSaved mappings to:", OUTPUT_MAPPING_JSON, flush=True)


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

metadata_columns_saved = False

sample_indices = np.arange(n_samples)

for fold, (train_idx, val_idx) in enumerate(kf.split(sample_indices)):

    print("\n==================================================", flush=True)
    print(f"FOLD {fold + 1}/{N_SPLITS}", flush=True)
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
        amr_val=amr_val_source
    )

    if len(valid_cols) == 0:
        print("Skipping fold: no valid antibiotics", flush=True)
        continue

    selected_antibiotics = antibiotics_all[valid_cols]

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

    num_items = amr_train.shape[1]

    print("Valid antibiotics:", num_items, flush=True)
    print("Antibiotics:", list(selected_antibiotics), flush=True)

    print("Building category maps...", flush=True)

    category_maps = build_category_maps(
        selected_antibiotics=selected_antibiotics,
        species_labels_all=species_labels,
        hospital_all=hospital_all,
        year_all=year_all
    )

    print("Computing train-only epidemiological features...", flush=True)

    epi_stats = compute_train_prevalence_features(
        amr_train=amr_train,
        species_train=species_train,
        num_species=num_species,
        num_items=num_items
    )

    print("Computing train-only antibiotic correlation clusters...", flush=True)

    corr_clusters = compute_antibiotic_correlation_clusters(
        amr_train=amr_train,
        n_clusters=N_CORR_CLUSTERS
    )

    actual_n_corr_clusters = N_CORR_CLUSTERS

    print("Building metadata matrices...", flush=True)

    metadata_train = build_metadata_matrix(
        selected_antibiotics=selected_antibiotics,
        species_labels=species_labels_train,
        species_ids=species_train,
        hospital=hospital_train,
        year=year_train,
        category_maps=category_maps,
        epi_stats=epi_stats,
        corr_clusters=corr_clusters,
        n_corr_clusters=actual_n_corr_clusters
    )

    metadata_val = build_metadata_matrix(
        selected_antibiotics=selected_antibiotics,
        species_labels=species_labels_val,
        species_ids=species_val,
        hospital=hospital_val,
        year=year_val,
        category_maps=category_maps,
        epi_stats=epi_stats,
        corr_clusters=corr_clusters,
        n_corr_clusters=actual_n_corr_clusters
    )

    metadata_dim = metadata_train.shape[1]

    print("Metadata dim:", metadata_dim, flush=True)

    if not metadata_columns_saved:
        metadata_columns = make_metadata_columns(
            category_maps=category_maps,
            n_corr_clusters=actual_n_corr_clusters
        )

        with open(OUTPUT_METADATA_COLUMNS_JSON, "w") as f:
            json.dump(metadata_columns, f, indent=4)

        print("Saved metadata columns to:", OUTPUT_METADATA_COLUMNS_JSON, flush=True)

        metadata_columns_saved = True

    train_dataset = GlobalMetadataRecDataset(
        X=X_train,
        species_ids=species_train,
        amr=amr_train,
        metadata_matrix=metadata_train
    )

    val_dataset = GlobalMetadataRecDataset(
        X=X_val,
        species_ids=species_val,
        amr=amr_val,
        metadata_matrix=metadata_val
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

    model = GlobalMetadataSpeciesNCF(
        num_feat=num_feat,
        num_items=num_items,
        num_species=num_species,
        metadata_dim=metadata_dim,
        drug_emb_dim=DRUG_EMB_DIM,
        species_emb_dim=SPECIES_EMB_DIM,
        metadata_emb_dim=METADATA_EMB_DIM,
        hidden_dims=HIDDEN_DIMS,
        lr=LR
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor="loss_val",
                patience=PATIENCE,
                mode="min"
            )
        ],
        logger=False,
        enable_checkpointing=False
    )

    trainer.fit(model, loader_train, loader_val)

    preds_val = predict_all_pairs(
        model=model,
        X=X_val,
        species_ids=species_val,
        metadata_matrix=metadata_val,
        num_items=num_items,
        batch_size=128
    )

    auc_global, auc_macro_antibiotic, antibiotic_aucs = compute_global_metrics(
        y_true=amr_val,
        preds=preds_val
    )

    print("Fold global AUC:", auc_global, flush=True)
    print("Fold macro antibiotic AUC:", auc_macro_antibiotic, flush=True)

    fold_results.append(
        {
            "fold": fold,
            "n_train_samples": X_train.shape[0],
            "n_val_samples": X_val.shape[0],
            "n_antibiotics": num_items,
            "n_train_pairs": len(train_dataset),
            "n_val_pairs": len(val_dataset),
            "metadata_dim": metadata_dim,
            "global_auc": auc_global,
            "macro_antibiotic_auc": auc_macro_antibiotic,
            "antibiotics": ";".join(map(str, selected_antibiotics))
        }
    )

    species_rows = compute_per_species_metrics(
        fold=fold,
        y_true=amr_val,
        preds=preds_val,
        species_ids=species_val,
        id_to_species=id_to_species
    )

    species_results.extend(species_rows)

    antibiotic_rows = compute_per_antibiotic_metrics(
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
    del metadata_train
    del metadata_val

    gc.collect()


# ============================================================
# SAVE RESULTS
# ============================================================

df_folds = pd.DataFrame(fold_results)
df_species = pd.DataFrame(species_results)
df_antibiotics = pd.DataFrame(antibiotic_results)

print("\n==================================================")
print("FINAL 5-FOLD GLOBAL METADATA RESULTS")
print("==================================================")
print(df_folds)

if len(df_folds) > 0:
    print("\nMean global AUC:", df_folds["global_auc"].mean())
    print("Std global AUC:", df_folds["global_auc"].std())

    print("\nMean macro antibiotic AUC:", df_folds["macro_antibiotic_auc"].mean())
    print("Std macro antibiotic AUC:", df_folds["macro_antibiotic_auc"].std())

df_folds.to_csv(OUTPUT_FOLD_CSV, index=False)
df_species.to_csv(OUTPUT_SPECIES_CSV, index=False)
df_antibiotics.to_csv(OUTPUT_ANTIBIOTIC_CSV, index=False)

print("\nSaved:")
print(OUTPUT_FOLD_CSV)
print(OUTPUT_SPECIES_CSV)
print(OUTPUT_ANTIBIOTIC_CSV)
print(OUTPUT_MAPPING_JSON)
print(OUTPUT_METADATA_COLUMNS_JSON)