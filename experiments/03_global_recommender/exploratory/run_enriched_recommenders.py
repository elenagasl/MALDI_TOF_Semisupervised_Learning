# ============================================================
# ENRICHED GLOBAL RECOMMENDERS FOR MALDI-TOF AMR PREDICTION
#
# Runs two enriched experiments:
#   1) enriched_species_only_head
#   2) enriched_implicit_hypernetwork
#
# Enrichment added to both models:
#   - Antibiotic biological metadata:
#       ATC2, ATC3, ATC4, mechanism, beta-lactamase inhibitor status
#   - Epidemiological train-only prevalence features:
#       computed INSIDE each fold using train only to avoid leakage
#
# GPU optimized:
#   - automatic CUDA detection
#   - mixed precision on CUDA
#   - larger batch sizes by default
#   - pin_memory / persistent workers when using GPU

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
# ANTIBIOTIC METADATA DICTIONARY 
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

try:
    ANTIBIOTIC_METADATA = antibiotic_metadata  # allows running inside a notebook where it already exists
except NameError:
    ANTIBIOTIC_METADATA = {}

if len(ANTIBIOTIC_METADATA) == 0:
    raise RuntimeError(
        "ANTIBIOTIC_METADATA is empty. Paste your antibiotic_metadata dictionary "
        "in the section marked 'PASTE YOUR ANTIBIOTIC METADATA DICTIONARY HERE'."
    )


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl"

OUTPUT_FOLD_CSV = "enriched_recommender_fold_results.csv"
OUTPUT_SPECIES_CSV = "enriched_recommender_per_species.csv"
OUTPUT_ANTIBIOTIC_CSV = "enriched_recommender_per_antibiotic.csv"
OUTPUT_SUMMARY_CSV = "enriched_recommender_summary.csv"
OUTPUT_MAPPING_JSON = "enriched_recommender_mappings.json"

N_SPLITS = 5
RANDOM_STATE = 42

# Antibiotic filtering per fold
MIN_TRAIN_OBS_PER_ANTIBIOTIC = 50
MIN_VAL_OBS_PER_ANTIBIOTIC = 5

# Training
MAX_EPOCHS = 300
PATIENCE = 15
LR = 1e-3

# GPU/CPU execution
USE_CUDA = torch.cuda.is_available()
ACCELERATOR = "gpu" if USE_CUDA else "cpu"
DEVICES = 1
PRECISION = "16-mixed" if USE_CUDA else "32-true"

# Batch sizes. Increase if your GPU has enough memory.
BATCH_SIZE = 1024 if USE_CUDA else 256
VAL_BATCH_SIZE = 2048 if USE_CUDA else 512
PRED_BATCH_SIZE = 1024 if USE_CUDA else 128

# Dataloader
NUM_WORKERS = min(8, os.cpu_count() or 1) if USE_CUDA else 0
PIN_MEMORY = bool(USE_CUDA)
PERSISTENT_WORKERS = bool(NUM_WORKERS > 0)

# Model dimensions
MALDI_EMB_DIM = 32
DRUG_EMB_DIM = 16
SPECIES_EMB_DIM = 16
ATC2_EMB_DIM = 4
ATC3_EMB_DIM = 6
ATC4_EMB_DIM = 8
MECHANISM_EMB_DIM = 6
EPI_EMB_DIM = 16
HYPERNET_HIDDEN_DIM = 64
HIDDEN_DIMS = [128, 64]

# Epidemiological prevalence smoothing
SMOOTH_ALPHA = 20.0

# Models to run
MODEL_MODES = [
    "enriched_species_only_head",
    "enriched_implicit_hypernetwork",
]

# CPU/GPU performance settings
CPU_THREADS = min(8, os.cpu_count() or 1)
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)

if USE_CUDA:
    torch.set_float32_matmul_precision("medium")

pl.seed_everything(RANDOM_STATE, workers=True)

print("==================================================", flush=True)
print("DEVICE CONFIG", flush=True)
print("==================================================", flush=True)
print("CUDA available:", USE_CUDA, flush=True)
if USE_CUDA:
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("Accelerator:", ACCELERATOR, flush=True)
print("Precision:", PRECISION, flush=True)
print("Batch size:", BATCH_SIZE, flush=True)
print("Val batch size:", VAL_BATCH_SIZE, flush=True)
print("Prediction batch size:", PRED_BATCH_SIZE, flush=True)
print("Num workers:", NUM_WORKERS, flush=True)


# ============================================================
# DATASET
# ============================================================

class EnrichedGlobalRecDataset(Dataset):
    """
    Each item:
        (sample, species, antibiotic, epi_features) -> resistance

    Inputs:
        - MALDI spectrum
        - species_id
        - antibiotic_id
        - epidemiological features computed from train only

    Output:
        - AMR label: 0 susceptible, 1 resistant
    """

    def __init__(self, X, species_ids, amr, epi_matrix):
        self.X = np.asarray(X, dtype=np.float32)
        self.species_ids = np.asarray(species_ids, dtype=np.int64)
        self.amr = np.asarray(amr)
        self.epi_matrix = np.asarray(epi_matrix, dtype=np.float32)

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

        species_id = self.species_ids[sample_idx]
        maldi = self.X[sample_idx]
        epi_features = self.epi_matrix[species_id, drug_id]

        return (
            torch.tensor(maldi, dtype=torch.float32),
            torch.tensor(species_id, dtype=torch.long),
            torch.tensor(drug_id, dtype=torch.long),
            torch.tensor(epi_features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


# ============================================================
# METADATA UTILS
# ============================================================

def clean_category(value, unknown_token):
    if value is None:
        return unknown_token
    value = str(value)
    if value.strip() == "" or value.lower() == "none" or value.lower() == "nan":
        return unknown_token
    return value


def get_antibiotic_meta(antibiotic_name):
    """Return metadata dict for an antibiotic, using robust fallback."""
    name = str(antibiotic_name)

    if name in ANTIBIOTIC_METADATA:
        return ANTIBIOTIC_METADATA[name]

    # Sometimes names may have whitespace or accidental variants.
    name_stripped = name.strip()
    if name_stripped in ANTIBIOTIC_METADATA:
        return ANTIBIOTIC_METADATA[name_stripped]

    return {
        "ATC2": None,
        "ATC3": None,
        "ATC4": None,
        "mechanism": None,
        "beta_lactamase_inhibitor": None,
    }


def build_metadata_arrays(antibiotics_all):
    """
    Build categorical vocabularies and per-antibiotic metadata arrays.
    These are built over all antibiotics in the payload, then subset per fold.
    """
    atc2_values = ["UNK_ATC2"]
    atc3_values = ["UNK_ATC3"]
    atc4_values = ["UNK_ATC4"]
    mechanism_values = ["unknown"]

    rows = []

    for ab in antibiotics_all:
        meta = get_antibiotic_meta(ab)

        atc2 = clean_category(meta.get("ATC2"), "UNK_ATC2")
        atc3 = clean_category(meta.get("ATC3"), "UNK_ATC3")
        atc4 = clean_category(meta.get("ATC4"), "UNK_ATC4")
        mechanism = clean_category(meta.get("mechanism"), "unknown")

        beta = meta.get("beta_lactamase_inhibitor")
        beta_known = 1.0 if beta is not None else 0.0
        beta = 0.0 if beta is None else float(beta)

        atc2_values.append(atc2)
        atc3_values.append(atc3)
        atc4_values.append(atc4)
        mechanism_values.append(mechanism)

        rows.append(
            {
                "antibiotic": str(ab),
                "ATC2": atc2,
                "ATC3": atc3,
                "ATC4": atc4,
                "mechanism": mechanism,
                "beta_lactamase_inhibitor": beta,
                "beta_lactamase_known": beta_known,
                "metadata_found": int(str(ab) in ANTIBIOTIC_METADATA),
            }
        )

    atc2_vocab = {v: i for i, v in enumerate(sorted(set(atc2_values)))}
    atc3_vocab = {v: i for i, v in enumerate(sorted(set(atc3_values)))}
    atc4_vocab = {v: i for i, v in enumerate(sorted(set(atc4_values)))}
    mechanism_vocab = {v: i for i, v in enumerate(sorted(set(mechanism_values)))}

    atc2_ids = []
    atc3_ids = []
    atc4_ids = []
    mechanism_ids = []
    beta_values = []
    beta_known_values = []
    metadata_found = []

    for row in rows:
        atc2_ids.append(atc2_vocab[row["ATC2"]])
        atc3_ids.append(atc3_vocab[row["ATC3"]])
        atc4_ids.append(atc4_vocab[row["ATC4"]])
        mechanism_ids.append(mechanism_vocab[row["mechanism"]])
        beta_values.append(row["beta_lactamase_inhibitor"])
        beta_known_values.append(row["beta_lactamase_known"])
        metadata_found.append(row["metadata_found"])

    metadata_arrays = {
        "atc2_ids": np.asarray(atc2_ids, dtype=np.int64),
        "atc3_ids": np.asarray(atc3_ids, dtype=np.int64),
        "atc4_ids": np.asarray(atc4_ids, dtype=np.int64),
        "mechanism_ids": np.asarray(mechanism_ids, dtype=np.int64),
        "beta_values": np.asarray(beta_values, dtype=np.float32),
        "beta_known_values": np.asarray(beta_known_values, dtype=np.float32),
        "metadata_found": np.asarray(metadata_found, dtype=np.int64),
    }

    vocabs = {
        "atc2_vocab": atc2_vocab,
        "atc3_vocab": atc3_vocab,
        "atc4_vocab": atc4_vocab,
        "mechanism_vocab": mechanism_vocab,
    }

    metadata_df = pd.DataFrame(rows)

    return metadata_arrays, vocabs, metadata_df


# ============================================================
# PREVALENCE FEATURES
# ============================================================

def safe_div(num, den, default):
    if den <= 0:
        return default
    return num / den


def count_resistant_and_observed(values):
    obs = values[~np.isnan(values)]
    n_obs = len(obs)
    n_res = int(np.sum(obs == 1)) if n_obs > 0 else 0
    return n_res, n_obs


def smooth_prevalence(n_res, n_obs, prior, alpha=20.0):
    return float((n_res + alpha * prior) / (n_obs + alpha))


def compute_train_prevalence_features(
    amr_train,
    species_train,
    num_species,
    selected_atc3_ids,
    selected_atc4_ids,
    selected_mechanism_ids,
    alpha=20.0,
):
    """
    Compute epidemiological features using TRAIN ONLY.

    Returns:
        epi_matrix: [num_species, num_items, epi_dim]

    Feature order:
        0  prev_antibiotic
        1  prev_species
        2  prev_species_antibiotic_smooth
        3  prev_mechanism
        4  prev_species_mechanism_smooth
        5  prev_atc3
        6  prev_species_atc3_smooth
        7  prev_atc4
        8  prev_species_atc4_smooth
        9  log_n_obs_species_antibiotic_norm
        10 log_n_obs_antibiotic_norm
        11 log_n_obs_species_norm
    """
    amr_train = np.asarray(amr_train)
    species_train = np.asarray(species_train, dtype=np.int64)

    n_samples, num_items = amr_train.shape

    all_obs = amr_train[~np.isnan(amr_train)]
    global_prior = float(np.mean(all_obs == 1)) if len(all_obs) > 0 else 0.5

    # --------------------------------------------------------
    # Antibiotic prevalence
    # --------------------------------------------------------
    prev_antibiotic = np.zeros(num_items, dtype=np.float32)
    n_obs_antibiotic = np.zeros(num_items, dtype=np.float32)

    for j in range(num_items):
        n_res, n_obs = count_resistant_and_observed(amr_train[:, j])
        n_obs_antibiotic[j] = n_obs
        prev_antibiotic[j] = smooth_prevalence(n_res, n_obs, global_prior, alpha)

    # --------------------------------------------------------
    # Species prevalence
    # --------------------------------------------------------
    prev_species = np.zeros(num_species, dtype=np.float32)
    n_obs_species = np.zeros(num_species, dtype=np.float32)

    for sp in range(num_species):
        rows = species_train == sp
        if not np.any(rows):
            prev_species[sp] = global_prior
            n_obs_species[sp] = 0
            continue

        vals = amr_train[rows, :]
        n_res, n_obs = count_resistant_and_observed(vals.reshape(-1))
        n_obs_species[sp] = n_obs
        prev_species[sp] = smooth_prevalence(n_res, n_obs, global_prior, alpha)

    # --------------------------------------------------------
    # Species-antibiotic prevalence
    # --------------------------------------------------------
    prev_species_antibiotic = np.zeros((num_species, num_items), dtype=np.float32)
    n_obs_species_antibiotic = np.zeros((num_species, num_items), dtype=np.float32)

    for sp in range(num_species):
        rows = species_train == sp
        for j in range(num_items):
            vals = amr_train[rows, j] if np.any(rows) else np.asarray([])
            n_res, n_obs = count_resistant_and_observed(vals)
            n_obs_species_antibiotic[sp, j] = n_obs
            prev_species_antibiotic[sp, j] = smooth_prevalence(
                n_res,
                n_obs,
                prior=float(prev_antibiotic[j]),
                alpha=alpha,
            )

    # --------------------------------------------------------
    # Group-level prevalence: mechanism, ATC3, ATC4
    # --------------------------------------------------------
    def compute_group_prevalence(group_ids):
        unique_groups = sorted(set(map(int, group_ids)))
        group_to_index = {g: i for i, g in enumerate(unique_groups)}
        n_groups = len(unique_groups)

        prev_group = np.zeros(n_groups, dtype=np.float32)
        n_obs_group = np.zeros(n_groups, dtype=np.float32)

        prev_species_group = np.zeros((num_species, n_groups), dtype=np.float32)
        n_obs_species_group = np.zeros((num_species, n_groups), dtype=np.float32)

        for g in unique_groups:
            g_idx = group_to_index[g]
            drug_mask = np.asarray(group_ids) == g

            vals = amr_train[:, drug_mask]
            n_res, n_obs = count_resistant_and_observed(vals.reshape(-1))
            n_obs_group[g_idx] = n_obs
            prev_group[g_idx] = smooth_prevalence(n_res, n_obs, global_prior, alpha)

            for sp in range(num_species):
                rows = species_train == sp
                if not np.any(rows):
                    prev_species_group[sp, g_idx] = prev_group[g_idx]
                    continue

                vals_sp = amr_train[rows, :][:, drug_mask]
                n_res_sp, n_obs_sp = count_resistant_and_observed(vals_sp.reshape(-1))
                n_obs_species_group[sp, g_idx] = n_obs_sp
                prev_species_group[sp, g_idx] = smooth_prevalence(
                    n_res_sp,
                    n_obs_sp,
                    prior=float(prev_group[g_idx]),
                    alpha=alpha,
                )

        group_index_per_drug = np.asarray([group_to_index[int(g)] for g in group_ids], dtype=np.int64)

        return prev_group, prev_species_group, group_index_per_drug

    prev_mechanism, prev_species_mechanism, mechanism_index_per_drug = compute_group_prevalence(
        selected_mechanism_ids
    )

    prev_atc3, prev_species_atc3, atc3_index_per_drug = compute_group_prevalence(
        selected_atc3_ids
    )

    prev_atc4, prev_species_atc4, atc4_index_per_drug = compute_group_prevalence(
        selected_atc4_ids
    )

    # --------------------------------------------------------
    # Log count normalization
    # --------------------------------------------------------
    max_log_sa = max(1.0, float(np.max(np.log1p(n_obs_species_antibiotic))))
    max_log_ant = max(1.0, float(np.max(np.log1p(n_obs_antibiotic))))
    max_log_sp = max(1.0, float(np.max(np.log1p(n_obs_species))))

    epi_dim = 12
    epi_matrix = np.zeros((num_species, num_items, epi_dim), dtype=np.float32)

    for sp in range(num_species):
        for j in range(num_items):
            mech_idx = mechanism_index_per_drug[j]
            atc3_idx = atc3_index_per_drug[j]
            atc4_idx = atc4_index_per_drug[j]

            epi_matrix[sp, j, 0] = prev_antibiotic[j]
            epi_matrix[sp, j, 1] = prev_species[sp]
            epi_matrix[sp, j, 2] = prev_species_antibiotic[sp, j]
            epi_matrix[sp, j, 3] = prev_mechanism[mech_idx]
            epi_matrix[sp, j, 4] = prev_species_mechanism[sp, mech_idx]
            epi_matrix[sp, j, 5] = prev_atc3[atc3_idx]
            epi_matrix[sp, j, 6] = prev_species_atc3[sp, atc3_idx]
            epi_matrix[sp, j, 7] = prev_atc4[atc4_idx]
            epi_matrix[sp, j, 8] = prev_species_atc4[sp, atc4_idx]
            epi_matrix[sp, j, 9] = np.log1p(n_obs_species_antibiotic[sp, j]) / max_log_sa
            epi_matrix[sp, j, 10] = np.log1p(n_obs_antibiotic[j]) / max_log_ant
            epi_matrix[sp, j, 11] = np.log1p(n_obs_species[sp]) / max_log_sp

    return epi_matrix


# ============================================================
# MODEL COMPONENTS
# ============================================================

class EnrichedAntibioticEncoder(nn.Module):
    """
    Antibiotic encoder enriched with biological metadata.

    Input:
        drug_id [batch_size]

    Uses registered per-drug metadata arrays:
        ATC2, ATC3, ATC4, mechanism, beta-lactamase inhibitor

    Output:
        enriched antibiotic embedding
    """

    def __init__(
        self,
        num_items,
        num_atc2,
        num_atc3,
        num_atc4,
        num_mechanisms,
        selected_atc2_ids,
        selected_atc3_ids,
        selected_atc4_ids,
        selected_mechanism_ids,
        selected_beta_values,
        selected_beta_known_values,
        drug_emb_dim=16,
        atc2_emb_dim=4,
        atc3_emb_dim=6,
        atc4_emb_dim=8,
        mechanism_emb_dim=6,
    ):
        super().__init__()

        self.drug_embedding = nn.Embedding(num_items, drug_emb_dim)
        self.atc2_embedding = nn.Embedding(num_atc2, atc2_emb_dim)
        self.atc3_embedding = nn.Embedding(num_atc3, atc3_emb_dim)
        self.atc4_embedding = nn.Embedding(num_atc4, atc4_emb_dim)
        self.mechanism_embedding = nn.Embedding(num_mechanisms, mechanism_emb_dim)

        self.register_buffer(
            "atc2_ids",
            torch.tensor(selected_atc2_ids, dtype=torch.long),
        )
        self.register_buffer(
            "atc3_ids",
            torch.tensor(selected_atc3_ids, dtype=torch.long),
        )
        self.register_buffer(
            "atc4_ids",
            torch.tensor(selected_atc4_ids, dtype=torch.long),
        )
        self.register_buffer(
            "mechanism_ids",
            torch.tensor(selected_mechanism_ids, dtype=torch.long),
        )
        self.register_buffer(
            "beta_values",
            torch.tensor(selected_beta_values, dtype=torch.float32),
        )
        self.register_buffer(
            "beta_known_values",
            torch.tensor(selected_beta_known_values, dtype=torch.float32),
        )

        self.output_dim = (
            drug_emb_dim
            + atc2_emb_dim
            + atc3_emb_dim
            + atc4_emb_dim
            + mechanism_emb_dim
            + 2
        )

    def forward(self, drug_id):
        drug_id = drug_id.long()

        drug_emb = self.drug_embedding(drug_id)

        atc2_emb = self.atc2_embedding(self.atc2_ids[drug_id])
        atc3_emb = self.atc3_embedding(self.atc3_ids[drug_id])
        atc4_emb = self.atc4_embedding(self.atc4_ids[drug_id])
        mechanism_emb = self.mechanism_embedding(self.mechanism_ids[drug_id])

        beta = self.beta_values[drug_id].view(-1, 1)
        beta_known = self.beta_known_values[drug_id].view(-1, 1)

        return torch.cat(
            [
                drug_emb,
                atc2_emb,
                atc3_emb,
                atc4_emb,
                mechanism_emb,
                beta,
                beta_known,
            ],
            dim=-1,
        )


class EpidemiologyEncoder(nn.Module):
    """Small MLP for train-only prevalence features."""

    def __init__(self, epi_input_dim, epi_emb_dim=16):
        super().__init__()
        self.output_dim = epi_emb_dim
        self.net = nn.Sequential(
            nn.Linear(epi_input_dim, epi_emb_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, epi_features):
        return self.net(epi_features.float())


class GlobalMALDIEncoder(nn.Module):
    """Global MALDI encoder shared by all species."""

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


class SpeciesOnlyMALDIHeadEncoder(nn.Module):
    """
    Species-only head model:
        MALDI -> shared backbone -> species-specific head -> MALDI embedding

    This is the enriched version of your previous species_only model.
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
                    nn.GELU(),
                )
                for _ in range(num_species)
            ]
        )

    def forward(self, maldi, species_id):
        z = self.backbone(maldi.float())
        h_species = torch.zeros(
            (z.shape[0], self.output_dim),
            dtype=z.dtype,
            device=z.device,
        )

        unique_species = torch.unique(species_id.long())

        for sp in unique_species:
            sp_int = int(sp.item())
            mask = species_id.long() == sp
            h_species[mask] = self.species_heads[sp_int](z[mask])

        return h_species


class SpeciesHypernetwork(nn.Module):
    """species_embedding -> species-specific residual correction."""

    def __init__(self, species_emb_dim=16, maldi_emb_dim=32, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(species_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, maldi_emb_dim),
        )

    def forward(self, species_emb):
        return self.net(species_emb)


# ============================================================
# MODELS
# ============================================================

class EnrichedSpeciesOnlyHeadNCF(pl.LightningModule):
    """
    Enriched species-only head recommender.

    Input to final MLP:
        species-adapted MALDI embedding
        + species embedding
        + enriched antibiotic embedding
        + epidemiological train-only prevalence embedding
    """

    def __init__(
        self,
        num_feat,
        num_items,
        num_species,
        metadata_config,
        epi_input_dim,
        maldi_emb_dim=32,
        species_emb_dim=16,
        hidden_dims=[128, 64],
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["metadata_config"])
        self.lr = lr

        self.maldi_encoder = SpeciesOnlyMALDIHeadEncoder(
            input_dim=num_feat,
            num_species=num_species,
            maldi_emb_dim=maldi_emb_dim,
        )

        self.species_embedding = nn.Embedding(num_species, species_emb_dim)

        self.antibiotic_encoder = EnrichedAntibioticEncoder(
            num_items=num_items,
            **metadata_config,
        )

        self.epi_encoder = EpidemiologyEncoder(
            epi_input_dim=epi_input_dim,
            epi_emb_dim=EPI_EMB_DIM,
        )

        fusion_input_dim = (
            maldi_emb_dim
            + species_emb_dim
            + self.antibiotic_encoder.output_dim
            + self.epi_encoder.output_dim
        )

        self.fusion_input_dim = fusion_input_dim

        sizes = [fusion_input_dim] + list(hidden_dims) + [1]
        layers = []

        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.mlp = nn.Sequential(*layers)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, maldi, species_id, drug_id, epi_features):
        maldi_emb = self.maldi_encoder(maldi, species_id)
        species_emb = self.species_embedding(species_id.long())
        antibiotic_emb = self.antibiotic_encoder(drug_id.long())
        epi_emb = self.epi_encoder(epi_features.float())

        x = torch.cat(
            [maldi_emb, species_emb, antibiotic_emb, epi_emb],
            dim=-1,
        )

        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        maldi, species_id, drug_id, epi_features, labels = batch
        logits = self.forward(maldi, species_id, drug_id, epi_features)
        labels = labels.view(-1, 1).float()
        loss = self.loss_fn(logits, labels)
        self.log("loss_tr", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        maldi, species_id, drug_id, epi_features, labels = batch
        logits = self.forward(maldi, species_id, drug_id, epi_features)
        labels = labels.view(-1, 1).float()
        loss = self.loss_fn(logits, labels)
        self.log("loss_val", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class EnrichedImplicitHypernetworkNCF(pl.LightningModule):
    """
    Enriched implicit species hypernetwork recommender.

    MALDI branch:
        MALDI -> global MALDI encoder -> z_maldi

    Species branch:
        species embedding -> hypernetwork -> delta_species
        z_final = z_maldi + alpha * delta_species

    Final MLP input:
        z_final
        + species embedding
        + enriched antibiotic embedding
        + epidemiological train-only prevalence embedding
    """

    def __init__(
        self,
        num_feat,
        num_items,
        num_species,
        metadata_config,
        epi_input_dim,
        maldi_emb_dim=32,
        species_emb_dim=16,
        hypernet_hidden_dim=64,
        hidden_dims=[128, 64],
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["metadata_config"])
        self.lr = lr

        self.maldi_encoder = GlobalMALDIEncoder(
            input_dim=num_feat,
            maldi_emb_dim=maldi_emb_dim,
        )

        self.species_embedding = nn.Embedding(num_species, species_emb_dim)

        self.species_hypernetwork = SpeciesHypernetwork(
            species_emb_dim=species_emb_dim,
            maldi_emb_dim=maldi_emb_dim,
            hidden_dim=hypernet_hidden_dim,
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.antibiotic_encoder = EnrichedAntibioticEncoder(
            num_items=num_items,
            **metadata_config,
        )

        self.epi_encoder = EpidemiologyEncoder(
            epi_input_dim=epi_input_dim,
            epi_emb_dim=EPI_EMB_DIM,
        )

        fusion_input_dim = (
            maldi_emb_dim
            + species_emb_dim
            + self.antibiotic_encoder.output_dim
            + self.epi_encoder.output_dim
        )

        self.fusion_input_dim = fusion_input_dim

        sizes = [fusion_input_dim] + list(hidden_dims) + [1]
        layers = []

        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.mlp = nn.Sequential(*layers)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, maldi, species_id, drug_id, epi_features):
        z_maldi = self.maldi_encoder(maldi)
        species_emb = self.species_embedding(species_id.long())
        delta_species = self.species_hypernetwork(species_emb)
        z_final = z_maldi + self.alpha * delta_species

        antibiotic_emb = self.antibiotic_encoder(drug_id.long())
        epi_emb = self.epi_encoder(epi_features.float())

        x = torch.cat(
            [z_final, species_emb, antibiotic_emb, epi_emb],
            dim=-1,
        )

        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        maldi, species_id, drug_id, epi_features, labels = batch
        logits = self.forward(maldi, species_id, drug_id, epi_features)
        labels = labels.view(-1, 1).float()
        loss = self.loss_fn(logits, labels)
        self.log("loss_tr", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("alpha", self.alpha.detach(), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        maldi, species_id, drug_id, epi_features, labels = batch
        logits = self.forward(maldi, species_id, drug_id, epi_features)
        labels = labels.view(-1, 1).float()
        loss = self.loss_fn(logits, labels)
        self.log("loss_val", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("alpha_val", self.alpha.detach(), prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ============================================================
# GENERAL UTILS
# ============================================================

def clean_amr(amr):
    """Keep only 0, 1 and NaN. Everything else becomes NaN."""
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
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
    )


def count_observed_with_two_classes(col):
    obs = col[~np.isnan(col)]
    n_obs = len(obs)
    has_two_classes = n_obs > 0 and len(np.unique(obs)) > 1
    return n_obs, has_two_classes


def select_valid_antibiotics(amr_train, amr_val):
    """
    Select antibiotics valid in a fold.

    Criteria:
        - enough observed labels in train
        - both classes present in train
        - enough observed labels in validation
        - both classes present in validation
    """
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


def predict_all_pairs(model, X, species_ids, num_items, epi_matrix, batch_size=128):
    """
    Predict resistance probabilities for all validation sample-antibiotic pairs.

    Output:
        preds_matrix [n_samples, num_items]
    """
    model.eval()
    device = model.device

    X = np.asarray(X, dtype=np.float32)
    species_ids = np.asarray(species_ids, dtype=np.int64)
    n_samples = X.shape[0]

    preds_matrix = np.zeros((n_samples, num_items), dtype=np.float32)

    epi_tensor = torch.tensor(epi_matrix, dtype=torch.float32, device=device)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            maldi_batch = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=device,
            )

            species_batch = torch.tensor(
                species_ids[start:end],
                dtype=torch.long,
                device=device,
            )

            batch_size_real = maldi_batch.shape[0]
            drug_preds = []

            for drug_id in range(num_items):
                drug_batch = torch.full(
                    size=(batch_size_real,),
                    fill_value=drug_id,
                    dtype=torch.long,
                    device=device,
                )

                epi_batch = epi_tensor[species_batch, drug_id]

                logits = model(
                    maldi=maldi_batch,
                    species_id=species_batch,
                    drug_id=drug_batch,
                    epi_features=epi_batch,
                )

                probs = torch.sigmoid(logits).view(-1)
                drug_preds.append(probs)

            drug_preds = torch.stack(drug_preds, dim=1)
            preds_matrix[start:end] = drug_preds.cpu().numpy()

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


def compute_per_species_metrics(model_mode, fold, y_true, preds, species_ids, id_to_species):
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
                "auc": auc,
            }
        )

    return rows


def compute_per_antibiotic_metrics(model_mode, fold, y_true, preds, selected_antibiotics):
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
                "auc": auc,
            }
        )

    return rows


def make_metadata_config_for_fold(metadata_arrays, valid_cols, vocabs):
    """Subset metadata arrays to selected antibiotics for this fold."""
    return {
        "num_atc2": len(vocabs["atc2_vocab"]),
        "num_atc3": len(vocabs["atc3_vocab"]),
        "num_atc4": len(vocabs["atc4_vocab"]),
        "num_mechanisms": len(vocabs["mechanism_vocab"]),
        "selected_atc2_ids": metadata_arrays["atc2_ids"][valid_cols],
        "selected_atc3_ids": metadata_arrays["atc3_ids"][valid_cols],
        "selected_atc4_ids": metadata_arrays["atc4_ids"][valid_cols],
        "selected_mechanism_ids": metadata_arrays["mechanism_ids"][valid_cols],
        "selected_beta_values": metadata_arrays["beta_values"][valid_cols],
        "selected_beta_known_values": metadata_arrays["beta_known_values"][valid_cols],
        "drug_emb_dim": DRUG_EMB_DIM,
        "atc2_emb_dim": ATC2_EMB_DIM,
        "atc3_emb_dim": ATC3_EMB_DIM,
        "atc4_emb_dim": ATC4_EMB_DIM,
        "mechanism_emb_dim": MECHANISM_EMB_DIM,
    }


def instantiate_model(model_mode, num_feat, num_items, num_species, metadata_config, epi_dim):
    if model_mode == "enriched_species_only_head":
        return EnrichedSpeciesOnlyHeadNCF(
            num_feat=num_feat,
            num_items=num_items,
            num_species=num_species,
            metadata_config=metadata_config,
            epi_input_dim=epi_dim,
            maldi_emb_dim=MALDI_EMB_DIM,
            species_emb_dim=SPECIES_EMB_DIM,
            hidden_dims=HIDDEN_DIMS,
            lr=LR,
        )

    if model_mode == "enriched_implicit_hypernetwork":
        return EnrichedImplicitHypernetworkNCF(
            num_feat=num_feat,
            num_items=num_items,
            num_species=num_species,
            metadata_config=metadata_config,
            epi_input_dim=epi_dim,
            maldi_emb_dim=MALDI_EMB_DIM,
            species_emb_dim=SPECIES_EMB_DIM,
            hypernet_hidden_dim=HYPERNET_HIDDEN_DIM,
            hidden_dims=HIDDEN_DIMS,
            lr=LR,
        )

    raise ValueError(f"Unknown model_mode: {model_mode}")


# ============================================================
# LOAD DATA
# ============================================================

print("\n==================================================", flush=True)
print("LOADING DATA", flush=True)
print("==================================================", flush=True)

with open(DATA_PATH, "rb") as f:
    payload = pickle.load(f)

X_all = np.asarray(payload["data"], dtype=np.float32)
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
species_to_id = {sp: i for i, sp in enumerate(unique_species)}
id_to_species = {i: sp for sp, i in species_to_id.items()}

species_ids_all = np.array(
    [species_to_id[sp] for sp in species_labels],
    dtype=np.int64,
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

print("\nBuilding antibiotic metadata arrays...", flush=True)
metadata_arrays, vocabs, metadata_df = build_metadata_arrays(antibiotics_all)

print("Metadata found for", int(metadata_arrays["metadata_found"].sum()), "/", len(antibiotics_all), "antibiotics", flush=True)
print("ATC2 vocab:", vocabs["atc2_vocab"], flush=True)
print("ATC3 vocab size:", len(vocabs["atc3_vocab"]), flush=True)
print("ATC4 vocab size:", len(vocabs["atc4_vocab"]), flush=True)
print("Mechanism vocab:", vocabs["mechanism_vocab"], flush=True)

metadata_df.to_csv("enriched_antibiotic_metadata_used.csv", index=False)

mapping_payload = {
    "species_to_id": {str(k): int(v) for k, v in species_to_id.items()},
    "id_to_species": {str(k): str(v) for k, v in id_to_species.items()},
    "antibiotics": [str(a) for a in antibiotics_all],
    "modes": MODEL_MODES,
    "vocabs": vocabs,
    "model_config": {
        "MALDI_EMB_DIM": MALDI_EMB_DIM,
        "DRUG_EMB_DIM": DRUG_EMB_DIM,
        "SPECIES_EMB_DIM": SPECIES_EMB_DIM,
        "ATC2_EMB_DIM": ATC2_EMB_DIM,
        "ATC3_EMB_DIM": ATC3_EMB_DIM,
        "ATC4_EMB_DIM": ATC4_EMB_DIM,
        "MECHANISM_EMB_DIM": MECHANISM_EMB_DIM,
        "EPI_EMB_DIM": EPI_EMB_DIM,
        "HYPERNET_HIDDEN_DIM": HYPERNET_HIDDEN_DIM,
        "SMOOTH_ALPHA": SMOOTH_ALPHA,
    },
}

with open(OUTPUT_MAPPING_JSON, "w") as f:
    json.dump(mapping_payload, f, indent=4)

print("Saved mappings to:", OUTPUT_MAPPING_JSON, flush=True)
print("Saved metadata table to: enriched_antibiotic_metadata_used.csv", flush=True)


# ============================================================
# 5-FOLD TRAINING FOR BOTH ENRICHED MODELS
# ============================================================

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_results = []
species_results = []
antibiotic_results = []

sample_indices = np.arange(n_samples)

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
            amr_val=amr_val_source,
        )

        if len(valid_cols) == 0:
            print("Skipping fold: no valid antibiotics", flush=True)
            continue

        selected_antibiotics = antibiotics_all[valid_cols]

        X_train = np.asarray(X_train_source, dtype=np.float32)
        X_val = np.asarray(X_val_source, dtype=np.float32)

        species_train = np.asarray(species_train_source, dtype=np.int64)
        species_val = np.asarray(species_val_source, dtype=np.int64)

        amr_train = amr_train_source[:, valid_cols]
        amr_val = amr_val_source[:, valid_cols]

        num_items = amr_train.shape[1]

        print("Valid antibiotics:", num_items, flush=True)
        print("Antibiotics:", list(selected_antibiotics), flush=True)

        metadata_config = make_metadata_config_for_fold(
            metadata_arrays=metadata_arrays,
            valid_cols=valid_cols,
            vocabs=vocabs,
        )

        print("Computing train-only epidemiological prevalence features...", flush=True)
        epi_matrix = compute_train_prevalence_features(
            amr_train=amr_train,
            species_train=species_train,
            num_species=num_species,
            selected_atc3_ids=metadata_config["selected_atc3_ids"],
            selected_atc4_ids=metadata_config["selected_atc4_ids"],
            selected_mechanism_ids=metadata_config["selected_mechanism_ids"],
            alpha=SMOOTH_ALPHA,
        )

        epi_dim = epi_matrix.shape[-1]
        print("Epidemiology feature dim:", epi_dim, flush=True)

        train_dataset = EnrichedGlobalRecDataset(
            X=X_train,
            species_ids=species_train,
            amr=amr_train,
            epi_matrix=epi_matrix,
        )

        val_dataset = EnrichedGlobalRecDataset(
            X=X_val,
            species_ids=species_val,
            amr=amr_val,
            epi_matrix=epi_matrix,
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
            shuffle=True,
        )

        loader_val = make_loader(
            dataset=val_dataset,
            batch_size=VAL_BATCH_SIZE,
            shuffle=False,
        )

        model = instantiate_model(
            model_mode=model_mode,
            num_feat=num_feat,
            num_items=num_items,
            num_species=num_species,
            metadata_config=metadata_config,
            epi_dim=epi_dim,
        )

        print("Fusion input dim:", model.fusion_input_dim, flush=True)
        if hasattr(model, "alpha"):
            print("Initial alpha:", float(model.alpha.detach().cpu()), flush=True)

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
            benchmark=USE_CUDA,
        )

        trainer.fit(model, loader_train, loader_val)

        final_alpha = np.nan
        if hasattr(model, "alpha"):
            final_alpha = float(model.alpha.detach().cpu())
            print("Final alpha:", final_alpha, flush=True)

        print("Predicting all validation sample-antibiotic pairs...", flush=True)
        preds_val = predict_all_pairs(
            model=model,
            X=X_val,
            species_ids=species_val,
            num_items=num_items,
            epi_matrix=epi_matrix,
            batch_size=PRED_BATCH_SIZE,
        )

        auc_global, auc_macro_antibiotic, antibiotic_aucs = compute_global_metrics(
            y_true=amr_val,
            preds=preds_val,
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
                "maldi_emb_dim": MALDI_EMB_DIM,
                "species_emb_dim": SPECIES_EMB_DIM,
                "drug_emb_dim": DRUG_EMB_DIM,
                "epi_dim": epi_dim,
                "epi_emb_dim": EPI_EMB_DIM,
                "final_alpha": final_alpha,
                "global_auc": auc_global,
                "macro_antibiotic_auc": auc_macro_antibiotic,
                "antibiotics": ";".join(map(str, selected_antibiotics)),
            }
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

        antibiotic_rows = compute_per_antibiotic_metrics(
            model_mode=model_mode,
            fold=fold,
            y_true=amr_val,
            preds=preds_val,
            selected_antibiotics=selected_antibiotics,
        )
        antibiotic_results.extend(antibiotic_rows)

        # Save partial results after every fold, so nothing is lost if the job stops.
        pd.DataFrame(fold_results).to_csv(OUTPUT_FOLD_CSV, index=False)
        pd.DataFrame(species_results).to_csv(OUTPUT_SPECIES_CSV, index=False)
        pd.DataFrame(antibiotic_results).to_csv(OUTPUT_ANTIBIOTIC_CSV, index=False)

        del model
        del trainer
        del loader_train
        del loader_val
        del train_dataset
        del val_dataset
        del preds_val
        gc.collect()

        if USE_CUDA:
            torch.cuda.empty_cache()


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

df_folds = pd.DataFrame(fold_results)
df_species = pd.DataFrame(species_results)
df_antibiotics = pd.DataFrame(antibiotic_results)

print("\n==================================================", flush=True)
print("FINAL FOLD RESULTS", flush=True)
print("==================================================", flush=True)
print(df_folds, flush=True)

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
                "mean_final_alpha": df_mode["final_alpha"].mean(),
                "std_final_alpha": df_mode["final_alpha"].std(),
                "n_folds": len(df_mode),
            }
        )

    df_summary = pd.DataFrame(summary_rows)

    print("\n==================================================", flush=True)
    print("SUMMARY", flush=True)
    print("==================================================", flush=True)
    print(df_summary, flush=True)

else:
    df_summary = pd.DataFrame()

# Additional aggregated tables
if len(df_species) > 0:
    df_species_summary = (
        df_species
        .groupby(["model_mode", "species"], as_index=False)
        .agg(
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            mean_n_val_pairs=("n_val_pairs", "mean"),
            n_folds=("fold", "nunique"),
        )
    )
else:
    df_species_summary = pd.DataFrame()

if len(df_antibiotics) > 0:
    df_antibiotic_summary = (
        df_antibiotics
        .groupby(["model_mode", "antibiotic"], as_index=False)
        .agg(
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            mean_n_val_pairs=("n_val_pairs", "mean"),
            n_folds=("fold", "nunique"),
        )
    )
else:
    df_antibiotic_summary = pd.DataFrame()

df_folds.to_csv(OUTPUT_FOLD_CSV, index=False)
df_species.to_csv(OUTPUT_SPECIES_CSV, index=False)
df_antibiotics.to_csv(OUTPUT_ANTIBIOTIC_CSV, index=False)
df_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
df_species_summary.to_csv("enriched_recommender_per_species_summary.csv", index=False)
df_antibiotic_summary.to_csv("enriched_recommender_per_antibiotic_summary.csv", index=False)

print("\nSaved:", flush=True)
print(OUTPUT_FOLD_CSV, flush=True)
print(OUTPUT_SPECIES_CSV, flush=True)
print(OUTPUT_ANTIBIOTIC_CSV, flush=True)
print(OUTPUT_SUMMARY_CSV, flush=True)
print(OUTPUT_MAPPING_JSON, flush=True)
print("enriched_antibiotic_metadata_used.csv", flush=True)
print("enriched_recommender_per_species_summary.csv", flush=True)
print("enriched_recommender_per_antibiotic_summary.csv", flush=True)
