#!/usr/bin/env python3
# ============================================================
# Resistance-oriented VAE for MALDI-TOF AMR prediction
# Multi-species training + early stopping by validation AUC
# ------------------------------------------------------------
# Correct architecture:
#   x -> q_phi(w | x)
#   w -> p_theta(y | w)      AMR prediction
#   w -> p_theta(x | w)      spectrum reconstruction
#
# Outputs:
#   figuras/ : figures per species
#   aucs/    : per-species AUC CSVs + final summary CSVs
# ============================================================

import os
import re
import copy
import pickle
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.manifold import TSNE


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/src/VAE_Generative_AI")
DATA_PATH = Path("/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS.pkl")

FIG_DIR = BASE_DIR / "figuras"
AUC_DIR = BASE_DIR / "aucs"

FIG_DIR.mkdir(parents=True, exist_ok=True)
AUC_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SPECIES AND ANTIBIOTICS
# ============================================================

species_list = [
    ("Staphylococcus", "Aureus"),
    ("Staphylococcus", "Epidermidis"),
    ("Escherichia", "Coli"),
    ("Klebsiella", "Pneumoniae"),
    ("Pseudomonas", "Aeruginosa"),
    ("Enterobacter", "Cloacae"),
    ("Proteus", "Mirabilis"),
    ("Staphylococcus", "Hominis"),
    ("Serratia", "Marcescens"),
    ("Staphylococcus", "Capitis"),
    ("Enterococcus", "Faecium"),
    ("Klebsiella", "Oxytoca"),
    ("Klebsiella", "Variicola"),
    ("Citrobacter", "Koseri"),
    ("Enterococcus", "Faecalis"),
    ("Staphylococcus", "Lugdunensis"),
    ("Citrobacter", "Freundii"),
    ("Morganella", "Morganii"),
    ("Proteus", "Vulgaris"),
    ("Staphylococcus", "Haemolyticus"),
    ("Candida", "Albicans"),
    ("Streptococcus", "Pneumoniae"),
    ("Stenotrophomonas", "Maltophilia"),
    ("Campylobacter", "Jejuni"),
    ("Haemophilus", "Influenzae"),
]

amr_antibiotics = [
    '5-Fluorocytosine','Amikacin','Amoxicillin','Amoxicillin-Clavulanic acid',
    'Amoxicillin-Clavulanic acid_uncomplicated_HWI','Amphotericin B','Ampicillin',
    'Ampicillin-Sulbactam','Anidulafungin','Azithromycin','Aztreonam','Bacitracin',
    'Benzylpenicillin','Benzylpenicillin_others','Benzylpenicillin_with_meningitis',
    'Benzylpenicillin_with_pneumonia','Caspofungin','Cefalotin-Cefazolin','Cefazolin',
    'Cefepime','Cefixime','Cefotaxime','Cefoxitin','Cefoxitin_screen','Cefpodoxime',
    'Ceftarolin','Ceftazidime','Ceftazidime-Avibactam','Ceftobiprole',
    'Ceftolozane-Tazobactam','Ceftriaxone','Cefuroxime','Cefuroxime.1',
    'Chloramphenicol','Ciprofloxacin','Clarithromycin','Clindamycin',
    'Clindamycin_induced','Colistin','Cotrimoxazol','Cotrimoxazole','Daptomycin',
    'Doxycycline','Ertapenem','Erythromycin','Ethambutol_5mg-l','Fluconazole',
    'Fosfomycin','Fusidic acid','Gentamicin','Gentamicin_high_level','Imipenem',
    'Isavuconazole','Isoniazid_.1mg-l','Isoniazid_.4mg-l','Itraconazole',
    'Levofloxacin','Linezolid','MRSA','Meropenem','Meropenem-Vaborbactam',
    'Meropenem_with_meningitis','Meropenem_with_pneumonia',
    'Meropenem_without_meningitis','Metronidazole','Micafungin','Minocycline',
    'Moxifloxacin','Mupirocin','Nitrofurantoin','Norfloxacin','Novobiocin',
    'Ofloxacin','Oxacillin','Pefloxacin','Penicillin',
    'Penicillin_with_endokarditis','Penicillin_with_meningitis',
    'Penicillin_with_other_infections','Penicillin_with_pneumonia',
    'Penicillin_without_endokarditis','Penicillin_without_meningitis',
    'Piperacillin','Piperacillin-Tazobactam','Polymyxin B','Posaconazole',
    'Pristinamycin','Pyrazinamide','Rifampicin','Rifampicin_1mg-l',
    'Sparfloxacin','Strepomycin_high_level','Streptomycin','Teicoplanin',
    'Teicoplanin_GRD','Telithromycin','Tetracycline','Ticarcillin',
    'Ticarcillin-Clavulan acid','Tigecycline','Tobramycin','Vancomycin',
    'Vancomycin_GRD','Voriconazole'
]


# ============================================================
# CONFIG
# ============================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K_REMOVE = 0
EPS = 1e-8

LATENT_DIM_W = 64
ENC_H1 = 512
ENC_H2 = 128
DEC_Y_H = 64
DEC_X_H1 = 256
DEC_X_H2 = 1024

BATCH_SIZE = 64
MAX_EPOCHS = 100
LR = 1e-3

TEST_SIZE = 0.20
VAL_SIZE_FROM_TRAIN = 0.20

EARLY_STOP_PATIENCE = 12
EARLY_STOP_MIN_DELTA = 1e-4

LAMBDA_AMR = 5.0
LAMBDA_SPEC = 0.05
BETA_W = 1e-4
KL_WARMUP = 30
USE_FOCAL_LOSS = False

MIN_SAMPLES_PER_SPECIES = 30
MIN_OBSERVED_AMR_VALUES = 20

DO_TSNE = True
TSNE_MAX_SAMPLES = 1200
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# UTILS
# ============================================================

def safe_filename(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_species_name(text):
    text = str(text)
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def species_key(genus, species):
    return f"{genus}_{species}"


def species_mask(labels, genus, species):
    target = normalize_species_name(species_key(genus, species))
    normalized = np.array([normalize_species_name(v) for v in labels])
    return normalized == target


def remove_top_k_peaks_per_spectrum(X, top_k=1):
    X_new = X.copy()
    if top_k <= 0:
        return X_new
    for i in range(X_new.shape[0]):
        idx = np.argpartition(X_new[i], -top_k)[-top_k:]
        X_new[i, idx] = 0
    return X_new


def kl_anneal(epoch, warmup=30):
    return min(1.0, epoch / max(1, warmup))


def prepare_y_target(y):
    """
    Prepare AMR targets for masked loss.

    y is used only as target.
    It is never passed into the model.
    """
    y_mask = (~torch.isnan(y)).float()
    y_filled = torch.nan_to_num(y, nan=0.0)
    return y_mask, y_filled


# ============================================================
# DATASET
# ============================================================

class ResistanceSpectrumDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ============================================================
# MODEL
# ============================================================

class ResistanceOrientedVAE(nn.Module):
    def __init__(self, input_dim, num_antibiotics):
        super().__init__()

        self.enc_w = nn.Sequential(
            nn.Linear(input_dim, ENC_H1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ENC_H1, ENC_H2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.w_mu = nn.Linear(ENC_H2, LATENT_DIM_W)
        self.w_logvar = nn.Linear(ENC_H2, LATENT_DIM_W)

        self.dec_y = nn.Sequential(
            nn.Linear(LATENT_DIM_W, DEC_Y_H),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(DEC_Y_H, num_antibiotics),
        )

        self.dec_x = nn.Sequential(
            nn.Linear(LATENT_DIM_W, DEC_X_H1),
            nn.ReLU(),
            nn.Linear(DEC_X_H1, DEC_X_H2),
            nn.ReLU(),
        )

        self.presence_head = nn.Linear(DEC_X_H2, input_dim)
        self.mu_head = nn.Linear(DEC_X_H2, input_dim)
        self.sig_head = nn.Linear(DEC_X_H2, input_dim)

    def reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, x):
        h = self.enc_w(x)

        w_mu = self.w_mu(h)
        w_logvar = self.w_logvar(h)
        w = self.reparam(w_mu, w_logvar)

        y_logits = self.dec_y(w)

        hx = self.dec_x(w)
        pres_logits = self.presence_head(hx)
        mu_x = self.mu_head(hx)
        sig_x = F.softplus(self.sig_head(hx)) + 1e-4

        return {
            "w": w,
            "w_mu": w_mu,
            "w_logvar": w_logvar,
            "y_logits": y_logits,
            "pres_logits": pres_logits,
            "mu_x": mu_x,
            "sig_x": sig_x,
        }


# ============================================================
# LOSSES
# ============================================================

def kl_normal(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def masked_bce_with_logits(logits, targets, mask):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def focal_loss_masked(logits, targets, mask, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p)
    alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
    loss = -alpha_t * (1 - pt).pow(gamma) * torch.log(pt + EPS)
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def spectrum_hurdle_loss(x, pres_logits, mu_x, sig_x):
    peak_mask = (x > 0).float()

    bce_presence = F.binary_cross_entropy_with_logits(
        pres_logits,
        peak_mask,
        reduction="none",
    )

    gaussian_nll = ((x - mu_x) ** 2) / (2 * sig_x ** 2) + torch.log(sig_x)
    loss = bce_presence + gaussian_nll * peak_mask
    return loss.mean()


# ============================================================
# METRICS
# ============================================================

def safe_auc(y_true, y_score):
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def compute_aucs(y_true, y_score, antibiotic_names):
    rows = []

    for j, ab in enumerate(antibiotic_names):
        yt = y_true[:, j]
        ys = y_score[:, j]
        mask = ~np.isnan(yt)

        n_obs = int(mask.sum())
        n_pos = int(np.nansum(yt[mask] == 1))
        n_neg = int(np.nansum(yt[mask] == 0))
        auc = safe_auc(yt, ys)

        rows.append({
            "antibiotic": ab,
            "auc": auc,
            "n_obs": n_obs,
            "n_pos": n_pos,
            "n_neg": n_neg,
        })

    per_antibiotic = pd.DataFrame(rows)
    micro_auc = safe_auc(y_true.reshape(-1), y_score.reshape(-1))

    if per_antibiotic["auc"].notna().any():
        macro_auc = float(np.nanmean(per_antibiotic["auc"].values))
    else:
        macro_auc = np.nan

    return per_antibiotic, micro_auc, macro_auc


@torch.no_grad()
def collect_outputs(model, loader):
    model.eval()

    y_true_all, y_prob_all = [], []
    w_mu_all, x_true_all, x_mu_all = [], [], []

    for x, y in loader:
        x = x.to(DEVICE)

        # IMPORTANT: y is not passed into the model.
        out = model(x)
        y_prob = torch.sigmoid(out["y_logits"])

        y_true_all.append(y.numpy())
        y_prob_all.append(y_prob.cpu().numpy())
        w_mu_all.append(out["w_mu"].cpu().numpy())
        x_true_all.append(x.cpu().numpy())
        x_mu_all.append(out["mu_x"].cpu().numpy())

    return {
        "y_true": np.concatenate(y_true_all, axis=0),
        "y_prob": np.concatenate(y_prob_all, axis=0),
        "w_mu": np.concatenate(w_mu_all, axis=0),
        "x_true": np.concatenate(x_true_all, axis=0),
        "x_mu": np.concatenate(x_mu_all, axis=0),
    }


@torch.no_grad()
def evaluate_spectrum_reconstruction(model, loader):
    model.eval()
    mse_all, mae_all, cosine_all = [], [], []

    for x, _ in loader:
        x = x.to(DEVICE)
        out = model(x)
        x_rec = out["mu_x"]

        mse = F.mse_loss(x_rec, x, reduction="none").mean(dim=1)
        mae = F.l1_loss(x_rec, x, reduction="none").mean(dim=1)
        cosine = F.cosine_similarity(x_rec, x, dim=1)

        mse_all.append(mse.cpu().numpy())
        mae_all.append(mae.cpu().numpy())
        cosine_all.append(cosine.cpu().numpy())

    mse_all = np.concatenate(mse_all)
    mae_all = np.concatenate(mae_all)
    cosine_all = np.concatenate(cosine_all)

    return {
        "spectrum_mse": float(mse_all.mean()),
        "spectrum_mae": float(mae_all.mean()),
        "spectrum_cosine": float(cosine_all.mean()),
    }


# ============================================================
# PLOTS
# ============================================================

def save_training_plots(history, species_name):
    prefix = safe_filename(species_name)

    plt.figure(figsize=(9, 5))
    plt.plot(history["train_loss"], label="Total loss")
    plt.plot(history["train_amr_loss"], label="AMR loss")
    plt.plot(history["train_spec_loss"], label="Spectrum loss")
    plt.plot(history["train_kl_w"], label="KL(w)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training losses - {species_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_training_losses.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(history["val_micro_auc"], label="Validation micro-AUC")
    plt.plot(history["val_macro_auc"], label="Validation macro-AUC")
    plt.axvline(history["best_epoch"], linestyle="--", label=f"Best epoch {history['best_epoch']}")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"Validation AUC - {species_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_validation_auc.png", dpi=200)
    plt.close()


def save_auc_barplot(per_ab_auc, species_name):
    prefix = safe_filename(species_name)
    df = per_ab_auc.dropna(subset=["auc"]).copy()

    if len(df) == 0:
        return

    df = df.sort_values("auc", ascending=True)

    height = max(6, 0.22 * len(df))
    plt.figure(figsize=(10, height))
    plt.barh(df["antibiotic"], df["auc"])
    plt.xlabel("Test AUC")
    plt.ylabel("Antibiotic")
    plt.xlim(0.0, 1.0)
    plt.title(f"Per-antibiotic test AUC - {species_name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_per_antibiotic_auc.png", dpi=200)
    plt.close()


def save_roc_plot(y_true, y_score, antibiotic_names, species_name, max_curves=35):
    prefix = safe_filename(species_name)

    valid = []
    for j, ab in enumerate(antibiotic_names):
        auc = safe_auc(y_true[:, j], y_score[:, j])
        if not np.isnan(auc):
            valid.append((j, ab, auc))

    if len(valid) == 0:
        return

    valid = sorted(valid, key=lambda x: x[2], reverse=True)[:max_curves]

    plt.figure(figsize=(8, 7))
    for j, ab, auc in valid:
        yt = y_true[:, j]
        ys = y_score[:, j]
        mask = ~np.isnan(yt)
        fpr, tpr, _ = roc_curve(yt[mask], ys[mask])
        plt.plot(fpr, tpr, linewidth=1.2, label=f"{ab} ({auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curves - {species_name}")
    plt.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_roc_curves.png", dpi=200)
    plt.close()


def make_tsne(data, perplexity=30, n_iter=1000, seed=42):
    if len(data) < 10:
        return None

    perplexity = min(perplexity, max(5, (len(data) - 1) // 3))

    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )

    try:
        return TSNE(max_iter=n_iter, **kwargs).fit_transform(data)
    except TypeError:
        return TSNE(n_iter=n_iter, **kwargs).fit_transform(data)


def resistance_burden(y):
    return np.nansum(y, axis=1)


def save_embedding_plot(emb, labels, title, out_path):
    if emb is None:
        return

    labels = np.asarray(labels)
    unique = np.unique(labels)

    plt.figure(figsize=(8, 7))
    for lab in unique:
        idx = labels == lab
        plt.scatter(emb[idx, 0], emb[idx, 1], s=16, alpha=0.75, label=str(lab))

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    if len(unique) <= 15:
        plt.legend(markerscale=1.5, fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_tsne_plots(test_out, species_name):
    if not DO_TSNE:
        return

    prefix = safe_filename(species_name)
    w = test_out["w_mu"]
    y = test_out["y_true"]

    if len(w) < 10:
        return

    if len(w) > TSNE_MAX_SAMPLES:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(w), size=TSNE_MAX_SAMPLES, replace=False)
        w = w[idx]
        y = y[idx]

    emb = make_tsne(w, perplexity=TSNE_PERPLEXITY, n_iter=TSNE_N_ITER, seed=SEED)

    if emb is None:
        return

    burden = resistance_burden(y).astype(int)
    save_embedding_plot(
        emb,
        burden,
        f"t-SNE of w colored by resistance burden - {species_name}",
        FIG_DIR / f"{prefix}_tsne_burden.png",
    )


def save_spectrum_reconstruction_plot(test_out, species_name, n_examples=5):
    prefix = safe_filename(species_name)

    x_true = test_out["x_true"]
    x_rec = test_out["x_mu"]

    if len(x_true) == 0:
        return

    n_examples = min(n_examples, len(x_true))

    plt.figure(figsize=(13, 4))
    for i in range(n_examples):
        plt.plot(x_true[i], linewidth=0.8, alpha=0.6, label="True" if i == 0 else None)
        plt.plot(x_rec[i], linewidth=0.8, alpha=0.6, linestyle="--", label="Reconstructed" if i == 0 else None)

    plt.xlabel("m/z bin")
    plt.ylabel("log1p intensity")
    plt.title(f"Spectrum reconstruction examples - {species_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_spectrum_reconstruction_examples.png", dpi=200)
    plt.close()


# ============================================================
# TRAINING FOR ONE SPECIES
# ============================================================

def build_loaders(X_sp, y_sp):
    n = len(X_sp)
    idx = np.arange(n)

    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=VAL_SIZE_FROM_TRAIN,
        random_state=SEED,
        shuffle=True,
    )

    train_loader = DataLoader(
        ResistanceSpectrumDataset(X_sp[train_idx], y_sp[train_idx]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        ResistanceSpectrumDataset(X_sp[val_idx], y_sp[val_idx]),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        ResistanceSpectrumDataset(X_sp[test_idx], y_sp[test_idx]),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    split_info = {
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }

    return train_loader, val_loader, test_loader, split_info


def train_one_species(X_sp, y_sp, antibiotic_names, species_name):
    input_dim = X_sp.shape[1]
    num_antibiotics = y_sp.shape[1]

    train_loader, val_loader, test_loader, split_info = build_loaders(X_sp, y_sp)

    model = ResistanceOrientedVAE(
        input_dim=input_dim,
        num_antibiotics=num_antibiotics,
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history = {
        "train_loss": [],
        "train_amr_loss": [],
        "train_spec_loss": [],
        "train_kl_w": [],
        "val_micro_auc": [],
        "val_macro_auc": [],
        "best_epoch": 0,
    }

    best_metric = -np.inf
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0

    print("\n============================================================")
    print(f"Training species: {species_name}")
    print(f"Samples: {len(X_sp)} | antibiotics: {num_antibiotics}")
    print(f"Split: {split_info}")
    print("============================================================")

    for ep in range(MAX_EPOCHS):
        model.train()

        running_loss = 0.0
        running_amr = 0.0
        running_spec = 0.0
        running_kl = 0.0

        beta_w = kl_anneal(ep, warmup=KL_WARMUP) * BETA_W

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad()

            # IMPORTANT: y is not passed into the model.
            out = model(x)

            y_mask, y_filled = prepare_y_target(y)

            if USE_FOCAL_LOSS:
                l_amr = focal_loss_masked(out["y_logits"], y_filled, y_mask)
            else:
                l_amr = masked_bce_with_logits(out["y_logits"], y_filled, y_mask)

            l_spec = spectrum_hurdle_loss(
                x,
                out["pres_logits"],
                out["mu_x"],
                out["sig_x"],
            )

            l_kl = kl_normal(out["w_mu"], out["w_logvar"])

            loss = (
                LAMBDA_AMR * l_amr
                + LAMBDA_SPEC * l_spec
                + beta_w * l_kl
            )

            loss.backward()
            opt.step()

            running_loss += loss.item()
            running_amr += l_amr.item()
            running_spec += l_spec.item()
            running_kl += l_kl.item()

        avg_loss = running_loss / max(1, len(train_loader))
        avg_amr = running_amr / max(1, len(train_loader))
        avg_spec = running_spec / max(1, len(train_loader))
        avg_kl = running_kl / max(1, len(train_loader))

        val_out = collect_outputs(model, val_loader)
        _, val_micro_auc, val_macro_auc = compute_aucs(
            val_out["y_true"],
            val_out["y_prob"],
            antibiotic_names,
        )

        history["train_loss"].append(avg_loss)
        history["train_amr_loss"].append(avg_amr)
        history["train_spec_loss"].append(avg_spec)
        history["train_kl_w"].append(avg_kl)
        history["val_micro_auc"].append(val_micro_auc)
        history["val_macro_auc"].append(val_macro_auc)

        monitor = val_micro_auc
        if np.isnan(monitor):
            monitor = -np.inf

        improved = monitor > best_metric + EARLY_STOP_MIN_DELTA

        if improved:
            best_metric = monitor
            best_epoch = ep
            history["best_epoch"] = ep
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"{species_name} | "
            f"Epoch {ep:03d} | "
            f"loss={avg_loss:.4f} | "
            f"amr={avg_amr:.4f} | "
            f"spec={avg_spec:.4f} | "
            f"kl={avg_kl:.4f} | "
            f"val_micro_auc={val_micro_auc:.4f} | "
            f"val_macro_auc={val_macro_auc:.4f} | "
            f"best={best_metric:.4f} @ {best_epoch}"
        )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(
                f"Early stopping for {species_name}: "
                f"no validation micro-AUC improvement for {EARLY_STOP_PATIENCE} epochs."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_out = collect_outputs(model, test_loader)

    per_ab_auc, test_micro_auc, test_macro_auc = compute_aucs(
        test_out["y_true"],
        test_out["y_prob"],
        antibiotic_names,
    )

    spec_metrics = evaluate_spectrum_reconstruction(model, test_loader)

    per_ab_auc.insert(0, "species", species_name)
    per_ab_auc["test_micro_auc_species"] = test_micro_auc
    per_ab_auc["test_macro_auc_species"] = test_macro_auc

    prefix = safe_filename(species_name)
    per_species_auc_path = AUC_DIR / f"{prefix}_aucs_per_antibiotic.csv"
    per_ab_auc.to_csv(per_species_auc_path, index=False)

    save_training_plots(history, species_name)
    save_auc_barplot(per_ab_auc, species_name)
    save_roc_plot(test_out["y_true"], test_out["y_prob"], antibiotic_names, species_name)
    save_tsne_plots(test_out, species_name)
    save_spectrum_reconstruction_plot(test_out, species_name)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "species": species_name,
            "antibiotic_names": antibiotic_names,
            "input_dim": input_dim,
            "num_antibiotics": num_antibiotics,
            "best_epoch": best_epoch,
            "best_val_micro_auc": best_metric if best_metric != -np.inf else np.nan,
        },
        AUC_DIR / f"{prefix}_best_model.pt",
    )

    summary_row = {
        "species": species_name,
        "n_total": len(X_sp),
        "n_train": split_info["n_train"],
        "n_val": split_info["n_val"],
        "n_test": split_info["n_test"],
        "num_antibiotics": len(antibiotic_names),
        "best_epoch": best_epoch,
        "best_val_micro_auc": best_metric if best_metric != -np.inf else np.nan,
        "final_test_micro_auc": test_micro_auc,
        "final_test_macro_auc": test_macro_auc,
        **spec_metrics,
        "auc_csv": str(per_species_auc_path),
    }

    return summary_row, per_ab_auc


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Using device: {DEVICE}")
    print(f"Base dir: {BASE_DIR}")
    print(f"Data path: {DATA_PATH}")
    print(f"Figures dir: {FIG_DIR}")
    print(f"AUC dir: {AUC_DIR}")

    with open(DATA_PATH, "rb") as f:
        payload = pickle.load(f)

    X = payload["data"]
    y_species = np.array(payload["label"])
    amr = payload["amr"]
    antibiotics_in_file = list(payload["antibiotics"])

    print("\nRaw shapes")
    print("X:", X.shape)
    print("AMR:", amr.shape)
    print("Number of antibiotics in file:", len(antibiotics_in_file))

    present_antibiotics = [ab for ab in amr_antibiotics if ab in antibiotics_in_file]
    missing_antibiotics = [ab for ab in amr_antibiotics if ab not in antibiotics_in_file]

    print("\nRequested antibiotics:", len(amr_antibiotics))
    print("Present antibiotics:", len(present_antibiotics))
    print("Missing antibiotics:", len(missing_antibiotics))

    if len(present_antibiotics) == 0:
        raise ValueError("None of the requested antibiotics are present in the pickle.")

    if len(missing_antibiotics) > 0:
        missing_path = AUC_DIR / "missing_requested_antibiotics.csv"
        pd.DataFrame({"missing_antibiotic": missing_antibiotics}).to_csv(missing_path, index=False)
        print(f"Saved missing requested antibiotics to: {missing_path}")

    ab_idx = [antibiotics_in_file.index(ab) for ab in present_antibiotics]

    X = remove_top_k_peaks_per_spectrum(X, TOP_K_REMOVE)
    X = np.log1p(X).astype(np.float32)
    amr_selected = amr[:, ab_idx].astype(np.float32)

    summary_rows = []
    all_auc_rows = []
    skipped_rows = []

    for genus, species in species_list:
        sp_name = species_key(genus, species)
        mask = species_mask(y_species, genus, species)
        n_sp = int(mask.sum())

        if n_sp < MIN_SAMPLES_PER_SPECIES:
            print(f"\nSkipping {sp_name}: only {n_sp} samples.")
            skipped_rows.append({
                "species": sp_name,
                "reason": "too_few_samples",
                "n_samples": n_sp,
            })
            continue

        X_sp = X[mask]
        y_sp = amr_selected[mask]

        observed_values = int((~np.isnan(y_sp)).sum())

        if observed_values < MIN_OBSERVED_AMR_VALUES:
            print(f"\nSkipping {sp_name}: only {observed_values} observed AMR values.")
            skipped_rows.append({
                "species": sp_name,
                "reason": "too_few_observed_amr_values",
                "n_samples": n_sp,
                "n_observed_amr_values": observed_values,
            })
            continue

        flat = y_sp.reshape(-1)
        flat = flat[~np.isnan(flat)]

        if len(np.unique(flat)) < 2:
            print(f"\nSkipping {sp_name}: only one AMR class observed.")
            skipped_rows.append({
                "species": sp_name,
                "reason": "only_one_amr_class",
                "n_samples": n_sp,
                "n_observed_amr_values": observed_values,
            })
            continue

        try:
            summary_row, per_ab_auc = train_one_species(
                X_sp=X_sp,
                y_sp=y_sp,
                antibiotic_names=present_antibiotics,
                species_name=sp_name,
            )

            summary_rows.append(summary_row)
            all_auc_rows.append(per_ab_auc)

        except Exception as e:
            print(f"\nERROR in {sp_name}: {repr(e)}")
            skipped_rows.append({
                "species": sp_name,
                "reason": "error",
                "error": repr(e),
                "n_samples": n_sp,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = AUC_DIR / "aucs_species_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved species summary to: {summary_path}")

    if len(all_auc_rows) > 0:
        all_auc_df = pd.concat(all_auc_rows, ignore_index=True)
    else:
        all_auc_df = pd.DataFrame()

    all_auc_path = AUC_DIR / "aucs_all_species_all_antibiotics.csv"
    all_auc_df.to_csv(all_auc_path, index=False)
    print(f"Saved all antibiotic-level AUCs to: {all_auc_path}")

    skipped_df = pd.DataFrame(skipped_rows)
    skipped_path = AUC_DIR / "skipped_species.csv"
    skipped_df.to_csv(skipped_path, index=False)
    print(f"Saved skipped species report to: {skipped_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
