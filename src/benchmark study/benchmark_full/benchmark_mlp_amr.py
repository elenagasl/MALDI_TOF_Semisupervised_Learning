#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark MLP (Paper-style) for AMR prediction from MALDI-TOF spectra
- Binary MLP per (species, antibiotic)
- LPS multiclass MLP (patterns as classes)
- Direct multilabel MLP (one logit per antibiotic)

Implements:
1) Uses combined pickle:
   /export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS.pkl

   Expected payload structure:
       X = payload["data"]
       y_species = payload["label"]
       amr = payload["amr"]
       antibiotics = list(payload["antibiotics"])

2) For each species:
   - select a subset of antibiotics with the maximum possible number of antibiotics
   - requiring at least MIN_COMPLETE_SAMPLES samples with:
       * no missing spectra values
       * no missing AMR values for all selected antibiotics
   - after that, antibiotics with only one class (all 0 or all 1) are removed
     because binary evaluation would not be meaningful

3) Builds LPS patterns
4) Removes rare patterns (min_count=11)
5) Stratified 80/20 split by patterns
6) Bayesian CV with Optuna: 1 fold, 200 trials
7) Train with max_epochs=1200, early stopping patience=50
8) No scaler fit (assumes spectra already preprocessed/scaled)
9) Save models + best hyperparams + test predictions/metrics
10) Parallelize Optuna with n_jobs=5 on CPU

Extras:
- Clear section prints per species (flush=True)
- tqdm bars for antibiotics
- tqdm bars for Optuna trials (binary/LPS/multilabel)
- Optional fold-level prints (toggle via --print_folds)
"""

import os
import json
import math
import time
import random
import pickle
import argparse
import warnings
import threading
from datetime import datetime
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

DEFAULT_PICKLE = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS.pkl"

TARGET_SPECIES = [
    "Staphylococcus_Aureus",
    "Staphylococcus_Epidermidis",
    "Escherichia_Coli",
    "Klebsiella_Pneumoniae",
    "Pseudomonas_Aeruginosa",
    #"Enterobacter_Cloacae",
    "Proteus_Mirabilis",
    #"Staphylococcus_Hominis",
    #"Serratia_Marcescens",
    #"Staphylococcus_Capitis",
    "Enterococcus_Faecium",
    #"Klebsiella_Oxytoca",
    #"Klebsiella_Variicola",
    #"Citrobacter_Koseri",
    #"Enterococcus_Faecalis",
    #"Staphylococcus_Lugdunensis",
    #"Citrobacter_Freundii",
    #"Morganella_Morganii",
    #"Proteus_Vulgaris",
    #"Staphylococcus_Haemolyticus",
    #"Candida_Albicans",
    #"Streptococcus_Pneumoniae",
    #"Stenotrophomonas_Maltophilia",
    #"Campylobacter_Jejuni",
    #"Haemophilus_Influenzae",
]

SEED = 42
TEST_SIZE = 0.20
VAL_SIZE_IN_TRAIN = 0.20
MIN_COMPLETE_SAMPLES = 500
MIN_PATTERN_COUNT = 11

N_TRIALS = 200
OPTUNA_N_JOBS = 5

BATCH_SIZE = 64
MAX_EPOCHS = 1200
PATIENCE = 50

DEVICE = torch.device("cpu")

# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

# =========================================================
# UTILS
# =========================================================

def normalize_species_name(x: str) -> str:
    """
    Converts species names to Genus_Species format.
    Handles examples like:
    - 'Staphylococcus aureus'
    - 'Staphylococcus_Aureus'
    - 'staphylococcus aureus'
    """
    if pd.isna(x):
        return x
    x = str(x).strip().replace(" ", "_")
    parts = [p for p in x.split("_") if p]
    if len(parts) >= 2:
        return "_".join([parts[0].capitalize(), parts[1].capitalize()] + parts[2:])
    return x.capitalize()

def make_output_dir(base_dir: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base_dir, f"benchmark_mlp_{stamp}")
    os.makedirs(outdir, exist_ok=True)
    return outdir

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def numpy_to_torch_loader(X, y, batch_size=64, shuffle=True):
    x_t = torch.tensor(X, dtype=torch.float32)
    if y.ndim == 1:
        if np.issubdtype(y.dtype, np.integer):
            y_t = torch.tensor(y, dtype=torch.long)
        else:
            y_t = torch.tensor(y, dtype=torch.float32)
    else:
        y_t = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def activation_from_name(name: str):
    if name == "identity":
        return nn.Identity()
    if name == "logistic":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")

def optimizer_from_name(name: str, params, lr: float):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown solver: {name}")

def weighted_f1_mean_binary_columns(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        if len(np.unique(yt)) < 2:
            vals.append(np.nan)
        else:
            vals.append(f1_score(yt, yp, average="weighted", zero_division=0))
    vals = np.array(vals, dtype=float)
    if np.all(np.isnan(vals)):
        return np.nan
    return float(np.nanmean(vals))

def metrics_for_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "hl": float(hamming_loss(y_true, y_pred)),
        "wf1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_samples": int(len(y_true)),
        "n_resistant": int(np.sum(y_true == 1)),
        "n_susceptible": int(np.sum(y_true == 0)),
    }

def metrics_for_multilabel_binary_decomposition(y_true: np.ndarray, y_pred: np.ndarray, antibiotics: list) -> pd.DataFrame:
    rows = []
    for j, ab in enumerate(antibiotics):
        yt = y_true[:, j].astype(int)
        yp = y_pred[:, j].astype(int)
        if len(np.unique(yt)) < 2:
            status = "constant_ground_truth"
            row = {
                "antibiotic": ab,
                "acc": np.nan,
                "hl": np.nan,
                "wf1": np.nan,
                "precision_weighted": np.nan,
                "recall_weighted": np.nan,
                "n_samples": len(yt),
                "n_resistant": int(np.sum(yt == 1)),
                "n_susceptible": int(np.sum(yt == 0)),
                "status": status,
            }
        else:
            m = metrics_for_binary(yt, yp)
            m["antibiotic"] = ab
            m["status"] = "ok"
            row = m
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_metrics_from_df(df_metrics: pd.DataFrame, model_name: str, species: str) -> dict:
    ok = df_metrics[df_metrics["status"] == "ok"].copy()
    return {
        "species": species,
        "model": model_name,
        "n_antibiotics": int(df_metrics.shape[0]),
        "n_antibiotics_evaluable": int(ok.shape[0]),
        "mean_acc": float(ok["acc"].mean()) if len(ok) else np.nan,
        "mean_hl": float(ok["hl"].mean()) if len(ok) else np.nan,
        "mean_wf1": float(ok["wf1"].mean()) if len(ok) else np.nan,
        "median_wf1": float(ok["wf1"].median()) if len(ok) else np.nan,
    }

def build_pattern_strings(Y: np.ndarray) -> np.ndarray:
    return np.array(["".join(map(str, row.astype(int).tolist())) for row in Y], dtype=object)

def filter_rare_patterns(X, Y, patterns, min_count=11):
    counts = Counter(patterns.tolist())
    keep = np.array([counts[p] >= min_count for p in patterns], dtype=bool)
    return X[keep], Y[keep], patterns[keep], counts

def stratified_train_test_by_patterns(X, Y, patterns, test_size=0.2, seed=42):
    idx = np.arange(len(patterns))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=patterns,
    )
    return train_idx, test_idx

def encode_patterns(patterns: np.ndarray):
    uniq = sorted(pd.unique(patterns).tolist())
    p2i = {p: i for i, p in enumerate(uniq)}
    i2p = {i: p for p, i in p2i.items()}
    y = np.array([p2i[p] for p in patterns], dtype=np.int64)
    return y, p2i, i2p

def decode_pattern_indices_to_binary_matrix(pred_idx: np.ndarray, i2p: dict, n_antibiotics: int) -> np.ndarray:
    mats = []
    for idx in pred_idx:
        p = i2p[int(idx)]
        arr = np.array(list(map(int, list(p))), dtype=int)
        if len(arr) != n_antibiotics:
            raise ValueError("Decoded pattern length does not match n_antibiotics.")
        mats.append(arr)
    return np.vstack(mats)

def safe_species_dir_name(species: str) -> str:
    return species.replace("/", "_").replace(" ", "_")

# =========================================================
# MODEL
# =========================================================

class PaperMLP(nn.Module):
    def __init__(self, input_dim, output_dim, layer1, layer2, layer3, activation):
        super().__init__()
        act = activation_from_name(activation)
        self.net = nn.Sequential(
            nn.Linear(input_dim, layer1),
            act,
            nn.Linear(layer1, layer2),
            act.__class__() if not isinstance(act, nn.Identity) else nn.Identity(),
            nn.Linear(layer2, layer3),
            act.__class__() if not isinstance(act, nn.Identity) else nn.Identity(),
            nn.Linear(layer3, output_dim),
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# TRAIN / EVAL
# =========================================================

def predict_binary_logits(model, X, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=DEVICE)
            logits = model(xb).cpu().numpy().reshape(-1)
            preds.append(logits)
    return np.concatenate(preds)

def predict_multilabel_logits(model, X, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=DEVICE)
            logits = model(xb).cpu().numpy()
            preds.append(logits)
    return np.vstack(preds)

def predict_multiclass_logits(model, X, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=DEVICE)
            logits = model(xb).cpu().numpy()
            preds.append(logits)
    return np.vstack(preds)

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    task_type,
    params,
    max_epochs=1200,
    patience=50,
    batch_size=64,
    print_folds=False,
):
    """
    task_type:
        - 'binary'
        - 'multilabel'
        - 'multiclass'
    """

    input_dim = X_train.shape[1]

    if task_type == "binary":
        output_dim = 1
        criterion = nn.BCEWithLogitsLoss()
    elif task_type == "multilabel":
        output_dim = y_train.shape[1]
        criterion = nn.BCEWithLogitsLoss()
    elif task_type == "multiclass":
        output_dim = int(np.max(y_train)) + 1
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported task_type={task_type}")

    model = PaperMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        layer1=params["layer1"],
        layer2=params["layer2"],
        layer3=params["layer3"],
        activation=params["activation"],
    ).to(DEVICE)

    optimizer = optimizer_from_name(params["solver"], model.parameters(), params["learning_rate"])

    if task_type == "binary":
        train_loader = numpy_to_torch_loader(X_train, y_train.astype(np.float32), batch_size=batch_size, shuffle=True)
    elif task_type == "multilabel":
        train_loader = numpy_to_torch_loader(X_train, y_train.astype(np.float32), batch_size=batch_size, shuffle=True)
    else:
        train_loader = numpy_to_torch_loader(X_train, y_train.astype(np.int64), batch_size=batch_size, shuffle=True)

    best_state = None
    best_score = -np.inf
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            out = model(xb)

            if task_type == "binary":
                yb = yb.float().view(-1, 1)
                loss = criterion(out, yb)
            elif task_type == "multilabel":
                yb = yb.float()
                loss = criterion(out, yb)
            else:
                yb = yb.long()
                loss = criterion(out, yb)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # validation metric = weighted F1
        if task_type == "binary":
            val_logits = predict_binary_logits(model, X_val)
            val_pred = (1 / (1 + np.exp(-val_logits)) >= 0.5).astype(int)
            val_score = f1_score(y_val, val_pred, average="weighted", zero_division=0)

        elif task_type == "multilabel":
            val_logits = predict_multilabel_logits(model, X_val)
            val_pred = (1 / (1 + np.exp(-val_logits)) >= 0.5).astype(int)
            val_score = weighted_f1_mean_binary_columns(y_val, val_pred)

        else:  # multiclass
            val_logits = predict_multiclass_logits(model, X_val)
            val_pred_class = np.argmax(val_logits, axis=1)
            val_score = f1_score(y_val, val_pred_class, average="weighted", zero_division=0)

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else np.nan,
            "val_wf1": float(val_score) if val_score == val_score else np.nan,
        })

        if print_folds:
            print(
                f"    epoch={epoch:04d} | train_loss={np.mean(train_losses):.6f} | val_wf1={val_score:.6f}",
                flush=True
            )

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, float(best_score), int(best_epoch), history

# =========================================================
# OPTUNA
# =========================================================

class TqdmOptunaCallback:
    def __init__(self, pbar):
        self.pbar = pbar
        self.lock = threading.Lock()

    def __call__(self, study, trial):
        with self.lock:
            self.pbar.update(1)

def suggest_hparams(trial):
    return {
        "activation": trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"]),
        "solver": trial.suggest_categorical("solver", ["sgd", "adam"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True),
        "layer1": trial.suggest_int("layer1", 10, 500),
        "layer2": trial.suggest_int("layer2", 10, 500),
        "layer3": trial.suggest_int("layer3", 10, 500),
    }

def optimize_binary(
    X_train,
    y_train,
    n_trials=200,
    n_jobs=5,
    max_epochs=1200,
    patience=50,
    batch_size=64,
    seed=42,
    print_folds=False,
    desc="Optuna-Binary"
):
    idx = np.arange(len(y_train))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=VAL_SIZE_IN_TRAIN,
        random_state=seed,
        stratify=y_train,
    )
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    sampler = TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        params = suggest_hparams(trial)
        _, best_val, _, _ = train_model(
            X_tr, y_tr, X_va, y_va,
            task_type="binary",
            params=params,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            print_folds=print_folds,
        )
        return best_val

    with tqdm(total=n_trials, desc=desc, leave=False) as pbar:
        cb = TqdmOptunaCallback(pbar)
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[cb], show_progress_bar=False)

    return study.best_params, float(study.best_value), study

def optimize_multiclass_lps(
    X_train,
    y_train_cls,
    patterns_train,
    n_trials=200,
    n_jobs=5,
    max_epochs=1200,
    patience=50,
    batch_size=64,
    seed=42,
    print_folds=False,
    desc="Optuna-LPS"
):
    idx = np.arange(len(y_train_cls))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=VAL_SIZE_IN_TRAIN,
        random_state=seed,
        stratify=patterns_train,
    )
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train_cls[tr_idx], y_train_cls[va_idx]

    sampler = TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        params = suggest_hparams(trial)
        _, best_val, _, _ = train_model(
            X_tr, y_tr, X_va, y_va,
            task_type="multiclass",
            params=params,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            print_folds=print_folds,
        )
        return best_val

    with tqdm(total=n_trials, desc=desc, leave=False) as pbar:
        cb = TqdmOptunaCallback(pbar)
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[cb], show_progress_bar=False)

    return study.best_params, float(study.best_value), study

def optimize_multilabel(
    X_train,
    Y_train,
    patterns_train,
    n_trials=200,
    n_jobs=5,
    max_epochs=1200,
    patience=50,
    batch_size=64,
    seed=42,
    print_folds=False,
    desc="Optuna-Multilabel"
):
    idx = np.arange(len(Y_train))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=VAL_SIZE_IN_TRAIN,
        random_state=seed,
        stratify=patterns_train,
    )
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    Y_tr, Y_va = Y_train[tr_idx], Y_train[va_idx]

    sampler = TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        params = suggest_hparams(trial)
        _, best_val, _, _ = train_model(
            X_tr, Y_tr, X_va, Y_va,
            task_type="multilabel",
            params=params,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            print_folds=print_folds,
        )
        return best_val

    with tqdm(total=n_trials, desc=desc, leave=False) as pbar:
        cb = TqdmOptunaCallback(pbar)
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[cb], show_progress_bar=False)

    return study.best_params, float(study.best_value), study

# =========================================================
# SUBSET SELECTION
# =========================================================

def select_antibiotic_subset_max_k(
    X_sp: np.ndarray,
    amr_sp: pd.DataFrame,
    min_complete_samples: int = 1500,
):
    """
    Greedy backward elimination:
    Start with all antibiotics, remove the antibiotic with the largest missingness
    until the number of complete samples (valid spectra + no NaNs in selected AMR cols)
    is at least min_complete_samples.

    Primary goal: maximize number of antibiotics.
    Secondary effect: antibiotics with more complete data are preferred.

    Returns:
        selected_antibiotics, keep_mask_complete, info_dict
    """
    feat_ok = ~np.isnan(X_sp).any(axis=1)
    ab_list = list(amr_sp.columns)

    if len(ab_list) == 0:
        return [], np.zeros(len(amr_sp), dtype=bool), {"reason": "no_antibiotics"}

    current = ab_list.copy()

    while len(current) > 0:
        amr_ok = ~amr_sp[current].isna().any(axis=1).values
        keep = feat_ok & amr_ok
        n_complete = int(np.sum(keep))

        if n_complete >= min_complete_samples:
            return current, keep, {
                "n_antibiotics": len(current),
                "n_complete_samples": n_complete,
                "removed_antibiotics": [ab for ab in ab_list if ab not in current],
            }

        miss_counts = amr_sp[current].isna().sum(axis=0).sort_values(ascending=False)
        worst_ab = miss_counts.index[0]
        current.remove(worst_ab)

    return [], np.zeros(len(amr_sp), dtype=bool), {
        "reason": "could_not_reach_min_complete_samples",
        "n_complete_samples": 0
    }

# =========================================================
# DATA LOADING
# =========================================================

def load_payload(pickle_path: str):
    with open(pickle_path, "rb") as f:
        payload = pickle.load(f)

    X = payload["data"]
    y_species = payload["label"]
    amr = payload["amr"]
    antibiotics = list(payload["antibiotics"])

    X = np.asarray(X)
    y_species = np.asarray([normalize_species_name(s) for s in y_species])

    if isinstance(amr, pd.DataFrame):
        amr_df = amr.copy()
        if list(amr_df.columns) != antibiotics:
            amr_df.columns = antibiotics
    else:
        amr_df = pd.DataFrame(np.asarray(amr), columns=antibiotics)

    # force numeric if possible
    for c in amr_df.columns:
        amr_df[c] = pd.to_numeric(amr_df[c], errors="coerce")

    return X, y_species, amr_df, antibiotics

# =========================================================
# MAIN BENCHMARK
# =========================================================

def benchmark_species(
    species: str,
    X_all: np.ndarray,
    y_species_all: np.ndarray,
    amr_all: pd.DataFrame,
    out_root: str,
    min_complete_samples: int,
    min_pattern_count: int,
    n_trials: int,
    n_jobs: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    print_folds: bool = False,
):
    print("=" * 100, flush=True)
    print(f"[SPECIES] {species}", flush=True)

    species_dir = os.path.join(out_root, safe_species_dir_name(species))
    os.makedirs(species_dir, exist_ok=True)

    mask_sp = (y_species_all == species)
    n_total_species = int(np.sum(mask_sp))
    print(f"  Total samples in pickle for species: {n_total_species}", flush=True)

    if n_total_species == 0:
        print("  -> species not found. Skipping.", flush=True)
        return {"species": species, "status": "species_not_found"}

    X_sp = X_all[mask_sp]
    amr_sp = amr_all.loc[mask_sp].reset_index(drop=True)

    # Step 2: select antibiotics subset
    selected_abs, keep_complete, subset_info = select_antibiotic_subset_max_k(
        X_sp=X_sp,
        amr_sp=amr_sp,
        min_complete_samples=min_complete_samples,
    )

    save_json(subset_info, os.path.join(species_dir, f"{species}__subset_selection_info.json"))

    if len(selected_abs) == 0:
        print("  -> could not find antibiotic subset satisfying minimum complete samples. Skipping.", flush=True)
        return {"species": species, "status": "no_valid_antibiotic_subset"}

    print(f"  Selected antibiotics before constant-label filtering: {len(selected_abs)}", flush=True)
    print(f"  Complete samples with selected set: {int(np.sum(keep_complete))}", flush=True)

    X_sp = X_sp[keep_complete]
    Y_sp = amr_sp.loc[keep_complete, selected_abs].astype(int).values

    # Remove antibiotics with only one class (constant)
    keep_ab = []
    dropped_constant_abs = []
    for j, ab in enumerate(selected_abs):
        uniq = np.unique(Y_sp[:, j])
        if len(uniq) >= 2:
            keep_ab.append(ab)
        else:
            dropped_constant_abs.append(ab)

    if len(dropped_constant_abs) > 0:
        print(f"  Dropping antibiotics with a single class after complete-case filtering: {dropped_constant_abs}", flush=True)

    if len(keep_ab) == 0:
        print("  -> after dropping constant antibiotics, none remain. Skipping.", flush=True)
        return {"species": species, "status": "all_antibiotics_constant"}

    selected_abs = keep_ab
    keep_ab = []
    dropped_constant_abs = []
    for j, ab in enumerate(selected_abs):
        uniq = np.unique(Y_sp[:, j])
        if len(uniq) >= 2:
            keep_ab.append(ab)
        else:
            dropped_constant_abs.append(ab)
    if len(dropped_constant_abs) > 0:
        print(f"  Dropping antibiotics with a single class after complete-case filtering: {dropped_constant_abs}", flush=True)

    if len(keep_ab) == 0:
        print("  -> after dropping constant antibiotics, none remain. Skipping.", flush=True)
        return {"species": species, "status": "all_antibiotics_constant"}

    selected_abs = keep_ab
    Y_sp = amr_sp.loc[keep_complete, selected_abs].astype(int).values
    # Step 3: LPS patterns
    patterns = build_pattern_strings(Y_sp)
    pattern_counts_before = Counter(patterns.tolist())
    pd.DataFrame({
        "pattern": list(pattern_counts_before.keys()),
        "count": list(pattern_counts_before.values())
    }).sort_values("count", ascending=False).to_csv(
        os.path.join(species_dir, f"{species}__pattern_counts_before_filter.csv"),
        index=False
    )

    # Step 4: remove rare patterns
    X_sp, Y_sp, patterns, pattern_counts = filter_rare_patterns(
        X_sp, Y_sp, patterns, min_count=min_pattern_count
    )

    if len(X_sp) < min_complete_samples:
        print(f"  -> after rare-pattern filtering, samples={len(X_sp)} < {min_complete_samples}. Skipping.", flush=True)
        return {"species": species, "status": "too_few_after_pattern_filter"}

    pattern_counts_after = Counter(patterns.tolist())
    pd.DataFrame({
        "pattern": list(pattern_counts_after.keys()),
        "count": list(pattern_counts_after.values())
    }).sort_values("count", ascending=False).to_csv(
        os.path.join(species_dir, f"{species}__pattern_counts_after_filter.csv"),
        index=False
    )

    print(f"  Final antibiotics used: {len(selected_abs)}", flush=True)
    print(f"  Antibiotics: {selected_abs}", flush=True)
    print(f"  Samples after rare-pattern filtering: {len(X_sp)}", flush=True)
    print(f"  Number of remaining patterns: {len(np.unique(patterns))}", flush=True)

    save_json({
        "species": species,
        "n_total_species_samples": n_total_species,
        "n_complete_samples_before_pattern_filter": int(np.sum(keep_complete)),
        "n_samples_after_pattern_filter": int(len(X_sp)),
        "selected_antibiotics": selected_abs,
        "dropped_constant_antibiotics": dropped_constant_abs,
        "min_pattern_count": int(min_pattern_count),
    }, os.path.join(species_dir, f"{species}__data_summary.json"))

    # Step 5: stratified 80/20 split by patterns
    train_idx, test_idx = stratified_train_test_by_patterns(X_sp, Y_sp, patterns, test_size=TEST_SIZE, seed=SEED)

    X_train, X_test = X_sp[train_idx], X_sp[test_idx]
    Y_train, Y_test = Y_sp[train_idx], Y_sp[test_idx]
    patterns_train, patterns_test = patterns[train_idx], patterns[test_idx]

    print(f"  Train size: {len(train_idx)} | Test size: {len(test_idx)}", flush=True)

    # Encode LPS multiclass labels
    y_train_lps, p2i, i2p = encode_patterns(patterns_train)
    y_test_lps = np.array([p2i[p] for p in patterns_test], dtype=np.int64)

    # Save split metadata
    save_json({
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "n_train_patterns": int(len(np.unique(patterns_train))),
        "n_test_patterns": int(len(np.unique(patterns_test))),
        "pattern_to_index": p2i,
    }, os.path.join(species_dir, f"{species}__split_info.json"))

    # =====================================================
    # 1) BINARY CLASSIFIERS
    # =====================================================
    binary_models_dir = os.path.join(species_dir, "binary_models")
    os.makedirs(binary_models_dir, exist_ok=True)

    binary_rows = []
    binary_summary_rows = []

    print("  [BINARY] Starting per-antibiotic training...", flush=True)

    for ab_idx in tqdm(range(len(selected_abs)), desc=f"{species} | Binary antibiotics", leave=False):
        ab = selected_abs[ab_idx]
        y_train_bin = Y_train[:, ab_idx].astype(int)
        y_test_bin = Y_test[:, ab_idx].astype(int)

        if len(np.unique(y_train_bin)) < 2 or len(np.unique(y_test_bin)) < 2:
            print(f"    -> {ab}: skipped because train/test has a single class.", flush=True)
            binary_summary_rows.append({
                "species": species,
                "model": "binary",
                "antibiotic": ab,
                "status": "skipped_single_class",
                "acc": np.nan,
                "hl": np.nan,
                "wf1": np.nan,
                "precision_weighted": np.nan,
                "recall_weighted": np.nan,
                "n_samples": len(y_test_bin),
                "n_resistant": int(np.sum(y_test_bin == 1)),
                "n_susceptible": int(np.sum(y_test_bin == 0)),
            })
            continue

        best_params, best_cv_score, study = optimize_binary(
            X_train=X_train,
            y_train=y_train_bin,
            n_trials=n_trials,
            n_jobs=n_jobs,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            seed=SEED,
            print_folds=print_folds,
            desc=f"{species} | {ab} | Optuna-Binary"
        )

        # Final train/val split again inside train_model is only used for ES selection
        tr_idx2, va_idx2 = train_test_split(
            np.arange(len(y_train_bin)),
            test_size=VAL_SIZE_IN_TRAIN,
            random_state=SEED,
            stratify=y_train_bin,
        )
        model_bin, best_val_score, best_epoch, hist_bin = train_model(
            X_train[tr_idx2], y_train_bin[tr_idx2],
            X_train[va_idx2], y_train_bin[va_idx2],
            task_type="binary",
            params=best_params,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            print_folds=print_folds,
        )

        test_logits = predict_binary_logits(model_bin, X_test)
        test_prob = 1 / (1 + np.exp(-test_logits))
        test_pred = (test_prob >= 0.5).astype(int)

        m = metrics_for_binary(y_test_bin, test_pred)
        row = {
            "species": species,
            "model": "binary",
            "antibiotic": ab,
            "status": "ok",
            **m,
            "cv_best_wf1": best_cv_score,
            "val_best_wf1": best_val_score,
            "best_epoch": best_epoch,
        }
        binary_summary_rows.append(row)

        # save model
        model_path = os.path.join(binary_models_dir, f"{species}__{ab}__binary_model.pt")
        torch.save({
            "state_dict": model_bin.state_dict(),
            "params": best_params,
            "input_dim": int(X_train.shape[1]),
            "output_dim": 1,
            "species": species,
            "antibiotic": ab,
            "task_type": "binary",
        }, model_path)

        save_json(best_params, os.path.join(binary_models_dir, f"{species}__{ab}__binary_best_params.json"))
        pd.DataFrame(hist_bin).to_csv(
            os.path.join(binary_models_dir, f"{species}__{ab}__binary_train_history.csv"),
            index=False
        )
        pd.DataFrame({
            "y_true": y_test_bin,
            "y_prob": test_prob,
            "y_pred": test_pred,
        }).to_csv(
            os.path.join(binary_models_dir, f"{species}__{ab}__binary_test_predictions.csv"),
            index=False
        )

    df_binary = pd.DataFrame(binary_summary_rows)
    df_binary.to_csv(os.path.join(species_dir, f"{species}__binary_metrics.csv"), index=False)

    binary_overall = {
        "species": species,
        "model": "binary",
        "n_antibiotics": int(len(selected_abs)),
        "n_antibiotics_evaluable": int(np.sum(df_binary["status"] == "ok")) if len(df_binary) else 0,
        "mean_acc": float(df_binary.loc[df_binary["status"] == "ok", "acc"].mean()) if len(df_binary) else np.nan,
        "mean_hl": float(df_binary.loc[df_binary["status"] == "ok", "hl"].mean()) if len(df_binary) else np.nan,
        "mean_wf1": float(df_binary.loc[df_binary["status"] == "ok", "wf1"].mean()) if len(df_binary) else np.nan,
    }
    save_json(binary_overall, os.path.join(species_dir, f"{species}__binary_overall.json"))

    # =====================================================
    # 2) LPS MULTICLASS
    # =====================================================
    print("  [LPS] Starting multiclass pattern training...", flush=True)

    best_params_lps, best_cv_score_lps, study_lps = optimize_multiclass_lps(
        X_train=X_train,
        y_train_cls=y_train_lps,
        patterns_train=patterns_train,
        n_trials=n_trials,
        n_jobs=n_jobs,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        seed=SEED,
        print_folds=print_folds,
        desc=f"{species} | Optuna-LPS"
    )

    tr_idx_lps, va_idx_lps = train_test_split(
        np.arange(len(y_train_lps)),
        test_size=VAL_SIZE_IN_TRAIN,
        random_state=SEED,
        stratify=patterns_train,
    )

    model_lps, best_val_lps, best_epoch_lps, hist_lps = train_model(
        X_train[tr_idx_lps], y_train_lps[tr_idx_lps],
        X_train[va_idx_lps], y_train_lps[va_idx_lps],
        task_type="multiclass",
        params=best_params_lps,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        print_folds=print_folds,
    )

    test_logits_lps = predict_multiclass_logits(model_lps, X_test)
    test_pred_class_lps = np.argmax(test_logits_lps, axis=1)
    test_pred_bin_lps = decode_pattern_indices_to_binary_matrix(
        test_pred_class_lps,
        i2p=i2p,
        n_antibiotics=len(selected_abs)
    )

    df_lps_metrics = metrics_for_multilabel_binary_decomposition(
        y_true=Y_test,
        y_pred=test_pred_bin_lps,
        antibiotics=selected_abs,
    )
    df_lps_metrics.to_csv(os.path.join(species_dir, f"{species}__lps_metrics.csv"), index=False)

    lps_overall = aggregate_metrics_from_df(df_lps_metrics, model_name="lps_multiclass", species=species)
    lps_overall["cv_best_wf1"] = best_cv_score_lps
    lps_overall["val_best_wf1"] = best_val_lps
    lps_overall["best_epoch"] = best_epoch_lps
    save_json(lps_overall, os.path.join(species_dir, f"{species}__lps_overall.json"))

    torch.save({
        "state_dict": model_lps.state_dict(),
        "params": best_params_lps,
        "input_dim": int(X_train.shape[1]),
        "output_dim": int(len(p2i)),
        "species": species,
        "task_type": "lps_multiclass",
        "pattern_to_index": p2i,
        "index_to_pattern": i2p,
        "antibiotics": selected_abs,
    }, os.path.join(species_dir, f"{species}__lps_model.pt"))

    save_json(best_params_lps, os.path.join(species_dir, f"{species}__lps_best_params.json"))
    pd.DataFrame(hist_lps).to_csv(os.path.join(species_dir, f"{species}__lps_train_history.csv"), index=False)
    pd.DataFrame({
        "y_true_pattern": patterns_test,
        "y_pred_pattern": [i2p[int(i)] for i in test_pred_class_lps],
        "y_true_class": y_test_lps,
        "y_pred_class": test_pred_class_lps,
    }).to_csv(os.path.join(species_dir, f"{species}__lps_test_predictions_patterns.csv"), index=False)

    pd.DataFrame(
        np.hstack([Y_test, test_pred_bin_lps]),
        columns=[f"{ab}_true" for ab in selected_abs] + [f"{ab}_pred" for ab in selected_abs]
    ).to_csv(os.path.join(species_dir, f"{species}__lps_test_predictions_binary_decomposed.csv"), index=False)

    # =====================================================
    # 3) DIRECT MULTILABEL
    # =====================================================
    print("  [MULTILABEL] Starting direct multilabel training...", flush=True)

    best_params_ml, best_cv_score_ml, study_ml = optimize_multilabel(
        X_train=X_train,
        Y_train=Y_train,
        patterns_train=patterns_train,
        n_trials=n_trials,
        n_jobs=n_jobs,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        seed=SEED,
        print_folds=print_folds,
        desc=f"{species} | Optuna-Multilabel"
    )

    tr_idx_ml, va_idx_ml = train_test_split(
        np.arange(len(Y_train)),
        test_size=VAL_SIZE_IN_TRAIN,
        random_state=SEED,
        stratify=patterns_train,
    )

    model_ml, best_val_ml, best_epoch_ml, hist_ml = train_model(
        X_train[tr_idx_ml], Y_train[tr_idx_ml],
        X_train[va_idx_ml], Y_train[va_idx_ml],
        task_type="multilabel",
        params=best_params_ml,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        print_folds=print_folds,
    )

    test_logits_ml = predict_multilabel_logits(model_ml, X_test)
    test_prob_ml = 1 / (1 + np.exp(-test_logits_ml))
    test_pred_ml = (test_prob_ml >= 0.5).astype(int)

    df_ml_metrics = metrics_for_multilabel_binary_decomposition(
        y_true=Y_test,
        y_pred=test_pred_ml,
        antibiotics=selected_abs,
    )
    df_ml_metrics.to_csv(os.path.join(species_dir, f"{species}__multilabel_metrics.csv"), index=False)

    ml_overall = aggregate_metrics_from_df(df_ml_metrics, model_name="direct_multilabel", species=species)
    ml_overall["cv_best_wf1"] = best_cv_score_ml
    ml_overall["val_best_wf1"] = best_val_ml
    ml_overall["best_epoch"] = best_epoch_ml
    save_json(ml_overall, os.path.join(species_dir, f"{species}__multilabel_overall.json"))

    torch.save({
        "state_dict": model_ml.state_dict(),
        "params": best_params_ml,
        "input_dim": int(X_train.shape[1]),
        "output_dim": int(len(selected_abs)),
        "species": species,
        "task_type": "direct_multilabel",
        "antibiotics": selected_abs,
    }, os.path.join(species_dir, f"{species}__multilabel_model.pt"))

    save_json(best_params_ml, os.path.join(species_dir, f"{species}__multilabel_best_params.json"))
    pd.DataFrame(hist_ml).to_csv(os.path.join(species_dir, f"{species}__multilabel_train_history.csv"), index=False)

    pred_df_ml = pd.DataFrame(test_prob_ml, columns=[f"{ab}_prob" for ab in selected_abs])
    for j, ab in enumerate(selected_abs):
        pred_df_ml[f"{ab}_true"] = Y_test[:, j]
        pred_df_ml[f"{ab}_pred"] = test_pred_ml[:, j]
    pred_df_ml.to_csv(os.path.join(species_dir, f"{species}__multilabel_test_predictions.csv"), index=False)

    # =====================================================
    # COMPARISON TABLE
    # =====================================================
    comparison_rows = []

    # binary rows
    for _, r in df_binary.iterrows():
        comparison_rows.append({
            "species": species,
            "antibiotic": r["antibiotic"],
            "binary_acc": r["acc"],
            "binary_hl": r["hl"],
            "binary_wf1": r["wf1"],
        })

    comp = pd.DataFrame(comparison_rows)
    comp = comp.merge(
        df_lps_metrics[["antibiotic", "acc", "hl", "wf1"]].rename(columns={
            "acc": "lps_acc",
            "hl": "lps_hl",
            "wf1": "lps_wf1",
        }),
        on="antibiotic",
        how="left"
    )
    comp = comp.merge(
        df_ml_metrics[["antibiotic", "acc", "hl", "wf1"]].rename(columns={
            "acc": "multilabel_acc",
            "hl": "multilabel_hl",
            "wf1": "multilabel_wf1",
        }),
        on="antibiotic",
        how="left"
    )
    comp.to_csv(os.path.join(species_dir, f"{species}__comparison_per_antibiotic.csv"), index=False)

    species_summary = {
        "species": species,
        "status": "ok",
        "n_samples_train": int(len(X_train)),
        "n_samples_test": int(len(X_test)),
        "n_antibiotics": int(len(selected_abs)),
        "antibiotics": selected_abs,
        "binary_mean_wf1": binary_overall["mean_wf1"],
        "lps_mean_wf1": lps_overall["mean_wf1"],
        "multilabel_mean_wf1": ml_overall["mean_wf1"],
    }
    save_json(species_summary, os.path.join(species_dir, f"{species}__species_summary.json"))

    print(f"  [DONE] {species}", flush=True)
    print(f"    Binary mean WF1:      {binary_overall['mean_wf1']:.6f}", flush=True)
    print(f"    LPS mean WF1:         {lps_overall['mean_wf1']:.6f}", flush=True)
    print(f"    Multilabel mean WF1:  {ml_overall['mean_wf1']:.6f}", flush=True)

    return species_summary

# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Paper-style MLP benchmark for AMR prediction from MALDI-TOF spectra")
    parser.add_argument("--pickle_path", type=str, default=DEFAULT_PICKLE)
    parser.add_argument("--outdir", type=str, default="./benchmark_outputs")
    parser.add_argument("--min_complete_samples", type=int, default=MIN_COMPLETE_SAMPLES)
    parser.add_argument("--min_pattern_count", type=int, default=MIN_PATTERN_COUNT)
    parser.add_argument("--test_size", type=float, default=TEST_SIZE)
    parser.add_argument("--val_size_in_train", type=float, default=VAL_SIZE_IN_TRAIN)
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    parser.add_argument("--n_jobs", type=int, default=OPTUNA_N_JOBS)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--print_folds", action="store_true")
    return parser.parse_args()

# =========================================================
# MAIN
# =========================================================

def main():
    global SEED, TEST_SIZE, VAL_SIZE_IN_TRAIN

    args = parse_args()
    SEED = args.seed
    TEST_SIZE = args.test_size
    VAL_SIZE_IN_TRAIN = args.val_size_in_train
    set_seed(SEED)

    os.makedirs(args.outdir, exist_ok=True)
    out_root = make_output_dir(args.outdir)

    print("=" * 100, flush=True)
    print("LOADING COMBINED PICKLE...", flush=True)
    print(f"Pickle path: {args.pickle_path}", flush=True)

    X, y_species, amr_df, antibiotics = load_payload(args.pickle_path)

    print(f"X shape: {X.shape}", flush=True)
    print(f"Number of samples: {len(X)}", flush=True)
    print(f"Number of antibiotics in payload: {len(antibiotics)}", flush=True)
    print(f"Unique species in payload: {len(pd.unique(y_species))}", flush=True)

    save_json({
        "pickle_path": args.pickle_path,
        "X_shape": list(X.shape),
        "n_samples": int(len(X)),
        "n_antibiotics_payload": int(len(antibiotics)),
        "target_species": TARGET_SPECIES,
        "seed": int(SEED),
        "test_size": float(TEST_SIZE),
        "val_size_in_train": float(VAL_SIZE_IN_TRAIN),
        "min_complete_samples": int(args.min_complete_samples),
        "min_pattern_count": int(args.min_pattern_count),
        "n_trials": int(args.n_trials),
        "n_jobs": int(args.n_jobs),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
    }, os.path.join(out_root, "run_config.json"))

    all_species_results = []

    for species in TARGET_SPECIES:
        try:
            res = benchmark_species(
                species=species,
                X_all=X,
                y_species_all=y_species,
                amr_all=amr_df,
                out_root=out_root,
                min_complete_samples=args.min_complete_samples,
                min_pattern_count=args.min_pattern_count,
                n_trials=args.n_trials,
                n_jobs=args.n_jobs,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                print_folds=args.print_folds,
            )
            all_species_results.append(res)
        except Exception as e:
            print(f"[ERROR] {species}: {repr(e)}", flush=True)
            all_species_results.append({
                "species": species,
                "status": "error",
                "error": repr(e),
            })

    df_all = pd.DataFrame(all_species_results)
    df_all.to_csv(os.path.join(out_root, "all_species_summary.csv"), index=False)

    # Global summary table if possible
    ok_df = df_all[df_all["status"] == "ok"].copy()
    if len(ok_df) > 0:
        ok_df.to_csv(os.path.join(out_root, "all_species_summary_ok_only.csv"), index=False)

    print("=" * 100, flush=True)
    print("BENCHMARK FINISHED", flush=True)
    print(f"Results saved in: {out_root}", flush=True)

if __name__ == "__main__":
    main()