#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark MLP (Paper-style) for AMR prediction from MALDI-TOF spectra (DRIAMS-A pickle)
- Binary MLP per (species, antibiotic)
- LPS multiclass MLP (patterns as classes)
- Direct multilabel MLP (one logit per antibiotic)

Implements:
1) Remove duplicates
2) Species-specific antibiotic subsets
3) Drop samples with NaNs in features or required antibiotics
4) Build LPS patterns
5) Remove rare patterns (min_count configurable, default=5)
6) Stratified 80/20 split by patterns
7) Bayesian CV with Optuna: 5 folds, 200 trials
8) StandardScaler fit on train only
9-11) Train with max_epochs=1200, early stopping patience=50
12) Save models + best hyperparams + test predictions/metrics

PROGRESS UX improvements added:
- Clear section prints per species (flush=True)
- tqdm bars for antibiotics
- tqdm bars for Optuna trials (binary/LPS/multilabel)
- Optional fold-level prints (toggle via --print_folds)

Notes:
- Parallelization (n_jobs) is applied at SPECIES level to avoid GPU contention.
- If you run on GPU, parallel training processes can compete for VRAM. Consider --n_jobs 1 on GPU.
"""

import os
import sys
import json
import time
import pickle
import hashlib
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

# --- Force unbuffered-ish stdout for nohup ---
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    # Python 3.7+
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from joblib import Parallel, delayed, dump
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import optuna

warnings.filterwarnings("ignore")


# -------------------------
# Config: Species antibiotics
# -------------------------
SPECIES_ANTIBIOTICS: Dict[str, List[str]] = {
    "Staphylococcus_Aureus": ["Oxacillin", "Clindamycin", "Fusidic acid"],
    "Escherichia_Coli": ["Ciprofloxacin", "Ceftriaxone", "Piperacillin-Tazobactam", "Cefepime"],
    "Klebsiella_Pneumoniae": ["Ciprofloxacin", "Ceftriaxone", "Imipenem", "Meropenem"],
    "Pseudomonas_Aeruginosa": ["Ciprofloxacin", "Imipenem", "Meropenem"],
}


# -------------------------
# Torch models
# -------------------------
class Identity(nn.Module):
    def forward(self, x):
        return x


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "logistic":
        return nn.Sigmoid()
    if name == "identity":
        return Identity()
    raise ValueError(f"Unknown activation: {name}")


class MLPBinary(nn.Module):
    def __init__(self, input_dim: int, layer1: int, layer2: int, layer3: int, activation: str):
        super().__init__()
        act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(input_dim, layer1), act,
            nn.Linear(layer1, layer2), act,
            nn.Linear(layer2, layer3), act,
            nn.Linear(layer3, 1),
        )

    def forward(self, x):
        return self.net(x)


class MLPLPS(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, layer1: int, layer2: int, layer3: int, activation: str):
        super().__init__()
        act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(input_dim, layer1), act,
            nn.Linear(layer1, layer2), act,
            nn.Linear(layer2, layer3), act,
            nn.Linear(layer3, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class MLPMultiLabel(nn.Module):
    def __init__(self, input_dim: int, n_outputs: int, layer1: int, layer2: int, layer3: int, activation: str):
        super().__init__()
        act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(input_dim, layer1), act,
            nn.Linear(layer1, layer2), act,
            nn.Linear(layer2, layer3), act,
            nn.Linear(layer3, n_outputs),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Utils: metrics
# -------------------------
def wf1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted"))


def evaluate_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    y_true, y_pred: shape (n_samples, n_labels) with {0,1}
    Metrics are computed antibiotic-by-antibiotic (binary) and averaged.
    """
    n_labels = y_true.shape[1]
    wf1s, accs, hls = [], [], []
    for j in range(n_labels):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        wf1s.append(f1_score(yt, yp, average="weighted"))
        accs.append(accuracy_score(yt, yp))
        hls.append(hamming_loss(yt, yp))
    return {
        "WF1": float(np.mean(wf1s)),
        "ACC": float(np.mean(accs)),
        "HL": float(np.mean(hls)),
    }


def lps_index_to_multilabel(pred_indices: np.ndarray, class_to_pattern: Dict[int, str]) -> np.ndarray:
    preds = []
    for idx in pred_indices:
        pat = class_to_pattern[int(idx)]
        preds.append([int(c) for c in pat])
    return np.asarray(preds, dtype=int)


def evaluate_lps_patterns(
    y_true_patterns: np.ndarray,
    y_pred_indices: np.ndarray,
    class_to_pattern: Dict[int, str],
) -> Dict[str, float]:
    # IMPORTANT: metrics are computed antibiotic-by-antibiotic (binary), not exact-pattern accuracy
    y_true_multi = np.asarray([[int(c) for c in p] for p in y_true_patterns], dtype=int)
    y_pred_multi = lps_index_to_multilabel(y_pred_indices, class_to_pattern)
    return evaluate_multilabel_metrics(y_true_multi, y_pred_multi)


# -------------------------
# Utils: training loops (Torch)
# -------------------------
@torch.no_grad()
def _predict_binary(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    logits = model(X)
    probs = torch.sigmoid(logits).view(-1)
    return (probs > 0.5).long().cpu().numpy()


@torch.no_grad()
def _predict_multilabel(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    logits = model(X)
    probs = torch.sigmoid(logits)
    return (probs > 0.5).long().cpu().numpy()


def train_binary_with_early_stopping(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    max_epochs: int = 1200,
    patience: int = 50,
    batch_size: int = 128,
) -> Tuple[Dict[str, Any], float]:
    criterion = nn.BCEWithLogitsLoss()

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    Xva = torch.tensor(X_val, dtype=torch.float32).to(device)
    yva_np = y_val.astype(int)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    best_state = None
    best_score = -np.inf
    bad = 0

    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds = _predict_binary(model, Xva)
        score = wf1_binary(yva_np, preds)

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state, float(best_score)


def train_lps_with_early_stopping(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train: np.ndarray,
    y_train_class: np.ndarray,
    X_val: np.ndarray,
    y_val_class: np.ndarray,
    class_to_pattern: Dict[int, str],
    device: torch.device,
    max_epochs: int = 1200,
    patience: int = 50,
    batch_size: int = 128,
) -> Tuple[Dict[str, Any], float]:
    criterion = nn.CrossEntropyLoss()

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train_class, dtype=torch.long).to(device)
    Xva = torch.tensor(X_val, dtype=torch.float32).to(device)
    yva = y_val_class.astype(int)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    best_state = None
    best_wf1 = -np.inf
    bad = 0

    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(Xva)
            pred_class = torch.argmax(logits_val, dim=1).cpu().numpy()

        y_true_patterns = np.array([class_to_pattern[int(i)] for i in yva], dtype=object)
        metrics = evaluate_lps_patterns(y_true_patterns, pred_class, class_to_pattern)
        wf1 = metrics["WF1"]

        if wf1 > best_wf1:
            best_wf1 = wf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state, float(best_wf1)


def train_multilabel_with_early_stopping(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    max_epochs: int = 1200,
    patience: int = 50,
    batch_size: int = 128,
) -> Tuple[Dict[str, Any], float]:
    criterion = nn.BCEWithLogitsLoss()

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train, dtype=torch.float32).to(device)
    Xva = torch.tensor(X_val, dtype=torch.float32).to(device)
    yva = y_val.astype(int)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    best_state = None
    best_wf1 = -np.inf
    bad = 0

    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds = _predict_multilabel(model, Xva)
        metrics = evaluate_multilabel_metrics(yva, preds)
        wf1 = metrics["WF1"]

        if wf1 > best_wf1:
            best_wf1 = wf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state, float(best_wf1)


# -------------------------
# Duplicates removal
# -------------------------
def remove_duplicates(X: np.ndarray, y_species: np.ndarray, amr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove duplicate rows based on (X row bytes + species + amr row).
    Returns filtered (X, y_species, amr, kept_indices).
    """
    seen = set()
    keep = []
    for i in tqdm(range(X.shape[0]), desc="Removing duplicates", leave=False, file=sys.stdout):
        h = hashlib.md5()
        h.update(X[i].tobytes())
        h.update(str(y_species[i]).encode("utf-8"))
        h.update(np.asarray(amr[i]).tobytes())  # amr row can have nans; include bytes representation
        key = h.hexdigest()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    keep = np.asarray(keep, dtype=int)
    return X[keep], y_species[keep], amr[keep], keep


# -------------------------
# Optuna search spaces (paper table)
# -------------------------
def suggest_hparams_paper(trial: optuna.Trial) -> Dict[str, Any]:
    # activation, solver, lr log [1e-6, 1e-2], layers [10, 500]
    return {
        "activation": trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"]),
        "solver": trial.suggest_categorical("solver", ["sgd", "adam"]),
        "lr": trial.suggest_float("lr", 1e-6, 1e-2, log=True),
        "layer1": trial.suggest_int("layer1", 10, 500),
        "layer2": trial.suggest_int("layer2", 10, 500),
        "layer3": trial.suggest_int("layer3", 10, 500),
    }


def make_optimizer(params: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    lr = float(params["lr"])
    if params["solver"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    return torch.optim.SGD(model.parameters(), lr=lr)


# -------------------------
# Per-species pipeline
# -------------------------
@dataclass
class SpeciesRunOutputs:
    species: str
    binary_rows: List[Dict[str, Any]]
    lps_row: Dict[str, Any]
    multi_row: Dict[str, Any]


def _optuna_with_tqdm(study: optuna.Study, objective_fn, n_trials: int, desc: str) -> None:
    """
    Run optuna with a visible tqdm bar + best value postfix.
    """
    with tqdm(total=n_trials, desc=desc, leave=False, file=sys.stdout, mininterval=0.5) as pbar:
        def callback(study_: optuna.Study, trial: optuna.trial.FrozenTrial):
            pbar.update(1)
            if study_.best_trial is not None:
                pbar.set_postfix(best=round(float(study_.best_value), 4))
        study.optimize(objective_fn, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)


def run_species_pipeline(
    species: str,
    X: np.ndarray,
    amr_df: pd.DataFrame,
    ab_list: List[str],
    outdir: str,
    test_size: float,
    random_state: int,
    n_cv: int,
    n_trials: int,
    min_pattern_count: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    optuna_sampler_seed: int = 42,
    print_folds: bool = False,
) -> SpeciesRunOutputs:
    sp_dir = os.path.join(outdir, species)
    os.makedirs(sp_dir, exist_ok=True)

    print("\n" + "=" * 90, flush=True)
    print(f"[START] Species: {species}", flush=True)
    print(f"         Antibiotics: {ab_list}", flush=True)
    print("=" * 90, flush=True)

    # ---- Filter species + required antibiotics
    print(f"[{species}] Loading + filtering...", flush=True)
    df_sp = amr_df[amr_df["species"] == species].copy()
    before_n = len(df_sp)

    df_sp = df_sp.dropna(subset=ab_list)
    after_ab_n = len(df_sp)

    idx = df_sp.index.values
    feat_ok = ~np.isnan(X[idx]).any(axis=1)
    df_sp = df_sp.loc[idx[feat_ok]]
    after_feat_n = len(df_sp)

    print(f"[{species}] isolates: {before_n} -> after ab NaNs: {after_ab_n} -> after feature NaNs: {after_feat_n}", flush=True)

    if len(df_sp) == 0:
        raise RuntimeError(f"[{species}] No data after NaN filtering.")

    # ---- Build patterns (LPS)
    print(f"[{species}] Building LPS patterns...", flush=True)
    df_sp[ab_list] = df_sp[ab_list].astype(int)
    df_sp["pattern"] = df_sp[ab_list].astype(str).agg("".join, axis=1)

    counts = df_sp["pattern"].value_counts()
    valid = counts[counts >= min_pattern_count].index
    df_sp = df_sp[df_sp["pattern"].isin(valid)].copy()

    print(f"[{species}] patterns: {counts.shape[0]} -> kept: {df_sp['pattern'].nunique()} (min_count={min_pattern_count})", flush=True)
    print(f"[{species}] isolates after pattern filtering: {len(df_sp)}", flush=True)

    if df_sp["pattern"].nunique() < 2:
        raise RuntimeError(f"[{species}] Not enough patterns after filtering (need >=2).")

    # ---- Stratified split by pattern 80/20
    print(f"[{species}] Train/test split stratified by pattern (test_size={test_size})...", flush=True)
    indices = df_sp.index.values
    y_pat = df_sp["pattern"].values
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y_pat,
        random_state=random_state,
    )
    print(f"[{species}] Train size: {len(train_idx)} | Test size: {len(test_idx)}", flush=True)

    # ---- StandardScaler fit on train only
    print(f"[{species}] StandardScaler fit on train...", flush=True)
    scaler = StandardScaler()
    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    dump(scaler, os.path.join(sp_dir, "scaler.joblib"))
    np.save(os.path.join(sp_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(sp_dir, "test_idx.npy"), test_idx)

    # -------------------------
    # 9) Binary models per antibiotic
    # -------------------------
    print(f"\n[{species}] ===== Binary training (one model per antibiotic) =====", flush=True)
    binary_rows: List[Dict[str, Any]] = []
    binary_models_dir = os.path.join(sp_dir, "binary_models")
    os.makedirs(binary_models_dir, exist_ok=True)

    for ab in tqdm(ab_list, desc=f"{species} | Binary antibiotics", file=sys.stdout, mininterval=0.5):
        y_train = df_sp.loc[train_idx, ab].astype(int).values
        y_test = df_sp.loc[test_idx, ab].astype(int).values

        print(f"\n[{species} | {ab}] Optuna search starting... (trials={n_trials}, cv={n_cv})", flush=True)

        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_state)

        def objective(trial: optuna.Trial) -> float:
            params = suggest_hparams_paper(trial)
            fold_scores = []
            for fold_id, (tr_i, va_i) in enumerate(skf.split(X_train, y_train)):
                if print_folds:
                    print(f"    [{species} | {ab}] Fold {fold_id+1}/{n_cv}", flush=True)
                X_tr, X_va = X_train[tr_i], X_train[va_i]
                y_tr, y_va = y_train[tr_i], y_train[va_i]

                model = MLPBinary(
                    input_dim=X_train.shape[1],
                    layer1=params["layer1"],
                    layer2=params["layer2"],
                    layer3=params["layer3"],
                    activation=params["activation"],
                ).to(device)
                optim = make_optimizer(params, model)

                _, best_val = train_binary_with_early_stopping(
                    model, optim,
                    X_tr, y_tr, X_va, y_va,
                    device=device,
                    max_epochs=max_epochs,
                    patience=patience,
                    batch_size=batch_size,
                )
                fold_scores.append(best_val)
            return float(np.mean(fold_scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=optuna_sampler_seed),
        )
        _optuna_with_tqdm(study, objective, n_trials=n_trials, desc=f"{species} | {ab} | Optuna")

        best_params = study.best_params
        best_cv_wf1 = float(study.best_value)

        print(f"[{species} | {ab}] [OPTUNA DONE] Best CV WF1={best_cv_wf1:.4f} | params={best_params}", flush=True)

        print(f"[{species} | {ab}] Retraining with validation early stopping (10% split)...", flush=True)
        X_trf, X_vaf, y_trf, y_vaf = train_test_split(
            X_train, y_train,
            test_size=0.1,
            stratify=y_train,
            random_state=random_state,
        )

        final_model = MLPBinary(
            input_dim=X_train.shape[1],
            layer1=best_params["layer1"],
            layer2=best_params["layer2"],
            layer3=best_params["layer3"],
            activation=best_params["activation"],
        ).to(device)
        final_optim = make_optimizer(best_params, final_model)

        _, best_val_wf1 = train_binary_with_early_stopping(
            final_model, final_optim,
            X_trf, y_trf, X_vaf, y_vaf,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
        )

        final_model.eval()
        Xte_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = _predict_binary(final_model, Xte_t)

        wf1 = wf1_binary(y_test, y_pred)
        acc = float(accuracy_score(y_test, y_pred))
        hl = float(hamming_loss(y_test, y_pred))

        print(f"[{species} | {ab}] [TEST] WF1={wf1:.4f} | ACC={acc:.4f} | HL={hl:.4f} | best_val_WF1={best_val_wf1:.4f}", flush=True)

        model_path = os.path.join(binary_models_dir, f"{species}__{ab}__binary.pt")
        torch.save(
            {
                "model_state_dict": final_model.state_dict(),
                "best_params": best_params,
                "best_cv_wf1": best_cv_wf1,
                "best_retrain_val_wf1": best_val_wf1,
                "species": species,
                "antibiotic": ab,
            },
            model_path,
        )

        pred_path = os.path.join(binary_models_dir, f"{species}__{ab}__test_preds.csv")
        pd.DataFrame({"index": test_idx, "y_true": y_test.astype(int), "y_pred": y_pred.astype(int)}).to_csv(pred_path, index=False)

        with open(os.path.join(binary_models_dir, f"{species}__{ab}__optuna_best.json"), "w") as f:
            json.dump({"best_params": best_params, "best_value": best_cv_wf1}, f, indent=2)

        binary_rows.append(
            {
                "Species": species,
                "Task": "binary",
                "Antibiotic": ab,
                "CV_WF1": best_cv_wf1,
                "Retrain_Val_WF1": best_val_wf1,
                "Test_WF1": wf1,
                "Test_ACC": acc,
                "Test_HL": hl,
                **{f"hp_{k}": v for k, v in best_params.items()},
                "model_path": model_path,
            }
        )

    # -------------------------
    # 10) LPS multiclass (patterns)
    # -------------------------
    print(f"\n[{species}] ===== LPS multiclass training (patterns) =====", flush=True)
    lps_dir = os.path.join(sp_dir, "lps_model")
    os.makedirs(lps_dir, exist_ok=True)

    df_train = df_sp.loc[train_idx]
    df_test = df_sp.loc[test_idx]

    patterns = df_train["pattern"].unique().tolist()
    pattern_to_class = {p: i for i, p in enumerate(patterns)}
    class_to_pattern = {i: p for p, i in pattern_to_class.items()}

    y_train_class = df_train["pattern"].map(pattern_to_class).values.astype(int)
    y_test_patterns = df_test["pattern"].values.astype(object)

    print(f"[{species}] LPS classes (patterns) on train: {len(pattern_to_class)}", flush=True)

    skf_lps = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_state)

    def objective_lps(trial: optuna.Trial) -> float:
        params = suggest_hparams_paper(trial)
        fold_scores = []
        for fold_id, (tr_i, va_i) in enumerate(skf_lps.split(X_train, y_train_class)):
            if print_folds:
                print(f"    [{species} | LPS] Fold {fold_id+1}/{n_cv}", flush=True)
            X_tr, X_va = X_train[tr_i], X_train[va_i]
            y_tr, y_va = y_train_class[tr_i], y_train_class[va_i]

            model = MLPLPS(
                input_dim=X_train.shape[1],
                n_classes=len(pattern_to_class),
                layer1=params["layer1"],
                layer2=params["layer2"],
                layer3=params["layer3"],
                activation=params["activation"],
            ).to(device)
            optim = make_optimizer(params, model)

            _, best_val = train_lps_with_early_stopping(
                model, optim,
                X_tr, y_tr, X_va, y_va,
                class_to_pattern=class_to_pattern,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
            )
            fold_scores.append(best_val)
        return float(np.mean(fold_scores))

    study_lps = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=optuna_sampler_seed),
    )
    _optuna_with_tqdm(study_lps, objective_lps, n_trials=n_trials, desc=f"{species} | LPS | Optuna")

    best_params_lps = study_lps.best_params
    best_cv_wf1_lps = float(study_lps.best_value)

    print(f"[{species} | LPS] [OPTUNA DONE] Best CV WF1={best_cv_wf1_lps:.4f} | params={best_params_lps}", flush=True)

    print(f"[{species} | LPS] Retraining with validation early stopping (10% split)...", flush=True)
    X_trf, X_vaf, y_trf, y_vaf = train_test_split(
        X_train, y_train_class,
        test_size=0.1,
        stratify=y_train_class,
        random_state=random_state,
    )

    final_lps = MLPLPS(
        input_dim=X_train.shape[1],
        n_classes=len(pattern_to_class),
        layer1=best_params_lps["layer1"],
        layer2=best_params_lps["layer2"],
        layer3=best_params_lps["layer3"],
        activation=best_params_lps["activation"],
    ).to(device)
    optim_lps = make_optimizer(best_params_lps, final_lps)

    _, best_val_wf1_lps = train_lps_with_early_stopping(
        final_lps, optim_lps,
        X_trf, y_trf, X_vaf, y_vaf,
        class_to_pattern=class_to_pattern,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
    )

    print(f"[{species} | LPS] Testing...", flush=True)
    final_lps.eval()
    Xte_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = final_lps(Xte_t)
        pred_class = torch.argmax(logits, dim=1).cpu().numpy()

    lps_metrics = evaluate_lps_patterns(y_test_patterns, pred_class, class_to_pattern)
    print(f"[{species} | LPS] [TEST] WF1={lps_metrics['WF1']:.4f} | ACC={lps_metrics['ACC']:.4f} | HL={lps_metrics['HL']:.4f} | best_val_WF1={best_val_wf1_lps:.4f}", flush=True)

    lps_model_path = os.path.join(lps_dir, f"{species}__lps.pt")
    torch.save(
        {
            "model_state_dict": final_lps.state_dict(),
            "best_params": best_params_lps,
            "best_cv_wf1": best_cv_wf1_lps,
            "best_retrain_val_wf1": best_val_wf1_lps,
            "pattern_to_class": pattern_to_class,
            "class_to_pattern": class_to_pattern,
            "species": species,
            "antibiotics": ab_list,
        },
        lps_model_path,
    )
    with open(os.path.join(lps_dir, f"{species}__lps_optuna_best.json"), "w") as f:
        json.dump({"best_params": best_params_lps, "best_value": best_cv_wf1_lps}, f, indent=2)

    y_pred_multi = lps_index_to_multilabel(pred_class, class_to_pattern)
    y_true_multi = np.asarray([[int(c) for c in p] for p in y_test_patterns], dtype=int)

    pd.DataFrame({"index": test_idx, "y_true_pattern": y_test_patterns, "y_pred_class": pred_class}).to_csv(
        os.path.join(lps_dir, f"{species}__lps_test_preds.csv"), index=False
    )

    cols_true = [f"true_{ab}" for ab in ab_list]
    cols_pred = [f"pred_{ab}" for ab in ab_list]
    pd.DataFrame(
        np.concatenate([y_true_multi, y_pred_multi], axis=1),
        columns=cols_true + cols_pred,
        index=test_idx,
    ).reset_index(names="index").to_csv(os.path.join(lps_dir, f"{species}__lps_test_preds_multilabel.csv"), index=False)

    lps_row = {
        "Species": species,
        "Task": "lps_multiclass",
        "Antibiotic": "ALL",
        "CV_WF1": best_cv_wf1_lps,
        "Retrain_Val_WF1": best_val_wf1_lps,
        "Test_WF1": lps_metrics["WF1"],
        "Test_ACC": lps_metrics["ACC"],
        "Test_HL": lps_metrics["HL"],
        **{f"hp_{k}": v for k, v in best_params_lps.items()},
        "model_path": lps_model_path,
        "n_patterns": int(len(pattern_to_class)),
    }

    # -------------------------
    # 11) Direct multilabel
    # -------------------------
    print(f"\n[{species}] ===== Direct multilabel training =====", flush=True)
    multi_dir = os.path.join(sp_dir, "multilabel_model")
    os.makedirs(multi_dir, exist_ok=True)

    y_train_multi = df_train[ab_list].astype(int).values
    y_test_multi = df_test[ab_list].astype(int).values

    strat_labels = df_train["pattern"].values
    skf_multi = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_state)

    def objective_multi(trial: optuna.Trial) -> float:
        params = suggest_hparams_paper(trial)
        fold_scores = []
        for fold_id, (tr_i, va_i) in enumerate(skf_multi.split(X_train, strat_labels)):
            if print_folds:
                print(f"    [{species} | DirectMulti] Fold {fold_id+1}/{n_cv}", flush=True)
            X_tr, X_va = X_train[tr_i], X_train[va_i]
            y_tr, y_va = y_train_multi[tr_i], y_train_multi[va_i]

            model = MLPMultiLabel(
                input_dim=X_train.shape[1],
                n_outputs=len(ab_list),
                layer1=params["layer1"],
                layer2=params["layer2"],
                layer3=params["layer3"],
                activation=params["activation"],
            ).to(device)
            optim = make_optimizer(params, model)

            _, best_val = train_multilabel_with_early_stopping(
                model, optim,
                X_tr, y_tr, X_va, y_va,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
            )
            fold_scores.append(best_val)
        return float(np.mean(fold_scores))

    study_multi = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=optuna_sampler_seed),
    )
    _optuna_with_tqdm(study_multi, objective_multi, n_trials=n_trials, desc=f"{species} | DirectMulti | Optuna")

    best_params_multi = study_multi.best_params
    best_cv_wf1_multi = float(study_multi.best_value)

    print(f"[{species} | DirectMulti] [OPTUNA DONE] Best CV WF1={best_cv_wf1_multi:.4f} | params={best_params_multi}", flush=True)

    print(f"[{species} | DirectMulti] Retraining with validation early stopping (10% split)...", flush=True)
    X_trf, X_vaf, y_trf, y_vaf = train_test_split(
        X_train, y_train_multi,
        test_size=0.1,
        stratify=strat_labels,
        random_state=random_state,
    )

    final_multi = MLPMultiLabel(
        input_dim=X_train.shape[1],
        n_outputs=len(ab_list),
        layer1=best_params_multi["layer1"],
        layer2=best_params_multi["layer2"],
        layer3=best_params_multi["layer3"],
        activation=best_params_multi["activation"],
    ).to(device)
    optim_multi = make_optimizer(best_params_multi, final_multi)

    _, best_val_wf1_multi = train_multilabel_with_early_stopping(
        final_multi, optim_multi,
        X_trf, y_trf, X_vaf, y_vaf,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
    )

    print(f"[{species} | DirectMulti] Testing...", flush=True)
    final_multi.eval()
    Xte_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_pred_multi = _predict_multilabel(final_multi, Xte_t)
    multi_metrics = evaluate_multilabel_metrics(y_test_multi, y_pred_multi)

    print(f"[{species} | DirectMulti] [TEST] WF1={multi_metrics['WF1']:.4f} | ACC={multi_metrics['ACC']:.4f} | HL={multi_metrics['HL']:.4f} | best_val_WF1={best_val_wf1_multi:.4f}", flush=True)

    multi_model_path = os.path.join(multi_dir, f"{species}__multilabel.pt")
    torch.save(
        {
            "model_state_dict": final_multi.state_dict(),
            "best_params": best_params_multi,
            "best_cv_wf1": best_cv_wf1_multi,
            "best_retrain_val_wf1": best_val_wf1_multi,
            "species": species,
            "antibiotics": ab_list,
        },
        multi_model_path,
    )
    with open(os.path.join(multi_dir, f"{species}__multilabel_optuna_best.json"), "w") as f:
        json.dump({"best_params": best_params_multi, "best_value": best_cv_wf1_multi}, f, indent=2)

    cols_true = [f"true_{ab}" for ab in ab_list]
    cols_pred = [f"pred_{ab}" for ab in ab_list]
    pd.DataFrame(
        np.concatenate([y_test_multi, y_pred_multi], axis=1),
        columns=cols_true + cols_pred,
        index=test_idx,
    ).reset_index(names="index").to_csv(os.path.join(multi_dir, f"{species}__multilabel_test_preds.csv"), index=False)

    multi_row = {
        "Species": species,
        "Task": "direct_multilabel",
        "Antibiotic": "ALL",
        "CV_WF1": best_cv_wf1_multi,
        "Retrain_Val_WF1": best_val_wf1_multi,
        "Test_WF1": multi_metrics["WF1"],
        "Test_ACC": multi_metrics["ACC"],
        "Test_HL": multi_metrics["HL"],
        **{f"hp_{k}": v for k, v in best_params_multi.items()},
        "model_path": multi_model_path,
    }

    all_rows = pd.DataFrame(binary_rows + [lps_row, multi_row])
    all_rows.to_csv(os.path.join(sp_dir, f"{species}__summary.csv"), index=False)

    print(f"\n[END] Species: {species} done.", flush=True)
    return SpeciesRunOutputs(species=species, binary_rows=binary_rows, lps_row=lps_row, multi_row=multi_row)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True, help="Path to DRIAMS-A pickle file")
    ap.add_argument("--outdir", required=True, help="Output directory to save models/results")
    ap.add_argument("--n_jobs", type=int, default=5, help="Parallel jobs (species-level). Recommended 1 on GPU.")
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_cv", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--min_pattern_count", type=int, default=5, help="Remove patterns with count < this value")
    ap.add_argument("--max_epochs", type=int, default=1200)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--cpu_only", action="store_true", help="Force CPU even if CUDA is available")
    ap.add_argument("--skip_duplicates_removal", action="store_true", help="Skip duplicate removal step")
    ap.add_argument("--print_folds", action="store_true", help="Print fold-by-fold progress inside CV (very verbose).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"benchmark_mlp_{run_id}")
    os.makedirs(outdir, exist_ok=True)

    device = torch.device("cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu"))
    with open(os.path.join(outdir, "run_config.json"), "w") as f:
        json.dump(vars(args) | {"device": str(device)}, f, indent=2)

    print(f"[INFO] Output: {outdir}", flush=True)
    print(f"[INFO] Device: {device}", flush=True)

    with open(args.pickle, "rb") as f:
        payload = pickle.load(f)

    X = np.asarray(payload["data"])
    y_species = np.asarray(payload["label"])
    amr = np.asarray(payload["amr"])
    antibiotics = list(payload["antibiotics"])

    print(f"[INFO] Loaded X={X.shape}, amr={amr.shape}, species={y_species.shape}, antibiotics={len(antibiotics)}", flush=True)

    if not args.skip_duplicates_removal:
        X, y_species, amr, kept_idx = remove_duplicates(X, y_species, amr)
        print(f"[INFO] After duplicates removal: X={X.shape}", flush=True)
        np.save(os.path.join(outdir, "kept_indices_after_dedup.npy"), kept_idx)

    amr_df = pd.DataFrame(amr, columns=antibiotics)
    amr_df["species"] = y_species

    for sp, ab_list in SPECIES_ANTIBIOTICS.items():
        missing = [ab for ab in ab_list if ab not in amr_df.columns]
        if missing:
            raise RuntimeError(f"[ERROR] Missing antibiotics in dataset for {sp}: {missing}")

    species_list = list(SPECIES_ANTIBIOTICS.keys())

    n_jobs = int(args.n_jobs)
    if device.type == "cuda" and n_jobs > 1:
        print("[WARN] CUDA detected and n_jobs>1. This can cause VRAM contention. Consider --n_jobs 1.", flush=True)

    print("\n================ STARTING SPECIES PIPELINES ================\n", flush=True)

    outputs: List[SpeciesRunOutputs] = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_species_pipeline)(
            species=sp,
            X=X,
            amr_df=amr_df,
            ab_list=SPECIES_ANTIBIOTICS[sp],
            outdir=outdir,
            test_size=args.test_size,
            random_state=args.random_state,
            n_cv=args.n_cv,
            n_trials=args.n_trials,
            min_pattern_count=args.min_pattern_count,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=device,
            optuna_sampler_seed=args.random_state,
            print_folds=args.print_folds,
        )
        for sp in species_list
    )

    all_rows = []
    for o in outputs:
        all_rows.extend(o.binary_rows)
        all_rows.append(o.lps_row)
        all_rows.append(o.multi_row)

    results_df = pd.DataFrame(all_rows)

    col_order = [
        "Species", "Task", "Antibiotic",
        "CV_WF1", "Retrain_Val_WF1", "Test_WF1", "Test_ACC", "Test_HL",
        "hp_activation", "hp_solver", "hp_lr", "hp_layer1", "hp_layer2", "hp_layer3",
        "n_patterns", "model_path",
    ]
    for c in col_order:
        if c not in results_df.columns:
            results_df[c] = np.nan
    results_df = results_df[col_order]

    results_df.to_csv(os.path.join(outdir, "ALL_RESULTS.csv"), index=False)

    print("\n==================== ALL RESULTS (top by Test_WF1) ====================", flush=True)
    print(results_df.sort_values("Test_WF1", ascending=False).head(30).to_string(index=False), flush=True)

    print("\n==================== SUMMARY: mean over antibiotics (binary) per species ====================", flush=True)
    bin_df = results_df[results_df["Task"] == "binary"].copy()
    if len(bin_df) > 0:
        g = bin_df.groupby("Species")[["Test_WF1", "Test_ACC", "Test_HL"]].mean().reset_index()
        print(g.sort_values("Test_WF1", ascending=False).to_string(index=False), flush=True)

    print("\n==================== SUMMARY: LPS vs Multilabel per species ====================", flush=True)
    comp_df = results_df[results_df["Task"].isin(["lps_multiclass", "direct_multilabel"])].copy()
    if len(comp_df) > 0:
        print(comp_df.sort_values(["Species", "Task"]).to_string(index=False), flush=True)

    print(f"\n[INFO] Done. Results saved in: {outdir}", flush=True)


if __name__ == "__main__":
    main()