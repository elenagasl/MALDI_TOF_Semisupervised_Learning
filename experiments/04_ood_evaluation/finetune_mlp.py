#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OOD FINE-TUNING EXPERIMENT ON MS-UMG / GÖTTINGEN

Models:
    1. Binary MLP per species-antibiotic
    2. Multilabel MLP per species with masked BCE loss

Protocol:
    For each species shared between source dataset and MS-UMG:
        1. Train source model using original dataset.
        2. Split MS-UMG samples into:
            - fine-tuning split
            - OOD test split
        3. Evaluate source-trained model on MS-UMG test.
        4. Fine-tune model on MS-UMG fine-tuning split.
        5. Evaluate fine-tuned model on MS-UMG test.

Outputs:
    - ood_global_results.csv
    - ood_species_results.csv
    - ood_species_antibiotic_results.csv
    - ood_species_summary.csv
    - ood_global_summary.json

Recommended usage:

nohup python run_ood_finetuning_mlp_msumg_gpu.py \
  --source-pickle "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl" \
  --msumg-pickle "/export/data_ml4ds/bacteria_id/MALDIAlign_Alex/MSUMG_study_full.pkl" \
  --output-dir "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/src/benchmark study/benchmark_full/benchmark_outputs/ood_msumg_finetuning_mlp_gpu" \
  --device cuda \
  --n-runs 5 \
  --finetune-frac 0.2 \
  --source-epochs 300 \
  --finetune-epochs 80 \
  --batch-size 256 \
  --val-batch-size 2048 \
  > ood_msumg_finetuning.log 2>&1 &
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import os
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_PARAMS = {
    "layer1": 128,
    "layer2": 64,
    "layer3": 32,
    "activation": "gelu",
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
}

DEFAULT_BINARY_PARAMS = {
    "layer1": 128,
    "layer2": 64,
    "layer3": 32,
    "activation": "gelu",
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
}

DEFAULT_MULTILABEL_PARAMS = {
    "layer1": 256,
    "layer2": 128,
    "layer3": 64,
    "activation": "gelu",
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
}


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class ExperimentConfig:
    n_runs: int = 5
    random_seed: int = 42

    finetune_frac: float = 0.2

    source_epochs: int = 300
    finetune_epochs: int = 80

    source_patience: int = 20
    finetune_patience: int = 12

    batch_size: int = 256
    val_batch_size: int = 2048

    min_source_obs_per_antibiotic: int = 50
    min_finetune_obs_per_antibiotic: int = 10
    min_test_obs_per_antibiotic: int = 10

    min_source_samples_species: int = 50
    min_msumg_samples_species: int = 30

    finetune_lr_factor: float = 0.1

    device: str = "auto"
    num_workers: int = 0
    use_amp: bool = True
    compile_model: bool = False


# ============================================================
# LOGGING
# ============================================================

def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "ood_finetuning_run.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


# ============================================================
# GENERAL UTILS
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device(device_arg)


def json_dump(obj: Any, path: Path) -> None:
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=convert)


def maybe_float(v: Any) -> Any:
    if v is None:
        return np.nan
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def clean_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# NAME NORMALIZATION
# ============================================================

def canonical_species_name(x: Any) -> str:
    """
    Converts:
        Escherichia_Coli -> escherichia_coli
        Escherichia coli -> escherichia_coli
        Enterobacter_cloacae_complex -> enterobacter_cloacae_complex
    """

    s = str(x).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    s_low = s.lower()

    # Harmonise Enterobacter group
    enterobacter_aliases = {
        "enterobacter_cloacae",
        "enterobacter_cloacae_complex",
        "enterobacter_hormaechei",
        "enterobacter_asburiae",
        "enterobacter_kobei",
        "enterobacter_ludwigii",
        "enterobacter_roggenkampii",
    }

    if s_low in enterobacter_aliases:
        return "enterobacter_cloacae_complex"

    return s_low


def display_species_name(canonical: str) -> str:
    if canonical == "enterobacter_cloacae_complex":
        return "Enterobacter cloacae complex"

    parts = canonical.split("_")

    if len(parts) >= 2:
        genus = parts[0].capitalize()
        species = " ".join(parts[1:])
        return f"{genus} {species}"

    return canonical.replace("_", " ").title()


def canonical_antibiotic_name(x: Any) -> str:
    """
    Robust antibiotic key:
        Ampicillin+Sulbactam -> ampicillinsulbactam
        Ampicillin-Sulbactam -> ampicillinsulbactam
        Amoxicillin-Clavulanic acid -> amoxicillinclavulanicacid
    """

    s = str(x).strip().lower()

    if s in {"combined_code", "nan", "none", ""}:
        return ""

    s = s.replace("β", "beta")
    s = s.replace("ß", "ss")
    s = re.sub(r"[^a-z0-9]", "", s)

    return s


def build_index_by_canonical(names: Sequence[str]) -> Dict[str, int]:
    mapping = {}

    for i, name in enumerate(names):
        key = canonical_antibiotic_name(name)

        if not key:
            continue

        if key not in mapping:
            mapping[key] = i

    return mapping


# ============================================================
# PICKLE LOADING
# ============================================================

def load_pickle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def coerce_X_to_2d_float32(X: Any) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        arr = X.values
    else:
        arr = np.asarray(X)

    if arr.dtype == object:
        arr = np.vstack([np.asarray(row, dtype=np.float32) for row in arr])

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"X must be 2D. Found shape: {arr.shape}")

    return arr


def normalize_single_amr_value(v: Any) -> float:
    if v is None:
        return np.nan

    if isinstance(v, (np.floating, float, np.integer, int, bool)):
        try:
            fv = float(v)
        except Exception:
            return np.nan

        if math.isnan(fv):
            return np.nan

        if fv == 0.0:
            return 0.0

        if fv == 1.0:
            return 1.0

        return np.nan

    s = str(v).strip().upper()

    if s in {"R", "I", "1", "TRUE", "RESISTANT", "RESISTANCE"}:
        return 1.0

    if s in {"S", "0", "FALSE", "SUSCEPTIBLE"}:
        return 0.0

    return np.nan


def coerce_amr_to_matrix(amr: Any) -> np.ndarray:
    if isinstance(amr, pd.DataFrame):
        arr = amr.values
    else:
        arr = np.asarray(amr)

    if arr.ndim != 2:
        raise ValueError(f"AMR must be 2D. Found shape: {arr.shape}")

    out = np.empty(arr.shape, dtype=np.float32)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            out[i, j] = normalize_single_amr_value(arr[i, j])

    return out


def load_dataset_standard(path: str, dataset_name: str) -> Dict[str, Any]:
    payload = load_pickle(path)

    required = {"data", "label", "amr", "antibiotics"}
    missing = required - set(payload.keys())

    if missing:
        raise KeyError(f"{dataset_name} pickle missing required keys: {missing}")

    X = coerce_X_to_2d_float32(payload["data"])
    labels_raw = np.asarray(payload["label"]).astype(str)
    species = np.array([canonical_species_name(x) for x in labels_raw], dtype=object)

    amr = coerce_amr_to_matrix(payload["amr"])
    antibiotics = [str(a) for a in list(payload["antibiotics"])]

    if len(X) != len(species) or len(X) != len(amr):
        raise ValueError(
            f"{dataset_name}: inconsistent lengths: X={len(X)}, labels={len(species)}, amr={len(amr)}"
        )

    if amr.shape[1] != len(antibiotics):
        raise ValueError(
            f"{dataset_name}: amr.shape[1]={amr.shape[1]} but len(antibiotics)={len(antibiotics)}"
        )

    # Remove invalid spectra only. Do NOT remove AMR NaNs.
    valid_x = np.isfinite(X).all(axis=1)

    X = X[valid_x].astype(np.float32, copy=False)
    species = species[valid_x]
    amr = amr[valid_x].astype(np.float32, copy=False)

    return {
        "X": X,
        "species": species,
        "amr": amr,
        "antibiotics": antibiotics,
        "raw_payload_keys": list(payload.keys()),
    }


# ============================================================
# ALIGN SOURCE AND MSUMG ANTIBIOTICS
# ============================================================

def align_antibiotics(
    source_antibiotics: Sequence[str],
    msumg_antibiotics: Sequence[str],
) -> Tuple[List[int], List[int], List[str]]:
    source_map = build_index_by_canonical(source_antibiotics)
    msumg_map = build_index_by_canonical(msumg_antibiotics)

    common_keys = sorted(set(source_map.keys()) & set(msumg_map.keys()))

    source_idxs = []
    msumg_idxs = []
    names = []

    for key in common_keys:
        s_idx = source_map[key]
        m_idx = msumg_map[key]

        source_idxs.append(s_idx)
        msumg_idxs.append(m_idx)

        # prefer source display name
        names.append(str(source_antibiotics[s_idx]))

    return source_idxs, msumg_idxs, names


# ============================================================
# METRICS
# ============================================================

def safe_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_score)

    if np.sum(valid) == 0:
        return np.nan

    y_true = y_true[valid]
    y_score = y_score[valid]

    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def compute_micro_macro_auc_matrix(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float, List[float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_score)

    micro_auc = np.nan

    if np.sum(valid) > 0:
        y_flat = y_true[valid]
        p_flat = y_score[valid]

        if len(np.unique(y_flat)) > 1:
            micro_auc = safe_binary_auc(y_flat, p_flat)

    aucs = []

    for j in range(y_true.shape[1]):
        auc_j = safe_binary_auc(y_true[:, j], y_score[:, j])
        aucs.append(auc_j)

    valid_aucs = [a for a in aucs if not np.isnan(a)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else np.nan

    return micro_auc, macro_auc, aucs


def count_obs_and_classes(y: np.ndarray) -> Tuple[int, int, int, bool]:
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y) & ((y == 0) | (y == 1))
    obs = y[valid]

    n = int(len(obs))
    n0 = int(np.sum(obs == 0))
    n1 = int(np.sum(obs == 1))

    return n, n0, n1, bool(n0 > 0 and n1 > 0)


# ============================================================
# MODEL
# ============================================================

def activation_from_name(name: str) -> nn.Module:
    name = str(name).lower().strip()

    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()

    return nn.GELU()


class PaperMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer1: int,
        layer2: int,
        layer3: int,
        activation: str,
    ):
        super().__init__()

        def act():
            return activation_from_name(activation)

        self.net = nn.Sequential(
            nn.Linear(input_dim, layer1),
            act(),
            nn.Dropout(0.15),

            nn.Linear(layer1, layer2),
            act(),
            nn.Dropout(0.15),

            nn.Linear(layer2, layer3),
            act(),

            nn.Linear(layer3, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def build_model(
    input_dim: int,
    output_dim: int,
    params: Dict[str, Any],
    device: torch.device,
    compile_model: bool = False,
) -> nn.Module:
    model = PaperMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        layer1=int(params["layer1"]),
        layer2=int(params["layer2"]),
        layer3=int(params["layer3"]),
        activation=str(params["activation"]),
    ).to(device)

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            logging.warning("torch.compile failed, continuing without compile: %s", e)

    return model


def build_optimizer(
    model: nn.Module,
    params: Dict[str, Any],
    lr_override: Optional[float] = None,
) -> torch.optim.Optimizer:
    lr = float(params.get("learning_rate", 1e-3))
    wd = float(params.get("weight_decay", 0.0))

    if lr_override is not None:
        lr = float(lr_override)

    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


# ============================================================
# DATALOADERS
# ============================================================

def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    x_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))

    ds = TensorDataset(x_t, y_t)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ============================================================
# TRAINING HELPERS
# ============================================================

class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = int(patience)
        self.best_loss = np.inf
        self.best_state = None
        self.counter = 0

    def step(self, loss: float, model: nn.Module) -> bool:
        if loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def masked_bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    mask = torch.isfinite(targets)

    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    clean_targets = torch.where(mask, targets, torch.zeros_like(targets))

    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        clean_targets,
        reduction="none",
    )

    return loss[mask].mean()


def train_binary_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    cfg: ExperimentConfig,
    device: torch.device,
    epochs: int,
    patience: int,
    lr_override: Optional[float] = None,
) -> nn.Module:
    optimizer = build_optimizer(model, params, lr_override=lr_override)
    criterion = nn.BCEWithLogitsLoss()

    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    loader = make_loader(
        X_train,
        y_train.reshape(-1, 1),
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    stopper = EarlyStopper(patience=patience)

    for epoch in range(epochs):
        model.train()

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb).view(-1, 1)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        val_loss = evaluate_binary_loss(model, X_val, y_val, criterion, cfg, device)

        if stopper.step(val_loss, model):
            break

    stopper.restore(model)

    return model


def evaluate_binary_loss(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    criterion: nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
) -> float:
    model.eval()

    losses = []
    use_amp = cfg.use_amp and device.type == "cuda"

    with torch.no_grad():
        for start in range(0, len(X_val), cfg.val_batch_size):
            end = min(start + cfg.val_batch_size, len(X_val))

            xb = torch.from_numpy(X_val[start:end].astype(np.float32)).to(device, non_blocking=True)
            yb = torch.from_numpy(y_val[start:end].reshape(-1, 1).astype(np.float32)).to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb).view(-1, 1)
                loss = criterion(logits, yb)

            losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else np.inf


def train_multilabel_model(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    params: Dict[str, Any],
    cfg: ExperimentConfig,
    device: torch.device,
    epochs: int,
    patience: int,
    lr_override: Optional[float] = None,
) -> nn.Module:
    optimizer = build_optimizer(model, params, lr_override=lr_override)

    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    loader = make_loader(
        X_train,
        Y_train,
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    stopper = EarlyStopper(patience=patience)

    for epoch in range(epochs):
        model.train()

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                loss = masked_bce_with_logits_loss(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        val_loss = evaluate_multilabel_loss(model, X_val, Y_val, cfg, device)

        if stopper.step(val_loss, model):
            break

    stopper.restore(model)

    return model


def evaluate_multilabel_loss(
    model: nn.Module,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> float:
    model.eval()

    losses = []
    use_amp = cfg.use_amp and device.type == "cuda"

    with torch.no_grad():
        for start in range(0, len(X_val), cfg.val_batch_size):
            end = min(start + cfg.val_batch_size, len(X_val))

            xb = torch.from_numpy(X_val[start:end].astype(np.float32)).to(device, non_blocking=True)
            yb = torch.from_numpy(Y_val[start:end].astype(np.float32)).to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                loss = masked_bce_with_logits_loss(logits, yb)

            if torch.isfinite(loss):
                losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else np.inf


# ============================================================
# PREDICTION
# ============================================================

def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    preds = []
    use_amp = cfg.use_amp and device.type == "cuda"

    with torch.no_grad():
        for start in range(0, len(X), cfg.val_batch_size):
            end = min(start + cfg.val_batch_size, len(X))

            xb = torch.from_numpy(X[start:end].astype(np.float32)).to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds.append(probs)

    if not preds:
        return np.empty((0, 1), dtype=np.float32)

    return np.concatenate(preds, axis=0)


# ============================================================
# SPECIES SPLITTING
# ============================================================

def split_msumg_species_indices(
    amr_sp: np.ndarray,
    finetune_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample-level split.

    Stratification helper:
        uses number of observed labels and mean resistance as rough bins.
        Falls back to random split if stratification fails.
    """

    n = amr_sp.shape[0]
    idx = np.arange(n)

    observed_counts = np.sum(np.isfinite(amr_sp), axis=1)
    resistance_mean = np.nanmean(amr_sp, axis=1)
    resistance_mean = np.nan_to_num(resistance_mean, nan=-1.0)

    obs_bin = pd.qcut(
        observed_counts,
        q=min(4, len(np.unique(observed_counts))),
        labels=False,
        duplicates="drop",
    )

    if np.all(resistance_mean == -1):
        strat = obs_bin.astype(str)
    else:
        try:
            res_bin = pd.qcut(
                resistance_mean,
                q=min(4, len(np.unique(resistance_mean))),
                labels=False,
                duplicates="drop",
            )
            strat = np.array([f"{a}_{b}" for a, b in zip(obs_bin, res_bin)], dtype=object)
        except Exception:
            strat = obs_bin.astype(str)

    try:
        ft_idx, test_idx = train_test_split(
            idx,
            train_size=finetune_frac,
            random_state=seed,
            shuffle=True,
            stratify=strat,
        )
    except Exception:
        ft_idx, test_idx = train_test_split(
            idx,
            train_size=finetune_frac,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    return np.asarray(ft_idx), np.asarray(test_idx)


# ============================================================
# VALID ANTIBIOTIC SELECTION
# ============================================================

def select_valid_antibiotics_for_binary(
    source_amr_sp: np.ndarray,
    ft_amr_sp: np.ndarray,
    test_amr_sp: np.ndarray,
    cfg: ExperimentConfig,
) -> List[int]:
    valid = []

    n_ab = source_amr_sp.shape[1]

    for j in range(n_ab):
        n_source, _, _, source_two = count_obs_and_classes(source_amr_sp[:, j])
        n_ft, _, _, ft_two = count_obs_and_classes(ft_amr_sp[:, j])
        n_test, _, _, test_two = count_obs_and_classes(test_amr_sp[:, j])

        if n_source < cfg.min_source_obs_per_antibiotic:
            continue
        if not source_two:
            continue

        if n_ft < cfg.min_finetune_obs_per_antibiotic:
            continue
        if not ft_two:
            continue

        if n_test < cfg.min_test_obs_per_antibiotic:
            continue
        if not test_two:
            continue

        valid.append(j)

    return valid


def select_valid_antibiotics_for_multilabel(
    source_amr_sp: np.ndarray,
    ft_amr_sp: np.ndarray,
    test_amr_sp: np.ndarray,
    cfg: ExperimentConfig,
) -> List[int]:
    """
    Same criteria as binary to keep direct comparability.
    """

    return select_valid_antibiotics_for_binary(
        source_amr_sp,
        ft_amr_sp,
        test_amr_sp,
        cfg,
    )


# ============================================================
# MAIN SPECIES EXPERIMENT
# ============================================================

def run_species_experiment(
    run_id: int,
    species: str,
    antibiotic_names: List[str],
    X_source_sp: np.ndarray,
    amr_source_sp: np.ndarray,
    X_msumg_sp: np.ndarray,
    amr_msumg_sp: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
        species_records, species_antibiotic_records
    """

    seed = cfg.random_seed + run_id * 1000 + abs(hash(species)) % 1000

    ft_idx, test_idx = split_msumg_species_indices(
        amr_msumg_sp,
        finetune_frac=cfg.finetune_frac,
        seed=seed,
    )

    X_ft = X_msumg_sp[ft_idx]
    X_test = X_msumg_sp[test_idx]

    amr_ft = amr_msumg_sp[ft_idx]
    amr_test = amr_msumg_sp[test_idx]

    valid_binary_cols = select_valid_antibiotics_for_binary(
        source_amr_sp=amr_source_sp,
        ft_amr_sp=amr_ft,
        test_amr_sp=amr_test,
        cfg=cfg,
    )

    valid_multilabel_cols = select_valid_antibiotics_for_multilabel(
        source_amr_sp=amr_source_sp,
        ft_amr_sp=amr_ft,
        test_amr_sp=amr_test,
        cfg=cfg,
    )

    n_ab = amr_source_sp.shape[1]

    binary_pred_before = np.full_like(amr_test, fill_value=np.nan, dtype=np.float32)
    binary_pred_after = np.full_like(amr_test, fill_value=np.nan, dtype=np.float32)

    multilabel_pred_before = np.full_like(amr_test, fill_value=np.nan, dtype=np.float32)
    multilabel_pred_after = np.full_like(amr_test, fill_value=np.nan, dtype=np.float32)

    species_antibiotic_records = []

    logging.info(
        "[Run %d | %s] FT samples=%d | Test samples=%d | Valid binary AB=%d | Valid multilabel AB=%d",
        run_id,
        display_species_name(species),
        len(ft_idx),
        len(test_idx),
        len(valid_binary_cols),
        len(valid_multilabel_cols),
    )

    # --------------------------------------------------------
    # Binary models
    # --------------------------------------------------------

    for j in valid_binary_cols:
        ab_name = antibiotic_names[j]

        source_col = amr_source_sp[:, j]
        ft_col = amr_ft[:, j]
        test_col = amr_test[:, j]

        source_mask = np.isfinite(source_col) & ((source_col == 0) | (source_col == 1))
        ft_mask = np.isfinite(ft_col) & ((ft_col == 0) | (ft_col == 1))
        test_mask = np.isfinite(test_col) & ((test_col == 0) | (test_col == 1))

        X_source_ab = X_source_sp[source_mask]
        y_source_ab = source_col[source_mask].astype(np.float32)

        X_ft_ab = X_ft[ft_mask]
        y_ft_ab = ft_col[ft_mask].astype(np.float32)

        X_test_ab = X_test[test_mask]
        y_test_ab = test_col[test_mask].astype(np.float32)

        n_source, source_s, source_r, _ = count_obs_and_classes(y_source_ab)
        n_ft, ft_s, ft_r, _ = count_obs_and_classes(y_ft_ab)
        n_test, test_s, test_r, _ = count_obs_and_classes(y_test_ab)

        try:
            params = DEFAULT_BINARY_PARAMS

            model = build_model(
                input_dim=X_source_sp.shape[1],
                output_dim=1,
                params=params,
                device=device,
                compile_model=cfg.compile_model,
            )

            # source validation = small random split from source observed data
            if len(y_source_ab) >= 20:
                try:
                    tr_idx, val_idx = train_test_split(
                        np.arange(len(y_source_ab)),
                        test_size=0.15,
                        random_state=seed,
                        stratify=y_source_ab,
                    )
                except Exception:
                    tr_idx, val_idx = train_test_split(
                        np.arange(len(y_source_ab)),
                        test_size=0.15,
                        random_state=seed,
                        stratify=None,
                    )
            else:
                tr_idx = np.arange(len(y_source_ab))
                val_idx = np.arange(len(y_source_ab))

            model = train_binary_model(
                model=model,
                X_train=X_source_ab[tr_idx],
                y_train=y_source_ab[tr_idx],
                X_val=X_source_ab[val_idx],
                y_val=y_source_ab[val_idx],
                params=params,
                cfg=cfg,
                device=device,
                epochs=cfg.source_epochs,
                patience=cfg.source_patience,
            )

            p_before = predict_proba(model, X_test_ab, cfg, device).reshape(-1)
            binary_pred_before[test_mask, j] = p_before.astype(np.float32)

            auc_before = safe_binary_auc(y_test_ab, p_before)

            # Fine-tune copy
            ft_model = copy.deepcopy(model).to(device)

            ft_lr = float(params["learning_rate"]) * float(cfg.finetune_lr_factor)

            # validation for FT: use source validation? Better: small split from FT if enough
            if len(y_ft_ab) >= 20:
                try:
                    ft_tr_idx, ft_val_idx = train_test_split(
                        np.arange(len(y_ft_ab)),
                        test_size=0.2,
                        random_state=seed,
                        stratify=y_ft_ab,
                    )
                except Exception:
                    ft_tr_idx, ft_val_idx = train_test_split(
                        np.arange(len(y_ft_ab)),
                        test_size=0.2,
                        random_state=seed,
                        stratify=None,
                    )
            else:
                ft_tr_idx = np.arange(len(y_ft_ab))
                ft_val_idx = np.arange(len(y_ft_ab))

            ft_model = train_binary_model(
                model=ft_model,
                X_train=X_ft_ab[ft_tr_idx],
                y_train=y_ft_ab[ft_tr_idx],
                X_val=X_ft_ab[ft_val_idx],
                y_val=y_ft_ab[ft_val_idx],
                params=params,
                cfg=cfg,
                device=device,
                epochs=cfg.finetune_epochs,
                patience=cfg.finetune_patience,
                lr_override=ft_lr,
            )

            p_after = predict_proba(ft_model, X_test_ab, cfg, device).reshape(-1)
            binary_pred_after[test_mask, j] = p_after.astype(np.float32)

            auc_after = safe_binary_auc(y_test_ab, p_after)

            del model
            del ft_model
            clean_cuda()

        except Exception as e:
            logging.exception(
                "[Run %d | %s | %s] Binary model failed: %s",
                run_id,
                display_species_name(species),
                ab_name,
                e,
            )
            auc_before = np.nan
            auc_after = np.nan

        species_antibiotic_records.append(
            {
                "run": run_id,
                "species": display_species_name(species),
                "species_key": species,
                "antibiotic": ab_name,
                "model_type": "binary",
                "n_source_obs": n_source,
                "n_source_susceptible": source_s,
                "n_source_resistant": source_r,
                "n_finetune_obs": n_ft,
                "n_finetune_susceptible": ft_s,
                "n_finetune_resistant": ft_r,
                "n_test_obs": n_test,
                "n_test_susceptible": test_s,
                "n_test_resistant": test_r,
                "ood_auc_before_finetuning": maybe_float(auc_before),
                "ood_auc_after_finetuning": maybe_float(auc_after),
                "delta_auc_after_minus_before": maybe_float(
                    auc_after - auc_before
                    if pd.notna(auc_after) and pd.notna(auc_before)
                    else np.nan
                ),
            }
        )

    # --------------------------------------------------------
    # Multilabel model
    # --------------------------------------------------------

    if len(valid_multilabel_cols) > 0:
        cols = valid_multilabel_cols
        params = DEFAULT_MULTILABEL_PARAMS

        Y_source_ml = amr_source_sp[:, cols].astype(np.float32)
        Y_ft_ml = amr_ft[:, cols].astype(np.float32)
        Y_test_ml = amr_test[:, cols].astype(np.float32)

        try:
            model = build_model(
                input_dim=X_source_sp.shape[1],
                output_dim=len(cols),
                params=params,
                device=device,
                compile_model=cfg.compile_model,
            )

            # source validation split at sample level
            src_idx = np.arange(len(X_source_sp))

            if len(src_idx) >= 20:
                tr_idx, val_idx = train_test_split(
                    src_idx,
                    test_size=0.15,
                    random_state=seed,
                    shuffle=True,
                )
            else:
                tr_idx = src_idx
                val_idx = src_idx

            model = train_multilabel_model(
                model=model,
                X_train=X_source_sp[tr_idx],
                Y_train=Y_source_ml[tr_idx],
                X_val=X_source_sp[val_idx],
                Y_val=Y_source_ml[val_idx],
                params=params,
                cfg=cfg,
                device=device,
                epochs=cfg.source_epochs,
                patience=cfg.source_patience,
            )

            p_before_local = predict_proba(model, X_test, cfg, device)
            multilabel_pred_before[:, cols] = p_before_local.astype(np.float32)

            ft_model = copy.deepcopy(model).to(device)

            ft_lr = float(params["learning_rate"]) * float(cfg.finetune_lr_factor)

            ft_sample_idx = np.arange(len(X_ft))

            if len(ft_sample_idx) >= 20:
                ft_tr_idx, ft_val_idx = train_test_split(
                    ft_sample_idx,
                    test_size=0.2,
                    random_state=seed,
                    shuffle=True,
                )
            else:
                ft_tr_idx = ft_sample_idx
                ft_val_idx = ft_sample_idx

            ft_model = train_multilabel_model(
                model=ft_model,
                X_train=X_ft[ft_tr_idx],
                Y_train=Y_ft_ml[ft_tr_idx],
                X_val=X_ft[ft_val_idx],
                Y_val=Y_ft_ml[ft_val_idx],
                params=params,
                cfg=cfg,
                device=device,
                epochs=cfg.finetune_epochs,
                patience=cfg.finetune_patience,
                lr_override=ft_lr,
            )

            p_after_local = predict_proba(ft_model, X_test, cfg, device)
            multilabel_pred_after[:, cols] = p_after_local.astype(np.float32)

            # per-antibiotic records
            _, _, aucs_before = compute_micro_macro_auc_matrix(
                y_true=amr_test,
                y_score=multilabel_pred_before,
            )

            _, _, aucs_after = compute_micro_macro_auc_matrix(
                y_true=amr_test,
                y_score=multilabel_pred_after,
            )

            for j in cols:
                ab_name = antibiotic_names[j]

                n_source, source_s, source_r, _ = count_obs_and_classes(amr_source_sp[:, j])
                n_ft, ft_s, ft_r, _ = count_obs_and_classes(amr_ft[:, j])
                n_test, test_s, test_r, _ = count_obs_and_classes(amr_test[:, j])

                auc_before = aucs_before[j]
                auc_after = aucs_after[j]

                species_antibiotic_records.append(
                    {
                        "run": run_id,
                        "species": display_species_name(species),
                        "species_key": species,
                        "antibiotic": ab_name,
                        "model_type": "multilabel",
                        "n_source_obs": n_source,
                        "n_source_susceptible": source_s,
                        "n_source_resistant": source_r,
                        "n_finetune_obs": n_ft,
                        "n_finetune_susceptible": ft_s,
                        "n_finetune_resistant": ft_r,
                        "n_test_obs": n_test,
                        "n_test_susceptible": test_s,
                        "n_test_resistant": test_r,
                        "ood_auc_before_finetuning": maybe_float(auc_before),
                        "ood_auc_after_finetuning": maybe_float(auc_after),
                        "delta_auc_after_minus_before": maybe_float(
                            auc_after - auc_before
                            if pd.notna(auc_after) and pd.notna(auc_before)
                            else np.nan
                        ),
                    }
                )

            del model
            del ft_model
            clean_cuda()

        except Exception as e:
            logging.exception(
                "[Run %d | %s] Multilabel model failed: %s",
                run_id,
                display_species_name(species),
                e,
            )

    # --------------------------------------------------------
    # Species-level metrics
    # --------------------------------------------------------

    bin_micro_before, bin_macro_before, _ = compute_micro_macro_auc_matrix(
        y_true=amr_test,
        y_score=binary_pred_before,
    )

    bin_micro_after, bin_macro_after, _ = compute_micro_macro_auc_matrix(
        y_true=amr_test,
        y_score=binary_pred_after,
    )

    ml_micro_before, ml_macro_before, _ = compute_micro_macro_auc_matrix(
        y_true=amr_test,
        y_score=multilabel_pred_before,
    )

    ml_micro_after, ml_macro_after, _ = compute_micro_macro_auc_matrix(
        y_true=amr_test,
        y_score=multilabel_pred_after,
    )

    species_records = [
        {
            "run": run_id,
            "species": display_species_name(species),
            "species_key": species,
            "model_type": "binary",
            "n_source_samples": int(len(X_source_sp)),
            "n_msumg_samples": int(len(X_msumg_sp)),
            "n_finetune_samples": int(len(X_ft)),
            "n_test_samples": int(len(X_test)),
            "n_valid_antibiotics": int(len(valid_binary_cols)),
            "valid_antibiotics": ";".join([antibiotic_names[j] for j in valid_binary_cols]),
            "ood_micro_auc_before_finetuning": maybe_float(bin_micro_before),
            "ood_macro_auc_before_finetuning": maybe_float(bin_macro_before),
            "ood_micro_auc_after_finetuning": maybe_float(bin_micro_after),
            "ood_macro_auc_after_finetuning": maybe_float(bin_macro_after),
            "delta_micro_auc_after_minus_before": maybe_float(
                bin_micro_after - bin_micro_before
                if pd.notna(bin_micro_after) and pd.notna(bin_micro_before)
                else np.nan
            ),
            "delta_macro_auc_after_minus_before": maybe_float(
                bin_macro_after - bin_macro_before
                if pd.notna(bin_macro_after) and pd.notna(bin_macro_before)
                else np.nan
            ),
            "n_test_pairs_predicted_before": int(np.sum(np.isfinite(amr_test) & np.isfinite(binary_pred_before))),
            "n_test_pairs_predicted_after": int(np.sum(np.isfinite(amr_test) & np.isfinite(binary_pred_after))),
        },
        {
            "run": run_id,
            "species": display_species_name(species),
            "species_key": species,
            "model_type": "multilabel",
            "n_source_samples": int(len(X_source_sp)),
            "n_msumg_samples": int(len(X_msumg_sp)),
            "n_finetune_samples": int(len(X_ft)),
            "n_test_samples": int(len(X_test)),
            "n_valid_antibiotics": int(len(valid_multilabel_cols)),
            "valid_antibiotics": ";".join([antibiotic_names[j] for j in valid_multilabel_cols]),
            "ood_micro_auc_before_finetuning": maybe_float(ml_micro_before),
            "ood_macro_auc_before_finetuning": maybe_float(ml_macro_before),
            "ood_micro_auc_after_finetuning": maybe_float(ml_micro_after),
            "ood_macro_auc_after_finetuning": maybe_float(ml_macro_after),
            "delta_micro_auc_after_minus_before": maybe_float(
                ml_micro_after - ml_micro_before
                if pd.notna(ml_micro_after) and pd.notna(ml_micro_before)
                else np.nan
            ),
            "delta_macro_auc_after_minus_before": maybe_float(
                ml_macro_after - ml_macro_before
                if pd.notna(ml_macro_after) and pd.notna(ml_macro_before)
                else np.nan
            ),
            "n_test_pairs_predicted_before": int(np.sum(np.isfinite(amr_test) & np.isfinite(multilabel_pred_before))),
            "n_test_pairs_predicted_after": int(np.sum(np.isfinite(amr_test) & np.isfinite(multilabel_pred_after))),
        },
    ]

    return species_records, species_antibiotic_records


# ============================================================
# SUMMARY TABLES
# ============================================================

def summarize_species_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []

    group_cols = ["species", "species_key", "model_type"]

    for keys, sub in df.groupby(group_cols):
        species, species_key, model_type = keys

        row = {
            "species": species,
            "species_key": species_key,
            "model_type": model_type,
            "n_runs": int(sub["run"].nunique()),
            "mean_source_samples": float(sub["n_source_samples"].mean()),
            "mean_msumg_samples": float(sub["n_msumg_samples"].mean()),
            "mean_finetune_samples": float(sub["n_finetune_samples"].mean()),
            "mean_test_samples": float(sub["n_test_samples"].mean()),
            "mean_valid_antibiotics": float(sub["n_valid_antibiotics"].mean()),
            "micro_auc_before_mean": float(sub["ood_micro_auc_before_finetuning"].mean(skipna=True)),
            "micro_auc_before_std": float(sub["ood_micro_auc_before_finetuning"].std(skipna=True, ddof=1)),
            "macro_auc_before_mean": float(sub["ood_macro_auc_before_finetuning"].mean(skipna=True)),
            "macro_auc_before_std": float(sub["ood_macro_auc_before_finetuning"].std(skipna=True, ddof=1)),
            "micro_auc_after_mean": float(sub["ood_micro_auc_after_finetuning"].mean(skipna=True)),
            "micro_auc_after_std": float(sub["ood_micro_auc_after_finetuning"].std(skipna=True, ddof=1)),
            "macro_auc_after_mean": float(sub["ood_macro_auc_after_finetuning"].mean(skipna=True)),
            "macro_auc_after_std": float(sub["ood_macro_auc_after_finetuning"].std(skipna=True, ddof=1)),
            "delta_micro_auc_mean": float(sub["delta_micro_auc_after_minus_before"].mean(skipna=True)),
            "delta_micro_auc_std": float(sub["delta_micro_auc_after_minus_before"].std(skipna=True, ddof=1)),
            "delta_macro_auc_mean": float(sub["delta_macro_auc_after_minus_before"].mean(skipna=True)),
            "delta_macro_auc_std": float(sub["delta_macro_auc_after_minus_before"].std(skipna=True, ddof=1)),
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            by=["model_type", "micro_auc_after_mean", "species"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    return out


def summarize_global_results(species_df: pd.DataFrame) -> pd.DataFrame:
    if species_df.empty:
        return pd.DataFrame()

    rows = []

    for run, sub_run in species_df.groupby("run"):
        for model_type, sub in sub_run.groupby("model_type"):
            rows.append(
                {
                    "run": int(run),
                    "model_type": model_type,
                    "n_species": int(sub["species"].nunique()),
                    "mean_micro_auc_before": float(sub["ood_micro_auc_before_finetuning"].mean(skipna=True)),
                    "mean_macro_auc_before": float(sub["ood_macro_auc_before_finetuning"].mean(skipna=True)),
                    "mean_micro_auc_after": float(sub["ood_micro_auc_after_finetuning"].mean(skipna=True)),
                    "mean_macro_auc_after": float(sub["ood_macro_auc_after_finetuning"].mean(skipna=True)),
                    "mean_delta_micro_auc": float(sub["delta_micro_auc_after_minus_before"].mean(skipna=True)),
                    "mean_delta_macro_auc": float(sub["delta_macro_auc_after_minus_before"].mean(skipna=True)),
                }
            )

    return pd.DataFrame(rows)


def format_pm(mean_val: float, std_val: float, decimals: int = 4) -> str:
    if mean_val is None or pd.isna(mean_val):
        return "NaN"
    if std_val is None or pd.isna(std_val):
        return f"{mean_val:.{decimals}f}"
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


def build_public_summary(species_summary: pd.DataFrame) -> pd.DataFrame:
    if species_summary.empty:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["Species"] = species_summary["species"]
    out["Model"] = species_summary["model_type"]

    out["Micro AUC before FT"] = [
        format_pm(m, s)
        for m, s in zip(species_summary["micro_auc_before_mean"], species_summary["micro_auc_before_std"])
    ]

    out["Micro AUC after FT"] = [
        format_pm(m, s)
        for m, s in zip(species_summary["micro_auc_after_mean"], species_summary["micro_auc_after_std"])
    ]

    out["Δ Micro AUC"] = [
        format_pm(m, s)
        for m, s in zip(species_summary["delta_micro_auc_mean"], species_summary["delta_micro_auc_std"])
    ]

    out["Macro AUC before FT"] = [
        format_pm(m, s)
        for m, s in zip(species_summary["macro_auc_before_mean"], species_summary["macro_auc_before_std"])
    ]

    out["Macro AUC after FT"] = [
        format_pm(m, s)
        for m, s in zip(species_summary["macro_auc_after_mean"], species_summary["macro_auc_after_std"])
    ]

    out["Δ Macro AUC"] = [
        format_pm(m, s)
        for m, s in zip(species_summary["delta_macro_auc_mean"], species_summary["delta_macro_auc_std"])
    ]

    out["Mean valid antibiotics"] = species_summary["mean_valid_antibiotics"].round(1)
    out["Mean FT samples"] = species_summary["mean_finetune_samples"].round(1)
    out["Mean test samples"] = species_summary["mean_test_samples"].round(1)

    return out


# ============================================================
# ARGPARSE
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OOD fine-tuning experiment on MS-UMG using binary and multilabel MLPs."
    )

    parser.add_argument("--source-pickle", type=str, required=True)
    parser.add_argument("--msumg-pickle", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--species", nargs="*", default=None)

    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--finetune-frac", type=float, default=0.2)

    parser.add_argument("--source-epochs", type=int, default=300)
    parser.add_argument("--finetune-epochs", type=int, default=80)

    parser.add_argument("--source-patience", type=int, default=20)
    parser.add_argument("--finetune-patience", type=int, default=12)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=2048)

    parser.add_argument("--min-source-obs-per-antibiotic", type=int, default=50)
    parser.add_argument("--min-finetune-obs-per-antibiotic", type=int, default=10)
    parser.add_argument("--min-test-obs-per-antibiotic", type=int, default=10)

    parser.add_argument("--min-source-samples-species", type=int, default=50)
    parser.add_argument("--min-msumg-samples-species", type=int, default=30)

    parser.add_argument("--finetune-lr-factor", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile-model", action="store_true")

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)

    cfg = ExperimentConfig(
        n_runs=args.n_runs,
        random_seed=args.random_seed,
        finetune_frac=args.finetune_frac,
        source_epochs=args.source_epochs,
        finetune_epochs=args.finetune_epochs,
        source_patience=args.source_patience,
        finetune_patience=args.finetune_patience,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        min_source_obs_per_antibiotic=args.min_source_obs_per_antibiotic,
        min_finetune_obs_per_antibiotic=args.min_finetune_obs_per_antibiotic,
        min_test_obs_per_antibiotic=args.min_test_obs_per_antibiotic,
        min_source_samples_species=args.min_source_samples_species,
        min_msumg_samples_species=args.min_msumg_samples_species,
        finetune_lr_factor=args.finetune_lr_factor,
        device=args.device,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        compile_model=args.compile_model,
    )

    seed_everything(cfg.random_seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    device = choose_device(cfg.device)

    logging.info("=" * 100)
    logging.info("OOD FINE-TUNING EXPERIMENT START")
    logging.info("=" * 100)
    logging.info("Args: %s", vars(args))
    logging.info("Config: %s", asdict(cfg))
    logging.info("Device: %s", device)

    if device.type == "cuda":
        logging.info("CUDA device: %s", torch.cuda.get_device_name(device))

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    logging.info("Loading source dataset: %s", args.source_pickle)
    source = load_dataset_standard(args.source_pickle, "source")

    logging.info("Loading MS-UMG dataset: %s", args.msumg_pickle)
    msumg = load_dataset_standard(args.msumg_pickle, "MS-UMG")

    source_X = source["X"]
    source_species = source["species"]
    source_amr_full = source["amr"]
    source_antibiotics = source["antibiotics"]

    msumg_X = msumg["X"]
    msumg_species = msumg["species"]
    msumg_amr_full = msumg["amr"]
    msumg_antibiotics = msumg["antibiotics"]

    logging.info("Source X shape: %s", source_X.shape)
    logging.info("Source AMR shape: %s", source_amr_full.shape)
    logging.info("MS-UMG X shape: %s", msumg_X.shape)
    logging.info("MS-UMG AMR shape: %s", msumg_amr_full.shape)

    if source_X.shape[1] != msumg_X.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: source={source_X.shape[1]}, MS-UMG={msumg_X.shape[1]}"
        )

    # --------------------------------------------------------
    # Align antibiotics
    # --------------------------------------------------------

    src_ab_idx, msumg_ab_idx, common_ab_names = align_antibiotics(
        source_antibiotics,
        msumg_antibiotics,
    )

    if len(common_ab_names) == 0:
        raise RuntimeError("No common antibiotics found between source and MS-UMG.")

    source_amr = source_amr_full[:, src_ab_idx]
    msumg_amr = msumg_amr_full[:, msumg_ab_idx]

    logging.info("Common antibiotics: %d", len(common_ab_names))
    logging.info("Common antibiotics names: %s", common_ab_names)

    pd.DataFrame(
        {
            "common_antibiotic": common_ab_names,
            "source_index": src_ab_idx,
            "msumg_index": msumg_ab_idx,
            "source_name": [source_antibiotics[i] for i in src_ab_idx],
            "msumg_name": [msumg_antibiotics[i] for i in msumg_ab_idx],
        }
    ).to_csv(output_dir / "common_antibiotics_mapping.csv", index=False)

    # --------------------------------------------------------
    # Common species
    # --------------------------------------------------------

    source_species_set = set(source_species.tolist())
    msumg_species_set = set(msumg_species.tolist())

    common_species = sorted(source_species_set & msumg_species_set)

    if args.species:
        requested = {canonical_species_name(s) for s in args.species}
        common_species = [s for s in common_species if s in requested]

    if not common_species:
        raise RuntimeError("No common species found between source and MS-UMG.")

    # Filter species with enough samples
    filtered_species = []

    for sp in common_species:
        n_src = int(np.sum(source_species == sp))
        n_mg = int(np.sum(msumg_species == sp))

        if n_src < cfg.min_source_samples_species:
            continue

        if n_mg < cfg.min_msumg_samples_species:
            continue

        filtered_species.append(sp)

    common_species = filtered_species

    logging.info("Common species selected: %d", len(common_species))
    logging.info("Common species: %s", [display_species_name(s) for s in common_species])

    pd.DataFrame(
        {
            "species_key": common_species,
            "species": [display_species_name(s) for s in common_species],
            "n_source_samples": [int(np.sum(source_species == s)) for s in common_species],
            "n_msumg_samples": [int(np.sum(msumg_species == s)) for s in common_species],
        }
    ).to_csv(output_dir / "common_species_counts.csv", index=False)

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    all_species_records = []
    all_species_antibiotic_records = []

    t_start = time.time()

    for run_id in range(cfg.n_runs):
        logging.info("=" * 100)
        logging.info("RUN %d/%d", run_id + 1, cfg.n_runs)
        logging.info("=" * 100)

        for sp_i, species in enumerate(common_species, start=1):
            logging.info("-" * 100)
            logging.info(
                "Run %d/%d | Species %d/%d | %s",
                run_id + 1,
                cfg.n_runs,
                sp_i,
                len(common_species),
                display_species_name(species),
            )

            source_mask = source_species == species
            msumg_mask = msumg_species == species

            X_source_sp = source_X[source_mask]
            amr_source_sp = source_amr[source_mask]

            X_msumg_sp = msumg_X[msumg_mask]
            amr_msumg_sp = msumg_amr[msumg_mask]

            try:
                species_records, species_ab_records = run_species_experiment(
                    run_id=run_id,
                    species=species,
                    antibiotic_names=common_ab_names,
                    X_source_sp=X_source_sp,
                    amr_source_sp=amr_source_sp,
                    X_msumg_sp=X_msumg_sp,
                    amr_msumg_sp=amr_msumg_sp,
                    cfg=cfg,
                    device=device,
                )

                all_species_records.extend(species_records)
                all_species_antibiotic_records.extend(species_ab_records)

            except Exception as e:
                logging.exception(
                    "[Run %d | %s] Species experiment failed: %s",
                    run_id,
                    display_species_name(species),
                    e,
                )

            # Save intermediate outputs
            pd.DataFrame(all_species_records).to_csv(
                output_dir / "ood_species_results.csv",
                index=False,
            )

            pd.DataFrame(all_species_antibiotic_records).to_csv(
                output_dir / "ood_species_antibiotic_results.csv",
                index=False,
            )

            clean_cuda()

    # --------------------------------------------------------
    # Final summaries
    # --------------------------------------------------------

    df_species = pd.DataFrame(all_species_records)
    df_ab = pd.DataFrame(all_species_antibiotic_records)

    df_species.to_csv(output_dir / "ood_species_results.csv", index=False)
    df_ab.to_csv(output_dir / "ood_species_antibiotic_results.csv", index=False)

    df_species_summary = summarize_species_results(df_species)
    df_species_summary.to_csv(output_dir / "ood_species_summary.csv", index=False)

    df_public = build_public_summary(df_species_summary)
    df_public.to_csv(output_dir / "ood_species_summary_public_table.csv", index=False)

    with open(output_dir / "ood_species_summary_public_table.md", "w", encoding="utf-8") as f:
        if not df_public.empty:
            f.write(df_public.to_markdown(index=False))
        else:
            f.write("No results.")

    df_global_runs = summarize_global_results(df_species)
    df_global_runs.to_csv(output_dir / "ood_global_results.csv", index=False)

    global_summary = {
        "config": asdict(cfg),
        "source_pickle": args.source_pickle,
        "msumg_pickle": args.msumg_pickle,
        "device": str(device),
        "n_common_antibiotics": int(len(common_ab_names)),
        "common_antibiotics": common_ab_names,
        "n_common_species": int(len(common_species)),
        "common_species": [display_species_name(s) for s in common_species],
        "elapsed_seconds": float(time.time() - t_start),
    }

    if not df_global_runs.empty:
        for model_type, sub in df_global_runs.groupby("model_type"):
            global_summary[model_type] = {
                "mean_micro_auc_before_mean": float(sub["mean_micro_auc_before"].mean(skipna=True)),
                "mean_micro_auc_before_std": float(sub["mean_micro_auc_before"].std(skipna=True, ddof=1)),
                "mean_macro_auc_before_mean": float(sub["mean_macro_auc_before"].mean(skipna=True)),
                "mean_macro_auc_before_std": float(sub["mean_macro_auc_before"].std(skipna=True, ddof=1)),
                "mean_micro_auc_after_mean": float(sub["mean_micro_auc_after"].mean(skipna=True)),
                "mean_micro_auc_after_std": float(sub["mean_micro_auc_after"].std(skipna=True, ddof=1)),
                "mean_macro_auc_after_mean": float(sub["mean_macro_auc_after"].mean(skipna=True)),
                "mean_macro_auc_after_std": float(sub["mean_macro_auc_after"].std(skipna=True, ddof=1)),
                "mean_delta_micro_auc_mean": float(sub["mean_delta_micro_auc"].mean(skipna=True)),
                "mean_delta_micro_auc_std": float(sub["mean_delta_micro_auc"].std(skipna=True, ddof=1)),
                "mean_delta_macro_auc_mean": float(sub["mean_delta_macro_auc"].mean(skipna=True)),
                "mean_delta_macro_auc_std": float(sub["mean_delta_macro_auc"].std(skipna=True, ddof=1)),
            }

    json_dump(global_summary, output_dir / "ood_global_summary.json")

    logging.info("=" * 100)
    logging.info("OOD FINE-TUNING EXPERIMENT COMPLETED")
    logging.info("=" * 100)
    logging.info("Output dir: %s", output_dir)
    logging.info("Species results: %s", output_dir / "ood_species_results.csv")
    logging.info("Species-antibiotic results: %s", output_dir / "ood_species_antibiotic_results.csv")
    logging.info("Species summary: %s", output_dir / "ood_species_summary.csv")
    logging.info("Public table: %s", output_dir / "ood_species_summary_public_table.csv")
    logging.info("Global results: %s", output_dir / "ood_global_results.csv")
    logging.info("Global summary: %s", output_dir / "ood_global_summary.json")


if __name__ == "__main__":
    main()