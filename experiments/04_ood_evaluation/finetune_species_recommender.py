#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OOD GÖTTINGEN / MS-UMG EXPERIMENT
SPECIES-SPECIFIC RECOMMENDER WITH MLP INTERACTOR

Source training domain:
    MARISMA + DRIAMS combined pickle

OOD target domain:
    Göttingen / MS-UMG pickle

Model per species:
    Input = (MALDI spectrum, antibiotic)

    MALDI embedder:
        6000 -> 512 -> 256 -> 128 -> 64

    Antibiotic embedding:
        antibiotic_id -> 30

    Interaction MLP:
        concat(z_MALDI, z_AB) = 64 + 30 = 94
        94 -> 64 -> 32 -> 1

    Output:
        resistance logit -> sigmoid -> P(resistant)

Protocol:
    For each species shared between source and MS-UMG:
        1. Train source model using MARISMA + DRIAMS.
        2. Split MS-UMG samples into:
            - fine-tuning subset
            - OOD test subset
        3. Evaluate source-trained model on MS-UMG test.
        4. Fine-tune on MS-UMG fine-tuning subset.
        5. Re-evaluate on the same MS-UMG test subset.

Metrics:
    - Micro AUC before fine-tuning
    - Macro AUC before fine-tuning
    - Micro AUC after fine-tuning
    - Macro AUC after fine-tuning
    - Delta micro/macro AUC

Recommended command:

nohup python run_ood_gottingen_species_recommender_mlp_interactor_gpu.py \
  --source-pickle "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl" \
  --ood-pickle "/export/data_ml4ds/bacteria_id/MALDIAlign_Alex/MSUMG_study_full.pkl" \
  --output-dir "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/src/benchmark study/benchmark_full/benchmark_outputs/ood_gottingen_species_recommender_mlp_interactor" \
  --device cuda \
  --n-runs 5 \
  --finetune-frac 0.2 \
  --source-epochs 1200 \
  --finetune-epochs 200 \
  --source-patience 50 \
  --finetune-patience 25 \
  --batch-size 128 \
  --val-batch-size 2048 \
  > ood_gottingen_recommender_mlp.log 2>&1 &
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ExperimentConfig:
    n_runs: int = 5
    random_seed: int = 42

    finetune_frac: float = 0.2

    source_epochs: int = 1200
    finetune_epochs: int = 200

    source_patience: int = 50
    finetune_patience: int = 25

    batch_size: int = 128
    val_batch_size: int = 2048

    lr: float = 1e-3
    finetune_lr_factor: float = 0.1
    weight_decay: float = 0.0

    min_source_samples_species: int = 50
    min_ood_samples_species: int = 50

    min_source_obs_per_antibiotic: int = 50
    min_finetune_obs_per_antibiotic: int = 10
    min_test_obs_per_antibiotic: int = 10

    maldi_hidden_dims: Tuple[int, ...] = (512, 256, 128, 64)
    maldi_emb_dim: int = 64
    antibiotic_emb_dim: int = 30
    interactor_hidden_dims: Tuple[int, ...] = (64, 32)

    dropout: float = 0.2

    device: str = "auto"
    num_workers: int = 0
    use_amp: bool = True
    compile_model: bool = False


# ============================================================
# LOGGING
# ============================================================

def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "ood_gottingen_recommender_mlp_run.log"

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


def clean_cuda() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


# ============================================================
# NORMALIZATION
# ============================================================

def canonical_species_name(x: Any) -> str:
    """
    Normalizes species names across source and MS-UMG.

    Examples:
        Escherichia_Coli -> escherichia_coli
        Escherichia coli -> escherichia_coli
        Enterobacter_cloacae_complex -> enterobacter_cloacae_complex
    """

    s = str(x).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    s_low = s.lower()

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


def display_species_name(x: str) -> str:
    if x == "enterobacter_cloacae_complex":
        return "Enterobacter cloacae complex"

    parts = str(x).split("_")

    if len(parts) >= 2:
        genus = parts[0].capitalize()
        species = " ".join(parts[1:])
        return f"{genus} {species}"

    return str(x).replace("_", " ").title()


def canonical_antibiotic_name(x: Any) -> str:
    """
    Robust matching key for antibiotic names.

    Examples:
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


# ============================================================
# DATA LOADING
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


def load_standard_dataset(path: str, dataset_name: str) -> Dict[str, Any]:
    payload = load_pickle(path)

    required = {"data", "label", "amr", "antibiotics"}
    missing = required - set(payload.keys())

    if missing:
        raise KeyError(f"{dataset_name} pickle missing required keys: {missing}")

    X = coerce_X_to_2d_float32(payload["data"])

    species_raw = np.asarray(payload["label"]).astype(str)
    species = np.array([canonical_species_name(x) for x in species_raw], dtype=object)

    amr = coerce_amr_to_matrix(payload["amr"])

    antibiotics = [str(a) for a in list(payload["antibiotics"])]

    if len(X) != len(species) or len(X) != len(amr):
        raise ValueError(
            f"{dataset_name}: inconsistent lengths between X, species and AMR."
        )

    if amr.shape[1] != len(antibiotics):
        raise ValueError(
            f"{dataset_name}: amr.shape[1]={amr.shape[1]} but len(antibiotics)={len(antibiotics)}"
        )

    # Remove invalid spectra only. Do not remove AMR NaNs.
    valid_x = np.isfinite(X).all(axis=1)

    X = X[valid_x].astype(np.float32, copy=False)
    species = species[valid_x]
    amr = amr[valid_x].astype(np.float32, copy=False)

    return {
        "X": X,
        "species": species,
        "amr": amr,
        "antibiotics": antibiotics,
        "payload_keys": list(payload.keys()),
    }


def align_antibiotics(
    source_antibiotics: Sequence[str],
    ood_antibiotics: Sequence[str],
) -> Tuple[List[int], List[int], List[str]]:
    source_map = build_index_by_canonical(source_antibiotics)
    ood_map = build_index_by_canonical(ood_antibiotics)

    common_keys = sorted(set(source_map.keys()) & set(ood_map.keys()))

    source_idxs = []
    ood_idxs = []
    names = []

    for key in common_keys:
        s_idx = source_map[key]
        o_idx = ood_map[key]

        source_idxs.append(s_idx)
        ood_idxs.append(o_idx)

        names.append(str(source_antibiotics[s_idx]))

    return source_idxs, ood_idxs, names


# ============================================================
# PAIR DATASET
# ============================================================

class PairRecommenderDataset(Dataset):
    """
    Each item:
        (MALDI spectrum, antibiotic_id) -> AMR label

    Expands all observed sample-antibiotic pairs.
    """

    def __init__(
        self,
        X: np.ndarray,
        amr: np.ndarray,
        valid_cols: Sequence[int],
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.amr = np.asarray(amr, dtype=np.float32)
        self.valid_cols = list(valid_cols)

        local_amr = self.amr[:, self.valid_cols]

        valid = np.isfinite(local_amr) & ((local_amr == 0) | (local_amr == 1))

        sample_idx, local_drug_idx = np.where(valid)

        labels = local_amr[sample_idx, local_drug_idx].astype(np.float32)

        self.sample_idx = sample_idx.astype(np.int64)
        self.drug_idx = local_drug_idx.astype(np.int64)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        sample_id = self.sample_idx[idx]
        drug_id = self.drug_idx[idx]

        return (
            torch.from_numpy(self.X[sample_id]).float(),
            torch.tensor(drug_id, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def make_pair_loader(
    X: np.ndarray,
    amr: np.ndarray,
    valid_cols: Sequence[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = PairRecommenderDataset(
        X=X,
        amr=amr,
        valid_cols=valid_cols,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ============================================================
# MODEL
# ============================================================

class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float,
        final_activation: bool = False,
    ):
        super().__init__()

        dims = [input_dim] + list(hidden_dims) + [output_dim]

        layers = []

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))

        if final_activation:
            layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class SpeciesRecommenderMLPInteractor(nn.Module):
    """
    MALDI embedder:
        6000 -> 512 -> 256 -> 128 -> 64

    Antibiotic embedding:
        antibiotic_id -> 30

    Interaction MLP:
        64 + 30 -> 64 -> 32 -> 1
    """

    def __init__(
        self,
        input_dim: int,
        num_antibiotics: int,
        maldi_hidden_dims: Sequence[int],
        maldi_emb_dim: int,
        antibiotic_emb_dim: int,
        interactor_hidden_dims: Sequence[int],
        dropout: float,
    ):
        super().__init__()

        if len(maldi_hidden_dims) > 0 and maldi_hidden_dims[-1] == maldi_emb_dim:
            encoder_hidden = list(maldi_hidden_dims[:-1])
        else:
            encoder_hidden = list(maldi_hidden_dims)

        self.maldi_encoder = MLPBlock(
            input_dim=input_dim,
            hidden_dims=encoder_hidden,
            output_dim=maldi_emb_dim,
            dropout=dropout,
            final_activation=True,
        )

        self.antibiotic_embedding = nn.Embedding(
            num_embeddings=num_antibiotics,
            embedding_dim=antibiotic_emb_dim,
        )

        self.interactor = MLPBlock(
            input_dim=maldi_emb_dim + antibiotic_emb_dim,
            hidden_dims=interactor_hidden_dims,
            output_dim=1,
            dropout=dropout,
            final_activation=False,
        )

    def forward(self, maldi: torch.Tensor, antibiotic_id: torch.Tensor) -> torch.Tensor:
        z_maldi = self.maldi_encoder(maldi)
        z_ab = self.antibiotic_embedding(antibiotic_id.long())

        h = torch.cat([z_maldi, z_ab], dim=-1)

        logits = self.interactor(h).view(-1)

        return logits


def build_model(
    input_dim: int,
    num_antibiotics: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> nn.Module:
    model = SpeciesRecommenderMLPInteractor(
        input_dim=input_dim,
        num_antibiotics=num_antibiotics,
        maldi_hidden_dims=cfg.maldi_hidden_dims,
        maldi_emb_dim=cfg.maldi_emb_dim,
        antibiotic_emb_dim=cfg.antibiotic_emb_dim,
        interactor_hidden_dims=cfg.interactor_hidden_dims,
        dropout=cfg.dropout,
    ).to(device)

    if cfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            logging.warning("torch.compile failed. Continuing without compile. Error: %s", e)

    return model


# ============================================================
# TRAINING
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


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    amr_train: np.ndarray,
    X_val: np.ndarray,
    amr_val: np.ndarray,
    valid_cols: Sequence[int],
    cfg: ExperimentConfig,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
) -> nn.Module:
    train_loader = make_pair_loader(
        X=X_train,
        amr=amr_train,
        valid_cols=valid_cols,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_loader = make_pair_loader(
        X=X_val,
        amr=amr_val,
        valid_cols=valid_cols,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=cfg.weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss()

    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    stopper = EarlyStopper(patience=patience)

    for epoch in range(epochs):
        model.train()

        train_losses = []

        for xb, ab_id, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            ab_id = ab_id.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb, ab_id)
                loss = criterion(logits, yb.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu().item()))

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            criterion=criterion,
            cfg=cfg,
            device=device,
        )

        if (epoch + 1) % 25 == 0 or epoch == 0:
            logging.info(
                "Epoch %d/%d | train_loss=%.5f | val_loss=%.5f",
                epoch + 1,
                epochs,
                float(np.mean(train_losses)) if train_losses else np.nan,
                val_loss,
            )

        if stopper.step(val_loss, model):
            logging.info(
                "Early stopping at epoch %d | best_val_loss=%.5f",
                epoch + 1,
                stopper.best_loss,
            )
            break

    stopper.restore(model)

    return model


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
) -> float:
    model.eval()

    losses = []
    use_amp = cfg.use_amp and device.type == "cuda"

    with torch.no_grad():
        for xb, ab_id, yb in loader:
            xb = xb.to(device, non_blocking=True)
            ab_id = ab_id.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb, ab_id)
                loss = criterion(logits, yb.float())

            losses.append(float(loss.detach().cpu().item()))

    if not losses:
        return np.inf

    return float(np.mean(losses))


# ============================================================
# PREDICTION
# ============================================================

def predict_matrix(
    model: nn.Module,
    X: np.ndarray,
    num_antibiotics: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    X = np.asarray(X, dtype=np.float32)

    n_samples = X.shape[0]
    preds = np.full((n_samples, num_antibiotics), np.nan, dtype=np.float32)

    use_amp = cfg.use_amp and device.type == "cuda"

    with torch.no_grad():
        for start in range(0, n_samples, cfg.val_batch_size):
            end = min(start + cfg.val_batch_size, n_samples)

            xb = torch.from_numpy(X[start:end]).float().to(device, non_blocking=True)
            bsz = xb.shape[0]

            batch_preds = []

            for drug_id in range(num_antibiotics):
                ab_id = torch.full(
                    size=(bsz,),
                    fill_value=drug_id,
                    dtype=torch.long,
                    device=device,
                )

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(xb, ab_id)

                probs = torch.sigmoid(logits).detach().cpu().numpy()
                batch_preds.append(probs)

            batch_preds = np.stack(batch_preds, axis=1)

            preds[start:end] = batch_preds.astype(np.float32)

    return preds


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

    antibiotic_aucs = []

    for j in range(y_true.shape[1]):
        auc_j = safe_binary_auc(y_true[:, j], y_score[:, j])
        antibiotic_aucs.append(auc_j)

    valid_aucs = [a for a in antibiotic_aucs if not np.isnan(a)]

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else np.nan

    return micro_auc, macro_auc, antibiotic_aucs


def count_obs_and_classes(y: np.ndarray) -> Tuple[int, int, int, bool]:
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(y) & ((y == 0) | (y == 1))
    obs = y[valid]

    n = int(len(obs))
    n0 = int(np.sum(obs == 0))
    n1 = int(np.sum(obs == 1))

    return n, n0, n1, bool(n0 > 0 and n1 > 0)


def select_valid_antibiotics(
    source_amr_sp: np.ndarray,
    ft_amr_sp: np.ndarray,
    test_amr_sp: np.ndarray,
    cfg: ExperimentConfig,
) -> List[int]:
    valid_cols = []

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

        valid_cols.append(j)

    return valid_cols


# ============================================================
# SPLITTING
# ============================================================

def split_ood_samples(
    amr_ood_sp: np.ndarray,
    finetune_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = amr_ood_sp.shape[0]
    idx = np.arange(n)

    observed_counts = np.sum(np.isfinite(amr_ood_sp), axis=1)
    resistance_mean = np.nanmean(amr_ood_sp, axis=1)
    resistance_mean = np.nan_to_num(resistance_mean, nan=-1.0)

    try:
        obs_q = min(4, len(np.unique(observed_counts)))
        obs_bin = pd.qcut(
            observed_counts,
            q=obs_q,
            labels=False,
            duplicates="drop",
        )
    except Exception:
        obs_bin = np.zeros(n, dtype=int)

    try:
        res_q = min(4, len(np.unique(resistance_mean)))
        res_bin = pd.qcut(
            resistance_mean,
            q=res_q,
            labels=False,
            duplicates="drop",
        )
        strat = np.array([f"{a}_{b}" for a, b in zip(obs_bin, res_bin)], dtype=object)
    except Exception:
        strat = np.asarray(obs_bin).astype(str)

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


def split_train_val(
    n_samples: int,
    seed: int,
    val_size: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_samples)

    if n_samples < 5:
        return idx, idx

    try:
        train_idx, val_idx = train_test_split(
            idx,
            test_size=val_size,
            random_state=seed,
            shuffle=True,
        )
    except Exception:
        train_idx = idx
        val_idx = idx

    return np.asarray(train_idx), np.asarray(val_idx)


# ============================================================
# SPECIES EXPERIMENT
# ============================================================

def run_species_experiment(
    run_id: int,
    species: str,
    antibiotics: List[str],
    X_source_sp: np.ndarray,
    amr_source_sp: np.ndarray,
    X_ood_sp: np.ndarray,
    amr_ood_sp: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    seed = cfg.random_seed + run_id * 1000 + (abs(hash(species)) % 1000)

    ft_idx, test_idx = split_ood_samples(
        amr_ood_sp=amr_ood_sp,
        finetune_frac=cfg.finetune_frac,
        seed=seed,
    )

    X_ft = X_ood_sp[ft_idx]
    amr_ft = amr_ood_sp[ft_idx]

    X_test = X_ood_sp[test_idx]
    amr_test = amr_ood_sp[test_idx]

    valid_cols = select_valid_antibiotics(
        source_amr_sp=amr_source_sp,
        ft_amr_sp=amr_ft,
        test_amr_sp=amr_test,
        cfg=cfg,
    )

    logging.info(
        "[Run %d | %s] source=%d | ood=%d | ft=%d | test=%d | valid_ab=%d",
        run_id,
        display_species_name(species),
        len(X_source_sp),
        len(X_ood_sp),
        len(X_ft),
        len(X_test),
        len(valid_cols),
    )

    if len(valid_cols) == 0:
        record = {
            "run": run_id,
            "species": display_species_name(species),
            "species_key": species,
            "n_source_samples": int(len(X_source_sp)),
            "n_ood_samples": int(len(X_ood_sp)),
            "n_finetune_samples": int(len(X_ft)),
            "n_test_samples": int(len(X_test)),
            "n_valid_antibiotics": 0,
            "valid_antibiotics": "",
            "micro_auc_before_finetuning": np.nan,
            "macro_auc_before_finetuning": np.nan,
            "micro_auc_after_finetuning": np.nan,
            "macro_auc_after_finetuning": np.nan,
            "delta_micro_auc_after_minus_before": np.nan,
            "delta_macro_auc_after_minus_before": np.nan,
            "n_test_pairs": 0,
        }

        return [record], []

    selected_antibiotics = [antibiotics[j] for j in valid_cols]

    # --------------------------------------------------------
    # Train source model on MARISMA + DRIAMS
    # --------------------------------------------------------

    source_train_idx, source_val_idx = split_train_val(
        n_samples=len(X_source_sp),
        seed=seed,
        val_size=0.15,
    )

    model = build_model(
        input_dim=X_source_sp.shape[1],
        num_antibiotics=len(valid_cols),
        cfg=cfg,
        device=device,
    )

    model = train_model(
        model=model,
        X_train=X_source_sp[source_train_idx],
        amr_train=amr_source_sp[source_train_idx],
        X_val=X_source_sp[source_val_idx],
        amr_val=amr_source_sp[source_val_idx],
        valid_cols=valid_cols,
        cfg=cfg,
        device=device,
        epochs=cfg.source_epochs,
        patience=cfg.source_patience,
        lr=cfg.lr,
    )

    # --------------------------------------------------------
    # Evaluate before fine-tuning on MS-UMG OOD test
    # --------------------------------------------------------

    preds_before_local = predict_matrix(
        model=model,
        X=X_test,
        num_antibiotics=len(valid_cols),
        cfg=cfg,
        device=device,
    )

    amr_test_local = amr_test[:, valid_cols]

    micro_before, macro_before, aucs_before = compute_micro_macro_auc_matrix(
        y_true=amr_test_local,
        y_score=preds_before_local,
    )

    # --------------------------------------------------------
    # Fine-tune on MS-UMG fine-tuning subset
    # --------------------------------------------------------

    ft_train_idx, ft_val_idx = split_train_val(
        n_samples=len(X_ft),
        seed=seed,
        val_size=0.2,
    )

    ft_lr = cfg.lr * cfg.finetune_lr_factor

    model = train_model(
        model=model,
        X_train=X_ft[ft_train_idx],
        amr_train=amr_ft[ft_train_idx],
        X_val=X_ft[ft_val_idx],
        amr_val=amr_ft[ft_val_idx],
        valid_cols=valid_cols,
        cfg=cfg,
        device=device,
        epochs=cfg.finetune_epochs,
        patience=cfg.finetune_patience,
        lr=ft_lr,
    )

    # --------------------------------------------------------
    # Evaluate after fine-tuning on same MS-UMG OOD test
    # --------------------------------------------------------

    preds_after_local = predict_matrix(
        model=model,
        X=X_test,
        num_antibiotics=len(valid_cols),
        cfg=cfg,
        device=device,
    )

    micro_after, macro_after, aucs_after = compute_micro_macro_auc_matrix(
        y_true=amr_test_local,
        y_score=preds_after_local,
    )

    n_test_pairs = int(np.sum(np.isfinite(amr_test_local)))

    species_record = {
        "run": run_id,
        "species": display_species_name(species),
        "species_key": species,
        "n_source_samples": int(len(X_source_sp)),
        "n_ood_samples": int(len(X_ood_sp)),
        "n_finetune_samples": int(len(X_ft)),
        "n_test_samples": int(len(X_test)),
        "n_valid_antibiotics": int(len(valid_cols)),
        "valid_antibiotics": ";".join(selected_antibiotics),
        "micro_auc_before_finetuning": maybe_float(micro_before),
        "macro_auc_before_finetuning": maybe_float(macro_before),
        "micro_auc_after_finetuning": maybe_float(micro_after),
        "macro_auc_after_finetuning": maybe_float(macro_after),
        "delta_micro_auc_after_minus_before": maybe_float(
            micro_after - micro_before
            if pd.notna(micro_after) and pd.notna(micro_before)
            else np.nan
        ),
        "delta_macro_auc_after_minus_before": maybe_float(
            macro_after - macro_before
            if pd.notna(macro_after) and pd.notna(macro_before)
            else np.nan
        ),
        "n_test_pairs": n_test_pairs,
    }

    antibiotic_records = []

    for local_j, global_j in enumerate(valid_cols):
        ab_name = antibiotics[global_j]

        n_source, source_s, source_r, _ = count_obs_and_classes(amr_source_sp[:, global_j])
        n_ft, ft_s, ft_r, _ = count_obs_and_classes(amr_ft[:, global_j])
        n_test, test_s, test_r, _ = count_obs_and_classes(amr_test[:, global_j])

        auc_before = aucs_before[local_j]
        auc_after = aucs_after[local_j]

        antibiotic_records.append(
            {
                "run": run_id,
                "species": display_species_name(species),
                "species_key": species,
                "antibiotic": ab_name,
                "antibiotic_index": int(global_j),
                "n_source_obs": int(n_source),
                "n_source_susceptible": int(source_s),
                "n_source_resistant": int(source_r),
                "n_finetune_obs": int(n_ft),
                "n_finetune_susceptible": int(ft_s),
                "n_finetune_resistant": int(ft_r),
                "n_test_obs": int(n_test),
                "n_test_susceptible": int(test_s),
                "n_test_resistant": int(test_r),
                "auc_before_finetuning": maybe_float(auc_before),
                "auc_after_finetuning": maybe_float(auc_after),
                "delta_auc_after_minus_before": maybe_float(
                    auc_after - auc_before
                    if pd.notna(auc_after) and pd.notna(auc_before)
                    else np.nan
                ),
            }
        )

    del model
    clean_cuda()

    return [species_record], antibiotic_records


# ============================================================
# SUMMARIES
# ============================================================

def summarize_species(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []

    for (species, species_key), sub in df.groupby(["species", "species_key"]):
        row = {
            "species": species,
            "species_key": species_key,
            "n_runs": int(sub["run"].nunique()),
            "mean_source_samples": float(sub["n_source_samples"].mean()),
            "mean_ood_samples": float(sub["n_ood_samples"].mean()),
            "mean_finetune_samples": float(sub["n_finetune_samples"].mean()),
            "mean_test_samples": float(sub["n_test_samples"].mean()),
            "mean_valid_antibiotics": float(sub["n_valid_antibiotics"].mean()),
            "micro_auc_before_mean": float(sub["micro_auc_before_finetuning"].mean(skipna=True)),
            "micro_auc_before_std": float(sub["micro_auc_before_finetuning"].std(skipna=True, ddof=1)),
            "macro_auc_before_mean": float(sub["macro_auc_before_finetuning"].mean(skipna=True)),
            "macro_auc_before_std": float(sub["macro_auc_before_finetuning"].std(skipna=True, ddof=1)),
            "micro_auc_after_mean": float(sub["micro_auc_after_finetuning"].mean(skipna=True)),
            "micro_auc_after_std": float(sub["micro_auc_after_finetuning"].std(skipna=True, ddof=1)),
            "macro_auc_after_mean": float(sub["macro_auc_after_finetuning"].mean(skipna=True)),
            "macro_auc_after_std": float(sub["macro_auc_after_finetuning"].std(skipna=True, ddof=1)),
            "delta_micro_auc_mean": float(sub["delta_micro_auc_after_minus_before"].mean(skipna=True)),
            "delta_micro_auc_std": float(sub["delta_micro_auc_after_minus_before"].std(skipna=True, ddof=1)),
            "delta_macro_auc_mean": float(sub["delta_macro_auc_after_minus_before"].mean(skipna=True)),
            "delta_macro_auc_std": float(sub["delta_macro_auc_after_minus_before"].std(skipna=True, ddof=1)),
            "mean_test_pairs": float(sub["n_test_pairs"].mean()),
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            by=["micro_auc_after_mean", "species"],
            ascending=[False, True],
        ).reset_index(drop=True)

    return out


def summarize_global(species_df: pd.DataFrame) -> pd.DataFrame:
    if species_df.empty:
        return pd.DataFrame()

    rows = []

    for run, sub in species_df.groupby("run"):
        rows.append(
            {
                "run": int(run),
                "n_species": int(sub["species"].nunique()),
                "mean_micro_auc_before": float(sub["micro_auc_before_finetuning"].mean(skipna=True)),
                "mean_macro_auc_before": float(sub["macro_auc_before_finetuning"].mean(skipna=True)),
                "mean_micro_auc_after": float(sub["micro_auc_after_finetuning"].mean(skipna=True)),
                "mean_macro_auc_after": float(sub["macro_auc_after_finetuning"].mean(skipna=True)),
                "mean_delta_micro_auc": float(sub["delta_micro_auc_after_minus_before"].mean(skipna=True)),
                "mean_delta_macro_auc": float(sub["delta_macro_auc_after_minus_before"].mean(skipna=True)),
                "total_test_pairs": int(sub["n_test_pairs"].sum()),
            }
        )

    return pd.DataFrame(rows)


def format_pm(mean_val: float, std_val: float, decimals: int = 4) -> str:
    if mean_val is None or pd.isna(mean_val):
        return "NaN"

    if std_val is None or pd.isna(std_val):
        return f"{mean_val:.{decimals}f}"

    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


def build_public_species_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    out = pd.DataFrame()

    out["Species"] = summary["species"]

    out["Micro AUC before FT"] = [
        format_pm(m, s)
        for m, s in zip(summary["micro_auc_before_mean"], summary["micro_auc_before_std"])
    ]

    out["Micro AUC after FT"] = [
        format_pm(m, s)
        for m, s in zip(summary["micro_auc_after_mean"], summary["micro_auc_after_std"])
    ]

    out["Δ Micro AUC"] = [
        format_pm(m, s)
        for m, s in zip(summary["delta_micro_auc_mean"], summary["delta_micro_auc_std"])
    ]

    out["Macro AUC before FT"] = [
        format_pm(m, s)
        for m, s in zip(summary["macro_auc_before_mean"], summary["macro_auc_before_std"])
    ]

    out["Macro AUC after FT"] = [
        format_pm(m, s)
        for m, s in zip(summary["macro_auc_after_mean"], summary["macro_auc_after_std"])
    ]

    out["Δ Macro AUC"] = [
        format_pm(m, s)
        for m, s in zip(summary["delta_macro_auc_mean"], summary["delta_macro_auc_std"])
    ]

    out["Mean valid antibiotics"] = summary["mean_valid_antibiotics"].round(1)
    out["Mean FT samples"] = summary["mean_finetune_samples"].round(1)
    out["Mean test samples"] = summary["mean_test_samples"].round(1)
    out["Mean test pairs"] = summary["mean_test_pairs"].round(1)

    return out


# ============================================================
# ARGPARSE
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Göttingen/MS-UMG OOD experiment with species-specific recommender MLP interactor."
    )

    parser.add_argument("--source-pickle", type=str, required=True)
    parser.add_argument("--ood-pickle", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--species", nargs="*", default=None)

    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--finetune-frac", type=float, default=0.2)

    parser.add_argument("--source-epochs", type=int, default=1200)
    parser.add_argument("--finetune-epochs", type=int, default=200)
    parser.add_argument("--source-patience", type=int, default=50)
    parser.add_argument("--finetune-patience", type=int, default=25)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-batch-size", type=int, default=2048)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--min-source-samples-species", type=int, default=50)
    parser.add_argument("--min-ood-samples-species", type=int, default=50)

    parser.add_argument("--min-source-obs-per-antibiotic", type=int, default=50)
    parser.add_argument("--min-finetune-obs-per-antibiotic", type=int, default=10)
    parser.add_argument("--min-test-obs-per-antibiotic", type=int, default=10)

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
        lr=args.lr,
        finetune_lr_factor=args.finetune_lr_factor,
        weight_decay=args.weight_decay,
        min_source_samples_species=args.min_source_samples_species,
        min_ood_samples_species=args.min_ood_samples_species,
        min_source_obs_per_antibiotic=args.min_source_obs_per_antibiotic,
        min_finetune_obs_per_antibiotic=args.min_finetune_obs_per_antibiotic,
        min_test_obs_per_antibiotic=args.min_test_obs_per_antibiotic,
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
    logging.info("GÖTTINGEN/MS-UMG OOD SPECIES-SPECIFIC RECOMMENDER MLP INTERACTOR")
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
    source = load_standard_dataset(args.source_pickle, "source")

    logging.info("Loading OOD dataset: %s", args.ood_pickle)
    ood = load_standard_dataset(args.ood_pickle, "MS-UMG")

    X_source_all = source["X"]
    species_source_all = source["species"]
    amr_source_full = source["amr"]
    source_antibiotics = source["antibiotics"]

    X_ood_all = ood["X"]
    species_ood_all = ood["species"]
    amr_ood_full = ood["amr"]
    ood_antibiotics = ood["antibiotics"]

    logging.info("Source X shape: %s", X_source_all.shape)
    logging.info("Source AMR shape: %s", amr_source_full.shape)
    logging.info("OOD X shape: %s", X_ood_all.shape)
    logging.info("OOD AMR shape: %s", amr_ood_full.shape)

    if X_source_all.shape[1] != X_ood_all.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: source={X_source_all.shape[1]}, OOD={X_ood_all.shape[1]}"
        )

    # --------------------------------------------------------
    # Align antibiotics
    # --------------------------------------------------------

    src_ab_idx, ood_ab_idx, common_antibiotics = align_antibiotics(
        source_antibiotics=source_antibiotics,
        ood_antibiotics=ood_antibiotics,
    )

    if len(common_antibiotics) == 0:
        raise RuntimeError("No common antibiotics found between source and OOD.")

    amr_source_all = amr_source_full[:, src_ab_idx]
    amr_ood_all = amr_ood_full[:, ood_ab_idx]

    logging.info("Common antibiotics: %d", len(common_antibiotics))
    logging.info("Common antibiotic names: %s", common_antibiotics)

    pd.DataFrame(
        {
            "common_antibiotic": common_antibiotics,
            "source_index": src_ab_idx,
            "ood_index": ood_ab_idx,
            "source_name": [source_antibiotics[i] for i in src_ab_idx],
            "ood_name": [ood_antibiotics[i] for i in ood_ab_idx],
        }
    ).to_csv(output_dir / "common_antibiotics_mapping.csv", index=False)

    # --------------------------------------------------------
    # Common species
    # --------------------------------------------------------

    source_species_set = set(species_source_all.tolist())
    ood_species_set = set(species_ood_all.tolist())

    common_species = sorted(source_species_set & ood_species_set)

    if args.species:
        requested = {canonical_species_name(s) for s in args.species}
        common_species = [s for s in common_species if s in requested]

    filtered_species = []

    for sp in common_species:
        n_source = int(np.sum(species_source_all == sp))
        n_ood = int(np.sum(species_ood_all == sp))

        if n_source < cfg.min_source_samples_species:
            continue

        if n_ood < cfg.min_ood_samples_species:
            continue

        filtered_species.append(sp)

    common_species = filtered_species

    if not common_species:
        raise RuntimeError("No common species passed the sample-count filters.")

    species_counts_df = pd.DataFrame(
        {
            "species_key": common_species,
            "species": [display_species_name(s) for s in common_species],
            "n_source_samples": [int(np.sum(species_source_all == s)) for s in common_species],
            "n_ood_samples": [int(np.sum(species_ood_all == s)) for s in common_species],
        }
    ).sort_values("n_ood_samples", ascending=False)

    species_counts_df.to_csv(output_dir / "common_species_counts.csv", index=False)

    logging.info("Common species selected: %d", len(common_species))
    logging.info("Species: %s", [display_species_name(s) for s in common_species])

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    all_species_records = []
    all_antibiotic_records = []

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

            source_sp_mask = species_source_all == species
            ood_sp_mask = species_ood_all == species

            X_source_sp = X_source_all[source_sp_mask]
            amr_source_sp = amr_source_all[source_sp_mask]

            X_ood_sp = X_ood_all[ood_sp_mask]
            amr_ood_sp = amr_ood_all[ood_sp_mask]

            try:
                species_records, antibiotic_records = run_species_experiment(
                    run_id=run_id,
                    species=species,
                    antibiotics=common_antibiotics,
                    X_source_sp=X_source_sp,
                    amr_source_sp=amr_source_sp,
                    X_ood_sp=X_ood_sp,
                    amr_ood_sp=amr_ood_sp,
                    cfg=cfg,
                    device=device,
                )

                all_species_records.extend(species_records)
                all_antibiotic_records.extend(antibiotic_records)

            except Exception as e:
                logging.exception(
                    "Failed species experiment | run=%d | species=%s | error=%s",
                    run_id,
                    display_species_name(species),
                    e,
                )

            # Incremental save
            pd.DataFrame(all_species_records).to_csv(
                output_dir / "ood_gottingen_species_results.csv",
                index=False,
            )

            pd.DataFrame(all_antibiotic_records).to_csv(
                output_dir / "ood_gottingen_species_antibiotic_results.csv",
                index=False,
            )

            clean_cuda()

    # --------------------------------------------------------
    # Final outputs
    # --------------------------------------------------------

    df_species = pd.DataFrame(all_species_records)
    df_ab = pd.DataFrame(all_antibiotic_records)

    df_species.to_csv(output_dir / "ood_gottingen_species_results.csv", index=False)
    df_ab.to_csv(output_dir / "ood_gottingen_species_antibiotic_results.csv", index=False)

    df_species_summary = summarize_species(df_species)
    df_species_summary.to_csv(output_dir / "ood_gottingen_species_summary.csv", index=False)

    df_public = build_public_species_table(df_species_summary)
    df_public.to_csv(output_dir / "ood_gottingen_species_summary_public_table.csv", index=False)

    with open(output_dir / "ood_gottingen_species_summary_public_table.md", "w", encoding="utf-8") as f:
        if not df_public.empty:
            f.write(df_public.to_markdown(index=False))
        else:
            f.write("No results.")

    df_global = summarize_global(df_species)
    df_global.to_csv(output_dir / "ood_gottingen_global_results.csv", index=False)

    global_summary = {
        "config": asdict(cfg),
        "source_pickle": args.source_pickle,
        "ood_pickle": args.ood_pickle,
        "device": str(device),
        "n_common_antibiotics": int(len(common_antibiotics)),
        "common_antibiotics": common_antibiotics,
        "n_species": int(len(common_species)),
        "species": [display_species_name(s) for s in common_species],
        "elapsed_seconds": float(time.time() - t_start),
    }

    if not df_global.empty:
        global_summary["mean_micro_auc_before"] = float(df_global["mean_micro_auc_before"].mean(skipna=True))
        global_summary["std_micro_auc_before"] = float(df_global["mean_micro_auc_before"].std(skipna=True, ddof=1))

        global_summary["mean_macro_auc_before"] = float(df_global["mean_macro_auc_before"].mean(skipna=True))
        global_summary["std_macro_auc_before"] = float(df_global["mean_macro_auc_before"].std(skipna=True, ddof=1))

        global_summary["mean_micro_auc_after"] = float(df_global["mean_micro_auc_after"].mean(skipna=True))
        global_summary["std_micro_auc_after"] = float(df_global["mean_micro_auc_after"].std(skipna=True, ddof=1))

        global_summary["mean_macro_auc_after"] = float(df_global["mean_macro_auc_after"].mean(skipna=True))
        global_summary["std_macro_auc_after"] = float(df_global["mean_macro_auc_after"].std(skipna=True, ddof=1))

        global_summary["mean_delta_micro_auc"] = float(df_global["mean_delta_micro_auc"].mean(skipna=True))
        global_summary["std_delta_micro_auc"] = float(df_global["mean_delta_micro_auc"].std(skipna=True, ddof=1))

        global_summary["mean_delta_macro_auc"] = float(df_global["mean_delta_macro_auc"].mean(skipna=True))
        global_summary["std_delta_macro_auc"] = float(df_global["mean_delta_macro_auc"].std(skipna=True, ddof=1))

    json_dump(global_summary, output_dir / "ood_gottingen_global_summary.json")

    logging.info("=" * 100)
    logging.info("GÖTTINGEN/MS-UMG OOD EXPERIMENT COMPLETED")
    logging.info("=" * 100)
    logging.info("Output dir: %s", output_dir)
    logging.info("Species results: %s", output_dir / "ood_gottingen_species_results.csv")
    logging.info("Species-antibiotic results: %s", output_dir / "ood_gottingen_species_antibiotic_results.csv")
    logging.info("Species summary: %s", output_dir / "ood_gottingen_species_summary.csv")
    logging.info("Public table: %s", output_dir / "ood_gottingen_species_summary_public_table.csv")
    logging.info("Global results: %s", output_dir / "ood_gottingen_global_results.csv")
    logging.info("Global summary: %s", output_dir / "ood_gottingen_global_summary.json")


if __name__ == "__main__":
    main()