#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OOD Evaluation — Global Species-Conditioned Recommender (Göttingen / MS-UMG)
=============================================================================

Evaluates the external generalisation of the global species-conditioned
recommender under cross-site out-of-distribution (OOD) conditions
(Paper Objective 4).

Source training domain : MARISMA + DRIAMS (combined pickle)
OOD target domain      : Göttingen / MS-UMG independent cohort

Model architecture
------------------
Input = (MALDI spectrum [6000 bins], species_id, antibiotic_id)

    MALDI encoder:       6000 → 128 → 64 → 32
    Species embedding:   species_id → R^{16}
    Antibiotic embedding: antibiotic_id → R^{16}
    Fusion MLP:          [32 + 16 + 16] → 128 → 64 → 1
    Output:              logit → sigmoid → P(resistant)

Evaluation protocol (per run)
------------------------------
1. Identify species and antibiotics present in both source and Göttingen.
2. Train the global recommender on the full source dataset.
3. Split the Göttingen cohort per species into:
    - Fine-tuning subset  (``finetune_frac`` of each species)
    - OOD test subset     (remaining samples)
4. Evaluate the source-trained model on Göttingen test (zero-shot OOD).
5. Fine-tune the model on the Göttingen fine-tuning subset.
6. Re-evaluate the fine-tuned model on the same test subset.

Metrics
-------
Reported before and after fine-tuning:
    - Global micro AUC (all observed sample–antibiotic pairs)
    - Global macro AUC (mean per-antibiotic AUC)
    - Per-species micro/macro AUC
    - Per-antibiotic AUC

Example usage
-------------
nohup python finetune_global_recommender.py \\
  --source-pickle "/data/COMBINED_MARISMA_DRIAMS_samples.pkl" \\
  --ood-pickle "/data/MSUMG_study_full.pkl" \\
  --output-dir "/results/ood_global_recommender" \\
  --device cuda --n-runs 5 --finetune-frac 0.2 \\
  --source-epochs 300 --finetune-epochs 100 \\
  --source-patience 15 --finetune-patience 10 \\
  --batch-size 128 --val-batch-size 2048 \\
  > ood_global_recommender.log 2>&1 &
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

    source_epochs: int = 300
    finetune_epochs: int = 100

    source_patience: int = 15
    finetune_patience: int = 10

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

    drug_emb_dim: int = 16
    species_emb_dim: int = 16
    fusion_hidden_dims: Tuple[int, ...] = (128, 64)

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

    log_file = output_dir / "ood_gottingen_global_recommender_run.log"

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


# ============================================================
# NORMALIZATION
# ============================================================

def canonical_species_name(x: Any) -> str:
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
# DATASET
# ============================================================

class GlobalRecDataset(Dataset):
    """
    Each item:
        (sample, species_id, antibiotic_id) -> resistance label
    """

    def __init__(
        self,
        X: np.ndarray,
        species_ids: np.ndarray,
        amr: np.ndarray,
        valid_cols: Sequence[int],
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.species_ids = np.asarray(species_ids, dtype=np.int64)
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
            torch.tensor(self.species_ids[sample_id], dtype=torch.long),
            torch.tensor(drug_id, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def make_loader(
    X: np.ndarray,
    species_ids: np.ndarray,
    amr: np.ndarray,
    valid_cols: Sequence[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = GlobalRecDataset(
        X=X,
        species_ids=species_ids,
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

class MALDIEncoder(nn.Module):
    """
    MALDI encoder:
        input_dim -> 128 -> 64 -> 32
    """

    def __init__(self, input_dim: int, dropout: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class GlobalSpeciesNCF(nn.Module):
    """
    Global species-conditioned recommender.

    MALDI spectrum -> 32
    species ID -> species embedding
    antibiotic ID -> antibiotic embedding

    concat -> MLP -> resistance logit
    """

    def __init__(
        self,
        num_feat: int,
        num_items: int,
        num_species: int,
        drug_emb_dim: int,
        species_emb_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ):
        super().__init__()

        self.maldi_encoder = MALDIEncoder(
            input_dim=num_feat,
            dropout=dropout,
        )

        self.drug_embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=drug_emb_dim,
        )

        self.species_embedding = nn.Embedding(
            num_embeddings=num_species,
            embedding_dim=species_emb_dim,
        )

        fusion_input_dim = 32 + drug_emb_dim + species_emb_dim

        dims = [fusion_input_dim] + list(hidden_dims) + [1]

        layers = []

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        maldi: torch.Tensor,
        species_id: torch.Tensor,
        drug_id: torch.Tensor,
    ) -> torch.Tensor:
        maldi_emb = self.maldi_encoder(maldi)
        species_emb = self.species_embedding(species_id.long())
        drug_emb = self.drug_embedding(drug_id.long())

        x = torch.cat(
            [maldi_emb, species_emb, drug_emb],
            dim=-1,
        )

        logits = self.mlp(x).view(-1)

        return logits


def build_model(
    num_feat: int,
    num_items: int,
    num_species: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> nn.Module:
    model = GlobalSpeciesNCF(
        num_feat=num_feat,
        num_items=num_items,
        num_species=num_species,
        drug_emb_dim=cfg.drug_emb_dim,
        species_emb_dim=cfg.species_emb_dim,
        hidden_dims=cfg.fusion_hidden_dims,
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
    species_train: np.ndarray,
    amr_train: np.ndarray,
    X_val: np.ndarray,
    species_val: np.ndarray,
    amr_val: np.ndarray,
    valid_cols: Sequence[int],
    cfg: ExperimentConfig,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
) -> nn.Module:
    train_loader = make_loader(
        X=X_train,
        species_ids=species_train,
        amr=amr_train,
        valid_cols=valid_cols,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_loader = make_loader(
        X=X_val,
        species_ids=species_val,
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

        for xb, sp_id, drug_id, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            sp_id = sp_id.to(device, non_blocking=True)
            drug_id = drug_id.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb, sp_id, drug_id)
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
        for xb, sp_id, drug_id, yb in loader:
            xb = xb.to(device, non_blocking=True)
            sp_id = sp_id.to(device, non_blocking=True)
            drug_id = drug_id.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb, sp_id, drug_id)
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
    species_ids: np.ndarray,
    num_antibiotics: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> np.ndarray:
    """
    Reconstructs samples x antibiotics prediction matrix.
    """

    model.eval()

    X = np.asarray(X, dtype=np.float32)
    species_ids = np.asarray(species_ids, dtype=np.int64)

    n_samples = X.shape[0]

    preds = np.full((n_samples, num_antibiotics), np.nan, dtype=np.float32)

    use_amp = cfg.use_amp and device.type == "cuda"

    with torch.no_grad():
        for start in range(0, n_samples, cfg.val_batch_size):
            end = min(start + cfg.val_batch_size, n_samples)

            xb = torch.from_numpy(X[start:end]).float().to(device, non_blocking=True)

            sp_batch = torch.from_numpy(species_ids[start:end]).long().to(device, non_blocking=True)

            bsz = xb.shape[0]

            batch_preds = []

            for drug_id in range(num_antibiotics):
                drug_batch = torch.full(
                    size=(bsz,),
                    fill_value=drug_id,
                    dtype=torch.long,
                    device=device,
                )

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(xb, sp_batch, drug_batch)

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
    source_amr_train: np.ndarray,
    source_amr_val: np.ndarray,
    ft_amr: np.ndarray,
    test_amr: np.ndarray,
    cfg: ExperimentConfig,
) -> List[int]:
    valid_cols = []

    n_ab = source_amr_train.shape[1]

    for j in range(n_ab):
        n_source_train, _, _, source_train_two = count_obs_and_classes(source_amr_train[:, j])
        n_source_val, _, _, source_val_two = count_obs_and_classes(source_amr_val[:, j])
        n_ft, _, _, ft_two = count_obs_and_classes(ft_amr[:, j])
        n_test, _, _, test_two = count_obs_and_classes(test_amr[:, j])

        if n_source_train < cfg.min_source_obs_per_antibiotic:
            continue

        if not source_train_two:
            continue

        if n_source_val < 5:
            continue

        if not source_val_two:
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


def compute_per_species_rows(
    run_id: int,
    y_true: np.ndarray,
    p_before: np.ndarray,
    p_after: np.ndarray,
    species_ids: np.ndarray,
    id_to_species: Dict[int, str],
) -> List[Dict[str, Any]]:
    rows = []

    for sp_id in sorted(np.unique(species_ids)):
        mask = species_ids == sp_id

        y_sp = y_true[mask]
        p_before_sp = p_before[mask]
        p_after_sp = p_after[mask]

        micro_before, macro_before, _ = compute_micro_macro_auc_matrix(y_sp, p_before_sp)
        micro_after, macro_after, _ = compute_micro_macro_auc_matrix(y_sp, p_after_sp)

        rows.append(
            {
                "run": run_id,
                "species_id": int(sp_id),
                "species_key": id_to_species[int(sp_id)],
                "species": display_species_name(id_to_species[int(sp_id)]),
                "n_test_samples": int(np.sum(mask)),
                "n_test_pairs": int(np.sum(np.isfinite(y_sp))),
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
            }
        )

    return rows


def compute_per_antibiotic_rows(
    run_id: int,
    y_true: np.ndarray,
    p_before: np.ndarray,
    p_after: np.ndarray,
    antibiotic_names: Sequence[str],
) -> List[Dict[str, Any]]:
    rows = []

    _, _, aucs_before = compute_micro_macro_auc_matrix(y_true, p_before)
    _, _, aucs_after = compute_micro_macro_auc_matrix(y_true, p_after)

    for j, ab_name in enumerate(antibiotic_names):
        col = y_true[:, j]
        n_obs, n_s, n_r, _ = count_obs_and_classes(col)

        auc_before = aucs_before[j]
        auc_after = aucs_after[j]

        rows.append(
            {
                "run": run_id,
                "antibiotic_id": int(j),
                "antibiotic": str(ab_name),
                "n_test_obs": int(n_obs),
                "n_test_susceptible": int(n_s),
                "n_test_resistant": int(n_r),
                "auc_before_finetuning": maybe_float(auc_before),
                "auc_after_finetuning": maybe_float(auc_after),
                "delta_auc_after_minus_before": maybe_float(
                    auc_after - auc_before
                    if pd.notna(auc_after) and pd.notna(auc_before)
                    else np.nan
                ),
            }
        )

    return rows


# ============================================================
# SPLITTING
# ============================================================

def split_train_val(
    n_samples: int,
    seed: int,
    val_size: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_samples)

    if n_samples < 5:
        return idx, idx

    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
    )

    return np.asarray(train_idx), np.asarray(val_idx)


def split_ood_by_species(
    species_ids_ood: np.ndarray,
    amr_ood: np.ndarray,
    finetune_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits OOD samples species-by-species so every species contributes
    to fine-tuning and OOD test when possible.
    """

    all_ft = []
    all_test = []

    for sp_id in sorted(np.unique(species_ids_ood)):
        sp_mask = species_ids_ood == sp_id
        local_global_idx = np.where(sp_mask)[0]

        amr_sp = amr_ood[local_global_idx]

        n = len(local_global_idx)

        if n < 5:
            all_test.extend(local_global_idx.tolist())
            continue

        observed_counts = np.sum(np.isfinite(amr_sp), axis=1)
        resistance_mean = np.nanmean(amr_sp, axis=1)
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
            ft_local, test_local = train_test_split(
                np.arange(n),
                train_size=finetune_frac,
                random_state=seed + int(sp_id),
                shuffle=True,
                stratify=strat,
            )
        except Exception:
            ft_local, test_local = train_test_split(
                np.arange(n),
                train_size=finetune_frac,
                random_state=seed + int(sp_id),
                shuffle=True,
                stratify=None,
            )

        all_ft.extend(local_global_idx[ft_local].tolist())
        all_test.extend(local_global_idx[test_local].tolist())

    return np.asarray(all_ft, dtype=np.int64), np.asarray(all_test, dtype=np.int64)


# ============================================================
# SUMMARIES
# ============================================================

def summarize_species(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []

    for (species, species_key), sub in df.groupby(["species", "species_key"]):
        rows.append(
            {
                "species": species,
                "species_key": species_key,
                "n_runs": int(sub["run"].nunique()),
                "mean_test_samples": float(sub["n_test_samples"].mean()),
                "mean_test_pairs": float(sub["n_test_pairs"].mean()),
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
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            by=["micro_auc_after_mean", "species"],
            ascending=[False, True],
        ).reset_index(drop=True)

    return out


def summarize_global(df: pd.DataFrame) -> Dict[str, Any]:
    out = {}

    if df.empty:
        return out

    for col in [
        "micro_auc_before_finetuning",
        "macro_auc_before_finetuning",
        "micro_auc_after_finetuning",
        "macro_auc_after_finetuning",
        "delta_micro_auc_after_minus_before",
        "delta_macro_auc_after_minus_before",
    ]:
        out[f"{col}_mean"] = float(df[col].mean(skipna=True))
        out[f"{col}_std"] = float(df[col].std(skipna=True, ddof=1))

    return out


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

    out["Mean test samples"] = summary["mean_test_samples"].round(1)
    out["Mean test pairs"] = summary["mean_test_pairs"].round(1)

    return out


# ============================================================
# ARGPARSE
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Göttingen/MS-UMG OOD experiment with global species-conditioned recommender."
    )

    parser.add_argument("--source-pickle", type=str, required=True)
    parser.add_argument("--ood-pickle", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--species", nargs="*", default=None)

    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--finetune-frac", type=float, default=0.2)

    parser.add_argument("--source-epochs", type=int, default=300)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--source-patience", type=int, default=15)
    parser.add_argument("--finetune-patience", type=int, default=10)

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
    logging.info("GÖTTINGEN/MS-UMG OOD GLOBAL SPECIES-CONDITIONED RECOMMENDER")
    logging.info("=" * 100)
    logging.info("Args: %s", vars(args))
    logging.info("Config: %s", asdict(cfg))
    logging.info("Device: %s", device)

    if device.type == "cuda":
        logging.info("CUDA device: %s", torch.cuda.get_device_name(device))

    # --------------------------------------------------------
    # Load source and OOD
    # --------------------------------------------------------

    logging.info("Loading source dataset: %s", args.source_pickle)
    source = load_standard_dataset(args.source_pickle, "source")

    logging.info("Loading OOD dataset: %s", args.ood_pickle)
    ood = load_standard_dataset(args.ood_pickle, "MS-UMG")

    X_source_full = source["X"]
    species_source_full = source["species"]
    amr_source_full = source["amr"]
    source_antibiotics = source["antibiotics"]

    X_ood_full = ood["X"]
    species_ood_full = ood["species"]
    amr_ood_full = ood["amr"]
    ood_antibiotics = ood["antibiotics"]

    logging.info("Source X shape: %s", X_source_full.shape)
    logging.info("Source AMR shape: %s", amr_source_full.shape)
    logging.info("OOD X shape: %s", X_ood_full.shape)
    logging.info("OOD AMR shape: %s", amr_ood_full.shape)

    if X_source_full.shape[1] != X_ood_full.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: source={X_source_full.shape[1]}, OOD={X_ood_full.shape[1]}"
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

    amr_source_full = amr_source_full[:, src_ab_idx]
    amr_ood_full = amr_ood_full[:, ood_ab_idx]

    pd.DataFrame(
        {
            "common_antibiotic": common_antibiotics,
            "source_index": src_ab_idx,
            "ood_index": ood_ab_idx,
            "source_name": [source_antibiotics[i] for i in src_ab_idx],
            "ood_name": [ood_antibiotics[i] for i in ood_ab_idx],
        }
    ).to_csv(output_dir / "common_antibiotics_mapping.csv", index=False)

    logging.info("Common antibiotics: %d", len(common_antibiotics))
    logging.info("Common antibiotic names: %s", common_antibiotics)

    # --------------------------------------------------------
    # Common species
    # --------------------------------------------------------

    source_species_set = set(species_source_full.tolist())
    ood_species_set = set(species_ood_full.tolist())

    common_species = sorted(source_species_set & ood_species_set)

    if args.species:
        requested = {canonical_species_name(s) for s in args.species}
        common_species = [s for s in common_species if s in requested]

    filtered_species = []

    for sp in common_species:
        n_source = int(np.sum(species_source_full == sp))
        n_ood = int(np.sum(species_ood_full == sp))

        if n_source < cfg.min_source_samples_species:
            continue

        if n_ood < cfg.min_ood_samples_species:
            continue

        filtered_species.append(sp)

    common_species = filtered_species

    if not common_species:
        raise RuntimeError("No common species passed the sample-count filters.")

    species_to_id = {sp: i for i, sp in enumerate(common_species)}
    id_to_species = {i: sp for sp, i in species_to_id.items()}

    pd.DataFrame(
        {
            "species_key": common_species,
            "species": [display_species_name(s) for s in common_species],
            "species_id": [species_to_id[s] for s in common_species],
            "n_source_samples": [int(np.sum(species_source_full == s)) for s in common_species],
            "n_ood_samples": [int(np.sum(species_ood_full == s)) for s in common_species],
        }
    ).sort_values("n_ood_samples", ascending=False).to_csv(
        output_dir / "common_species_counts.csv",
        index=False,
    )

    json_dump(
        {
            "species_to_id": species_to_id,
            "id_to_species": {str(k): v for k, v in id_to_species.items()},
            "common_antibiotics": common_antibiotics,
        },
        output_dir / "mappings.json",
    )

    logging.info("Common species selected: %d", len(common_species))
    logging.info("Species: %s", [display_species_name(s) for s in common_species])

    # Filter source and OOD to common species
    source_mask = np.isin(species_source_full, common_species)
    ood_mask = np.isin(species_ood_full, common_species)

    X_source = X_source_full[source_mask]
    species_source = species_source_full[source_mask]
    amr_source = amr_source_full[source_mask]

    X_ood = X_ood_full[ood_mask]
    species_ood = species_ood_full[ood_mask]
    amr_ood = amr_ood_full[ood_mask]

    species_source_ids = np.array([species_to_id[s] for s in species_source], dtype=np.int64)
    species_ood_ids = np.array([species_to_id[s] for s in species_ood], dtype=np.int64)

    logging.info("Filtered source samples: %d", len(X_source))
    logging.info("Filtered OOD samples: %d", len(X_ood))

    # --------------------------------------------------------
    # Runs
    # --------------------------------------------------------

    global_records = []
    species_records = []
    antibiotic_records = []

    t_start = time.time()

    for run_id in range(cfg.n_runs):
        logging.info("=" * 100)
        logging.info("RUN %d/%d", run_id + 1, cfg.n_runs)
        logging.info("=" * 100)

        seed = cfg.random_seed + run_id * 1000

        # Source split for early stopping
        source_train_idx, source_val_idx = split_train_val(
            n_samples=len(X_source),
            seed=seed,
            val_size=0.15,
        )

        # OOD split by species
        ft_idx, test_idx = split_ood_by_species(
            species_ids_ood=species_ood_ids,
            amr_ood=amr_ood,
            finetune_frac=cfg.finetune_frac,
            seed=seed,
        )

        X_source_train = X_source[source_train_idx]
        species_source_train = species_source_ids[source_train_idx]
        amr_source_train = amr_source[source_train_idx]

        X_source_val = X_source[source_val_idx]
        species_source_val = species_source_ids[source_val_idx]
        amr_source_val = amr_source[source_val_idx]

        X_ft = X_ood[ft_idx]
        species_ft = species_ood_ids[ft_idx]
        amr_ft = amr_ood[ft_idx]

        X_test = X_ood[test_idx]
        species_test = species_ood_ids[test_idx]
        amr_test = amr_ood[test_idx]

        valid_cols = select_valid_antibiotics(
            source_amr_train=amr_source_train,
            source_amr_val=amr_source_val,
            ft_amr=amr_ft,
            test_amr=amr_test,
            cfg=cfg,
        )

        if len(valid_cols) == 0:
            logging.warning("Run %d skipped: no valid antibiotics.", run_id)
            continue

        selected_antibiotics = [common_antibiotics[j] for j in valid_cols]

        logging.info("Run %d | valid antibiotics=%d", run_id, len(valid_cols))
        logging.info("Selected antibiotics: %s", selected_antibiotics)

        # ----------------------------------------------------
        # Train source global model
        # ----------------------------------------------------

        model = build_model(
            num_feat=X_source.shape[1],
            num_items=len(valid_cols),
            num_species=len(common_species),
            cfg=cfg,
            device=device,
        )

        model = train_model(
            model=model,
            X_train=X_source_train,
            species_train=species_source_train,
            amr_train=amr_source_train,
            X_val=X_source_val,
            species_val=species_source_val,
            amr_val=amr_source_val,
            valid_cols=valid_cols,
            cfg=cfg,
            device=device,
            epochs=cfg.source_epochs,
            patience=cfg.source_patience,
            lr=cfg.lr,
        )

        # ----------------------------------------------------
        # Evaluate before FT
        # ----------------------------------------------------

        preds_before = predict_matrix(
            model=model,
            X=X_test,
            species_ids=species_test,
            num_antibiotics=len(valid_cols),
            cfg=cfg,
            device=device,
        )

        amr_test_local = amr_test[:, valid_cols]

        micro_before, macro_before, _ = compute_micro_macro_auc_matrix(
            y_true=amr_test_local,
            y_score=preds_before,
        )

        # ----------------------------------------------------
        # Fine-tune on Göttingen FT subset
        # ----------------------------------------------------

        ft_train_idx, ft_val_idx = split_train_val(
            n_samples=len(X_ft),
            seed=seed,
            val_size=0.2,
        )

        ft_lr = cfg.lr * cfg.finetune_lr_factor

        model = train_model(
            model=model,
            X_train=X_ft[ft_train_idx],
            species_train=species_ft[ft_train_idx],
            amr_train=amr_ft[ft_train_idx],
            X_val=X_ft[ft_val_idx],
            species_val=species_ft[ft_val_idx],
            amr_val=amr_ft[ft_val_idx],
            valid_cols=valid_cols,
            cfg=cfg,
            device=device,
            epochs=cfg.finetune_epochs,
            patience=cfg.finetune_patience,
            lr=ft_lr,
        )

        # ----------------------------------------------------
        # Evaluate after FT
        # ----------------------------------------------------

        preds_after = predict_matrix(
            model=model,
            X=X_test,
            species_ids=species_test,
            num_antibiotics=len(valid_cols),
            cfg=cfg,
            device=device,
        )

        micro_after, macro_after, _ = compute_micro_macro_auc_matrix(
            y_true=amr_test_local,
            y_score=preds_after,
        )

        n_test_pairs = int(np.sum(np.isfinite(amr_test_local)))

        global_record = {
            "run": run_id,
            "n_species": int(len(common_species)),
            "n_source_samples": int(len(X_source)),
            "n_ood_samples": int(len(X_ood)),
            "n_source_train_samples": int(len(X_source_train)),
            "n_source_val_samples": int(len(X_source_val)),
            "n_finetune_samples": int(len(X_ft)),
            "n_test_samples": int(len(X_test)),
            "n_valid_antibiotics": int(len(valid_cols)),
            "n_test_pairs": int(n_test_pairs),
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
            "valid_antibiotics": ";".join(selected_antibiotics),
        }

        global_records.append(global_record)

        logging.info(
            "Run %d result | micro before=%.4f | micro after=%.4f | macro before=%.4f | macro after=%.4f",
            run_id,
            micro_before if pd.notna(micro_before) else np.nan,
            micro_after if pd.notna(micro_after) else np.nan,
            macro_before if pd.notna(macro_before) else np.nan,
            macro_after if pd.notna(macro_after) else np.nan,
        )

        species_records.extend(
            compute_per_species_rows(
                run_id=run_id,
                y_true=amr_test_local,
                p_before=preds_before,
                p_after=preds_after,
                species_ids=species_test,
                id_to_species=id_to_species,
            )
        )

        antibiotic_records.extend(
            compute_per_antibiotic_rows(
                run_id=run_id,
                y_true=amr_test_local,
                p_before=preds_before,
                p_after=preds_after,
                antibiotic_names=selected_antibiotics,
            )
        )

        # Incremental save
        pd.DataFrame(global_records).to_csv(output_dir / "ood_gottingen_global_results.csv", index=False)
        pd.DataFrame(species_records).to_csv(output_dir / "ood_gottingen_species_results.csv", index=False)
        pd.DataFrame(antibiotic_records).to_csv(output_dir / "ood_gottingen_antibiotic_results.csv", index=False)

        del model
        clean_cuda()

    # --------------------------------------------------------
    # Final outputs
    # --------------------------------------------------------

    df_global = pd.DataFrame(global_records)
    df_species = pd.DataFrame(species_records)
    df_antibiotic = pd.DataFrame(antibiotic_records)

    df_global.to_csv(output_dir / "ood_gottingen_global_results.csv", index=False)
    df_species.to_csv(output_dir / "ood_gottingen_species_results.csv", index=False)
    df_antibiotic.to_csv(output_dir / "ood_gottingen_antibiotic_results.csv", index=False)

    df_species_summary = summarize_species(df_species)
    df_species_summary.to_csv(output_dir / "ood_gottingen_species_summary.csv", index=False)

    df_public = build_public_species_table(df_species_summary)
    df_public.to_csv(output_dir / "ood_gottingen_species_summary_public_table.csv", index=False)

    with open(output_dir / "ood_gottingen_species_summary_public_table.md", "w", encoding="utf-8") as f:
        if not df_public.empty:
            f.write(df_public.to_markdown(index=False))
        else:
            f.write("No results.")

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
        "global_summary": summarize_global(df_global),
    }

    json_dump(global_summary, output_dir / "ood_gottingen_global_summary.json")

    logging.info("=" * 100)
    logging.info("GÖTTINGEN GLOBAL RECOMMENDER OOD EXPERIMENT COMPLETED")
    logging.info("=" * 100)
    logging.info("Output dir: %s", output_dir)
    logging.info("Global results: %s", output_dir / "ood_gottingen_global_results.csv")
    logging.info("Species results: %s", output_dir / "ood_gottingen_species_results.csv")
    logging.info("Species summary: %s", output_dir / "ood_gottingen_species_summary.csv")
    logging.info("Public species table: %s", output_dir / "ood_gottingen_species_summary_public_table.csv")
    logging.info("Antibiotic results: %s", output_dir / "ood_gottingen_antibiotic_results.csv")
    logging.info("Global summary: %s", output_dir / "ood_gottingen_global_summary.json")


if __name__ == "__main__":
    main()