#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OOD GÖTTINGEN / MS-UMG EXPERIMENT
MULTIHEAD AND HYPERNETWORK NO-ENRICHMENT RECOMMENDERS

Source training domain:
    MARISMA + DRIAMS combined pickle

OOD target domain:
    Göttingen / MS-UMG pickle

Models evaluated:

1) species_multihead_no_enrichment
    Shared MALDI backbone:
        input_dim -> 512 -> 128
    Species-specific MALDI head:
        128 -> 64 -> 32
    Antibiotic embedding:
        antibiotic_id -> 16
    Species-specific recommender MLP:
        [z_species_maldi ; z_ab] = 32 + 16 = 48
        48 -> 32 -> 1

2) implicit_species_hypernetwork_no_enrichment
    Global MALDI encoder:
        input_dim -> 512 -> 128 -> 64 -> 32
    Species embedding:
        species_id -> 16
    Species hypernetwork:
        16 -> 64 -> 64 -> 32
    Residual adaptation:
        z_final = z_maldi + alpha * delta_species
    Antibiotic embedding:
        antibiotic_id -> 16
    Species-specific recommender MLP:
        [z_final ; z_ab] = 32 + 16 = 48
        48 -> 32 -> 1

Protocol:
    For each run and each model_mode:
        1. Keep species common to source and Göttingen.
        2. Align antibiotics common to source and Göttingen.
        3. Split source into train/val for early stopping.
        4. Split Göttingen by species into fine-tuning/test.
        5. Select antibiotics valid in source train, source val, Göttingen FT, Göttingen test.
        6. Train on source.
        7. Evaluate OOD before fine-tuning.
        8. Fine-tune on Göttingen FT.
        9. Evaluate OOD after fine-tuning.

Outputs:
    - ood_gottingen_multihead_hypernetwork_global_results.csv
    - ood_gottingen_multihead_hypernetwork_species_results.csv
    - ood_gottingen_multihead_hypernetwork_species_summary.csv
    - ood_gottingen_multihead_hypernetwork_species_summary_public_table.csv
    - ood_gottingen_multihead_hypernetwork_antibiotic_results.csv
    - ood_gottingen_multihead_hypernetwork_species_antibiotic_results.csv
    - ood_gottingen_multihead_hypernetwork_global_summary.json

Recommended command:

nohup python run_ood_gottingen_multihead_hypernetwork_gpu.py \
  --source-pickle "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl" \
  --ood-pickle "/export/data_ml4ds/bacteria_id/MALDIAlign_Alex/MSUMG_study_full.pkl" \
  --output-dir "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/src/benchmark study/benchmark_full/benchmark_outputs/ood_gottingen_multihead_hypernetwork" \
  --device cuda \
  --n-runs 5 \
  --finetune-frac 0.2 \
  --source-epochs 300 \
  --finetune-epochs 100 \
  --source-patience 15 \
  --finetune-patience 10 \
  --batch-size 1024 \
  --val-batch-size 2048 \
  --pred-batch-size 512 \
  > ood_gottingen_multihead_hypernetwork.log 2>&1 &
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

MODEL_MODES = [
    "species_multihead_no_enrichment",
    "implicit_species_hypernetwork_no_enrichment",
]


@dataclass
class ExperimentConfig:
    n_runs: int = 5
    random_seed: int = 42
    finetune_frac: float = 0.2

    source_epochs: int = 300
    finetune_epochs: int = 100
    source_patience: int = 15
    finetune_patience: int = 10

    batch_size: int = 1024
    val_batch_size: int = 2048
    pred_batch_size: int = 512

    lr: float = 1e-3
    finetune_lr_factor: float = 0.1
    weight_decay: float = 0.0

    min_source_samples_species: int = 50
    min_ood_samples_species: int = 50
    min_source_obs_per_antibiotic: int = 50
    min_source_val_obs_per_antibiotic: int = 5
    min_finetune_obs_per_antibiotic: int = 10
    min_test_obs_per_antibiotic: int = 10

    maldi_emb_dim: int = 32
    drug_emb_dim: int = 16
    hypernet_species_emb_dim: int = 16
    hypernet_hidden_dims: Tuple[int, ...] = (64, 64)
    alpha_init: float = 0.05
    recommender_hidden_dims: Tuple[int, ...] = (32,)
    dropout: float = 0.2

    device: str = "auto"
    num_workers: int = 4
    use_amp: bool = True
    compile_model: bool = False


# ============================================================
# LOGGING AND UTILS
# ============================================================


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "ood_gottingen_multihead_hypernetwork_run.log"

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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    s = str(x).strip().replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_").lower()

    enterobacter_aliases = {
        "enterobacter_cloacae",
        "enterobacter_cloacae_complex",
        "enterobacter_hormaechei",
        "enterobacter_asburiae",
        "enterobacter_kobei",
        "enterobacter_ludwigii",
        "enterobacter_roggenkampii",
    }
    if s in enterobacter_aliases:
        return "enterobacter_cloacae_complex"
    return s


def display_species_name(x: str) -> str:
    if x == "enterobacter_cloacae_complex":
        return "Enterobacter cloacae complex"
    parts = str(x).split("_")
    if len(parts) >= 2:
        return f"{parts[0].capitalize()} {' '.join(parts[1:])}"
    return str(x).replace("_", " ").title()


def canonical_antibiotic_name(x: Any) -> str:
    s = str(x).strip().lower()
    if s in {"combined_code", "nan", "none", ""}:
        return ""
    s = s.replace("β", "beta").replace("ß", "ss")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def build_index_by_canonical(names: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, name in enumerate(names):
        key = canonical_antibiotic_name(name)
        if key and key not in out:
            out[key] = i
    return out


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
    arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    if arr.dtype == object:
        arr = np.vstack([np.asarray(row, dtype=np.float32) for row in arr])
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D. Found shape: {arr.shape}")
    return arr


def coerce_amr_to_matrix(amr: Any) -> np.ndarray:
    arr = amr.values if isinstance(amr, pd.DataFrame) else np.asarray(amr)
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
        raise ValueError(f"{dataset_name}: inconsistent lengths between X, species and AMR.")
    if amr.shape[1] != len(antibiotics):
        raise ValueError(f"{dataset_name}: amr.shape[1]={amr.shape[1]} but len(antibiotics)={len(antibiotics)}")

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


def align_antibiotics(source_antibiotics: Sequence[str], ood_antibiotics: Sequence[str]) -> Tuple[List[int], List[int], List[str]]:
    source_map = build_index_by_canonical(source_antibiotics)
    ood_map = build_index_by_canonical(ood_antibiotics)
    common_keys = sorted(set(source_map.keys()) & set(ood_map.keys()))

    source_idxs, ood_idxs, names = [], [], []
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


class RecDataset(Dataset):
    """
    Each item:
        (MALDI sample, species_id, antibiotic_id) -> AMR label

    valid_cols are global aligned antibiotic columns, but drug_id is local
    within selected valid_cols: 0..len(valid_cols)-1.
    """

    def __init__(self, X: np.ndarray, species_ids: np.ndarray, amr: np.ndarray, valid_cols: Sequence[int]):
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
        s = self.sample_idx[idx]
        return (
            torch.from_numpy(self.X[s]).float(),
            torch.tensor(self.species_ids[s], dtype=torch.long),
            torch.tensor(self.drug_idx[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def make_loader(X, species_ids, amr, valid_cols, batch_size, shuffle, num_workers):
    ds = RecDataset(X=X, species_ids=species_ids, amr=amr, valid_cols=valid_cols)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


# ============================================================
# MODEL COMPONENTS
# ============================================================


class GlobalMALDIEncoder(nn.Module):
    def __init__(self, input_dim: int, maldi_emb_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.output_dim = maldi_emb_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, maldi_emb_dim),
            nn.GELU(),
        )

    def forward(self, maldi):
        return self.encoder(maldi.float())


class SpeciesHeadMALDIEncoder(nn.Module):
    def __init__(self, input_dim: int, num_species: int, maldi_emb_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.output_dim = maldi_emb_dim
        self.num_species = num_species
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.species_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, maldi_emb_dim),
                nn.GELU(),
            )
            for _ in range(num_species)
        ])

    def forward(self, maldi, species_id):
        z_shared = self.backbone(maldi.float())
        species_id = species_id.long()
        out = None
        for sp in torch.unique(species_id):
            sp_int = int(sp.item())
            mask = species_id == sp
            head_out = self.species_heads[sp_int](z_shared[mask])
            if out is None:
                out = torch.empty(
                    size=(z_shared.shape[0], self.output_dim),
                    dtype=head_out.dtype,
                    device=head_out.device,
                )
            out[mask] = head_out
        return out


class SpeciesHyperNetwork(nn.Module):
    def __init__(self, species_emb_dim: int = 16, hidden_dims: Sequence[int] = (64, 64), output_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        sizes = [species_emb_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, species_emb):
        return self.network(species_emb)


class SpeciesSpecificRecommenderMLP(nn.Module):
    def __init__(self, num_species: int, input_dim: int = 48, hidden_dims: Sequence[int] = (32,), output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.output_dim = output_dim
        self.species_mlps = nn.ModuleList([
            self._make_mlp(input_dim, hidden_dims, output_dim, dropout)
            for _ in range(num_species)
        ])

    @staticmethod
    def _make_mlp(input_dim, hidden_dims, output_dim, dropout):
        sizes = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        return nn.Sequential(*layers)

    def forward(self, x, species_id):
        species_id = species_id.long()
        out = None
        for sp in torch.unique(species_id):
            sp_int = int(sp.item())
            mask = species_id == sp
            mlp_out = self.species_mlps[sp_int](x[mask])
            if out is None:
                out = torch.empty(
                    size=(x.shape[0], self.output_dim),
                    dtype=mlp_out.dtype,
                    device=mlp_out.device,
                )
            out[mask] = mlp_out
        return out


class NoEnrichmentRecommender(nn.Module):
    def __init__(
        self,
        num_feat: int,
        num_items: int,
        num_species: int,
        model_mode: str,
        maldi_emb_dim: int = 32,
        drug_emb_dim: int = 16,
        hypernet_species_emb_dim: int = 16,
        hypernet_hidden_dims: Sequence[int] = (64, 64),
        alpha_init: float = 0.05,
        hidden_dims: Sequence[int] = (32,),
        dropout: float = 0.2,
    ):
        super().__init__()
        if model_mode not in MODEL_MODES:
            raise ValueError(f"Unknown model_mode: {model_mode}")

        self.model_mode = model_mode
        self.maldi_emb_dim = maldi_emb_dim
        self.drug_emb_dim = drug_emb_dim

        if model_mode == "species_multihead_no_enrichment":
            self.maldi_encoder = SpeciesHeadMALDIEncoder(
                input_dim=num_feat,
                num_species=num_species,
                maldi_emb_dim=maldi_emb_dim,
                dropout=dropout,
            )
            self.species_embedding = None
            self.species_hypernetwork = None
            self.alpha = None
        else:
            self.maldi_encoder = GlobalMALDIEncoder(
                input_dim=num_feat,
                maldi_emb_dim=maldi_emb_dim,
                dropout=dropout,
            )
            self.species_embedding = nn.Embedding(num_species, hypernet_species_emb_dim)
            self.species_hypernetwork = SpeciesHyperNetwork(
                species_emb_dim=hypernet_species_emb_dim,
                hidden_dims=hypernet_hidden_dims,
                output_dim=maldi_emb_dim,
                dropout=dropout,
            )
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.drug_embedding = nn.Embedding(num_items, drug_emb_dim)
        self.input_dim = maldi_emb_dim + drug_emb_dim
        self.recommender_mlp = SpeciesSpecificRecommenderMLP(
            num_species=num_species,
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

    def _encode_maldi(self, maldi, species_id):
        if self.model_mode == "species_multihead_no_enrichment":
            return self.maldi_encoder(maldi, species_id)
        z_maldi = self.maldi_encoder(maldi)
        species_emb = self.species_embedding(species_id.long())
        delta_species = self.species_hypernetwork(species_emb)
        return z_maldi + self.alpha * delta_species

    def get_alpha_value(self) -> float:
        if self.alpha is None:
            return np.nan
        return float(self.alpha.detach().cpu().item())

    def forward(self, maldi, species_id, drug_id):
        z_final = self._encode_maldi(maldi, species_id)
        drug_emb = self.drug_embedding(drug_id.long())
        x = torch.cat([z_final, drug_emb], dim=-1)
        return self.recommender_mlp(x, species_id).view(-1)


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


def build_model(num_feat, num_items, num_species, model_mode, cfg: ExperimentConfig, device: torch.device) -> nn.Module:
    model = NoEnrichmentRecommender(
        num_feat=num_feat,
        num_items=num_items,
        num_species=num_species,
        model_mode=model_mode,
        maldi_emb_dim=cfg.maldi_emb_dim,
        drug_emb_dim=cfg.drug_emb_dim,
        hypernet_species_emb_dim=cfg.hypernet_species_emb_dim,
        hypernet_hidden_dims=cfg.hypernet_hidden_dims,
        alpha_init=cfg.alpha_init,
        hidden_dims=cfg.recommender_hidden_dims,
        dropout=cfg.dropout,
    ).to(device)

    if cfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            logging.warning("torch.compile failed. Continuing without compile. Error: %s", e)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def evaluate_loss(model, loader, criterion, cfg: ExperimentConfig, device: torch.device) -> float:
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

    return float(np.mean(losses)) if losses else np.inf


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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
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

        val_loss = evaluate_loss(model, val_loader, criterion, cfg, device)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            alpha = model.get_alpha_value() if hasattr(model, "get_alpha_value") else np.nan
            logging.info(
                "Epoch %d/%d | train_loss=%.5f | val_loss=%.5f | alpha=%.5f",
                epoch + 1,
                epochs,
                float(np.mean(train_losses)) if train_losses else np.nan,
                val_loss,
                alpha if np.isfinite(alpha) else np.nan,
            )

        if stopper.step(val_loss, model):
            logging.info("Early stopping at epoch %d | best_val_loss=%.5f", epoch + 1, stopper.best_loss)
            break

    stopper.restore(model)
    return model


# ============================================================
# PREDICTION AND METRICS
# ============================================================


def predict_matrix(model, X, species_ids, num_items, cfg: ExperimentConfig, device: torch.device) -> np.ndarray:
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    species_ids = np.asarray(species_ids, dtype=np.int64)
    n = X.shape[0]
    preds = np.full((n, num_items), np.nan, dtype=np.float32)
    use_amp = cfg.use_amp and device.type == "cuda"

    with torch.no_grad():
        for start in range(0, n, cfg.pred_batch_size):
            end = min(start + cfg.pred_batch_size, n)
            xb = torch.from_numpy(X[start:end]).float().to(device, non_blocking=True)
            sp_batch = torch.from_numpy(species_ids[start:end]).long().to(device, non_blocking=True)
            bsz = xb.shape[0]
            batch_preds = []

            for drug_id in range(num_items):
                drug_batch = torch.full((bsz,), drug_id, dtype=torch.long, device=device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(xb, sp_batch, drug_batch)
                batch_preds.append(torch.sigmoid(logits).detach().cpu().numpy())

            preds[start:end] = np.stack(batch_preds, axis=1).astype(np.float32)
    return preds


def safe_binary_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    if np.sum(valid) == 0:
        return np.nan
    y = y_true[valid]
    p = y_score[valid]
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


def compute_micro_macro_auc_matrix(y_true, y_score) -> Tuple[float, float, List[float]]:
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
        antibiotic_aucs.append(safe_binary_auc(y_true[:, j], y_score[:, j]))

    valid_aucs = [a for a in antibiotic_aucs if np.isfinite(a)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else np.nan
    return micro_auc, macro_auc, antibiotic_aucs


def count_obs_and_classes(y) -> Tuple[int, int, int, bool]:
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y) & ((y == 0) | (y == 1))
    obs = y[valid]
    n = int(len(obs))
    n0 = int(np.sum(obs == 0))
    n1 = int(np.sum(obs == 1))
    return n, n0, n1, bool(n0 > 0 and n1 > 0)


def select_valid_antibiotics(source_amr_train, source_amr_val, ft_amr, test_amr, cfg: ExperimentConfig) -> List[int]:
    valid_cols = []
    for j in range(source_amr_train.shape[1]):
        n_tr, _, _, tr_two = count_obs_and_classes(source_amr_train[:, j])
        n_val, _, _, val_two = count_obs_and_classes(source_amr_val[:, j])
        n_ft, _, _, ft_two = count_obs_and_classes(ft_amr[:, j])
        n_test, _, _, test_two = count_obs_and_classes(test_amr[:, j])

        if n_tr < cfg.min_source_obs_per_antibiotic:
            continue
        if not tr_two:
            continue
        if n_val < cfg.min_source_val_obs_per_antibiotic:
            continue
        if not val_two:
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


def compute_per_species_rows(run_id, model_mode, y_true, p_before, p_after, species_ids, id_to_species):
    rows = []
    for sp_id in sorted(np.unique(species_ids)):
        mask = species_ids == sp_id
        y_sp = y_true[mask]
        pb = p_before[mask]
        pa = p_after[mask]
        micro_b, macro_b, _ = compute_micro_macro_auc_matrix(y_sp, pb)
        micro_a, macro_a, _ = compute_micro_macro_auc_matrix(y_sp, pa)
        rows.append({
            "run": run_id,
            "model_mode": model_mode,
            "species_id": int(sp_id),
            "species_key": id_to_species[int(sp_id)],
            "species": display_species_name(id_to_species[int(sp_id)]),
            "n_test_samples": int(np.sum(mask)),
            "n_test_pairs": int(np.sum(np.isfinite(y_sp))),
            "micro_auc_before_finetuning": maybe_float(micro_b),
            "macro_auc_before_finetuning": maybe_float(macro_b),
            "micro_auc_after_finetuning": maybe_float(micro_a),
            "macro_auc_after_finetuning": maybe_float(macro_a),
            "delta_micro_auc_after_minus_before": maybe_float(micro_a - micro_b if pd.notna(micro_a) and pd.notna(micro_b) else np.nan),
            "delta_macro_auc_after_minus_before": maybe_float(macro_a - macro_b if pd.notna(macro_a) and pd.notna(macro_b) else np.nan),
        })
    return rows


def compute_per_antibiotic_rows(run_id, model_mode, y_true, p_before, p_after, antibiotic_names):
    rows = []
    _, _, aucs_b = compute_micro_macro_auc_matrix(y_true, p_before)
    _, _, aucs_a = compute_micro_macro_auc_matrix(y_true, p_after)
    for j, ab in enumerate(antibiotic_names):
        n_obs, n_s, n_r, _ = count_obs_and_classes(y_true[:, j])
        b = aucs_b[j]
        a = aucs_a[j]
        rows.append({
            "run": run_id,
            "model_mode": model_mode,
            "antibiotic_id": int(j),
            "antibiotic": str(ab),
            "n_test_obs": int(n_obs),
            "n_test_susceptible": int(n_s),
            "n_test_resistant": int(n_r),
            "auc_before_finetuning": maybe_float(b),
            "auc_after_finetuning": maybe_float(a),
            "delta_auc_after_minus_before": maybe_float(a - b if pd.notna(a) and pd.notna(b) else np.nan),
        })
    return rows


def compute_per_species_antibiotic_rows(run_id, model_mode, y_true, p_before, p_after, species_ids, id_to_species, antibiotic_names):
    rows = []
    for sp_id in sorted(np.unique(species_ids)):
        sp_mask = species_ids == sp_id
        y_sp = y_true[sp_mask]
        pb_sp = p_before[sp_mask]
        pa_sp = p_after[sp_mask]
        for j, ab in enumerate(antibiotic_names):
            n_obs, n_s, n_r, _ = count_obs_and_classes(y_sp[:, j])
            auc_b = safe_binary_auc(y_sp[:, j], pb_sp[:, j])
            auc_a = safe_binary_auc(y_sp[:, j], pa_sp[:, j])
            rows.append({
                "run": run_id,
                "model_mode": model_mode,
                "species_id": int(sp_id),
                "species_key": id_to_species[int(sp_id)],
                "species": display_species_name(id_to_species[int(sp_id)]),
                "antibiotic_id": int(j),
                "antibiotic": str(ab),
                "n_test_obs": int(n_obs),
                "n_test_susceptible": int(n_s),
                "n_test_resistant": int(n_r),
                "auc_before_finetuning": maybe_float(auc_b),
                "auc_after_finetuning": maybe_float(auc_a),
                "delta_auc_after_minus_before": maybe_float(auc_a - auc_b if pd.notna(auc_a) and pd.notna(auc_b) else np.nan),
            })
    return rows


# ============================================================
# SPLITTING
# ============================================================


def split_train_val(n_samples: int, seed: int, val_size: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_samples)
    if n_samples < 5:
        return idx, idx
    train_idx, val_idx = train_test_split(idx, test_size=val_size, random_state=seed, shuffle=True)
    return np.asarray(train_idx), np.asarray(val_idx)


def split_ood_by_species(species_ids_ood, amr_ood, finetune_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    all_ft, all_test = [], []
    for sp_id in sorted(np.unique(species_ids_ood)):
        global_idx = np.where(species_ids_ood == sp_id)[0]
        amr_sp = amr_ood[global_idx]
        n = len(global_idx)
        if n < 5:
            all_test.extend(global_idx.tolist())
            continue

        observed_counts = np.sum(np.isfinite(amr_sp), axis=1)
        resistance_mean = np.nanmean(amr_sp, axis=1)
        resistance_mean = np.nan_to_num(resistance_mean, nan=-1.0)

        try:
            obs_bin = pd.qcut(observed_counts, q=min(4, len(np.unique(observed_counts))), labels=False, duplicates="drop")
        except Exception:
            obs_bin = np.zeros(n, dtype=int)

        try:
            res_bin = pd.qcut(resistance_mean, q=min(4, len(np.unique(resistance_mean))), labels=False, duplicates="drop")
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

        all_ft.extend(global_idx[ft_local].tolist())
        all_test.extend(global_idx[test_local].tolist())

    return np.asarray(all_ft, dtype=np.int64), np.asarray(all_test, dtype=np.int64)


# ============================================================
# SUMMARIES
# ============================================================


def summarize_species(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (model_mode, species, species_key), sub in df.groupby(["model_mode", "species", "species_key"]):
        rows.append({
            "model_mode": model_mode,
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
        })
    return pd.DataFrame(rows).sort_values(["model_mode", "micro_auc_after_mean"], ascending=[True, False]).reset_index(drop=True)


def summarize_global(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if df.empty:
        return out
    for model_mode, sub in df.groupby("model_mode"):
        out[model_mode] = {}
        for col in [
            "micro_auc_before_finetuning",
            "macro_auc_before_finetuning",
            "micro_auc_after_finetuning",
            "macro_auc_after_finetuning",
            "delta_micro_auc_after_minus_before",
            "delta_macro_auc_after_minus_before",
            "final_alpha_before_finetuning",
            "final_alpha_after_finetuning",
        ]:
            if col in sub.columns:
                out[model_mode][f"{col}_mean"] = float(sub[col].mean(skipna=True))
                out[model_mode][f"{col}_std"] = float(sub[col].std(skipna=True, ddof=1))
    return out


def format_pm(mean_val, std_val, decimals: int = 4) -> str:
    if mean_val is None or pd.isna(mean_val):
        return "NaN"
    if std_val is None or pd.isna(std_val):
        return f"{mean_val:.{decimals}f}"
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


def build_public_species_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["Model"] = summary["model_mode"]
    out["Species"] = summary["species"]
    out["Micro AUC before FT"] = [format_pm(m, s) for m, s in zip(summary["micro_auc_before_mean"], summary["micro_auc_before_std"])]
    out["Micro AUC after FT"] = [format_pm(m, s) for m, s in zip(summary["micro_auc_after_mean"], summary["micro_auc_after_std"])]
    out["Δ Micro AUC"] = [format_pm(m, s) for m, s in zip(summary["delta_micro_auc_mean"], summary["delta_micro_auc_std"])]
    out["Macro AUC before FT"] = [format_pm(m, s) for m, s in zip(summary["macro_auc_before_mean"], summary["macro_auc_before_std"])]
    out["Macro AUC after FT"] = [format_pm(m, s) for m, s in zip(summary["macro_auc_after_mean"], summary["macro_auc_after_std"])]
    out["Δ Macro AUC"] = [format_pm(m, s) for m, s in zip(summary["delta_macro_auc_mean"], summary["delta_macro_auc_std"])]
    out["Mean test samples"] = summary["mean_test_samples"].round(1)
    out["Mean test pairs"] = summary["mean_test_pairs"].round(1)
    return out


# ============================================================
# ARGPARSE
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Göttingen/MS-UMG OOD experiment for multihead and hypernetwork recommenders."
    )
    parser.add_argument("--source-pickle", type=str, required=True)
    parser.add_argument("--ood-pickle", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-modes", nargs="*", default=MODEL_MODES)
    parser.add_argument("--species", nargs="*", default=None)

    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--finetune-frac", type=float, default=0.2)

    parser.add_argument("--source-epochs", type=int, default=300)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--source-patience", type=int, default=15)
    parser.add_argument("--finetune-patience", type=int, default=10)

    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--val-batch-size", type=int, default=2048)
    parser.add_argument("--pred-batch-size", type=int, default=512)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--min-source-samples-species", type=int, default=50)
    parser.add_argument("--min-ood-samples-species", type=int, default=50)
    parser.add_argument("--min-source-obs-per-antibiotic", type=int, default=50)
    parser.add_argument("--min-source-val-obs-per-antibiotic", type=int, default=5)
    parser.add_argument("--min-finetune-obs-per-antibiotic", type=int, default=10)
    parser.add_argument("--min-test-obs-per-antibiotic", type=int, default=10)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
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
        pred_batch_size=args.pred_batch_size,
        lr=args.lr,
        finetune_lr_factor=args.finetune_lr_factor,
        weight_decay=args.weight_decay,
        min_source_samples_species=args.min_source_samples_species,
        min_ood_samples_species=args.min_ood_samples_species,
        min_source_obs_per_antibiotic=args.min_source_obs_per_antibiotic,
        min_source_val_obs_per_antibiotic=args.min_source_val_obs_per_antibiotic,
        min_finetune_obs_per_antibiotic=args.min_finetune_obs_per_antibiotic,
        min_test_obs_per_antibiotic=args.min_test_obs_per_antibiotic,
        device=args.device,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        compile_model=args.compile_model,
    )

    seed_everything(cfg.random_seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.benchmark = True

    device = choose_device(cfg.device)
    logging.info("=" * 100)
    logging.info("GÖTTINGEN/MS-UMG OOD MULTIHEAD + HYPERNETWORK RECOMMENDERS")
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

    X_source_full = source["X"]
    species_source_full = source["species"]
    amr_source_full = source["amr"]
    source_antibiotics = source["antibiotics"]

    X_ood_full = ood["X"]
    species_ood_full = ood["species"]
    amr_ood_full = ood["amr"]
    ood_antibiotics = ood["antibiotics"]

    if X_source_full.shape[1] != X_ood_full.shape[1]:
        raise ValueError(f"Feature dimensions differ: source={X_source_full.shape[1]}, OOD={X_ood_full.shape[1]}")

    # --------------------------------------------------------
    # Align antibiotics
    # --------------------------------------------------------
    src_ab_idx, ood_ab_idx, common_antibiotics = align_antibiotics(source_antibiotics, ood_antibiotics)
    if len(common_antibiotics) == 0:
        raise RuntimeError("No common antibiotics found between source and OOD.")

    amr_source_full = amr_source_full[:, src_ab_idx]
    amr_ood_full = amr_ood_full[:, ood_ab_idx]

    pd.DataFrame({
        "common_antibiotic": common_antibiotics,
        "source_index": src_ab_idx,
        "ood_index": ood_ab_idx,
        "source_name": [source_antibiotics[i] for i in src_ab_idx],
        "ood_name": [ood_antibiotics[i] for i in ood_ab_idx],
    }).to_csv(output_dir / "common_antibiotics_mapping.csv", index=False)

    # --------------------------------------------------------
    # Common species
    # --------------------------------------------------------
    common_species = sorted(set(species_source_full.tolist()) & set(species_ood_full.tolist()))
    if args.species:
        requested = {canonical_species_name(s) for s in args.species}
        common_species = [s for s in common_species if s in requested]

    filtered_species = []
    for sp in common_species:
        n_source = int(np.sum(species_source_full == sp))
        n_ood = int(np.sum(species_ood_full == sp))
        if n_source >= cfg.min_source_samples_species and n_ood >= cfg.min_ood_samples_species:
            filtered_species.append(sp)
    common_species = filtered_species
    if not common_species:
        raise RuntimeError("No common species passed sample-count filters.")

    species_to_id = {sp: i for i, sp in enumerate(common_species)}
    id_to_species = {i: sp for sp, i in species_to_id.items()}

    pd.DataFrame({
        "species_key": common_species,
        "species": [display_species_name(s) for s in common_species],
        "species_id": [species_to_id[s] for s in common_species],
        "n_source_samples": [int(np.sum(species_source_full == s)) for s in common_species],
        "n_ood_samples": [int(np.sum(species_ood_full == s)) for s in common_species],
    }).sort_values("n_ood_samples", ascending=False).to_csv(output_dir / "common_species_counts.csv", index=False)

    json_dump({
        "species_to_id": species_to_id,
        "id_to_species": {str(k): v for k, v in id_to_species.items()},
        "common_antibiotics": common_antibiotics,
        "model_modes": args.model_modes,
    }, output_dir / "mappings.json")

    logging.info("Common antibiotics: %d", len(common_antibiotics))
    logging.info("Common species selected: %d", len(common_species))
    logging.info("Species: %s", [display_species_name(s) for s in common_species])

    # Filter source/OOD to common species
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

    global_records = []
    species_records = []
    antibiotic_records = []
    species_antibiotic_records = []
    t_start = time.time()

    for model_mode in args.model_modes:
        if model_mode not in MODEL_MODES:
            raise ValueError(f"Invalid model mode: {model_mode}")

        logging.info("#" * 100)
        logging.info("MODEL MODE: %s", model_mode)
        logging.info("#" * 100)

        for run_id in range(cfg.n_runs):
            logging.info("=" * 100)
            logging.info("MODE %s | RUN %d/%d", model_mode, run_id + 1, cfg.n_runs)
            logging.info("=" * 100)

            seed = cfg.random_seed + run_id * 1000
            source_train_idx, source_val_idx = split_train_val(len(X_source), seed=seed, val_size=0.15)
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
                logging.warning("Mode %s | run %d skipped: no valid antibiotics", model_mode, run_id)
                continue

            selected_antibiotics = [common_antibiotics[j] for j in valid_cols]
            logging.info("Mode %s | run %d | valid antibiotics=%d", model_mode, run_id, len(valid_cols))
            logging.info("Selected antibiotics: %s", selected_antibiotics)

            model = build_model(
                num_feat=X_source.shape[1],
                num_items=len(valid_cols),
                num_species=len(common_species),
                model_mode=model_mode,
                cfg=cfg,
                device=device,
            )
            n_params = count_trainable_parameters(model)
            logging.info("Trainable parameters: %d", n_params)

            # ---------------- Source training ----------------
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

            alpha_before = model.get_alpha_value() if hasattr(model, "get_alpha_value") else np.nan

            # ---------------- OOD before FT ----------------
            preds_before = predict_matrix(
                model=model,
                X=X_test,
                species_ids=species_test,
                num_items=len(valid_cols),
                cfg=cfg,
                device=device,
            )
            amr_test_local = amr_test[:, valid_cols]
            micro_before, macro_before, _ = compute_micro_macro_auc_matrix(amr_test_local, preds_before)

            # ---------------- Fine-tuning ----------------
            ft_train_idx, ft_val_idx = split_train_val(len(X_ft), seed=seed, val_size=0.2)
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

            alpha_after = model.get_alpha_value() if hasattr(model, "get_alpha_value") else np.nan

            # ---------------- OOD after FT ----------------
            preds_after = predict_matrix(
                model=model,
                X=X_test,
                species_ids=species_test,
                num_items=len(valid_cols),
                cfg=cfg,
                device=device,
            )
            micro_after, macro_after, _ = compute_micro_macro_auc_matrix(amr_test_local, preds_after)

            n_test_pairs = int(np.sum(np.isfinite(amr_test_local)))
            global_record = {
                "run": run_id,
                "model_mode": model_mode,
                "n_species": int(len(common_species)),
                "n_source_samples": int(len(X_source)),
                "n_ood_samples": int(len(X_ood)),
                "n_source_train_samples": int(len(X_source_train)),
                "n_source_val_samples": int(len(X_source_val)),
                "n_finetune_samples": int(len(X_ft)),
                "n_test_samples": int(len(X_test)),
                "n_valid_antibiotics": int(len(valid_cols)),
                "n_test_pairs": n_test_pairs,
                "trainable_parameters": int(n_params),
                "final_alpha_before_finetuning": maybe_float(alpha_before),
                "final_alpha_after_finetuning": maybe_float(alpha_after),
                "micro_auc_before_finetuning": maybe_float(micro_before),
                "macro_auc_before_finetuning": maybe_float(macro_before),
                "micro_auc_after_finetuning": maybe_float(micro_after),
                "macro_auc_after_finetuning": maybe_float(macro_after),
                "delta_micro_auc_after_minus_before": maybe_float(micro_after - micro_before if pd.notna(micro_after) and pd.notna(micro_before) else np.nan),
                "delta_macro_auc_after_minus_before": maybe_float(macro_after - macro_before if pd.notna(macro_after) and pd.notna(macro_before) else np.nan),
                "valid_antibiotics": ";".join(selected_antibiotics),
            }
            global_records.append(global_record)

            logging.info(
                "Mode %s | run %d result | micro before=%.4f after=%.4f | macro before=%.4f after=%.4f | alpha before=%.4f after=%.4f",
                model_mode,
                run_id,
                micro_before if pd.notna(micro_before) else np.nan,
                micro_after if pd.notna(micro_after) else np.nan,
                macro_before if pd.notna(macro_before) else np.nan,
                macro_after if pd.notna(macro_after) else np.nan,
                alpha_before if np.isfinite(alpha_before) else np.nan,
                alpha_after if np.isfinite(alpha_after) else np.nan,
            )

            species_records.extend(
                compute_per_species_rows(
                    run_id=run_id,
                    model_mode=model_mode,
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
                    model_mode=model_mode,
                    y_true=amr_test_local,
                    p_before=preds_before,
                    p_after=preds_after,
                    antibiotic_names=selected_antibiotics,
                )
            )
            species_antibiotic_records.extend(
                compute_per_species_antibiotic_rows(
                    run_id=run_id,
                    model_mode=model_mode,
                    y_true=amr_test_local,
                    p_before=preds_before,
                    p_after=preds_after,
                    species_ids=species_test,
                    id_to_species=id_to_species,
                    antibiotic_names=selected_antibiotics,
                )
            )

            # Incremental save
            pd.DataFrame(global_records).to_csv(output_dir / "ood_gottingen_multihead_hypernetwork_global_results.csv", index=False)
            pd.DataFrame(species_records).to_csv(output_dir / "ood_gottingen_multihead_hypernetwork_species_results.csv", index=False)
            pd.DataFrame(antibiotic_records).to_csv(output_dir / "ood_gottingen_multihead_hypernetwork_antibiotic_results.csv", index=False)
            pd.DataFrame(species_antibiotic_records).to_csv(output_dir / "ood_gottingen_multihead_hypernetwork_species_antibiotic_results.csv", index=False)

            del model
            clean_cuda()

    # --------------------------------------------------------
    # Final outputs
    # --------------------------------------------------------
    df_global = pd.DataFrame(global_records)
    df_species = pd.DataFrame(species_records)
    df_antibiotic = pd.DataFrame(antibiotic_records)
    df_species_antibiotic = pd.DataFrame(species_antibiotic_records)

    df_global.to_csv(output_dir / "ood_gottingen_multihead_hypernetwork_global_results.csv", index=False)
    df_species.to_csv(output_dir / "ood_gottingen_multihead_hypernetwork_species_results.csv", index=False)
    df_antibiotic.to_csv(output_dir / "ood_gottingen_multihead_hypernetwork_antibiotic_results.csv", index=False)
    df_species_antibiotic.to_csv(output_dir / "ood_gottingen_multihead_hypernetwork_species_antibiotic_results.csv", index=False)

    df_species_summary = summarize_species(df_species)
    df_species_summary.to_csv(output_dir / "ood_gottingen_multihead_hypernetwork_species_summary.csv", index=False)

    df_public = build_public_species_table(df_species_summary)
    df_public.to_csv(output_dir / "ood_gottingen_multihead_hypernetwork_species_summary_public_table.csv", index=False)

    with open(output_dir / "ood_gottingen_multihead_hypernetwork_species_summary_public_table.md", "w", encoding="utf-8") as f:
        if not df_public.empty:
            f.write(df_public.to_markdown(index=False))
        else:
            f.write("No results.")

    global_summary = {
        "config": asdict(cfg),
        "source_pickle": args.source_pickle,
        "ood_pickle": args.ood_pickle,
        "device": str(device),
        "model_modes": args.model_modes,
        "n_common_antibiotics": int(len(common_antibiotics)),
        "common_antibiotics": common_antibiotics,
        "n_species": int(len(common_species)),
        "species": [display_species_name(s) for s in common_species],
        "elapsed_seconds": float(time.time() - t_start),
        "global_summary": summarize_global(df_global),
    }
    json_dump(global_summary, output_dir / "ood_gottingen_multihead_hypernetwork_global_summary.json")

    logging.info("=" * 100)
    logging.info("GÖTTINGEN MULTIHEAD + HYPERNETWORK OOD EXPERIMENT COMPLETED")
    logging.info("=" * 100)
    logging.info("Output dir: %s", output_dir)
    logging.info("Global results: %s", output_dir / "ood_gottingen_multihead_hypernetwork_global_results.csv")
    logging.info("Species summary: %s", output_dir / "ood_gottingen_multihead_hypernetwork_species_summary.csv")
    logging.info("Public species table: %s", output_dir / "ood_gottingen_multihead_hypernetwork_species_summary_public_table.csv")
    logging.info("Antibiotic results: %s", output_dir / "ood_gottingen_multihead_hypernetwork_antibiotic_results.csv")
    logging.info("Species-antibiotic results: %s", output_dir / "ood_gottingen_multihead_hypernetwork_species_antibiotic_results.csv")
    logging.info("Global summary: %s", output_dir / "ood_gottingen_multihead_hypernetwork_global_summary.json")


if __name__ == "__main__":
    main()
