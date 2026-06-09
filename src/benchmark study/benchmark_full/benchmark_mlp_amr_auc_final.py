#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAIR MLP BENCHMARK FOR MALDI-TOF AMR PREDICTION

Objetivo:
    Reevaluar modelos MLP species-specific de forma comparable al recommender global.

Modelos evaluados:
    1) Binary MLP:
        - un modelo por especie y antibiótico
        - usa solo muestras observadas para ese antibiótico

    2) Multilabel MLP:
        - un modelo por especie
        - una salida/logit por antibiótico
        - permite NaN usando masked BCE loss
        - una muestra contribuye solo en los antibióticos donde tiene etiqueta conocida

NO evalúa LPS:
    - LPS obliga a eliminar muestras con cualquier NaN en los antibióticos seleccionados
    - por tanto no es comparable con el recommender global

Split:
    - KFold global sobre todas las muestras, igual que el recommender global
    - después se evalúa cada especie dentro de cada fold

Métricas:
    Por especie y fold:
        - binary_micro_auc
        - binary_macro_auc
        - multilabel_micro_auc
        - multilabel_macro_auc

    Global por fold:
        - global binary micro/macro
        - global multilabel micro/macro

Outputs:
    - fair_mlp_all_species_fold_results.csv
    - fair_mlp_species_summary.csv
    - fair_mlp_species_antibiotic_fold_results.csv
    - fair_mlp_global_fold_results.csv
    - fair_mlp_global_summary.json
    - fair_mlp_run.log

Uso típico:

nohup python run_fair_mlp_binary_multilabel_auc.py \
  --pickle-path "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl" \
  --benchmark-roots \
    "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/src/benchmark study/benchmark_full/benchmark_outputs/benchmark_mlp_20260314_200946" \
    "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/src/benchmark study/benchmark_full/benchmark_outputs/benchmark_mlp_20260316_112641" \
  --output-dir "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/src/benchmark study/benchmark_full/benchmark_outputs/fair_mlp_no_lps_20260603" \
  --device auto \
  > fair_mlp_no_lps.log 2>&1 &

Si NO quieres usar params antiguos:
nohup python run_fair_mlp_binary_multilabel_auc.py \
  --pickle-path "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl" \
  --output-dir "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/src/benchmark study/benchmark_full/benchmark_outputs/fair_mlp_no_lps_default_params" \
  --device auto \
  > fair_mlp_no_lps.log 2>&1 &
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
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_PARAMS = {
    "layer1": 128,
    "layer2": 64,
    "layer3": 32,
    "activation": "gelu",
    "solver": "adam",
    "learning_rate": 1e-3,
}

DEFAULT_MIN_TRAIN_OBS_PER_ANTIBIOTIC = 50
DEFAULT_MIN_VAL_OBS_PER_ANTIBIOTIC = 5


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class TrainConfig:
    n_splits: int = 5
    random_seed: int = 42
    epochs: int = 300
    patience: int = 15
    batch_size: int = 64
    val_batch_size: int = 256
    num_workers: int = 0
    device: str = "auto"
    min_train_obs_per_antibiotic: int = DEFAULT_MIN_TRAIN_OBS_PER_ANTIBIOTIC
    min_val_obs_per_antibiotic: int = DEFAULT_MIN_VAL_OBS_PER_ANTIBIOTIC


@dataclass
class SpeciesParams:
    species: str
    multilabel_params: Dict[str, Any]
    multilabel_params_source: str
    binary_params_by_antibiotic: Dict[str, Dict[str, Any]]
    binary_params_source_by_antibiotic: Dict[str, str]


# ============================================================
# Logging
# ============================================================

def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "fair_mlp_run.log"

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
# General utils
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


def json_load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def pretty_species_name(species: str) -> str:
    return str(species).replace("_", " ")


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
# Pickle loading / normalization
# ============================================================

def load_pickle_payload(pickle_path: str) -> Dict[str, Any]:
    with open(pickle_path, "rb") as f:
        payload = pickle.load(f)

    required_keys = {"data", "label", "amr", "antibiotics"}
    missing = required_keys - set(payload.keys())

    if missing:
        raise KeyError(f"El pickle no contiene las claves requeridas: {missing}")

    return payload


def coerce_X_to_2d_float32(X: Any) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        arr = X.values
    else:
        arr = np.asarray(X)

    if arr.dtype == object:
        arr = np.vstack([np.asarray(row, dtype=np.float32) for row in arr])

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"X debe ser 2D. Shape encontrado: {arr.shape}")

    return arr


def normalize_species_labels(y_species: Any) -> np.ndarray:
    if isinstance(y_species, pd.Series):
        return y_species.astype(str).values

    return np.asarray(y_species).astype(str)


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

    s = str(v).strip().lower()

    if s in {"1", "true", "r", "resistant", "resistance"}:
        return 1.0

    if s in {"0", "false", "s", "susceptible"}:
        return 0.0

    if s in {"nan", "none", "", "na", "n/a", "i", "intermediate"}:
        return np.nan

    return np.nan


def coerce_amr_to_matrix(amr: Any) -> np.ndarray:
    if isinstance(amr, pd.DataFrame):
        arr = amr.values
    else:
        arr = np.asarray(amr)

    if arr.ndim != 2:
        raise ValueError(f"AMR debe ser 2D. Shape encontrado: {arr.shape}")

    out = np.empty(arr.shape, dtype=np.float32)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            out[i, j] = normalize_single_amr_value(arr[i, j])

    return out


def clean_spectra_and_labels(
    X_all: np.ndarray,
    species_all: np.ndarray,
    amr_all: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Elimina solo muestras con espectro inválido.

    Importante:
        NO elimina muestras por NaN en AMR.
        Los NaN en AMR se gestionan después por máscara.
    """

    x_valid = np.isfinite(X_all).all(axis=1)

    X_clean = X_all[x_valid].astype(np.float32, copy=False)
    species_clean = species_all[x_valid]
    amr_clean = amr_all[x_valid].astype(np.float32, copy=False)

    return X_clean, species_clean, amr_clean


# ============================================================
# Params discovery
# ============================================================

def sanitize_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Adapta params antiguos de Optuna al formato esperado.
    Si falta algo, rellena con defaults.
    """

    out = dict(DEFAULT_PARAMS)

    if params is not None:
        for k in out.keys():
            if k in params:
                out[k] = params[k]

    out["layer1"] = int(out["layer1"])
    out["layer2"] = int(out["layer2"])
    out["layer3"] = int(out["layer3"])
    out["activation"] = str(out["activation"])
    out["solver"] = str(out["solver"])
    out["learning_rate"] = float(out["learning_rate"])

    return out


def discover_params_for_species(
    species: str,
    antibiotics: Sequence[str],
    benchmark_roots: Optional[Sequence[str]],
) -> SpeciesParams:
    """
    Busca params antiguos de Optuna.

    Espera estructura del benchmark antiguo:
        root/species/species__multilabel_best_params.json
        root/species/binary_models/species__ANTIBIOTIC__binary_best_params.json

    Si no encuentra params:
        usa DEFAULT_PARAMS.
    """

    multilabel_params = sanitize_params(None)
    multilabel_source = "default"

    binary_params_by_antibiotic = {
        str(ab): sanitize_params(None) for ab in antibiotics
    }
    binary_sources_by_antibiotic = {
        str(ab): "default" for ab in antibiotics
    }

    if not benchmark_roots:
        return SpeciesParams(
            species=species,
            multilabel_params=multilabel_params,
            multilabel_params_source=multilabel_source,
            binary_params_by_antibiotic=binary_params_by_antibiotic,
            binary_params_source_by_antibiotic=binary_sources_by_antibiotic,
        )

    candidate_species_dirs: List[Path] = []

    for root in benchmark_roots:
        root_path = Path(root)

        if not root_path.exists():
            logging.warning("Benchmark root no existe y se ignora: %s", root_path)
            continue

        species_dir = root_path / species

        if species_dir.exists() and species_dir.is_dir():
            candidate_species_dirs.append(species_dir)

    if not candidate_species_dirs:
        return SpeciesParams(
            species=species,
            multilabel_params=multilabel_params,
            multilabel_params_source=multilabel_source,
            binary_params_by_antibiotic=binary_params_by_antibiotic,
            binary_params_source_by_antibiotic=binary_sources_by_antibiotic,
        )

    # Prioriza el último root por orden lexicográfico, como en tu script antiguo
    candidate_species_dirs = sorted(candidate_species_dirs, key=lambda p: str(p))

    for species_dir in candidate_species_dirs:
        ml_path = species_dir / f"{species}__multilabel_best_params.json"

        if ml_path.exists():
            try:
                multilabel_params = sanitize_params(json_load(ml_path))
                multilabel_source = str(ml_path)
            except Exception as e:
                logging.warning("[%s] No se pudieron cargar params multilabel de %s: %s", species, ml_path, e)

        binary_dir = species_dir / "binary_models"

        if binary_dir.exists():
            for ab in antibiotics:
                ab_str = str(ab)
                bin_path = binary_dir / f"{species}__{ab_str}__binary_best_params.json"

                if bin_path.exists():
                    try:
                        binary_params_by_antibiotic[ab_str] = sanitize_params(json_load(bin_path))
                        binary_sources_by_antibiotic[ab_str] = str(bin_path)
                    except Exception as e:
                        logging.warning(
                            "[%s | %s] No se pudieron cargar params binarios de %s: %s",
                            species,
                            ab_str,
                            bin_path,
                            e,
                        )

    return SpeciesParams(
        species=species,
        multilabel_params=multilabel_params,
        multilabel_params_source=multilabel_source,
        binary_params_by_antibiotic=binary_params_by_antibiotic,
        binary_params_source_by_antibiotic=binary_sources_by_antibiotic,
    )


# ============================================================
# Model
# ============================================================

def activation_from_name(name: str) -> nn.Module:
    name = str(name).strip().lower()

    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name in {"sigmoid", "logistic"}:
        return nn.Sigmoid()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    if name == "identity":
        return nn.Identity()

    logging.warning("Activación no reconocida '%s'. Se usa GELU.", name)
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

        def new_act() -> nn.Module:
            return activation_from_name(activation)

        self.net = nn.Sequential(
            nn.Linear(input_dim, layer1),
            new_act(),
            nn.Linear(layer1, layer2),
            new_act(),
            nn.Linear(layer2, layer3),
            new_act(),
            nn.Linear(layer3, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def build_optimizer(model: nn.Module, params: Dict[str, Any]) -> torch.optim.Optimizer:
    solver = str(params.get("solver", "adam")).lower()
    lr = float(params.get("learning_rate", 1e-3))

    if solver == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)

    if solver == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)

    if solver == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    logging.warning("Solver no reconocido '%s'. Se usa Adam.", solver)
    return torch.optim.Adam(model.parameters(), lr=lr)


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_score: Optional[float] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.counter = 0

    def step(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = float(score)
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False

        self.counter += 1

        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ============================================================
# Datasets / loaders
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
    )


# ============================================================
# Metrics
# ============================================================

def safe_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)

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
    """
    y_true:
        shape (N, A), valores 0/1/NaN

    y_score:
        shape (N, A), probabilidades o NaN

    micro:
        AUC sobre todos los pares observados y predichos

    macro:
        media de AUCs por antibiótico, ignorando antibióticos no evaluables
    """

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


def count_observed_with_two_classes(col: np.ndarray) -> Tuple[int, bool]:
    valid = np.isfinite(col) & ((col == 0) | (col == 1))
    obs = col[valid]

    n_obs = int(len(obs))
    has_two_classes = n_obs > 0 and len(np.unique(obs)) > 1

    return n_obs, has_two_classes


def select_valid_antibiotics_for_species_fold(
    amr_train_sp: np.ndarray,
    amr_val_sp: np.ndarray,
    min_train_obs: int,
    min_val_obs: int,
) -> List[int]:
    """
    Selecciona antibióticos evaluables dentro de una especie y fold.

    Criterios:
        - suficientes observaciones en train
        - dos clases en train
        - suficientes observaciones en val
        - dos clases en val
    """

    valid_cols = []

    for j in range(amr_train_sp.shape[1]):
        train_col = amr_train_sp[:, j]
        val_col = amr_val_sp[:, j]

        n_train, train_two_classes = count_observed_with_two_classes(train_col)
        n_val, val_two_classes = count_observed_with_two_classes(val_col)

        if n_train < min_train_obs:
            continue
        if not train_two_classes:
            continue
        if n_val < min_val_obs:
            continue
        if not val_two_classes:
            continue

        valid_cols.append(j)

    return valid_cols


# ============================================================
# Prediction
# ============================================================

def predict_logits(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()

    preds = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))

            xb = torch.from_numpy(X[start:end].astype(np.float32)).to(device)
            logits = model(xb).detach().cpu().numpy()

            preds.append(logits)

    if not preds:
        return np.empty((0, 1), dtype=np.float32)

    return np.concatenate(preds, axis=0)


# ============================================================
# Training: binary
# ============================================================

def train_binary_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    train_config: TrainConfig,
    device: torch.device,
) -> nn.Module:
    model = PaperMLP(
        input_dim=X_train.shape[1],
        output_dim=1,
        layer1=int(params["layer1"]),
        layer2=int(params["layer2"]),
        layer3=int(params["layer3"]),
        activation=str(params["activation"]),
    ).to(device)

    optimizer = build_optimizer(model, params)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = make_loader(
        X=X_train,
        y=y_train.reshape(-1, 1),
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )

    stopper = EarlyStopper(patience=train_config.patience)

    for epoch in range(train_config.epochs):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb).view(-1, 1)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

        # validation loss
        model.eval()

        with torch.no_grad():
            x_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
            y_val_t = torch.from_numpy(y_val.reshape(-1, 1).astype(np.float32)).to(device)

            logits_val = model(x_val_t).view(-1, 1)
            val_loss = criterion(logits_val, y_val_t).item()

        if stopper.step(val_loss, model):
            break

    stopper.restore(model)

    return model


# ============================================================
# Training: multilabel with masked BCE
# ============================================================

def masked_bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    targets contiene 0/1/NaN.

    La loss se calcula solo donde targets no es NaN.
    """

    mask = torch.isfinite(targets)

    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    clean_targets = torch.where(mask, targets, torch.zeros_like(targets))

    raw_loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        clean_targets,
        reduction="none",
    )

    masked_loss = raw_loss[mask].mean()

    return masked_loss


def train_multilabel_mlp(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    params: Dict[str, Any],
    train_config: TrainConfig,
    device: torch.device,
) -> nn.Module:
    model = PaperMLP(
        input_dim=X_train.shape[1],
        output_dim=Y_train.shape[1],
        layer1=int(params["layer1"]),
        layer2=int(params["layer2"]),
        layer3=int(params["layer3"]),
        activation=str(params["activation"]),
    ).to(device)

    optimizer = build_optimizer(model, params)

    train_loader = make_loader(
        X=X_train,
        y=Y_train,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )

    stopper = EarlyStopper(patience=train_config.patience)

    for epoch in range(train_config.epochs):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = masked_bce_with_logits_loss(logits, yb)

            loss.backward()
            optimizer.step()

        # validation masked loss
        model.eval()

        val_losses = []

        with torch.no_grad():
            for start in range(0, len(X_val), train_config.val_batch_size):
                end = min(start + train_config.val_batch_size, len(X_val))

                xb = torch.from_numpy(X_val[start:end].astype(np.float32)).to(device)
                yb = torch.from_numpy(Y_val[start:end].astype(np.float32)).to(device)

                logits = model(xb)
                val_loss = masked_bce_with_logits_loss(logits, yb)

                if torch.isfinite(val_loss):
                    val_losses.append(float(val_loss.item()))

        if val_losses:
            mean_val_loss = float(np.mean(val_losses))
        else:
            mean_val_loss = np.inf

        if stopper.step(mean_val_loss, model):
            break

    stopper.restore(model)

    return model


# ============================================================
# Evaluation per species and fold
# ============================================================

def evaluate_species_fold(
    fold: int,
    species: str,
    antibiotics_all: Sequence[str],
    X_train_sp: np.ndarray,
    X_val_sp: np.ndarray,
    amr_train_sp: np.ndarray,
    amr_val_sp: np.ndarray,
    params: SpeciesParams,
    train_config: TrainConfig,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve:
        - fold record por especie
        - records por especie-antibiótico
        - y_val_binary, p_val_binary
        - y_val_multilabel, p_val_multilabel

    Las matrices y/p devueltas tienen shape (n_val_sp, n_antibiotics_total),
    con NaN donde no hay etiqueta/predicción.
    """

    n_antibiotics_total = amr_train_sp.shape[1]

    valid_cols = select_valid_antibiotics_for_species_fold(
        amr_train_sp=amr_train_sp,
        amr_val_sp=amr_val_sp,
        min_train_obs=train_config.min_train_obs_per_antibiotic,
        min_val_obs=train_config.min_val_obs_per_antibiotic,
    )

    species_antibiotic_records: List[Dict[str, Any]] = []

    binary_pred_matrix = np.full_like(amr_val_sp, fill_value=np.nan, dtype=np.float32)
    multilabel_pred_matrix = np.full_like(amr_val_sp, fill_value=np.nan, dtype=np.float32)

    if len(valid_cols) == 0:
        record = {
            "fold": fold,
            "species": species,
            "species_pretty": pretty_species_name(species),
            "n_train_samples": int(X_train_sp.shape[0]),
            "n_val_samples": int(X_val_sp.shape[0]),
            "n_valid_antibiotics": 0,
            "binary_micro_auc": np.nan,
            "binary_macro_auc": np.nan,
            "multilabel_micro_auc": np.nan,
            "multilabel_macro_auc": np.nan,
            "n_binary_val_pairs": 0,
            "n_multilabel_val_pairs": 0,
            "multilabel_params_source": params.multilabel_params_source,
        }

        return (
            record,
            species_antibiotic_records,
            amr_val_sp.copy(),
            binary_pred_matrix,
            amr_val_sp.copy(),
            multilabel_pred_matrix,
        )

    selected_antibiotics = [str(antibiotics_all[j]) for j in valid_cols]

    logging.info(
        "[Fold %d | %s] Valid antibiotics: %d | %s",
        fold,
        species,
        len(valid_cols),
        selected_antibiotics,
    )

    # --------------------------------------------------------
    # 1) Binary MLP por antibiótico
    # --------------------------------------------------------

    for j in valid_cols:
        antibiotic = str(antibiotics_all[j])

        train_col = amr_train_sp[:, j]
        val_col = amr_val_sp[:, j]

        train_mask = np.isfinite(train_col) & ((train_col == 0) | (train_col == 1))
        val_mask = np.isfinite(val_col) & ((val_col == 0) | (val_col == 1))

        X_train_ab = X_train_sp[train_mask]
        y_train_ab = train_col[train_mask].astype(np.float32)

        X_val_ab = X_val_sp[val_mask]
        y_val_ab = val_col[val_mask].astype(np.float32)

        n_train_obs = int(len(y_train_ab))
        n_val_obs = int(len(y_val_ab))

        ab_auc_binary = np.nan

        if (
            n_train_obs >= train_config.min_train_obs_per_antibiotic
            and n_val_obs >= train_config.min_val_obs_per_antibiotic
            and len(np.unique(y_train_ab)) > 1
            and len(np.unique(y_val_ab)) > 1
        ):
            bin_params = params.binary_params_by_antibiotic.get(antibiotic, sanitize_params(None))
            bin_source = params.binary_params_source_by_antibiotic.get(antibiotic, "default")

            try:
                bin_model = train_binary_mlp(
                    X_train=X_train_ab,
                    y_train=y_train_ab,
                    X_val=X_val_ab,
                    y_val=y_val_ab,
                    params=bin_params,
                    train_config=train_config,
                    device=device,
                )

                logits_val = predict_logits(
                    model=bin_model,
                    X=X_val_ab,
                    device=device,
                    batch_size=train_config.val_batch_size,
                ).reshape(-1)

                prob_val = sigmoid_np(logits_val)

                binary_pred_matrix[val_mask, j] = prob_val.astype(np.float32)

                ab_auc_binary = safe_binary_auc(y_val_ab, prob_val)

                del bin_model
                torch.cuda.empty_cache()

            except Exception as e:
                bin_source = params.binary_params_source_by_antibiotic.get(antibiotic, "default")
                logging.exception(
                    "[Fold %d | %s | %s] Error entrenando binary MLP: %s",
                    fold,
                    species,
                    antibiotic,
                    e,
                )
        else:
            bin_source = params.binary_params_source_by_antibiotic.get(antibiotic, "default")

        species_antibiotic_records.append(
            {
                "fold": fold,
                "species": species,
                "species_pretty": pretty_species_name(species),
                "antibiotic": antibiotic,
                "antibiotic_index": int(j),
                "n_train_obs": n_train_obs,
                "n_val_obs": n_val_obs,
                "binary_auc": maybe_float(ab_auc_binary),
                "binary_params_source": bin_source,
                "multilabel_auc": np.nan,
                "multilabel_params_source": params.multilabel_params_source,
            }
        )

        gc.collect()

    # --------------------------------------------------------
    # 2) Multilabel MLP por especie con masked loss
    # --------------------------------------------------------

    Y_train_ml = amr_train_sp[:, valid_cols].astype(np.float32)
    Y_val_ml = amr_val_sp[:, valid_cols].astype(np.float32)

    try:
        ml_model = train_multilabel_mlp(
            X_train=X_train_sp,
            Y_train=Y_train_ml,
            X_val=X_val_sp,
            Y_val=Y_val_ml,
            params=params.multilabel_params,
            train_config=train_config,
            device=device,
        )

        ml_logits = predict_logits(
            model=ml_model,
            X=X_val_sp,
            device=device,
            batch_size=train_config.val_batch_size,
        )

        ml_prob = sigmoid_np(ml_logits).astype(np.float32)

        multilabel_pred_matrix[:, valid_cols] = ml_prob

        del ml_model
        torch.cuda.empty_cache()

    except Exception as e:
        logging.exception(
            "[Fold %d | %s] Error entrenando multilabel MLP: %s",
            fold,
            species,
            e,
        )

    # calcular AUC multilabel por antibiótico y rellenar records
    for local_idx, j in enumerate(valid_cols):
        antibiotic = str(antibiotics_all[j])

        val_col = amr_val_sp[:, j]
        pred_col = multilabel_pred_matrix[:, j]

        ml_auc_j = safe_binary_auc(val_col, pred_col)

        for rec in species_antibiotic_records:
            if rec["antibiotic_index"] == int(j):
                rec["multilabel_auc"] = maybe_float(ml_auc_j)
                break

    # --------------------------------------------------------
    # Métricas species-fold
    # --------------------------------------------------------

    binary_micro_auc, binary_macro_auc, _ = compute_micro_macro_auc_matrix(
        y_true=amr_val_sp,
        y_score=binary_pred_matrix,
    )

    multilabel_micro_auc, multilabel_macro_auc, _ = compute_micro_macro_auc_matrix(
        y_true=amr_val_sp,
        y_score=multilabel_pred_matrix,
    )

    n_binary_val_pairs = int(np.sum(np.isfinite(amr_val_sp) & np.isfinite(binary_pred_matrix)))
    n_multilabel_val_pairs = int(np.sum(np.isfinite(amr_val_sp) & np.isfinite(multilabel_pred_matrix)))

    record = {
        "fold": fold,
        "species": species,
        "species_pretty": pretty_species_name(species),
        "n_train_samples": int(X_train_sp.shape[0]),
        "n_val_samples": int(X_val_sp.shape[0]),
        "n_valid_antibiotics": int(len(valid_cols)),
        "valid_antibiotics": ";".join(selected_antibiotics),
        "binary_micro_auc": maybe_float(binary_micro_auc),
        "binary_macro_auc": maybe_float(binary_macro_auc),
        "multilabel_micro_auc": maybe_float(multilabel_micro_auc),
        "multilabel_macro_auc": maybe_float(multilabel_macro_auc),
        "n_binary_val_pairs": n_binary_val_pairs,
        "n_multilabel_val_pairs": n_multilabel_val_pairs,
        "multilabel_params_source": params.multilabel_params_source,
    }

    return (
        record,
        species_antibiotic_records,
        amr_val_sp.copy(),
        binary_pred_matrix,
        amr_val_sp.copy(),
        multilabel_pred_matrix,
    )


# ============================================================
# Summaries
# ============================================================

def summarize_species_fold_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []

    for species, sub in df.groupby("species"):
        row = {
            "species": species,
            "species_pretty": pretty_species_name(species),
            "n_folds_with_results": int(len(sub)),
            "mean_train_samples": float(sub["n_train_samples"].mean()),
            "mean_val_samples": float(sub["n_val_samples"].mean()),
            "mean_valid_antibiotics": float(sub["n_valid_antibiotics"].mean()),
            "mean_binary_val_pairs": float(sub["n_binary_val_pairs"].mean()),
            "mean_multilabel_val_pairs": float(sub["n_multilabel_val_pairs"].mean()),
            "binary_micro_auc_mean": float(sub["binary_micro_auc"].mean(skipna=True)),
            "binary_micro_auc_std": float(sub["binary_micro_auc"].std(skipna=True, ddof=1)),
            "binary_macro_auc_mean": float(sub["binary_macro_auc"].mean(skipna=True)),
            "binary_macro_auc_std": float(sub["binary_macro_auc"].std(skipna=True, ddof=1)),
            "multilabel_micro_auc_mean": float(sub["multilabel_micro_auc"].mean(skipna=True)),
            "multilabel_micro_auc_std": float(sub["multilabel_micro_auc"].std(skipna=True, ddof=1)),
            "multilabel_macro_auc_mean": float(sub["multilabel_macro_auc"].mean(skipna=True)),
            "multilabel_macro_auc_std": float(sub["multilabel_macro_auc"].std(skipna=True, ddof=1)),
        }

        # mejor modelo por micro AUC medio
        candidates = {
            "binary_micro": row["binary_micro_auc_mean"],
            "multilabel_micro": row["multilabel_micro_auc_mean"],
        }

        valid_candidates = {
            k: v for k, v in candidates.items()
            if v is not None and not np.isnan(v)
        }

        if valid_candidates:
            best_name = max(valid_candidates, key=valid_candidates.get)
            best_auc = valid_candidates[best_name]
        else:
            best_name = None
            best_auc = np.nan

        row["best_model_by_micro_auc"] = best_name
        row["best_micro_auc"] = best_auc

        rows.append(row)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            by=["best_micro_auc", "species"],
            ascending=[False, True],
        ).reset_index(drop=True)

    return out


def format_mean_std(mean_val: float, std_val: float, decimals: int = 4) -> str:
    if mean_val is None or np.isnan(mean_val):
        return "NaN"

    if std_val is None or np.isnan(std_val):
        return f"{mean_val:.{decimals}f}"

    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


def build_public_species_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    out = pd.DataFrame()

    out["Species"] = summary_df["species_pretty"]

    out["Binary micro AUC"] = [
        format_mean_std(m, s)
        for m, s in zip(summary_df["binary_micro_auc_mean"], summary_df["binary_micro_auc_std"])
    ]

    out["Binary macro AUC"] = [
        format_mean_std(m, s)
        for m, s in zip(summary_df["binary_macro_auc_mean"], summary_df["binary_macro_auc_std"])
    ]

    out["Multilabel micro AUC"] = [
        format_mean_std(m, s)
        for m, s in zip(summary_df["multilabel_micro_auc_mean"], summary_df["multilabel_micro_auc_std"])
    ]

    out["Multilabel macro AUC"] = [
        format_mean_std(m, s)
        for m, s in zip(summary_df["multilabel_macro_auc_mean"], summary_df["multilabel_macro_auc_std"])
    ]

    out["Best model by micro AUC"] = summary_df["best_model_by_micro_auc"]

    out["Best micro AUC"] = summary_df["best_micro_auc"].map(
        lambda x: f"{x:.4f}" if pd.notna(x) else "NaN"
    )

    out["Mean val samples"] = summary_df["mean_val_samples"].round(1)
    out["Mean valid antibiotics"] = summary_df["mean_valid_antibiotics"].round(1)
    out["Mean multilabel val pairs"] = summary_df["mean_multilabel_val_pairs"].round(1)

    return out


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fair MLP benchmark without LPS complete-case filtering."
    )

    parser.add_argument(
        "--pickle-path",
        type=str,
        required=True,
        help="Ruta al pickle COMBINED_MARISMA_DRIAMS_samples.pkl o equivalente.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directorio donde guardar resultados.",
    )

    parser.add_argument(
        "--benchmark-roots",
        nargs="*",
        default=None,
        help=(
            "Opcional. Roots antiguos de benchmark MLP para cargar params Optuna. "
            "Si una especie/antibiótico no tiene params, se usan defaults."
        ),
    )

    parser.add_argument(
        "--species",
        nargs="*",
        default=None,
        help=(
            "Opcional. Lista de especies concretas. "
            "Si no se indica, se usan todas las especies del pickle, como el recommender global."
        ),
    )

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0")

    parser.add_argument(
        "--min-train-obs-per-antibiotic",
        type=int,
        default=DEFAULT_MIN_TRAIN_OBS_PER_ANTIBIOTIC,
        help="Mínimo de observaciones en train por especie-antibiótico.",
    )

    parser.add_argument(
        "--min-val-obs-per-antibiotic",
        type=int,
        default=DEFAULT_MIN_VAL_OBS_PER_ANTIBIOTIC,
        help="Mínimo de observaciones en validación por especie-antibiótico.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)

    train_config = TrainConfig(
        n_splits=args.n_splits,
        random_seed=args.random_seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        min_train_obs_per_antibiotic=args.min_train_obs_per_antibiotic,
        min_val_obs_per_antibiotic=args.min_val_obs_per_antibiotic,
    )

    logging.info("Starting fair MLP benchmark without LPS.")
    logging.info("Args: %s", vars(args))
    logging.info("Train config: %s", asdict(train_config))

    seed_everything(train_config.random_seed)

    device = choose_device(train_config.device)

    logging.info("Device selected: %s", device)

    if device.type == "cuda":
        logging.info("CUDA device name: %s", torch.cuda.get_device_name(device))

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    logging.info("Loading pickle: %s", args.pickle_path)

    payload = load_pickle_payload(args.pickle_path)

    X_all = coerce_X_to_2d_float32(payload["data"])
    species_all = normalize_species_labels(payload["label"])
    amr_all = coerce_amr_to_matrix(payload["amr"])

    antibiotics_all = [str(a) for a in list(payload["antibiotics"])]

    if len(X_all) != len(species_all) or len(X_all) != len(amr_all):
        raise ValueError("Longitudes inconsistentes entre data, label y amr.")

    if len(antibiotics_all) != amr_all.shape[1]:
        raise ValueError(
            f"Número de antibióticos no coincide: len(antibiotics)={len(antibiotics_all)} "
            f"pero amr.shape[1]={amr_all.shape[1]}"
        )

    X_all, species_all, amr_all = clean_spectra_and_labels(
        X_all=X_all,
        species_all=species_all,
        amr_all=amr_all,
    )

    n_samples = X_all.shape[0]
    n_features = X_all.shape[1]
    n_antibiotics = amr_all.shape[1]

    logging.info("Samples after removing invalid spectra: %d", n_samples)
    logging.info("Features: %d", n_features)
    logging.info("Antibiotics: %d", n_antibiotics)

    unique_species = sorted(np.unique(species_all).tolist())

    if args.species:
        requested = set(args.species)
        unique_species = [sp for sp in unique_species if sp in requested]
        missing = sorted(requested - set(unique_species))

        if missing:
            logging.warning("Requested species not found in pickle: %s", missing)

    if not unique_species:
        raise RuntimeError("No species selected.")

    logging.info("Species to evaluate (%d): %s", len(unique_species), unique_species)

    # --------------------------------------------------------
    # Global KFold over samples, like recommender
    # --------------------------------------------------------

    kf = KFold(
        n_splits=train_config.n_splits,
        shuffle=True,
        random_state=train_config.random_seed,
    )

    sample_indices = np.arange(n_samples)

    all_species_fold_records: List[Dict[str, Any]] = []
    all_species_antibiotic_records: List[Dict[str, Any]] = []
    global_fold_records: List[Dict[str, Any]] = []

    # Caches params per species
    species_params_cache: Dict[str, SpeciesParams] = {}

    t_global_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(sample_indices)):
        logging.info("=" * 100)
        logging.info("FOLD %d/%d", fold + 1, train_config.n_splits)
        logging.info("=" * 100)

        X_train_global = X_all[train_idx]
        X_val_global = X_all[val_idx]

        species_train_global = species_all[train_idx]
        species_val_global = species_all[val_idx]

        amr_train_global = amr_all[train_idx]
        amr_val_global = amr_all[val_idx]

        logging.info("Global train samples: %d", X_train_global.shape[0])
        logging.info("Global val samples: %d", X_val_global.shape[0])

        # matrices globales de predicción para este fold
        binary_global_y_list = []
        binary_global_p_list = []

        multilabel_global_y_list = []
        multilabel_global_p_list = []

        for sp_i, species in enumerate(unique_species, start=1):
            logging.info("-" * 100)
            logging.info(
                "Fold %d | Species %d/%d: %s",
                fold + 1,
                sp_i,
                len(unique_species),
                species,
            )

            sp_train_mask = species_train_global == species
            sp_val_mask = species_val_global == species

            n_train_sp = int(np.sum(sp_train_mask))
            n_val_sp = int(np.sum(sp_val_mask))

            if n_train_sp == 0 or n_val_sp == 0:
                logging.warning(
                    "[Fold %d | %s] Skipping: n_train=%d, n_val=%d",
                    fold,
                    species,
                    n_train_sp,
                    n_val_sp,
                )
                continue

            X_train_sp = X_train_global[sp_train_mask]
            X_val_sp = X_val_global[sp_val_mask]

            amr_train_sp = amr_train_global[sp_train_mask]
            amr_val_sp = amr_val_global[sp_val_mask]

            if species not in species_params_cache:
                species_params_cache[species] = discover_params_for_species(
                    species=species,
                    antibiotics=antibiotics_all,
                    benchmark_roots=args.benchmark_roots,
                )

                logging.info(
                    "[%s] Multilabel params source: %s",
                    species,
                    species_params_cache[species].multilabel_params_source,
                )

            params = species_params_cache[species]

            t0 = time.time()

            try:
                (
                    species_fold_record,
                    species_ab_records,
                    y_val_binary,
                    p_val_binary,
                    y_val_multilabel,
                    p_val_multilabel,
                ) = evaluate_species_fold(
                    fold=fold,
                    species=species,
                    antibiotics_all=antibiotics_all,
                    X_train_sp=X_train_sp,
                    X_val_sp=X_val_sp,
                    amr_train_sp=amr_train_sp,
                    amr_val_sp=amr_val_sp,
                    params=params,
                    train_config=train_config,
                    device=device,
                )

                elapsed = time.time() - t0

                species_fold_record["elapsed_seconds"] = elapsed

                all_species_fold_records.append(species_fold_record)
                all_species_antibiotic_records.extend(species_ab_records)

                # global aggregation
                binary_valid = np.isfinite(y_val_binary) & np.isfinite(p_val_binary)

                if np.any(binary_valid):
                    binary_global_y_list.append(y_val_binary[binary_valid])
                    binary_global_p_list.append(p_val_binary[binary_valid])

                multilabel_valid = np.isfinite(y_val_multilabel) & np.isfinite(p_val_multilabel)

                if np.any(multilabel_valid):
                    multilabel_global_y_list.append(y_val_multilabel[multilabel_valid])
                    multilabel_global_p_list.append(p_val_multilabel[multilabel_valid])

                logging.info(
                    "[Fold %d | %s] Done in %.2f min | "
                    "Binary micro=%.4f macro=%.4f | "
                    "Multilabel micro=%.4f macro=%.4f",
                    fold,
                    species,
                    elapsed / 60,
                    species_fold_record["binary_micro_auc"]
                    if pd.notna(species_fold_record["binary_micro_auc"]) else np.nan,
                    species_fold_record["binary_macro_auc"]
                    if pd.notna(species_fold_record["binary_macro_auc"]) else np.nan,
                    species_fold_record["multilabel_micro_auc"]
                    if pd.notna(species_fold_record["multilabel_micro_auc"]) else np.nan,
                    species_fold_record["multilabel_macro_auc"]
                    if pd.notna(species_fold_record["multilabel_macro_auc"]) else np.nan,
                )

            except Exception as e:
                logging.exception(
                    "[Fold %d | %s] Fatal error evaluating species: %s",
                    fold,
                    species,
                    e,
                )

            gc.collect()

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # --------------------------------------------------------
        # Global fold metrics over all species-specific predictions
        # --------------------------------------------------------

        binary_global_micro_auc = np.nan
        multilabel_global_micro_auc = np.nan

        if binary_global_y_list:
            y_bin = np.concatenate(binary_global_y_list)
            p_bin = np.concatenate(binary_global_p_list)
            binary_global_micro_auc = safe_binary_auc(y_bin, p_bin)
            n_binary_pairs_global = int(len(y_bin))
        else:
            n_binary_pairs_global = 0

        if multilabel_global_y_list:
            y_ml = np.concatenate(multilabel_global_y_list)
            p_ml = np.concatenate(multilabel_global_p_list)
            multilabel_global_micro_auc = safe_binary_auc(y_ml, p_ml)
            n_multilabel_pairs_global = int(len(y_ml))
        else:
            n_multilabel_pairs_global = 0

        # macro global as mean of species macro values in that fold
        fold_species_df = pd.DataFrame(
            [r for r in all_species_fold_records if r["fold"] == fold]
        )

        if not fold_species_df.empty:
            binary_global_macro_species_auc = float(
                fold_species_df["binary_macro_auc"].mean(skipna=True)
            )
            multilabel_global_macro_species_auc = float(
                fold_species_df["multilabel_macro_auc"].mean(skipna=True)
            )
        else:
            binary_global_macro_species_auc = np.nan
            multilabel_global_macro_species_auc = np.nan

        global_fold_record = {
            "fold": fold,
            "n_train_samples": int(len(train_idx)),
            "n_val_samples": int(len(val_idx)),
            "n_species_evaluated": int(fold_species_df["species"].nunique())
            if not fold_species_df.empty else 0,
            "binary_global_micro_auc": maybe_float(binary_global_micro_auc),
            "binary_global_macro_species_auc": maybe_float(binary_global_macro_species_auc),
            "multilabel_global_micro_auc": maybe_float(multilabel_global_micro_auc),
            "multilabel_global_macro_species_auc": maybe_float(multilabel_global_macro_species_auc),
            "n_binary_val_pairs": n_binary_pairs_global,
            "n_multilabel_val_pairs": n_multilabel_pairs_global,
        }

        global_fold_records.append(global_fold_record)

        logging.info(
            "[Fold %d] GLOBAL | Binary micro=%.4f | Multilabel micro=%.4f",
            fold,
            binary_global_micro_auc if pd.notna(binary_global_micro_auc) else np.nan,
            multilabel_global_micro_auc if pd.notna(multilabel_global_micro_auc) else np.nan,
        )

        # Save intermediate results after each fold
        pd.DataFrame(all_species_fold_records).to_csv(
            output_dir / "fair_mlp_all_species_fold_results.csv",
            index=False,
        )

        pd.DataFrame(all_species_antibiotic_records).to_csv(
            output_dir / "fair_mlp_species_antibiotic_fold_results.csv",
            index=False,
        )

        pd.DataFrame(global_fold_records).to_csv(
            output_dir / "fair_mlp_global_fold_results.csv",
            index=False,
        )

    # ========================================================
    # Final outputs
    # ========================================================

    df_species_fold = pd.DataFrame(all_species_fold_records)
    df_species_ab = pd.DataFrame(all_species_antibiotic_records)
    df_global_fold = pd.DataFrame(global_fold_records)

    df_species_summary = summarize_species_fold_results(df_species_fold)
    df_public_species_summary = build_public_species_table(df_species_summary)

    df_species_fold.to_csv(
        output_dir / "fair_mlp_all_species_fold_results.csv",
        index=False,
    )

    df_species_ab.to_csv(
        output_dir / "fair_mlp_species_antibiotic_fold_results.csv",
        index=False,
    )

    df_global_fold.to_csv(
        output_dir / "fair_mlp_global_fold_results.csv",
        index=False,
    )

    df_species_summary.to_csv(
        output_dir / "fair_mlp_species_summary.csv",
        index=False,
    )

    df_public_species_summary.to_csv(
        output_dir / "fair_mlp_species_summary_public_table.csv",
        index=False,
    )

    with open(output_dir / "fair_mlp_species_summary_public_table.md", "w", encoding="utf-8") as f:
        if not df_public_species_summary.empty:
            f.write(df_public_species_summary.to_markdown(index=False))
        else:
            f.write("No results.")

    # Global summary JSON
    global_summary = {
        "train_config": asdict(train_config),
        "device": str(device),
        "pickle_path": args.pickle_path,
        "benchmark_roots": args.benchmark_roots,
        "n_samples_after_cleaning_invalid_spectra": int(n_samples),
        "n_features": int(n_features),
        "n_antibiotics": int(n_antibiotics),
        "n_species_requested": int(len(unique_species)),
        "species": unique_species,
        "elapsed_seconds": float(time.time() - t_global_start),
    }

    if not df_global_fold.empty:
        for col in [
            "binary_global_micro_auc",
            "binary_global_macro_species_auc",
            "multilabel_global_micro_auc",
            "multilabel_global_macro_species_auc",
        ]:
            global_summary[f"{col}_mean"] = float(df_global_fold[col].mean(skipna=True))
            global_summary[f"{col}_std"] = float(df_global_fold[col].std(skipna=True, ddof=1))

    json_dump(
        global_summary,
        output_dir / "fair_mlp_global_summary.json",
    )

    logging.info("=" * 100)
    logging.info("FAIR MLP BENCHMARK COMPLETED")
    logging.info("Results saved in: %s", output_dir)
    logging.info("Species fold results: %s", output_dir / "fair_mlp_all_species_fold_results.csv")
    logging.info("Species summary: %s", output_dir / "fair_mlp_species_summary.csv")
    logging.info("Public table: %s", output_dir / "fair_mlp_species_summary_public_table.csv")
    logging.info("Antibiotic fold results: %s", output_dir / "fair_mlp_species_antibiotic_fold_results.csv")
    logging.info("Global fold results: %s", output_dir / "fair_mlp_global_fold_results.csv")
    logging.info("Global summary: %s", output_dir / "fair_mlp_global_summary.json")


if __name__ == "__main__":
    main()