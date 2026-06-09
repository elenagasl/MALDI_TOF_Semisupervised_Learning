#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluación comparativa de AUCs para modelos MLP ya optimizados con Optuna.

Qué hace:
1. Descubre especies en múltiples directorios de benchmark.
2. Carga los hiperparámetros:
   - LPS
   - Multilabel
   - Binarios por antibiótico
3. Reconstruye el dataset por especie a partir del pickle principal:
   - filtra por especie
   - usa solo selected_antibiotics
   - elimina NaNs en espectro/AMR
   - construye patrones LPS
   - elimina patrones raros según min_pattern_count
4. Ejecuta 10 splits 80/20 estratificados por patrón LPS
5. Entrena:
   - modelo binario por antibiótico
   - modelo multilabel
   - modelo LPS multiclass
6. Calcula AUC por fold y resume media/std
7. Guarda todos los resultados en disco

Uso típico:
nohup python run_auc_benchmark_mlp.py \
  --pickle-path "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS.pkl" \
  --benchmark-roots \
    "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/src/benchmark study/benchmark_full/benchmark_outputs/benchmark_mlp_20260314_200946" \
    "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/src/benchmark study/benchmark_full/benchmark_outputs/benchmark_mlp_20260316_112641" \
  --output-dir "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/src/benchmark study/benchmark_full/benchmark_outputs/auc_re_evaluation_20260326" \
  --device auto \
  > auc_eval.log 2>&1 &
"""

from __future__ import annotations

import argparse
import copy
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
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Config / dataclasses
# ============================================================

@dataclass
class SpeciesBenchmarkPaths:
    species: str
    root_dir: str
    species_dir: str
    data_summary_json: str
    lps_params_json: str
    multilabel_params_json: str
    binary_models_dir: str
    binary_param_files: Dict[str, str]


@dataclass
class SpeciesDataConfig:
    species: str
    selected_antibiotics: List[str]
    min_pattern_count: int
    n_total_species_samples: Optional[int] = None
    n_complete_samples_before_pattern_filter: Optional[int] = None
    n_samples_after_pattern_filter: Optional[int] = None
    dropped_constant_antibiotics: Optional[List[str]] = None


@dataclass
class PreparedSpeciesDataset:
    species: str
    X: np.ndarray                      # shape (N, D), float32
    Y: np.ndarray                      # shape (N, A), float32/binary
    lps_strings: np.ndarray            # shape (N,), strings like "101010"
    selected_antibiotics: List[str]
    min_pattern_count: int


@dataclass
class TrainConfig:
    epochs: int = 1200
    patience: int = 50
    batch_size: int = 128
    test_size: float = 0.2
    n_splits: int = 10
    random_seed: int = 42
    num_workers: int = 0
    device: str = "auto"


# ============================================================
# Logging
# ============================================================

def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


# ============================================================
# Utils
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


def pretty_species_name(species: str) -> str:
    return species.replace("_", " ")


def json_load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_markdown_table(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))


def maybe_float(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


# ============================================================
# Activations / model
# ============================================================

def activation_from_name(name: str) -> nn.Module:
    name = str(name).strip().lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "logistic":
        return nn.Sigmoid()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Activación no soportada: {name}")


class PaperMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layer1: int, layer2: int, layer3: int, activation: str):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Descubrimiento de benchmarks
# ============================================================

def discover_species_benchmarks(benchmark_roots: Sequence[str]) -> Dict[str, SpeciesBenchmarkPaths]:
    """
    Descubre las especies disponibles en varios roots.
    Si una especie aparece en varios roots, se prioriza el root más reciente
    por orden lexicográfico del nombre de carpeta.
    """
    root_paths = [Path(r) for r in benchmark_roots]
    for r in root_paths:
        if not r.exists():
            raise FileNotFoundError(f"No existe benchmark root: {r}")

    discovered: Dict[str, List[SpeciesBenchmarkPaths]] = {}

    for root in root_paths:
        for species_dir in root.iterdir():
            if not species_dir.is_dir():
                continue

            species = species_dir.name
            data_summary_json = species_dir / f"{species}__data_summary.json"
            lps_params_json = species_dir / f"{species}__lps_best_params.json"
            multilabel_params_json = species_dir / f"{species}__multilabel_best_params.json"
            binary_models_dir = species_dir / "binary_models"

            if not data_summary_json.exists():
                continue
            if not lps_params_json.exists():
                continue
            if not multilabel_params_json.exists():
                continue
            if not binary_models_dir.exists():
                continue

            binary_param_files: Dict[str, str] = {}
            for p in binary_models_dir.glob(f"{species}__*__binary_best_params.json"):
                stem = p.stem  # species__Antibiotic__binary_best_params
                antibiotic = stem.replace(f"{species}__", "").replace("__binary_best_params", "")
                binary_param_files[antibiotic] = str(p)

            sp = SpeciesBenchmarkPaths(
                species=species,
                root_dir=str(root),
                species_dir=str(species_dir),
                data_summary_json=str(data_summary_json),
                lps_params_json=str(lps_params_json),
                multilabel_params_json=str(multilabel_params_json),
                binary_models_dir=str(binary_models_dir),
                binary_param_files=binary_param_files
            )
            discovered.setdefault(species, []).append(sp)

    if not discovered:
        raise RuntimeError("No se encontraron especies válidas en los benchmark roots proporcionados.")

    resolved: Dict[str, SpeciesBenchmarkPaths] = {}
    for species, candidates in discovered.items():
        # priorizar directorio root más reciente por nombre
        candidates_sorted = sorted(candidates, key=lambda x: Path(x.root_dir).name)
        chosen = candidates_sorted[-1]
        resolved[species] = chosen

        if len(candidates) > 1:
            logging.warning(
                "Especie %s encontrada en múltiples roots. Se usará: %s",
                species,
                chosen.root_dir
            )

    return resolved


def load_species_data_config(paths: SpeciesBenchmarkPaths) -> SpeciesDataConfig:
    data = json_load(Path(paths.data_summary_json))
    return SpeciesDataConfig(
        species=data["species"],
        selected_antibiotics=list(data["selected_antibiotics"]),
        min_pattern_count=int(data["min_pattern_count"]),
        n_total_species_samples=data.get("n_total_species_samples"),
        n_complete_samples_before_pattern_filter=data.get("n_complete_samples_before_pattern_filter"),
        n_samples_after_pattern_filter=data.get("n_samples_after_pattern_filter"),
        dropped_constant_antibiotics=data.get("dropped_constant_antibiotics", []),
    )


# ============================================================
# Carga / preparación del pickle
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
        if isinstance(v, float) and np.isnan(v):
            return np.nan
        fv = float(v)
        if fv in (0.0, 1.0):
            return fv
        if math.isnan(fv):
            return np.nan
        # cualquier otro valor numérico se considera inválido
        return np.nan

    s = str(v).strip().lower()
    if s in {"1", "true", "r", "resistant", "resistance"}:
        return 1.0
    if s in {"0", "false", "s", "susceptible"}:
        return 0.0
    if s in {"nan", "none", "", "i", "intermediate"}:
        return np.nan

    return np.nan


def coerce_amr_matrix(amr: Any, payload_antibiotics: Sequence[str], selected_antibiotics: Sequence[str]) -> np.ndarray:
    """
    Devuelve matriz AMR shape (N, len(selected_antibiotics)) con valores {0,1,np.nan}
    """
    if isinstance(amr, pd.DataFrame):
        missing_cols = [ab for ab in selected_antibiotics if ab not in amr.columns]
        if missing_cols:
            raise KeyError(f"Antibióticos seleccionados no encontrados en amr DataFrame: {missing_cols}")

        sub = amr.loc[:, list(selected_antibiotics)].copy()
        arr = sub.values
    else:
        arr_all = np.asarray(amr)
        if arr_all.ndim != 2:
            raise ValueError(f"AMR debe ser 2D. Shape encontrado: {arr_all.shape}")

        ab_to_idx = {ab: i for i, ab in enumerate(payload_antibiotics)}
        missing_cols = [ab for ab in selected_antibiotics if ab not in ab_to_idx]
        if missing_cols:
            raise KeyError(f"Antibióticos seleccionados no encontrados en payload['antibiotics']: {missing_cols}")

        idxs = [ab_to_idx[ab] for ab in selected_antibiotics]
        arr = arr_all[:, idxs]

    out = np.empty(arr.shape, dtype=np.float32)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            out[i, j] = normalize_single_amr_value(arr[i, j])

    return out


def build_lps_strings(y_binary: np.ndarray) -> np.ndarray:
    """
    y_binary: shape (N, A), valores 0/1
    Devuelve patrones tipo "101010"
    """
    y_int = y_binary.astype(int)
    return np.array(["".join(map(str, row.tolist())) for row in y_int], dtype=object)


def prepare_species_dataset(
    payload: Dict[str, Any],
    species_config: SpeciesDataConfig
) -> PreparedSpeciesDataset:
    species = species_config.species
    selected_antibiotics = list(species_config.selected_antibiotics)

    X_all = coerce_X_to_2d_float32(payload["data"])
    y_species_all = normalize_species_labels(payload["label"])
    amr_all = coerce_amr_matrix(payload["amr"], payload["antibiotics"], selected_antibiotics)

    if len(X_all) != len(y_species_all) or len(X_all) != len(amr_all):
        raise ValueError("Longitudes inconsistentes entre X, label y amr.")

    species_mask = (y_species_all == species)
    X_sp = X_all[species_mask]
    Y_sp = amr_all[species_mask]

    if X_sp.shape[0] == 0:
        raise ValueError(f"No se encontraron muestras para la especie '{species}' en el pickle.")

    # filtrar NaNs en espectros o AMR
    mask_x_valid = np.isfinite(X_sp).all(axis=1)
    mask_y_valid = np.isfinite(Y_sp).all(axis=1)
    valid_mask = mask_x_valid & mask_y_valid

    X_valid = X_sp[valid_mask]
    Y_valid = Y_sp[valid_mask]

    if X_valid.shape[0] == 0:
        raise ValueError(f"Tras filtrar NaNs, no quedan muestras válidas para '{species}'.")

    lps_strings = build_lps_strings(Y_valid)

    # filtrar patrones raros
    vc = pd.Series(lps_strings).value_counts()
    keep_patterns = set(vc[vc >= species_config.min_pattern_count].index.tolist())
    keep_mask = np.array([p in keep_patterns for p in lps_strings], dtype=bool)

    X_final = X_valid[keep_mask].astype(np.float32, copy=False)
    Y_final = Y_valid[keep_mask].astype(np.float32, copy=False)
    lps_final = lps_strings[keep_mask]

    if X_final.shape[0] == 0:
        raise ValueError(f"Tras filtrar patrones raros, no quedan muestras para '{species}'.")

    logging.info(
        "[%s] Muestras finales: %d | Dim input: %d | Antibióticos: %s | Nº patrones LPS: %d",
        species,
        X_final.shape[0],
        X_final.shape[1],
        selected_antibiotics,
        len(np.unique(lps_final))
    )

    return PreparedSpeciesDataset(
        species=species,
        X=X_final,
        Y=Y_final,
        lps_strings=lps_final,
        selected_antibiotics=selected_antibiotics,
        min_pattern_count=species_config.min_pattern_count,
    )


# ============================================================
# Training helpers
# ============================================================

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.best_state = None
        self.counter = 0

    def step(self, score: float, model: nn.Module) -> bool:
        """
        score: mayor es mejor
        Devuelve True si hay que parar.
        """
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def build_optimizer(model: nn.Module, solver: str, learning_rate: float) -> torch.optim.Optimizer:
    solver = str(solver).lower()
    if solver == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if solver == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if solver == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    raise ValueError(f"Solver no soportado: {solver}")


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    x_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)


def predict_logits(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size].astype(np.float32)).to(device)
            logits = model(xb).detach().cpu().numpy()
            preds.append(logits)
    return np.concatenate(preds, axis=0)


def safe_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def safe_macro_multilabel_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, List[float]]:
    """
    AUC macro sobre columnas. Ignora columnas constantes en test.
    """
    aucs = []
    for j in range(y_true.shape[1]):
        auc = safe_binary_auc(y_true[:, j], y_score[:, j])
        aucs.append(auc)

    valid = [a for a in aucs if not np.isnan(a)]
    if not valid:
        return np.nan, aucs
    return float(np.mean(valid)), aucs


def safe_macro_ovr_multiclass_auc(y_true_int: np.ndarray, probas: np.ndarray, class_ids: Sequence[int]) -> Tuple[float, List[float]]:
    """
    AUC macro OVR calculada manualmente por clase.
    Ignora clases no presentes en test.
    """
    aucs = []
    y_true_int = np.asarray(y_true_int).astype(int)

    for class_id in class_ids:
        y_bin = (y_true_int == class_id).astype(int)
        auc = safe_binary_auc(y_bin, probas[:, class_id])
        aucs.append(auc)

    valid = [a for a in aucs if not np.isnan(a)]
    if not valid:
        return np.nan, aucs
    return float(np.mean(valid)), aucs


def train_single_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dim: int,
    params: Dict[str, Any],
    train_config: TrainConfig,
    device: torch.device,
    task_type: str,
) -> nn.Module:
    """
    task_type in {"binary", "multilabel", "multiclass_lps"}
    """
    input_dim = X_train.shape[1]

    model = PaperMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        layer1=int(params["layer1"]),
        layer2=int(params["layer2"]),
        layer3=int(params["layer3"]),
        activation=str(params["activation"]),
    ).to(device)

    optimizer = build_optimizer(model, solver=str(params["solver"]), learning_rate=float(params["learning_rate"]))

    if task_type in {"binary", "multilabel"}:
        criterion = nn.BCEWithLogitsLoss()
    elif task_type == "multiclass_lps":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"task_type no reconocido: {task_type}")

    train_loader = make_loader(
        X_train,
        y_train if task_type != "multiclass_lps" else y_train.reshape(-1, 1),
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )

    early_stopper = EarlyStopping(patience=train_config.patience, min_delta=0.0)

    for epoch in range(train_config.epochs):
        model.train()
        epoch_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)

            if task_type == "multiclass_lps":
                yb = yb.squeeze(1).long().to(device)
            else:
                yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)

            if task_type == "binary":
                logits = logits.view(-1)
                yb = yb.view(-1)

            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # validación con AUC para early stopping
        logits_val = predict_logits(model, X_val, device=device)
        stop_score = np.nan

        if task_type == "binary":
            prob_val = 1.0 / (1.0 + np.exp(-logits_val.reshape(-1)))
            stop_score = safe_binary_auc(y_val.reshape(-1), prob_val)

        elif task_type == "multilabel":
            prob_val = 1.0 / (1.0 + np.exp(-logits_val))
            stop_score, _ = safe_macro_multilabel_auc(y_val, prob_val)

        elif task_type == "multiclass_lps":
            # softmax
            logits_val = logits_val - logits_val.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits_val)
            prob_val = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            class_ids = list(range(output_dim))
            stop_score, _ = safe_macro_ovr_multiclass_auc(y_val.astype(int), prob_val, class_ids)

        if np.isnan(stop_score):
            # si el AUC es indefinido por cualquier motivo, usar la pérdida media negativa
            stop_score = -float(np.mean(epoch_losses))

        should_stop = early_stopper.step(stop_score, model)
        if should_stop:
            break

    early_stopper.restore(model)
    return model


# ============================================================
# Evaluación por especie
# ============================================================

def make_splits(lps_strings: np.ndarray, n_splits: int, test_size: float, random_seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_seed
    )
    dummy_X = np.zeros((len(lps_strings), 1), dtype=np.float32)

    splits = []
    for train_idx, test_idx in splitter.split(dummy_X, lps_strings):
        splits.append((train_idx, test_idx))
    return splits


def encode_lps_labels(train_lps: np.ndarray, test_lps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    unique_train = sorted(np.unique(train_lps).tolist())
    lps_to_idx = {k: i for i, k in enumerate(unique_train)}
    idx_to_lps = {i: k for k, i in lps_to_idx.items()}

    y_train = np.array([lps_to_idx[x] for x in train_lps], dtype=np.int64)
    # asumimos que por estratificación los test patterns también están en train
    y_test = np.array([lps_to_idx[x] for x in test_lps], dtype=np.int64)
    return y_train, y_test, lps_to_idx, idx_to_lps


def evaluate_species(
    dataset: PreparedSpeciesDataset,
    benchmark_paths: SpeciesBenchmarkPaths,
    train_config: TrainConfig,
    device: torch.device,
    species_output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Devuelve:
    - fold_results_df
    - species_summary_dict
    """
    species = dataset.species
    X = dataset.X
    Y = dataset.Y
    lps_strings = dataset.lps_strings
    antibiotics = dataset.selected_antibiotics

    species_output_dir.mkdir(parents=True, exist_ok=True)

    # cargar hiperparámetros
    lps_params = json_load(Path(benchmark_paths.lps_params_json))
    multilabel_params = json_load(Path(benchmark_paths.multilabel_params_json))

    binary_params: Dict[str, Dict[str, Any]] = {}
    for ab in antibiotics:
        if ab not in benchmark_paths.binary_param_files:
            logging.warning("[%s] No hay best_params binario para antibiótico %s. Se omitirá.", species, ab)
            continue
        binary_params[ab] = json_load(Path(benchmark_paths.binary_param_files[ab]))

    if not binary_params:
        logging.warning("[%s] No hay modelos binarios válidos. Binary AUC quedará NaN.", species)

    splits = make_splits(
        lps_strings=lps_strings,
        n_splits=train_config.n_splits,
        test_size=train_config.test_size,
        random_seed=train_config.random_seed
    )

    fold_records: List[Dict[str, Any]] = []
    binary_antibiotic_records: List[Dict[str, Any]] = []

    for fold_id, (train_idx, test_idx) in enumerate(splits):
        logging.info("[%s] Fold %d/%d", species, fold_id + 1, len(splits))

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        lps_train, lps_test = lps_strings[train_idx], lps_strings[test_idx]

        # --------------------------------------------------------
        # Multilabel
        # --------------------------------------------------------
        ml_model = train_single_model(
            X_train=X_train,
            y_train=Y_train,
            X_val=X_test,
            y_val=Y_test,
            output_dim=Y_train.shape[1],
            params=multilabel_params,
            train_config=train_config,
            device=device,
            task_type="multilabel",
        )

        ml_logits = predict_logits(ml_model, X_test, device=device)
        ml_prob = 1.0 / (1.0 + np.exp(-ml_logits))
        multilabel_auc_macro, multilabel_auc_per_ab = safe_macro_multilabel_auc(Y_test, ml_prob)

        # --------------------------------------------------------
        # LPS multiclass
        # --------------------------------------------------------
        y_lps_train, y_lps_test, lps_to_idx, idx_to_lps = encode_lps_labels(lps_train, lps_test)

        lps_model = train_single_model(
            X_train=X_train,
            y_train=y_lps_train,
            X_val=X_test,
            y_val=y_lps_test,
            output_dim=len(lps_to_idx),
            params=lps_params,
            train_config=train_config,
            device=device,
            task_type="multiclass_lps",
        )

        lps_logits = predict_logits(lps_model, X_test, device=device)
        lps_logits = lps_logits - lps_logits.max(axis=1, keepdims=True)
        lps_exp = np.exp(lps_logits)
        lps_prob = lps_exp / lps_exp.sum(axis=1, keepdims=True)
        
        # --------------------------------------------------------
        # LPS → convertir a probabilidades por antibiótico
        # --------------------------------------------------------

        # construir matriz de patrones binarios
        # shape: (n_classes, n_antibiotics)
        pattern_matrix = np.array([
            [int(bit) for bit in idx_to_lps[i]]
            for i in range(len(idx_to_lps))
        ], dtype=np.float32)

        # lps_prob: (N_samples, n_classes)
        # queremos: (N_samples, n_antibiotics)

        lps_ab_prob = lps_prob @ pattern_matrix

        # --------------------------------------------------------
        # calcular AUC por antibiótico
        # --------------------------------------------------------

        lps_ab_aucs = []
        for j in range(len(antibiotics)):
            auc = safe_binary_auc(Y_test[:, j], lps_ab_prob[:, j])
            lps_ab_aucs.append(auc)

        valid = [a for a in lps_ab_aucs if not np.isnan(a)]
        lps_auc_macro = float(np.mean(valid)) if valid else np.nan
        

        # --------------------------------------------------------
        # Binarios por antibiótico
        # --------------------------------------------------------
        binary_fold_aucs = []
        for ab_idx, ab in enumerate(antibiotics):
            if ab not in binary_params:
                continue

            y_train_ab = Y_train[:, ab_idx].reshape(-1, 1)
            y_test_ab = Y_test[:, ab_idx].reshape(-1)

            # Si train es constante no se puede entrenar útilmente
            if len(np.unique(y_train_ab.reshape(-1))) < 2:
                logging.warning(
                    "[%s] Fold %d | Antibiótico %s con train constante. Se omite AUC binaria.",
                    species, fold_id, ab
                )
                ab_auc = np.nan
            else:
                bin_model = train_single_model(
                    X_train=X_train,
                    y_train=y_train_ab,
                    X_val=X_test,
                    y_val=y_test_ab.reshape(-1, 1),
                    output_dim=1,
                    params=binary_params[ab],
                    train_config=train_config,
                    device=device,
                    task_type="binary",
                )

                bin_logits = predict_logits(bin_model, X_test, device=device).reshape(-1)
                bin_prob = 1.0 / (1.0 + np.exp(-bin_logits))
                ab_auc = safe_binary_auc(y_test_ab, bin_prob)

            binary_antibiotic_records.append({
                "species": species,
                "species_pretty": pretty_species_name(species),
                "fold": fold_id,
                "antibiotic": ab,
                "auc": maybe_float(ab_auc),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            })

            binary_fold_aucs.append(ab_auc)

        valid_binary_fold_aucs = [a for a in binary_fold_aucs if not np.isnan(a)]
        binary_auc_macro = float(np.mean(valid_binary_fold_aucs)) if valid_binary_fold_aucs else np.nan

        fold_records.append({
            "species": species,
            "species_pretty": pretty_species_name(species),
            "fold": fold_id,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_antibiotics": int(len(antibiotics)),
            "n_lps_classes_train": int(len(np.unique(y_lps_train))),
            "binary_auc": maybe_float(binary_auc_macro),
            "multilabel_auc": maybe_float(multilabel_auc_macro),
            "lps_auc": maybe_float(lps_auc_macro),
        })

    fold_df = pd.DataFrame(fold_records)
    binary_ab_df = pd.DataFrame(binary_antibiotic_records)

    fold_df.to_csv(species_output_dir / "fold_results.csv", index=False)
    binary_ab_df.to_csv(species_output_dir / "binary_antibiotic_fold_results.csv", index=False)

    summary = {
        "species": species,
        "species_pretty": pretty_species_name(species),
        "n_samples_final": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "n_antibiotics": int(len(antibiotics)),
        "selected_antibiotics": list(antibiotics),
        "n_unique_lps_patterns": int(len(np.unique(lps_strings))),
        "binary_auc_mean": maybe_float(fold_df["binary_auc"].mean(skipna=True)),
        "binary_auc_std": maybe_float(fold_df["binary_auc"].std(ddof=1, skipna=True)),
        "multilabel_auc_mean": maybe_float(fold_df["multilabel_auc"].mean(skipna=True)),
        "multilabel_auc_std": maybe_float(fold_df["multilabel_auc"].std(ddof=1, skipna=True)),
        "lps_auc_mean": maybe_float(fold_df["lps_auc"].mean(skipna=True)),
        "lps_auc_std": maybe_float(fold_df["lps_auc"].std(ddof=1, skipna=True)),
    }

    # mejor de los tres
    model_scores = {
        "binary": summary["binary_auc_mean"],
        "multilabel": summary["multilabel_auc_mean"],
        "lps": summary["lps_auc_mean"],
    }
    valid_scores = {k: v for k, v in model_scores.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    if valid_scores:
        best_model = max(valid_scores, key=valid_scores.get)
        best_auc = valid_scores[best_model]
    else:
        best_model = None
        best_auc = np.nan

    summary["best_ours"] = best_model
    summary["best_auc"] = maybe_float(best_auc)

    json_dump(summary, species_output_dir / "species_summary.json")
    return fold_df, summary


# ============================================================
# Tabla final
# ============================================================

def format_auc_pm(mean_val: Any, std_val: Any) -> str:
    if mean_val is None or (isinstance(mean_val, float) and np.isnan(mean_val)):
        return "NaN"
    if std_val is None or (isinstance(std_val, float) and np.isnan(std_val)):
        return f"{mean_val:.4f}"
    return f"{mean_val:.4f} ± {std_val:.4f}"


def build_final_summary_table(species_summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for s in species_summaries:
        rows.append({
            "Species": s["species_pretty"],
            "Binary AUC": s["binary_auc_mean"],
            "Binary AUC Std": s["binary_auc_std"],
            "Multilabel AUC": s["multilabel_auc_mean"],
            "Multilabel AUC Std": s["multilabel_auc_std"],
            "LPS AUC": s["lps_auc_mean"],
            "LPS AUC Std": s["lps_auc_std"],
            "Best (Ours)": s["best_ours"],
            "Best AUC": s["best_auc"],
            "N samples": s["n_samples_final"],
            "N antibiotics": s["n_antibiotics"],
            "N LPS patterns": s["n_unique_lps_patterns"],
        })

    df = pd.DataFrame(rows)

    # ordenar por mejor AUC descendente
    if not df.empty:
        df = df.sort_values(by=["Best AUC", "Species"], ascending=[False, True]).reset_index(drop=True)

    return df


def build_public_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    out["Binary AUC"] = [
        format_auc_pm(m, s) for m, s in zip(out["Binary AUC"], out["Binary AUC Std"])
    ]
    out["Multilabel AUC"] = [
        format_auc_pm(m, s) for m, s in zip(out["Multilabel AUC"], out["Multilabel AUC Std"])
    ]
    out["LPS AUC"] = [
        format_auc_pm(m, s) for m, s in zip(out["LPS AUC"], out["LPS AUC Std"])
    ]

    out = out[[
        "Species",
        "Binary AUC",
        "Multilabel AUC",
        "LPS AUC",
        "Best (Ours)",
        "Best AUC",
        "N samples",
        "N antibiotics",
        "N LPS patterns",
    ]]

    out["Best AUC"] = out["Best AUC"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
    return out


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluación comparativa de AUCs para MLP benchmark")

    parser.add_argument(
        "--pickle-path",
        type=str,
        required=True,
        help="Ruta al pickle COMBINED_MARISMA_DRIAMS.pkl"
    )
    parser.add_argument(
        "--benchmark-roots",
        nargs="+",
        required=True,
        help="Una o varias carpetas root de benchmark outputs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directorio donde guardar todos los resultados"
    )

    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0")
    parser.add_argument(
        "--species",
        nargs="*",
        default=None,
        help="Opcional: lista de especies concretas a evaluar. Si no se indica, se evalúan todas."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)

    logging.info("Iniciando evaluación de AUCs")
    logging.info("Args: %s", vars(args))

    train_config = TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        test_size=args.test_size,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
        device=args.device,
    )

    seed_everything(train_config.random_seed)
    device = choose_device(train_config.device)
    logging.info("Device seleccionado: %s", device)

    payload = load_pickle_payload(args.pickle_path)
    discovered = discover_species_benchmarks(args.benchmark_roots)

    if args.species:
        requested = set(args.species)
        discovered = {k: v for k, v in discovered.items() if k in requested}
        missing = requested - set(discovered.keys())
        if missing:
            logging.warning("Especies solicitadas no encontradas en benchmark roots: %s", sorted(missing))

    if not discovered:
        raise RuntimeError("No quedan especies a evaluar tras aplicar el filtro --species.")

    logging.info("Especies a evaluar (%d): %s", len(discovered), sorted(discovered.keys()))

    all_fold_dfs = []
    all_species_summaries = []

    for idx, species in enumerate(sorted(discovered.keys()), start=1):
        logging.info("=" * 80)
        logging.info("Procesando especie %d/%d: %s", idx, len(discovered), species)

        bench_paths = discovered[species]
        species_cfg = load_species_data_config(bench_paths)
        dataset = prepare_species_dataset(payload, species_cfg)

        species_output_dir = output_dir / species

        t0 = time.time()
        try:
            fold_df, species_summary = evaluate_species(
                dataset=dataset,
                benchmark_paths=bench_paths,
                train_config=train_config,
                device=device,
                species_output_dir=species_output_dir,
            )
            elapsed = time.time() - t0
            species_summary["elapsed_seconds"] = elapsed
            json_dump(species_summary, species_output_dir / "species_summary.json")

            all_fold_dfs.append(fold_df)
            all_species_summaries.append(species_summary)

            logging.info(
                "[%s] Completado en %.2f min | Binary=%.4f | Multilabel=%.4f | LPS=%.4f | Best=%s",
                species,
                elapsed / 60.0,
                species_summary["binary_auc_mean"] if species_summary["binary_auc_mean"] is not None else np.nan,
                species_summary["multilabel_auc_mean"] if species_summary["multilabel_auc_mean"] is not None else np.nan,
                species_summary["lps_auc_mean"] if species_summary["lps_auc_mean"] is not None else np.nan,
                species_summary["best_ours"],
            )
        except Exception as e:
            logging.exception("Error procesando especie %s: %s", species, e)

    # Guardados globales
    if all_fold_dfs:
        global_fold_df = pd.concat(all_fold_dfs, ignore_index=True)
    else:
        global_fold_df = pd.DataFrame()

    global_fold_df.to_csv(output_dir / "all_species_fold_results.csv", index=False)

    summary_df = build_final_summary_table(all_species_summaries)
    summary_df.to_csv(output_dir / "summary_numeric.csv", index=False)

    public_df = build_public_table(summary_df)
    public_df.to_csv(output_dir / "summary_table.csv", index=False)
    save_markdown_table(public_df, output_dir / "summary_table.md")

    # JSON global
    json_dump(
        {
            "train_config": asdict(train_config),
            "device": str(device),
            "n_species_ok": int(len(all_species_summaries)),
            "species": all_species_summaries,
        },
        output_dir / "global_summary.json"
    )

    logging.info("=" * 80)
    logging.info("Proceso completado.")
    logging.info("Resultados en: %s", output_dir)
    logging.info("Tabla final CSV: %s", output_dir / "summary_table.csv")
    logging.info("Tabla final Markdown: %s", output_dir / "summary_table.md")


if __name__ == "__main__":
    main()