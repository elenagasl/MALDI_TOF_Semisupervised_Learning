from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class OODFoldSplit:
    fold: int
    adapt_train_idx: np.ndarray
    adapt_val_idx: np.ndarray
    test_idx: np.ndarray


def make_species_folds(indices: np.ndarray, n_folds: int, val_size: float, random_seed: int) -> list[FoldSplit]:
    indices = np.asarray(indices)
    if indices.size < n_folds:
        return []
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    folds: list[FoldSplit] = []
    for fold, (train_val_pos, test_pos) in enumerate(splitter.split(indices), start=1):
        train_val_idx = indices[train_val_pos]
        test_idx = indices[test_pos]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            random_state=random_seed + fold,
            shuffle=True,
        )
        folds.append(FoldSplit(fold, np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)))
    return folds


def make_global_folds(species: np.ndarray, n_folds: int, val_size: float, random_seed: int) -> list[FoldSplit]:
    species_array = species.astype(str)
    parts_by_fold: dict[int, dict[str, list[np.ndarray]]] = {
        fold: {"train": [], "val": [], "test": []} for fold in range(1, n_folds + 1)
    }
    for species_name in sorted(np.unique(species_array)):
        species_idx = np.where(species_array == species_name)[0]
        for split in make_species_folds(species_idx, n_folds, val_size, random_seed):
            parts_by_fold[split.fold]["train"].append(split.train_idx)
            parts_by_fold[split.fold]["val"].append(split.val_idx)
            parts_by_fold[split.fold]["test"].append(split.test_idx)

    folds: list[FoldSplit] = []
    for fold in range(1, n_folds + 1):
        parts = parts_by_fold[fold]
        if not parts["train"] or not parts["val"] or not parts["test"]:
            continue
        folds.append(
            FoldSplit(
                fold=fold,
                train_idx=np.concatenate(parts["train"]),
                val_idx=np.concatenate(parts["val"]),
                test_idx=np.concatenate(parts["test"]),
            )
        )
    return folds


def _split_ood_species(
    local_indices: np.ndarray,
    amr: np.ndarray,
    adaptation_fraction: float,
    finetune_val_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if local_indices.size < 5:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), local_indices.astype(np.int64)

    amr_sp = amr[local_indices]
    observed_counts = np.sum(np.isfinite(amr_sp), axis=1)
    resistance_mean = np.nanmean(amr_sp, axis=1)
    resistance_mean = np.nan_to_num(resistance_mean, nan=-1.0)
    try:
        obs_bin = pd.qcut(observed_counts, q=min(4, len(np.unique(observed_counts))), labels=False, duplicates="drop")
    except Exception:
        obs_bin = np.zeros(local_indices.size, dtype=int)
    try:
        res_bin = pd.qcut(resistance_mean, q=min(4, len(np.unique(resistance_mean))), labels=False, duplicates="drop")
        strat = np.asarray([f"{a}_{b}" for a, b in zip(obs_bin, res_bin)], dtype=object)
    except Exception:
        strat = np.asarray(obs_bin).astype(str)

    try:
        adapt_pos, test_pos = train_test_split(
            np.arange(local_indices.size),
            train_size=adaptation_fraction,
            random_state=random_seed,
            shuffle=True,
            stratify=strat,
        )
    except Exception:
        adapt_pos, test_pos = train_test_split(
            np.arange(local_indices.size),
            train_size=adaptation_fraction,
            random_state=random_seed,
            shuffle=True,
        )

    adapt_idx = local_indices[adapt_pos]
    test_idx = local_indices[test_pos]
    if adapt_idx.size < 2:
        return adapt_idx.astype(np.int64), adapt_idx.astype(np.int64), test_idx.astype(np.int64)
    try:
        ft_train, ft_val = train_test_split(
            adapt_idx,
            test_size=finetune_val_size,
            random_state=random_seed + 101,
            shuffle=True,
        )
    except Exception:
        ft_train, ft_val = adapt_idx, adapt_idx
    return np.asarray(ft_train, dtype=np.int64), np.asarray(ft_val, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def make_ood_folds(
    species: np.ndarray,
    amr: np.ndarray,
    n_folds: int,
    adaptation_fraction: float,
    finetune_val_size: float,
    random_seed: int,
) -> list[OODFoldSplit]:
    species_array = species.astype(str)
    folds: list[OODFoldSplit] = []
    for fold in range(1, n_folds + 1):
        adapt_train_parts: list[np.ndarray] = []
        adapt_val_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        for species_name in sorted(np.unique(species_array)):
            species_idx = np.where(species_array == species_name)[0]
            train_idx, val_idx, test_idx = _split_ood_species(
                species_idx,
                amr,
                adaptation_fraction,
                finetune_val_size,
                random_seed + fold + abs(hash(species_name)) % 1000,
            )
            if train_idx.size:
                adapt_train_parts.append(train_idx)
            if val_idx.size:
                adapt_val_parts.append(val_idx)
            if test_idx.size:
                test_parts.append(test_idx)
        folds.append(
            OODFoldSplit(
                fold=fold,
                adapt_train_idx=np.concatenate(adapt_train_parts) if adapt_train_parts else np.array([], dtype=np.int64),
                adapt_val_idx=np.concatenate(adapt_val_parts) if adapt_val_parts else np.array([], dtype=np.int64),
                test_idx=np.concatenate(test_parts) if test_parts else np.array([], dtype=np.int64),
            )
        )
    return folds

