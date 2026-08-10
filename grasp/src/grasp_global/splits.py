from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold, train_test_split


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def make_species_folds(
    indices: np.ndarray,
    n_folds: int,
    val_size: float,
    random_seed: int,
) -> list[FoldSplit]:
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


def make_global_folds(
    species: np.ndarray,
    n_folds: int,
    val_size: float,
    random_seed: int,
) -> list[FoldSplit]:
    """Build global folds by applying the same per-species split rule as the baseline experiment."""
    species_array = species.astype(str)
    global_folds: dict[int, dict[str, list[np.ndarray]]] = {
        fold: {"train": [], "val": [], "test": []} for fold in range(1, n_folds + 1)
    }
    for species_name in sorted(np.unique(species_array)):
        species_idx = np.where(species_array == species_name)[0]
        for split in make_species_folds(species_idx, n_folds, val_size, random_seed):
            global_folds[split.fold]["train"].append(split.train_idx)
            global_folds[split.fold]["val"].append(split.val_idx)
            global_folds[split.fold]["test"].append(split.test_idx)

    folds: list[FoldSplit] = []
    for fold in range(1, n_folds + 1):
        parts = global_folds[fold]
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

