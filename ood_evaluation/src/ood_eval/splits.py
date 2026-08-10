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


def make_global_source_folds(species: np.ndarray, n_folds: int, val_size: float, random_seed: int) -> list[FoldSplit]:
    species_array = species.astype(str)
    acc = {fold: {"train": [], "val": [], "test": []} for fold in range(1, n_folds + 1)}
    for species_name in sorted(np.unique(species_array)):
        species_idx = np.where(species_array == species_name)[0]
        for split in make_species_folds(species_idx, n_folds, val_size, random_seed):
            acc[split.fold]["train"].append(split.train_idx)
            acc[split.fold]["val"].append(split.val_idx)
            acc[split.fold]["test"].append(split.test_idx)

    folds: list[FoldSplit] = []
    for fold in range(1, n_folds + 1):
        parts = acc[fold]
        folds.append(
            FoldSplit(
                fold=fold,
                train_idx=np.concatenate(parts["train"]),
                val_idx=np.concatenate(parts["val"]),
                test_idx=np.concatenate(parts["test"]),
            )
        )
    return folds


def make_ood_folds(
    species: np.ndarray,
    n_folds: int,
    adaptation_fraction: float,
    finetune_val_size: float,
    random_seed: int,
) -> list[OODFoldSplit]:
    species_array = species.astype(str)
    acc = {fold: {"adapt_train": [], "adapt_val": [], "test": []} for fold in range(1, n_folds + 1)}
    for fold in range(1, n_folds + 1):
        for species_name in sorted(np.unique(species_array)):
            species_idx = np.where(species_array == species_name)[0]
            if species_idx.size < 5:
                continue
            adapt_idx, test_idx = train_test_split(
                species_idx,
                train_size=adaptation_fraction,
                random_state=random_seed + fold,
                shuffle=True,
            )
            if adapt_idx.size >= 2:
                adapt_train, adapt_val = train_test_split(
                    adapt_idx,
                    test_size=finetune_val_size,
                    random_state=random_seed + 1000 + fold,
                    shuffle=True,
                )
            else:
                adapt_train, adapt_val = adapt_idx, adapt_idx
            acc[fold]["adapt_train"].append(np.asarray(adapt_train))
            acc[fold]["adapt_val"].append(np.asarray(adapt_val))
            acc[fold]["test"].append(np.asarray(test_idx))

    folds: list[OODFoldSplit] = []
    for fold in range(1, n_folds + 1):
        parts = acc[fold]
        folds.append(
            OODFoldSplit(
                fold=fold,
                adapt_train_idx=np.concatenate(parts["adapt_train"]),
                adapt_val_idx=np.concatenate(parts["adapt_val"]),
                test_idx=np.concatenate(parts["test"]),
            )
        )
    return folds

