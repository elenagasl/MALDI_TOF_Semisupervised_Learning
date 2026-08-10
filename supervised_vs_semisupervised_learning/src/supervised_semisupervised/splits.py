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
    """Create common train/val/test splits for all models within one species."""
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
        folds.append(
            FoldSplit(
                fold=fold,
                train_idx=np.asarray(train_idx),
                val_idx=np.asarray(val_idx),
                test_idx=np.asarray(test_idx),
            )
        )
    return folds


def iter_species_indices(species: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return sorted species names and corresponding sample indices."""
    values = sorted(str(s) for s in np.unique(species))
    return [(value, np.where(species.astype(str) == value)[0]) for value in values]

