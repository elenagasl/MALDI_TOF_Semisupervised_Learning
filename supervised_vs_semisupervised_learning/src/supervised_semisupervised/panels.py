from __future__ import annotations

import numpy as np


def observed_mask(amr: np.ndarray) -> np.ndarray:
    return ~np.isnan(amr)


def select_complete_profile_panel(
    amr: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    min_train_samples: int,
    min_val_samples: int,
) -> list[int]:
    """Greedily maximize antibiotics with complete AST profiles.

    Antibiotics are ordered by observed counts across train+val. Each antibiotic is
    accepted if the complete-profile sample counts remain above the requested
    train and validation thresholds.
    """
    obs = observed_mask(amr)
    availability = obs[np.concatenate([train_idx, val_idx])].sum(axis=0)
    candidates = list(np.argsort(-availability))
    selected: list[int] = []

    for antibiotic_idx in candidates:
        proposed = selected + [int(antibiotic_idx)]
        train_complete = obs[np.ix_(train_idx, proposed)].all(axis=1).sum()
        val_complete = obs[np.ix_(val_idx, proposed)].all(axis=1).sum()
        if train_complete >= min_train_samples and val_complete >= min_val_samples:
            selected = proposed

    return selected


def select_sparse_panel(
    amr: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    min_train_samples: int,
    min_val_samples: int,
) -> list[int]:
    """Select antibiotics with enough observed labels in train and validation."""
    obs = observed_mask(amr)
    train_counts = obs[train_idx].sum(axis=0)
    val_counts = obs[val_idx].sum(axis=0)
    valid = np.where((train_counts >= min_train_samples) & (val_counts >= min_val_samples))[0]
    return [int(i) for i in valid]


def complete_profile_indices(amr: np.ndarray, sample_idx: np.ndarray, panel: list[int]) -> np.ndarray:
    if not panel:
        return np.array([], dtype=int)
    mask = observed_mask(amr[np.ix_(sample_idx, panel)]).all(axis=1)
    return np.asarray(sample_idx)[mask]


def labeled_indices_for_antibiotic(
    amr: np.ndarray,
    sample_idx: np.ndarray,
    antibiotic_idx: int,
) -> np.ndarray:
    mask = observed_mask(amr[sample_idx, antibiotic_idx])
    return np.asarray(sample_idx)[mask]

