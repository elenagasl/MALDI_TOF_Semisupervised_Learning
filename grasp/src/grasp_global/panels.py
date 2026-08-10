from __future__ import annotations

import numpy as np


def select_sparse_panel(
    amr: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    min_train_samples: int,
    min_val_samples: int,
) -> list[int]:
    observed = ~np.isnan(amr)
    train_counts = observed[train_idx].sum(axis=0)
    val_counts = observed[val_idx].sum(axis=0)
    valid = np.where((train_counts >= min_train_samples) & (val_counts >= min_val_samples))[0]
    return [int(i) for i in valid]

