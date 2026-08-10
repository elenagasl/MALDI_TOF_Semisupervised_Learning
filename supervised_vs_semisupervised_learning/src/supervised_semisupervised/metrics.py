from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    mask = ~np.isnan(y_true) & ~np.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def micro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return safe_auc(y_true.reshape(-1), y_score.reshape(-1))


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    aucs: list[float] = []
    for col in range(y_true.shape[1]):
        auc = safe_auc(y_true[:, col], y_score[:, col])
        if not np.isnan(auc):
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan")


def patient_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    aucs: list[float] = []
    for row in range(y_true.shape[0]):
        auc = safe_auc(y_true[row], y_score[row])
        if not np.isnan(auc):
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan")

