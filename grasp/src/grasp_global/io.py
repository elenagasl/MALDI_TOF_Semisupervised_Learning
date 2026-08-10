from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PicklePayload:
    X: np.ndarray
    species: np.ndarray
    amr: np.ndarray
    antibiotics: list[str]


def load_combined_pickle(path: str | Path) -> PicklePayload:
    with Path(path).expanduser().open("rb") as handle:
        payload = pickle.load(handle)

    required = {"data", "label", "amr", "antibiotics"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"Missing keys in pickle payload: {sorted(missing)}")

    X = np.asarray(payload["data"], dtype=np.float32)
    species = np.asarray(payload["label"])
    amr = np.asarray(payload["amr"], dtype=np.float32)
    antibiotics = [str(a) for a in list(payload["antibiotics"])]

    if X.ndim != 2 or amr.ndim != 2:
        raise ValueError(f"Expected X and amr to be 2D, got X={X.shape}, amr={amr.shape}")
    if X.shape[0] != species.shape[0] or X.shape[0] != amr.shape[0]:
        raise ValueError("Inconsistent sample counts between X, species and amr")
    if amr.shape[1] != len(antibiotics):
        raise ValueError("AMR columns do not match antibiotics list length")

    return PicklePayload(X=X, species=species, amr=amr, antibiotics=antibiotics)

