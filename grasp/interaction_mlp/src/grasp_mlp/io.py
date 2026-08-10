from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PicklePayload:
    X: np.ndarray
    species: np.ndarray
    amr: np.ndarray
    antibiotics: list[str]


def canonical_species_key(species_name: object) -> str:
    name = str(species_name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def load_pickle(path: str | Path) -> PicklePayload:
    with Path(path).expanduser().open("rb") as handle:
        payload = pickle.load(handle)

    required = {"data", "label", "amr", "antibiotics"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"Missing keys in {path}: {sorted(missing)}")

    X = np.asarray(payload["data"], dtype=np.float32)
    species = np.asarray(payload["label"]).astype(str)
    amr = np.asarray(payload["amr"], dtype=np.float32)
    antibiotics = [str(a) for a in list(payload["antibiotics"])]

    if X.ndim != 2 or amr.ndim != 2:
        raise ValueError(f"Expected 2D X and AMR, got X={X.shape}, amr={amr.shape}")
    if X.shape[0] != species.shape[0] or X.shape[0] != amr.shape[0]:
        raise ValueError("Inconsistent sample counts between data, label and AMR")
    if amr.shape[1] != len(antibiotics):
        raise ValueError("AMR columns do not match antibiotics list length")

    return PicklePayload(X=X, species=species, amr=amr, antibiotics=antibiotics)


def filter_excluded_species(
    payload: PicklePayload,
    excluded_species: tuple[str, ...],
) -> tuple[PicklePayload, list[str], int]:
    if not excluded_species:
        return payload, [], 0
    excluded_keys = {canonical_species_key(name) for name in excluded_species}
    species = payload.species.astype(str)
    matched = sorted({str(name) for name in np.unique(species) if canonical_species_key(name) in excluded_keys})
    keep_mask = np.asarray([canonical_species_key(name) not in excluded_keys for name in species], dtype=bool)
    filtered = PicklePayload(
        X=payload.X[keep_mask],
        species=payload.species[keep_mask],
        amr=payload.amr[keep_mask],
        antibiotics=payload.antibiotics,
    )
    return filtered, matched, int(np.sum(~keep_mask))


def align_payloads(ind: PicklePayload, ood: PicklePayload) -> tuple[PicklePayload, PicklePayload]:
    shared_species = sorted(set(ind.species.astype(str)).intersection(set(ood.species.astype(str))))
    shared_antibiotics = [ab for ab in ind.antibiotics if ab in set(ood.antibiotics)]
    if not shared_species:
        raise ValueError("No shared species between in-distribution and OOD payloads.")
    if not shared_antibiotics:
        raise ValueError("No shared antibiotics between in-distribution and OOD payloads.")

    ind_ab_idx = [ind.antibiotics.index(ab) for ab in shared_antibiotics]
    ood_ab_idx = [ood.antibiotics.index(ab) for ab in shared_antibiotics]
    ind_mask = np.isin(ind.species.astype(str), shared_species)
    ood_mask = np.isin(ood.species.astype(str), shared_species)
    ind_rows = np.where(ind_mask)[0]
    ood_rows = np.where(ood_mask)[0]

    aligned_ind = PicklePayload(
        X=ind.X[ind_rows],
        species=ind.species[ind_rows].astype(str),
        amr=ind.amr[np.ix_(ind_rows, ind_ab_idx)],
        antibiotics=shared_antibiotics,
    )
    aligned_ood = PicklePayload(
        X=ood.X[ood_rows],
        species=ood.species[ood_rows].astype(str),
        amr=ood.amr[np.ix_(ood_rows, ood_ab_idx)],
        antibiotics=shared_antibiotics,
    )
    if aligned_ind.X.shape[1] != aligned_ood.X.shape[1]:
        raise ValueError(
            f"Input dimensions differ after alignment: in={aligned_ind.X.shape[1]}, ood={aligned_ood.X.shape[1]}"
        )
    return aligned_ind, aligned_ood

