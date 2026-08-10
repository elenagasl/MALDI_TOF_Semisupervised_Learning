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
    name = str(species_name).strip().replace(" ", "_")
    name = re.sub(r"_+", "_", name).strip("_").lower()
    enterobacter_aliases = {
        "enterobacter_cloacae",
        "enterobacter_cloacae_complex",
        "enterobacter_hormaechei",
        "enterobacter_asburiae",
        "enterobacter_kobei",
        "enterobacter_ludwigii",
        "enterobacter_roggenkampii",
    }
    if name in enterobacter_aliases:
        return "enterobacter_cloacae_complex"
    return name


def load_pickle(path: str | Path) -> PicklePayload:
    with Path(path).expanduser().open("rb") as handle:
        payload = pickle.load(handle)

    required = {"data", "label", "amr", "antibiotics"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(
            f"Missing keys in {path}: {sorted(missing)}. "
            "Expected keys: data, label, amr, antibiotics."
        )

    X = np.asarray(payload["data"], dtype=np.float32)
    species = np.asarray([canonical_species_key(x) for x in np.asarray(payload["label"])], dtype=object)
    amr = np.asarray(payload["amr"], dtype=np.float32)
    antibiotics = [str(a) for a in list(payload["antibiotics"])]

    if X.ndim != 2 or amr.ndim != 2:
        raise ValueError(f"Expected 2D X and amr, got X={X.shape}, amr={amr.shape}")
    if X.shape[0] != species.shape[0] or X.shape[0] != amr.shape[0]:
        raise ValueError("Inconsistent sample counts between data, label and amr")
    if amr.shape[1] != len(antibiotics):
        raise ValueError("Number of AMR columns does not match antibiotics list")

    return PicklePayload(X=X, species=species, amr=amr, antibiotics=antibiotics)


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

    aligned_ind = PicklePayload(
        X=ind.X[ind_mask],
        species=ind.species[ind_mask].astype(str),
        amr=ind.amr[np.ix_(np.where(ind_mask)[0], ind_ab_idx)],
        antibiotics=shared_antibiotics,
    )
    aligned_ood = PicklePayload(
        X=ood.X[ood_mask],
        species=ood.species[ood_mask].astype(str),
        amr=ood.amr[np.ix_(np.where(ood_mask)[0], ood_ab_idx)],
        antibiotics=shared_antibiotics,
    )
    if aligned_ind.X.shape[1] != aligned_ood.X.shape[1]:
        raise ValueError(
            "Input dimensions differ after alignment: "
            f"in={aligned_ind.X.shape[1]}, ood={aligned_ood.X.shape[1]}"
        )
    return aligned_ind, aligned_ood

def filter_excluded_species(payload: PicklePayload, excluded_species: tuple[str, ...]) -> PicklePayload:
    if not excluded_species:
        return payload
    excluded = {canonical_species_key(species_name) for species_name in excluded_species}
    species = payload.species.astype(str)
    keep_mask = np.asarray([canonical_species_key(name) not in excluded for name in species], dtype=bool)
    if keep_mask.all():
        return payload
    return PicklePayload(
        X=payload.X[keep_mask],
        species=payload.species[keep_mask],
        amr=payload.amr[keep_mask],
        antibiotics=payload.antibiotics,
    )
