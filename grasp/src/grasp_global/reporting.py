from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import safe_auc


def _patient_auc_values(y_true: np.ndarray, y_score: np.ndarray) -> list[float]:
    values: list[float] = []
    for row in range(y_true.shape[0]):
        auc = safe_auc(y_true[row], y_score[row])
        if not np.isnan(auc):
            values.append(auc)
    return values


def _load_prediction_block(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    return {
        "path": path,
        "model": path.parent.name,
        "species": str(data["species_name"].item()),
        "fold": int(data["fold"].item()),
        "test_idx": data["test_idx"],
        "antibiotic_indices": data["antibiotic_indices"],
        "y_true": data["y_true"],
        "y_score": data["y_score"],
    }


def _prediction_blocks(output_dir: Path) -> list[dict[str, object]]:
    prediction_dir = output_dir / "predictions"
    if not prediction_dir.exists():
        return []
    return [_load_prediction_block(path) for path in sorted(prediction_dir.glob("*/*.npz"))]


def _pooled_observed_values(blocks: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    y_true_values: list[np.ndarray] = []
    y_score_values: list[np.ndarray] = []
    for block in blocks:
        y_true = np.asarray(block["y_true"])
        y_score = np.asarray(block["y_score"])
        mask = ~np.isnan(y_true) & ~np.isnan(y_score)
        y_true_values.append(y_true[mask])
        y_score_values.append(y_score[mask])
    if not y_true_values:
        return np.array([]), np.array([])
    return np.concatenate(y_true_values), np.concatenate(y_score_values)


def _antibiotic_values(
    blocks: list[dict[str, object]],
    antibiotic_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_values: list[np.ndarray] = []
    y_score_values: list[np.ndarray] = []
    for block in blocks:
        antibiotic_indices = list(np.asarray(block["antibiotic_indices"]).astype(int))
        if antibiotic_idx not in antibiotic_indices:
            continue
        local_col = antibiotic_indices.index(antibiotic_idx)
        y_true_col = np.asarray(block["y_true"])[:, local_col]
        y_score_col = np.asarray(block["y_score"])[:, local_col]
        mask = ~np.isnan(y_true_col) & ~np.isnan(y_score_col)
        y_true_values.append(y_true_col[mask])
        y_score_values.append(y_score_col[mask])
    if not y_true_values:
        return np.array([]), np.array([])
    return np.concatenate(y_true_values), np.concatenate(y_score_values)


def write_final_reports(output_dir: Path, antibiotics: list[str]) -> None:
    blocks = _prediction_blocks(output_dir)
    if not blocks:
        return

    global_rows: list[dict[str, object]] = []
    species_rows: list[dict[str, object]] = []
    antibiotic_rows_by_model: dict[str, list[dict[str, object]]] = {}

    for model in sorted({str(block["model"]) for block in blocks}):
        model_blocks = [block for block in blocks if block["model"] == model]
        patient_values: list[float] = []
        species_micro_values: list[float] = []
        species_antibiotic_auc_values: list[float] = []

        for species in sorted({str(block["species"]) for block in model_blocks}):
            species_blocks = [block for block in model_blocks if block["species"] == species]
            species_patient_values: list[float] = []
            for block in species_blocks:
                species_patient_values.extend(
                    _patient_auc_values(np.asarray(block["y_true"]), np.asarray(block["y_score"]))
                )

            y_true_micro, y_score_micro = _pooled_observed_values(species_blocks)
            species_micro_auc = safe_auc(y_true_micro, y_score_micro)

            species_antibiotic_values: list[float] = []
            antibiotic_indices = sorted(
                {
                    int(antibiotic_idx)
                    for block in species_blocks
                    for antibiotic_idx in np.asarray(block["antibiotic_indices"]).astype(int)
                }
            )
            for antibiotic_idx in antibiotic_indices:
                y_true_col, y_score_col = _antibiotic_values(species_blocks, antibiotic_idx)
                auc = safe_auc(y_true_col, y_score_col)
                row = {
                    "model": model,
                    "species": species,
                    "antibiotic_idx": antibiotic_idx,
                    "antibiotic": antibiotics[antibiotic_idx],
                    "auc": auc,
                    "n_observed": int(y_true_col.size),
                }
                antibiotic_rows_by_model.setdefault(model, []).append(row)
                if not np.isnan(auc):
                    species_antibiotic_values.append(auc)
                    species_antibiotic_auc_values.append(auc)

            if not np.isnan(species_micro_auc):
                species_micro_values.append(species_micro_auc)
            patient_values.extend(species_patient_values)
            species_rows.append(
                {
                    "model": model,
                    "species": species,
                    "patient_auc_by_sample": float(np.mean(species_patient_values))
                    if species_patient_values
                    else float("nan"),
                    "n_patient_auc_samples": len(species_patient_values),
                    "micro_auc_by_species": species_micro_auc,
                    "macro_auc_by_species_antibiotic": float(np.mean(species_antibiotic_values))
                    if species_antibiotic_values
                    else float("nan"),
                    "n_species_antibiotic_aucs": len(species_antibiotic_values),
                }
            )

        global_rows.append(
            {
                "model": model,
                "patient_auc_by_sample": float(np.mean(patient_values)) if patient_values else float("nan"),
                "n_patient_auc_samples": len(patient_values),
                "micro_auc_by_species": float(np.mean(species_micro_values))
                if species_micro_values
                else float("nan"),
                "n_species_micro_aucs": len(species_micro_values),
                "macro_auc_by_species_antibiotic": float(np.mean(species_antibiotic_auc_values))
                if species_antibiotic_auc_values
                else float("nan"),
                "n_species_antibiotic_aucs": len(species_antibiotic_auc_values),
            }
        )

    pd.DataFrame(global_rows).to_csv(output_dir / "global_metrics_by_model.csv", index=False)
    pd.DataFrame(species_rows).to_csv(output_dir / "species_metrics_by_model.csv", index=False)
    antibiotic_dir = output_dir / "auc_by_species_antibiotic"
    antibiotic_dir.mkdir(exist_ok=True)
    for model, rows in antibiotic_rows_by_model.items():
        pd.DataFrame(rows).to_csv(antibiotic_dir / f"{model}.csv", index=False)

