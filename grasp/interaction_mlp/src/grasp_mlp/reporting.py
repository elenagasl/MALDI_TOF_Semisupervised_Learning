from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import safe_auc


def _patient_values(y_true: np.ndarray, y_score: np.ndarray) -> list[float]:
    values: list[float] = []
    for row in range(y_true.shape[0]):
        auc = safe_auc(y_true[row], y_score[row])
        if not np.isnan(auc):
            values.append(auc)
    return values


def _load_blocks(output_dir: Path) -> list[dict[str, object]]:
    prediction_dir = output_dir / "predictions"
    blocks: list[dict[str, object]] = []
    if not prediction_dir.exists():
        return blocks
    for path in sorted(prediction_dir.glob("*/*.npz")):
        data = np.load(path, allow_pickle=True)
        blocks.append(
            {
                "model": path.parent.name,
                "scenario": str(data["scenario"].item()) if "scenario" in data.files else "",
                "species": str(data["species_name"].item()),
                "fold": int(data["fold"].item()),
                "antibiotic_indices": data["antibiotic_indices"],
                "y_true": data["y_true"],
                "y_score": data["y_score"],
            }
        )
    return blocks


def _pooled(blocks: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    y_true_values: list[np.ndarray] = []
    y_score_values: list[np.ndarray] = []
    for block in blocks:
        y_true = np.asarray(block["y_true"])
        y_score = np.asarray(block["y_score"])
        mask = np.isfinite(y_true) & np.isfinite(y_score)
        y_true_values.append(y_true[mask])
        y_score_values.append(y_score[mask])
    if not y_true_values:
        return np.array([]), np.array([])
    return np.concatenate(y_true_values), np.concatenate(y_score_values)


def _antibiotic_values(blocks: list[dict[str, object]], antibiotic_idx: int) -> tuple[np.ndarray, np.ndarray]:
    y_true_values: list[np.ndarray] = []
    y_score_values: list[np.ndarray] = []
    for block in blocks:
        antibiotic_indices = list(np.asarray(block["antibiotic_indices"]).astype(int))
        if antibiotic_idx not in antibiotic_indices:
            continue
        local_col = antibiotic_indices.index(antibiotic_idx)
        y_true = np.asarray(block["y_true"])[:, local_col]
        y_score = np.asarray(block["y_score"])[:, local_col]
        mask = np.isfinite(y_true) & np.isfinite(y_score)
        y_true_values.append(y_true[mask])
        y_score_values.append(y_score[mask])
    if not y_true_values:
        return np.array([]), np.array([])
    return np.concatenate(y_true_values), np.concatenate(y_score_values)


def write_reports(output_dir: Path, antibiotics: list[str]) -> None:
    blocks = _load_blocks(output_dir)
    if not blocks:
        return

    has_scenario = any(str(block["scenario"]) for block in blocks)
    global_rows: list[dict[str, object]] = []
    species_rows: list[dict[str, object]] = []
    task_rows_by_key: dict[str, list[dict[str, object]]] = {}

    if has_scenario:
        keys = sorted({(str(block["scenario"]), str(block["model"])) for block in blocks})
    else:
        keys = [("", model) for model in sorted({str(block["model"]) for block in blocks})]

    for scenario, model in keys:
        model_blocks = [
            block
            for block in blocks
            if str(block["model"]) == model and (not has_scenario or str(block["scenario"]) == scenario)
        ]
        patient_values: list[float] = []
        species_micro_values: list[float] = []
        task_auc_values: list[float] = []

        for species in sorted({str(block["species"]) for block in model_blocks}):
            species_blocks = [block for block in model_blocks if str(block["species"]) == species]
            species_patient_values: list[float] = []
            for block in species_blocks:
                species_patient_values.extend(_patient_values(np.asarray(block["y_true"]), np.asarray(block["y_score"])))

            y_true_micro, y_score_micro = _pooled(species_blocks)
            species_micro_auc = safe_auc(y_true_micro, y_score_micro)

            species_task_values: list[float] = []
            antibiotic_indices = sorted(
                {
                    int(antibiotic_idx)
                    for block in species_blocks
                    for antibiotic_idx in np.asarray(block["antibiotic_indices"]).astype(int)
                }
            )
            for antibiotic_idx in antibiotic_indices:
                y_true_ab, y_score_ab = _antibiotic_values(species_blocks, antibiotic_idx)
                auc = safe_auc(y_true_ab, y_score_ab)
                row = {
                    "model": model,
                    "species": species,
                    "antibiotic_idx": antibiotic_idx,
                    "antibiotic": antibiotics[antibiotic_idx],
                    "auc": auc,
                    "n_observed": int(y_true_ab.size),
                }
                if has_scenario:
                    row = {"scenario": scenario, **row}
                task_key = f"{scenario}__{model}" if has_scenario else model
                task_rows_by_key.setdefault(task_key, []).append(row)
                if not np.isnan(auc):
                    species_task_values.append(auc)
                    task_auc_values.append(auc)

            if not np.isnan(species_micro_auc):
                species_micro_values.append(species_micro_auc)
            patient_values.extend(species_patient_values)

            species_row = {
                "model": model,
                "species": species,
                "patient_auc_by_sample": float(np.mean(species_patient_values)) if species_patient_values else float("nan"),
                "n_patient_auc_samples": len(species_patient_values),
                "micro_auc_by_species": species_micro_auc,
                "macro_auc_by_species_antibiotic": float(np.mean(species_task_values)) if species_task_values else float("nan"),
                "n_species_antibiotic_aucs": len(species_task_values),
            }
            if has_scenario:
                species_row = {"scenario": scenario, **species_row}
            species_rows.append(species_row)

        global_row = {
            "model": model,
            "patient_auc_by_sample": float(np.mean(patient_values)) if patient_values else float("nan"),
            "n_patient_auc_samples": len(patient_values),
            "micro_auc_by_species": float(np.mean(species_micro_values)) if species_micro_values else float("nan"),
            "n_species_micro_aucs": len(species_micro_values),
            "macro_auc_by_species_antibiotic": float(np.mean(task_auc_values)) if task_auc_values else float("nan"),
            "n_species_antibiotic_aucs": len(task_auc_values),
        }
        if has_scenario:
            global_row = {"scenario": scenario, **global_row}
        global_rows.append(global_row)

    pd.DataFrame(global_rows).to_csv(output_dir / "global_metrics_by_model.csv", index=False)
    pd.DataFrame(species_rows).to_csv(output_dir / "species_metrics_by_model.csv", index=False)
    task_dir = output_dir / "auc_by_species_antibiotic"
    task_dir.mkdir(exist_ok=True)
    for key, rows in task_rows_by_key.items():
        pd.DataFrame(rows).to_csv(task_dir / f"{key}.csv", index=False)

