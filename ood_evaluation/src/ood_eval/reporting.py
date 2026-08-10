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


def _blocks(output_dir: Path) -> list[dict[str, object]]:
    prediction_dir = output_dir / "predictions"
    blocks: list[dict[str, object]] = []
    for path in sorted(prediction_dir.glob("*/*.npz")):
        data = np.load(path, allow_pickle=True)
        blocks.append(
            {
                "model": path.parent.name,
                "scenario": str(data["scenario"].item()),
                "species": str(data["species_name"].item()),
                "fold": int(data["fold"].item()),
                "antibiotic_indices": data["antibiotic_indices"],
                "y_true": data["y_true"],
                "y_score": data["y_score"],
            }
        )
    return blocks


def _pooled(blocks: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    yt: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for block in blocks:
        y_true = np.asarray(block["y_true"])
        y_score = np.asarray(block["y_score"])
        mask = ~np.isnan(y_true) & ~np.isnan(y_score)
        yt.append(y_true[mask])
        ys.append(y_score[mask])
    if not yt:
        return np.array([]), np.array([])
    return np.concatenate(yt), np.concatenate(ys)


def _antibiotic_values(blocks: list[dict[str, object]], antibiotic_idx: int) -> tuple[np.ndarray, np.ndarray]:
    yt: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for block in blocks:
        antibiotic_indices = list(np.asarray(block["antibiotic_indices"]).astype(int))
        if antibiotic_idx not in antibiotic_indices:
            continue
        local_col = antibiotic_indices.index(antibiotic_idx)
        y_true = np.asarray(block["y_true"])[:, local_col]
        y_score = np.asarray(block["y_score"])[:, local_col]
        mask = ~np.isnan(y_true) & ~np.isnan(y_score)
        yt.append(y_true[mask])
        ys.append(y_score[mask])
    if not yt:
        return np.array([]), np.array([])
    return np.concatenate(yt), np.concatenate(ys)


def write_reports(output_dir: Path, antibiotics: list[str]) -> None:
    blocks = _blocks(output_dir)
    if not blocks:
        return

    global_rows: list[dict[str, object]] = []
    species_rows: list[dict[str, object]] = []
    task_rows_by_model: dict[str, list[dict[str, object]]] = {}

    keys = sorted({(str(b["scenario"]), str(b["model"])) for b in blocks})
    for scenario, model in keys:
        model_blocks = [b for b in blocks if b["scenario"] == scenario and b["model"] == model]
        patient_values: list[float] = []
        species_micro_values: list[float] = []
        task_auc_values: list[float] = []

        for species in sorted({str(b["species"]) for b in model_blocks}):
            species_blocks = [b for b in model_blocks if b["species"] == species]
            species_patient_values: list[float] = []
            for block in species_blocks:
                species_patient_values.extend(_patient_values(np.asarray(block["y_true"]), np.asarray(block["y_score"])))
            y_true_micro, y_score_micro = _pooled(species_blocks)
            species_micro = safe_auc(y_true_micro, y_score_micro)

            species_task_values: list[float] = []
            antibiotic_indices = sorted(
                {
                    int(a)
                    for block in species_blocks
                    for a in np.asarray(block["antibiotic_indices"]).astype(int)
                }
            )
            for antibiotic_idx in antibiotic_indices:
                y_true_ab, y_score_ab = _antibiotic_values(species_blocks, antibiotic_idx)
                auc = safe_auc(y_true_ab, y_score_ab)
                task_rows_by_model.setdefault(f"{scenario}__{model}", []).append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "species": species,
                        "antibiotic_idx": antibiotic_idx,
                        "antibiotic": antibiotics[antibiotic_idx],
                        "auc": auc,
                        "n_observed": int(y_true_ab.size),
                    }
                )
                if not np.isnan(auc):
                    species_task_values.append(auc)
                    task_auc_values.append(auc)

            if not np.isnan(species_micro):
                species_micro_values.append(species_micro)
            patient_values.extend(species_patient_values)
            species_rows.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "species": species,
                    "patient_auc_by_sample": float(np.mean(species_patient_values)) if species_patient_values else float("nan"),
                    "n_patient_auc_samples": len(species_patient_values),
                    "micro_auc_by_species": species_micro,
                    "macro_auc_by_species_antibiotic": float(np.mean(species_task_values)) if species_task_values else float("nan"),
                    "n_species_antibiotic_aucs": len(species_task_values),
                }
            )

        global_rows.append(
            {
                "scenario": scenario,
                "model": model,
                "patient_auc_by_sample": float(np.mean(patient_values)) if patient_values else float("nan"),
                "n_patient_auc_samples": len(patient_values),
                "micro_auc_by_species": float(np.mean(species_micro_values)) if species_micro_values else float("nan"),
                "n_species_micro_aucs": len(species_micro_values),
                "macro_auc_by_species_antibiotic": float(np.mean(task_auc_values)) if task_auc_values else float("nan"),
                "n_species_antibiotic_aucs": len(task_auc_values),
            }
        )

    pd.DataFrame(global_rows).to_csv(output_dir / "global_metrics_by_model.csv", index=False)
    pd.DataFrame(species_rows).to_csv(output_dir / "species_metrics_by_model.csv", index=False)
    task_dir = output_dir / "auc_by_species_antibiotic"
    task_dir.mkdir(exist_ok=True)
    for key, rows in task_rows_by_model.items():
        pd.DataFrame(rows).to_csv(task_dir / f"{key}.csv", index=False)

