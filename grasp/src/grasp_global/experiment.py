from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, GRASPConfig, TrainingConfig
from .datasets import GlobalInteractionDataset
from .io import PicklePayload
from .metrics import safe_auc
from .models import build_model
from .panels import select_sparse_panel
from .reporting import write_final_reports
from .splits import make_global_folds, make_species_folds
from .training import interaction_loss, predict_grasp, train_with_early_stopping


MODEL_NAMES = ("species_conditioned_grasp", "multihead_grasp", "hypernetwork_grasp")
DEFAULT_EXCLUDED_SPECIES = ("Candida albicans", "Streptococcus pneumoniae")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_species(species: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    values = sorted(str(s) for s in np.unique(species.astype(str)))
    mapping = {species_name: idx for idx, species_name in enumerate(values)}
    codes = np.asarray([mapping[str(s)] for s in species.astype(str)], dtype=np.int64)
    return codes, mapping


def canonical_species_key(species_name: object) -> str:
    name = str(species_name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


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


def patient_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    values: list[float] = []
    for row in range(y_true.shape[0]):
        auc = safe_auc(y_true[row], y_score[row])
        if not np.isnan(auc):
            values.append(auc)
    return float(np.mean(values)) if values else float("nan")


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    values: list[float] = []
    for col in range(y_true.shape[1]):
        auc = safe_auc(y_true[:, col], y_score[:, col])
        if not np.isnan(auc):
            values.append(auc)
    return float(np.mean(values)) if values else float("nan")


def save_fold_predictions(
    output_dir: Path,
    model_name: str,
    species_name: str,
    fold: int,
    test_idx: np.ndarray,
    antibiotic_indices: list[int],
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> None:
    safe_species = species_name.replace(" ", "_").replace("/", "_")
    path = output_dir / "predictions" / model_name / f"{safe_species}_fold_{fold}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        model_name=np.asarray(model_name),
        species_name=np.asarray(species_name),
        fold=np.asarray(fold),
        test_idx=test_idx,
        antibiotic_indices=np.asarray(antibiotic_indices),
        y_true=y_true,
        y_score=y_score,
    )


def run_one_model_fold(
    payload: PicklePayload,
    species_codes: np.ndarray,
    species_mapping: dict[str, int],
    model_name: str,
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    antibiotic_indices: list[int],
    evaluation_panels: dict[tuple[str, int], list[int]],
    evaluation_panel_mode: str,
    exp_cfg: ExperimentConfig,
    train_cfg: TrainingConfig,
    grasp_cfg: GRASPConfig,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    model = build_model(
        model_name=model_name,
        input_dim=payload.X.shape[1],
        n_antibiotics=len(antibiotic_indices),
        n_species=len(species_mapping),
        cfg=grasp_cfg,
    )
    model, history = train_with_early_stopping(
        model=model,
        train_dataset=GlobalInteractionDataset(payload.X, species_codes, payload.amr, train_idx, antibiotic_indices),
        val_dataset=GlobalInteractionDataset(payload.X, species_codes, payload.amr, val_idx, antibiotic_indices),
        loss_fn=interaction_loss,
        optimizer_name=grasp_cfg.optimizer,
        learning_rate=grasp_cfg.learning_rate,
        training_config=train_cfg,
        device=device,
        val_X=payload.X,
        val_species_codes=species_codes,
        val_amr=payload.amr,
        val_sample_indices=val_idx,
        val_antibiotic_indices=antibiotic_indices,
    )

    rows: list[dict[str, object]] = []
    species_array = payload.species.astype(str)
    antibiotic_to_local_col = {antibiotic_idx: local_col for local_col, antibiotic_idx in enumerate(antibiotic_indices)}
    for species_name in sorted(np.unique(species_array[test_idx])):
        species_test_idx = test_idx[species_array[test_idx] == species_name]
        full_y_true = payload.amr[np.ix_(species_test_idx, antibiotic_indices)]
        full_y_score = predict_grasp(
            model,
            payload.X,
            species_codes,
            species_test_idx,
            n_antibiotics=len(antibiotic_indices),
            device=device,
            batch_size=train_cfg.prediction_batch_size,
        )

        if evaluation_panel_mode == "species":
            species_panel = evaluation_panels.get((species_name, fold), [])
            local_cols = [
                antibiotic_to_local_col[antibiotic_idx]
                for antibiotic_idx in species_panel
                if antibiotic_idx in antibiotic_to_local_col
            ]
            eval_antibiotic_indices = [
                antibiotic_indices[local_col]
                for local_col in local_cols
            ]
            if not local_cols:
                continue
            y_true = full_y_true[:, local_cols]
            y_score = full_y_score[:, local_cols]
        elif evaluation_panel_mode == "global":
            eval_antibiotic_indices = antibiotic_indices
            y_true = full_y_true
            y_score = full_y_score
        else:
            raise ValueError(f"Unsupported evaluation_panel_mode: {evaluation_panel_mode}")

        save_fold_predictions(
            output_dir,
            model_name,
            species_name,
            fold,
            species_test_idx,
            eval_antibiotic_indices,
            y_true,
            y_score,
        )
        rows.append(
            {
                "model": model_name,
                "species": species_name,
                "fold": fold,
                "n_train_samples": len(train_idx),
                "n_val_samples": len(val_idx),
                "n_test_samples": len(species_test_idx),
                "n_antibiotics": len(eval_antibiotic_indices),
                "training_panel_n_antibiotics": len(antibiotic_indices),
                "evaluation_panel": evaluation_panel_mode,
                "best_val_loss": history["best_val_loss"],
                "best_val_patient_auc": history["best_val_patient_auc"],
                "best_epoch": history["best_epoch"],
                "early_stopping_metric": history["early_stopping_metric"],
                "optimizer": grasp_cfg.optimizer,
                "learning_rate": grasp_cfg.learning_rate,
                "dropout": grasp_cfg.dropout,
                "weight_decay": train_cfg.weight_decay,
                "patient_auc": patient_auc(y_true, y_score),
                "micro_auc": safe_auc(y_true.reshape(-1), y_score.reshape(-1)),
                "macro_auc": macro_auc(y_true, y_score),
            }
        )
    return rows


def run_experiment(
    payload: PicklePayload,
    output_dir: Path,
    exp_cfg: ExperimentConfig,
    train_cfg: TrainingConfig,
    grasp_cfg: GRASPConfig,
    device: torch.device,
    model_names: tuple[str, ...] = MODEL_NAMES,
    excluded_species: tuple[str, ...] = DEFAULT_EXCLUDED_SPECIES,
    evaluation_panel_mode: str = "species",
) -> pd.DataFrame:
    set_seed(exp_cfg.random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    original_n_samples = payload.X.shape[0]
    excluded_keys = {canonical_species_key(species_name) for species_name in excluded_species}
    original_species = payload.species.astype(str)
    matched_excluded_species = sorted(
        {
            str(species_name)
            for species_name in np.unique(original_species)
            if canonical_species_key(species_name) in excluded_keys
        }
    )
    payload = filter_excluded_species(payload, excluded_species)
    excluded_n_samples = original_n_samples - payload.X.shape[0]

    species_codes, species_mapping = encode_species(payload.species)
    species_array = payload.species.astype(str)
    evaluation_panels: dict[tuple[str, int], list[int]] = {}
    for species_name in sorted(np.unique(species_array)):
        species_idx = np.where(species_array == species_name)[0]
        for species_split in make_species_folds(
            species_idx,
            exp_cfg.n_folds,
            exp_cfg.val_size,
            exp_cfg.random_seed,
        ):
            evaluation_panels[(species_name, species_split.fold)] = select_sparse_panel(
                payload.amr,
                species_split.train_idx,
                species_split.val_idx,
                exp_cfg.min_train_samples,
                exp_cfg.min_val_samples,
            )

    with (output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "experiment": asdict(exp_cfg),
                "training": asdict(train_cfg),
                "grasp": asdict(grasp_cfg),
                "device": str(device),
                "species_mapping": species_mapping,
                "models": list(model_names),
                "excluded_species": list(excluded_species),
                "matched_excluded_species": matched_excluded_species,
                "excluded_n_samples": int(excluded_n_samples),
                "evaluation_panel_mode": evaluation_panel_mode,
                "evaluation_panel_note": "species mode matches the semi-supervised species/fold sparse panels",
            },
            handle,
            indent=2,
        )

    rows: list[dict[str, object]] = []
    folds = make_global_folds(payload.species, exp_cfg.n_folds, exp_cfg.val_size, exp_cfg.random_seed)
    for split in folds:
        antibiotic_indices = select_sparse_panel(
            payload.amr,
            split.train_idx,
            split.val_idx,
            exp_cfg.min_train_samples,
            exp_cfg.min_val_samples,
        )
        if not antibiotic_indices:
            continue

        for model_name in model_names:
            rows.extend(
                run_one_model_fold(
                    payload,
                    species_codes,
                    species_mapping,
                    model_name,
                    split.fold,
                    split.train_idx,
                    split.val_idx,
                    split.test_idx,
                    antibiotic_indices,
                    evaluation_panels,
                    evaluation_panel_mode,
                    exp_cfg,
                    train_cfg,
                    grasp_cfg,
                    device,
                    output_dir,
                )
            )
            metrics_df = pd.DataFrame(rows)
            metrics_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)

    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        summary = (
            metrics_df.groupby("model", as_index=False)[["patient_auc", "micro_auc", "macro_auc"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    write_final_reports(output_dir, payload.antibiotics)
    return metrics_df
