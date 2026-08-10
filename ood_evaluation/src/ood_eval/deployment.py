from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from grasp_global.datasets import GlobalInteractionDataset
from grasp_global.models import build_model
from grasp_global.training import (
    interaction_loss,
    predict_grasp,
    train_with_early_stopping as train_grasp_with_early_stopping,
)
from grasp_mlp.config import MODEL_NAME as GRASP_MLP_MODEL_NAME
from grasp_mlp.config import ModelConfig as GRASPMlpModelConfig
from grasp_mlp.config import TrainingConfig as GRASPMlpTrainingConfig
from grasp_mlp.datasets import SpeciesAwareMultiOutputDataset
from grasp_mlp.models import SpeciesAwareGlobalMLP
from grasp_mlp.training import (
    masked_multioutput_loss as grasp_mlp_loss,
    predict_species_aware_mlp,
    train_with_early_stopping as train_grasp_mlp_with_early_stopping,
)

from .config import ExperimentConfig, GRASPConfig, TrainingConfig
from .experiment import (
    ALL_MODELS,
    BASELINE_MODELS,
    GRASP_MODELS,
    encode_species,
    metric_row,
    run_binary_species,
    run_multioutput_species,
    run_species_recommender,
    save_predictions,
)
from .io import PicklePayload
from .legacy_global import (
    count_obs_and_classes,
    select_valid_antibiotics_legacy,
    split_ood_by_species,
    split_train_val,
)
from .metrics import macro_auc, patient_auc, safe_auc
from .reporting import write_reports

DEPLOYMENT_MODELS = ALL_MODELS + (GRASP_MLP_MODEL_NAME,)

def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finetune_config(cfg: TrainingConfig) -> TrainingConfig:
    return TrainingConfig(
        batch_size=cfg.batch_size,
        prediction_batch_size=cfg.prediction_batch_size,
        max_epochs=cfg.finetune_epochs,
        patience=cfg.finetune_patience,
        finetune_epochs=cfg.finetune_epochs,
        finetune_patience=cfg.finetune_patience,
        num_workers=cfg.num_workers,
        weight_decay=cfg.weight_decay,
        early_stopping_metric=cfg.early_stopping_metric,
        pin_memory=cfg.pin_memory,
    )


def _grasp_mlp_training_config(cfg: TrainingConfig) -> GRASPMlpTrainingConfig:
    return GRASPMlpTrainingConfig(
        batch_size=cfg.batch_size,
        prediction_batch_size=cfg.prediction_batch_size,
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
        finetune_epochs=cfg.finetune_epochs,
        finetune_patience=cfg.finetune_patience,
        weight_decay=cfg.weight_decay,
        early_stopping_metric=cfg.early_stopping_metric,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )


def _grasp_mlp_finetune_config(cfg: TrainingConfig) -> GRASPMlpTrainingConfig:
    return GRASPMlpTrainingConfig(
        batch_size=cfg.batch_size,
        prediction_batch_size=cfg.prediction_batch_size,
        max_epochs=cfg.finetune_epochs,
        patience=cfg.finetune_patience,
        finetune_epochs=cfg.finetune_epochs,
        finetune_patience=cfg.finetune_patience,
        weight_decay=cfg.weight_decay,
        early_stopping_metric=cfg.early_stopping_metric,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )


def _filter_common_species(
    ind: PicklePayload,
    ood: PicklePayload,
    min_source_samples_species: int,
    min_ood_samples_species: int,
) -> tuple[PicklePayload, PicklePayload, list[str]]:
    common_species = sorted(set(ind.species.astype(str)).intersection(set(ood.species.astype(str))))
    kept_species = [
        species_name
        for species_name in common_species
        if int(np.sum(ind.species.astype(str) == species_name)) >= min_source_samples_species
        and int(np.sum(ood.species.astype(str) == species_name)) >= min_ood_samples_species
    ]
    if not kept_species:
        raise ValueError("No common species passed deployment sample-count filters.")

    ind_mask = np.isin(ind.species.astype(str), kept_species)
    ood_mask = np.isin(ood.species.astype(str), kept_species)
    return (
        PicklePayload(X=ind.X[ind_mask], species=ind.species[ind_mask], amr=ind.amr[ind_mask], antibiotics=ind.antibiotics),
        PicklePayload(X=ood.X[ood_mask], species=ood.species[ood_mask], amr=ood.amr[ood_mask], antibiotics=ood.antibiotics),
        kept_species,
    )


def _select_species_panel(
    ind_amr: np.ndarray,
    source_train_idx: np.ndarray,
    source_val_idx: np.ndarray,
    ft_idx: np.ndarray,
    test_idx: np.ndarray,
    ood_amr: np.ndarray,
    min_source_obs_per_antibiotic: int,
    min_source_val_obs_per_antibiotic: int,
    min_finetune_obs_per_antibiotic: int,
    min_test_obs_per_antibiotic: int,
) -> list[int]:
    return select_valid_antibiotics_legacy(
        source_amr_train=ind_amr[source_train_idx],
        source_amr_val=ind_amr[source_val_idx],
        ft_amr=ood_amr[ft_idx],
        test_amr=ood_amr[test_idx],
        min_source_obs=min_source_obs_per_antibiotic,
        min_source_val_obs=min_source_val_obs_per_antibiotic,
        min_finetune_obs=min_finetune_obs_per_antibiotic,
        min_test_obs=min_test_obs_per_antibiotic,
    )


def _split_ft_train_val(ft_idx: np.ndarray, seed: int, val_size: float) -> tuple[np.ndarray, np.ndarray]:
    local_train, local_val = split_train_val(len(ft_idx), seed, val_size)
    return ft_idx[local_train], ft_idx[local_val]


def _has_two_classes(y: np.ndarray) -> bool:
    _, _, _, two = count_obs_and_classes(y)
    return two


def _deployment_summary_from_predictions(output_dir: Path, antibiotics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prediction_dir = output_dir / "predictions"
    for model_dir in sorted(prediction_dir.glob("*")):
        if not model_dir.is_dir():
            continue
        blocks: list[dict[str, object]] = []
        for path in sorted(model_dir.glob("*.npz")):
            data = np.load(path, allow_pickle=True)
            blocks.append(
                {
                    "model": model_dir.name,
                    "scenario": str(data["scenario"].item()),
                    "fold": int(data["fold"].item()),
                    "species": str(data["species_name"].item()),
                    "antibiotic_indices": np.asarray(data["antibiotic_indices"]).astype(int),
                    "y_true": np.asarray(data["y_true"], dtype=float),
                    "y_score": np.asarray(data["y_score"], dtype=float),
                }
            )
        keys = sorted({(b["scenario"], b["fold"]) for b in blocks})
        for scenario, fold in keys:
            group = [b for b in blocks if b["scenario"] == scenario and b["fold"] == fold]
            yt_all: list[np.ndarray] = []
            ys_all: list[np.ndarray] = []
            patient_values: list[float] = []
            species_micro_values: list[float] = []
            task_values: list[float] = []
            species_antibiotic_pairs: set[tuple[str, int]] = set()
            antibiotic_set: set[int] = set()

            for block in group:
                y_true = block["y_true"]
                y_score = block["y_score"]
                mask = np.isfinite(y_true) & np.isfinite(y_score)
                yt_all.append(y_true[mask])
                ys_all.append(y_score[mask])

                for row in range(y_true.shape[0]):
                    auc = safe_auc(y_true[row], y_score[row])
                    if np.isfinite(auc):
                        patient_values.append(auc)

                species_micro = safe_auc(y_true.reshape(-1), y_score.reshape(-1))
                if np.isfinite(species_micro):
                    species_micro_values.append(species_micro)

                for local_col, antibiotic_idx in enumerate(block["antibiotic_indices"]):
                    antibiotic_set.add(int(antibiotic_idx))
                    species_antibiotic_pairs.add((str(block["species"]), int(antibiotic_idx)))
                    auc = safe_auc(y_true[:, local_col], y_score[:, local_col])
                    if np.isfinite(auc):
                        task_values.append(auc)

            y_true_flat = np.concatenate(yt_all) if yt_all else np.array([])
            y_score_flat = np.concatenate(ys_all) if ys_all else np.array([])
            rows.append(
                {
                    "scenario": scenario,
                    "model": model_dir.name,
                    "fold": fold,
                    "pooled_micro_auc": safe_auc(y_true_flat, y_score_flat),
                    "patient_auc_by_sample": float(np.mean(patient_values)) if patient_values else np.nan,
                    "mean_species_micro_auc": float(np.mean(species_micro_values)) if species_micro_values else np.nan,
                    "macro_auc_by_species_antibiotic": float(np.mean(task_values)) if task_values else np.nan,
                    "n_patient_auc_samples": len(patient_values),
                    "n_species_micro_aucs": len(species_micro_values),
                    "n_species_antibiotic_aucs": len(task_values),
                    "n_species_antibiotic_pairs_predicted": len(species_antibiotic_pairs),
                    "n_antibiotics_predicted": len(antibiotic_set),
                    "n_test_pairs_predicted": int(y_true_flat.size),
                    "antibiotics_predicted": ";".join(antibiotics[i] for i in sorted(antibiotic_set)),
                }
            )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary.to_csv(output_dir / "deployment_metrics_by_fold.csv", index=False)
        summary.groupby(["scenario", "model"], as_index=False).agg(
            pooled_micro_auc_mean=("pooled_micro_auc", "mean"),
            pooled_micro_auc_std=("pooled_micro_auc", "std"),
            patient_auc_by_sample_mean=("patient_auc_by_sample", "mean"),
            patient_auc_by_sample_std=("patient_auc_by_sample", "std"),
            mean_species_micro_auc_mean=("mean_species_micro_auc", "mean"),
            mean_species_micro_auc_std=("mean_species_micro_auc", "std"),
            macro_auc_by_species_antibiotic_mean=("macro_auc_by_species_antibiotic", "mean"),
            macro_auc_by_species_antibiotic_std=("macro_auc_by_species_antibiotic", "std"),
            n_test_pairs_predicted_mean=("n_test_pairs_predicted", "mean"),
            n_species_antibiotic_pairs_predicted_mean=("n_species_antibiotic_pairs_predicted", "mean"),
            n_antibiotics_predicted_mean=("n_antibiotics_predicted", "mean"),
        ).to_csv(output_dir / "deployment_metrics_summary.csv", index=False)
    return summary


def _run_global_grasp_deployment(
    ind: PicklePayload,
    ood: PicklePayload,
    ind_species_codes: np.ndarray,
    ood_species_codes: np.ndarray,
    model_name: str,
    run_id: int,
    source_train_idx: np.ndarray,
    source_val_idx: np.ndarray,
    ft_idx: np.ndarray,
    ft_train_idx: np.ndarray,
    ft_val_idx: np.ndarray,
    test_idx: np.ndarray,
    panel: list[int],
    train_cfg: TrainingConfig,
    grasp_cfg: GRASPConfig,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    model = build_model(model_name, ind.X.shape[1], len(panel), len(np.unique(ind_species_codes)), grasp_cfg)
    model, source_history = train_grasp_with_early_stopping(
        model,
        GlobalInteractionDataset(ind.X, ind_species_codes, ind.amr, source_train_idx, panel),
        GlobalInteractionDataset(ind.X, ind_species_codes, ind.amr, source_val_idx, panel),
        interaction_loss,
        grasp_cfg.optimizer,
        grasp_cfg.learning_rate,
        train_cfg,
        device,
        val_X=ind.X,
        val_species_codes=ind_species_codes,
        val_amr=ind.amr,
        val_sample_indices=source_val_idx,
        val_antibiotic_indices=panel,
    )

    before = predict_grasp(
        model,
        ood.X,
        ood_species_codes,
        test_idx,
        len(panel),
        device,
        batch_size=train_cfg.prediction_batch_size,
    )

    ft_model = copy.deepcopy(model)
    ft_model, ft_history = train_grasp_with_early_stopping(
        ft_model,
        GlobalInteractionDataset(ood.X, ood_species_codes, ood.amr, ft_train_idx, panel),
        GlobalInteractionDataset(ood.X, ood_species_codes, ood.amr, ft_val_idx, panel),
        interaction_loss,
        grasp_cfg.optimizer,
        grasp_cfg.learning_rate,
        _finetune_config(train_cfg),
        device,
        val_X=ood.X,
        val_species_codes=ood_species_codes,
        val_amr=ood.amr,
        val_sample_indices=ft_val_idx,
        val_antibiotic_indices=panel,
    )
    after = predict_grasp(
        ft_model,
        ood.X,
        ood_species_codes,
        test_idx,
        len(panel),
        device,
        batch_size=train_cfg.prediction_batch_size,
    )

    y_true = ood.amr[np.ix_(test_idx, panel)]
    species_array = ood.species.astype(str)
    for species_name in sorted(np.unique(species_array[test_idx])):
        species_test_idx = test_idx[species_array[test_idx] == species_name]
        local_mask = species_array[test_idx] == species_name
        save_predictions(output_dir, "zero_shot", model_name, species_name, run_id + 1, species_test_idx, panel, y_true[local_mask], before[local_mask])
        save_predictions(output_dir, "finetuned", model_name, species_name, run_id + 1, species_test_idx, panel, y_true[local_mask], after[local_mask])

    rows: list[dict[str, object]] = []
    for scenario, score in (("zero_shot", before), ("finetuned", after)):
        rows.append(
            {
                "scenario": scenario,
                "model": model_name,
                "fold": run_id + 1,
                "training_scope": "global",
                "n_source_samples_available": int(ind.X.shape[0]),
                "n_ood_samples_available": int(ood.X.shape[0]),
                "n_train_samples": int(len(source_train_idx)),
                "n_val_samples": int(len(source_val_idx)),
                "n_finetune_samples": int(len(ft_idx)),
                "n_finetune_train_samples": int(len(ft_train_idx)),
                "n_finetune_val_samples": int(len(ft_val_idx)),
                "n_test_samples": int(len(test_idx)),
                "n_antibiotics": int(len(panel)),
                "n_test_pairs": int(np.isfinite(y_true).sum()),
                "patient_auc": patient_auc(y_true, score),
                "pooled_micro_auc": safe_auc(y_true.reshape(-1), score.reshape(-1)),
                "macro_auc": macro_auc(y_true, score),
                "best_source_epoch": source_history.get("best_epoch", np.nan),
                "best_source_val_loss": source_history.get("best_val_loss", np.nan),
                "best_finetune_epoch": ft_history.get("best_epoch", np.nan),
                "best_finetune_val_loss": ft_history.get("best_val_loss", np.nan),
                "valid_antibiotics": ";".join(ind.antibiotics[j] for j in panel),
            }
        )
    return rows


def _run_grasp_mlp_deployment(
    ind: PicklePayload,
    ood: PicklePayload,
    ind_species_codes: np.ndarray,
    ood_species_codes: np.ndarray,
    run_id: int,
    source_train_idx: np.ndarray,
    source_val_idx: np.ndarray,
    ft_idx: np.ndarray,
    ft_train_idx: np.ndarray,
    ft_val_idx: np.ndarray,
    test_idx: np.ndarray,
    panel: list[int],
    train_cfg: TrainingConfig,
    grasp_cfg: GRASPConfig,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    model_cfg = GRASPMlpModelConfig(dropout=grasp_cfg.dropout, learning_rate=grasp_cfg.learning_rate)
    source_train_cfg = _grasp_mlp_training_config(train_cfg)
    ft_train_cfg = _grasp_mlp_finetune_config(train_cfg)

    model = SpeciesAwareGlobalMLP(
        input_dim=ind.X.shape[1],
        n_species=len(np.unique(ind_species_codes)),
        n_antibiotics=len(panel),
        cfg=model_cfg,
    )
    model, source_history = train_grasp_mlp_with_early_stopping(
        model,
        SpeciesAwareMultiOutputDataset(ind.X, ind_species_codes, ind.amr, source_train_idx, panel),
        SpeciesAwareMultiOutputDataset(ind.X, ind_species_codes, ind.amr, source_val_idx, panel),
        grasp_mlp_loss,
        model_cfg.optimizer,
        model_cfg.learning_rate,
        source_train_cfg,
        device,
        val_X=ind.X,
        val_species_codes=ind_species_codes,
        val_amr=ind.amr,
        val_sample_indices=source_val_idx,
        val_antibiotic_indices=panel,
    )
    before = predict_species_aware_mlp(
        model,
        ood.X,
        ood_species_codes,
        test_idx,
        device,
        batch_size=train_cfg.prediction_batch_size,
    )

    ft_model = copy.deepcopy(model)
    ft_model, ft_history = train_grasp_mlp_with_early_stopping(
        ft_model,
        SpeciesAwareMultiOutputDataset(ood.X, ood_species_codes, ood.amr, ft_train_idx, panel),
        SpeciesAwareMultiOutputDataset(ood.X, ood_species_codes, ood.amr, ft_val_idx, panel),
        grasp_mlp_loss,
        model_cfg.optimizer,
        model_cfg.learning_rate,
        ft_train_cfg,
        device,
        val_X=ood.X,
        val_species_codes=ood_species_codes,
        val_amr=ood.amr,
        val_sample_indices=ft_val_idx,
        val_antibiotic_indices=panel,
    )
    after = predict_species_aware_mlp(
        ft_model,
        ood.X,
        ood_species_codes,
        test_idx,
        device,
        batch_size=train_cfg.prediction_batch_size,
    )

    y_true = ood.amr[np.ix_(test_idx, panel)]
    species_array = ood.species.astype(str)
    for species_name in sorted(np.unique(species_array[test_idx])):
        species_test_idx = test_idx[species_array[test_idx] == species_name]
        local_mask = species_array[test_idx] == species_name
        save_predictions(
            output_dir,
            "zero_shot",
            GRASP_MLP_MODEL_NAME,
            species_name,
            run_id + 1,
            species_test_idx,
            panel,
            y_true[local_mask],
            before[local_mask],
        )
        save_predictions(
            output_dir,
            "finetuned",
            GRASP_MLP_MODEL_NAME,
            species_name,
            run_id + 1,
            species_test_idx,
            panel,
            y_true[local_mask],
            after[local_mask],
        )

    rows: list[dict[str, object]] = []
    for scenario, score in (("zero_shot", before), ("finetuned", after)):
        rows.append(
            {
                "scenario": scenario,
                "model": GRASP_MLP_MODEL_NAME,
                "fold": run_id + 1,
                "training_scope": "global",
                "n_source_samples_available": int(ind.X.shape[0]),
                "n_ood_samples_available": int(ood.X.shape[0]),
                "n_train_samples": int(len(source_train_idx)),
                "n_val_samples": int(len(source_val_idx)),
                "n_finetune_samples": int(len(ft_idx)),
                "n_finetune_train_samples": int(len(ft_train_idx)),
                "n_finetune_val_samples": int(len(ft_val_idx)),
                "n_test_samples": int(len(test_idx)),
                "n_antibiotics": int(len(panel)),
                "n_test_pairs": int(np.isfinite(y_true).sum()),
                "patient_auc": patient_auc(y_true, score),
                "pooled_micro_auc": safe_auc(y_true.reshape(-1), score.reshape(-1)),
                "macro_auc": macro_auc(y_true, score),
                "best_source_epoch": source_history.get("best_epoch", np.nan),
                "best_source_val_loss": source_history.get("best_val_loss", np.nan),
                "best_finetune_epoch": ft_history.get("best_epoch", np.nan),
                "best_finetune_val_loss": ft_history.get("best_val_loss", np.nan),
                "valid_antibiotics": ";".join(ind.antibiotics[j] for j in panel),
            }
        )
    return rows


def run_deployment_experiment(
    ind: PicklePayload,
    ood: PicklePayload,
    output_dir: Path,
    exp_cfg: ExperimentConfig,
    train_cfg: TrainingConfig,
    grasp_cfg: GRASPConfig,
    device: torch.device,
    model_names: tuple[str, ...] = ALL_MODELS,
    source_val_size: float = 0.15,
    finetune_val_size: float = 0.2,
    min_source_samples_species: int = 50,
    min_ood_samples_species: int = 50,
    min_source_obs_per_antibiotic: int = 50,
    min_source_val_obs_per_antibiotic: int = 5,
    min_finetune_obs_per_antibiotic: int = 10,
    min_test_obs_per_antibiotic: int = 10,
) -> pd.DataFrame:
    _set_seed(exp_cfg.random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)

    ind, ood, kept_species = _filter_common_species(
        ind,
        ood,
        min_source_samples_species=min_source_samples_species,
        min_ood_samples_species=min_ood_samples_species,
    )
    ind_species_codes, species_mapping = encode_species(ind.species)
    ood_species_codes = np.asarray([species_mapping[str(s)] for s in ood.species.astype(str)], dtype=np.int64)

    model_names = tuple(m for m in model_names if m in DEPLOYMENT_MODELS)
    if not model_names:
        raise ValueError(f"deployment_ood supports: {DEPLOYMENT_MODELS}")

    with (output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "protocol": "deployment_ood",
                "experiment": asdict(exp_cfg),
                "training": asdict(train_cfg),
                "grasp": asdict(grasp_cfg),
                "device": str(device),
                "models": list(model_names),
                "species_mapping": species_mapping,
                "antibiotics": ind.antibiotics,
                "source_val_size": source_val_size,
                "finetune_val_size": finetune_val_size,
                "min_source_samples_species": min_source_samples_species,
                "min_ood_samples_species": min_ood_samples_species,
                "min_source_obs_per_antibiotic": min_source_obs_per_antibiotic,
                "min_source_val_obs_per_antibiotic": min_source_val_obs_per_antibiotic,
                "min_finetune_obs_per_antibiotic": min_finetune_obs_per_antibiotic,
                "min_test_obs_per_antibiotic": min_test_obs_per_antibiotic,
            },
            handle,
            indent=2,
        )

    pd.DataFrame(
        {
            "species": kept_species,
            "n_source_samples": [int(np.sum(ind.species.astype(str) == s)) for s in kept_species],
            "n_ood_samples": [int(np.sum(ood.species.astype(str) == s)) for s in kept_species],
        }
    ).to_csv(output_dir / "deployment_species_counts.csv", index=False)

    rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    species_array = ind.species.astype(str)
    ood_species_array = ood.species.astype(str)
    baseline_exp_cfg = replace(
        exp_cfg,
        min_train_samples=min_source_obs_per_antibiotic,
        min_val_samples=min_source_val_obs_per_antibiotic,
    )

    for run_id in range(exp_cfg.n_folds):
        seed = exp_cfg.random_seed + run_id * 1000
        source_train_idx, source_val_idx = split_train_val(ind.X.shape[0], seed, source_val_size)
        ft_idx, test_idx = split_ood_by_species(ood_species_codes, ood.amr, exp_cfg.ood_adaptation_fraction, seed)
        ft_train_idx, ft_val_idx = _split_ft_train_val(ft_idx, seed + 2000, finetune_val_size)

        if any(model_name in model_names for model_name in GRASP_MODELS) or GRASP_MLP_MODEL_NAME in model_names:
            global_panel = _select_species_panel(
                ind.amr,
                source_train_idx,
                source_val_idx,
                ft_idx,
                test_idx,
                ood.amr,
                min_source_obs_per_antibiotic,
                min_source_val_obs_per_antibiotic,
                min_finetune_obs_per_antibiotic,
                min_test_obs_per_antibiotic,
            )
            coverage_rows.append(
                {
                    "fold": run_id + 1,
                    "scope": "global",
                    "model_family": "global",
                    "species": "ALL",
                    "n_train_samples": int(len(source_train_idx)),
                    "n_val_samples": int(len(source_val_idx)),
                    "n_finetune_samples": int(len(ft_idx)),
                    "n_test_samples": int(len(test_idx)),
                    "n_antibiotics": int(len(global_panel)),
                    "n_test_pairs": int(np.isfinite(ood.amr[np.ix_(test_idx, global_panel)]).sum()) if global_panel else 0,
                }
            )
            if global_panel:
                for model_name in GRASP_MODELS:
                    if model_name in model_names:
                        rows.extend(
                            _run_global_grasp_deployment(
                                ind,
                                ood,
                                ind_species_codes,
                                ood_species_codes,
                                model_name,
                                run_id,
                                source_train_idx,
                                source_val_idx,
                                ft_idx,
                                ft_train_idx,
                                ft_val_idx,
                                test_idx,
                                global_panel,
                                train_cfg,
                                grasp_cfg,
                                device,
                                output_dir,
                            )
                        )
                        pd.DataFrame(rows).to_csv(output_dir / "deployment_raw_metrics_incremental.csv", index=False)
                if GRASP_MLP_MODEL_NAME in model_names:
                    rows.extend(
                        _run_grasp_mlp_deployment(
                            ind,
                            ood,
                            ind_species_codes,
                            ood_species_codes,
                            run_id,
                            source_train_idx,
                            source_val_idx,
                            ft_idx,
                            ft_train_idx,
                            ft_val_idx,
                            test_idx,
                            global_panel,
                            train_cfg,
                            grasp_cfg,
                            device,
                            output_dir,
                        )
                    )
                    pd.DataFrame(rows).to_csv(output_dir / "deployment_raw_metrics_incremental.csv", index=False)

        for species_name in kept_species:
            source_species_idx = np.where(species_array == species_name)[0]
            ood_species_ft = ft_idx[ood_species_array[ft_idx] == species_name]
            ood_species_test = test_idx[ood_species_array[test_idx] == species_name]
            if source_species_idx.size < 5 or ood_species_test.size == 0:
                continue
            local_train, local_val = split_train_val(source_species_idx.size, seed + abs(hash(species_name)) % 1000, source_val_size)
            source_species_train = source_species_idx[local_train]
            source_species_val = source_species_idx[local_val]
            species_ft_train, species_ft_val = _split_ft_train_val(
                ood_species_ft,
                seed + 3000 + abs(hash(species_name)) % 1000,
                finetune_val_size,
            )
            species_panel = _select_species_panel(
                ind.amr,
                source_species_train,
                source_species_val,
                ood_species_ft,
                ood_species_test,
                ood.amr,
                min_source_obs_per_antibiotic,
                min_source_val_obs_per_antibiotic,
                min_finetune_obs_per_antibiotic,
                min_test_obs_per_antibiotic,
            )
            coverage_rows.append(
                {
                    "fold": run_id + 1,
                    "scope": "species",
                    "model_family": "baseline",
                    "species": species_name,
                    "n_train_samples": int(len(source_species_train)),
                    "n_val_samples": int(len(source_species_val)),
                    "n_finetune_samples": int(len(ood_species_ft)),
                    "n_test_samples": int(len(ood_species_test)),
                    "n_antibiotics": int(len(species_panel)),
                    "n_test_pairs": int(np.isfinite(ood.amr[np.ix_(ood_species_test, species_panel)]).sum()) if species_panel else 0,
                }
            )
            if not species_panel:
                continue
            if "semisupervised_binary_mlp" in model_names:
                rows.extend(
                    run_binary_species(
                        ind,
                        ood,
                        species_name,
                        run_id + 1,
                        source_species_train,
                        source_species_val,
                        species_ft_train,
                        species_ft_val,
                        ood_species_test,
                        species_panel,
                        baseline_exp_cfg,
                        train_cfg,
                        device,
                        output_dir,
                    )
                )
            if "semisupervised_multioutput_mlp" in model_names:
                rows.extend(
                    run_multioutput_species(
                        ind,
                        ood,
                        species_name,
                        run_id + 1,
                        source_species_train,
                        source_species_val,
                        species_ft_train,
                        species_ft_val,
                        ood_species_test,
                        species_panel,
                        baseline_exp_cfg,
                        train_cfg,
                        device,
                        output_dir,
                    )
                )
            if "species_recommender" in model_names:
                rows.extend(
                    run_species_recommender(
                        ind,
                        ood,
                        species_name,
                        run_id + 1,
                        source_species_train,
                        source_species_val,
                        species_ft_train,
                        species_ft_val,
                        ood_species_test,
                        species_panel,
                        train_cfg,
                        device,
                        output_dir,
                    )
                )
            pd.DataFrame(rows).to_csv(output_dir / "deployment_raw_metrics_incremental.csv", index=False)
            pd.DataFrame(coverage_rows).to_csv(output_dir / "deployment_coverage_incremental.csv", index=False)

    metrics = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage_rows)
    metrics.to_csv(output_dir / "deployment_raw_metrics.csv", index=False)
    coverage.to_csv(output_dir / "deployment_coverage_by_fold.csv", index=False)
    if not coverage.empty:
        coverage.groupby(["scope", "model_family"], as_index=False).agg(
            n_train_samples_mean=("n_train_samples", "mean"),
            n_val_samples_mean=("n_val_samples", "mean"),
            n_finetune_samples_mean=("n_finetune_samples", "mean"),
            n_test_samples_mean=("n_test_samples", "mean"),
            n_antibiotics_mean=("n_antibiotics", "mean"),
            n_test_pairs_mean=("n_test_pairs", "mean"),
        ).to_csv(output_dir / "deployment_coverage_summary.csv", index=False)

    write_reports(output_dir, ind.antibiotics)
    _deployment_summary_from_predictions(output_dir, ind.antibiotics)
    return metrics
