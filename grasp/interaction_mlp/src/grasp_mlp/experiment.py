from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DEFAULT_EXCLUDED_SPECIES, MODEL_NAME, ExperimentConfig, ModelConfig, TrainingConfig
from .datasets import SpeciesAwareMultiOutputDataset
from .io import PicklePayload, filter_excluded_species
from .metrics import macro_auc, patient_auc, safe_auc
from .models import SpeciesAwareGlobalMLP
from .panels import select_sparse_panel
from .reporting import write_reports
from .splits import FoldSplit, make_global_folds, make_ood_folds, make_species_folds
from .training import (
    finetune_training_config,
    masked_multioutput_loss,
    predict_species_aware_mlp,
    train_with_early_stopping,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_species(species: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    values = sorted(str(s) for s in np.unique(species.astype(str)))
    mapping = {name: idx for idx, name in enumerate(values)}
    codes = np.asarray([mapping[str(s)] for s in species.astype(str)], dtype=np.int64)
    return codes, mapping


def _species_panels(
    payload: PicklePayload,
    exp_cfg: ExperimentConfig,
) -> dict[tuple[str, int], list[int]]:
    species_array = payload.species.astype(str)
    panels: dict[tuple[str, int], list[int]] = {}
    for species_name in sorted(np.unique(species_array)):
        species_idx = np.where(species_array == species_name)[0]
        for split in make_species_folds(species_idx, exp_cfg.n_folds, exp_cfg.val_size, exp_cfg.random_seed):
            panels[(species_name, split.fold)] = select_sparse_panel(
                payload.amr,
                split.train_idx,
                split.val_idx,
                exp_cfg.min_train_samples,
                exp_cfg.min_val_samples,
            )
    return panels


def _save_predictions(
    output_dir: Path,
    model_name: str,
    species_name: str,
    fold: int,
    test_idx: np.ndarray,
    antibiotic_indices: list[int],
    y_true: np.ndarray,
    y_score: np.ndarray,
    scenario: str | None = None,
) -> None:
    safe_species = species_name.replace(" ", "_").replace("/", "_")
    prefix = f"{scenario}_" if scenario else ""
    path = output_dir / "predictions" / model_name / f"{prefix}{safe_species}_fold_{fold}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model_name": np.asarray(model_name),
        "species_name": np.asarray(species_name),
        "fold": np.asarray(fold),
        "test_idx": test_idx,
        "antibiotic_indices": np.asarray(antibiotic_indices),
        "y_true": y_true,
        "y_score": y_score,
    }
    if scenario is not None:
        payload["scenario"] = np.asarray(scenario)
    np.savez_compressed(path, **payload)


def _metric_row(
    model_name: str,
    species_name: str,
    fold: int,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    n_antibiotics: int,
    history: dict[str, object],
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    scenario: str | None = None,
) -> dict[str, object]:
    row = {
        "model": model_name,
        "species": species_name,
        "fold": fold,
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "n_test_samples": n_test,
        "n_antibiotics": n_antibiotics,
        "best_val_loss": history.get("best_val_loss", float("nan")),
        "best_val_patient_auc": history.get("best_val_patient_auc", float("nan")),
        "best_epoch": history.get("best_epoch", float("nan")),
        "early_stopping_metric": history.get("early_stopping_metric", train_cfg.early_stopping_metric),
        "optimizer": model_cfg.optimizer,
        "learning_rate": model_cfg.learning_rate,
        "dropout": model_cfg.dropout,
        "weight_decay": train_cfg.weight_decay,
        "patient_auc": patient_auc(y_true, y_score),
        "micro_auc": safe_auc(y_true.reshape(-1), y_score.reshape(-1)),
        "macro_auc": macro_auc(y_true, y_score),
    }
    if scenario is not None:
        row = {"scenario": scenario, **row}
    return row


def _build_model(payload: PicklePayload, n_species: int, n_antibiotics: int, model_cfg: ModelConfig) -> SpeciesAwareGlobalMLP:
    return SpeciesAwareGlobalMLP(
        input_dim=payload.X.shape[1],
        n_species=n_species,
        n_antibiotics=n_antibiotics,
        cfg=model_cfg,
    )


def _train_model(
    payload: PicklePayload,
    species_codes: np.ndarray,
    n_species: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    antibiotic_indices: list[int],
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
) -> tuple[SpeciesAwareGlobalMLP, dict[str, object]]:
    model = _build_model(payload, n_species, len(antibiotic_indices), model_cfg)
    return train_with_early_stopping(
        model=model,
        train_dataset=SpeciesAwareMultiOutputDataset(payload.X, species_codes, payload.amr, train_idx, antibiotic_indices),
        val_dataset=SpeciesAwareMultiOutputDataset(payload.X, species_codes, payload.amr, val_idx, antibiotic_indices),
        loss_fn=masked_multioutput_loss,
        optimizer_name=model_cfg.optimizer,
        learning_rate=model_cfg.learning_rate,
        training_config=train_cfg,
        device=device,
        val_X=payload.X,
        val_species_codes=species_codes,
        val_amr=payload.amr,
        val_sample_indices=val_idx,
        val_antibiotic_indices=antibiotic_indices,
    )


def _evaluate_species_blocks(
    payload: PicklePayload,
    species_codes: np.ndarray,
    model: SpeciesAwareGlobalMLP,
    split: FoldSplit,
    training_panel: list[int],
    species_panels: dict[tuple[str, int], list[int]],
    evaluation_panel_mode: str,
    train_cfg: TrainingConfig,
    device: torch.device,
    output_dir: Path,
    history: dict[str, object],
    model_cfg: ModelConfig,
    scenario: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    species_array = payload.species.astype(str)
    panel_to_local = {antibiotic_idx: local_col for local_col, antibiotic_idx in enumerate(training_panel)}
    for species_name in sorted(np.unique(species_array[split.test_idx])):
        species_test_idx = split.test_idx[species_array[split.test_idx] == species_name]
        full_y_true = payload.amr[np.ix_(species_test_idx, training_panel)]
        full_y_score = predict_species_aware_mlp(
            model,
            payload.X,
            species_codes,
            species_test_idx,
            device,
            train_cfg.prediction_batch_size,
        )
        if evaluation_panel_mode == "species":
            species_panel = species_panels.get((species_name, split.fold), [])
            local_cols = [panel_to_local[idx] for idx in species_panel if idx in panel_to_local]
            if not local_cols:
                continue
            eval_antibiotic_indices = [training_panel[col] for col in local_cols]
            y_true = full_y_true[:, local_cols]
            y_score = full_y_score[:, local_cols]
        elif evaluation_panel_mode == "global":
            eval_antibiotic_indices = training_panel
            y_true = full_y_true
            y_score = full_y_score
        else:
            raise ValueError(f"Unsupported evaluation_panel_mode: {evaluation_panel_mode}")

        _save_predictions(output_dir, MODEL_NAME, species_name, split.fold, species_test_idx, eval_antibiotic_indices, y_true, y_score, scenario)
        rows.append(
            _metric_row(
                MODEL_NAME,
                species_name,
                split.fold,
                y_true,
                y_score,
                len(split.train_idx),
                len(split.val_idx),
                len(species_test_idx),
                len(eval_antibiotic_indices),
                history,
                model_cfg,
                train_cfg,
                scenario,
            )
        )
    return rows


def run_ind_experiment(
    payload: PicklePayload,
    output_dir: Path,
    exp_cfg: ExperimentConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    excluded_species: tuple[str, ...] = DEFAULT_EXCLUDED_SPECIES,
    evaluation_panel_mode: str = "species",
) -> pd.DataFrame:
    set_seed(exp_cfg.random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    original_n_samples = payload.X.shape[0]
    payload, matched_excluded_species, excluded_n_samples = filter_excluded_species(payload, excluded_species)
    species_codes, species_mapping = encode_species(payload.species)
    species_panels = _species_panels(payload, exp_cfg)
    folds = make_global_folds(payload.species, exp_cfg.n_folds, exp_cfg.val_size, exp_cfg.random_seed)

    with (output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "experiment": asdict(exp_cfg),
                "model": asdict(model_cfg),
                "training": asdict(train_cfg),
                "device": str(device),
                "species_mapping": species_mapping,
                "excluded_species": list(excluded_species),
                "matched_excluded_species": matched_excluded_species,
                "excluded_n_samples": excluded_n_samples,
                "original_n_samples": int(original_n_samples),
                "evaluation_panel_mode": evaluation_panel_mode,
            },
            handle,
            indent=2,
        )

    rows: list[dict[str, object]] = []
    for split in folds:
        panel = select_sparse_panel(payload.amr, split.train_idx, split.val_idx, exp_cfg.min_train_samples, exp_cfg.min_val_samples)
        if not panel:
            continue
        model, history = _train_model(payload, species_codes, len(species_mapping), split.train_idx, split.val_idx, panel, model_cfg, train_cfg, device)
        rows.extend(
            _evaluate_species_blocks(
                payload,
                species_codes,
                model,
                split,
                panel,
                species_panels,
                evaluation_panel_mode,
                train_cfg,
                device,
                output_dir,
                history,
                model_cfg,
            )
        )
        pd.DataFrame(rows).to_csv(output_dir / "metrics_by_fold.csv", index=False)

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        summary = metrics.groupby("model", as_index=False)[["patient_auc", "micro_auc", "macro_auc"]].agg(["mean", "std"]).reset_index()
        summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    write_reports(output_dir, payload.antibiotics)
    return metrics


def run_ood_experiment(
    ind: PicklePayload,
    ood: PicklePayload,
    output_dir: Path,
    exp_cfg: ExperimentConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    excluded_species: tuple[str, ...] = DEFAULT_EXCLUDED_SPECIES,
    evaluation_panel_mode: str = "species",
) -> pd.DataFrame:
    set_seed(exp_cfg.random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    ind, matched_ind, excluded_ind = filter_excluded_species(ind, excluded_species)
    ood, matched_ood, excluded_ood = filter_excluded_species(ood, excluded_species)
    ind_species_codes, species_mapping = encode_species(ind.species)
    ood_species_codes = np.asarray([species_mapping[str(s)] for s in ood.species.astype(str)], dtype=np.int64)
    source_folds = make_global_folds(ind.species, exp_cfg.n_folds, exp_cfg.val_size, exp_cfg.random_seed)
    ood_folds = make_ood_folds(
        ood.species,
        ood.amr,
        exp_cfg.n_folds,
        exp_cfg.ood_adaptation_fraction,
        exp_cfg.finetune_val_size,
        exp_cfg.random_seed,
    )
    species_panels = _species_panels(ind, exp_cfg)

    with (output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "experiment": asdict(exp_cfg),
                "model": asdict(model_cfg),
                "training": asdict(train_cfg),
                "device": str(device),
                "species_mapping": species_mapping,
                "excluded_species": list(excluded_species),
                "matched_excluded_species_in": matched_ind,
                "matched_excluded_species_ood": matched_ood,
                "excluded_n_samples_in": excluded_ind,
                "excluded_n_samples_ood": excluded_ood,
                "evaluation_panel_mode": evaluation_panel_mode,
            },
            handle,
            indent=2,
        )

    rows: list[dict[str, object]] = []
    ind_species_array = ind.species.astype(str)
    ood_species_array = ood.species.astype(str)
    for source_split, ood_split in zip(source_folds, ood_folds, strict=True):
        panel = select_sparse_panel(ind.amr, source_split.train_idx, source_split.val_idx, exp_cfg.min_train_samples, exp_cfg.min_val_samples)
        if not panel:
            continue
        model, history = _train_model(ind, ind_species_codes, len(species_mapping), source_split.train_idx, source_split.val_idx, panel, model_cfg, train_cfg, device)

        ft_model = copy.deepcopy(model)
        ft_history = dict(history)
        if ood_split.adapt_train_idx.size > 0 and ood_split.adapt_val_idx.size > 0:
            ft_cfg = finetune_training_config(train_cfg)
            ft_model, ft_history = train_with_early_stopping(
                model=ft_model,
                train_dataset=SpeciesAwareMultiOutputDataset(ood.X, ood_species_codes, ood.amr, ood_split.adapt_train_idx, panel),
                val_dataset=SpeciesAwareMultiOutputDataset(ood.X, ood_species_codes, ood.amr, ood_split.adapt_val_idx, panel),
                loss_fn=masked_multioutput_loss,
                optimizer_name=model_cfg.optimizer,
                learning_rate=model_cfg.learning_rate,
                training_config=ft_cfg,
                device=device,
                val_X=ood.X,
                val_species_codes=ood_species_codes,
                val_amr=ood.amr,
                val_sample_indices=ood_split.adapt_val_idx,
                val_antibiotic_indices=panel,
            )

        panel_to_local = {antibiotic_idx: local_col for local_col, antibiotic_idx in enumerate(panel)}
        for species_name in sorted(np.unique(ood_species_array[ood_split.test_idx])):
            ood_species_test = ood_split.test_idx[ood_species_array[ood_split.test_idx] == species_name]
            if ood_species_test.size == 0:
                continue
            if evaluation_panel_mode == "species":
                source_species_idx = np.where(ind_species_array == species_name)[0]
                source_species_folds = make_species_folds(source_species_idx, exp_cfg.n_folds, exp_cfg.val_size, exp_cfg.random_seed)
                if len(source_species_folds) < source_split.fold:
                    continue
                species_panel = species_panels.get((species_name, source_split.fold), [])
                local_cols = [panel_to_local[idx] for idx in species_panel if idx in panel_to_local]
                if not local_cols:
                    continue
                eval_antibiotic_indices = [panel[col] for col in local_cols]
            elif evaluation_panel_mode == "global":
                local_cols = list(range(len(panel)))
                eval_antibiotic_indices = panel
            else:
                raise ValueError(f"Unsupported evaluation_panel_mode: {evaluation_panel_mode}")

            y_true_full = ood.amr[np.ix_(ood_species_test, panel)]
            zero_score_full = predict_species_aware_mlp(model, ood.X, ood_species_codes, ood_species_test, device, train_cfg.prediction_batch_size)
            ft_score_full = predict_species_aware_mlp(ft_model, ood.X, ood_species_codes, ood_species_test, device, train_cfg.prediction_batch_size)
            y_true = y_true_full[:, local_cols]
            zero_score = zero_score_full[:, local_cols]
            ft_score = ft_score_full[:, local_cols]

            for scenario, score, scenario_history in (
                ("zero_shot", zero_score, history),
                ("finetuned", ft_score, ft_history),
            ):
                _save_predictions(
                    output_dir,
                    MODEL_NAME,
                    species_name,
                    source_split.fold,
                    ood_species_test,
                    eval_antibiotic_indices,
                    y_true,
                    score,
                    scenario,
                )
                rows.append(
                    _metric_row(
                        MODEL_NAME,
                        species_name,
                        source_split.fold,
                        y_true,
                        score,
                        len(source_split.train_idx),
                        len(source_split.val_idx),
                        len(ood_species_test),
                        len(eval_antibiotic_indices),
                        scenario_history,
                        model_cfg,
                        train_cfg,
                        scenario,
                    )
                )
        pd.DataFrame(rows).to_csv(output_dir / "metrics_by_fold.csv", index=False)

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        summary = (
            metrics.groupby(["scenario", "model"], as_index=False)[["patient_auc", "micro_auc", "macro_auc"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    write_reports(output_dir, ind.antibiotics)
    return metrics

