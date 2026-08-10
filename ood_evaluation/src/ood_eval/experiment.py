from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from supervised_semisupervised.datasets import BinaryDataset, InteractionDataset, MultiOutputDataset
from supervised_semisupervised.config import MLPConfig
from supervised_semisupervised.models import BinaryMLP, MultiOutputMLP, SpeciesRecommender
from supervised_semisupervised.training import (
    binary_loss,
    masked_multioutput_loss,
    predict_binary,
    predict_multioutput,
    predict_recommender,
    recommender_loss,
    train_with_early_stopping,
)
from grasp_global.datasets import GlobalInteractionDataset
from grasp_global.models import build_model
from grasp_global.training import (
    interaction_loss,
    predict_grasp,
    train_with_early_stopping as train_grasp_with_early_stopping,
)

from .config import ExperimentConfig, GRASPConfig, TrainingConfig
from .io import PicklePayload
from .metrics import macro_auc, patient_auc, safe_auc
from .panels import labeled_indices_for_antibiotic, select_sparse_panel
from .reporting import write_reports
from .splits import make_global_source_folds, make_ood_folds, make_species_folds


BASELINE_MODELS = ("semisupervised_binary_mlp", "semisupervised_multioutput_mlp", "species_recommender")
GRASP_MODELS = ("species_conditioned_grasp", "multihead_grasp", "hypernetwork_grasp")
ALL_MODELS = BASELINE_MODELS + GRASP_MODELS


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finetune_config(cfg: TrainingConfig) -> TrainingConfig:
    return TrainingConfig(
        batch_size=cfg.batch_size,
        prediction_batch_size=cfg.prediction_batch_size,
        max_epochs=cfg.finetune_epochs,
        patience=cfg.finetune_patience,
        num_workers=cfg.num_workers,
        weight_decay=cfg.weight_decay,
        early_stopping_metric=cfg.early_stopping_metric,
        pin_memory=cfg.pin_memory,
    )


def encode_species(species: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    values = sorted(str(s) for s in np.unique(species.astype(str)))
    mapping = {name: i for i, name in enumerate(values)}
    return np.asarray([mapping[str(s)] for s in species.astype(str)], dtype=np.int64), mapping


def fixed_mlp_config(exp_cfg: ExperimentConfig) -> MLPConfig:
    return MLPConfig(
        hidden_dims=exp_cfg.mlp_hidden_dims,
        activation=exp_cfg.mlp_activation,
        optimizer=exp_cfg.mlp_optimizer,
        learning_rate=exp_cfg.mlp_learning_rate,
    )


def save_predictions(
    output_dir: Path,
    scenario: str,
    model_name: str,
    species_name: str,
    fold: int,
    test_idx: np.ndarray,
    antibiotic_indices: list[int],
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> None:
    safe_species = species_name.replace(" ", "_").replace("/", "_")
    path = output_dir / "predictions" / model_name / f"{scenario}_{safe_species}_fold_{fold}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scenario=np.asarray(scenario),
        model_name=np.asarray(model_name),
        species_name=np.asarray(species_name),
        fold=np.asarray(fold),
        test_idx=test_idx,
        antibiotic_indices=np.asarray(antibiotic_indices),
        y_true=y_true,
        y_score=y_score,
    )


def metric_row(
    scenario: str,
    model_name: str,
    species_name: str,
    fold: int,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    n_antibiotics: int,
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "model": model_name,
        "species": species_name,
        "fold": fold,
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "n_test_samples": n_test,
        "n_antibiotics": n_antibiotics,
        "patient_auc": patient_auc(y_true, y_score),
        "micro_auc": safe_auc(y_true.reshape(-1), y_score.reshape(-1)),
        "macro_auc": macro_auc(y_true, y_score),
    }


def run_binary_species(
    ind: PicklePayload,
    ood: PicklePayload,
    species_name: str,
    fold: int,
    source_train_idx: np.ndarray,
    source_val_idx: np.ndarray,
    ood_adapt_train_idx: np.ndarray,
    ood_adapt_val_idx: np.ndarray,
    ood_test_idx: np.ndarray,
    panel: list[int],
    exp_cfg: ExperimentConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y_true = ood.amr[np.ix_(ood_test_idx, panel)]
    zero_score = np.full_like(y_true, np.nan, dtype=np.float32)
    ft_score = np.full_like(y_true, np.nan, dtype=np.float32)

    for local_col, antibiotic_idx in enumerate(panel):
        train_labeled = labeled_indices_for_antibiotic(ind.amr, source_train_idx, antibiotic_idx)
        val_labeled = labeled_indices_for_antibiotic(ind.amr, source_val_idx, antibiotic_idx)
        if len(train_labeled) < exp_cfg.min_train_samples or len(val_labeled) < exp_cfg.min_val_samples:
            continue
        y_train = ind.amr[train_labeled, antibiotic_idx]
        y_val = ind.amr[val_labeled, antibiotic_idx]
        if np.unique(y_train).size < 2 or np.unique(y_val).size < 2:
            continue
        cfg = fixed_mlp_config(exp_cfg)
        model = BinaryMLP(ind.X.shape[1], cfg.hidden_dims, cfg.activation)
        model, _ = train_with_early_stopping(
            model,
            BinaryDataset(ind.X[train_labeled], y_train),
            BinaryDataset(ind.X[val_labeled], y_val),
            binary_loss,
            cfg.optimizer,
            cfg.learning_rate,
            train_cfg,
            device,
        )
        zero_score[:, local_col] = predict_binary(model, ood.X[ood_test_idx], device)

        ft_model = copy.deepcopy(model)
        ft_train = labeled_indices_for_antibiotic(ood.amr, ood_adapt_train_idx, antibiotic_idx)
        ft_val = labeled_indices_for_antibiotic(ood.amr, ood_adapt_val_idx, antibiotic_idx)
        if ft_train.size > 0 and ft_val.size > 0 and np.unique(ood.amr[ft_train, antibiotic_idx]).size >= 2:
            ft_model, _ = train_with_early_stopping(
                ft_model,
                BinaryDataset(ood.X[ft_train], ood.amr[ft_train, antibiotic_idx]),
                BinaryDataset(ood.X[ft_val], ood.amr[ft_val, antibiotic_idx]),
                binary_loss,
                cfg.optimizer,
                cfg.learning_rate,
                finetune_config(train_cfg),
                device,
            )
        ft_score[:, local_col] = predict_binary(ft_model, ood.X[ood_test_idx], device)

    for scenario, score in (("zero_shot", zero_score), ("finetuned", ft_score)):
        save_predictions(output_dir, scenario, "semisupervised_binary_mlp", species_name, fold, ood_test_idx, panel, y_true, score)
        rows.append(metric_row(scenario, "semisupervised_binary_mlp", species_name, fold, y_true, score, len(source_train_idx), len(source_val_idx), len(ood_test_idx), len(panel)))
    return rows


def run_multioutput_species(
    ind: PicklePayload,
    ood: PicklePayload,
    species_name: str,
    fold: int,
    source_train_idx: np.ndarray,
    source_val_idx: np.ndarray,
    ood_adapt_train_idx: np.ndarray,
    ood_adapt_val_idx: np.ndarray,
    ood_test_idx: np.ndarray,
    panel: list[int],
    exp_cfg: ExperimentConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    y_train = ind.amr[np.ix_(source_train_idx, panel)]
    y_val = ind.amr[np.ix_(source_val_idx, panel)]
    cfg = fixed_mlp_config(exp_cfg)
    model = MultiOutputMLP(ind.X.shape[1], len(panel), cfg.hidden_dims, cfg.activation)
    model, _ = train_with_early_stopping(
        model,
        MultiOutputDataset(ind.X[source_train_idx], y_train),
        MultiOutputDataset(ind.X[source_val_idx], y_val),
        masked_multioutput_loss,
        cfg.optimizer,
        cfg.learning_rate,
        train_cfg,
        device,
    )
    y_true = ood.amr[np.ix_(ood_test_idx, panel)]
    zero_score = predict_multioutput(model, ood.X[ood_test_idx], device)

    ft_model = copy.deepcopy(model)
    if ood_adapt_train_idx.size > 0 and ood_adapt_val_idx.size > 0:
        ft_model, _ = train_with_early_stopping(
            ft_model,
            MultiOutputDataset(ood.X[ood_adapt_train_idx], ood.amr[np.ix_(ood_adapt_train_idx, panel)]),
            MultiOutputDataset(ood.X[ood_adapt_val_idx], ood.amr[np.ix_(ood_adapt_val_idx, panel)]),
            masked_multioutput_loss,
            cfg.optimizer,
            cfg.learning_rate,
            finetune_config(train_cfg),
            device,
        )
    ft_score = predict_multioutput(ft_model, ood.X[ood_test_idx], device)

    rows: list[dict[str, object]] = []
    for scenario, score in (("zero_shot", zero_score), ("finetuned", ft_score)):
        save_predictions(output_dir, scenario, "semisupervised_multioutput_mlp", species_name, fold, ood_test_idx, panel, y_true, score)
        rows.append(metric_row(scenario, "semisupervised_multioutput_mlp", species_name, fold, y_true, score, len(source_train_idx), len(source_val_idx), len(ood_test_idx), len(panel)))
    return rows


def run_species_recommender(
    ind: PicklePayload,
    ood: PicklePayload,
    species_name: str,
    fold: int,
    source_train_idx: np.ndarray,
    source_val_idx: np.ndarray,
    ood_adapt_train_idx: np.ndarray,
    ood_adapt_val_idx: np.ndarray,
    ood_test_idx: np.ndarray,
    panel: list[int],
    train_cfg: TrainingConfig,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    model = SpeciesRecommender(ind.X.shape[1], len(panel))
    model, _ = train_with_early_stopping(
        model,
        InteractionDataset(ind.X, ind.amr, source_train_idx, panel),
        InteractionDataset(ind.X, ind.amr, source_val_idx, panel),
        recommender_loss,
        "adam",
        1e-3,
        train_cfg,
        device,
    )
    y_true = ood.amr[np.ix_(ood_test_idx, panel)]
    zero_score = predict_recommender(model, ood.X[ood_test_idx], len(panel), device)

    ft_model = copy.deepcopy(model)
    if ood_adapt_train_idx.size > 0 and ood_adapt_val_idx.size > 0:
        ft_model, _ = train_with_early_stopping(
            ft_model,
            InteractionDataset(ood.X, ood.amr, ood_adapt_train_idx, panel),
            InteractionDataset(ood.X, ood.amr, ood_adapt_val_idx, panel),
            recommender_loss,
            "adam",
            1e-3,
            finetune_config(train_cfg),
            device,
        )
    ft_score = predict_recommender(ft_model, ood.X[ood_test_idx], len(panel), device)

    rows: list[dict[str, object]] = []
    for scenario, score in (("zero_shot", zero_score), ("finetuned", ft_score)):
        save_predictions(output_dir, scenario, "species_recommender", species_name, fold, ood_test_idx, panel, y_true, score)
        rows.append(metric_row(scenario, "species_recommender", species_name, fold, y_true, score, len(source_train_idx), len(source_val_idx), len(ood_test_idx), len(panel)))
    return rows


def run_grasp_model(
    ind: PicklePayload,
    ood: PicklePayload,
    ind_species_codes: np.ndarray,
    ood_species_codes: np.ndarray,
    n_species: int,
    model_name: str,
    fold: int,
    source_train_idx: np.ndarray,
    source_val_idx: np.ndarray,
    ood_adapt_train_idx: np.ndarray,
    ood_adapt_val_idx: np.ndarray,
    ood_test_idx: np.ndarray,
    panel: list[int],
    exp_cfg: ExperimentConfig,
    train_cfg: TrainingConfig,
    grasp_cfg: GRASPConfig,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    model = build_model(model_name, ind.X.shape[1], len(panel), n_species, grasp_cfg)
    model, _ = train_grasp_with_early_stopping(
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
    ft_model = copy.deepcopy(model)
    if ood_adapt_train_idx.size > 0 and ood_adapt_val_idx.size > 0:
        ft_model, _ = train_grasp_with_early_stopping(
            ft_model,
            GlobalInteractionDataset(ood.X, ood_species_codes, ood.amr, ood_adapt_train_idx, panel),
            GlobalInteractionDataset(ood.X, ood_species_codes, ood.amr, ood_adapt_val_idx, panel),
            interaction_loss,
            grasp_cfg.optimizer,
            grasp_cfg.learning_rate,
            finetune_config(train_cfg),
            device,
            val_X=ood.X,
            val_species_codes=ood_species_codes,
            val_amr=ood.amr,
            val_sample_indices=ood_adapt_val_idx,
            val_antibiotic_indices=panel,
        )

    rows: list[dict[str, object]] = []
    species_array = ood.species.astype(str)
    for species_name in sorted(np.unique(species_array[ood_test_idx])):
        species_test_idx = ood_test_idx[species_array[ood_test_idx] == species_name]
        y_true = ood.amr[np.ix_(species_test_idx, panel)]
        zero_score = predict_grasp(
            model,
            ood.X,
            ood_species_codes,
            species_test_idx,
            len(panel),
            device,
            batch_size=train_cfg.prediction_batch_size,
        )
        ft_score = predict_grasp(
            ft_model,
            ood.X,
            ood_species_codes,
            species_test_idx,
            len(panel),
            device,
            batch_size=train_cfg.prediction_batch_size,
        )
        for scenario, score in (("zero_shot", zero_score), ("finetuned", ft_score)):
            save_predictions(output_dir, scenario, model_name, species_name, fold, species_test_idx, panel, y_true, score)
            rows.append(metric_row(scenario, model_name, species_name, fold, y_true, score, len(source_train_idx), len(source_val_idx), len(species_test_idx), len(panel)))
    return rows


def run_experiment(
    ind: PicklePayload,
    ood: PicklePayload,
    output_dir: Path,
    exp_cfg: ExperimentConfig,
    train_cfg: TrainingConfig,
    grasp_cfg: GRASPConfig,
    device: torch.device,
    model_names: tuple[str, ...] = ALL_MODELS,
) -> pd.DataFrame:
    set_seed(exp_cfg.random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)

    ind_species_codes, species_mapping = encode_species(ind.species)
    ood_species_codes = np.asarray([species_mapping[str(s)] for s in ood.species.astype(str)], dtype=np.int64)
    source_global_folds = make_global_source_folds(ind.species, exp_cfg.n_folds, exp_cfg.val_size, exp_cfg.random_seed)
    ood_folds = make_ood_folds(ood.species, exp_cfg.n_folds, exp_cfg.ood_adaptation_fraction, exp_cfg.finetune_val_size, exp_cfg.random_seed)

    with (output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "experiment": asdict(exp_cfg),
                "training": asdict(train_cfg),
                "grasp": asdict(grasp_cfg),
                "device": str(device),
                "models": list(model_names),
                "species_mapping": species_mapping,
                "antibiotics": ind.antibiotics,
            },
            handle,
            indent=2,
        )

    rows: list[dict[str, object]] = []
    species_array = ind.species.astype(str)
    ood_species_array = ood.species.astype(str)

    for source_split, ood_split in zip(source_global_folds, ood_folds, strict=True):
        panel = select_sparse_panel(ind.amr, source_split.train_idx, source_split.val_idx, exp_cfg.min_train_samples, exp_cfg.min_val_samples)
        if not panel:
            continue

        for species_name in sorted(np.unique(species_array)):
            source_species_idx = np.where(species_array == species_name)[0]
            source_species_folds = make_species_folds(source_species_idx, exp_cfg.n_folds, exp_cfg.val_size, exp_cfg.random_seed)
            if len(source_species_folds) < source_split.fold:
                continue
            source_species_split = source_species_folds[source_split.fold - 1]
            ood_species_train = ood_split.adapt_train_idx[ood_species_array[ood_split.adapt_train_idx] == species_name]
            ood_species_val = ood_split.adapt_val_idx[ood_species_array[ood_split.adapt_val_idx] == species_name]
            ood_species_test = ood_split.test_idx[ood_species_array[ood_split.test_idx] == species_name]
            if ood_species_test.size == 0:
                continue

            species_panel = select_sparse_panel(ind.amr, source_species_split.train_idx, source_species_split.val_idx, exp_cfg.min_train_samples, exp_cfg.min_val_samples)
            species_panel = [ab for ab in species_panel if ab in panel]
            if not species_panel:
                continue

            if "semisupervised_binary_mlp" in model_names:
                rows.extend(run_binary_species(ind, ood, species_name, source_split.fold, source_species_split.train_idx, source_species_split.val_idx, ood_species_train, ood_species_val, ood_species_test, species_panel, exp_cfg, train_cfg, device, output_dir))
            if "semisupervised_multioutput_mlp" in model_names:
                rows.extend(run_multioutput_species(ind, ood, species_name, source_split.fold, source_species_split.train_idx, source_species_split.val_idx, ood_species_train, ood_species_val, ood_species_test, species_panel, exp_cfg, train_cfg, device, output_dir))
            if "species_recommender" in model_names:
                rows.extend(run_species_recommender(ind, ood, species_name, source_split.fold, source_species_split.train_idx, source_species_split.val_idx, ood_species_train, ood_species_val, ood_species_test, species_panel, train_cfg, device, output_dir))

        for model_name in GRASP_MODELS:
            if model_name in model_names:
                rows.extend(run_grasp_model(ind, ood, ind_species_codes, ood_species_codes, len(species_mapping), model_name, source_split.fold, source_split.train_idx, source_split.val_idx, ood_split.adapt_train_idx, ood_split.adapt_val_idx, ood_split.test_idx, panel, exp_cfg, train_cfg, grasp_cfg, device, output_dir))

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
