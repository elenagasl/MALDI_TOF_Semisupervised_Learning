from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, MLPConfig, RecommenderConfig, TrainingConfig
from .datasets import BinaryDataset, InteractionDataset, MultiOutputDataset
from .io import PicklePayload
from .metrics import macro_auc, micro_auc, patient_auc
from .models import BinaryMLP, MultiOutputMLP, SpeciesRecommender
from .panels import complete_profile_indices, labeled_indices_for_antibiotic, select_complete_profile_panel, select_sparse_panel
from .reporting import write_final_reports
from .splits import make_species_folds
from .training import (
    binary_loss,
    masked_multioutput_loss,
    predict_binary,
    predict_multioutput,
    predict_recommender,
    recommender_loss,
    train_with_early_stopping,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    species_name: str,
    fold: int,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    n_antibiotics: int,
) -> dict[str, object]:
    return {
        "model": model_name,
        "species": species_name,
        "fold": fold,
        "n_train_samples": n_train_samples,
        "n_val_samples": n_val_samples,
        "n_test_samples": n_test_samples,
        "n_antibiotics": n_antibiotics,
        "patient_auc": patient_auc(y_true, y_score),
        "micro_auc": micro_auc(y_true, y_score),
        "macro_auc": macro_auc(y_true, y_score),
    }


def fixed_mlp_config(exp_cfg: ExperimentConfig) -> MLPConfig:
    return MLPConfig(
        hidden_dims=exp_cfg.mlp_hidden_dims,
        activation=exp_cfg.mlp_activation,
        optimizer=exp_cfg.mlp_optimizer,
        learning_rate=exp_cfg.mlp_learning_rate,
    )


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


def train_supervised_binary(
    payload: PicklePayload,
    species_name: str,
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    panel: list[int],
    exp_cfg: ExperimentConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    if not panel:
        return []

    train_complete = complete_profile_indices(payload.amr, train_idx, panel)
    val_complete = complete_profile_indices(payload.amr, val_idx, panel)
    X_test = payload.X[test_idx]
    y_true = payload.amr[np.ix_(test_idx, panel)]
    y_score = np.full_like(y_true, np.nan, dtype=np.float32)

    for local_col, antibiotic_idx in enumerate(panel):
        y_train = payload.amr[train_complete, antibiotic_idx]
        y_val = payload.amr[val_complete, antibiotic_idx]
        if np.unique(y_train).size < 2 or np.unique(y_val).size < 2:
            continue
        best_cfg = fixed_mlp_config(exp_cfg)
        model = BinaryMLP(payload.X.shape[1], best_cfg.hidden_dims, best_cfg.activation)
        model, _ = train_with_early_stopping(
            model,
            BinaryDataset(payload.X[train_complete], y_train),
            BinaryDataset(payload.X[val_complete], y_val),
            binary_loss,
            best_cfg.optimizer,
            best_cfg.learning_rate,
            train_cfg,
            device,
        )
        y_score[:, local_col] = predict_binary(model, X_test, device)

    save_fold_predictions(
        output_dir,
        "supervised_binary_mlp",
        species_name,
        fold,
        test_idx,
        panel,
        y_true,
        y_score,
    )
    return [
        evaluate_predictions(
            y_true,
            y_score,
            "supervised_binary_mlp",
            species_name,
            fold,
            len(train_complete),
            len(val_complete),
            len(test_idx),
            len(panel),
        )
    ]


def train_supervised_multioutput(
    payload: PicklePayload,
    species_name: str,
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    panel: list[int],
    exp_cfg: ExperimentConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    if not panel:
        return []

    train_complete = complete_profile_indices(payload.amr, train_idx, panel)
    val_complete = complete_profile_indices(payload.amr, val_idx, panel)
    if len(train_complete) < exp_cfg.min_train_samples or len(val_complete) < exp_cfg.min_val_samples:
        return []

    y_train = payload.amr[np.ix_(train_complete, panel)]
    y_val = payload.amr[np.ix_(val_complete, panel)]
    best_cfg = fixed_mlp_config(exp_cfg)
    model = MultiOutputMLP(payload.X.shape[1], len(panel), best_cfg.hidden_dims, best_cfg.activation)
    model, _ = train_with_early_stopping(
        model,
        MultiOutputDataset(payload.X[train_complete], y_train),
        MultiOutputDataset(payload.X[val_complete], y_val),
        masked_multioutput_loss,
        best_cfg.optimizer,
        best_cfg.learning_rate,
        train_cfg,
        device,
    )
    y_true = payload.amr[np.ix_(test_idx, panel)]
    y_score = predict_multioutput(model, payload.X[test_idx], device)
    save_fold_predictions(
        output_dir,
        "supervised_multioutput_mlp",
        species_name,
        fold,
        test_idx,
        panel,
        y_true,
        y_score,
    )
    return [
        evaluate_predictions(
            y_true,
            y_score,
            "supervised_multioutput_mlp",
            species_name,
            fold,
            len(train_complete),
            len(val_complete),
            len(test_idx),
            len(panel),
        )
    ]


def train_semisupervised_binary(
    payload: PicklePayload,
    species_name: str,
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    panel: list[int],
    exp_cfg: ExperimentConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    if not panel:
        return []

    X_test = payload.X[test_idx]
    y_true = payload.amr[np.ix_(test_idx, panel)]
    y_score = np.full_like(y_true, np.nan, dtype=np.float32)
    trained_counts: list[int] = []
    val_counts: list[int] = []

    for local_col, antibiotic_idx in enumerate(panel):
        train_labeled = labeled_indices_for_antibiotic(payload.amr, train_idx, antibiotic_idx)
        val_labeled = labeled_indices_for_antibiotic(payload.amr, val_idx, antibiotic_idx)
        if len(train_labeled) < exp_cfg.min_train_samples or len(val_labeled) < exp_cfg.min_val_samples:
            continue
        y_train = payload.amr[train_labeled, antibiotic_idx]
        y_val = payload.amr[val_labeled, antibiotic_idx]
        if np.unique(y_train).size < 2 or np.unique(y_val).size < 2:
            continue
        best_cfg = fixed_mlp_config(exp_cfg)
        model = BinaryMLP(payload.X.shape[1], best_cfg.hidden_dims, best_cfg.activation)
        model, _ = train_with_early_stopping(
            model,
            BinaryDataset(payload.X[train_labeled], y_train),
            BinaryDataset(payload.X[val_labeled], y_val),
            binary_loss,
            best_cfg.optimizer,
            best_cfg.learning_rate,
            train_cfg,
            device,
        )
        y_score[:, local_col] = predict_binary(model, X_test, device)
        trained_counts.append(len(train_labeled))
        val_counts.append(len(val_labeled))

    save_fold_predictions(
        output_dir,
        "semisupervised_binary_mlp",
        species_name,
        fold,
        test_idx,
        panel,
        y_true,
        y_score,
    )
    return [
        evaluate_predictions(
            y_true,
            y_score,
            "semisupervised_binary_mlp",
            species_name,
            fold,
            int(np.mean(trained_counts)) if trained_counts else 0,
            int(np.mean(val_counts)) if val_counts else 0,
            len(test_idx),
            len(panel),
        )
    ]


def train_semisupervised_multioutput(
    payload: PicklePayload,
    species_name: str,
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    panel: list[int],
    exp_cfg: ExperimentConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    if not panel:
        return []

    y_train = payload.amr[np.ix_(train_idx, panel)]
    y_val = payload.amr[np.ix_(val_idx, panel)]
    best_cfg = fixed_mlp_config(exp_cfg)
    model = MultiOutputMLP(payload.X.shape[1], len(panel), best_cfg.hidden_dims, best_cfg.activation)
    model, _ = train_with_early_stopping(
        model,
        MultiOutputDataset(payload.X[train_idx], y_train),
        MultiOutputDataset(payload.X[val_idx], y_val),
        masked_multioutput_loss,
        best_cfg.optimizer,
        best_cfg.learning_rate,
        train_cfg,
        device,
    )
    y_true = payload.amr[np.ix_(test_idx, panel)]
    y_score = predict_multioutput(model, payload.X[test_idx], device)
    save_fold_predictions(
        output_dir,
        "semisupervised_multioutput_mlp",
        species_name,
        fold,
        test_idx,
        panel,
        y_true,
        y_score,
    )
    return [
        evaluate_predictions(
            y_true,
            y_score,
            "semisupervised_multioutput_mlp",
            species_name,
            fold,
            len(train_idx),
            len(val_idx),
            len(test_idx),
            len(panel),
        )
    ]


def train_species_recommender(
    payload: PicklePayload,
    species_name: str,
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    panel: list[int],
    train_cfg: TrainingConfig,
    rec_cfg: RecommenderConfig,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    if not panel:
        return []

    model = SpeciesRecommender(
        input_dim=payload.X.shape[1],
        n_antibiotics=len(panel),
        maldi_hidden_dims=rec_cfg.maldi_dims[1:],
        antibiotic_embedding_dim=rec_cfg.antibiotic_embedding_dim,
        interaction_hidden_dims=rec_cfg.interaction_dims[:-1],
        activation=rec_cfg.activation,
    )
    model, _ = train_with_early_stopping(
        model,
        InteractionDataset(payload.X, payload.amr, train_idx, panel),
        InteractionDataset(payload.X, payload.amr, val_idx, panel),
        recommender_loss,
        rec_cfg.optimizer,
        rec_cfg.learning_rate,
        train_cfg,
        device,
    )
    y_true = payload.amr[np.ix_(test_idx, panel)]
    y_score = predict_recommender(model, payload.X[test_idx], len(panel), device)
    save_fold_predictions(
        output_dir,
        "species_recommender",
        species_name,
        fold,
        test_idx,
        panel,
        y_true,
        y_score,
    )
    return [
        evaluate_predictions(
            y_true,
            y_score,
            "species_recommender",
            species_name,
            fold,
            len(train_idx),
            len(val_idx),
            len(test_idx),
            len(panel),
        )
    ]


def run_experiment(
    payload: PicklePayload,
    output_dir: Path,
    exp_cfg: ExperimentConfig,
    train_cfg: TrainingConfig,
    rec_cfg: RecommenderConfig,
    device: torch.device,
    only_species: list[str] | None = None,
) -> pd.DataFrame:
    set_seed(exp_cfg.random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "experiment": asdict(exp_cfg),
                "training": asdict(train_cfg),
                "recommender": asdict(rec_cfg),
                "device": str(device),
            },
            handle,
            indent=2,
        )

    rows: list[dict[str, object]] = []
    species_array = payload.species.astype(str)
    species_values = sorted(np.unique(species_array))
    if only_species:
        requested = set(only_species)
        species_values = [s for s in species_values if s in requested]

    for species_name in species_values:
        species_idx = np.where(species_array == species_name)[0]
        folds = make_species_folds(
            species_idx,
            n_folds=exp_cfg.n_folds,
            val_size=exp_cfg.val_size,
            random_seed=exp_cfg.random_seed,
        )
        for split in folds:
            supervised_panel = select_complete_profile_panel(
                payload.amr,
                split.train_idx,
                split.val_idx,
                exp_cfg.min_train_samples,
                exp_cfg.min_val_samples,
            )
            sparse_panel = select_sparse_panel(
                payload.amr,
                split.train_idx,
                split.val_idx,
                exp_cfg.min_train_samples,
                exp_cfg.min_val_samples,
            )

            rows.extend(
                train_supervised_binary(
                    payload,
                    species_name,
                    split.fold,
                    split.train_idx,
                    split.val_idx,
                    split.test_idx,
                    supervised_panel,
                    exp_cfg,
                    train_cfg,
                    device,
                    output_dir,
                )
            )
            rows.extend(
                train_supervised_multioutput(
                    payload,
                    species_name,
                    split.fold,
                    split.train_idx,
                    split.val_idx,
                    split.test_idx,
                    supervised_panel,
                    exp_cfg,
                    train_cfg,
                    device,
                    output_dir,
                )
            )
            rows.extend(
                train_semisupervised_binary(
                    payload,
                    species_name,
                    split.fold,
                    split.train_idx,
                    split.val_idx,
                    split.test_idx,
                    sparse_panel,
                    exp_cfg,
                    train_cfg,
                    device,
                    output_dir,
                )
            )
            rows.extend(
                train_semisupervised_multioutput(
                    payload,
                    species_name,
                    split.fold,
                    split.train_idx,
                    split.val_idx,
                    split.test_idx,
                    sparse_panel,
                    exp_cfg,
                    train_cfg,
                    device,
                    output_dir,
                )
            )
            rows.extend(
                train_species_recommender(
                    payload,
                    species_name,
                    split.fold,
                    split.train_idx,
                    split.val_idx,
                    split.test_idx,
                    sparse_panel,
                    train_cfg,
                    rec_cfg,
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
