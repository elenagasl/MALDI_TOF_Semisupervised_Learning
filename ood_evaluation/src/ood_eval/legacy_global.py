from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

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
from .reporting import write_reports


LEGACY_GLOBAL_MODELS = ("multihead_grasp", "hypernetwork_grasp", "species_conditioned_grasp")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_species(species: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    values = sorted(str(s) for s in np.unique(species.astype(str)))
    mapping = {name: i for i, name in enumerate(values)}
    return np.asarray([mapping[str(s)] for s in species.astype(str)], dtype=np.int64), mapping


def count_obs_and_classes(y: np.ndarray) -> tuple[int, int, int, bool]:
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y) & ((y == 0) | (y == 1))
    obs = y[valid]
    n = int(obs.size)
    n0 = int(np.sum(obs == 0))
    n1 = int(np.sum(obs == 1))
    return n, n0, n1, bool(n0 > 0 and n1 > 0)


def select_valid_antibiotics_legacy(
    source_amr_train: np.ndarray,
    source_amr_val: np.ndarray,
    ft_amr: np.ndarray,
    test_amr: np.ndarray,
    min_source_obs: int,
    min_source_val_obs: int,
    min_finetune_obs: int,
    min_test_obs: int,
) -> list[int]:
    valid_cols: list[int] = []
    for j in range(source_amr_train.shape[1]):
        n_tr, _, _, tr_two = count_obs_and_classes(source_amr_train[:, j])
        n_val, _, _, val_two = count_obs_and_classes(source_amr_val[:, j])
        n_ft, _, _, ft_two = count_obs_and_classes(ft_amr[:, j])
        n_test, _, _, test_two = count_obs_and_classes(test_amr[:, j])
        if n_tr < min_source_obs or not tr_two:
            continue
        if n_val < min_source_val_obs or not val_two:
            continue
        if n_ft < min_finetune_obs or not ft_two:
            continue
        if n_test < min_test_obs or not test_two:
            continue
        valid_cols.append(j)
    return valid_cols


def split_train_val(n_samples: int, seed: int, val_size: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_samples)
    if n_samples < 5:
        return idx, idx
    train_idx, val_idx = train_test_split(idx, test_size=val_size, random_state=seed, shuffle=True)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def split_ood_by_species(
    species_codes: np.ndarray,
    amr: np.ndarray,
    finetune_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_ft: list[int] = []
    all_test: list[int] = []
    for sp_id in sorted(np.unique(species_codes)):
        global_idx = np.where(species_codes == sp_id)[0]
        amr_sp = amr[global_idx]
        n = len(global_idx)
        if n < 5:
            all_test.extend(global_idx.tolist())
            continue

        observed_counts = np.sum(np.isfinite(amr_sp), axis=1)
        resistance_mean = np.nanmean(amr_sp, axis=1)
        resistance_mean = np.nan_to_num(resistance_mean, nan=-1.0)
        try:
            obs_bin = pd.qcut(observed_counts, q=min(4, len(np.unique(observed_counts))), labels=False, duplicates="drop")
        except Exception:
            obs_bin = np.zeros(n, dtype=int)
        try:
            res_bin = pd.qcut(resistance_mean, q=min(4, len(np.unique(resistance_mean))), labels=False, duplicates="drop")
            strat = np.asarray([f"{a}_{b}" for a, b in zip(obs_bin, res_bin)], dtype=object)
        except Exception:
            strat = np.asarray(obs_bin).astype(str)

        try:
            ft_local, test_local = train_test_split(
                np.arange(n),
                train_size=finetune_frac,
                random_state=seed + int(sp_id),
                shuffle=True,
                stratify=strat,
            )
        except Exception:
            ft_local, test_local = train_test_split(
                np.arange(n),
                train_size=finetune_frac,
                random_state=seed + int(sp_id),
                shuffle=True,
                stratify=None,
            )
        all_ft.extend(global_idx[ft_local].tolist())
        all_test.extend(global_idx[test_local].tolist())
    return np.asarray(all_ft, dtype=np.int64), np.asarray(all_test, dtype=np.int64)


def save_predictions(
    output_dir: Path,
    scenario: str,
    model_name: str,
    species_name: str,
    run_id: int,
    test_idx: np.ndarray,
    antibiotic_indices: list[int],
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> None:
    safe_species = species_name.replace(" ", "_").replace("/", "_")
    path = output_dir / "predictions" / model_name / f"{scenario}_{safe_species}_run_{run_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scenario=np.asarray(scenario),
        model_name=np.asarray(model_name),
        species_name=np.asarray(species_name),
        fold=np.asarray(run_id + 1),
        run=np.asarray(run_id),
        test_idx=test_idx,
        antibiotic_indices=np.asarray(antibiotic_indices),
        y_true=y_true,
        y_score=y_score,
    )


def species_rows(
    run_id: int,
    model_name: str,
    species_names: np.ndarray,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    test_species = species_names[test_idx]
    for species_name in sorted(np.unique(test_species)):
        mask = test_species == species_name
        y_sp = y_true[mask]
        before_sp = before[mask]
        after_sp = after[mask]
        rows.append(
            {
                "run": run_id,
                "model": model_name,
                "species": species_name,
                "n_test_samples": int(mask.sum()),
                "n_test_pairs": int(np.isfinite(y_sp).sum()),
                "micro_auc_before_finetuning": safe_auc(y_sp.reshape(-1), before_sp.reshape(-1)),
                "macro_auc_before_finetuning": macro_auc(y_sp, before_sp),
                "patient_auc_before_finetuning": patient_auc(y_sp, before_sp),
                "micro_auc_after_finetuning": safe_auc(y_sp.reshape(-1), after_sp.reshape(-1)),
                "macro_auc_after_finetuning": macro_auc(y_sp, after_sp),
                "patient_auc_after_finetuning": patient_auc(y_sp, after_sp),
            }
        )
    return rows


def run_legacy_global_experiment(
    ind: PicklePayload,
    ood: PicklePayload,
    output_dir: Path,
    exp_cfg: ExperimentConfig,
    train_cfg: TrainingConfig,
    grasp_cfg: GRASPConfig,
    device: torch.device,
    model_names: tuple[str, ...],
    source_val_size: float = 0.15,
    finetune_val_size: float = 0.2,
    finetune_lr_factor: float = 0.1,
    min_source_samples_species: int = 50,
    min_ood_samples_species: int = 50,
    min_source_obs_per_antibiotic: int = 50,
    min_source_val_obs_per_antibiotic: int = 5,
    min_finetune_obs_per_antibiotic: int = 10,
    min_test_obs_per_antibiotic: int = 10,
) -> pd.DataFrame:
    set_seed(exp_cfg.random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)

    common_species = sorted(set(ind.species.astype(str)).intersection(set(ood.species.astype(str))))
    kept_species = [
        species_name
        for species_name in common_species
        if int(np.sum(ind.species.astype(str) == species_name)) >= min_source_samples_species
        and int(np.sum(ood.species.astype(str) == species_name)) >= min_ood_samples_species
    ]
    if not kept_species:
        raise ValueError("No common species passed legacy sample-count filters.")
    ind_mask = np.isin(ind.species.astype(str), kept_species)
    ood_mask = np.isin(ood.species.astype(str), kept_species)
    ind = PicklePayload(X=ind.X[ind_mask], species=ind.species[ind_mask], amr=ind.amr[ind_mask], antibiotics=ind.antibiotics)
    ood = PicklePayload(X=ood.X[ood_mask], species=ood.species[ood_mask], amr=ood.amr[ood_mask], antibiotics=ood.antibiotics)

    ind_species_codes, species_mapping = encode_species(ind.species)
    ood_species_codes = np.asarray([species_mapping[str(s)] for s in ood.species.astype(str)], dtype=np.int64)
    id_to_species = {v: k for k, v in species_mapping.items()}
    model_names = tuple(m for m in model_names if m in LEGACY_GLOBAL_MODELS)
    if not model_names:
        raise ValueError(f"legacy_global_ood only supports: {LEGACY_GLOBAL_MODELS}")

    legacy_train_cfg = TrainingConfig(
        batch_size=train_cfg.batch_size,
        prediction_batch_size=train_cfg.prediction_batch_size,
        max_epochs=train_cfg.max_epochs,
        patience=train_cfg.patience,
        finetune_epochs=train_cfg.finetune_epochs,
        finetune_patience=train_cfg.finetune_patience,
        num_workers=train_cfg.num_workers,
        weight_decay=train_cfg.weight_decay,
        early_stopping_metric="loss",
        pin_memory=train_cfg.pin_memory,
    )
    legacy_ft_cfg = TrainingConfig(
        batch_size=train_cfg.batch_size,
        prediction_batch_size=train_cfg.prediction_batch_size,
        max_epochs=train_cfg.finetune_epochs,
        patience=train_cfg.finetune_patience,
        finetune_epochs=train_cfg.finetune_epochs,
        finetune_patience=train_cfg.finetune_patience,
        num_workers=train_cfg.num_workers,
        weight_decay=train_cfg.weight_decay,
        early_stopping_metric="loss",
        pin_memory=train_cfg.pin_memory,
    )

    with (output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "protocol": "legacy_global_ood",
                "experiment": asdict(exp_cfg),
                "training": asdict(legacy_train_cfg),
                "grasp": asdict(grasp_cfg),
                "device": str(device),
                "models": list(model_names),
                "species_mapping": species_mapping,
                "antibiotics": ind.antibiotics,
                "source_val_size": source_val_size,
                "finetune_val_size": finetune_val_size,
                "finetune_lr_factor": finetune_lr_factor,
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

    global_rows: list[dict[str, object]] = []
    all_species_rows: list[dict[str, object]] = []

    for model_name in model_names:
        for run_id in range(exp_cfg.n_folds):
            seed = exp_cfg.random_seed + run_id * 1000
            source_train_idx, source_val_idx = split_train_val(ind.X.shape[0], seed, source_val_size)
            ft_idx, test_idx = split_ood_by_species(ood_species_codes, ood.amr, exp_cfg.ood_adaptation_fraction, seed)
            if ft_idx.size == 0 or test_idx.size == 0:
                continue
            valid_cols = select_valid_antibiotics_legacy(
                source_amr_train=ind.amr[source_train_idx],
                source_amr_val=ind.amr[source_val_idx],
                ft_amr=ood.amr[ft_idx],
                test_amr=ood.amr[test_idx],
                min_source_obs=min_source_obs_per_antibiotic,
                min_source_val_obs=min_source_val_obs_per_antibiotic,
                min_finetune_obs=min_finetune_obs_per_antibiotic,
                min_test_obs=min_test_obs_per_antibiotic,
            )
            if not valid_cols:
                continue

            model = build_model(model_name, ind.X.shape[1], len(valid_cols), len(species_mapping), grasp_cfg)
            model, history = train_grasp_with_early_stopping(
                model,
                GlobalInteractionDataset(ind.X, ind_species_codes, ind.amr, source_train_idx, valid_cols),
                GlobalInteractionDataset(ind.X, ind_species_codes, ind.amr, source_val_idx, valid_cols),
                interaction_loss,
                grasp_cfg.optimizer,
                grasp_cfg.learning_rate,
                legacy_train_cfg,
                device,
            )
            y_true = ood.amr[np.ix_(test_idx, valid_cols)]
            before = predict_grasp(
                model,
                ood.X,
                ood_species_codes,
                test_idx,
                len(valid_cols),
                device,
                batch_size=train_cfg.prediction_batch_size,
            )

            ft_train_local, ft_val_local = split_train_val(len(ft_idx), seed + 2000, finetune_val_size)
            ft_train_idx = ft_idx[ft_train_local]
            ft_val_idx = ft_idx[ft_val_local]
            ft_model = copy.deepcopy(model)
            ft_model, ft_history = train_grasp_with_early_stopping(
                ft_model,
                GlobalInteractionDataset(ood.X, ood_species_codes, ood.amr, ft_train_idx, valid_cols),
                GlobalInteractionDataset(ood.X, ood_species_codes, ood.amr, ft_val_idx, valid_cols),
                interaction_loss,
                grasp_cfg.optimizer,
                grasp_cfg.learning_rate * finetune_lr_factor,
                legacy_ft_cfg,
                device,
            )
            after = predict_grasp(
                ft_model,
                ood.X,
                ood_species_codes,
                test_idx,
                len(valid_cols),
                device,
                batch_size=train_cfg.prediction_batch_size,
            )

            for sp_id in sorted(np.unique(ood_species_codes[test_idx])):
                sp_name = id_to_species[int(sp_id)]
                sp_idx = test_idx[ood_species_codes[test_idx] == sp_id]
                local_mask = ood_species_codes[test_idx] == sp_id
                save_predictions(
                    output_dir,
                    "zero_shot",
                    model_name,
                    sp_name,
                    run_id,
                    sp_idx,
                    valid_cols,
                    y_true[local_mask],
                    before[local_mask],
                )
                save_predictions(
                    output_dir,
                    "finetuned",
                    model_name,
                    sp_name,
                    run_id,
                    sp_idx,
                    valid_cols,
                    y_true[local_mask],
                    after[local_mask],
                )

            micro_before = safe_auc(y_true.reshape(-1), before.reshape(-1))
            macro_before = macro_auc(y_true, before)
            micro_after = safe_auc(y_true.reshape(-1), after.reshape(-1))
            macro_after = macro_auc(y_true, after)
            global_rows.append(
                {
                    "run": run_id,
                    "model": model_name,
                    "n_species": int(len(species_mapping)),
                    "n_source_samples": int(ind.X.shape[0]),
                    "n_ood_samples": int(ood.X.shape[0]),
                    "n_source_train_samples": int(len(source_train_idx)),
                    "n_source_val_samples": int(len(source_val_idx)),
                    "n_finetune_samples": int(len(ft_idx)),
                    "n_test_samples": int(len(test_idx)),
                    "n_valid_antibiotics": int(len(valid_cols)),
                    "n_test_pairs": int(np.isfinite(y_true).sum()),
                    "best_source_epoch": history.get("best_epoch", np.nan),
                    "best_source_val_loss": history.get("best_val_loss", np.nan),
                    "best_finetune_epoch": ft_history.get("best_epoch", np.nan),
                    "best_finetune_val_loss": ft_history.get("best_val_loss", np.nan),
                    "micro_auc_before_finetuning": micro_before,
                    "macro_auc_before_finetuning": macro_before,
                    "patient_auc_before_finetuning": patient_auc(y_true, before),
                    "micro_auc_after_finetuning": micro_after,
                    "macro_auc_after_finetuning": macro_after,
                    "patient_auc_after_finetuning": patient_auc(y_true, after),
                    "delta_micro_auc_after_minus_before": micro_after - micro_before
                    if np.isfinite(micro_after) and np.isfinite(micro_before)
                    else np.nan,
                    "delta_macro_auc_after_minus_before": macro_after - macro_before
                    if np.isfinite(macro_after) and np.isfinite(macro_before)
                    else np.nan,
                    "valid_antibiotics": ";".join(ind.antibiotics[j] for j in valid_cols),
                }
            )
            all_species_rows.extend(species_rows(run_id, model_name, ood.species.astype(str), test_idx, y_true, before, after))
            pd.DataFrame(global_rows).to_csv(output_dir / "legacy_global_results.csv", index=False)
            pd.DataFrame(all_species_rows).to_csv(output_dir / "legacy_species_results.csv", index=False)

    metrics = pd.DataFrame(global_rows)
    if not metrics.empty:
        summary = (
            metrics.groupby("model", as_index=False)[
                [
                    "micro_auc_before_finetuning",
                    "macro_auc_before_finetuning",
                    "patient_auc_before_finetuning",
                    "micro_auc_after_finetuning",
                    "macro_auc_after_finetuning",
                    "patient_auc_after_finetuning",
                ]
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.to_csv(output_dir / "legacy_global_summary.csv", index=False)
    write_reports(output_dir, ind.antibiotics)
    return metrics
