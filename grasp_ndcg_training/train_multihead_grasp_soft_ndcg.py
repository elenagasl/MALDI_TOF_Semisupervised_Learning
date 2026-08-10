#!/usr/bin/env python3
"""Train MultiHead GRASP with a differentiable clinical NDCG objective.

This script is intentionally separate from the original GRASP experiments. It
keeps the same pickle payload, folds, sparse antibiotic panel selection and
model implementation, but optimizes a listwise SoftNDCG surrogate instead of
binary cross-entropy on individual spectrum-antibiotic interactions.

Default model: multihead_grasp, chosen because it was the best GRASP individual
model at rank 1 in the finetuned soft clinical assessment.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
GRASPI_SRC = REPO_ROOT / "graspI" / "src"
CLINICAL_SRC = REPO_ROOT / "clinical_assesment"
sys.path.insert(0, str(GRASPI_SRC))
sys.path.insert(0, str(CLINICAL_SRC))

from generations import antibiotic_generation_metadata
from grasp_global.config import ExperimentConfig, GRASPConfig, TrainingConfig
from grasp_global.experiment import (
    DEFAULT_EXCLUDED_SPECIES,
    encode_species,
    filter_excluded_species,
    macro_auc,
    patient_auc,
    save_fold_predictions,
    set_seed,
)
from grasp_global.io import PicklePayload, load_combined_pickle
from grasp_global.metrics import safe_auc
from grasp_global.models import build_model
from grasp_global.panels import select_sparse_panel
from grasp_global.reporting import write_final_reports
from grasp_global.splits import make_global_folds
from grasp_global.training import get_device, make_optimizer, predict_grasp
from run_clinical_assessment import ANTIFUNGALS
from spectra import antibiotic_aware_category


LOCAL_PICKLE = (
    "/Users/elenagarciaarroyo/Documents/Máster/Generative AI "
    "/project_2/COMBINED_MARISMA_DRIAMS_samples.pkl"
)
SERVER_PICKLE = (
    "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/"
    "COMBINED_MARISMA_DRIAMS_samples.pkl"
)
MODEL_NAME = "multihead_grasp"


@dataclass(frozen=True)
class NDCGConfig:
    tau: float = 0.1
    ndcg_weight: float = 1.0
    bce_weight: float = 0.2
    generation_lambda: float = 0.2
    aware_lambda: float = 0.2


class SampleRankingDataset(Dataset):
    """One item per spectrum; labels are all selected antibiotics for that sample."""

    def __init__(
        self,
        X: np.ndarray,
        species_codes: np.ndarray,
        amr: np.ndarray,
        sample_indices: np.ndarray,
        antibiotic_indices: list[int],
    ) -> None:
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.antibiotic_indices = np.asarray(antibiotic_indices, dtype=np.int64)
        self.X = torch.as_tensor(X[self.sample_indices], dtype=torch.float32)
        self.species_idx = torch.as_tensor(species_codes[self.sample_indices], dtype=torch.long)
        y = amr[np.ix_(self.sample_indices, self.antibiotic_indices)]
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.observed = torch.isfinite(self.y)
        self.y = torch.nan_to_num(self.y, nan=0.0)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.species_idx[idx], self.y[idx], self.observed[idx]


def resolve_pickle(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    for candidate in (Path(SERVER_PICKLE), Path(LOCAL_PICKLE)):
        if candidate.exists():
            return candidate
    return Path(SERVER_PICKLE)


def antibiotic_penalties(
    antibiotic_indices: list[int],
    antibiotics: list[str],
    ndcg_cfg: NDCGConfig,
) -> np.ndarray:
    aware_cost = {"Access": 0.0, "Watch": 0.5, "Reserve": 1.0}
    penalties = []
    for idx in antibiotic_indices:
        antibiotic = antibiotics[int(idx)]
        generation = antibiotic_generation_metadata.get(antibiotic, {}).get("generation")
        generation_cost = 0.0 if generation is None else (float(generation) - 1.0) / 4.0
        category_cost = aware_cost.get(antibiotic_aware_category.get(antibiotic), 0.0)
        penalties.append(ndcg_cfg.generation_lambda * generation_cost + ndcg_cfg.aware_lambda * category_cost)
    return np.asarray(penalties, dtype=np.float32)


def forward_all_antibiotics(
    model: nn.Module,
    x: torch.Tensor,
    species_idx: torch.Tensor,
    n_antibiotics: int,
) -> torch.Tensor:
    batch_size = x.shape[0]
    x_rep = x[:, None, :].expand(batch_size, n_antibiotics, x.shape[1]).reshape(
        batch_size * n_antibiotics,
        x.shape[1],
    )
    species_rep = species_idx[:, None].expand(batch_size, n_antibiotics).reshape(-1)
    antibiotic_idx = torch.arange(n_antibiotics, device=x.device).repeat(batch_size)
    logits = model(x_rep, species_rep, antibiotic_idx)
    return logits.reshape(batch_size, n_antibiotics)


def ideal_dcg(relevance: torch.Tensor, observed: torch.Tensor, max_k: int) -> torch.Tensor:
    masked_relevance = relevance.masked_fill(~observed, -1.0)
    sorted_relevance = torch.sort(masked_relevance, descending=True, dim=1).values[:, :max_k]
    sorted_relevance = torch.clamp(sorted_relevance, min=0.0)
    discounts = 1.0 / torch.log2(
        torch.arange(2, max_k + 2, device=relevance.device, dtype=relevance.dtype)
    )
    return (sorted_relevance * discounts[None, :]).sum(dim=1)


def soft_ndcg_loss(
    logits_resistance: torch.Tensor,
    y: torch.Tensor,
    observed: torch.Tensor,
    penalties: torch.Tensor,
    tau: float,
    k_values: tuple[int, ...],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Differentiable NDCG surrogate using sigmoid pairwise soft ranks."""
    susceptibility = torch.sigmoid(-logits_resistance)
    predicted_utility = torch.clamp(susceptibility - penalties[None, :], min=0.0)
    relevance = torch.where(y == 0.0, torch.clamp(1.0 - penalties[None, :], min=0.0), torch.zeros_like(y))
    predicted_utility = predicted_utility.masked_fill(~observed, -1e6)
    relevance = relevance.masked_fill(~observed, 0.0)

    n_antibiotics = predicted_utility.shape[1]
    not_self = ~torch.eye(n_antibiotics, dtype=torch.bool, device=predicted_utility.device)
    valid_pair = observed[:, :, None] & observed[:, None, :] & not_self[None, :, :]
    pairwise = torch.sigmoid((predicted_utility[:, None, :] - predicted_utility[:, :, None]) / tau)
    soft_rank = 1.0 + pairwise.masked_fill(~valid_pair, 0.0).sum(dim=2)
    discounts = 1.0 / torch.log2(soft_rank + 1.0)
    max_k = max(k_values)
    idcg = ideal_dcg(relevance, observed, max_k).clamp_min(1e-8)

    metrics: dict[str, float] = {}
    losses = []
    for k in k_values:
        # Smooth top-k gate: positions with soft rank <= k contribute most.
        gate = torch.sigmoid((float(k) + 0.5 - soft_rank) / tau)
        dcg = (relevance * discounts * gate * observed.float()).sum(dim=1)
        ndcg = dcg / idcg
        valid = (observed.sum(dim=1) > 1) & (idcg > 1e-8)
        if valid.any():
            losses.append(1.0 - ndcg[valid].mean())
            metrics[f"soft_ndcg_at_{k}"] = float(ndcg[valid].detach().mean().cpu())
        else:
            metrics[f"soft_ndcg_at_{k}"] = float("nan")
    if not losses:
        return logits_resistance.sum() * 0.0, metrics
    return torch.stack(losses).mean(), metrics


def batch_loss(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    model: nn.Module,
    penalties: torch.Tensor,
    ndcg_cfg: NDCGConfig,
    k_values: tuple[int, ...],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    x, species_idx, y, observed = batch
    x = x.to(device, non_blocking=True)
    species_idx = species_idx.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    observed = observed.to(device, non_blocking=True)
    penalties = penalties.to(device, non_blocking=True)

    logits = forward_all_antibiotics(model, x, species_idx, penalties.shape[0])
    ndcg_loss, metrics = soft_ndcg_loss(
        logits,
        y,
        observed,
        penalties,
        tau=ndcg_cfg.tau,
        k_values=k_values,
    )
    bce = nn.functional.binary_cross_entropy_with_logits(logits[observed], y[observed]) if observed.any() else logits.sum() * 0.0
    total = ndcg_cfg.ndcg_weight * ndcg_loss + ndcg_cfg.bce_weight * bce
    metrics["loss"] = float(total.detach().cpu())
    metrics["soft_ndcg_loss"] = float(ndcg_loss.detach().cpu())
    metrics["bce_loss"] = float(bce.detach().cpu())
    return total, metrics


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    penalties: torch.Tensor,
    ndcg_cfg: NDCGConfig,
    k_values: tuple[int, ...],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    weights: list[int] = []
    for batch in loader:
        _, _, _, observed = batch
        _, metrics = batch_loss(batch, model, penalties, ndcg_cfg, k_values, device)
        rows.append(metrics)
        weights.append(int(observed.sum().item()))
    if not rows:
        return {"loss": float("inf"), **{f"soft_ndcg_at_{k}": float("nan") for k in k_values}}
    weights_array = np.asarray(weights, dtype=float)
    result = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        valid = np.isfinite(values)
        result[key] = float(np.average(values[valid], weights=weights_array[valid])) if valid.any() else float("nan")
    return result


def train_soft_ndcg(
    model: nn.Module,
    train_dataset: SampleRankingDataset,
    val_dataset: SampleRankingDataset,
    penalties: np.ndarray,
    train_cfg: TrainingConfig,
    grasp_cfg: GRASPConfig,
    ndcg_cfg: NDCGConfig,
    k_values: tuple[int, ...],
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    model = model.to(device)
    penalties_tensor = torch.as_tensor(penalties, dtype=torch.float32)
    optimizer = make_optimizer(
        model,
        name=grasp_cfg.optimizer,
        learning_rate=grasp_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory and device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory and device.type == "cuda",
    )

    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = batch_loss(batch, model, penalties_tensor, ndcg_cfg, k_values, device)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_loader(model, val_loader, penalties_tensor, ndcg_cfg, k_values, device)
        score = val_metrics.get("soft_ndcg_at_5", float("nan"))
        if math.isnan(score):
            score = -val_metrics["loss"]
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"epoch={epoch:04d} val_soft_ndcg@5={val_metrics.get('soft_ndcg_at_5', float('nan')):.4f} "
            f"val_loss={val_metrics.get('loss', float('nan')):.4f}"
        )
        if bad_epochs >= train_cfg.patience:
            break

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "early_stopping_metric": "soft_ndcg_at_5",
        **{f"best_val_{key}": value for key, value in best_metrics.items()},
    }


def clinical_ndcg_from_predictions(
    y_true: np.ndarray,
    y_score_resistance: np.ndarray,
    penalties: np.ndarray,
    k_values: tuple[int, ...],
) -> dict[int, float]:
    values: dict[int, list[float]] = {k: [] for k in k_values}
    for row in range(y_true.shape[0]):
        observed = np.isfinite(y_true[row]) & np.isfinite(y_score_resistance[row])
        if not observed.any():
            continue
        truth = y_true[row, observed]
        row_penalties = penalties[observed]
        relevance = np.where(truth == 0.0, np.maximum(0.0, 1.0 - row_penalties), 0.0)
        if not np.any(relevance > 0):
            continue
        score = np.maximum(0.0, 1.0 - y_score_resistance[row, observed] - row_penalties)
        order = np.argsort(-score, kind="stable")
        ideal = np.argsort(-relevance, kind="stable")
        for k in k_values:
            limit = min(k, len(relevance))
            discounts = 1.0 / np.log2(np.arange(2, limit + 2, dtype=float))
            dcg = float(np.sum(relevance[order[:limit]] * discounts))
            idcg = float(np.sum(relevance[ideal[:limit]] * discounts))
            if idcg > 0:
                values[k].append(dcg / idcg)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in values.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MultiHead GRASP with SoftNDCG.")
    parser.add_argument("--pickle", default=None, help="Combined in-distribution pickle. Defaults to server path, then local fallback.")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "multihead_grasp_soft_ndcg")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-train-samples", type=int, default=100)
    parser.add_argument("--min-val-samples", type=int, default=50)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32, help="Sample-level batch size. Lower than BCE because each batch expands over antibiotics.")
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--ndcg-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.2, help="Small BCE term stabilizes probability calibration.")
    parser.add_argument("--generation-lambda", type=float, default=0.2)
    parser.add_argument("--aware-lambda", type=float, default=0.2)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--exclude-species", nargs="*", default=list(DEFAULT_EXCLUDED_SPECIES))
    parser.add_argument("--evaluation-panel", choices=["global"], default="global")
    return parser.parse_args()


def run() -> pd.DataFrame:
    args = parse_args()
    pickle_path = resolve_pickle(args.pickle)
    payload = load_combined_pickle(pickle_path)
    payload = filter_excluded_species(payload, tuple(args.exclude_species))
    device = get_device(args.device)
    set_seed(args.random_seed)

    exp_cfg = ExperimentConfig(
        n_folds=args.n_folds,
        random_seed=args.random_seed,
        min_train_samples=args.min_train_samples,
        min_val_samples=args.min_val_samples,
        val_size=args.val_size,
    )
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        prediction_batch_size=args.prediction_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        weight_decay=args.weight_decay,
        early_stopping_metric="soft_ndcg_at_5",
        num_workers=args.num_workers,
    )
    grasp_cfg = GRASPConfig(dropout=args.dropout, learning_rate=args.learning_rate)
    ndcg_cfg = NDCGConfig(
        tau=args.tau,
        ndcg_weight=args.ndcg_weight,
        bce_weight=args.bce_weight,
        generation_lambda=args.generation_lambda,
        aware_lambda=args.aware_lambda,
    )
    k_values = tuple(sorted(set(args.k)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "pickle": str(pickle_path),
                "model": MODEL_NAME,
                "objective": "soft_clinical_ndcg",
                "experiment": asdict(exp_cfg),
                "training": asdict(train_cfg),
                "grasp": asdict(grasp_cfg),
                "ndcg": asdict(ndcg_cfg),
                "k": list(k_values),
                "device": str(device),
                "excluded_species": list(args.exclude_species),
                "evaluation_panel": args.evaluation_panel,
            },
            handle,
            indent=2,
        )

    species_codes, species_mapping = encode_species(payload.species)
    rows: list[dict[str, object]] = []
    for split in make_global_folds(payload.species, exp_cfg.n_folds, exp_cfg.val_size, exp_cfg.random_seed):
        antibiotic_indices = select_sparse_panel(
            payload.amr,
            split.train_idx,
            split.val_idx,
            exp_cfg.min_train_samples,
            exp_cfg.min_val_samples,
        )
        antibiotic_indices = [
            idx for idx in antibiotic_indices if payload.antibiotics[int(idx)] not in ANTIFUNGALS
        ]
        if not antibiotic_indices:
            continue
        penalties = antibiotic_penalties(antibiotic_indices, payload.antibiotics, ndcg_cfg)
        model = build_model(
            model_name=MODEL_NAME,
            input_dim=payload.X.shape[1],
            n_antibiotics=len(antibiotic_indices),
            n_species=len(species_mapping),
            cfg=grasp_cfg,
        )
        model, history = train_soft_ndcg(
            model=model,
            train_dataset=SampleRankingDataset(payload.X, species_codes, payload.amr, split.train_idx, antibiotic_indices),
            val_dataset=SampleRankingDataset(payload.X, species_codes, payload.amr, split.val_idx, antibiotic_indices),
            penalties=penalties,
            train_cfg=train_cfg,
            grasp_cfg=grasp_cfg,
            ndcg_cfg=ndcg_cfg,
            k_values=k_values,
            device=device,
        )

        species_array = payload.species.astype(str)
        for species_name in sorted(np.unique(species_array[split.test_idx])):
            species_test_idx = split.test_idx[species_array[split.test_idx] == species_name]
            y_true = payload.amr[np.ix_(species_test_idx, antibiotic_indices)]
            y_score = predict_grasp(
                model,
                payload.X,
                species_codes,
                species_test_idx,
                n_antibiotics=len(antibiotic_indices),
                device=device,
                batch_size=train_cfg.prediction_batch_size,
            )
            save_fold_predictions(
                args.output_dir,
                MODEL_NAME,
                species_name,
                split.fold,
                species_test_idx,
                antibiotic_indices,
                y_true,
                y_score,
            )
            ndcg = clinical_ndcg_from_predictions(y_true, y_score, penalties, k_values)
            row = {
                "model": MODEL_NAME,
                "objective": "soft_clinical_ndcg",
                "species": species_name,
                "fold": split.fold,
                "n_train_samples": len(split.train_idx),
                "n_val_samples": len(split.val_idx),
                "n_test_samples": len(species_test_idx),
                "n_antibiotics": len(antibiotic_indices),
                "best_epoch": history["best_epoch"],
                "best_val_soft_ndcg_at_5": history.get("best_val_soft_ndcg_at_5"),
                "best_val_loss": history.get("best_val_loss"),
                "patient_auc": patient_auc(y_true, y_score),
                "micro_auc": safe_auc(y_true.reshape(-1), y_score.reshape(-1)),
                "macro_auc": macro_auc(y_true, y_score),
            }
            row.update({f"clinical_ndcg_at_{k}": value for k, value in ndcg.items()})
            rows.append(row)
        metrics = pd.DataFrame(rows)
        metrics.to_csv(args.output_dir / "metrics_by_fold.csv", index=False)

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metric_cols = [
            "patient_auc",
            "micro_auc",
            "macro_auc",
            *[f"clinical_ndcg_at_{k}" for k in k_values],
        ]
        summary = metrics.groupby("model", as_index=False)[metric_cols].agg(["mean", "std"]).reset_index()
        summary.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    write_final_reports(args.output_dir, payload.antibiotics)
    print(f"Results written to {args.output_dir.resolve()}")
    return metrics


if __name__ == "__main__":
    run()
