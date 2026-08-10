#!/usr/bin/env python3
"""Wide-panel OOD clinical assessment.

The ideal ranking is built on the broad antibiotic panel available in the
global GRASP-style models. Baselines are evaluated on that same ideal panel;
missing model predictions are kept as non-recommendable candidates, so limited
coverage is penalized directly in NDCG.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from generations import antibiotic_generation_metadata
from run_clinical_assessment import ANTIFUNGALS, _load_antibiotics
from spectra import antibiotic_aware_category


GLOBAL_REFERENCE_MODEL = "species_conditioned_grasp"
GLOBAL_MODELS = (
    "species_conditioned_grasp",
    "multihead_grasp",
    "hypernetwork_grasp",
    "species_aware_global_mlp",
)
BASELINE_MODELS = (
    "species_recommender",
    "semisupervised_multioutput_mlp",
    "semisupervised_binary_mlp",
)
MODEL_LABELS = {
    "species_conditioned_grasp": "GRASP species-conditioned",
    "multihead_grasp": "GRASP multihead",
    "hypernetwork_grasp": "GRASP hypernetwork",
    "species_aware_global_mlp": "Species-aware global MLP",
    "ensemble_grasp": "GRASP ensemble (3)",
    "ensemble_global": "Global ensemble (4)",
    "species_recommender": "Species-specific recommender",
    "semisupervised_multioutput_mlp": "Semisupervised multioutput MLP",
    "semisupervised_binary_mlp": "Semisupervised binary MLP",
}
MODEL_GROUPS = {
    "species_conditioned_grasp": "global_grasp",
    "multihead_grasp": "global_grasp",
    "hypernetwork_grasp": "global_grasp",
    "species_aware_global_mlp": "global_mlp",
    "ensemble_grasp": "global_grasp_ensemble",
    "ensemble_global": "global_ensemble",
    "species_recommender": "baseline",
    "semisupervised_multioutput_mlp": "baseline",
    "semisupervised_binary_mlp": "baseline",
}
EVALUATED_SYSTEMS = (
    "species_conditioned_grasp",
    "multihead_grasp",
    "hypernetwork_grasp",
    "species_aware_global_mlp",
    "species_recommender",
    "semisupervised_multioutput_mlp",
    "semisupervised_binary_mlp",
)
ENSEMBLES = {
    "ensemble_grasp": ("species_conditioned_grasp", "multihead_grasp", "hypernetwork_grasp"),
    "ensemble_global": GLOBAL_MODELS,
}
SCENARIOS = ("zero_shot", "finetuned")
K_VALUES = (1, 3, 5, 10)
PREDICTED_RESISTANT_SCORE = -1_000_000.0
MISSING_PREDICTION_SCORE = -2_000_000.0


@dataclass(frozen=True)
class Config:
    results_root: Path
    output_dir: Path
    scenarios: tuple[str, ...]
    k_values: tuple[int, ...]
    target_fsr: float
    uncertainty_lambda: float
    generation_lambda: float
    aware_lambda: float


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_results = here.parent / "ood_evaluation" / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=default_results)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "results" / "wide_panel_clinical_assessment",
    )
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--k", nargs="+", type=int, default=list(K_VALUES))
    parser.add_argument("--target-fsr", type=float, default=0.01)
    parser.add_argument("--uncertainty-lambda", type=float, default=0.0)
    parser.add_argument("--generation-lambda", type=float, default=0.2)
    parser.add_argument("--aware-lambda", type=float, default=0.2)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Config:
    scenarios = tuple(dict.fromkeys(args.scenarios))
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
    k_values = tuple(sorted(set(args.k)))
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("All K values must be positive.")
    if not 0 <= args.target_fsr <= 1:
        raise ValueError("--target-fsr must be within [0, 1].")
    if min(args.uncertainty_lambda, args.generation_lambda, args.aware_lambda) < 0:
        raise ValueError("Penalty lambdas must be non-negative.")
    return Config(
        results_root=args.results_root,
        output_dir=args.output_dir,
        scenarios=scenarios,
        k_values=k_values,
        target_fsr=float(args.target_fsr),
        uncertainty_lambda=float(args.uncertainty_lambda),
        generation_lambda=float(args.generation_lambda),
        aware_lambda=float(args.aware_lambda),
    )


def model_prediction_dir(results_root: Path, model: str) -> Path:
    if model == "species_aware_global_mlp":
        return results_root / "ood_deployment_grasp_mlp" / "predictions" / model
    return results_root / "ood_deployment_all_models" / "predictions" / model


def prediction_path(results_root: Path, model: str, scenario: str, species: str, fold: int) -> Path:
    return model_prediction_dir(results_root, model) / f"{scenario}_{species}_fold_{fold}.npz"


def available_species(results_root: Path, scenarios: tuple[str, ...]) -> tuple[str, ...]:
    root = model_prediction_dir(results_root, GLOBAL_REFERENCE_MODEL)
    species: set[str] = set()
    pattern = re.compile(r"(zero_shot|finetuned)_(.+)_fold_(\d+)\.npz$")
    for path in root.glob("*.npz"):
        match = pattern.match(path.name)
        if match and match.group(1) in scenarios:
            species.add(match.group(2))
    return tuple(sorted(species))


def load_block(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {
            "test_idx": np.array(data["test_idx"], dtype=int),
            "antibiotic_indices": np.array(data["antibiotic_indices"], dtype=int),
            "y_true": np.array(data["y_true"], dtype=float),
            "y_score": np.array(data["y_score"], dtype=float),
        }


def clinical_penalty(antibiotic: str, config: Config) -> float:
    generation = antibiotic_generation_metadata.get(antibiotic, {}).get("generation")
    generation_cost = 0.0 if generation is None else (float(generation) - 1.0) / 4.0
    aware_cost = {"Access": 0.0, "Watch": 0.5, "Reserve": 1.0}.get(
        antibiotic_aware_category.get(antibiotic), 0.0
    )
    return config.generation_lambda * generation_cost + config.aware_lambda * aware_cost


def ndcg_at_k(relevance: np.ndarray, predicted_score: np.ndarray, k: int) -> float:
    if relevance.size == 0:
        return float("nan")
    limit = min(k, relevance.size)
    discounts = np.log2(np.arange(2, limit + 2, dtype=float))
    predicted_order = np.argsort(-predicted_score, kind="stable")[:limit]
    ideal_order = np.argsort(-relevance, kind="stable")[:limit]
    dcg = float(np.sum(relevance[predicted_order] / discounts))
    idcg = float(np.sum(relevance[ideal_order] / discounts))
    return dcg / idcg if idcg > 0 else float("nan")


def ndcg_values_from_order(
    relevance: np.ndarray,
    order: np.ndarray,
    ideal_order: np.ndarray,
    k_values: tuple[int, ...],
) -> dict[int, float]:
    limit = min(max(k_values), relevance.size)
    discounts = np.log2(np.arange(2, limit + 2, dtype=float))
    predicted_gain = np.cumsum(relevance[order[:limit]] / discounts)
    ideal_gain = np.cumsum(relevance[ideal_order[:limit]] / discounts)
    values: dict[int, float] = {}
    for k in k_values:
        idx = min(k, relevance.size) - 1
        values[k] = float(predicted_gain[idx] / ideal_gain[idx]) if ideal_gain[idx] > 0 else float("nan")
    return values


def fsr_threshold(y_true: list[np.ndarray], y_score: list[np.ndarray], target_fsr: float) -> float:
    if not y_true:
        return float("nan")
    truth = np.concatenate(y_true)
    score = np.concatenate(y_score)
    resistant_scores = np.sort(score[(truth == 1.0) & np.isfinite(score)])
    if resistant_scores.size == 0:
        return float("nan")
    allowed_false_susceptible = int(math.floor(target_fsr * resistant_scores.size))
    allowed_false_susceptible = min(allowed_false_susceptible, resistant_scores.size - 1)
    return float(resistant_scores[allowed_false_susceptible])


def threshold_diagnostics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    truth = y_true[valid]
    score = y_score[valid]
    predicted_resistant = score >= threshold
    true_resistant = truth == 1.0
    true_susceptible = truth == 0.0
    tp = int(np.sum(predicted_resistant & true_resistant))
    fn = int(np.sum(~predicted_resistant & true_resistant))
    tn = int(np.sum(~predicted_resistant & true_susceptible))
    fp = int(np.sum(predicted_resistant & true_susceptible))
    return {
        "n_observed_predictions": int(valid.sum()),
        "n_true_resistant": tp + fn,
        "n_true_susceptible": tn + fp,
        "resistance_sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "susceptibility_specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "false_susceptible_rate": fn / (tp + fn) if tp + fn else float("nan"),
        "false_resistant_rate": fp / (tn + fp) if tn + fp else float("nan"),
    }


def align_to_reference(
    reference: dict[str, np.ndarray],
    block: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = reference["y_true"].shape
    score = np.full(shape, np.nan, dtype=float)
    truth = np.full(shape, np.nan, dtype=float)
    available = np.zeros(shape, dtype=bool)
    if block is None:
        return truth, score, available

    row_map = {int(idx): pos for pos, idx in enumerate(block["test_idx"])}
    col_map = {int(idx): pos for pos, idx in enumerate(block["antibiotic_indices"])}
    ref_rows = [row_map.get(int(idx)) for idx in reference["test_idx"]]
    ref_cols = [col_map.get(int(idx)) for idx in reference["antibiotic_indices"]]

    for ref_r, block_r in enumerate(ref_rows):
        if block_r is None:
            continue
        for ref_c, block_c in enumerate(ref_cols):
            if block_c is None:
                continue
            truth[ref_r, ref_c] = block["y_true"][block_r, block_c]
            score[ref_r, ref_c] = block["y_score"][block_r, block_c]
            available[ref_r, ref_c] = np.isfinite(score[ref_r, ref_c])
    return truth, score, available


def make_system_blocks(
    results_root: Path,
    scenario: str,
    species: str,
    fold: int,
    reference: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    loaded = {
        model: load_block(prediction_path(results_root, model, scenario, species, fold))
        for model in (*GLOBAL_MODELS, *BASELINE_MODELS)
    }
    aligned: dict[str, dict[str, np.ndarray]] = {}
    for model, block in loaded.items():
        truth, score, available = align_to_reference(reference, block)
        aligned[model] = {
            "y_true": truth,
            "mean_probability": score,
            "std_probability": np.zeros_like(score),
            "available": available,
        }

    for ensemble, members in ENSEMBLES.items():
        if ensemble not in EVALUATED_SYSTEMS:
            continue
        scores = [aligned[member]["mean_probability"] for member in members]
        stacked = np.stack(scores)
        available = np.isfinite(stacked).all(axis=0)
        mean = np.full(reference["y_true"].shape, np.nan, dtype=float)
        std = np.zeros(reference["y_true"].shape, dtype=float)
        mean[available] = stacked[:, available].mean(axis=0)
        std[available] = stacked[:, available].std(axis=0)
        aligned[ensemble] = {
            "y_true": reference["y_true"],
            "mean_probability": mean,
            "std_probability": std,
            "available": available,
        }
    return aligned


def collect_threshold_inputs(
    config: Config,
    antibiotics: list[str],
    species_names: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], float]]:
    truth_by_key: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    score_by_key: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)

    for scenario in config.scenarios:
        for species in species_names:
            for fold in range(1, 6):
                reference = load_block(
                    prediction_path(config.results_root, GLOBAL_REFERENCE_MODEL, scenario, species, fold)
                )
                if reference is None:
                    continue
                systems = make_system_blocks(config.results_root, scenario, species, fold, reference)
                bacterial = np.array(
                    [
                        antibiotics[int(idx)] not in ANTIFUNGALS
                        for idx in reference["antibiotic_indices"]
                    ],
                    dtype=bool,
                )
                for system, block in systems.items():
                    observed = (
                        bacterial[None, :]
                        & block["available"]
                        & np.isfinite(reference["y_true"])
                        & np.isfinite(block["mean_probability"])
                    )
                    if observed.any():
                        key = (scenario, species, system)
                        truth_by_key[key].append(reference["y_true"][observed])
                        score_by_key[key].append(block["mean_probability"][observed])

    rows: list[dict[str, object]] = []
    thresholds: dict[tuple[str, str, str], float] = {}
    for scenario in config.scenarios:
        for species in species_names:
            for system in EVALUATED_SYSTEMS:
                key = (scenario, species, system)
                threshold = fsr_threshold(truth_by_key[key], score_by_key[key], config.target_fsr)
                thresholds[key] = threshold
                if truth_by_key[key]:
                    diagnostics = threshold_diagnostics(
                        np.concatenate(truth_by_key[key]),
                        np.concatenate(score_by_key[key]),
                        threshold,
                    )
                else:
                    diagnostics = {
                        "n_observed_predictions": 0,
                        "n_true_resistant": 0,
                        "n_true_susceptible": 0,
                        "resistance_sensitivity": float("nan"),
                        "susceptibility_specificity": float("nan"),
                        "false_susceptible_rate": float("nan"),
                        "false_resistant_rate": float("nan"),
                    }
                rows.append(
                    {
                        "scenario": scenario,
                        "species": species,
                        "system": system,
                        "system_label": MODEL_LABELS[system],
                        "model_group": MODEL_GROUPS[system],
                        "target_false_susceptible_rate": config.target_fsr,
                        "selected_resistance_threshold": threshold,
                        **diagnostics,
                    }
                )
    return pd.DataFrame(rows), thresholds


def evaluate(
    config: Config,
    antibiotics: list[str],
    species_names: tuple[str, ...],
    thresholds: dict[tuple[str, str, str], float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_counts: dict[tuple, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    rank_counts: dict[tuple, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))

    for scenario in config.scenarios:
        for species in species_names:
            for fold in range(1, 6):
                reference = load_block(
                    prediction_path(config.results_root, GLOBAL_REFERENCE_MODEL, scenario, species, fold)
                )
                if reference is None:
                    continue
                systems = make_system_blocks(config.results_root, scenario, species, fold, reference)
                antibiotic_names = [antibiotics[int(idx)] for idx in reference["antibiotic_indices"]]
                bacterial = np.array([name not in ANTIFUNGALS for name in antibiotic_names], dtype=bool)
                penalties = np.array(
                    [clinical_penalty(name, config) for name in antibiotic_names],
                    dtype=float,
                )

                for row_idx in range(reference["y_true"].shape[0]):
                    observed = bacterial & np.isfinite(reference["y_true"][row_idx])
                    if not observed.any():
                        continue
                    truth = reference["y_true"][row_idx, observed]
                    row_penalties = penalties[observed]
                    ideal_utility = np.where(
                        truth == 0.0,
                        np.maximum(0.0, 1.0 - row_penalties),
                        0.0,
                    )
                    if not np.any(ideal_utility > 0):
                        continue
                    ideal_order = np.argsort(-ideal_utility, kind="stable")

                    for system, block in systems.items():
                        threshold = thresholds[(scenario, species, system)]
                        available = block["available"][row_idx, observed]
                        p_res = block["mean_probability"][row_idx, observed]
                        p_std = block["std_probability"][row_idx, observed]

                        raw_score = np.full_like(ideal_utility, MISSING_PREDICTION_SCORE)
                        safety_score = np.full_like(ideal_utility, MISSING_PREDICTION_SCORE)
                        clinical_score = np.full_like(ideal_utility, MISSING_PREDICTION_SCORE)
                        raw_score[available] = 1.0 - p_res[available]
                        safety_score[available] = (
                            1.0 - p_res[available] - config.uncertainty_lambda * p_std[available]
                        )
                        predicted_susceptible = np.zeros_like(available, dtype=bool)
                        if np.isfinite(threshold):
                            predicted_susceptible[available] = p_res[available] < threshold
                        clinical_score[predicted_susceptible] = np.maximum(
                            0.0,
                            safety_score[predicted_susceptible] - row_penalties[predicted_susceptible],
                        )
                        clinical_score[available & ~predicted_susceptible] = PREDICTED_RESISTANT_SCORE

                        clinical_order = np.argsort(-clinical_score, kind="stable")
                        raw_order = np.argsort(-raw_score, kind="stable")
                        ndcg_by_ranking = {
                            "raw": ndcg_values_from_order(
                                ideal_utility, raw_order, ideal_order, config.k_values
                            ),
                            "clinical_reranked": ndcg_values_from_order(
                                ideal_utility, clinical_order, ideal_order, config.k_values
                            ),
                        }
                        for ranking, values in ndcg_by_ranking.items():
                            for k, ndcg in values.items():
                                if np.isfinite(ndcg):
                                    key = (scenario, species, fold, system, ranking, k)
                                    metric_counts[key]["sum_ndcg"] += float(ndcg)
                                    metric_counts[key]["n_evaluable_samples"] += 1
                                    metric_counts[key]["sum_candidate_antibiotics"] += int(observed.sum())
                                    metric_counts[key]["sum_available_predictions"] += int(available.sum())
                                    metric_counts[key]["sum_coverage"] += float(available.sum() / observed.sum())

                        for rank in (1, 2, 3):
                            for ranking, order, score in (
                                ("raw", raw_order, raw_score),
                                ("clinical_reranked", clinical_order, clinical_score),
                            ):
                                if order.size < rank:
                                    continue
                                pos = int(order[rank - 1])
                                key = (scenario, species, fold, system, ranking, rank)
                                rank_counts[key]["n_ranked"] += 1
                                rank_counts[key]["has_prediction"] += int(bool(available[pos]))
                                rank_counts[key]["true_resistant"] += int(bool(truth[pos] == 1.0))
                                rank_counts[key]["true_susceptible"] += int(bool(truth[pos] == 0.0))
                                rank_counts[key]["predicted_susceptible"] += int(
                                    bool(predicted_susceptible[pos])
                                )

    metric_rows: list[dict[str, object]] = []
    for key, counts in metric_counts.items():
        scenario, species, fold, system, ranking, k = key
        n = counts["n_evaluable_samples"]
        metric_rows.append(
            {
                "scenario": scenario,
                "species": species,
                "fold": fold,
                "system": system,
                "system_label": MODEL_LABELS[system],
                "model_group": MODEL_GROUPS[system],
                "ranking": ranking,
                "k": int(k),
                "mean_ndcg": counts["sum_ndcg"] / n,
                "n_evaluable_samples": int(n),
                "mean_candidate_antibiotics": counts["sum_candidate_antibiotics"] / n,
                "mean_available_predictions": counts["sum_available_predictions"] / n,
                "mean_coverage": counts["sum_coverage"] / n,
            }
        )

    rank_rows: list[dict[str, object]] = []
    for key, counts in rank_counts.items():
        scenario, species, fold, system, ranking, rank = key
        n = counts["n_ranked"]
        rank_rows.append(
            {
                "scenario": scenario,
                "species": species,
                "fold": fold,
                "system": system,
                "system_label": MODEL_LABELS[system],
                "model_group": MODEL_GROUPS[system],
                "ranking": ranking,
                "rank": int(rank),
                "n_ranked": int(n),
                "has_prediction_pct": counts["has_prediction"] / n,
                "true_susceptible_pct": counts["true_susceptible"] / n,
                "true_resistant_pct": counts["true_resistant"] / n,
                "predicted_susceptible_pct": counts["predicted_susceptible"] / n,
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(rank_rows)


def summarise_metrics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    finite = metrics[np.isfinite(metrics["mean_ndcg"])].copy()
    finite["weighted_ndcg"] = finite["mean_ndcg"] * finite["n_evaluable_samples"]
    finite["weighted_candidate_antibiotics"] = (
        finite["mean_candidate_antibiotics"] * finite["n_evaluable_samples"]
    )
    finite["weighted_available_predictions"] = (
        finite["mean_available_predictions"] * finite["n_evaluable_samples"]
    )
    finite["weighted_coverage"] = finite["mean_coverage"] * finite["n_evaluable_samples"]
    by_species = (
        finite.groupby(
            ["scenario", "species", "system", "system_label", "model_group", "ranking", "k"],
            sort=False,
        )
        .agg(
            sum_ndcg=("weighted_ndcg", "sum"),
            n_evaluable_samples=("n_evaluable_samples", "sum"),
            sum_candidate_antibiotics=("weighted_candidate_antibiotics", "sum"),
            sum_available_predictions=("weighted_available_predictions", "sum"),
            sum_coverage=("weighted_coverage", "sum"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
    )
    by_species["mean_ndcg"] = by_species["sum_ndcg"] / by_species["n_evaluable_samples"]
    by_species["mean_candidate_antibiotics"] = (
        by_species["sum_candidate_antibiotics"] / by_species["n_evaluable_samples"]
    )
    by_species["mean_available_predictions"] = (
        by_species["sum_available_predictions"] / by_species["n_evaluable_samples"]
    )
    by_species["mean_coverage"] = by_species["sum_coverage"] / by_species["n_evaluable_samples"]
    by_species = by_species.drop(
        columns=[
            "sum_ndcg",
            "sum_candidate_antibiotics",
            "sum_available_predictions",
            "sum_coverage",
        ]
    )
    global_rows: list[dict[str, object]] = []
    for key, group in by_species.groupby(
        ["scenario", "system", "system_label", "model_group", "ranking", "k"],
        sort=False,
    ):
        weights = group["n_evaluable_samples"].to_numpy(dtype=float)
        global_rows.append(
            {
                **dict(
                    zip(
                        ["scenario", "system", "system_label", "model_group", "ranking", "k"],
                        key,
                    )
                ),
                "weighted_mean_ndcg": float(
                    np.average(group["mean_ndcg"].to_numpy(dtype=float), weights=weights)
                ),
                "n_evaluable_samples": int(weights.sum()),
                "mean_candidate_antibiotics": float(
                    np.average(group["mean_candidate_antibiotics"], weights=weights)
                ),
                "mean_available_predictions": float(
                    np.average(group["mean_available_predictions"], weights=weights)
                ),
                "mean_coverage": float(np.average(group["mean_coverage"], weights=weights)),
                "n_species": int(group["species"].nunique()),
            }
        )
    return by_species, pd.DataFrame(global_rows)


def summarise_ranks(ranks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranks = ranks.copy()
    for column in (
        "has_prediction_pct",
        "true_susceptible_pct",
        "true_resistant_pct",
        "predicted_susceptible_pct",
    ):
        ranks[f"weighted_{column}"] = ranks[column] * ranks["n_ranked"]
    by_species = (
        ranks.groupby(
            ["scenario", "species", "system", "system_label", "model_group", "ranking", "rank"],
            sort=False,
        )
        .agg(
            n_ranked=("n_ranked", "sum"),
            has_prediction=("weighted_has_prediction_pct", "sum"),
            true_susceptible=("weighted_true_susceptible_pct", "sum"),
            true_resistant=("weighted_true_resistant_pct", "sum"),
            predicted_susceptible=("weighted_predicted_susceptible_pct", "sum"),
        )
        .reset_index()
    )
    by_species["has_prediction_pct"] = by_species["has_prediction"] / by_species["n_ranked"]
    by_species["true_susceptible_pct"] = by_species["true_susceptible"] / by_species["n_ranked"]
    by_species["true_resistant_pct"] = by_species["true_resistant"] / by_species["n_ranked"]
    by_species["predicted_susceptible_pct"] = (
        by_species["predicted_susceptible"] / by_species["n_ranked"]
    )
    by_species = by_species.drop(
        columns=["has_prediction", "true_susceptible", "true_resistant", "predicted_susceptible"]
    )
    global_rows: list[dict[str, object]] = []
    for key, group in by_species.groupby(
        ["scenario", "system", "system_label", "model_group", "ranking", "rank"],
        sort=False,
    ):
        weights = group["n_ranked"].to_numpy(dtype=float)
        global_rows.append(
            {
                **dict(
                    zip(
                        ["scenario", "system", "system_label", "model_group", "ranking", "rank"],
                        key,
                    )
                ),
                "n_ranked": int(weights.sum()),
                "has_prediction_pct": float(np.average(group["has_prediction_pct"], weights=weights)),
                "true_susceptible_pct": float(np.average(group["true_susceptible_pct"], weights=weights)),
                "true_resistant_pct": float(np.average(group["true_resistant_pct"], weights=weights)),
                "predicted_susceptible_pct": float(
                    np.average(group["predicted_susceptible_pct"], weights=weights)
                ),
                "n_species": int(group["species"].nunique()),
            }
        )
    return by_species, pd.DataFrame(global_rows)


def summarise_thresholds(thresholds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in thresholds.groupby(
        ["scenario", "system", "system_label", "model_group"], sort=False
    ):
        n_r = group["n_true_resistant"].sum()
        n_s = group["n_true_susceptible"].sum()
        rows.append(
            {
                **dict(zip(["scenario", "system", "system_label", "model_group"], key)),
                "mean_selected_resistance_threshold": group["selected_resistance_threshold"].mean(),
                "n_observed_predictions": int(group["n_observed_predictions"].sum()),
                "n_true_resistant": int(n_r),
                "n_true_susceptible": int(n_s),
                "resistance_sensitivity": float(
                    np.average(group["resistance_sensitivity"].fillna(0.0), weights=group["n_true_resistant"])
                )
                if n_r
                else float("nan"),
                "susceptibility_specificity": float(
                    np.average(
                        group["susceptibility_specificity"].fillna(0.0),
                        weights=group["n_true_susceptible"],
                    )
                )
                if n_s
                else float("nan"),
                "false_susceptible_rate": float(
                    np.average(group["false_susceptible_rate"].fillna(0.0), weights=group["n_true_resistant"])
                )
                if n_r
                else float("nan"),
                "false_resistant_rate": float(
                    np.average(group["false_resistant_rate"].fillna(0.0), weights=group["n_true_susceptible"])
                )
                if n_s
                else float("nan"),
                "n_species_with_predictions": int((group["n_observed_predictions"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    global_metrics: pd.DataFrame,
    global_thresholds: pd.DataFrame,
    global_ranks: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
        if frame.empty:
            return "_No rows._"
        display = frame.copy()
        for column in display.columns:
            if pd.api.types.is_float_dtype(display[column]):
                display[column] = display[column].map(
                    lambda value: "" if pd.isna(value) else format(float(value), floatfmt)
                )
            else:
                display[column] = display[column].map(str)
        columns = list(display.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in display.itertuples(index=False):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    focus = global_metrics[
        (global_metrics["ranking"] == "clinical_reranked")
        & (global_metrics["k"].isin([1, 3, 5]))
    ].copy()
    focus = focus.sort_values(["scenario", "k", "weighted_mean_ndcg"], ascending=[True, True, False])
    raw = global_metrics[global_metrics["ranking"] == "raw"][
        ["scenario", "system", "k", "weighted_mean_ndcg"]
    ].rename(columns={"weighted_mean_ndcg": "raw_ndcg"})
    clinical = global_metrics[global_metrics["ranking"] == "clinical_reranked"][
        ["scenario", "system", "k", "weighted_mean_ndcg"]
    ].rename(columns={"weighted_mean_ndcg": "clinical_ndcg"})
    delta = clinical.merge(raw, on=["scenario", "system", "k"], how="left")
    delta["clinical_minus_raw"] = delta["clinical_ndcg"] - delta["raw_ndcg"]

    lines = [
        "# Wide-panel clinical assessment",
        "",
        "Ideal ranking: broad global-panel antibiotics with observed AST.",
        "Missing baseline predictions remain in the candidate set and are ranked last.",
        "Decision thresholds target false susceptible rate <= 1% among available predictions.",
        "",
        "## Clinical NDCG",
        "",
        markdown_table(focus[
            [
                "scenario",
                "system_label",
                "k",
                "weighted_mean_ndcg",
                "mean_coverage",
                "n_evaluable_samples",
                "n_species",
            ]
        ]),
        "",
        "## Threshold Diagnostics",
        "",
        markdown_table(global_thresholds[
            [
                "scenario",
                "system_label",
                "mean_selected_resistance_threshold",
                "false_susceptible_rate",
                "resistance_sensitivity",
                "susceptibility_specificity",
                "n_observed_predictions",
                "n_species_with_predictions",
            ]
        ]),
        "",
        "## Clinical Minus Raw NDCG",
        "",
        markdown_table(delta.merge(
            global_metrics[["scenario", "system", "system_label"]].drop_duplicates(),
            on=["scenario", "system"],
            how="left",
        )[["scenario", "system_label", "k", "raw_ndcg", "clinical_ndcg", "clinical_minus_raw"]]
        .sort_values(["scenario", "k", "clinical_ndcg"], ascending=[True, True, False])),
        "",
        "## Rank 1 Safety",
        "",
        markdown_table(global_ranks[
            (global_ranks["ranking"] == "clinical_reranked") & (global_ranks["rank"] == 1)
        ][
            [
                "scenario",
                "system_label",
                "has_prediction_pct",
                "true_susceptible_pct",
                "true_resistant_pct",
                "predicted_susceptible_pct",
                "n_ranked",
            ]
        ]),
        "",
    ]
    (output_dir / "WIDE_PANEL_CLINICAL_ASSESSMENT_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    config = validate_args(parse_args())
    antibiotics = _load_antibiotics(config.results_root)
    species_names = available_species(config.results_root, config.scenarios)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    thresholds_by_species, thresholds = collect_threshold_inputs(config, antibiotics, species_names)
    metrics, ranks = evaluate(config, antibiotics, species_names, thresholds)
    metrics_by_species, metrics_global = summarise_metrics(metrics)
    ranks_by_species, ranks_global = summarise_ranks(ranks)
    thresholds_global = summarise_thresholds(thresholds_by_species)

    thresholds_by_species.to_csv(config.output_dir / "fsr_1pct_thresholds_by_species.csv", index=False)
    thresholds_global.to_csv(config.output_dir / "fsr_1pct_thresholds_global.csv", index=False)
    metrics.to_csv(config.output_dir / "wide_panel_ndcg_by_fold.csv", index=False)
    metrics_by_species.to_csv(config.output_dir / "wide_panel_ndcg_by_species.csv", index=False)
    metrics_global.to_csv(config.output_dir / "wide_panel_ndcg_global.csv", index=False)
    ranks.to_csv(config.output_dir / "wide_panel_rank_metrics_by_fold.csv", index=False)
    ranks_by_species.to_csv(config.output_dir / "wide_panel_rank_metrics_by_species.csv", index=False)
    ranks_global.to_csv(config.output_dir / "wide_panel_rank_metrics_global.csv", index=False)
    write_report(config.output_dir, metrics_global, thresholds_global, ranks_global)

    run_config = {
        "results_root": str(config.results_root.resolve()),
        "species": list(species_names),
        "scenarios": list(config.scenarios),
        "systems": list(EVALUATED_SYSTEMS),
        "global_reference_model": GLOBAL_REFERENCE_MODEL,
        "ideal_panel_policy": "global reference model observed bacterial AST",
        "missing_prediction_policy": "kept in candidate set and ranked last",
        "target_false_susceptible_rate": config.target_fsr,
        "k": list(config.k_values),
        "uncertainty_lambda": config.uncertainty_lambda,
        "generation_lambda": config.generation_lambda,
        "aware_lambda": config.aware_lambda,
        "antifungals_excluded": sorted(ANTIFUNGALS),
    }
    with (config.output_dir / "config.json").open("w") as handle:
        json.dump(run_config, handle, indent=2)
    print(f"Results written to {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
