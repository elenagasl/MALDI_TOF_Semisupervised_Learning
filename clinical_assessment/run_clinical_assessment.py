#!/usr/bin/env python3
"""Clinical ranking assessment for the four global OOD AMR models."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from generations import antibiotic_generation_metadata
from spectra import antibiotic_aware_category


GRASP_MODELS = (
    "species_conditioned_grasp",
    "multihead_grasp",
    "hypernetwork_grasp",
)
MLP_MODEL = "species_aware_global_mlp"
DEFAULT_SPECIES = (
    "escherichia_coli",
    "staphylococcus_aureus",
    "enterococcus_faecalis",
    "pseudomonas_aeruginosa",
    "klebsiella_pneumoniae",
)
DEFAULT_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
DEFAULT_K = (1, 3, 5, 10)
ANTIFUNGALS = frozenset(
    {
        "5-Fluorocytosine",
        "Amphotericin B",
        "Anidulafungin",
        "Caspofungin",
        "Fluconazole",
        "Isavuconazole",
        "Itraconazole",
        "Micafungin",
        "Posaconazole",
        "Voriconazole",
    }
)


@dataclass(frozen=True)
class AssessmentConfig:
    uncertainty_lambda: float
    generation_lambda: float
    aware_lambda: float
    thresholds: tuple[float, ...]
    k_values: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_results = here.parent / "ood_evaluation" / "results"
    parser = argparse.ArgumentParser(
        description="Evaluate raw, uncertainty-aware and clinically reranked OOD rankings."
    )
    parser.add_argument("--results-root", type=Path, default=default_results)
    parser.add_argument("--output-dir", type=Path, default=here / "results")
    parser.add_argument("--species", nargs="+", default=list(DEFAULT_SPECIES))
    parser.add_argument("--scenarios", nargs="+", default=["zero_shot", "finetuned"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_K))
    parser.add_argument("--uncertainty-lambda", type=float, default=0.5)
    parser.add_argument("--generation-lambda", type=float, default=0.2)
    parser.add_argument("--aware-lambda", type=float, default=0.2)
    return parser.parse_args()


def _normalise_species(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _validate_args(args: argparse.Namespace) -> AssessmentConfig:
    thresholds = tuple(sorted(set(args.thresholds)))
    k_values = tuple(sorted(set(args.k)))
    if not thresholds or any(t < 0 or t > 1 for t in thresholds):
        raise ValueError("Thresholds must be within [0, 1].")
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("All K values must be positive.")
    if any(x < 0 for x in (args.uncertainty_lambda, args.generation_lambda, args.aware_lambda)):
        raise ValueError("Penalty lambdas must be non-negative.")
    unknown_scenarios = set(args.scenarios) - {"zero_shot", "finetuned"}
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios: {sorted(unknown_scenarios)}")
    return AssessmentConfig(
        uncertainty_lambda=args.uncertainty_lambda,
        generation_lambda=args.generation_lambda,
        aware_lambda=args.aware_lambda,
        thresholds=thresholds,
        k_values=k_values,
    )


def _load_antibiotics(results_root: Path) -> list[str]:
    grasp_config = results_root / "ood_deployment_all_models" / "config.json"
    mlp_config = results_root / "ood_deployment_grasp_mlp" / "config.json"
    with grasp_config.open() as handle:
        grasp_antibiotics = json.load(handle)["antibiotics"]
    with mlp_config.open() as handle:
        mlp_antibiotics = json.load(handle)["antibiotics"]
    if grasp_antibiotics != mlp_antibiotics:
        raise ValueError("The GRASP and GRASP-MLP antibiotic panels are not aligned.")
    return list(grasp_antibiotics)


def _prediction_paths(
    results_root: Path, scenario: str, species: str, fold: int
) -> dict[str, Path]:
    filename = f"{scenario}_{species}_fold_{fold}.npz"
    base = results_root / "ood_deployment_all_models" / "predictions"
    paths = {model: base / model / filename for model in GRASP_MODELS}
    paths[MLP_MODEL] = (
        results_root
        / "ood_deployment_grasp_mlp"
        / "predictions"
        / MLP_MODEL
        / filename
    )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prediction files:\n" + "\n".join(missing))
    return paths


def _load_ensemble_block(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    blocks = {}
    for model, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            blocks[model] = {key: np.array(data[key]) for key in data.files}

    reference = blocks[GRASP_MODELS[0]]
    for model, block in blocks.items():
        for key in ("test_idx", "antibiotic_indices", "y_true"):
            if not np.array_equal(reference[key], block[key], equal_nan=True):
                raise ValueError(f"{model} is not aligned with the ensemble on {key}: {paths[model]}")
        if reference["y_score"].shape != block["y_score"].shape:
            raise ValueError(f"{model} has an incompatible y_score shape: {paths[model]}")

    scores = np.stack([blocks[model]["y_score"] for model in (*GRASP_MODELS, MLP_MODEL)])
    if not np.isfinite(scores).all():
        raise ValueError(f"Non-finite model scores found in {next(iter(paths.values()))}")
    return {
        "test_idx": reference["test_idx"],
        "antibiotic_indices": reference["antibiotic_indices"].astype(int),
        "y_true": reference["y_true"].astype(float),
        "mean_probability": scores.mean(axis=0),
        "std_probability": scores.std(axis=0),
    }


def _clinical_costs(
    antibiotic_indices: np.ndarray,
    antibiotics: list[str],
    config: AssessmentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    penalties = np.zeros(len(antibiotic_indices), dtype=float)
    keep = np.ones(len(antibiotic_indices), dtype=bool)
    aware_cost = {"Access": 0.0, "Watch": 0.5, "Reserve": 1.0}

    for local_idx, global_idx in enumerate(antibiotic_indices):
        antibiotic = antibiotics[global_idx]
        if antibiotic in ANTIFUNGALS:
            keep[local_idx] = False
            continue
        generation = antibiotic_generation_metadata.get(antibiotic, {}).get("generation")
        generation_cost = 0.0 if generation is None else (float(generation) - 1.0) / 4.0
        category = antibiotic_aware_category.get(antibiotic)
        category_cost = aware_cost.get(category, 0.0)
        penalties[local_idx] = (
            config.generation_lambda * generation_cost
            + config.aware_lambda * category_cost
        )
    return penalties, keep


def ndcg_at_k(relevance: np.ndarray, predicted_score: np.ndarray, k: int) -> float:
    """Return NDCG@K, or NaN when the ideal DCG is zero."""
    if relevance.ndim != 1 or predicted_score.shape != relevance.shape:
        raise ValueError("Relevance and predicted_score must be aligned 1-D arrays.")
    if relevance.size == 0:
        return float("nan")
    limit = min(k, relevance.size)
    discounts = np.log2(np.arange(2, limit + 2, dtype=float))
    # Stable sorting gives deterministic output for tied clinical utilities.
    predicted_order = np.argsort(-predicted_score, kind="stable")[:limit]
    ideal_order = np.argsort(-relevance, kind="stable")[:limit]
    dcg = float(np.sum(relevance[predicted_order] / discounts))
    idcg = float(np.sum(relevance[ideal_order] / discounts))
    return dcg / idcg if idcg > 0 else float("nan")


def _update_accumulator(
    accumulator: dict[tuple, list[float]],
    key: tuple,
    value: float,
) -> None:
    if np.isfinite(value):
        accumulator[key].append(float(value))


def _mean_or_nan(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def evaluate_block(
    block: dict[str, np.ndarray],
    antibiotics: list[str],
    scenario: str,
    species: str,
    fold: int,
    config: AssessmentConfig,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    y_true = block["y_true"]
    mean_probability = block["mean_probability"]
    std_probability = block["std_probability"]
    penalties, bacterial_columns = _clinical_costs(
        block["antibiotic_indices"], antibiotics, config
    )
    safety_score = 1.0 - mean_probability - config.uncertainty_lambda * std_probability

    metric_values: dict[tuple, list[float]] = defaultdict(list)
    delta_values: dict[tuple, list[float]] = defaultdict(list)
    candidate_counts: list[int] = []
    susceptible_counts: list[int] = []
    observed_truth: list[np.ndarray] = []
    observed_probability: list[np.ndarray] = []

    for row in range(y_true.shape[0]):
        observed = (
            bacterial_columns
            & np.isfinite(y_true[row])
            & np.isfinite(mean_probability[row])
            & np.isfinite(std_probability[row])
        )
        if not observed.any():
            continue

        truth = y_true[row, observed]
        row_penalties = penalties[observed]
        raw_score = 1.0 - mean_probability[row, observed]
        uncertainty_score = safety_score[row, observed]
        ideal_utility = np.where(truth == 0.0, np.maximum(0.0, 1.0 - row_penalties), 0.0)
        candidate_counts.append(int(observed.sum()))
        susceptible_counts.append(int((truth == 0.0).sum()))
        observed_truth.append(truth)
        observed_probability.append(mean_probability[row, observed])

        raw_ndcg: dict[int, float] = {}
        for k in config.k_values:
            raw_ndcg[k] = ndcg_at_k(ideal_utility, raw_score, k)
            uncertainty_ndcg = ndcg_at_k(ideal_utility, uncertainty_score, k)
            _update_accumulator(metric_values, ("raw", None, k), raw_ndcg[k])
            _update_accumulator(
                metric_values, ("uncertainty_adjusted", None, k), uncertainty_ndcg
            )

        row_mean_probability = mean_probability[row, observed]
        for threshold in config.thresholds:
            predicted_susceptible = row_mean_probability < threshold
            clinical_score = np.where(
                predicted_susceptible,
                np.maximum(0.0, uncertainty_score - row_penalties),
                0.0,
            )
            for k in config.k_values:
                clinical_ndcg = ndcg_at_k(ideal_utility, clinical_score, k)
                _update_accumulator(
                    metric_values, ("clinical_reranked", threshold, k), clinical_ndcg
                )
                if np.isfinite(clinical_ndcg) and np.isfinite(raw_ndcg[k]):
                    _update_accumulator(
                        delta_values,
                        (threshold, k),
                        clinical_ndcg - raw_ndcg[k],
                    )

    metric_rows: list[dict[str, object]] = []
    for (ranking, threshold, k), values in sorted(
        metric_values.items(),
        key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1], item[0][2]),
    ):
        metric_rows.append(
            {
                "scenario": scenario,
                "species": species,
                "fold": fold,
                "ranking": ranking,
                "resistance_threshold": threshold,
                "k": k,
                "mean_ndcg": _mean_or_nan(values),
                "std_ndcg_across_samples": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                "n_evaluable_samples": len(values),
                "mean_candidate_antibiotics": _mean_or_nan(candidate_counts),
                "mean_true_susceptible_antibiotics": _mean_or_nan(susceptible_counts),
            }
        )

    delta_rows: list[dict[str, object]] = []
    for (threshold, k), values in sorted(delta_values.items()):
        delta_rows.append(
            {
                "scenario": scenario,
                "species": species,
                "fold": fold,
                "resistance_threshold": threshold,
                "k": k,
                "mean_ndcg_delta_clinical_vs_raw": _mean_or_nan(values),
                "n_paired_samples": len(values),
            }
        )

    diagnostic_rows: list[dict[str, object]] = []
    all_truth = np.concatenate(observed_truth) if observed_truth else np.array([])
    all_probability = (
        np.concatenate(observed_probability) if observed_probability else np.array([])
    )
    for threshold in config.thresholds:
        predicted_resistant = all_probability >= threshold
        true_resistant = all_truth == 1.0
        true_susceptible = all_truth == 0.0
        tp = int(np.sum(predicted_resistant & true_resistant))
        fn = int(np.sum(~predicted_resistant & true_resistant))
        tn = int(np.sum(~predicted_resistant & true_susceptible))
        fp = int(np.sum(predicted_resistant & true_susceptible))
        diagnostic_rows.append(
            {
                "scenario": scenario,
                "species": species,
                "fold": fold,
                "resistance_threshold": threshold,
                "n_observed_pairs": int(all_truth.size),
                "n_true_resistant": tp + fn,
                "n_true_susceptible": tn + fp,
                "true_resistant_predicted_resistant": tp,
                "true_resistant_predicted_susceptible": fn,
                "true_susceptible_predicted_susceptible": tn,
                "true_susceptible_predicted_resistant": fp,
                "resistance_sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
                "susceptibility_specificity": tn / (tn + fp) if tn + fp else float("nan"),
                "false_susceptible_rate": fn / (tp + fn) if tp + fn else float("nan"),
                "false_resistant_rate": fp / (tn + fp) if tn + fp else float("nan"),
                "predicted_resistant_rate": (tp + fp) / all_truth.size
                if all_truth.size
                else float("nan"),
            }
        )
    return metric_rows, delta_rows, diagnostic_rows


def _summarise_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["scenario", "species", "ranking", "resistance_threshold", "k"]
    return (
        metrics.groupby(group_columns, dropna=False)
        .agg(
            mean_ndcg=("mean_ndcg", "mean"),
            std_ndcg_across_folds=("mean_ndcg", "std"),
            min_fold_ndcg=("mean_ndcg", "min"),
            max_fold_ndcg=("mean_ndcg", "max"),
            n_folds=("fold", "nunique"),
            n_evaluable_sample_folds=("n_evaluable_samples", "sum"),
            mean_candidate_antibiotics=("mean_candidate_antibiotics", "mean"),
            mean_true_susceptible_antibiotics=("mean_true_susceptible_antibiotics", "mean"),
        )
        .reset_index()
        .sort_values(group_columns, na_position="first")
    )


def _summarise_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["scenario", "species", "resistance_threshold", "k"]
    return (
        deltas.groupby(group_columns, dropna=False)
        .agg(
            mean_ndcg_delta_clinical_vs_raw=("mean_ndcg_delta_clinical_vs_raw", "mean"),
            std_delta_across_folds=("mean_ndcg_delta_clinical_vs_raw", "std"),
            positive_folds=("mean_ndcg_delta_clinical_vs_raw", lambda values: int((values > 0).sum())),
            n_folds=("fold", "nunique"),
            n_paired_sample_folds=("n_paired_samples", "sum"),
        )
        .reset_index()
        .sort_values(group_columns)
    )


def _overall_weighted_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = ["scenario", "ranking", "resistance_threshold", "k"]
    for key, group in metrics.groupby(groups, dropna=False):
        weights = group["n_evaluable_samples"].to_numpy(dtype=float)
        values = group["mean_ndcg"].to_numpy(dtype=float)
        valid = np.isfinite(values) & (weights > 0)
        rows.append(
            {
                **dict(zip(groups, key)),
                "weighted_mean_ndcg": float(np.average(values[valid], weights=weights[valid]))
                if valid.any()
                else float("nan"),
                "n_evaluable_sample_folds": int(weights[valid].sum()),
                "n_species": int(group["species"].nunique()),
                "n_folds": int(group[["species", "fold"]].drop_duplicates().shape[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(groups, na_position="first")


def _summarise_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    count_columns = [
        "n_observed_pairs",
        "n_true_resistant",
        "n_true_susceptible",
        "true_resistant_predicted_resistant",
        "true_resistant_predicted_susceptible",
        "true_susceptible_predicted_susceptible",
        "true_susceptible_predicted_resistant",
    ]
    rate_columns = [
        "resistance_sensitivity",
        "susceptibility_specificity",
        "false_susceptible_rate",
        "false_resistant_rate",
        "predicted_resistant_rate",
    ]
    return (
        diagnostics.groupby(
            ["scenario", "species", "resistance_threshold"], dropna=False
        )
        .agg(
            **{column: (column, "sum") for column in count_columns},
            **{f"mean_{column}": (column, "mean") for column in rate_columns},
            **{f"std_{column}_across_folds": (column, "std") for column in rate_columns},
            n_folds=("fold", "nunique"),
        )
        .reset_index()
        .sort_values(["scenario", "species", "resistance_threshold"])
    )


def main() -> None:
    args = parse_args()
    config = _validate_args(args)
    species_names = tuple(dict.fromkeys(_normalise_species(s) for s in args.species))
    scenarios = tuple(dict.fromkeys(args.scenarios))
    antibiotics = _load_antibiotics(args.results_root)

    metric_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for species in species_names:
            for fold in range(1, 6):
                paths = _prediction_paths(args.results_root, scenario, species, fold)
                block = _load_ensemble_block(paths)
                block_metrics, block_deltas, block_diagnostics = evaluate_block(
                    block, antibiotics, scenario, species, fold, config
                )
                metric_rows.extend(block_metrics)
                delta_rows.extend(block_deltas)
                diagnostic_rows.extend(block_diagnostics)
                print(f"Evaluated {scenario}: {species}, fold {fold}")

    metrics = pd.DataFrame(metric_rows)
    deltas = pd.DataFrame(delta_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "ndcg_by_fold.csv", index=False)
    _summarise_metrics(metrics).to_csv(args.output_dir / "ndcg_summary.csv", index=False)
    _overall_weighted_summary(metrics).to_csv(
        args.output_dir / "ndcg_overall_weighted_summary.csv", index=False
    )
    deltas.to_csv(args.output_dir / "clinical_vs_raw_delta_by_fold.csv", index=False)
    _summarise_deltas(deltas).to_csv(
        args.output_dir / "clinical_vs_raw_delta_summary.csv", index=False
    )
    diagnostics.to_csv(args.output_dir / "threshold_diagnostics_by_fold.csv", index=False)
    _summarise_diagnostics(diagnostics).to_csv(
        args.output_dir / "threshold_diagnostics_summary.csv", index=False
    )

    run_config = {
        "results_root": str(args.results_root.resolve()),
        "species": list(species_names),
        "scenarios": list(scenarios),
        "models": [*GRASP_MODELS, MLP_MODEL],
        "thresholds": list(config.thresholds),
        "k": list(config.k_values),
        "uncertainty_lambda": config.uncertainty_lambda,
        "generation_lambda": config.generation_lambda,
        "aware_lambda": config.aware_lambda,
        "antifungals_excluded": sorted(ANTIFUNGALS),
        "missing_ast_policy": "exclude_from_both_predicted_and_ideal_ranking",
        "generation_cost": "(generation - 1) / 4; zero when unavailable",
        "aware_cost": {"Access": 0.0, "Watch": 0.5, "Reserve": 1.0, "unavailable": 0.0},
        "primary_resistance_threshold": 0.20,
    }
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(run_config, handle, indent=2)
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
