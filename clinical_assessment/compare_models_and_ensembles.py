#!/usr/bin/env python3
"""Compare individual global models with GRASP-only and four-model ensembles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_clinical_assessment import (
    DEFAULT_SPECIES,
    DEFAULT_THRESHOLDS,
    GRASP_MODELS,
    MLP_MODEL,
    AssessmentConfig,
    _load_antibiotics,
    _normalise_species,
    _prediction_paths,
    evaluate_block,
)


SYSTEM_LABELS = {
    "species_conditioned_grasp": "GRASP species-conditioned",
    "multihead_grasp": "GRASP multihead",
    "hypernetwork_grasp": "GRASP hypernetwork",
    "species_aware_global_mlp": "GRASP-MLP",
    "ensemble_grasp": "Ensemble GRASP (3)",
    "ensemble_all": "Ensemble global (4)",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare individual OOD models and two ensembles."
    )
    parser.add_argument("--results-root", type=Path, default=here.parent / "ood_evaluation" / "results")
    parser.add_argument("--output-dir", type=Path, default=here / "results" / "model_comparison")
    parser.add_argument("--species", nargs="+", default=list(DEFAULT_SPECIES))
    parser.add_argument("--scenarios", nargs="+", default=["zero_shot", "finetuned"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--uncertainty-lambda", type=float, default=0.5)
    parser.add_argument("--generation-lambda", type=float, default=0.2)
    parser.add_argument("--aware-lambda", type=float, default=0.2)
    return parser.parse_args()


def load_system_blocks(paths: dict[str, Path]) -> dict[str, dict[str, np.ndarray]]:
    models: dict[str, dict[str, np.ndarray]] = {}
    for model, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            models[model] = {
                "test_idx": np.array(data["test_idx"]),
                "antibiotic_indices": np.array(data["antibiotic_indices"]).astype(int),
                "y_true": np.array(data["y_true"]).astype(float),
                "y_score": np.array(data["y_score"]).astype(float),
            }

    reference = models[GRASP_MODELS[0]]
    for model, block in models.items():
        for key in ("test_idx", "antibiotic_indices", "y_true"):
            if not np.array_equal(reference[key], block[key], equal_nan=True):
                raise ValueError(f"{model} is not aligned on {key}: {paths[model]}")
        if not np.isfinite(block["y_score"]).all():
            raise ValueError(f"{model} contains non-finite prediction scores.")

    common = {
        "test_idx": reference["test_idx"],
        "antibiotic_indices": reference["antibiotic_indices"],
        "y_true": reference["y_true"],
    }
    systems: dict[str, dict[str, np.ndarray]] = {}
    for model, block in models.items():
        systems[model] = {
            **common,
            "mean_probability": block["y_score"],
            # Individual models have no inter-model disagreement penalty.
            "std_probability": np.zeros_like(block["y_score"]),
        }

    grasp_scores = np.stack([models[model]["y_score"] for model in GRASP_MODELS])
    all_scores = np.stack(
        [models[model]["y_score"] for model in (*GRASP_MODELS, MLP_MODEL)]
    )
    systems["ensemble_grasp"] = {
        **common,
        "mean_probability": grasp_scores.mean(axis=0),
        "std_probability": grasp_scores.std(axis=0),
    }
    systems["ensemble_all"] = {
        **common,
        "mean_probability": all_scores.mean(axis=0),
        "std_probability": all_scores.std(axis=0),
    }
    return systems


def pooled_threshold_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["scenario", "species", "system", "resistance_threshold"]
    rows: list[dict[str, object]] = []
    for key, group in diagnostics.groupby(group_columns):
        tp = int(group["true_resistant_predicted_resistant"].sum())
        fn = int(group["true_resistant_predicted_susceptible"].sum())
        tn = int(group["true_susceptible_predicted_susceptible"].sum())
        fp = int(group["true_susceptible_predicted_resistant"].sum())
        sensitivity = tp / (tp + fn) if tp + fn else float("nan")
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        rows.append(
            {
                **dict(zip(group_columns, key)),
                "resistance_sensitivity": sensitivity,
                "susceptibility_specificity": specificity,
                "balanced_accuracy": (sensitivity + specificity) / 2,
                "false_susceptible_rate": 1 - sensitivity,
                "false_resistant_rate": 1 - specificity,
                "n_true_resistant_sample_folds": tp + fn,
                "n_true_susceptible_sample_folds": tn + fp,
            }
        )
    return pd.DataFrame(rows)


def select_thresholds(sweep: pd.DataFrame) -> pd.DataFrame:
    ordered = sweep.sort_values(
        [
            "scenario",
            "species",
            "system",
            "balanced_accuracy",
            "resistance_threshold",
        ],
        ascending=[True, True, True, False, True],
    )
    return (
        ordered.groupby(["scenario", "species", "system"], as_index=False)
        .first()
        .sort_values(["scenario", "species", "system"])
    )


def ndcg_at_selected_thresholds(
    metrics: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in selected.itertuples(index=False):
        base = metrics[
            (metrics["scenario"] == record.scenario)
            & (metrics["species"] == record.species)
            & (metrics["system"] == record.system)
            & metrics["ranking"].isin(["raw", "clinical_reranked"])
        ]
        base = base[
            (base["ranking"] == "raw")
            | (
                (base["ranking"] == "clinical_reranked")
                & np.isclose(
                    base["resistance_threshold"],
                    record.resistance_threshold,
                    equal_nan=False,
                )
            )
        ]
        for (ranking, k), group in base.groupby(["ranking", "k"]):
            rows.append(
                {
                    "scenario": record.scenario,
                    "species": record.species,
                    "system": record.system,
                    "system_label": SYSTEM_LABELS[record.system],
                    "selected_resistance_threshold": record.resistance_threshold,
                    "ranking": ranking,
                    "k": int(k),
                    "mean_ndcg": float(group["mean_ndcg"].mean()),
                    "std_ndcg_across_folds": float(group["mean_ndcg"].std(ddof=1)),
                    "n_folds": int(group["fold"].nunique()),
                    "n_evaluable_sample_folds": int(group["n_evaluable_samples"].sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scenario", "species", "system", "k", "ranking"]
    )


def comparison_wide(ndcg: pd.DataFrame) -> pd.DataFrame:
    wide = ndcg.pivot_table(
        index=[
            "scenario",
            "species",
            "system",
            "system_label",
            "selected_resistance_threshold",
        ],
        columns=["ranking", "k"],
        values="mean_ndcg",
    )
    wide.columns = [f"{ranking}_ndcg_at_{k}" for ranking, k in wide.columns]
    return wide.reset_index().sort_values(["scenario", "species", "system"])


def _markdown_table(frame: pd.DataFrame, columns: list[str], decimals: int = 3) -> str:
    display = frame[columns].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}"
        )
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    selected: pd.DataFrame,
    wide: pd.DataFrame,
) -> None:
    selected = selected.copy()
    selected["model"] = selected["system"].map(SYSTEM_LABELS)
    wide = wide.copy()
    wide["model"] = wide["system"].map(SYSTEM_LABELS)
    sections = [
        "# Comparación de modelos individuales y ensembles",
        "",
        "Los modelos individuales no reciben penalización por desacuerdo. "
        "Los ensembles de tres y cuatro modelos utilizan "
        "`S = 1 - mean(pR) - 0.5 * std(pR)`.",
        "",
        "Los umbrales se seleccionan post-hoc por especie, escenario y sistema "
        "maximizando balanced accuracy.",
    ]
    for scenario in ("zero_shot", "finetuned"):
        sections.extend(["", f"## {scenario}", ""])
        for species in sorted(selected["species"].unique()):
            sections.extend(["", f"### {species}", "", "#### Sensibilidad y especificidad", ""])
            threshold_table = selected[
                (selected["scenario"] == scenario) & (selected["species"] == species)
            ].sort_values("model")
            sections.append(
                _markdown_table(
                    threshold_table,
                    [
                        "model",
                        "resistance_threshold",
                        "resistance_sensitivity",
                        "susceptibility_specificity",
                        "balanced_accuracy",
                    ],
                )
            )
            sections.extend(["", "#### NDCG bruto y clínico", ""])
            ndcg_table = wide[
                (wide["scenario"] == scenario) & (wide["species"] == species)
            ].sort_values("model")
            sections.append(
                _markdown_table(
                    ndcg_table,
                    [
                        "model",
                        "raw_ndcg_at_1",
                        "clinical_reranked_ndcg_at_1",
                        "raw_ndcg_at_3",
                        "clinical_reranked_ndcg_at_3",
                        "raw_ndcg_at_5",
                        "clinical_reranked_ndcg_at_5",
                    ],
                )
            )
    (output_dir / "COMPARACION_MODELOS.md").write_text(
        "\n".join(sections) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    species_names = tuple(dict.fromkeys(_normalise_species(s) for s in args.species))
    scenarios = tuple(dict.fromkeys(args.scenarios))
    thresholds = tuple(sorted(set(args.thresholds)))
    antibiotics = _load_antibiotics(args.results_root)
    config = AssessmentConfig(
        uncertainty_lambda=args.uncertainty_lambda,
        generation_lambda=args.generation_lambda,
        aware_lambda=args.aware_lambda,
        thresholds=thresholds,
        k_values=(1, 3, 5),
    )

    metric_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for species in species_names:
            for fold in range(1, 6):
                systems = load_system_blocks(
                    _prediction_paths(args.results_root, scenario, species, fold)
                )
                for system, block in systems.items():
                    block_metrics, _, block_diagnostics = evaluate_block(
                        block, antibiotics, scenario, species, fold, config
                    )
                    for row in block_metrics:
                        row["system"] = system
                    for row in block_diagnostics:
                        row["system"] = system
                    metric_rows.extend(block_metrics)
                    diagnostic_rows.extend(block_diagnostics)
                print(f"Evaluated systems for {scenario}: {species}, fold {fold}")

    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    sweep = pooled_threshold_table(diagnostics)
    selected = select_thresholds(sweep)
    selected["system_label"] = selected["system"].map(SYSTEM_LABELS)
    ndcg = ndcg_at_selected_thresholds(metrics, selected)
    wide = comparison_wide(ndcg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "ndcg_by_fold_all_thresholds.csv", index=False)
    sweep.to_csv(args.output_dir / "sensitivity_specificity_threshold_sweep.csv", index=False)
    selected.to_csv(args.output_dir / "selected_thresholds_sensitivity_specificity.csv", index=False)
    ndcg.to_csv(args.output_dir / "ndcg_at_selected_thresholds_long.csv", index=False)
    wide.to_csv(args.output_dir / "ndcg_at_selected_thresholds_wide.csv", index=False)
    write_report(args.output_dir, selected, wide)
    print(f"Comparison written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
