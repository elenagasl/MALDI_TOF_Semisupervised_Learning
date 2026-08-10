#!/usr/bin/env python3
"""Compute clinically interpretable percentages for antibiotic recommendations."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from compare_models_and_ensembles import SYSTEM_LABELS, load_system_blocks
from run_clinical_assessment import (
    DEFAULT_SPECIES,
    DEFAULT_THRESHOLDS,
    AssessmentConfig,
    _clinical_costs,
    _load_antibiotics,
    _normalise_species,
    _prediction_paths,
)
from generations import antibiotic_generation_metadata
from spectra import antibiotic_aware_category


DEFAULT_K = (1, 3, 5)
SCENARIO_ORDER = {"finetuned": 0, "zero_shot": 1}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Calculate susceptibility and stewardship percentages for raw and "
            "clinically reranked antibiotic recommendations."
        )
    )
    parser.add_argument("--results-root", type=Path, default=here.parent / "ood_evaluation" / "results")
    parser.add_argument("--comparison-dir", type=Path, default=here / "results" / "model_comparison")
    parser.add_argument("--output-dir", type=Path, default=here / "results" / "recommendation_percentages")
    parser.add_argument("--species", nargs="+", default=list(DEFAULT_SPECIES))
    parser.add_argument("--scenarios", nargs="+", default=["finetuned", "zero_shot"])
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_K))
    parser.add_argument("--uncertainty-lambda", type=float, default=0.5)
    parser.add_argument("--generation-lambda", type=float, default=0.2)
    parser.add_argument("--aware-lambda", type=float, default=0.2)
    return parser.parse_args()


def _generation_metadata(antibiotic: str) -> tuple[str | None, float | None]:
    metadata = antibiotic_generation_metadata.get(antibiotic, {})
    family = metadata.get("generation_family")
    generation = metadata.get("generation")
    return family, float(generation) if generation is not None else None


def _aware_rank(category: str | None) -> int | None:
    ranks = {"Access": 0, "Watch": 1, "Reserve": 2}
    return ranks.get(category)


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def _format_percent(value: float) -> str:
    return "" if pd.isna(value) else f"{100 * value:.1f}%"


def _format_float(value: float, digits: int = 3) -> str:
    return "" if pd.isna(value) else f"{value:.{digits}f}"


def recommendation_rows(
    results_root: Path,
    antibiotics: list[str],
    selected: pd.DataFrame,
    species_names: tuple[str, ...],
    scenarios: tuple[str, ...],
    k_values: tuple[int, ...],
    config: AssessmentConfig,
) -> pd.DataFrame:
    selected_lookup = {
        (row.scenario, row.species, row.system): float(row.resistance_threshold)
        for row in selected.itertuples(index=False)
    }
    counters: dict[tuple, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

    for scenario in scenarios:
        for species in species_names:
            for fold in range(1, 6):
                systems = load_system_blocks(
                    _prediction_paths(results_root, scenario, species, fold)
                )
                for system, block in systems.items():
                    threshold = selected_lookup[(scenario, species, system)]
                    penalties, bacterial = _clinical_costs(
                        block["antibiotic_indices"], antibiotics, config
                    )
                    safety_score = (
                        1.0
                        - block["mean_probability"]
                        - config.uncertainty_lambda * block["std_probability"]
                    )
                    for row_idx in range(block["y_true"].shape[0]):
                        observed = (
                            bacterial
                            & np.isfinite(block["y_true"][row_idx])
                            & np.isfinite(block["mean_probability"][row_idx])
                            & np.isfinite(block["std_probability"][row_idx])
                        )
                        if not observed.any():
                            continue

                        local_columns = np.flatnonzero(observed)
                        truth = block["y_true"][row_idx, observed]
                        probability = block["mean_probability"][row_idx, observed]
                        true_susceptible = truth == 0.0
                        true_susceptible_columns = local_columns[true_susceptible]
                        susceptible_antibiotics = [
                            antibiotics[int(block["antibiotic_indices"][local_col])]
                            for local_col in true_susceptible_columns
                        ]

                        access_available = any(
                            antibiotic_aware_category.get(antibiotic) == "Access"
                            for antibiotic in susceptible_antibiotics
                        )
                        min_generation_by_family: dict[str, float] = {}
                        for antibiotic in susceptible_antibiotics:
                            family, generation = _generation_metadata(antibiotic)
                            if family is None or generation is None:
                                continue
                            min_generation_by_family[family] = min(
                                generation,
                                min_generation_by_family.get(family, generation),
                            )

                        raw_score = 1.0 - probability
                        clinical_score = np.maximum(
                            0.0, safety_score[row_idx, observed] - penalties[observed]
                        )
                        ranking_inputs = {
                            "raw": np.argsort(-raw_score, kind="stable"),
                            "clinical_reranked": np.argsort(-clinical_score, kind="stable"),
                        }
                        clinical_eligible = (probability < threshold) & (clinical_score > 0)

                        for ranking, order in ranking_inputs.items():
                            if ranking == "clinical_reranked":
                                order = np.array(
                                    [idx for idx in order if clinical_eligible[idx]], dtype=int
                                )
                            for k in k_values:
                                key = (
                                    scenario,
                                    species,
                                    system,
                                    SYSTEM_LABELS[system],
                                    ranking,
                                    threshold if ranking == "clinical_reranked" else np.nan,
                                    k,
                                )
                                counters[key]["n_evaluable_sample_folds"] += 1
                                chosen_positions = order[:k]
                                if chosen_positions.size:
                                    counters[key]["n_sample_folds_with_recommendation"] += 1

                                for position in chosen_positions:
                                    local_col = int(local_columns[position])
                                    global_idx = int(block["antibiotic_indices"][local_col])
                                    antibiotic = antibiotics[global_idx]
                                    category = antibiotic_aware_category.get(antibiotic)
                                    family, generation = _generation_metadata(antibiotic)
                                    is_susceptible = bool(block["y_true"][row_idx, local_col] == 0.0)

                                    counters[key]["n_recommended_slots"] += 1
                                    counters[key]["n_recommended_true_susceptible"] += int(is_susceptible)
                                    counters[key]["n_recommended_true_resistant"] += int(not is_susceptible)

                                    if family is not None and generation is not None:
                                        min_generation = min_generation_by_family.get(family)
                                        if min_generation is not None:
                                            counters[key]["n_generation_comparable_slots"] += 1
                                            counters[key]["n_generation_overuse"] += int(
                                                generation > min_generation
                                            )

                                    if access_available:
                                        counters[key]["n_slots_with_susceptible_access_available"] += 1
                                        aware_rank = _aware_rank(category)
                                        if aware_rank is not None and aware_rank > 0:
                                            counters[key]["n_watch_or_reserve_when_access_available"] += 1
                                            counters[key]["n_watch_when_access_available"] += int(
                                                category == "Watch"
                                            )
                                            counters[key]["n_reserve_when_access_available"] += int(
                                                category == "Reserve"
                                            )

    rows: list[dict[str, object]] = []
    for key, counts in counters.items():
        (
            scenario,
            species,
            system,
            system_label,
            ranking,
            selected_threshold,
            k,
        ) = key
        n_slots = counts["n_recommended_slots"]
        n_generation_comparable = counts["n_generation_comparable_slots"]
        n_access_available = counts["n_slots_with_susceptible_access_available"]
        rows.append(
            {
                "scenario": scenario,
                "species": species,
                "system": system,
                "system_label": system_label,
                "ranking": ranking,
                "selected_resistance_threshold": selected_threshold,
                "k": int(k),
                "n_evaluable_sample_folds": counts["n_evaluable_sample_folds"],
                "n_sample_folds_with_recommendation": counts[
                    "n_sample_folds_with_recommendation"
                ],
                "recommendation_coverage": _safe_rate(
                    counts["n_sample_folds_with_recommendation"],
                    counts["n_evaluable_sample_folds"],
                ),
                "n_recommended_slots": n_slots,
                "recommended_true_susceptible_pct": _safe_rate(
                    counts["n_recommended_true_susceptible"], n_slots
                ),
                "recommended_true_resistant_pct": _safe_rate(
                    counts["n_recommended_true_resistant"], n_slots
                ),
                "generation_overuse_pct_all_recommendations": _safe_rate(
                    counts["n_generation_overuse"], n_slots
                ),
                "generation_overuse_pct_comparable": _safe_rate(
                    counts["n_generation_overuse"], n_generation_comparable
                ),
                "n_generation_comparable_slots": n_generation_comparable,
                "watch_or_reserve_when_access_available_pct_all_recommendations": _safe_rate(
                    counts["n_watch_or_reserve_when_access_available"], n_slots
                ),
                "watch_or_reserve_when_access_available_pct_access_available": _safe_rate(
                    counts["n_watch_or_reserve_when_access_available"], n_access_available
                ),
                "watch_when_access_available_pct_access_available": _safe_rate(
                    counts["n_watch_when_access_available"], n_access_available
                ),
                "reserve_when_access_available_pct_access_available": _safe_rate(
                    counts["n_reserve_when_access_available"], n_access_available
                ),
                "n_slots_with_susceptible_access_available": n_access_available,
            }
        )

    result = pd.DataFrame(rows)
    result["_scenario_order"] = result["scenario"].map(SCENARIO_ORDER).fillna(99)
    return (
        result.sort_values(
            ["_scenario_order", "species", "system_label", "ranking", "k"]
        )
        .drop(columns="_scenario_order")
        .reset_index(drop=True)
    )


def global_summary(by_species: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["scenario", "system", "system_label", "ranking", "k"]
    count_columns = [
        "n_evaluable_sample_folds",
        "n_sample_folds_with_recommendation",
        "n_recommended_slots",
        "n_generation_comparable_slots",
        "n_slots_with_susceptible_access_available",
    ]
    # Reconstruct numerators from rounded-free rates and denominators.
    frame = by_species.copy()
    frame["n_recommended_true_susceptible"] = (
        frame["recommended_true_susceptible_pct"] * frame["n_recommended_slots"]
    )
    frame["n_recommended_true_resistant"] = (
        frame["recommended_true_resistant_pct"] * frame["n_recommended_slots"]
    )
    frame["n_generation_overuse"] = (
        frame["generation_overuse_pct_comparable"]
        * frame["n_generation_comparable_slots"]
    ).fillna(0.0)
    frame["n_watch_or_reserve_when_access_available"] = (
        frame["watch_or_reserve_when_access_available_pct_access_available"]
        * frame["n_slots_with_susceptible_access_available"]
    ).fillna(0.0)
    frame["n_watch_when_access_available"] = (
        frame["watch_when_access_available_pct_access_available"]
        * frame["n_slots_with_susceptible_access_available"]
    ).fillna(0.0)
    frame["n_reserve_when_access_available"] = (
        frame["reserve_when_access_available_pct_access_available"]
        * frame["n_slots_with_susceptible_access_available"]
    ).fillna(0.0)

    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_columns, sort=False):
        sums = {column: float(group[column].sum()) for column in count_columns}
        n_slots = int(sums["n_recommended_slots"])
        n_generation_comparable = int(sums["n_generation_comparable_slots"])
        n_access_available = int(sums["n_slots_with_susceptible_access_available"])
        rows.append(
            {
                **dict(zip(group_columns, key)),
                "n_species": int(group["species"].nunique()),
                "n_evaluable_sample_folds": int(sums["n_evaluable_sample_folds"]),
                "n_sample_folds_with_recommendation": int(
                    sums["n_sample_folds_with_recommendation"]
                ),
                "recommendation_coverage": _safe_rate(
                    int(sums["n_sample_folds_with_recommendation"]),
                    int(sums["n_evaluable_sample_folds"]),
                ),
                "n_recommended_slots": n_slots,
                "recommended_true_susceptible_pct": _safe_rate(
                    int(round(group["n_recommended_true_susceptible"].sum())),
                    n_slots,
                ),
                "recommended_true_resistant_pct": _safe_rate(
                    int(round(group["n_recommended_true_resistant"].sum())),
                    n_slots,
                ),
                "generation_overuse_pct_all_recommendations": _safe_rate(
                    int(round(group["n_generation_overuse"].sum())), n_slots
                ),
                "generation_overuse_pct_comparable": _safe_rate(
                    int(round(group["n_generation_overuse"].sum())),
                    n_generation_comparable,
                ),
                "n_generation_comparable_slots": n_generation_comparable,
                "watch_or_reserve_when_access_available_pct_all_recommendations": _safe_rate(
                    int(round(group["n_watch_or_reserve_when_access_available"].sum())),
                    n_slots,
                ),
                "watch_or_reserve_when_access_available_pct_access_available": _safe_rate(
                    int(round(group["n_watch_or_reserve_when_access_available"].sum())),
                    n_access_available,
                ),
                "watch_when_access_available_pct_access_available": _safe_rate(
                    int(round(group["n_watch_when_access_available"].sum())),
                    n_access_available,
                ),
                "reserve_when_access_available_pct_access_available": _safe_rate(
                    int(round(group["n_reserve_when_access_available"].sum())),
                    n_access_available,
                ),
                "n_slots_with_susceptible_access_available": n_access_available,
            }
        )
    result = pd.DataFrame(rows)
    result["_scenario_order"] = result["scenario"].map(SCENARIO_ORDER).fillna(99)
    return (
        result.sort_values(["_scenario_order", "system_label", "ranking", "k"])
        .drop(columns="_scenario_order")
        .reset_index(drop=True)
    )


def _fixed_width_table(frame: pd.DataFrame, columns: list[str], rename: dict[str, str]) -> str:
    display = frame[columns].rename(columns=rename).copy()
    for column in display.columns:
        if column.startswith("%") or column in {"cobertura"}:
            display[column] = display[column].map(_format_percent)
        elif pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: _format_float(value, 3))
        else:
            display[column] = display[column].map(str)
    widths = {
        column: max(len(str(column)), *(len(value) for value in display[column]))
        for column in display.columns
    }
    lines = [
        " ".join(str(column).rjust(widths[column]) for column in display.columns)
    ]
    for row in display.itertuples(index=False):
        lines.append(
            " ".join(
                str(value).rjust(widths[column])
                for column, value in zip(display.columns, row)
            )
        )
    return "\n".join(lines)


def write_text_report(output_path: Path, global_df: pd.DataFrame, by_species: pd.DataFrame) -> None:
    clinical_columns = [
        "scenario",
        "system_label",
        "k",
        "recommendation_coverage",
        "recommended_true_susceptible_pct",
        "recommended_true_resistant_pct",
        "generation_overuse_pct_comparable",
        "reserve_when_access_available_pct_access_available",
        "n_recommended_slots",
    ]
    rename = {
        "scenario": "escenario",
        "system_label": "sistema",
        "k": "K",
        "recommendation_coverage": "cobertura",
        "recommended_true_susceptible_pct": "% susceptible_real",
        "recommended_true_resistant_pct": "% resistente_real",
        "generation_overuse_pct_comparable": "% gen_mayor",
        "reserve_when_access_available_pct_access_available": "% Reserve habiendo Access",
        "n_recommended_slots": "n_recs",
    }
    sections = [
        "PORCENTAJES CLÍNICOS DE LAS RECOMENDACIONES",
        "=" * 150,
        "",
        "Definiciones:",
        "- % susceptible_real: recomendaciones cuyo AST observado era susceptible.",
        "- % gen_mayor: recomendaciones con generación mayor que alguna alternativa susceptible de la misma familia.",
        "- % Reserve habiendo Access: recomendaciones Reserve cuando existía al menos una alternativa Access susceptible.",
        "- Se excluyen AST missing y antifúngicos. En clinical_reranked solo cuentan recomendaciones con pR < umbral y score clínico > 0.",
        "",
        "1. GLOBAL - RANKING CLÍNICO",
        "-" * 150,
        _fixed_width_table(
            global_df[global_df["ranking"] == "clinical_reranked"],
            clinical_columns,
            rename,
        ),
        "",
        "2. GLOBAL - RANKING BRUTO",
        "-" * 150,
        _fixed_width_table(
            global_df[global_df["ranking"] == "raw"],
            clinical_columns,
            rename,
        ),
        "",
        "3. POR ESPECIE - RANKING CLÍNICO",
        "=" * 150,
    ]
    for scenario in ("finetuned", "zero_shot"):
        sections.extend(["", scenario.upper(), "-" * 150])
        for species in sorted(by_species["species"].unique()):
            subset = by_species[
                (by_species["scenario"] == scenario)
                & (by_species["species"] == species)
                & (by_species["ranking"] == "clinical_reranked")
            ]
            sections.extend(
                [
                    "",
                    species.upper(),
                    _fixed_width_table(subset, clinical_columns, rename),
                ]
            )
    output_path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    species_names = tuple(dict.fromkeys(_normalise_species(s) for s in args.species))
    scenarios = tuple(dict.fromkeys(args.scenarios))
    k_values = tuple(sorted(set(args.k)))
    selected = pd.read_csv(
        args.comparison_dir / "selected_thresholds_sensitivity_specificity.csv"
    )
    antibiotics = _load_antibiotics(args.results_root)
    config = AssessmentConfig(
        uncertainty_lambda=args.uncertainty_lambda,
        generation_lambda=args.generation_lambda,
        aware_lambda=args.aware_lambda,
        thresholds=tuple(DEFAULT_THRESHOLDS),
        k_values=k_values,
    )

    by_species = recommendation_rows(
        args.results_root,
        antibiotics,
        selected,
        species_names,
        scenarios,
        k_values,
        config,
    )
    global_df = global_summary(by_species)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_species.to_csv(args.output_dir / "recommendation_percentages_by_species.csv", index=False)
    global_df.to_csv(args.output_dir / "recommendation_percentages_global.csv", index=False)
    write_text_report(
        args.output_dir / "PORCENTAJES_RECOMENDACIONES_CLINICAS.txt",
        global_df,
        by_species,
    )
    print(f"Recommendation percentages written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
