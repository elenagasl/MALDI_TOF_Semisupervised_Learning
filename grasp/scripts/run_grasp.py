#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from grasp_global.config import ExperimentConfig, GRASPConfig, TrainingConfig
from grasp_global.experiment import MODEL_NAMES, run_experiment
from grasp_global.io import load_combined_pickle
from grasp_global.training import get_device


DEFAULT_PICKLE = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global GRASP recommender experiments.")
    parser.add_argument("--pickle", default=DEFAULT_PICKLE, help="Combined in-distribution pickle path.")
    parser.add_argument("--output-dir", required=True, help="Directory where metrics and predictions are saved.")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cuda:0, mps, or cpu.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-train-samples", type=int, default=100)
    parser.add_argument("--min-val-samples", type=int, default=50)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--early-stopping-metric",
        choices=["patient_auc", "loss"],
        default="patient_auc",
        help="Metric used to keep the best epoch. patient_auc maximizes validation sample-level AUC.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--exclude-species",
        nargs="*",
        default=["Candida albicans", "Streptococcus pneumoniae"],
        help="Species names to remove before training/evaluation.",
    )
    parser.add_argument(
        "--evaluation-panel",
        choices=["species", "global"],
        default="species",
        help="Use species to evaluate on the same species/fold sparse panels as semi-supervised baselines.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODEL_NAMES),
        choices=list(MODEL_NAMES),
        help="GRASP architectures to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_combined_pickle(args.pickle)
    device = get_device(args.device)

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
        early_stopping_metric=args.early_stopping_metric,
        num_workers=args.num_workers,
    )
    grasp_cfg = GRASPConfig(
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )

    metrics = run_experiment(
        payload=payload,
        output_dir=Path(args.output_dir),
        exp_cfg=exp_cfg,
        train_cfg=train_cfg,
        grasp_cfg=grasp_cfg,
        device=device,
        model_names=tuple(args.models),
        excluded_species=tuple(args.exclude_species),
        evaluation_panel_mode=args.evaluation_panel,
    )
    print(metrics)


if __name__ == "__main__":
    main()
