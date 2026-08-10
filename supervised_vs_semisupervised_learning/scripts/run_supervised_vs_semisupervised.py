#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from supervised_semisupervised.config import ExperimentConfig, RecommenderConfig, TrainingConfig
from supervised_semisupervised.io import load_combined_pickle
from supervised_semisupervised.experiment import run_experiment
from supervised_semisupervised.training import get_device


DEFAULT_PICKLE = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run supervised vs semi-supervised MALDI-TOF AMR experiments."
    )
    parser.add_argument("--pickle", default=DEFAULT_PICKLE, help="Combined in-distribution pickle path.")
    parser.add_argument("--output-dir", required=True, help="Directory where metrics and predictions are saved.")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cuda:0, mps, or cpu.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-train-samples", type=int, default=100)
    parser.add_argument("--min-val-samples", type=int, default=50)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--only-species",
        nargs="*",
        default=None,
        help="Optional species names for debugging or partial reruns.",
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
        max_epochs=args.max_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
    )
    rec_cfg = RecommenderConfig()

    metrics = run_experiment(
        payload=payload,
        output_dir=Path(args.output_dir),
        exp_cfg=exp_cfg,
        train_cfg=train_cfg,
        rec_cfg=rec_cfg,
        device=device,
        only_species=args.only_species,
    )
    print(metrics)


if __name__ == "__main__":
    main()
