#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from grasp_mlp.config import DEFAULT_EXCLUDED_SPECIES, ExperimentConfig, ModelConfig, TrainingConfig
from grasp_mlp.experiment import run_ood_experiment
from grasp_mlp.io import align_payloads, load_pickle
from grasp_mlp.training import get_device


DEFAULT_IN_PICKLE = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl"
DEFAULT_OOD_PICKLE = "/export/data_ml4ds/bacteria_id/MALDIAlign_Alex/MSUMG_study_full.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OOD species-aware global MLP experiment.")
    parser.add_argument("--in-pickle", default=DEFAULT_IN_PICKLE)
    parser.add_argument("--ood-pickle", default=DEFAULT_OOD_PICKLE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-train-samples", type=int, default=100)
    parser.add_argument("--min-val-samples", type=int, default=50)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--ood-adaptation-fraction", type=float, default=0.2)
    parser.add_argument("--finetune-val-size", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=300)
    parser.add_argument("--finetune-patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--species-embedding-dim", type=int, default=32)
    parser.add_argument("--early-stopping-metric", choices=["patient_auc", "loss"], default="patient_auc")
    parser.add_argument("--evaluation-panel", choices=["species", "global"], default="species")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--exclude-species",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_SPECIES),
        help="Species names to remove before training/evaluation.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["species_aware_global_mlp"],
        choices=["species_aware_global_mlp"],
        help="Only species_aware_global_mlp is implemented in this experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ind = load_pickle(args.in_pickle)
    ood = load_pickle(args.ood_pickle)
    ind, ood = align_payloads(ind, ood)
    device = get_device(args.device)
    exp_cfg = ExperimentConfig(
        n_folds=args.n_folds,
        random_seed=args.random_seed,
        min_train_samples=args.min_train_samples,
        min_val_samples=args.min_val_samples,
        val_size=args.val_size,
        ood_adaptation_fraction=args.ood_adaptation_fraction,
        finetune_val_size=args.finetune_val_size,
    )
    model_cfg = ModelConfig(
        species_embedding_dim=args.species_embedding_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        prediction_batch_size=args.prediction_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        finetune_epochs=args.finetune_epochs,
        finetune_patience=args.finetune_patience,
        weight_decay=args.weight_decay,
        early_stopping_metric=args.early_stopping_metric,
        num_workers=args.num_workers,
    )
    metrics = run_ood_experiment(
        ind=ind,
        ood=ood,
        output_dir=Path(args.output_dir),
        exp_cfg=exp_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        device=device,
        excluded_species=tuple(args.exclude_species),
        evaluation_panel_mode=args.evaluation_panel,
    )
    print(metrics)


if __name__ == "__main__":
    main()

