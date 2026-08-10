#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ood_evaluation" / "src"))
sys.path.insert(0, str(ROOT / "supervised_vs_semisupervised_learning" / "src"))
sys.path.insert(0, str(ROOT / "graspI" / "src"))
sys.path.insert(0, str(ROOT / "grasp_mlp" / "src"))

from supervised_semisupervised.training import get_device

from ood_eval.config import ExperimentConfig, GRASPConfig, TrainingConfig
from ood_eval.deployment import DEPLOYMENT_MODELS, run_deployment_experiment
from ood_eval.experiment import ALL_MODELS, run_experiment
from ood_eval.io import align_payloads, filter_excluded_species, load_pickle
from ood_eval.legacy_global import LEGACY_GLOBAL_MODELS, run_legacy_global_experiment


DEFAULT_IN_PICKLE = "/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl"
DEFAULT_OOD_PICKLE = "/export/data_ml4ds/bacteria_id/MALDIAlign_Alex/MSUMG_study_full.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OOD zero-shot and fine-tuning evaluation.")
    parser.add_argument("--in-pickle", default=DEFAULT_IN_PICKLE)
    parser.add_argument("--ood-pickle", default=DEFAULT_OOD_PICKLE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--protocol",
        choices=["fair", "standard", "legacy_global_ood", "deployment_ood"],
        default="fair",
        help=(
            "fair/standard runs the paired unified comparison. "
            "legacy_global_ood reproduces the old global GRASP OOD protocol. "
            "deployment_ood compares each model family using its natural deployable coverage."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-train-samples", type=int, default=100)
    parser.add_argument("--min-val-samples", type=int, default=50)
    parser.add_argument("--legacy-min-source-samples-species", type=int, default=50)
    parser.add_argument("--legacy-min-ood-samples-species", type=int, default=50)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--legacy-source-val-size", type=float, default=0.15)
    parser.add_argument("--ood-adaptation-fraction", type=float, default=0.2)
    parser.add_argument("--finetune-val-size", type=float, default=0.25)
    parser.add_argument("--finetune-lr-factor", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=300)
    parser.add_argument("--finetune-patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--min-source-obs-per-antibiotic", type=int, default=50)
    parser.add_argument("--min-source-val-obs-per-antibiotic", type=int, default=5)
    parser.add_argument("--min-finetune-obs-per-antibiotic", type=int, default=10)
    parser.add_argument("--min-test-obs-per-antibiotic", type=int, default=10)
    parser.add_argument(
        "--early-stopping-metric",
        choices=["patient_auc", "loss"],
        default="patient_auc",
        help="Metric used by GRASP early stopping. MLP baselines continue using validation loss.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--exclude-species",
        nargs="*",
        default=["Candida albicans", "Streptococcus pneumoniae"],
        help="Species names to remove before training/evaluation.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=sorted(set(ALL_MODELS) | set(LEGACY_GLOBAL_MODELS) | set(DEPLOYMENT_MODELS)),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ind = load_pickle(args.in_pickle)
    ood = load_pickle(args.ood_pickle)
    ind, ood = align_payloads(ind, ood)
    ind = filter_excluded_species(ind, tuple(args.exclude_species))
    ood = filter_excluded_species(ood, tuple(args.exclude_species))
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
    grasp_cfg = GRASPConfig(
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    if args.protocol == "legacy_global_ood":
        model_names = tuple(args.models) if args.models else ("multihead_grasp", "hypernetwork_grasp")
        metrics = run_legacy_global_experiment(
            ind=ind,
            ood=ood,
            output_dir=Path(args.output_dir),
            exp_cfg=exp_cfg,
            train_cfg=train_cfg,
            grasp_cfg=grasp_cfg,
            device=device,
            model_names=model_names,
            source_val_size=args.legacy_source_val_size,
            finetune_val_size=args.finetune_val_size,
            finetune_lr_factor=args.finetune_lr_factor,
            min_source_samples_species=args.legacy_min_source_samples_species,
            min_ood_samples_species=args.legacy_min_ood_samples_species,
            min_source_obs_per_antibiotic=args.min_source_obs_per_antibiotic,
            min_source_val_obs_per_antibiotic=args.min_source_val_obs_per_antibiotic,
            min_finetune_obs_per_antibiotic=args.min_finetune_obs_per_antibiotic,
            min_test_obs_per_antibiotic=args.min_test_obs_per_antibiotic,
        )
    elif args.protocol == "deployment_ood":
        model_names = tuple(args.models) if args.models else tuple(ALL_MODELS)
        metrics = run_deployment_experiment(
            ind=ind,
            ood=ood,
            output_dir=Path(args.output_dir),
            exp_cfg=exp_cfg,
            train_cfg=train_cfg,
            grasp_cfg=grasp_cfg,
            device=device,
            model_names=model_names,
            source_val_size=args.legacy_source_val_size,
            finetune_val_size=args.finetune_val_size,
            min_source_samples_species=args.legacy_min_source_samples_species,
            min_ood_samples_species=args.legacy_min_ood_samples_species,
            min_source_obs_per_antibiotic=args.min_source_obs_per_antibiotic,
            min_source_val_obs_per_antibiotic=args.min_source_val_obs_per_antibiotic,
            min_finetune_obs_per_antibiotic=args.min_finetune_obs_per_antibiotic,
            min_test_obs_per_antibiotic=args.min_test_obs_per_antibiotic,
        )
    else:
        model_names = tuple(args.models) if args.models else tuple(ALL_MODELS)
        metrics = run_experiment(
            ind=ind,
            ood=ood,
            output_dir=Path(args.output_dir),
            exp_cfg=exp_cfg,
            train_cfg=train_cfg,
            grasp_cfg=grasp_cfg,
            device=device,
            model_names=model_names,
        )
    print(metrics)


if __name__ == "__main__":
    main()
