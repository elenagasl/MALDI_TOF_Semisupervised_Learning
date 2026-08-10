from __future__ import annotations

from dataclasses import dataclass


MODEL_NAME = "species_aware_global_mlp"
DEFAULT_EXCLUDED_SPECIES = ("Candida albicans", "Streptococcus pneumoniae")


@dataclass(frozen=True)
class ExperimentConfig:
    n_folds: int = 5
    random_seed: int = 42
    min_train_samples: int = 100
    min_val_samples: int = 50
    val_size: float = 0.2
    ood_adaptation_fraction: float = 0.2
    finetune_val_size: float = 0.25


@dataclass(frozen=True)
class ModelConfig:
    maldi_hidden_dims: tuple[int, ...] = (512, 256, 128)
    species_embedding_dim: int = 32
    head_hidden_dims: tuple[int, ...] = (128, 64)
    activation: str = "relu"
    dropout: float = 0.1
    optimizer: str = "adam"
    learning_rate: float = 1e-4


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    prediction_batch_size: int = 128
    max_epochs: int = 1200
    patience: int = 30
    finetune_epochs: int = 300
    finetune_patience: int = 10
    weight_decay: float = 1e-5
    early_stopping_metric: str = "patient_auc"
    num_workers: int = 0
    pin_memory: bool = True

