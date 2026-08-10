from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    prediction_batch_size: int = 128
    max_epochs: int = 1200
    patience: int = 30
    weight_decay: float = 1e-5
    early_stopping_metric: str = "patient_auc"
    num_workers: int = 0
    pin_memory: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    n_folds: int = 5
    random_seed: int = 42
    min_train_samples: int = 100
    min_val_samples: int = 50
    val_size: float = 0.2


@dataclass(frozen=True)
class GRASPConfig:
    global_maldi_embedding_dim: int = 32
    antibiotic_embedding_dim: int = 16
    species_embedding_dim: int = 16
    multihead_shared_dims: tuple[int, ...] = (512, 128)
    multihead_head_dims: tuple[int, ...] = (64,)
    interaction_hidden_dim: int = 32
    hyper_hidden_dims: tuple[int, ...] = (64, 64)
    alpha_init: float = 0.05
    dropout: float = 0.1
    activation: str = "gelu"
    optimizer: str = "adam"
    learning_rate: float = 1e-4
