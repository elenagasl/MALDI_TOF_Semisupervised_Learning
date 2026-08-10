from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLPConfig:
    hidden_dims: tuple[int, int, int]
    activation: str
    optimizer: str
    learning_rate: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 1200
    patience: int = 15
    weight_decay: float = 0.0
    num_workers: int = 0
    pin_memory: bool = True


@dataclass(frozen=True)
class RecommenderConfig:
    maldi_dims: tuple[int, ...] = (6000, 512, 256, 128, 64)
    antibiotic_embedding_dim: int = 30
    interaction_dims: tuple[int, ...] = (64, 32, 1)
    activation: str = "relu"
    optimizer: str = "adam"
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class ExperimentConfig:
    n_folds: int = 5
    random_seed: int = 42
    min_train_samples: int = 100
    min_val_samples: int = 50
    val_size: float = 0.2
    mlp_hidden_dims: tuple[int, int, int] = (512, 256, 128)
    mlp_activation: str = "relu"
    mlp_optimizer: str = "adam"
    mlp_learning_rate: float = 1e-3
