from __future__ import annotations

import torch
from torch import nn


def activation_layer(name: str) -> nn.Module:
    key = name.lower()
    if key == "identity":
        return nn.Identity()
    if key == "logistic":
        return nn.Sigmoid()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


def make_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    activation: str,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation_layer(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class BinaryMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int, int],
        activation: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.network = make_mlp(input_dim, hidden_dims, 1, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class MultiOutputMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, int, int],
        activation: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.network = make_mlp(input_dim, hidden_dims, output_dim, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SpeciesRecommender(nn.Module):
    """Species-specific NCF recommender for MALDI-antibiotic interactions."""

    def __init__(
        self,
        input_dim: int,
        n_antibiotics: int,
        maldi_hidden_dims: tuple[int, ...] = (512, 256, 128, 64),
        antibiotic_embedding_dim: int = 30,
        interaction_hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.maldi_encoder = make_mlp(
            input_dim=input_dim,
            hidden_dims=maldi_hidden_dims[:-1],
            output_dim=maldi_hidden_dims[-1],
            activation=activation,
            dropout=dropout,
        )
        self.antibiotic_embedding = nn.Embedding(n_antibiotics, antibiotic_embedding_dim)
        interaction_input_dim = maldi_hidden_dims[-1] + antibiotic_embedding_dim
        self.interaction = make_mlp(
            input_dim=interaction_input_dim,
            hidden_dims=interaction_hidden_dims,
            output_dim=1,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, antibiotic_idx: torch.Tensor) -> torch.Tensor:
        h = self.maldi_encoder(x)
        e = self.antibiotic_embedding(antibiotic_idx)
        z = torch.cat([h, e], dim=-1)
        return self.interaction(z).squeeze(-1)

