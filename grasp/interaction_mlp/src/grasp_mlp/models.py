from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


def activation_layer(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "tanh":
        return nn.Tanh()
    if key == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def make_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    activation: str,
    dropout: float,
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


class SpeciesAwareGlobalMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_species: int,
        n_antibiotics: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.maldi_encoder = make_mlp(
            input_dim=input_dim,
            hidden_dims=cfg.maldi_hidden_dims[:-1],
            output_dim=cfg.maldi_hidden_dims[-1],
            activation=cfg.activation,
            dropout=cfg.dropout,
        )
        self.species_embedding = nn.Embedding(n_species, cfg.species_embedding_dim)
        head_input_dim = cfg.maldi_hidden_dims[-1] + cfg.species_embedding_dim
        self.prediction_head = make_mlp(
            input_dim=head_input_dim,
            hidden_dims=cfg.head_hidden_dims,
            output_dim=n_antibiotics,
            activation=cfg.activation,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor, species_idx: torch.Tensor) -> torch.Tensor:
        maldi_embedding = self.maldi_encoder(x.float())
        species_embedding = self.species_embedding(species_idx.long())
        return self.prediction_head(torch.cat([maldi_embedding, species_embedding], dim=-1))

