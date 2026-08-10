from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def cfg_get(cfg, name: str, default):
    return getattr(cfg, name, default)


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
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str,
    dropout: float = 0.0,
    final_activation: bool = False,
) -> nn.Sequential:
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation_layer(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if final_activation:
        layers.append(activation_layer(activation))
    return nn.Sequential(*layers)


class GlobalMALDIEncoder(nn.Module):
    """Shared MALDI encoder used by species-conditioned and hypernetwork GRASP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 32,
        activation: str = "gelu",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.encoder = make_mlp(
            input_dim=input_dim,
            hidden_dims=(512, 128, 64),
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
            final_activation=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.float())


class SpeciesHeadMALDIEncoder(nn.Module):
    """Shared MALDI backbone followed by one MALDI head per species."""

    def __init__(
        self,
        input_dim: int,
        n_species: int,
        shared_dims: Sequence[int] = (512, 128),
        head_dims: Sequence[int] = (64,),
        output_dim: int = 32,
        activation: str = "gelu",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if len(shared_dims) == 0:
            raise ValueError("shared_dims must contain at least one layer")
        self.output_dim = output_dim
        self.backbone = make_mlp(
            input_dim=input_dim,
            hidden_dims=shared_dims[:-1],
            output_dim=shared_dims[-1],
            activation=activation,
            dropout=dropout,
            final_activation=True,
        )
        self.species_heads = nn.ModuleList(
            [
                make_mlp(
                    input_dim=shared_dims[-1],
                    hidden_dims=head_dims,
                    output_dim=output_dim,
                    activation=activation,
                    dropout=dropout,
                    final_activation=True,
                )
                for _ in range(n_species)
            ]
        )

    def forward(self, x: torch.Tensor, species_idx: torch.Tensor) -> torch.Tensor:
        shared = self.backbone(x.float())
        species_idx = species_idx.long()
        out = torch.empty(shared.shape[0], self.output_dim, dtype=shared.dtype, device=shared.device)
        for species_code in torch.unique(species_idx):
            mask = species_idx == species_code
            out[mask] = self.species_heads[int(species_code.item())](shared[mask])
        return out


class SpeciesSpecificInteractor(nn.Module):
    """One interaction MLP per species, matching the benchmark-style global models."""

    def __init__(
        self,
        n_species: int,
        input_dim: int,
        hidden_dims: Sequence[int] = (32,),
        activation: str = "gelu",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mlps = nn.ModuleList(
            [
                make_mlp(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=1,
                    activation=activation,
                    dropout=dropout,
                    final_activation=False,
                )
                for _ in range(n_species)
            ]
        )

    def forward(self, x: torch.Tensor, species_idx: torch.Tensor) -> torch.Tensor:
        species_idx = species_idx.long()
        out = torch.empty(x.shape[0], 1, dtype=x.dtype, device=x.device)
        for species_code in torch.unique(species_idx):
            mask = species_idx == species_code
            out[mask] = self.mlps[int(species_code.item())](x[mask])
        return out.squeeze(-1)


class SpeciesConditionedGRASP(nn.Module):
    """Global recommender with species embedding concatenated to MALDI and antibiotic embeddings."""

    def __init__(
        self,
        input_dim: int,
        n_antibiotics: int,
        n_species: int,
        maldi_embedding_dim: int = 32,
        antibiotic_embedding_dim: int = 16,
        species_embedding_dim: int = 16,
        interaction_hidden_dim: int = 32,
        activation: str = "gelu",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.maldi_encoder = GlobalMALDIEncoder(input_dim, maldi_embedding_dim, activation, dropout)
        self.antibiotic_embedding = nn.Embedding(n_antibiotics, antibiotic_embedding_dim)
        self.species_embedding = nn.Embedding(n_species, species_embedding_dim)
        self.interaction = make_mlp(
            input_dim=maldi_embedding_dim + antibiotic_embedding_dim + species_embedding_dim,
            hidden_dims=(interaction_hidden_dim,),
            output_dim=1,
            activation=activation,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        species_idx: torch.Tensor,
        antibiotic_idx: torch.Tensor,
    ) -> torch.Tensor:
        z_maldi = self.maldi_encoder(x)
        z_drug = self.antibiotic_embedding(antibiotic_idx.long())
        z_species = self.species_embedding(species_idx.long())
        return self.interaction(torch.cat([z_maldi, z_drug, z_species], dim=-1)).squeeze(-1)


class MultiHeadGRASP(nn.Module):
    """Benchmark-style global multihead recommender."""

    def __init__(
        self,
        input_dim: int,
        n_antibiotics: int,
        n_species: int,
        shared_dims: Sequence[int] = (512, 128),
        head_dims: Sequence[int] = (64,),
        maldi_embedding_dim: int = 32,
        antibiotic_embedding_dim: int = 16,
        interaction_hidden_dim: int = 32,
        activation: str = "gelu",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.maldi_encoder = SpeciesHeadMALDIEncoder(
            input_dim=input_dim,
            n_species=n_species,
            shared_dims=shared_dims,
            head_dims=head_dims,
            output_dim=maldi_embedding_dim,
            activation=activation,
            dropout=dropout,
        )
        self.antibiotic_embedding = nn.Embedding(n_antibiotics, antibiotic_embedding_dim)
        self.interaction = SpeciesSpecificInteractor(
            n_species=n_species,
            input_dim=maldi_embedding_dim + antibiotic_embedding_dim,
            hidden_dims=(interaction_hidden_dim,),
            activation=activation,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        species_idx: torch.Tensor,
        antibiotic_idx: torch.Tensor,
    ) -> torch.Tensor:
        z_maldi = self.maldi_encoder(x, species_idx)
        z_drug = self.antibiotic_embedding(antibiotic_idx.long())
        return self.interaction(torch.cat([z_maldi, z_drug], dim=-1), species_idx)


class HypernetworkGRASP(nn.Module):
    """Benchmark-style hypernetwork GRASP with a learned alpha-scaled species correction."""

    def __init__(
        self,
        input_dim: int,
        n_antibiotics: int,
        n_species: int,
        maldi_embedding_dim: int = 32,
        antibiotic_embedding_dim: int = 16,
        species_embedding_dim: int = 16,
        hyper_hidden_dims: Sequence[int] = (64, 64),
        interaction_hidden_dim: int = 32,
        alpha_init: float = 0.05,
        activation: str = "gelu",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.maldi_encoder = GlobalMALDIEncoder(input_dim, maldi_embedding_dim, activation, dropout)
        self.species_embedding = nn.Embedding(n_species, species_embedding_dim)
        self.hypernetwork = make_mlp(
            input_dim=species_embedding_dim,
            hidden_dims=hyper_hidden_dims,
            output_dim=maldi_embedding_dim,
            activation=activation,
            dropout=dropout,
            final_activation=False,
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.antibiotic_embedding = nn.Embedding(n_antibiotics, antibiotic_embedding_dim)
        self.interaction = SpeciesSpecificInteractor(
            n_species=n_species,
            input_dim=maldi_embedding_dim + antibiotic_embedding_dim,
            hidden_dims=(interaction_hidden_dim,),
            activation=activation,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        species_idx: torch.Tensor,
        antibiotic_idx: torch.Tensor,
    ) -> torch.Tensor:
        z_maldi = self.maldi_encoder(x)
        z_species = self.species_embedding(species_idx.long())
        delta_species = self.hypernetwork(z_species)
        z_corrected = z_maldi + self.alpha * delta_species
        z_drug = self.antibiotic_embedding(antibiotic_idx.long())
        return self.interaction(torch.cat([z_corrected, z_drug], dim=-1), species_idx)


def build_model(
    model_name: str,
    input_dim: int,
    n_antibiotics: int,
    n_species: int,
    cfg,
) -> nn.Module:
    maldi_embedding_dim = cfg_get(cfg, "global_maldi_embedding_dim", 32)
    antibiotic_embedding_dim = cfg_get(cfg, "antibiotic_embedding_dim", 16)
    species_embedding_dim = cfg_get(cfg, "species_embedding_dim", 16)
    interaction_hidden_dim = cfg_get(cfg, "interaction_hidden_dim", 32)
    activation = cfg_get(cfg, "activation", "gelu")
    dropout = cfg_get(cfg, "dropout", 0.2)

    if model_name == "species_conditioned_grasp":
        return SpeciesConditionedGRASP(
            input_dim=input_dim,
            n_antibiotics=n_antibiotics,
            n_species=n_species,
            maldi_embedding_dim=maldi_embedding_dim,
            antibiotic_embedding_dim=antibiotic_embedding_dim,
            species_embedding_dim=species_embedding_dim,
            interaction_hidden_dim=interaction_hidden_dim,
            activation=activation,
            dropout=dropout,
        )
    if model_name == "multihead_grasp":
        return MultiHeadGRASP(
            input_dim=input_dim,
            n_antibiotics=n_antibiotics,
            n_species=n_species,
            shared_dims=cfg_get(cfg, "multihead_shared_dims", (512, 128)),
            head_dims=cfg_get(cfg, "multihead_head_dims", (64,)),
            maldi_embedding_dim=maldi_embedding_dim,
            antibiotic_embedding_dim=antibiotic_embedding_dim,
            interaction_hidden_dim=interaction_hidden_dim,
            activation=activation,
            dropout=dropout,
        )
    if model_name == "hypernetwork_grasp":
        return HypernetworkGRASP(
            input_dim=input_dim,
            n_antibiotics=n_antibiotics,
            n_species=n_species,
            maldi_embedding_dim=maldi_embedding_dim,
            antibiotic_embedding_dim=antibiotic_embedding_dim,
            species_embedding_dim=species_embedding_dim,
            hyper_hidden_dims=cfg_get(cfg, "hyper_hidden_dims", (64, 64)),
            interaction_hidden_dim=interaction_hidden_dim,
            alpha_init=cfg_get(cfg, "alpha_init", 0.05),
            activation=activation,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model_name: {model_name}")
