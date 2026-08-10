from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SpeciesAwareMultiOutputDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        species_codes: np.ndarray,
        amr: np.ndarray,
        sample_indices: np.ndarray,
        antibiotic_indices: list[int],
    ) -> None:
        y = amr[np.ix_(sample_indices, antibiotic_indices)]
        self.X = torch.as_tensor(X[sample_indices], dtype=torch.float32)
        self.species_codes = torch.as_tensor(species_codes[sample_indices], dtype=torch.long)
        self.y = torch.as_tensor(np.nan_to_num(y, nan=0.0), dtype=torch.float32)
        self.mask = torch.as_tensor(np.isfinite(y), dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.species_codes[idx], self.y[idx], self.mask[idx]

