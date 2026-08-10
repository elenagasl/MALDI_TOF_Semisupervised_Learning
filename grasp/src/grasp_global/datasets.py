from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class GlobalInteractionDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        species_codes: np.ndarray,
        amr: np.ndarray,
        sample_indices: np.ndarray,
        antibiotic_indices: list[int],
    ) -> None:
        sample_indices = np.asarray(sample_indices)
        antibiotic_indices_array = np.asarray(antibiotic_indices)
        sub_amr = amr[np.ix_(sample_indices, antibiotic_indices_array)]
        rows, cols = np.where(~np.isnan(sub_amr))

        self.X = torch.as_tensor(X[sample_indices[rows]], dtype=torch.float32)
        self.species_idx = torch.as_tensor(species_codes[sample_indices[rows]], dtype=torch.long)
        self.antibiotic_idx = torch.as_tensor(cols, dtype=torch.long)
        self.y = torch.as_tensor(sub_amr[rows, cols], dtype=torch.float32)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.species_idx[idx], self.antibiotic_idx[idx], self.y[idx]

