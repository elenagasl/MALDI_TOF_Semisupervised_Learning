from torch.utils.data import Dataset
import torch
import numpy as np


class PredContextRecDataset(Dataset):
    """
    Dataset tipo recommender con soporte para contexto AMR global.

    Cada item representa:
        (sample, target_antibiotic) -> resistance

    Devuelve:
        - maldi spectrum
        - target antibiotic id
        - label target
        - full AMR labels vector
        - full AMR mask vector

    La full AMR matrix se usa para:
        1) entrenar la cabeza auxiliar que predice el contexto global
        2) construir contexto real enmascarado durante training
    """

    def __init__(
        self,
        maldi,
        amr
    ):
        self.maldi = np.asarray(maldi)
        self.amr = np.asarray(amr)

        if self.maldi.shape[0] != self.amr.shape[0]:
            raise ValueError(
                "maldi and amr must have the same number of samples. "
                f"Got maldi={self.maldi.shape[0]}, amr={self.amr.shape[0]}"
            )

        valid_values = (
            (self.amr == 0) |
            (self.amr == 1) |
            np.isnan(self.amr)
        )

        if not np.all(valid_values):
            raise ValueError(
                "AMR matrix contains values different from 0, 1 or NaN."
            )

        self.pairs = []

        for sample_idx in range(self.amr.shape[0]):
            amr_vec = self.amr[sample_idx]
            observed = np.where(~np.isnan(amr_vec))[0]

            for drug_id in observed:
                value = amr_vec[drug_id]

                if value in [0, 1]:
                    self.pairs.append(
                        (sample_idx, drug_id, int(value))
                    )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample_idx, target_drug, label = self.pairs[idx]

        maldi = self.maldi[sample_idx]
        amr_vec = self.amr[sample_idx].copy()

        # Mask de valores realmente observados
        amr_mask = ~np.isnan(amr_vec)

        # Labels auxiliares: NaN -> 0, pero no contará en loss porque mask = 0
        amr_labels = amr_vec.copy()
        amr_labels[~amr_mask] = 0

        return (
            torch.tensor(maldi).float(),
            torch.tensor(target_drug).long(),
            torch.tensor(label).float(),
            torch.tensor(amr_labels).float(),
            torch.tensor(amr_mask).float()
        )