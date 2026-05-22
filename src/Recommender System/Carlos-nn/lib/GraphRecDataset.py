from torch.utils.data import Dataset
import torch
import numpy as np


class GraphRecDataset(Dataset):
    def __init__(
        self,
        maldi,
        amr
    ):
        """
        Dataset vectorial para el modelo con graph output layer.

        Cada item representa un sample completo.

        Devuelve:
            - maldi: espectro MALDI del sample
            - labels: vector de resistencias de tamaño num_items
            - mask: vector binario de tamaño num_items

        Donde:
            labels[j] = 0/1 si la resistencia al antibiótico j está observada
            labels[j] = 0 si está missing, pero mask[j] = 0

            mask[j] = 1 si hay etiqueta real para el antibiótico j
            mask[j] = 0 si el valor era NaN

        Importante:
            La loss se calculará únicamente donde mask == 1.
        """

        self.maldi = np.asarray(maldi)
        self.amr = np.asarray(amr)

        if self.maldi.shape[0] != self.amr.shape[0]:
            raise ValueError(
                "maldi and amr must have the same number of samples. "
                f"Got maldi={self.maldi.shape[0]}, amr={self.amr.shape[0]}"
            )

        # Seguridad:
        # después del preprocesado, AMR debería contener solo 0, 1 o NaN.
        valid_values = (
            (self.amr == 0) |
            (self.amr == 1) |
            np.isnan(self.amr)
        )

        if not np.all(valid_values):
            raise ValueError(
                "AMR matrix contains values different from 0, 1 or NaN. "
                "Clean AMR values before creating GraphRecDataset."
            )

    def __len__(self):
        return self.maldi.shape[0]

    def __getitem__(self, idx):

        maldi = self.maldi[idx]
        amr_vec = self.amr[idx].copy()

        # mask = 1 donde hay dato real
        mask = ~np.isnan(amr_vec)

        # labels: NaNs se ponen a 0, pero no contarán en la loss
        labels = amr_vec.copy()
        labels[~mask] = 0

        return (
            torch.tensor(maldi).float(),
            torch.tensor(labels).float(),
            torch.tensor(mask).float()
        )