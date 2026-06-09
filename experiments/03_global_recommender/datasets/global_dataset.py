from torch.utils.data import Dataset
import torch
import numpy as np


class RecDataset(Dataset):
    def __init__(
        self,
        maldi,
        sample_ids,
        sample_amr,
        antibiotic_families,
        empty_context_prob=0.05,
        min_context_fraction=0.5,
        max_context_fraction=1.0
    ):
        """
        Dataset para recommender contextual.

        Cada item representa:
            (sample, target_antibiotic) -> resistance

        Inputs devueltos:
            - maldi spectrum
            - target antibiotic id
            - target antibiotic family id
            - label del target antibiotic
            - AMR context vector
            - AMR context mask

        Parámetros:
            maldi:
                array con espectros MALDI, uno por sample.

            sample_ids:
                ids de samples alineados con maldi.

            sample_amr:
                dict {sample_id: AMR vector}.

            antibiotic_families:
                array/list de tamaño num_items con family_id por antibiótico.
                Si todavía no usamos familias, puede ser todo ceros.

            empty_context_prob:
                probabilidad de entrenar con contexto AMR totalmente vacío.
                Ahora la bajamos a 0.05 para que el modelo no aprenda a ignorar AMR.

            min_context_fraction / max_context_fraction:
                rango del porcentaje de resistencias observadas que se dejan visibles.
                Ahora usamos U(0.5, 1.0), menos agresivo que U(0.0, 1.0).
        """

        self.maldi = maldi
        self.sample_ids = sample_ids
        self.sample_amr = sample_amr
        self.antibiotic_families = np.asarray(antibiotic_families)

        self.empty_context_prob = empty_context_prob
        self.min_context_fraction = min_context_fraction
        self.max_context_fraction = max_context_fraction

        self.unique_samples = np.unique(sample_ids)

        # =========================
        # Construir pares válidos:
        # sample + antibiótico observado
        # =========================
        self.pairs = []

        for sid in self.unique_samples:
            amr_vec = self.sample_amr[sid]

            observed = np.where(~np.isnan(amr_vec))[0]

            for drug_id in observed:
                value = amr_vec[drug_id]

                # Seguridad extra: solo labels binarios
                if value in [0, 1]:
                    self.pairs.append((sid, drug_id, int(value)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        sid, target_drug, label = self.pairs[idx]

        # =========================
        # MALDI del sample
        # =========================
        maldi = self.maldi[self.sample_ids == sid][0]

        # =========================
        # AMR completo del sample
        # =========================
        amr_vec = self.sample_amr[sid].copy()

        # =========================
        # Máscara de valores reales conocidos
        # =========================
        observed_mask = ~np.isnan(amr_vec)

        # =========================
        # Evitar leakage:
        # el target antibiotic NUNCA entra como contexto
        # =========================
        observed_mask[target_drug] = False

        # =========================
        # Random masking del contexto AMR
        # =========================
        if np.random.rand() < self.empty_context_prob:
            # Pocos batches sin contexto para que el modelo siga siendo capaz
            # de funcionar en one-shot, pero sin aprender a ignorar AMR.
            context_mask = np.zeros_like(observed_mask, dtype=bool)

        else:
            # Menos agresivo que U(0,1):
            # forzamos a que, cuando hay contexto, sea razonablemente informativo.
            p = np.random.uniform(
                self.min_context_fraction,
                self.max_context_fraction
            )

            random_mask = np.random.rand(len(amr_vec)) < p
            context_mask = observed_mask & random_mask

        # =========================
        # AMR input visible
        # =========================
        amr_input = amr_vec.copy()

        # Todo lo no visible se pone a 0
        amr_input[~context_mask] = 0

        # Seguridad por si quedan NaNs
        amr_input[np.isnan(amr_input)] = 0

        # =========================
        # Family del target antibiotic
        # =========================
        target_family = self.antibiotic_families[target_drug]

        return (
            torch.tensor(maldi).float(),
            torch.tensor(target_drug).long(),
            torch.tensor(target_family).long(),
            torch.tensor(label).float(),
            torch.tensor(amr_input).float(),
            torch.tensor(context_mask).float()
        )