import numpy as np
import torch
from torch.utils.data import Dataset

class RecommenderDataset(Dataset):
    def __init__(self, X, Y):
        self.samples = []

        N, A = Y.shape

        for i in range(N):
            for j in range(A):
                if np.isnan(Y[i, j]):
                    continue

                self.samples.append({
                    "intensity": X[i],
                    "drug": j,
                    "label": Y[i, j],
                    "loc": i,
                    "drug_name": j
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "intensity": torch.tensor(s["intensity"], dtype=torch.float32),
            "mz": torch.zeros_like(torch.tensor(s["intensity"], dtype=torch.float32)),
            "drug": torch.tensor(s["drug"], dtype=torch.long),
            "label": torch.tensor(s["label"], dtype=torch.float32),
            "loc": s["loc"],
            "drug_name": s["drug_name"]
        }