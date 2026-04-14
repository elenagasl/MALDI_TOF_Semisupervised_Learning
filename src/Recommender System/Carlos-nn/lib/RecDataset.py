from torch.utils.data import Dataset
import torch
# Dataset class
class RecDataset(torch.utils.data.Dataset):
    def __init__(self, maldi, hospital, sample_type, drugs, labels):
        self.maldi = maldi
        self.hospital = hospital
        self.sample_type = sample_type
        self.drugs = drugs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.maldi[idx]).float(),
            torch.tensor(self.hospital[idx]).long(),
            torch.tensor(self.sample_type[idx]).long(),
            torch.tensor(self.drugs[idx]).long(),
            torch.tensor(self.labels[idx]).float()
        )