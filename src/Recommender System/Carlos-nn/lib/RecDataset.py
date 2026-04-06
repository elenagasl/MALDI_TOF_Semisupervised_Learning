from torch.utils.data import Dataset
import torch
# Dataset class
class RecDataset(Dataset):
    def __init__(self, samples, metadata, items, ratings):
        self.samples = samples
        self.metadata = metadata
        self.items = items
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.metadata[idx], dtype=torch.float32), torch.tensor(self.items[idx], dtype=torch.float32), torch.tensor(self.ratings[idx], dtype=torch.float32)  