import os
import torch
from torch.utils.data import Dataset

## expecting the correct time window directory as input

class MyDataset(Dataset):
    def __init__(self, preprocessed_dir):
        self.sample_paths = sorted([
            os.path.join(preprocessed_dir, f)
            for f in os.listdir(preprocessed_dir) if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        x, y = torch.load(self.sample_paths[idx])  # x: [3, T, 64, 64], y: [1, 64, 64]
        return x, y
