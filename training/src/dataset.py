import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, features, macro_features, targets, sequence_length=60):
        self.features = torch.FloatTensor(features)
        self.macro_features = torch.FloatTensor(macro_features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x_seq = self.features[idx : idx + self.sequence_length]
        # Macro features are static for the prediction point (or taken from the last step)
        # Here we take the macro state at the time of prediction (idx + sequence_length)
        x_macro = self.macro_features[idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return x_seq, x_macro, y
