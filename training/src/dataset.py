import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, features, macro_features, safe_features, targets, sequence_length=60):
        self.features = torch.FloatTensor(features)
        self.macro_features = torch.FloatTensor(macro_features)
        self.safe_features = torch.FloatTensor(safe_features)
        self.targets = torch.FloatTensor(targets)  # regression targets (returns)
        self.sequence_length = sequence_length
        
        # Create classification targets (up/down)
        self.cls_targets = torch.LongTensor((targets > 0).astype(int))  # 1 for up, 0 for down

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x_seq = self.features[idx : idx + self.sequence_length]
        # Macro and safe features are taken from the prediction point
        x_macro = self.macro_features[idx + self.sequence_length]
        x_safe = self.safe_features[idx + self.sequence_length]
        y_reg = self.targets[idx + self.sequence_length]
        y_cls = self.cls_targets[idx + self.sequence_length]
        return x_seq, x_macro, x_safe, y_cls, y_reg
