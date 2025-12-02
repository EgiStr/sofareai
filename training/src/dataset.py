import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    """
    Time series dataset for multi-task learning (classification + regression).
    
    IMPORTANT: Classification targets are computed from ORIGINAL (unscaled) returns,
    not from scaled values. This ensures correct up/down labeling.
    """
    def __init__(self, features, macro_features, safe_features, targets_scaled, sequence_length=60, original_targets=None):
        """
        Args:
            features: Scaled feature sequences
            macro_features: Scaled macro features
            safe_features: Scaled safe haven features
            targets_scaled: Scaled regression targets (for loss computation)
            sequence_length: Length of input sequences
            original_targets: ORIGINAL (unscaled) returns for classification labels
        """
        self.features = torch.FloatTensor(features)
        self.macro_features = torch.FloatTensor(macro_features)
        self.safe_features = torch.FloatTensor(safe_features)
        self.targets = torch.FloatTensor(targets_scaled)  # scaled regression targets
        self.sequence_length = sequence_length
        
        # Use original targets for classification if provided, otherwise use scaled
        # (scaled should only be used for backward compatibility)
        if original_targets is not None:
            self.cls_targets = torch.LongTensor((original_targets > 0).astype(int))
        else:
            # Fallback: assume targets are centered around 0.5 if scaled
            self.cls_targets = torch.LongTensor((targets_scaled > 0.5).astype(int))

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
