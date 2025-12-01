import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TimeSeriesTransformerModel

class TCNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation//2)
        self.norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out.transpose(1, 2)  # (batch, seq_len, hidden_size)

class TCNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TCNBlock(input_size if i == 0 else hidden_size, hidden_size, dilation=2**i))
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # Global average pooling
        x = x.transpose(1, 2)  # (batch, hidden_size, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, hidden_size)
        return x

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, modalities):
        # modalities: list of (batch, embed_dim)
        # Stack them: (num_modalities, batch, embed_dim)
        stacked = torch.stack(modalities, dim=0)
        # Transpose to batch_first: (batch, num_modalities, embed_dim)
        stacked = stacked.transpose(0, 1)
        # Use first modality as query
        query = stacked[:, 0:1]  # (batch, 1, embed_dim)
        key_value = stacked  # (batch, num_modalities, embed_dim)
        attn_out, _ = self.attention(query, key_value, key_value)
        fused = attn_out.squeeze(1)  # (batch, embed_dim)
        fused = self.norm(fused)
        fused = self.dropout(fused)
        return fused

class MultiTaskHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classifier = nn.Linear(input_size, 2)  # up/down
        self.regressor = nn.Linear(input_size, 1)   # return

    def forward(self, x):
        cls_out = self.classifier(x)
        reg_out = self.regressor(x)
        return cls_out, reg_out

class SofareM3(nn.Module):
    def __init__(self, micro_input_size, macro_input_size, safe_input_size, hidden_size=128, embed_dim=128):
        super().__init__()
        
        # Encoders
        self.micro_encoder = TimeSeriesTransformerModel.from_pretrained('huggingface/time-series-transformer-tourism-monthly')
        # Note: This is a placeholder; in practice, you'd need to adjust for your data
        # For now, we'll use a simple projection
        self.micro_proj = nn.Linear(micro_input_size, embed_dim)
        
        self.macro_encoder = nn.Sequential(
            nn.Linear(macro_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim)
        )
        
        self.safe_encoder = nn.Sequential(
            nn.Linear(safe_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim)
        )
        
        # Fusion
        self.fusion = AttentionFusion(embed_dim)
        
        # Head
        self.head = MultiTaskHead(embed_dim)

    def forward(self, x_micro, x_macro, x_safe):
        # Encode modalities
        # Micro: Use transformer (simplified)
        micro_out = self.micro_proj(x_micro.mean(dim=1))  # Simple pooling for now
        
        macro_out = self.macro_encoder(x_macro)
        
        safe_out = self.safe_encoder(x_safe)
        
        # Fusion
        modalities = [micro_out, macro_out, safe_out]
        fused = self.fusion(modalities)
        
        # Multi-task output
        cls_pred, reg_pred = self.head(fused)
        return cls_pred, reg_pred
