import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TimeSeriesTransformerModel

class TCNBlock(nn.Module):
    """Temporal Convolutional Block with residual connection and layer normalization."""
    def __init__(self, input_size, hidden_size, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation//2)
        self.norm = nn.LayerNorm(hidden_size)  # LayerNorm is better for time-series
        self.relu = nn.GELU()  # GELU is smoother than ReLU
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(input_size, hidden_size, 1) if input_size != hidden_size else nn.Identity()

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        residual = self.residual(x.transpose(1, 2)).transpose(1, 2)
        
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        out = self.conv(x)
        out = out.transpose(1, 2)  # (batch, seq_len, hidden_size)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Add residual
        out = out + residual
        return out

class TCNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TCNBlock(input_size if i == 0 else hidden_size, hidden_size, dilation=2**i, dropout=dropout))
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # Global average pooling
        x = x.transpose(1, 2)  # (batch, hidden_size, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, hidden_size)
        return x

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

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
    """Multi-task prediction head with deeper layers for better feature extraction."""
    def __init__(self, input_size, hidden_size=64, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)  # up/down
        )
        # Regression head - separate pathway
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # return prediction
        )

    def forward(self, x):
        shared = self.shared(x)
        cls_out = self.classifier(shared)
        reg_out = self.regressor(shared)
        return cls_out, reg_out

class SofareM3(nn.Module):
    """
    SOFARE Multi-Modal Model (M3) for cryptocurrency price prediction.
    
    Supports configurable hyperparameters for tuning:
    - hidden_size: Hidden layer dimension
    - embed_dim: Embedding dimension (must be divisible by num_heads)
    - num_heads: Number of attention heads
    - num_encoder_layers: Number of encoder layers
    - dropout: Dropout rate
    """
    def __init__(
        self, 
        micro_input_size, 
        macro_input_size, 
        safe_input_size, 
        hidden_size=128, 
        embed_dim=128,
        num_heads=4,
        num_encoder_layers=2,
        dropout=0.1
    ):
        super().__init__()
        
        # Validate embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            embed_dim = (embed_dim // num_heads) * num_heads
            
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        
        # Encoders
        self.micro_encoder = TimeSeriesTransformerModel.from_pretrained('huggingface/time-series-transformer-tourism-monthly')
        # Note: This is a placeholder; in practice, you'd need to adjust for your data
        # For now, we'll use a simple projection
        self.micro_proj = nn.Linear(micro_input_size, embed_dim)
        self.micro_dropout = nn.Dropout(dropout)
        
        # Macro encoder with configurable layers and LayerNorm
        macro_layers = []
        in_dim = macro_input_size
        for i in range(num_encoder_layers):
            out_dim = hidden_size if i < num_encoder_layers - 1 else embed_dim
            macro_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        self.macro_encoder = nn.Sequential(*macro_layers)
        
        # Safe encoder with configurable layers and LayerNorm
        safe_layers = []
        in_dim = safe_input_size
        for i in range(num_encoder_layers):
            out_dim = hidden_size if i < num_encoder_layers - 1 else embed_dim
            safe_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        self.safe_encoder = nn.Sequential(*safe_layers)
        
        # Fusion with configurable attention
        self.fusion = AttentionFusion(embed_dim, num_heads=num_heads, dropout=dropout)
        
        # Head with deeper layers
        self.head = MultiTaskHead(embed_dim, hidden_size=hidden_size, dropout=dropout)

    def forward(self, x_micro, x_macro, x_safe):
        # Encode modalities
        # Micro: Use transformer (simplified)
        micro_out = self.micro_proj(x_micro.mean(dim=1))  # Simple pooling for now
        micro_out = self.micro_dropout(micro_out)
        
        macro_out = self.macro_encoder(x_macro)
        
        safe_out = self.safe_encoder(x_safe)
        
        # Fusion
        modalities = [micro_out, macro_out, safe_out]
        fused = self.fusion(modalities)
        
        # Multi-task output
        cls_pred, reg_pred = self.head(fused)
        return cls_pred, reg_pred
