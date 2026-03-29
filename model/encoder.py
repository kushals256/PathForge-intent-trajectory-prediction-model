import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return x

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=2, nhead=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim*4,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        returns: [batch, hidden_dim]
        """
        x = self.input_proj(x)
        
        # TransformerEncoder with batch_first=True expects [B, S, E]
        # But our PositionalEncoding expects [S, B, E]
        x = x.transpose(0, 1) # [S, B, E]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1) # [B, S, E]
        
        out = self.transformer_encoder(x)
        # Take the features from the latest timestep
        return out[:, -1, :]
