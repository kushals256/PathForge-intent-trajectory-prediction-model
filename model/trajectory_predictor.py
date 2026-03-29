import torch
import torch.nn as nn
from model.encoder import TemporalEncoder
from model.social_attention import SocialAttention
from model.decoder import MultiModalDecoder

class TrajectoryPredictor(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, fut_len=6, K=3):
        super().__init__()

        self.encoder = TemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.social_attn = SocialAttention(hidden_dim=hidden_dim)
        self.decoder = MultiModalDecoder(hidden_dim=hidden_dim, fut_len=fut_len, K=K)

        # FIX #5: Dedicated 2-layer MLP for neighbor encoding instead of reusing encoder.input_proj
        self.social_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, hist, social, social_mask):
        B = hist.size(0)

        # 1. Encode target agent temporal history
        target_h = self.encoder(hist)

        # 2. Encode neighbors with dedicated MLP (FIX #5)
        max_n = social.size(1)
        social_flat = social.view(B * max_n, social.size(-1))
        neighbor_h_flat = self.social_encoder(social_flat)
        neighbor_h = neighbor_h_flat.view(B, max_n, -1)

        # 3. Social Attention
        context_h = self.social_attn(target_h, neighbor_h, mask=social_mask)

        # 4. Decode
        trajs, confs = self.decoder(context_h)

        return trajs, confs
