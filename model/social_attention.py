import torch
import torch.nn as nn

class SocialAttention(nn.Module):
    def __init__(self, hidden_dim=128, nhead=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)

        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, target_h, neighbor_h, mask=None):
        Q = target_h.unsqueeze(1)
        K = neighbor_h
        V = neighbor_h

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask
            all_masked = key_padding_mask.all(dim=1)
            key_padding_mask[all_masked, 0] = False

        attn_out, _ = self.mha(Q, K, V, key_padding_mask=key_padding_mask)
        attn_out = attn_out.squeeze(1)

        combined = torch.cat([target_h, attn_out], dim=1)
        out = self.combine(combined)

        # FIX #6: Residual skip connection — if social context is weak, fall back to target
        out = out + target_h

        return out
