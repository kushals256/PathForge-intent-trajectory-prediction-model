import torch
import torch.nn as nn

class MultiModalDecoder(nn.Module):
    def __init__(self, hidden_dim=128, fut_len=6, K=3):
        super().__init__()
        self.fut_len = fut_len
        self.K = K

        # FIX #4: Deeper 3-layer MLP with residual connection for trajectory heads
        self.traj_pre = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.traj_residual = nn.Linear(hidden_dim, hidden_dim)  # skip connection
        self.traj_out = nn.Linear(hidden_dim, K * fut_len * 2)

        # Confidence head with LayerNorm stabilization
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, K)
        )

    def forward(self, context_h):
        B = context_h.size(0)

        # Deeper trajectory decoding with residual
        h = self.traj_pre(context_h) + self.traj_residual(context_h)
        raw_trajs = self.traj_out(h)
        trajs = raw_trajs.view(B, self.K, self.fut_len, 2)

        confs = self.conf_head(context_h)

        return trajs, confs
