from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    if half == 0:
        return t.float().unsqueeze(-1)
    device = t.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ConditionalMLPDenoiser(nn.Module):
    """
    Minimal conditional MLP denoiser.
    Input/Output shape: [B, 20, N]
    """

    def __init__(
        self,
        window_length: int,
        n_assets: int,
        cond_dim: int = 4,
        hidden_dim: int = 128,
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.window_length = int(window_length)
        self.n_assets = int(n_assets)
        self.cond_dim = int(cond_dim)
        self.input_dim = self.window_length * self.n_assets

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.x_mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.input_dim),
        )
        self.time_embed_dim = int(time_embed_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B, 20, N], got shape={tuple(x_t.shape)}")
        bsz, seq_len, n_assets = x_t.shape
        if seq_len != self.window_length:
            raise ValueError(f"window_length mismatch: expected {self.window_length}, got {seq_len}")
        if n_assets != self.n_assets:
            raise ValueError(f"n_assets mismatch: expected {self.n_assets}, got {n_assets}")
        if c.ndim != 2 or c.shape[1] != self.cond_dim:
            raise ValueError(f"c must be [B, {self.cond_dim}], got shape={tuple(c.shape)}")

        x_flat = x_t.reshape(bsz, -1)
        t_emb = timestep_embedding(t, self.time_embed_dim)
        h = self.x_mlp[0](x_flat)
        h = h + self.time_mlp(t_emb) + self.cond_mlp(c)
        h = self.x_mlp[1:](h)
        return h.reshape(bsz, self.window_length, self.n_assets)
