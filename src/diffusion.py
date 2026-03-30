from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _extract(arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = arr.gather(0, t)
    return out.reshape((x_shape[0],) + (1,) * (len(x_shape) - 1))


@dataclass
class SampleResult:
    samples: torch.Tensor
    trajectory: Optional[torch.Tensor] = None
    trajectory_steps: Optional[np.ndarray] = None


class DDPM:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str = "cpu",
    ) -> None:
        self.timesteps = int(timesteps)
        self.device = torch.device(device)

        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alphas_cumprod = alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).clamp(min=1e-20)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def loss(
        self,
        model: torch.nn.Module,
        x_start: torch.Tensor,
        c: torch.Tensor,
        y_tail: torch.Tensor,
        tail_weight: float,
        normal_weight: float,
    ) -> torch.Tensor:
        bsz = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (bsz,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred_noise = model(x_t, t, c)

        per_sample_mse = (pred_noise - noise).pow(2).mean(dim=(1, 2))
        weights = torch.where(
            y_tail > 0,
            torch.full_like(per_sample_mse, float(tail_weight)),
            torch.full_like(per_sample_mse, float(normal_weight)),
        )
        return (per_sample_mse * weights).mean()

    @torch.no_grad()
    def p_sample(self, model: torch.nn.Module, x: torch.Tensor, t: int, c: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        t_batch = torch.full((bsz,), t, device=x.device, dtype=torch.long)

        beta_t = _extract(self.betas, t_batch, x.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
        sqrt_recip_alpha = _extract(self.sqrt_recip_alphas, t_batch, x.shape)

        eps_theta = model(x, t_batch, c)
        model_mean = sqrt_recip_alpha * (x - beta_t * eps_theta / sqrt_one_minus)

        if t == 0:
            return model_mean

        posterior_var_t = _extract(self.posterior_variance, t_batch, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        condition: torch.Tensor,
        window_length: int,
        n_assets: int,
        n_samples: int,
        return_trajectory: bool = False,
    ) -> SampleResult:
        device = condition.device
        x = torch.randn((n_samples, window_length, n_assets), device=device, dtype=torch.float32)

        if not return_trajectory:
            for t in reversed(range(self.timesteps)):
                x = self.p_sample(model, x, t=t, c=condition)
            return SampleResult(samples=x)

        states = [x.detach().clone()]  # x_T
        steps = [self.timesteps]
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, t=t, c=condition)
            states.append(x.detach().clone())  # x_t
            steps.append(t)

        trajectory = torch.stack(states, dim=0)
        trajectory_steps = np.asarray(steps, dtype=np.int32)
        return SampleResult(samples=x, trajectory=trajectory, trajectory_steps=trajectory_steps)
