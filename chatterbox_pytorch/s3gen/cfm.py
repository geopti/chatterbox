"""
Conditional Flow Matching for mel-spectrogram generation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .decoder import ConditionalDecoder


class CausalConditionalCFM(nn.Module):
    """
    Causal Conditional Flow Matching for mel generation.

    Implements the ODE-based generation from noise to mel-spectrogram
    using a learned velocity field.
    """

    def __init__(
        self,
        in_channels: int = 240,
        n_spks: int = 1,
        spk_emb_dim: int = 80,
        estimator: Optional[ConditionalDecoder] = None,
        sigma_min: float = 1e-4,
        t_scheduler: str = "cosine",
        training_cfg_rate: float = 0.2,
        inference_cfg_rate: float = 0.7,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.spk_emb_dim = spk_emb_dim
        self.sigma_min = sigma_min
        self.t_scheduler = t_scheduler
        self.training_cfg_rate = training_cfg_rate
        self.inference_cfg_rate = inference_cfg_rate

        # Velocity estimator (U-Net decoder)
        if estimator is None:
            estimator = ConditionalDecoder(
                in_channels=in_channels + spk_emb_dim + 80,  # x + mu + spk + cond
                out_channels=80,
            )
        self.estimator = estimator

    @property
    def dtype(self) -> torch.dtype:
        return self.estimator.dtype

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        noised_mels: Optional[torch.Tensor] = None,
        meanflow: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        """
        Generate mel-spectrogram using ODE solving.

        Args:
            mu: Encoder output of shape (batch, n_mels, time)
            mask: Output mask of shape (batch, 1, time)
            n_timesteps: Number of ODE steps
            temperature: Noise temperature
            spks: Speaker embedding of shape (batch, spk_dim)
            cond: Conditioning mel of shape (batch, n_mels, time)
            noised_mels: Pre-noised mels (for partial generation)
            meanflow: Whether to use meanflow (distilled) mode

        Returns:
            Tuple of (generated_mel, None)
        """
        # Sample initial noise
        z = torch.randn_like(mu) * temperature

        # Use pre-noised mels if provided
        if noised_mels is not None:
            prompt_len = mu.size(2) - noised_mels.size(2)
            z[..., prompt_len:] = noised_mels

        # Time steps for ODE solving
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)

        if not meanflow and self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        # Solve ODE
        if meanflow:
            return self._basic_euler(z, t_span, mu, mask, spks, cond), None
        else:
            return self._solve_euler_cfg(z, t_span, mu, mask, spks, cond), None

    def _solve_euler_cfg(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Euler ODE solver with classifier-free guidance."""
        in_dtype = x.dtype
        B, _, T = mu.shape

        # Prepare doubled batch for CFG
        x_in = torch.zeros(2 * B, 80, T, device=x.device, dtype=self.dtype)
        mask_in = torch.zeros(2 * B, 1, T, device=x.device, dtype=self.dtype)
        mu_in = torch.zeros(2 * B, 80, T, device=x.device, dtype=self.dtype)
        t_in = torch.zeros(2 * B, device=x.device, dtype=self.dtype)
        spks_in = torch.zeros(2 * B, 80, device=x.device, dtype=self.dtype)
        cond_in = torch.zeros(2 * B, 80, T, device=x.device, dtype=self.dtype)

        print("S3 Token -> Mel Inference...")
        for t, r in tqdm(zip(t_span[:-1], t_span[1:]), total=len(t_span) - 1):
            t = t.unsqueeze(0)
            r = r.unsqueeze(0)

            # Fill conditional batch (first half)
            x_in[:B] = x.to(self.dtype)
            x_in[B:] = x.to(self.dtype)
            mask_in[:B] = mask.to(self.dtype)
            mask_in[B:] = mask.to(self.dtype)
            mu_in[:B] = mu.to(self.dtype)
            # mu_in[B:] stays zero for unconditional
            t_in[:B] = t
            t_in[B:] = t
            spks_in[:B] = spks.to(self.dtype)
            # spks_in[B:] stays zero for unconditional
            cond_in[:B] = cond.to(self.dtype)
            # cond_in[B:] stays zero for unconditional

            # Get velocity estimate
            dxdt = self.estimator(
                x=x_in,
                mask=mask_in,
                mu=mu_in,
                t=t_in,
                spks=spks_in,
                cond=cond_in,
            )

            # CFG: combine conditional and unconditional
            dxdt_cond, dxdt_uncond = torch.split(dxdt, B, dim=0)
            dxdt = (1.0 + self.inference_cfg_rate) * dxdt_cond - self.inference_cfg_rate * dxdt_uncond

            # Euler step
            dt = r - t
            x = x + dt * dxdt.to(in_dtype)

        return x

    def _basic_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Basic Euler solver for meanflow (distilled) models."""
        in_dtype = x.dtype

        print("S3 Token -> Mel Inference (Meanflow)...")
        for t, r in tqdm(zip(t_span[:-1], t_span[1:]), total=len(t_span) - 1):
            t = t.unsqueeze(0)
            r = r.unsqueeze(0)

            dxdt = self.estimator(
                x=x.to(self.dtype),
                mask=mask.to(self.dtype),
                mu=mu.to(self.dtype),
                t=t.to(self.dtype),
                spks=spks.to(self.dtype),
                cond=cond.to(self.dtype),
            )

            dt = r - t
            x = x + dt * dxdt.to(in_dtype)

        return x

    def compute_loss(
        self,
        x1: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        spks: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CFM training loss.

        Args:
            x1: Target mel-spectrogram
            mask: Target mask
            mu: Encoder output
            spks: Speaker embedding
            cond: Conditioning mel

        Returns:
            Tuple of (loss, intermediate_y)
        """
        b, _, t = mu.shape

        # Random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t = 1 - torch.cos(t * 0.5 * torch.pi)

        # Sample noise
        z = torch.randn_like(x1)

        # Interpolate
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # CFG dropout during training
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        # Predict velocity
        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond)

        # MSE loss
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum")
        loss = loss / (torch.sum(mask) * u.shape[1])

        return loss, y
