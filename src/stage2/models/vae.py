from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.shortcut(x) + h


class Encoder(nn.Module):
    """(B, in_channels, H, W) -> (B, 2*latent_channels, H//2, W//2), i.e. (mu, logvar) moments."""

    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 256,
        latent_channels: int = 128,
        num_res_blocks: int = 2,
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        mid_channels = hidden_channels * 2

        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([
            ResnetBlock(hidden_channels, hidden_channels, num_groups=num_groups, dropout=dropout)
            for _ in range(num_res_blocks)
        ])
        self.downsample = nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, stride=2, padding=1)
        self.mid_blocks = nn.ModuleList([
            ResnetBlock(mid_channels, mid_channels, num_groups=num_groups, dropout=dropout)
            for _ in range(num_res_blocks)
        ])
        self.norm_out = nn.GroupNorm(num_groups, mid_channels)
        self.conv_out = nn.Conv2d(mid_channels, 2 * latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.downsample(h)
        for block in self.mid_blocks:
            h = block(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class Decoder(nn.Module):
    """(B, latent_channels, H, W) -> (B, out_channels, 2*H, 2*W)."""

    def __init__(
        self,
        out_channels: int = 768,
        hidden_channels: int = 256,
        latent_channels: int = 128,
        num_res_blocks: int = 2,
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        mid_channels = hidden_channels * 2

        self.conv_in = nn.Conv2d(latent_channels, mid_channels, kernel_size=3, padding=1)
        self.mid_blocks = nn.ModuleList([
            ResnetBlock(mid_channels, mid_channels, num_groups=num_groups, dropout=dropout)
            for _ in range(num_res_blocks)
        ])
        self.upsample_conv = nn.Conv2d(mid_channels, hidden_channels, kernel_size=3, padding=1)
        self.up_blocks = nn.ModuleList([
            ResnetBlock(hidden_channels, hidden_channels, num_groups=num_groups, dropout=dropout)
            for _ in range(num_res_blocks)
        ])
        self.norm_out = nn.GroupNorm(num_groups, hidden_channels)
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        for block in self.mid_blocks:
            h = block(h)
        h = F.interpolate(h, scale_factor=2.0, mode="nearest")
        h = self.upsample_conv(h)
        for block in self.up_blocks:
            h = block(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class DiagonalGaussianDistribution:
    """Per-position diagonal Gaussian over a (B, C, H, W) latent, parameterized as
    channel-concatenated (mean, logvar) moments of shape (B, 2*C, H, W)."""

    def __init__(self, parameters: torch.Tensor):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = self.std * self.std

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        """KL divergence to the standard normal prior, per sample, summed over (C, H, W)."""
        return 0.5 * torch.sum(self.mean * self.mean + self.var - 1.0 - self.logvar, dim=[1, 2, 3])


class PatchTokenVAE(nn.Module):
    """CNN VAE for DINOv2 patch tokens: (B, 768, 16, 16) <-> (B, 128, 8, 8)."""

    def __init__(
        self,
        in_channels: int = 768,
        latent_channels: int = 128,
        hidden_channels: int = 256,
        num_res_blocks: int = 2,
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_channels % num_groups != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by num_groups ({num_groups})"
            )
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            num_groups=num_groups,
            dropout=dropout,
        )
        # technically these 1x1 convolutions collapse with the 3x3 conv_out and conv_in, analogue to diffusers AutoEncoderKL
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        moments = self.quant_conv(self.encoder(x))
        return DiagonalGaussianDistribution(moments)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, sample_posterior: bool = True
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        x_rec = self.decode(z)
        return x_rec, posterior

    def loss(
        self, x: torch.Tensor, sample_posterior: bool = True, kl_weight: float = 1e-6
    ) -> Tuple[torch.Tensor, dict]:
        """MSE reconstruction + KL loss, batch-averaged. Convenience helper for simple training loops."""
        x_rec, posterior = self.forward(x, sample_posterior=sample_posterior)
        rec_loss = F.mse_loss(x_rec, x, reduction="mean")
        kl_loss = posterior.kl().mean()
        total = rec_loss + kl_weight * kl_loss
        return total, {"rec_loss": rec_loss.detach(), "kl_loss": kl_loss.detach(), "loss": total.detach()}
