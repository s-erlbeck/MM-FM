#!/usr/bin/env python3
"""
Compute normalization statistics (mean, variance) for encoder latents.

This script iterates through a dataset, extracts encoder latents, and computes
position-wise mean and variance statistics. These stats are used to normalize
latents before feeding them to the diffusion model (Stage 2).

Usage:
    # For ImageFolder (ImageNet):
    python src/scripts/compute_normalization_stats.py \
        --encoder-cls Dinov2withNorm \
        --encoder-config facebook/dinov2-with-registers-base \
        --data-path /path/to/imagenet/train \
        --output models/stats/dinov2/imagenet/stat.pt \
        --num-samples 50000 \
        --batch-size 64
"""

import argparse
import os
import sys

# Add src to path for imports
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from math import sqrt


def get_encoder(encoder_cls: str, encoder_config: str, encoder_params: dict = None):
    """Load encoder model."""
    from stage1.encoders import ARCHS

    if encoder_params is None:
        encoder_params = {}

    # Set default params based on encoder class
    if encoder_cls == 'Dinov2withNorm':
        encoder_params.setdefault('dinov2_path', encoder_config)
        encoder_params.setdefault('normalize', True)

    encoder_class = ARCHS[encoder_cls]
    encoder = encoder_class(**encoder_params)
    return encoder


def get_image_transform(image_size: int, encoder_config: str):
    """Get image transform matching encoder preprocessing."""
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(encoder_config)
    encoder_mean = proc.image_mean
    encoder_std = proc.image_std

    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=encoder_mean, std=encoder_std),
    ])
    return transform


def compute_stats_welford(encoder, dataloader, num_samples, device, reshape_to_2d=True, compute_mean=False):
    """
    Compute variance (and optionally mean) using Welford's online algorithm.
    This is numerically stable and memory efficient.

    Note: The existing stats files (e.g., dinov2/wReg_base/imagenet1k/stat.pt) use
    mean=None and only store variance. This matches the DINOv2/SiT training setup
    where latents are normalized by variance only (z / sqrt(var)).

    Args:
        encoder: Encoder model
        dataloader: DataLoader for images
        num_samples: Number of samples to process
        device: torch device
        reshape_to_2d: Whether to reshape latents to 2D (B, C, H, W)
        compute_mean: Whether to compute mean (default: False, matching existing stats)

    Returns:
        mean: Position-wise mean (None if compute_mean=False)
        var: Position-wise variance, shape (C, H, W)
    """
    encoder.eval()

    # Initialize accumulators (will be set on first batch)
    count = 0
    running_mean = None
    M2 = None  # Sum of squared differences from mean

    pbar = tqdm(total=num_samples, desc="Computing statistics")

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)

            # Encode to latents
            z, _ = encoder(images)  # (B, N, C), (B, C)

            if reshape_to_2d:
                b, n, c = z.shape
                h = w = int(sqrt(n))
                z = z.transpose(1, 2).view(b, c, h, w)  # (B, C, H, W)

            # Process each sample in batch
            for i in range(z.size(0)):
                if count >= num_samples:
                    break

                sample = z[i]  # (C, H, W) or (N, C)

                if running_mean is None:
                    running_mean = torch.zeros_like(sample)
                    M2 = torch.zeros_like(sample)

                count += 1
                delta = sample - running_mean
                running_mean = running_mean + delta / count
                delta2 = sample - running_mean
                M2 = M2 + delta * delta2

                pbar.update(1)

            if count >= num_samples:
                break

    pbar.close()

    # Compute variance
    if count < 2:
        var = torch.ones_like(running_mean)
    else:
        var = M2 / (count - 1)  # Bessel's correction

    print(f"Computed statistics over {count} samples")
    print(f"Var shape: {var.shape}")
    print(f"Var range: [{var.min().item():.6f}, {var.max().item():.6f}]")
    print(f"Var mean: {var.mean().item():.6f}")

    if compute_mean:
        print(f"Mean shape: {running_mean.shape}")
        print(f"Mean range: [{running_mean.min().item():.6f}, {running_mean.max().item():.6f}]")
        return running_mean.cpu(), var.cpu()
    else:
        # Return None for mean, matching existing stats format
        return None, var.cpu()


def main():
    parser = argparse.ArgumentParser(description="Compute normalization statistics for encoder latents")

    # Encoder settings
    parser.add_argument("--encoder-cls", type=str, required=True,
                        choices=['Dinov2withNorm', 'SigLIP2wNorm'],
                        help="Encoder class name")
    parser.add_argument("--encoder-config", type=str, required=True,
                        help="Encoder config path (HuggingFace model ID)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size for encoder")

    # Data settings
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to ImageFolder dataset")

    # Output settings
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for statistics file (*.pt)")

    # Compute settings
    parser.add_argument("--num-samples", type=int, default=50000,
                        help="Number of samples to use for statistics")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--compute-mean", action="store_true",
                        help="Also compute mean (default: False, variance-only like existing stats)")

    args = parser.parse_args()

    # Validate data source
    if args.data_path is None:
        raise ValueError("Must provide --data-path")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load encoder
    print(f"Loading encoder: {args.encoder_cls} ({args.encoder_config})")
    encoder = get_encoder(args.encoder_cls, args.encoder_config)
    encoder = encoder.to(device)
    encoder.eval()

    # Get transform
    transform = get_image_transform(args.image_size, args.encoder_config)

    # Create dataset and dataloader (ImageFolder)
    print(f"Loading ImageFolder from: {args.data_path}")
    dataset = ImageFolder(args.data_path, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Compute statistics
    print(f"Computing statistics over {args.num_samples} samples...")
    print(f"Compute mean: {args.compute_mean}")
    mean, var = compute_stats_welford(
        encoder, dataloader, args.num_samples, device,
        reshape_to_2d=True, compute_mean=args.compute_mean
    )

    # Save statistics (matching existing format: mean=None for variance-only)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    stats = {'mean': mean, 'var': var}
    torch.save(stats, args.output)
    print(f"Saved statistics to: {args.output}")

    # Print summary
    print("\nStatistics Summary:")
    print(f"  Mean: {'None (variance-only normalization)' if mean is None else f'shape {mean.shape}'}")
    print(f"  Var shape: {var.shape}")
    print(f"  Var (global mean): {var.mean().item():.6f}")
    print(f"  Std (global mean): {var.mean().sqrt().item():.6f}")


if __name__ == "__main__":
    main()
