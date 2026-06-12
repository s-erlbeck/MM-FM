#!/usr/bin/env python
"""
Compute FID using pre-computed reference statistics (mu, sigma).
Works with reference batches from compute_fid_reference.py.

Usage:
    python src/scripts/compute_fid_from_stats.py \
        --ref guided-diffusion/evaluations/VIRTUAL_imagenet256_labeled.npz \
        --samples results/.../samples.npz \
        --batch-size 64
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm


def compute_fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate FID between two Gaussian distributions."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        print(f'WARNING: FID calculation produces singular product; adding {eps} to diagonal')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def load_inception_model():
    """Load InceptionV3 detector (same as training uses)."""
    import pickle
    import dnnlib

    url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    with dnnlib.util.open_url(url, verbose=True) as f:
        model = pickle.load(f)
    return model


def compute_statistics_from_images(images, model, batch_size, device):
    """Compute mu and sigma from images using InceptionV3."""
    all_features = []
    num_batches = (len(images) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Computing sample features"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(images))
        batch = images[start_idx:end_idx]

        # Convert NHWC uint8 to NCHW tensor
        batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)

        with torch.no_grad():
            features = model(batch_tensor, return_features=True).to(torch.float64)
            all_features.append(features.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()
    mu = all_features.mean(axis=0)
    sigma = np.cov(all_features, rowvar=False)

    return mu, sigma


def main():
    parser = argparse.ArgumentParser(description="Compute FID using pre-computed reference statistics")
    parser.add_argument("--ref", type=str, required=True, help="Reference NPZ (with mu/sigma or arr_0)")
    parser.add_argument("--samples", type=str, required=True, help="Samples NPZ (with arr_0)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load reference statistics
    print(f"Loading reference from {args.ref}...")
    ref_data = np.load(args.ref)

    if 'mu' in ref_data and 'sigma' in ref_data:
        ref_mu = ref_data['mu']
        ref_sigma = ref_data['sigma']
        print(f"  Using pre-computed statistics (mu: {ref_mu.shape}, sigma: {ref_sigma.shape})")
    else:
        raise ValueError("Reference must contain pre-computed 'mu' and 'sigma'. "
                        "Use compute_fid_reference.py to generate.")

    # Load sample images
    print(f"Loading samples from {args.samples}...")
    sample_data = np.load(args.samples)

    if 'arr_0' not in sample_data:
        raise ValueError("Samples NPZ must contain 'arr_0' with images")

    samples = sample_data['arr_0']
    print(f"  Loaded {len(samples)} samples, shape: {samples.shape}")

    # Check image size matches
    sample_size = samples.shape[1]
    print(f"  Sample image size: {sample_size}x{sample_size}")

    # Load InceptionV3
    print("Loading InceptionV3 model...")
    model = load_inception_model().to(device).eval()

    # Compute sample statistics
    print("Computing sample statistics...")
    sample_mu, sample_sigma = compute_statistics_from_images(
        samples, model, args.batch_size, device
    )

    # Compute FID
    print("Computing FID...")
    fid = compute_fid_from_stats(ref_mu, ref_sigma, sample_mu, sample_sigma)

    print(f"\n{'='*50}")
    print(f"FID: {fid:.2f}")
    print(f"{'='*50}")

    return fid


if __name__ == "__main__":
    main()
