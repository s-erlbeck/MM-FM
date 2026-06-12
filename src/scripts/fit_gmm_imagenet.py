#!/usr/bin/env python3
"""
Fit Gaussian Mixture Model (GMM) on CLS tokens from ImageNet (ImageFolder format).

This script extracts CLS tokens from a vision encoder (e.g., SigLIP2, DINOv2) on
ImageNet and fits a GMM for mode-conditional generation.

Key features:
- Supports any encoder via command-line arguments (SigLIP2, DINOv2, etc.)
- Works with ImageFolder format (ImageNet structure)
- Multi-GPU distributed extraction for faster processing
- Sequential GMM fitting with full CPU utilization
- Saves GMM model for use in spatial_gmm_imagenet.py

Usage:
    # Single GPU with SigLIP2 (default)
    python src/scripts/fit_gmm_imagenet.py \
        --data-path /path/to/imagenet/train \
        --config configs/stage2/training/ImageNet256/DiTDH-XL_SigLIP2-B-UNCONDITIONAL.yaml \
        --output-dir results/clustering/siglip2-base-imagenet-gmm-8192-diag \
        --n-components 8192 \
        --covariance-type diag

    # Multi-GPU (recommended for large datasets)
    torchrun --nproc_per_node=8 src/scripts/fit_gmm_imagenet.py \
        --data-path /path/to/imagenet/train \
        --config configs/stage2/training/ImageNet256/DiTDH-XL_SigLIP2-B-UNCONDITIONAL.yaml \
        --output-dir results/clustering/siglip2-base-imagenet-gmm-8192-diag \
        --n-components 8192 \
        --covariance-type diag

    # With DINOv2 encoder
    torchrun --nproc_per_node=8 src/scripts/fit_gmm_imagenet.py \
        --data-path /path/to/imagenet/train \
        --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B-UNCONDITIONAL.yaml \
        --output-dir results/clustering/dinov2-base-imagenet-gmm-8192-diag \
        --n-components 8192 \
        --covariance-type diag
"""

# Fix OpenBLAS threading - must be set before importing numpy/sklearn
import os
os.environ['OPENBLAS_NUM_THREADS'] = '64'
os.environ['MKL_NUM_THREADS'] = '64'
os.environ['OMP_NUM_THREADS'] = '64'

import sys
import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# Add src to path
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from stage1.encoders import ARCHS
from transformers import AutoImageProcessor


def center_crop_arr(pil_image, image_size):
    """Center cropping implementation from ADM."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class ClassBalancedSubset(torch.utils.data.Dataset):
    """
    Wrapper that samples a percentage of data from each class while maintaining class balance.
    Only supports ImageFolder (requires .targets attribute).

    Args:
        base_dataset: The underlying dataset (must have .targets attribute)
        percentage: Fraction of data to use per class (0.0-1.0)
        seed: Random seed for reproducible sampling
    """
    def __init__(self, base_dataset, percentage=1.0, seed=42):
        self.base_dataset = base_dataset
        self.percentage = percentage

        if not hasattr(base_dataset, 'targets'):
            raise ValueError("ClassBalancedSubset only supports ImageFolder datasets")

        all_labels = base_dataset.targets

        # Build per-class index lists
        class_to_indices = {}
        for idx, label in enumerate(all_labels):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

        # Sample indices per class
        rng = np.random.RandomState(seed)
        self.selected_indices = []

        for class_id in sorted(class_to_indices.keys()):
            indices = class_to_indices[class_id]
            n_samples = max(1, int(len(indices) * percentage))
            sampled = rng.choice(indices, size=n_samples, replace=False)
            self.selected_indices.extend(sampled)

        self.selected_indices = sorted(self.selected_indices)

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        original_idx = self.selected_indices[idx]
        return self.base_dataset[original_idx]


def get_transform(image_size: int, encoder_mean: list, encoder_std: list):
    """Create deterministic transform for CLS token extraction (no random augmentation)."""
    normalize = transforms.Normalize(mean=encoder_mean, std=encoder_std)
    return transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def extract_cls_tokens_distributed(
    encoder,
    dataloader,
    device,
    rank: int = 0,
    world_size: int = 1,
    precision: str = "bf16",
):
    """
    Extract CLS tokens from encoder using ImageFolder dataset.

    Returns:
        cls_tokens: numpy array of shape (N, hidden_size)
    """
    encoder.eval()
    all_cls_tokens = []

    pbar = tqdm(dataloader, desc=f"GPU {rank}: Extracting CLS tokens") if rank == 0 else dataloader

    with torch.no_grad():
        for batch in pbar:
            images = batch[0].to(device, non_blocking=True)

            if precision == "bf16":
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, cls_tokens = encoder(images)  # (B, N, C), (B, C)
            else:
                _, cls_tokens = encoder(images)

            # Convert to float32 for numpy (numpy doesn't support bfloat16)
            all_cls_tokens.append(cls_tokens.float().cpu().numpy())

    cls_tokens = np.concatenate(all_cls_tokens, axis=0)
    print(f"Rank {rank}: Extracted {len(cls_tokens)} CLS tokens")

    return cls_tokens


def gather_cls_tokens(local_tokens: np.ndarray, rank: int, world_size: int):
    """
    Gather CLS tokens from all ranks to rank 0.
    """
    if world_size == 1:
        return local_tokens

    # Convert to tensor for distributed gathering
    local_tensor = torch.from_numpy(local_tokens).cuda()

    # Gather sizes first (tokens may vary per rank)
    local_size = torch.tensor([local_tensor.shape[0]], device='cuda')
    all_sizes = [torch.zeros(1, device='cuda', dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [int(s.item()) for s in all_sizes]

    max_size = max(all_sizes)
    hidden_size = local_tensor.shape[1]

    # Pad local tensor to max size
    if local_tensor.shape[0] < max_size:
        padding = torch.zeros(max_size - local_tensor.shape[0], hidden_size, device='cuda')
        local_tensor = torch.cat([local_tensor, padding], dim=0)

    # Gather all tensors
    all_tensors = [torch.zeros(max_size, hidden_size, device='cuda') for _ in range(world_size)]
    dist.all_gather(all_tensors, local_tensor)

    # Only rank 0 needs the full result
    if rank == 0:
        # Trim padding and concatenate
        all_tokens = []
        for i, tensor in enumerate(all_tensors):
            all_tokens.append(tensor[:all_sizes[i]].cpu().numpy())
        return np.concatenate(all_tokens, axis=0)
    else:
        return None


def fit_gmm(
    cls_tokens: np.ndarray,
    n_components: int,
    covariance_type: str = 'diag',
    init_params: str = 'k-means++',
    max_iter: int = 100,
    n_init: int = 1,
    reg_covar: float = 1e-6,
    random_state: int = 42,
    verbose: bool = True,
):
    """
    Fit a Gaussian Mixture Model on CLS tokens.
    """
    print(f"\nFitting GMM:")
    print(f"  n_components: {n_components}")
    print(f"  covariance_type: {covariance_type}")
    print(f"  init_params: {init_params}")
    print(f"  n_samples: {len(cls_tokens)}")
    print(f"  hidden_size: {cls_tokens.shape[1]}")

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        init_params=init_params,
        max_iter=max_iter,
        n_init=n_init,
        reg_covar=reg_covar,
        random_state=random_state,
        verbose=2 if verbose else 0,
        verbose_interval=10,
    )

    print("\nFitting GMM (this may take a while)...")
    labels = gmm.fit_predict(cls_tokens)

    # Compute metrics
    log_likelihood = gmm.score(cls_tokens) * len(cls_tokens)
    bic = gmm.bic(cls_tokens)
    aic = gmm.aic(cls_tokens)

    print(f"\nGMM fitting complete:")
    print(f"  Converged: {gmm.converged_}")
    print(f"  Iterations: {gmm.n_iter_}")
    print(f"  Log-likelihood: {log_likelihood:.2f}")
    print(f"  BIC: {bic:.2f}")
    print(f"  AIC: {aic:.2f}")

    # Compute clustering metrics (sample for speed)
    sample_size = min(10000, len(cls_tokens))
    sample_idx = np.random.choice(len(cls_tokens), sample_size, replace=False)
    sample_tokens = cls_tokens[sample_idx]
    sample_labels = labels[sample_idx]

    silhouette = silhouette_score(sample_tokens, sample_labels)
    calinski = calinski_harabasz_score(sample_tokens, sample_labels)
    davies = davies_bouldin_score(sample_tokens, sample_labels)

    print(f"\nClustering metrics (sampled {sample_size}):")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Calinski-Harabasz: {calinski:.2f}")
    print(f"  Davies-Bouldin: {davies:.4f}")

    metrics = {
        'n_components': n_components,
        'covariance_type': covariance_type,
        'init_params': init_params,
        'n_samples': len(cls_tokens),
        'hidden_size': cls_tokens.shape[1],
        'converged': gmm.converged_,
        'n_iter': gmm.n_iter_,
        'log_likelihood': log_likelihood,
        'bic': bic,
        'aic': aic,
        'silhouette_score': silhouette,
        'calinski_harabasz_score': calinski,
        'davies_bouldin_score': davies,
    }

    return gmm, labels, metrics


def plot_gmm_weights(weights: np.ndarray, output_path: str, n_components: int):
    """Plot GMM component weights distribution."""
    sorted_weights = np.sort(weights)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    axes[0].bar(range(len(weights)), sorted_weights, alpha=0.7)
    axes[0].axhline(y=1.0/n_components, color='r', linestyle='--',
                    label=f'Uniform (1/{n_components})')
    axes[0].set_xlabel('Component (sorted)')
    axes[0].set_ylabel('Weight')
    axes[0].set_title('GMM Component Weights')
    axes[0].legend()

    # Histogram
    axes[1].hist(weights, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=weights.mean(), color='b', linestyle='--', label=f'Mean: {weights.mean():.4f}')
    axes[1].set_xlabel('Weight')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Weight Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved weight plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit GMM on CLS tokens from ImageNet (ImageFolder format)"
    )

    # Data settings
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to ImageNet dataset root (ImageFolder structure)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config (for encoder settings)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save GMM model and results")

    # GMM settings
    parser.add_argument("--n-components", type=int, default=8192,
                        help="Number of GMM components (default: 8192)")
    parser.add_argument("--covariance-type", type=str, default='diag',
                        choices=['full', 'tied', 'diag', 'spherical'],
                        help="GMM covariance type (default: diag)")
    parser.add_argument("--init-params", type=str, default='k-means++',
                        choices=['kmeans', 'k-means++', 'random'],
                        help="GMM initialization method (default: k-means++)")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="Maximum EM iterations (default: 100)")
    parser.add_argument("--reg-covar", type=float, default=1e-6,
                        help="Covariance regularization (default: 1e-6)")

    # Processing settings
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size per GPU (default: 64)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--precision", type=str, default='bf16',
                        choices=['fp32', 'bf16'],
                        help="Precision for extraction (default: bf16)")
    parser.add_argument("--cls-tokens-path", type=str, default=None,
                        help="Path to pre-extracted CLS tokens (skip extraction)")

    # Data limiting
    parser.add_argument("--data-limit-percentage", type=float, default=1.0,
                        help="Percentage of data to use per class (0.0-1.0, default: 1.0)")
    parser.add_argument("--data-limit-seed", type=int, default=42,
                        help="Random seed for data limiting (default: 42)")

    args = parser.parse_args()

    # Initialize distributed
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    except:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print("="*80)
        print("GMM Fitting on ImageNet CLS Tokens")
        print("="*80)
        print(f"Data path: {args.data_path}")
        print(f"Config: {args.config}")
        print(f"Output: {args.output_dir}")
        print(f"GPUs: {world_size}")
        print(f"Components: {args.n_components}")
        print(f"Covariance: {args.covariance_type}")
        print("="*80)

    # Load config to get encoder settings
    from utils.train_utils import parse_configs
    rae_config, *_ = parse_configs(args.config)

    encoder_cls = rae_config.params.encoder_cls
    encoder_config_path = rae_config.params.encoder_config_path
    encoder_params = dict(rae_config.params.get('encoder_params', {}))
    image_size = rae_config.params.get('encoder_input_size', 224)

    if rank == 0:
        print(f"\nEncoder: {encoder_cls}")
        print(f"Config: {encoder_config_path}")
        print(f"Image size: {image_size}")

    # Check for pre-extracted CLS tokens
    cls_tokens = None
    if args.data_limit_percentage < 1.0:
        cls_tokens_path = args.cls_tokens_path or os.path.join(
            args.output_dir, f'cls_tokens_p{args.data_limit_percentage}_s{args.data_limit_seed}.npz'
        )
    else:
        cls_tokens_path = args.cls_tokens_path or os.path.join(args.output_dir, 'cls_tokens.npz')

    if os.path.exists(cls_tokens_path):
        if rank == 0:
            print(f"\nLoading pre-extracted CLS tokens from {cls_tokens_path}")
            data = np.load(cls_tokens_path)
            cls_tokens = data['cls_tokens']
            print(f"Loaded {len(cls_tokens)} CLS tokens")
    else:
        # Need to extract CLS tokens
        # Load encoder
        encoder_class = ARCHS[encoder_cls]
        encoder = encoder_class(**encoder_params).to(device)
        encoder.eval()

        # Get encoder preprocessing
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)

        if rank == 0:
            print(f"\nEncoder loaded: {encoder_cls}")
            print(f"Hidden size: {encoder.hidden_size}")

        # Create dataset with transform
        transform = get_transform(image_size, proc.image_mean, proc.image_std)
        dataset = ImageFolder(args.data_path, transform=transform)

        # Apply data limiting
        if args.data_limit_percentage < 1.0:
            if rank == 0:
                original_size = len(dataset)
                print(f"\nApplying data limit: {args.data_limit_percentage*100:.1f}% per class (seed={args.data_limit_seed})")
            dataset = ClassBalancedSubset(dataset, args.data_limit_percentage, args.data_limit_seed)
            if rank == 0:
                print(f"Dataset reduced from {original_size:,} to {len(dataset):,} samples")

        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # No shuffling for deterministic ordering
            drop_last=False
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        if rank == 0:
            print(f"\nDataset size: {len(dataset):,}")
            print(f"Samples per GPU: ~{len(dataset) // world_size:,}")

        # Extract CLS tokens
        local_cls_tokens = extract_cls_tokens_distributed(
            encoder, dataloader, device, rank, world_size, args.precision
        )

        # Gather to rank 0
        if world_size > 1:
            dist.barrier()

        cls_tokens = gather_cls_tokens(local_cls_tokens, rank, world_size)

        # Save CLS tokens (rank 0 only)
        if rank == 0 and cls_tokens is not None:
            np.savez_compressed(cls_tokens_path, cls_tokens=cls_tokens)
            print(f"Saved CLS tokens to {cls_tokens_path}")

    # Only rank 0 fits GMM
    if rank != 0:
        if world_size > 1:
            dist.destroy_process_group()
        return

    if cls_tokens is None:
        print("ERROR: No CLS tokens available")
        return

    print(f"\nCLS tokens shape: {cls_tokens.shape}")
    print(f"CLS tokens stats:")
    print(f"  Mean: {cls_tokens.mean():.6f}")
    print(f"  Std: {cls_tokens.std():.6f}")
    print(f"  Min: {cls_tokens.min():.6f}")
    print(f"  Max: {cls_tokens.max():.6f}")

    # Fit GMM
    gmm, labels, metrics = fit_gmm(
        cls_tokens,
        n_components=args.n_components,
        covariance_type=args.covariance_type,
        init_params=args.init_params,
        max_iter=args.max_iter,
        reg_covar=args.reg_covar,
    )

    # Save GMM model
    model_name = f"gmm_n{args.n_components}_{args.covariance_type}_{args.init_params}.pkl"
    model_path = os.path.join(args.output_dir, model_name)

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': gmm,
            'labels': labels,
            'config': {
                'n_components': args.n_components,
                'covariance_type': args.covariance_type,
                'init_params': args.init_params,
            },
            'metrics': metrics,
            'metadata': {
                'data_path': args.data_path,
                'encoder_cls': encoder_cls,
                'encoder_config_path': encoder_config_path,
                'image_size': image_size,
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(cls_tokens),
                'hidden_size': cls_tokens.shape[1],
                'data_limit_percentage': args.data_limit_percentage,
                'data_limit_seed': args.data_limit_seed,
            },
        }, f)

    print(f"\nSaved GMM model to {model_path}")

    # Save labels
    labels_path = os.path.join(args.output_dir, f"gmm_n{args.n_components}_{args.covariance_type}_{args.init_params}_labels.npy")
    np.save(labels_path, labels)
    print(f"Saved labels to {labels_path}")

    # Plot weights
    weights_path = os.path.join(args.output_dir, f"gmm_n{args.n_components}_{args.covariance_type}_{args.init_params}_weights.png")
    plot_gmm_weights(gmm.weights_, weights_path, args.n_components)

    # Print cluster distribution
    print("\nCluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Active clusters: {len(unique)} / {args.n_components}")
    print(f"  Largest cluster: {counts.max()} samples ({100*counts.max()/len(labels):.2f}%)")
    print(f"  Smallest cluster: {counts.min()} samples ({100*counts.min()/len(labels):.2f}%)")
    print(f"  Median cluster: {np.median(counts):.0f} samples")

    print("\n" + "="*80)
    print("GMM fitting complete!")
    print("="*80)
    print(f"\nTo create spatial GMM, run:")
    print(f"  torchrun --nproc_per_node=8 src/scripts/spatial_gmm_imagenet.py \\")
    print(f"    --cls-gmm-path {model_path} \\")
    print(f"    --data-path {args.data_path} \\")
    print(f"    --config {args.config} \\")
    print(f"    --output-dir {args.output_dir}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
