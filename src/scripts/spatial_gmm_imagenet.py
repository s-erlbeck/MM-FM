#!/usr/bin/env python3
"""
Create Spatial GMM from CLS GMM assignments on ImageNet (ImageFolder format).

This script transfers cluster assignments from CLS tokens to spatial tokens,
creating a GMM suitable for sampling spatial latent noise during Stage 2 training.

Key features:
- Uses same ImageNet samples as CLS GMM fitting (deterministic processing)
- Multi-GPU distributed extraction for faster processing
- Horizontal flip augmentation (doubles data for better Gaussian estimates)
- Memory-efficient streaming Gaussian estimation
- Spherical covariance (memory efficient: 1 scalar per component)

Algorithm:
1. Pass 1 (Distributed): Extract CLS + spatial tokens, assign clusters, save to temp files
2. Pass 2 (Rank 0 only): Stream through temp files, compute mean/variance per cluster
3. Construct spatial GMM with spherical covariance

Usage:
    # Multi-GPU (recommended)
    torchrun --nproc_per_node=8 src/scripts/spatial_gmm_imagenet.py \
        --cls-gmm-path results/clustering/siglip2-base-imagenet-gmm-8192-diag/gmm_n8192_diag_k-means++.pkl \
        --data-path /path/to/imagenet/train \
        --config configs/stage2/training/ImageNet256/DiTDH-XL_SigLIP2-B-UNCONDITIONAL.yaml \
        --output-dir results/clustering/siglip2-base-imagenet-gmm-8192-diag

    # Resume from Pass 2 (if Pass 1 completed but Pass 2 failed)
    python src/scripts/spatial_gmm_imagenet.py \
        --cls-gmm-path results/clustering/siglip2-base-imagenet-gmm-8192-diag/gmm_n8192_diag_k-means++.pkl \
        --data-path /path/to/imagenet/train \
        --config configs/stage2/training/ImageNet256/DiTDH-XL_SigLIP2-B-UNCONDITIONAL.yaml \
        --output-dir results/clustering/siglip2-base-imagenet-gmm-8192-diag \
        --skip-pass1
"""

# Set BLAS threading for fast numpy operations
import os
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'

import sys
import argparse
import pickle
import glob
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import gc
from PIL import Image

# Add src to path
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from stage1 import RAE
from utils.train_utils import parse_configs
from utils.model_utils import instantiate_from_config


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
    """
    def __init__(self, base_dataset, percentage=1.0, seed=42):
        self.base_dataset = base_dataset
        self.percentage = percentage

        if not hasattr(base_dataset, 'targets'):
            raise ValueError("ClassBalancedSubset only supports ImageFolder datasets")

        all_labels = base_dataset.targets

        class_to_indices = {}
        for idx, label in enumerate(all_labels):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

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


def get_transform(image_size: int):
    """Create transform matching training preprocessing (no flip - we do it manually)."""
    return transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
    ])


def apply_horizontal_flip(spatial_latents: torch.Tensor) -> torch.Tensor:
    """Apply horizontal flip to spatial latents (B, C, H, W)."""
    return torch.flip(spatial_latents, dims=[3])


def pass1_extract_and_assign(
    args,
    rae: RAE,
    cls_gmm,
    device,
    temp_dir: str,
    n_components: int,
    rank: int,
    world_size: int,
):
    """
    Pass 1: Extract spatial tokens, assign clusters via CLS GMM, save to temp files.

    Each GPU processes its shard of data and saves to rank-specific cluster files.
    """
    if rank == 0:
        print("\n" + "="*80)
        print("Pass 1: Extracting spatial tokens and assigning clusters")
        print("="*80)

    # Create dataset with transform
    transform = get_transform(args.image_size)
    dataset = ImageFolder(args.data_path, transform=transform)

    # Apply data limiting
    if args.data_limit_percentage < 1.0:
        if rank == 0:
            original_size = len(dataset)
            print(f"Applying data limit: {args.data_limit_percentage*100:.1f}% per class (seed={args.data_limit_seed})")
        dataset = ClassBalancedSubset(dataset, args.data_limit_percentage, args.data_limit_seed)
        if rank == 0:
            print(f"Dataset reduced from {original_size:,} to {len(dataset):,} samples")

    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,  # CRITICAL: No shuffling for deterministic ordering
        drop_last=False
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    if rank == 0:
        print(f"Total samples: {len(dataset):,}")
        print(f"Samples per GPU: ~{len(dataset) // world_size:,}")
        print(f"With flip augmentation: ~{2 * len(dataset):,} total")

    # Create rank-specific temp directory
    rank_temp_dir = os.path.join(temp_dir, f"rank_{rank}")
    os.makedirs(rank_temp_dir, exist_ok=True)

    # Track cluster counts and file handles
    cluster_counts = np.zeros(n_components, dtype=np.int64)
    cluster_buffers = {}
    flush_counter = 0

    rae.eval()
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"GPU {rank}: Extracting") if rank == 0 else loader

        for batch_idx, batch in enumerate(pbar):
            images = batch[0].to(device, non_blocking=True)

            # Extract spatial and CLS tokens
            if args.precision == "bf16":
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    spatial, cls_tokens = rae.encode(images, return_cls_token=True)
            else:
                spatial, cls_tokens = rae.encode(images, return_cls_token=True)

            # Create flipped version
            spatial_flip = apply_horizontal_flip(spatial)

            # Move to CPU and flatten (convert to float32 for numpy - numpy doesn't support bfloat16)
            spatial_flat = spatial.float().cpu().flatten(start_dim=1).numpy()  # (B, C*H*W)
            spatial_flip_flat = spatial_flip.float().cpu().flatten(start_dim=1).numpy()
            cls_flat = cls_tokens.float().cpu().numpy()

            # Assign clusters using CLS GMM
            cluster_ids = cls_gmm.predict(cls_flat)

            # Save both original and flipped to cluster-specific buffers
            for i in range(len(cluster_ids)):
                cluster_id = cluster_ids[i]

                if cluster_id not in cluster_buffers:
                    cluster_buffers[cluster_id] = []

                cluster_buffers[cluster_id].append(spatial_flat[i])
                cluster_buffers[cluster_id].append(spatial_flip_flat[i])
                cluster_counts[cluster_id] += 2

            # Flush to disk periodically
            if (batch_idx + 1) % args.flush_interval == 0:
                for cluster_id, tokens in cluster_buffers.items():
                    if len(tokens) > 0:
                        cluster_file = os.path.join(
                            rank_temp_dir,
                            f"cluster_{cluster_id}_part_{flush_counter}.npy"
                        )
                        np.save(cluster_file, np.array(tokens))

                cluster_buffers = {}
                flush_counter += 1

                if rank == 0:
                    print(f"  Flushed part {flush_counter} at batch {batch_idx + 1}")

    # Final flush
    for cluster_id, tokens in cluster_buffers.items():
        if len(tokens) > 0:
            cluster_file = os.path.join(
                rank_temp_dir,
                f"cluster_{cluster_id}_part_{flush_counter}.npy"
            )
            np.save(cluster_file, np.array(tokens))

    print(f"\nRank {rank}: Pass 1 complete!")
    print(f"  Total samples (with flip): {cluster_counts.sum():,}")

    return cluster_counts


def collect_part_files(temp_dir: str, n_components: int):
    """
    Collect all part file paths from all ranks (no merging).
    """
    print("\n" + "="*80)
    print("Collecting part files from all GPUs")
    print("="*80)

    # Auto-detect rank directories
    rank_dirs = sorted(glob.glob(os.path.join(temp_dir, "rank_*")))
    print(f"Found {len(rank_dirs)} rank directories")

    cluster_part_files = {}
    global_counts = np.zeros(n_components, dtype=np.int64)

    for cluster_id in tqdm(range(n_components), desc="Collecting"):
        all_part_files = []
        total_samples = 0

        for rank_dir in rank_dirs:
            part_files = sorted(glob.glob(
                os.path.join(rank_dir, f"cluster_{cluster_id}_part_*.npy")
            ))

            for part_file in part_files:
                if os.path.exists(part_file):
                    mmap_data = np.load(part_file, mmap_mode='r')
                    total_samples += mmap_data.shape[0]
                    all_part_files.append(part_file)

        if all_part_files:
            cluster_part_files[cluster_id] = all_part_files
            global_counts[cluster_id] = total_samples

    print(f"\nTotal samples: {global_counts.sum():,}")
    print(f"Active clusters: {(global_counts > 0).sum()} / {n_components}")

    return cluster_part_files, global_counts


def pass2_estimate_gaussians(
    cluster_part_files: dict,
    n_components: int,
    chunk_size: int = 100000,
):
    """
    Pass 2: Stream through part files, compute Gaussian parameters per cluster.

    Uses streaming computation for memory efficiency.
    """
    print("\n" + "="*80)
    print("Pass 2: Estimating Gaussian parameters (streaming)")
    print("="*80)

    # Determine spatial dimension
    D_spatial = None
    for cluster_id in range(n_components):
        if cluster_id in cluster_part_files:
            part_file = cluster_part_files[cluster_id][0]
            sample = np.load(part_file, mmap_mode='r')
            D_spatial = sample.shape[1]
            break

    if D_spatial is None:
        raise ValueError("No cluster files found!")

    print(f"Spatial dimension: {D_spatial}")
    print(f"Chunk size: {chunk_size:,}")

    means = np.zeros((n_components, D_spatial))
    variances = np.zeros(n_components)  # Spherical: one scalar per component
    cluster_counts = np.zeros(n_components, dtype=np.int64)

    for cluster_id in tqdm(range(n_components), desc="Processing clusters"):
        if cluster_id not in cluster_part_files:
            continue

        part_files = cluster_part_files[cluster_id]

        # Count total samples
        n_samples = 0
        for part_file in part_files:
            mmap = np.load(part_file, mmap_mode='r')
            n_samples += mmap.shape[0]

        cluster_counts[cluster_id] = n_samples

        if n_samples == 0:
            continue

        # Pass 1: Compute mean (streaming)
        sum_accumulator = np.zeros(D_spatial, dtype=np.float64)

        for part_file in part_files:
            mmap = np.load(part_file, mmap_mode='r')
            n_part = mmap.shape[0]

            for start in range(0, n_part, chunk_size):
                end = min(start + chunk_size, n_part)
                chunk = mmap[start:end]
                sum_accumulator += chunk.sum(axis=0)
                del chunk
            del mmap

        gc.collect()
        cluster_mean = sum_accumulator / n_samples
        means[cluster_id] = cluster_mean

        # Pass 2: Compute variance (streaming)
        variance_accumulator = np.zeros(D_spatial, dtype=np.float64)

        for part_file in part_files:
            mmap = np.load(part_file, mmap_mode='r')
            n_part = mmap.shape[0]

            for start in range(0, n_part, chunk_size):
                end = min(start + chunk_size, n_part)
                chunk = mmap[start:end]
                centered = chunk - cluster_mean
                variance_accumulator += (centered ** 2).sum(axis=0)
                del chunk, centered
            del mmap

        gc.collect()
        per_dim_variance = variance_accumulator / n_samples
        avg_variance = float(per_dim_variance.mean())
        variances[cluster_id] = avg_variance

        if cluster_id < 5 or cluster_id % 100 == 0:
            print(f"  Cluster {cluster_id}: {n_samples:,} samples, "
                  f"mean_norm={np.linalg.norm(cluster_mean):.3f}, "
                  f"avg_var={avg_variance:.6f}")

    # Handle empty clusters
    empty_clusters = np.where(cluster_counts == 0)[0]
    if len(empty_clusters) > 0:
        print(f"\nWARNING: {len(empty_clusters)} empty clusters")
        global_mean = means[cluster_counts > 0].mean(axis=0)
        global_var = variances[cluster_counts > 0].mean()
        for cluster_id in empty_clusters:
            means[cluster_id] = global_mean
            variances[cluster_id] = global_var

    return means, variances, cluster_counts


def construct_spatial_gmm(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
    n_components: int,
):
    """
    Construct GMM with spherical covariance from estimated parameters.
    """
    D_spatial = means.shape[1]

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='spherical',
        random_state=42,
    )

    gmm.means_ = means
    gmm.covariances_ = variances  # Scalars for spherical
    gmm.weights_ = weights
    gmm.precisions_cholesky_ = 1.0 / np.sqrt(variances + 1e-8)
    gmm.converged_ = True
    gmm.n_iter_ = 0
    gmm.lower_bound_ = -np.inf

    print(f"\nConstructed spatial GMM:")
    print(f"  Components: {n_components}")
    print(f"  Spatial dim: {D_spatial}")
    print(f"  Covariance: spherical")
    print(f"  Variance range: [{variances.min():.6f}, {variances.max():.6f}]")
    print(f"  Memory saved: ~{(n_components * D_spatial * D_spatial * 8) / 1024**4:.2f} TiB (spherical vs full)")

    return gmm


def main():
    parser = argparse.ArgumentParser(
        description="Create spatial GMM from CLS GMM assignments on ImageNet"
    )

    # Required arguments
    parser.add_argument("--cls-gmm-path", type=str, required=True,
                        help="Path to CLS GMM pickle file (from fit_gmm_imagenet.py)")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to ImageNet dataset root (ImageFolder structure)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config (for RAE settings)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save spatial GMM")

    # Processing settings
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image size (default: 256)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size per GPU (default: 32, reduced for memory)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--precision", type=str, default='bf16',
                        choices=['fp32', 'bf16'],
                        help="Precision (default: bf16)")
    parser.add_argument("--flush-interval", type=int, default=50,
                        help="Flush to disk every N batches (default: 50)")
    parser.add_argument("--chunk-size", type=int, default=100000,
                        help="Chunk size for streaming Gaussian estimation (default: 100k)")

    # Data limiting
    parser.add_argument("--data-limit-percentage", type=float, default=1.0,
                        help="Percentage of data to use per class (0.0-1.0, default: 1.0)")
    parser.add_argument("--data-limit-seed", type=int, default=42,
                        help="Random seed for data limiting (default: 42)")

    # Resume options
    parser.add_argument("--skip-pass1", action="store_true",
                        help="Skip Pass 1 and resume from Pass 2")
    parser.add_argument("--keep-temp-files", action="store_true",
                        help="Keep temporary cluster files after completion")

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
        print("Spatial GMM from CLS GMM Assignments (ImageNet)")
        print("="*80)
        print(f"CLS GMM: {args.cls_gmm_path}")
        print(f"Data path: {args.data_path}")
        print(f"Config: {args.config}")
        print(f"Output: {args.output_dir}")
        print(f"GPUs: {world_size}")
        print("="*80)

    temp_dir = os.path.join(args.output_dir, 'cluster_temp')
    if rank == 0:
        os.makedirs(temp_dir, exist_ok=True)

    # Load CLS GMM
    with open(args.cls_gmm_path, 'rb') as f:
        cls_gmm_data = pickle.load(f)

    cls_gmm = cls_gmm_data['model']
    cls_config = cls_gmm_data['config']
    n_components = cls_config['n_components']

    if rank == 0:
        print(f"\nLoaded CLS GMM:")
        print(f"  n_components: {n_components}")
        print(f"  covariance_type: {cls_config['covariance_type']}")

    # Get image size from config
    rae_config, *_ = parse_configs(args.config)
    args.image_size = rae_config.params.get('encoder_input_size', 224)

    if rank == 0:
        print(f"  Image size from config: {args.image_size}")

    # Load RAE (only if not skipping Pass 1)
    rae = None
    if not args.skip_pass1:
        rae = instantiate_from_config(rae_config).to(device)
        rae.eval()

        if rank == 0:
            print(f"\nLoaded RAE:")
            print(f"  Latent dim: {rae.latent_dim}")
            print(f"  Normalization: {rae.do_normalization}")

    # Pass 1: Extract and assign (all ranks)
    if not args.skip_pass1:
        local_counts = pass1_extract_and_assign(
            args, rae, cls_gmm, device, temp_dir, n_components, rank, world_size
        )

        if world_size > 1:
            dist.barrier()
    else:
        if rank == 0:
            print("\nSkipping Pass 1 (using existing cluster files)")
        if world_size > 1:
            dist.barrier()

    # Release non-zero ranks
    if world_size > 1:
        dist.destroy_process_group()
        if rank != 0:
            return

    # Only rank 0 continues
    if rank != 0:
        return

    # Collect part files
    cluster_part_files, global_counts = collect_part_files(temp_dir, n_components)

    # Pass 2: Estimate Gaussians
    means, variances, cluster_counts = pass2_estimate_gaussians(
        cluster_part_files, n_components, args.chunk_size
    )

    # Construct spatial GMM
    weights = cls_gmm.weights_
    spatial_gmm = construct_spatial_gmm(means, variances, weights, n_components)

    # Save
    model_name = f"spatial_gmm_from_cls_n{n_components}_{cls_config['covariance_type']}_{cls_config['init_params']}_flip.pkl"
    output_path = os.path.join(args.output_dir, model_name)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': spatial_gmm,
            'means': means,
            'variances': variances,
            'weights': weights,
            'cluster_counts': cluster_counts,
            'config': {
                'n_components': n_components,
                'covariance_type': 'spherical',
                'source_cls_gmm': args.cls_gmm_path,
                'source_cls_config': cls_config,
                'flip_augmentation': True,
            },
            'metadata': {
                'data_path': args.data_path,
                'timestamp': datetime.now().isoformat(),
                'num_samples': int(cluster_counts.sum()),
                'spatial_dim': means.shape[1],
            },
        }, f)

    print(f"\nSaved spatial GMM to {output_path}")

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total samples (with flip): {cluster_counts.sum():,}")
    print(f"Spatial dim: {means.shape[1]}")
    print(f"Components: {n_components}")
    print(f"Active clusters: {(cluster_counts > 0).sum()}")
    print(f"Mean variance: {variances.mean():.6f}")
    print(f"Variance range: [{variances.min():.6f}, {variances.max():.6f}]")

    # Cleanup temp files
    if not args.keep_temp_files:
        try:
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temp directory: {temp_dir}")
        except Exception as e:
            print(f"\nFailed to clean up: {e}")
    else:
        print(f"\nKept temp files in: {temp_dir}")

    print("\n" + "="*80)
    print("Complete!")
    print("="*80)
    print(f"\nTo use in training config:")
    print(f"gmm:")
    print(f"  enabled: true")
    print(f"  mode_conditional: true")
    print(f"  path: '{output_path}'")
    print(f"  cls_enabled: true")
    print(f"  cls_path: '{args.cls_gmm_path}'")


if __name__ == "__main__":
    main()
