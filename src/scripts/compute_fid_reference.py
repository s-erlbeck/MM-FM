#!/usr/bin/env python3
"""
Compute FID reference statistics from a dataset.

This script extracts InceptionV3 features from real images and computes
mean/covariance statistics for FID calculation during training.

Output format (NPZ file):
    - mu: Mean feature vector (shape: 2048)
    - sigma: Covariance matrix (shape: 2048 x 2048)

Usage:
    # For ImageNetDataset (ImageNet):
    python src/scripts/compute_fid_reference.py \
        --data-path /path/to/imagenet/train.zip \
        --output models/fid_refs/imagenet256.npz \
        --image-size 256 \
        --num-samples 50000 \
        --batch-size 64

    # For faster processing with multiple GPUs:
    torchrun --nproc_per_node=4 src/scripts/compute_fid_reference.py \
        --data-path /path/to/imagenet/train.zip \
        --output models/fid_refs/imagenet256.npz \
        --num-samples 50000
"""

import argparse
import os
import sys
import pickle

# Add src to path for imports
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from utils.data_utils import ImageNetDataset


class InceptionV3Detector(torch.nn.Module):
    """
    InceptionV3 feature detector for FID calculation.
    Adapted from EDM2: https://github.com/NVlabs/edm2
    """
    def __init__(self):
        super().__init__()
        self.feature_dim = 2048

        import dnnlib

        # Download and load the PyTorch InceptionV3 model
        url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        print(f"Loading InceptionV3 from: {url}")
        with dnnlib.util.open_url(url, verbose=True) as f:
            self.model = pickle.load(f)
        print("InceptionV3 loaded successfully")

    @torch.no_grad()
    def forward(self, x):
        """
        Extract InceptionV3 features from images.
        Args:
            x: Images in [0, 255] range, NCHW format, uint8
        Returns:
            features: Feature vectors of shape [N, 2048]
        """
        return self.model.to(x.device)(x, return_features=True)


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


def get_fid_transform(image_size: int):
    """Get transform for FID reference images."""
    return transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),  # Convert to uint8 [0, 255]
    ])


def is_distributed():
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Get current rank in distributed mode."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get world size in distributed mode."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def compute_fid_stats(detector, dataloader, num_samples, device, rank=0, world_size=1):
    """
    Compute mean and covariance for FID from a dataloader.

    Args:
        detector: InceptionV3Detector instance
        dataloader: DataLoader for images
        num_samples: Total number of samples to process
        device: torch device
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        mu: Mean feature vector (shape: 2048)
        sigma: Covariance matrix (shape: 2048 x 2048)
    """
    detector.eval()
    all_features = []

    samples_per_rank = (num_samples + world_size - 1) // world_size
    processed = 0

    pbar = tqdm(total=samples_per_rank, desc=f"[Rank {rank}] Extracting features") if rank == 0 else None

    with torch.no_grad():
        for batch in dataloader:
            if processed >= samples_per_rank:
                break

            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            # Ensure uint8 format for InceptionV3
            if images.dtype != torch.uint8:
                images = (images * 255).clamp(0, 255).to(torch.uint8)

            images = images.to(device)
            current_batch_size = min(images.size(0), samples_per_rank - processed)
            images = images[:current_batch_size]

            # Extract features (use float32 to save memory, sufficient for FID)
            features = detector(images).float()  # float32
            all_features.append(features)  # Keep on GPU

            processed += current_batch_size

            if pbar is not None:
                pbar.update(current_batch_size)

    if pbar is not None:
        pbar.close()

    # Concatenate local features (on GPU)
    local_features = torch.cat(all_features, dim=0)[:samples_per_rank]

    # Free the list of tensors to reduce memory before all_gather
    del all_features
    torch.cuda.empty_cache()

    if world_size > 1:
        # Gather features from all ranks (must be on GPU for NCCL)
        print(f"[Rank {rank}] Gathered {local_features.shape[0]} features, synchronizing...")
        gathered_list = [torch.zeros_like(local_features) for _ in range(world_size)]
        dist.all_gather(gathered_list, local_features)

        # Free local_features before concatenating gathered results
        del local_features
        torch.cuda.empty_cache()

        all_features = torch.cat(gathered_list, dim=0)[:num_samples].cpu()
        del gathered_list
        torch.cuda.empty_cache()
    else:
        all_features = local_features[:num_samples].cpu()
        del local_features
        torch.cuda.empty_cache()

    if rank == 0:
        print(f"Computing statistics from {all_features.shape[0]} features...")

    # Compute mean and covariance (on CPU, convert to float64 for precision)
    all_features_np = all_features.numpy().astype(np.float64)
    mu = np.mean(all_features_np, axis=0)
    sigma = np.cov(all_features_np, rowvar=False)

    return mu, sigma


def main():
    parser = argparse.ArgumentParser(description="Compute FID reference statistics")

    # Data settings
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to the train.zip archive (see ImageNetDataset)")

    # Output settings
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for FID reference file (*.npz)")

    # Image settings
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image size for FID (typically 256 or 299)")

    # Compute settings
    parser.add_argument("--num-samples", type=int, default=50000,
                        help="Number of samples to use for FID reference")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader workers")

    # Distributed settings (auto-detected from torchrun)
    parser.add_argument("--local-rank", type=int, default=0,
                        help="Local rank (auto-set by torchrun)")

    args = parser.parse_args()

    # Validate data source
    if args.data_path is None:
        raise ValueError("Must provide --data-path")

    # Initialize distributed if available
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        print(f"Using device: {device}")
        print(f"World size: {world_size}")

    # Load InceptionV3 detector
    detector = InceptionV3Detector().to(device)
    detector.eval()

    # Get transform
    transform = get_fid_transform(args.image_size)

    # Create dataset and dataloader (ImageNetDataset)
    if rank == 0:
        print(f"Loading ImageNetDataset from: {args.data_path}")

    full_dataset = ImageNetDataset(args.data_path, transform=transform)

    # Subsample if dataset is larger than num_samples
    if len(full_dataset) > args.num_samples:
        indices = torch.randperm(len(full_dataset))[:args.num_samples].tolist()
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    # Compute statistics
    if rank == 0:
        print(f"Computing FID statistics over {args.num_samples} samples...")

    mu, sigma = compute_fid_stats(
        detector, dataloader, args.num_samples, device, rank, world_size
    )

    # Only rank 0 saves the result
    if rank == 0:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        np.savez(args.output, mu=mu, sigma=sigma)
        print(f"Saved FID reference to: {args.output}")

        # Print summary
        print("\nFID Reference Summary:")
        print(f"  Samples: {args.num_samples}")
        print(f"  Image size: {args.image_size}")
        print(f"  Feature dim: {mu.shape[0]}")
        print(f"  Mean norm: {np.linalg.norm(mu):.4f}")
        print(f"  Sigma trace: {np.trace(sigma):.4f}")

    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
