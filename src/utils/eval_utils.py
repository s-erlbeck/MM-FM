"""
FID Evaluation utilities for training.
Adapted from EDM2: https://github.com/NVlabs/edm2
"""
import os
import sys
import pickle
import gc

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from scipy import linalg
from tqdm import tqdm as tqdm_std


#################################################################################
#                        FID Evaluation (PyTorch-based)                         #
#################################################################################

class InceptionV3Detector(torch.nn.Module):
    """
    InceptionV3 feature detector for FID calculation.
    This is a direct PyTorch translation of the TensorFlow InceptionV3 model,
    adapted from EDM2: https://github.com/NVlabs/edm2
    """
    def __init__(self):
        super().__init__()
        self.feature_dim = 2048

        # Add src directory to path to import local dnnlib
        src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        import dnnlib

        # Download and load the PyTorch InceptionV3 model using dnnlib
        url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with dnnlib.util.open_url(url, verbose=False) as f:
            self.model = pickle.load(f)

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


def compute_fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate FID between two Gaussian distributions.

    Args:
        mu1, sigma1: Mean and covariance of first distribution
        mu2, sigma2: Mean and covariance of second distribution
        eps: Small value for numerical stability

    Returns:
        FID score (float)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, f'Mean vectors have different shapes: {mu1.shape} vs {mu2.shape}'
    assert sigma1.shape == sigma2.shape, f'Covariance matrices have different shapes: {sigma1.shape} vs {sigma2.shape}'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f'WARNING: FID calculation produces singular product; adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_statistics_from_images(images, detector, batch_size, device, logger):
    """
    Compute mean and covariance from images using InceptionV3.

    Args:
        images: numpy array of shape [N, H, W, 3], uint8
        detector: InceptionV3Detector instance
        batch_size: Batch size for processing
        device: torch device
        logger: Logger instance

    Returns:
        mu, sigma: Mean and covariance of features
    """
    all_features = []
    num_batches = (len(images) + batch_size - 1) // batch_size

    for i in tqdm_std(range(num_batches), desc="Computing reference features"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(images))
        batch = images[start_idx:end_idx]

        # Convert NHWC to NCHW and to torch tensor
        batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)

        with torch.no_grad():
            features = detector(batch_tensor).to(torch.float64)
            all_features.append(features)

            # Free GPU memory for batch_tensor only
            del batch_tensor

    # Concatenate all features on GPU, then move to CPU
    all_features = torch.cat(all_features, dim=0).cpu()
    mu = all_features.mean(dim=0).numpy()
    sigma = np.cov(all_features.numpy(), rowvar=False)

    return mu, sigma


def evaluate_fid(
    rank: int,
    train_steps: int,
    rae,
    ema_model,
    eval_sampler,
    model_fn,
    sample_model_kwargs: dict,
    using_cfg: bool,
    autocast_kwargs: dict,
    image_size: int,
    num_fid_samples: int,
    fid_batch_size: int,
    ref_batch_path: str,
    logger,
    enable_wandb: bool,
    latent_size: tuple,
    gmm_sampler=None,
    num_classes: int = 1000,
    null_label: int = 1000,
    world_size: int = 1,
    use_y_conditioning: bool = True,
    use_mode_conditional: bool = False,
    wandb_utils=None,
):
    """
    Evaluate FID using multi-GPU distributed sampling and InceptionV3 inference.

    Args:
        rank: DDP rank
        train_steps: Current training step
        rae: RAE model for decoding
        ema_model: EMA model for sampling (used for generation)
        eval_sampler: Sampler function
        model_fn: Model forward function (using EMA model)
        sample_model_kwargs: Kwargs for model forward
        using_cfg: Whether using classifier-free guidance
        autocast_kwargs: Autocast kwargs
        image_size: Image resolution
        num_fid_samples: Number of samples to generate for FID
        fid_batch_size: Batch size for FID sampling
        ref_batch_path: Path to reference batch NPZ file
        logger: Logger
        enable_wandb: Whether to log to wandb
        latent_size: Size of latent space (C, H, W)
        gmm_sampler: GMMSampler for noise sampling (optional)
        num_classes: Number of classes
        null_label: Null label for unconditional generation
        world_size: Number of GPUs for distributed sampling
        use_y_conditioning: Whether the model uses class conditioning
        use_mode_conditional: Whether using mode-conditional generation
        wandb_utils: wandb utilities module for logging
    """
    if rank == 0:
        logger.info(f"[Step {train_steps}] Starting FID evaluation with {world_size} GPUs...")

    device = next(ema_model.parameters()).device

    try:
        # All ranks load InceptionV3 detector
        if rank == 0:
            logger.info("Loading InceptionV3 detector on all GPUs...")
        detector = InceptionV3Detector().to(device).eval()
        dist.barrier()

        # Only rank 0 loads reference statistics
        if rank == 0:
            logger.info(f"Loading reference statistics from {ref_batch_path}...")
            ref_data = np.load(ref_batch_path)
            if 'mu' in ref_data and 'sigma' in ref_data:
                # Pre-computed statistics
                ref_mu = ref_data['mu']
                ref_sigma = ref_data['sigma']
                logger.info("Using pre-computed reference statistics")
            else:
                # Need to compute statistics from reference images
                logger.info("Computing reference statistics from images...")
                ref_images = ref_data['arr_0']  # Shape: [N, H, W, 3], uint8
                ref_mu, ref_sigma = compute_statistics_from_images(ref_images, detector, fid_batch_size, device, logger)

        # Distributed sampling: each GPU generates num_fid_samples / world_size samples
        # Round up to nearest multiple of world_size
        total_samples = int(np.ceil(num_fid_samples / world_size) * world_size)
        samples_per_gpu = total_samples // world_size

        if rank == 0:
            logger.info(f"Generating {total_samples} samples ({samples_per_gpu} per GPU) using EMA model...")

        latent_dtype = torch.bfloat16 if autocast_kwargs.get('enabled', False) else torch.float32

        # Each GPU generates its portion of samples
        local_features = []
        num_batches = (samples_per_gpu + fid_batch_size - 1) // fid_batch_size

        # Set different random seed for each rank to get diverse samples
        generator = torch.Generator(device=device)
        generator.manual_seed(train_steps * world_size + rank)

        # Generate label pool based on conditioning mode
        if use_mode_conditional:
            # Mode-conditional: sample modes weighted by GMM component weights
            if rank == 0:
                logger.info("FID: Sampling modes weighted by GMM component weights")
            device_pool = gmm_sampler.sample_modes_weighted(samples_per_gpu, device, generator=generator)
        elif use_y_conditioning:
            # Conditional: generate equal-mode label pool (balanced across classes)
            if num_fid_samples % num_classes != 0:
                if rank == 0:
                    logger.info(f"WARNING: num_fid_samples ({num_fid_samples}) not divisible by num_classes ({num_classes}). "
                               f"Using closest divisible value: {(num_fid_samples // num_classes) * num_classes}")

            labels_per_class = num_fid_samples // num_classes
            base_pool = torch.arange(num_classes, dtype=torch.long, device=device).repeat_interleave(labels_per_class)

            # Shuffle the base pool for randomness
            permutation = torch.randperm(base_pool.numel(), generator=generator, device=device)
            base_pool = base_pool[permutation]

            # If total_samples > num_fid_samples (due to rounding up), fill remaining with random labels
            if total_samples > base_pool.numel():
                tail = torch.randint(0, num_classes, (total_samples - base_pool.numel(),), device=device, generator=generator)
                global_pool = torch.cat([base_pool, tail], dim=0)
            else:
                global_pool = base_pool

            # Each rank gets its portion of the global pool
            start = rank * samples_per_gpu
            end = start + samples_per_gpu
            device_pool = global_pool[start:end]
        else:
            # Unconditional: no labels needed
            device_pool = None

        pbar = tqdm_std(range(num_batches), desc=f"GPU {rank} generating samples") if rank == 0 else range(num_batches)

        for batch_idx in pbar:
            # Determine batch size for this iteration
            remaining = samples_per_gpu - batch_idx * fid_batch_size
            current_batch_size = min(fid_batch_size, remaining)

            # ================================================================
            # SAMPLE GENERATION (same hierarchy as train.py)
            # ================================================================
            # Hierarchy:
            #   use_y_conditioning: whether model uses labels (y)
            #   use_mode_conditional: when conditional, mode vs class labels

            if use_y_conditioning:
                # CONDITIONAL
                if use_mode_conditional:
                    # MODE-CONDITIONAL: y = GMM cluster modes
                    start_idx = batch_idx * fid_batch_size
                    end_idx = start_idx + current_batch_size
                    y = device_pool[start_idx:end_idx]

                    # Sample noise from GMM (required for mode-conditional)
                    zs, _ = gmm_sampler.sample(y, latent_size, device, latent_dtype)
                    batch_kwargs = {**sample_model_kwargs, 'y': y}

                else:
                    # CLASS-CONDITIONAL: y = ImageNet class labels
                    start_idx = batch_idx * fid_batch_size
                    end_idx = start_idx + current_batch_size
                    y_cond = device_pool[start_idx:end_idx]

                    if using_cfg:
                        # CFG: duplicate noise and labels
                        if gmm_sampler is not None:
                            # GMM + CLASS-CONDITIONAL with CFG
                            modes_cond = gmm_sampler.sample_modes_weighted(current_batch_size, device, generator=generator)
                            modes_uncond = gmm_sampler.sample_modes_weighted(current_batch_size, device, generator=generator)
                            zs_cond, _ = gmm_sampler.sample(modes_cond, latent_size, device, latent_dtype)
                            zs_uncond, _ = gmm_sampler.sample(modes_uncond, latent_size, device, latent_dtype)
                        else:
                            zs_cond = torch.randn(current_batch_size, *latent_size, device=device, dtype=latent_dtype, generator=generator)
                            zs_uncond = torch.randn(current_batch_size, *latent_size, device=device, dtype=latent_dtype, generator=generator)
                        zs = torch.cat([zs_cond, zs_uncond], dim=0)

                        y_null = torch.full((current_batch_size,), null_label, device=device)
                        y = torch.cat([y_cond, y_null], dim=0)
                        batch_kwargs = {**sample_model_kwargs, 'y': y}
                    else:
                        # No CFG
                        if gmm_sampler is not None:
                            # GMM + CLASS-CONDITIONAL without CFG
                            modes_for_noise = gmm_sampler.sample_modes_weighted(current_batch_size, device, generator=generator)
                            zs, _ = gmm_sampler.sample(modes_for_noise, latent_size, device, latent_dtype)
                        else:
                            zs = torch.randn(current_batch_size, *latent_size, device=device, dtype=latent_dtype, generator=generator)
                        batch_kwargs = {**sample_model_kwargs, 'y': y_cond}

            else:
                # UNCONDITIONAL
                if gmm_sampler is not None:
                    # UNCOND-GMM: sample modes for noise, model ignores labels
                    modes = gmm_sampler.sample_modes_weighted(current_batch_size, device, generator=generator)
                    zs, _ = gmm_sampler.sample(modes, latent_size, device, latent_dtype)
                else:
                    # Pure unconditional: isotropic Gaussian
                    zs = torch.randn(current_batch_size, *latent_size, device=device, dtype=latent_dtype, generator=generator)

                # Inherit autoguidance params from sample_model_kwargs (y=None for unconditional)
                batch_kwargs = {**sample_model_kwargs}

            with torch.no_grad():
                with autocast(**autocast_kwargs):
                    # Generate samples using EMA model
                    result = eval_sampler(zs, model_fn, **batch_kwargs)
                    samples = result[-1]  # Get last timestep

                    if using_cfg:
                        samples, _ = samples.chunk(2, dim=0)

                    # Decode to images
                    samples = rae.decode(samples.to(torch.float32))

                    # Resize if decoder output size doesn't match target image size
                    # (e.g., Scale-RAE decoder outputs 224x224 but we want 256x256 for FID)
                    if hasattr(rae, 'decoder_output_size') and rae.decoder_output_size != image_size:
                        samples = torch.nn.functional.interpolate(
                            samples,
                            size=(image_size, image_size),
                            mode='bicubic',
                            align_corners=False,
                        )

                    samples = samples.clamp(0, 1).mul(255).to(torch.uint8)  # [B, 3, H, W], uint8

                    # Extract features using InceptionV3 on GPU
                    features = detector(samples).to(torch.float64)
                    local_features.append(features)

                    # Free GPU memory immediately
                    del samples, result
                    torch.cuda.empty_cache()

        # Concatenate local features on GPU
        local_features = torch.cat(local_features, dim=0)[:samples_per_gpu]  # [samples_per_gpu, 2048]

        # Synchronize all ranks before gathering to catch any stragglers
        dist.barrier()

        # Gather all features using all_gather (more robust than gather)
        if rank == 0:
            logger.info("Gathering features from all GPUs...")

        # Use all_gather instead of gather for better reliability
        # All ranks allocate gather list and participate symmetrically
        gathered_features = [torch.zeros_like(local_features) for _ in range(world_size)]
        dist.all_gather(gathered_features, local_features)

        # Move to CPU and free GPU memory
        local_features = local_features.cpu()
        del local_features
        torch.cuda.empty_cache()

        # Only rank 0 computes FID
        if rank == 0:
            # Concatenate features from all GPUs and move to CPU
            all_features = torch.cat(gathered_features, dim=0)[:num_fid_samples].cpu()  # [num_fid_samples, 2048]

            logger.info("Computing sample statistics...")
            sample_mu = all_features.mean(dim=0).numpy()
            sample_sigma = np.cov(all_features.numpy(), rowvar=False)

            # Compute FID
            logger.info("Computing FID...")
            fid_score = compute_fid_from_stats(sample_mu, sample_sigma, ref_mu, ref_sigma)
            logger.info(f"[Step {train_steps}] FID: {fid_score:.2f}")

            # Log to wandb
            if enable_wandb and wandb_utils is not None:
                wandb_utils.log({"fid": fid_score, "train_step": train_steps})

        # Free GPU memory on all ranks (all_gather allocates on every rank)
        del gathered_features
        torch.cuda.empty_cache()

        # Barrier to ensure all ranks finish before continuing
        dist.barrier()

        # Clean up detector to free GPU memory before returning to training
        del detector
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error computing FID: {e}")
        import traceback
        traceback.print_exc()
        # Clean up detector even on error to prevent memory leak
        if 'detector' in locals():
            del detector
            gc.collect()
            torch.cuda.empty_cache()

    logger.info(f"[Step {train_steps}] FID evaluation complete.")
