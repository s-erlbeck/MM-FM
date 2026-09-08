# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained stage-2 model using DDP and
stores results for downstream metrics. For single-device sampling, use sample.py.

Supports:
- Class-conditional sampling (with CFG)
- Mode-conditional sampling (GMM-based)
- Unconditional sampling
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast
from tqdm import tqdm
from pathlib import Path

from omegaconf import OmegaConf
from utils.model_utils import instantiate_from_config, is_model_conditional
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport, Sampler
from stage2.transport.gmm_sampler import GMMSampler
from utils.train_utils import parse_configs


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def build_label_sampler(
    sampling_mode: str,
    num_classes: int,
    num_fid_samples: int,
    total_samples: int,
    samples_needed_this_device: int,
    batch_size: int,
    device: torch.device,
    rank: int,
    iterations: int,
    seed: int,
    label_counts_path: Optional[str] = None,
) -> Callable[[int], torch.Tensor]:
    """Create a callable that returns a batch of labels for the given step index."""

    if sampling_mode == "random":
        def random_sampler(_step_idx: int) -> torch.Tensor:
            return torch.randint(0, num_classes, (batch_size,), device=device)
        return random_sampler

    if sampling_mode == "equal":
        if num_fid_samples % num_classes != 0:
            raise ValueError(
                f"Equal label sampling requires num_fid_samples ({num_fid_samples}) to be divisible by num_classes ({num_classes})."
            )

        labels_per_class = num_fid_samples // num_classes
        base_pool = torch.arange(num_classes, dtype=torch.long).repeat_interleave(labels_per_class)

        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(base_pool.numel(), generator=generator)
        base_pool = base_pool[permutation]

        if total_samples > num_fid_samples:
            tail = torch.randint(0, num_classes, (total_samples - num_fid_samples,), generator=generator)
            global_pool = torch.cat([base_pool, tail], dim=0)
        else:
            global_pool = base_pool

        start = rank * samples_needed_this_device
        end = start + samples_needed_this_device
        device_pool = global_pool[start:end]
        device_pool = device_pool.view(iterations, batch_size)

        def equal_sampler(step_idx: int) -> torch.Tensor:
            labels = device_pool[step_idx]
            return labels.to(device)

        return equal_sampler

    raise ValueError(f"Unknown label sampling mode: {sampling_mode}")


def main(args):
    """Run sampling with distributed execution."""
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling with DDP requires at least one GPU. Use sample.py for single-device usage.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)
    latent_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # Parse config (12 return values)
    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
        gmm_config,
        fid_config,
        data_limit_config,
        data_config,  # WebDataset config (unused in sampling)
        vae_prior_config,
    ) = parse_configs(args.config)

    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 entries.")

    # The VAE prior is only wired into train.py / evaluate_fid so far. Sampling a
    # prior-trained model from N(0, I) produces garbage that looks like a training failure.
    if vae_prior_config is not None and vae_prior_config.get("enabled", False):
        raise NotImplementedError(
            "vae_prior.enabled is not supported by sample_ddp.py yet. Use the in-training "
            "FID evaluation (fid.enabled) instead."
        )

    misc = {} if misc_config is None else dict(misc_config)
    gmm_cfg = {} if gmm_config is None else dict(gmm_config)
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    if rank == 0:
        print(f"Using time_dist_shift={time_dist_shift:.4f}.")

    # Load models
    rae: RAE = instantiate_from_config(rae_config).to(device)
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    rae.eval()
    model.eval()

    # Detect conditioning mode from model
    use_y_conditioning = is_model_conditional(model)
    use_mode_conditional = gmm_cfg.get('mode_conditional', False)
    use_gmm = gmm_cfg.get('enabled', False)

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))

    # For mode-conditional, num_classes should be GMM components
    if use_mode_conditional:
        num_classes = int(model_config.get("params", {}).get("num_classes", 8192))

    if rank == 0:
        print("=" * 80)
        if use_mode_conditional:
            print("MODE-CONDITIONAL SAMPLING")
            print(f"  GMM spatial: {gmm_cfg.get('path')}")
            print(f"  GMM modes: {num_classes}")
        elif use_y_conditioning:
            print("CLASS-CONDITIONAL SAMPLING")
            print(f"  Classes: {num_classes}")
        else:
            print("UNCONDITIONAL SAMPLING")
        print("=" * 80)

    # Initialize GMM sampler if needed
    gmm_sampler = None
    if use_gmm:
        gmm_path = gmm_cfg.get('path')
        cls_gmm_path = gmm_cfg.get('cls_path') if gmm_cfg.get('cls_enabled', False) else None
        gmm_sampler = GMMSampler(
            gmm_path=gmm_path,
            cls_gmm_path_for_assignment=cls_gmm_path,
            cls_assignment_mode=gmm_cfg.get('cls_assignment_mode', 'soft'),
            verbose=(rank == 0),
        )
        if rank == 0:
            print(f"GMM sampler initialized: {gmm_sampler.num_classes} components")

    # Create transport and sampler
    transport_params = {}
    if transport_config is not None:
        transport_params = dict(transport_config.get("params", {}))
    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
        gmm_sampler=gmm_sampler,
    )
    sampler = Sampler(transport)

    sampler_config = {} if sampler_config is None else dict(sampler_config)
    sampler_mode = sampler_config.get("mode", "ODE")
    sampler_params = dict(sampler_config.get("params", {}))
    mode = sampler_mode.upper()

    if mode == "ODE":
        sample_fn = sampler.sample_ode(**sampler_params)
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    # Guidance config
    guidance_config = {} if guidance_config is None else dict(guidance_config)

    def guidance_value(key: str, default: float):
        if key in guidance_config:
            return guidance_config[key]
        dashed_key = key.replace("_", "-")
        return guidance_config.get(dashed_key, default)

    guidance_scale = float(guidance_config.get("scale", 1.0))
    guidance_method = guidance_config.get("method", "cfg")
    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))

    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guid_model_config = guidance_config.get("guidance_model")
        if guid_model_config is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guid_model_config).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward
        if rank == 0:
            print(f"Autoguidance enabled: scale={guidance_scale}, interval=[{t_min}, {t_max}]")

    # Generate output folder name
    model_target = model_config.get("target", "stage2")
    model_string_name = str(model_target).split(".")[-1]
    ckpt_path = model_config.get("ckpt")
    ckpt_string_name = "pretrained" if not ckpt_path else os.path.splitext(os.path.basename(str(ckpt_path)))[0]
    sampling_method = sampler_params.get("sampling_method", "na")
    num_steps = sampler_params.get("num_steps", sampler_params.get("steps", "na"))
    guidance_tag = f"cfg-{guidance_scale:.2f}"
    base_components = [model_string_name, ckpt_string_name, guidance_tag, f"bs{args.per_proc_batch_size}"]
    if mode == "ODE":
        detail_components = [mode, str(num_steps), str(sampling_method), args.precision]
    else:
        diffusion_form = sampler_params.get("diffusion_form", "na")
        last_step = sampler_params.get("last_step", "na")
        last_step_size = sampler_params.get("last_step_size", "na")
        detail_components = [mode, str(num_steps), str(sampling_method), str(diffusion_form), str(last_step), str(last_step_size), args.precision]
    folder_name = "-".join(component.replace(os.sep, "-") for component in base_components + detail_components)
    sample_folder_dir = os.path.join(args.sample_dir, folder_name)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Compute sample counts
    n = args.per_proc_batch_size
    global_batch_size = n * world_size
    existing = [name for name in os.listdir(sample_folder_dir) if (os.path.isfile(os.path.join(sample_folder_dir, name)) and name.endswith(".png"))]
    num_samples = len(existing)
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    if total_samples % world_size != 0:
        raise ValueError("Total samples must be divisible by world size.")
    samples_needed_this_gpu = total_samples // world_size
    if samples_needed_this_gpu % n != 0:
        raise ValueError("Per-rank sample count must be divisible by the per-GPU batch size.")
    iterations = samples_needed_this_gpu // n
    pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)
    total = (num_samples // world_size) * world_size

    # Determine if using CFG
    using_cfg = guidance_scale > 1.0 and use_y_conditioning and not use_mode_conditional

    # Build label sampler for class-conditional mode
    label_sampler = None
    if use_y_conditioning and not use_mode_conditional:
        label_sampler = build_label_sampler(
            args.label_sampling,
            num_classes,
            args.num_fid_samples,
            total_samples,
            samples_needed_this_gpu,
            n,
            device,
            rank,
            iterations,
            args.global_seed,
        )

    # ========================================================================
    # SAMPLING LOOP
    # ========================================================================
    for step_idx in pbar:
        with autocast(**autocast_kwargs):
            # ----------------------------------------------------------------
            # MODE-CONDITIONAL: sample modes weighted by GMM
            # ----------------------------------------------------------------
            if use_mode_conditional:
                y = gmm_sampler.sample_modes_weighted(n, device)
                z, _ = gmm_sampler.sample(y, latent_size, device, latent_dtype)

                if guidance_method == "autoguidance" and guid_model_forward is not None:
                    model_kwargs = dict(
                        y=y,
                        cfg_scale=guidance_scale,
                        cfg_interval=(t_min, t_max),
                        additional_model_forward=guid_model_forward,
                    )
                    model_fn = model.forward_with_autoguidance
                else:
                    model_kwargs = dict(y=y)
                    model_fn = model.forward

                samples = sample_fn(z, model_fn, **model_kwargs)[-1]

            # ----------------------------------------------------------------
            # CLASS-CONDITIONAL: sample class labels
            # ----------------------------------------------------------------
            elif use_y_conditioning:
                y = label_sampler(step_idx)

                # Sample noise (GMM if available, otherwise isotropic)
                if gmm_sampler is not None:
                    # GMM + CLASS-CONDITIONAL: sample modes for noise (y is for model)
                    modes_for_noise = gmm_sampler.sample_modes_weighted(n, device)
                    z, _ = gmm_sampler.sample(modes_for_noise, latent_size, device, latent_dtype)
                else:
                    z = torch.randn(n, *latent_size, device=device, dtype=latent_dtype)

                if using_cfg:
                    # CFG: need separate noise for unconditional branch
                    if gmm_sampler is not None:
                        # For unconditional branch, use marginal sampling (no class info)
                        modes_uncond = gmm_sampler.sample_modes_weighted(n, device)
                        z_uncond, _ = gmm_sampler.sample(modes_uncond, latent_size, device, latent_dtype)
                        z = torch.cat([z, z_uncond], dim=0)
                    else:
                        z = torch.cat([z, z], dim=0)
                    y_null = torch.full((n,), null_label, device=device)
                    y = torch.cat([y, y_null], dim=0)
                    model_kwargs = dict(
                        y=y,
                        cfg_scale=guidance_scale,
                        cfg_interval=(t_min, t_max),
                    )
                    if guidance_method == "autoguidance":
                        if guid_model_forward is None:
                            raise RuntimeError("Guidance model forward is not initialized.")
                        model_kwargs["additional_model_forward"] = guid_model_forward
                        model_fn = model.forward_with_autoguidance
                    else:
                        model_fn = model.forward_with_cfg

                    samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                    samples, _ = samples.chunk(2, dim=0)
                else:
                    if guidance_method == "autoguidance" and guid_model_forward is not None:
                        model_kwargs = dict(
                            y=y,
                            cfg_scale=guidance_scale,
                            cfg_interval=(t_min, t_max),
                            additional_model_forward=guid_model_forward,
                        )
                        model_fn = model.forward_with_autoguidance
                    else:
                        model_kwargs = dict(y=y)
                        model_fn = model.forward

                    samples = sample_fn(z, model_fn, **model_kwargs)[-1]

            # ----------------------------------------------------------------
            # UNCONDITIONAL: no labels
            # ----------------------------------------------------------------
            else:
                # Sample noise (GMM if available, otherwise isotropic)
                if gmm_sampler is not None:
                    modes = gmm_sampler.sample_modes_weighted(n, device)
                    z, _ = gmm_sampler.sample(modes, latent_size, device, latent_dtype)
                else:
                    z = torch.randn(n, *latent_size, device=device, dtype=latent_dtype)

                if guidance_method == "autoguidance" and guid_model_forward is not None:
                    model_kwargs = dict(
                        y=None,
                        cfg_scale=guidance_scale,
                        cfg_interval=(t_min, t_max),
                        additional_model_forward=guid_model_forward,
                    )
                    model_fn = model.forward_with_autoguidance
                else:
                    model_kwargs = dict()
                    model_fn = model.forward

                samples = sample_fn(z, model_fn, **model_kwargs)[-1]

            # Decode samples
            samples = rae.decode(samples)

            # Resize if target image size is specified and differs from decoder output
            # (e.g., Scale-RAE decoder outputs 224x224 but we want 256x256 for ImageNet256)
            if args.image_size is not None and hasattr(rae, 'decoder_output_size'):
                if rae.decoder_output_size != args.image_size:
                    samples = torch.nn.functional.interpolate(
                        samples,
                        size=(args.image_size, args.image_size),
                        mode='bicubic',
                        align_corners=False,
                    )

            samples = samples.clamp(0, 1)
            samples = samples.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples
        for local_idx, sample in enumerate(samples):
            index = local_idx * world_size + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

        total += global_batch_size
        dist.barrier()

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=4)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable TF32 matmuls (Ampere+). Disable if deterministic results are required.")
    parser.add_argument(
        "--label-sampling",
        type=str,
        choices=["random", "equal"],
        default="equal",
        help="Choose how to sample class labels when generating images (class-conditional only).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Target output image size. If specified and differs from decoder output, images will be resized. "
             "E.g., use 256 for ImageNet256 when decoder outputs 224.",
    )

    args = parser.parse_args()
    main(args)
