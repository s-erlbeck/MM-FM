#!/usr/bin/env python3
"""
Sanity-check the VAE-as-flow-source pipeline on real images:
RAE.encode -> VAE.encode/decode (two decoder_std levels, two encode modes) -> FM
transport -> RAE.decode. Saves one grid image per input.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid, save_image

# Add src to path
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fit_vae_imagenet import center_crop_arr
from stage1 import RAE
from stage2.transport import create_transport, Sampler
from stage2.transport.prior import load_patch_token_vae
from utils.data_utils import ImageNetDataset
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs

DEFAULT_CONFIG = Path(
    "configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B-UNCONDITIONAL-VAE-PRIOR.yaml"
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_sample_fn(transport_config, sampler_config, misc):
    shift_dim = misc.get("time_dist_shift_dim", 768 * 16 * 16)
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = (shift_dim / shift_base) ** 0.5
    transport = create_transport(**transport_config["params"], time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)
    mode, sampler_params = sampler_config["mode"], sampler_config["params"]
    if mode != "ODE":
        raise NotImplementedError(f"Only ODE sampling is supported, got sampler.mode={mode}.")
    return sampler.sample_ode(**sampler_params)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Config with stage_1/stage_2/transport/sampler/vae_prior sections.")
    parser.add_argument("--vae-ckpt", type=str, default=None,
                        help="PatchTokenVAE checkpoint (default: config's vae_prior.ckpt_path).")
    parser.add_argument("--fm-ckpt", type=str, required=True,
                        help="Trained FM (stage_2) checkpoint.")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the train.zip archive (see ImageNetDataset).")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("vae_fm_recon_out"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    (rae_config, stage2_config, transport_config, sampler_config, _guidance_config,
     misc, *_rest, vae_prior_config) = parse_configs(str(args.config))

    rae: RAE = instantiate_from_config(rae_config).to(device).eval()

    stage2_config = dict(stage2_config)
    stage2_config["ckpt"] = args.fm_ckpt
    model = instantiate_from_config(stage2_config).to(device).eval()

    vae_ckpt = args.vae_ckpt or (vae_prior_config.get("ckpt_path") if vae_prior_config else None)
    if vae_ckpt is None:
        raise ValueError("No --vae-ckpt given and none found in config's vae_prior.ckpt_path.")
    vae, _ = load_patch_token_vae(vae_ckpt, device)

    sample_fn = build_sample_fn(transport_config, sampler_config, misc)

    # Same deterministic center-crop as fit_vae_imagenet.py's get_transform(), minus its
    # Normalize step: rae.encode() already normalizes with the encoder's mean/std.
    image_size = rae_config.params.get("encoder_input_size", 224)
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
    ])
    dataset = ImageNetDataset(args.data_path, transform=transform)
    num_images = min(args.num_images, len(dataset))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(num_images):
        image, label = dataset[idx]
        image = image.unsqueeze(0).to(device)
        x1 = rae.encode(image)
        dist = vae.encode(x1.float())

        variant_names = ["mode", "sample"]
        z_by_variant = {"mode": dist.mode(), "sample": dist.sample()}
        std0_by_variant = {name: vae.decode(z_by_variant[name]) for name in variant_names}
        std1_by_variant = {
            name: std0_by_variant[name] + torch.randn_like(std0_by_variant[name])
            for name in variant_names
        }

        # always use decoder mode for transport
        x0_batch = torch.cat([std0_by_variant[name] for name in variant_names], dim=0)
        transported = sample_fn(x0_batch, model.forward)[-1]

        recon_shape = std0_by_variant[variant_names[0]].shape[-2:]
        panels = [F.interpolate(image, size=recon_shape, mode="bilinear", align_corners=False)]
        for i, name in enumerate(variant_names):
            panels.append(rae.decode(std0_by_variant[name]))
            panels.append(rae.decode(std1_by_variant[name]))
            panels.append(rae.decode(transported[i:i + 1]))

        grid = make_grid(torch.cat(panels, dim=0).clamp(0.0, 1.0), nrow=len(panels))
        out_path = args.output_dir / f"{idx:03d}_cls{label}.png"
        save_image(grid, out_path)
        print(f"[{idx + 1}/{num_images}] saved {out_path}")


if __name__ == "__main__":
    main()
