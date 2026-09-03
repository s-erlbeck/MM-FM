#!/usr/bin/env python3

import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

# Add src to path
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from stage1.encoders import ARCHS
from transformers import AutoImageProcessor
from utils.data_utils import ClassBalancedSubset, ImageNetDataset
from vae import PatchTokenVAE


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


def get_transform(image_size: int, encoder_mean: list, encoder_std: list):
    """Create deterministic transform for CLS token extraction (no random augmentation)."""
    normalize = transforms.Normalize(mean=encoder_mean, std=encoder_std)
    return transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Data settings
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the train_blurred.zip archive (see ImageNetDataset)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config (for encoder settings)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save model and results")

    # Processing settings
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size per GPU (default: 64)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--precision", type=str, default='bf16',
                        choices=['fp32', 'bf16'],
                        help="Precision for extraction (default: bf16)")

    # VAE settings
    parser.add_argument("--latent-channels", type=int, default=128,
                        help="VAE latent channels (default: 128)")
    parser.add_argument("--hidden-channels", type=int, default=256,
                        help="VAE hidden channels (default: 256)")
    parser.add_argument("--num-res-blocks", type=int, default=2,
                        help="ResNet blocks per VAE stage (default: 2)")
    parser.add_argument("--num-groups", type=int, default=32,
                        help="GroupNorm groups (default: 32)")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="VAE dropout (default: 0.0)")

    # Training settings
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="AdamW weight decay (default: 0.0)")
    parser.add_argument("--kl-weight", type=float, default=1e-6,
                        help="KL loss weight (default: 1e-6)")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Steps between progress bar updates (default: 50)")
    parser.add_argument("--save-every-steps", type=int, default=2000,
                        help="Steps between intra-epoch checkpoints, 0 to disable (default: 2000)")

    # Data limiting
    parser.add_argument("--data-limit-sample-percentage", type=float, default=1.0,
                        help="Percentage of samples to use per kept class (0.0-1.0, default: 1.0)")
    parser.add_argument("--data-limit-class-percentage", type=float, default=1.0,
                        help="Percentage of classes to keep (0.0-1.0, default: 1.0)")
    parser.add_argument("--data-limit-seed", type=int, default=42,
                        help="Random seed for data limiting (default: 42)")

    return parser


def load_encoder(config_path: str, device: torch.device):
    """Load the frozen vision encoder and its preprocessing settings from a training config."""
    from utils.train_utils import parse_configs
    rae_config, *_ = parse_configs(config_path)

    encoder_cls = rae_config.params.encoder_cls
    encoder_config_path = rae_config.params.encoder_config_path
    encoder_params = dict(rae_config.params.get('encoder_params', {}))
    image_size = rae_config.params.get('encoder_input_size', 224)

    print(f"\nEncoder: {encoder_cls}")
    print(f"Config: {encoder_config_path}")
    print(f"Image size: {image_size}")

    encoder_class = ARCHS[encoder_cls]
    encoder = encoder_class(**encoder_params).to(device)
    encoder.eval()

    proc = AutoImageProcessor.from_pretrained(encoder_config_path)
    print(f"\nEncoder loaded: {encoder_cls}")
    print(f"Hidden size: {encoder.hidden_size}")

    return encoder, image_size, proc


def build_dataloader(args: argparse.Namespace, image_size: int, proc) -> DataLoader:
    transform = get_transform(image_size, proc.image_mean, proc.image_std)
    dataset = ImageNetDataset(args.data_path, transform=transform)

    if args.data_limit_sample_percentage < 1.0 or args.data_limit_class_percentage < 1.0:
        original_size = len(dataset)
        print(
            f"\nApplying data limit: {args.data_limit_sample_percentage*100:.1f}% of samples from "
            f"{args.data_limit_class_percentage*100:.1f}% of classes (seed={args.data_limit_seed})"
        )
        dataset = ClassBalancedSubset(
            dataset,
            sample_percentage=args.data_limit_sample_percentage,
            class_percentage=args.data_limit_class_percentage,
            seed=args.data_limit_seed,
        )
        print(f"Dataset reduced from {original_size:,} to {len(dataset):,} samples")

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def build_vae(args: argparse.Namespace, in_channels: int, device: torch.device) -> PatchTokenVAE:
    vae = PatchTokenVAE(
        in_channels=in_channels,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        num_res_blocks=args.num_res_blocks,
        num_groups=args.num_groups,
        dropout=args.dropout,
    ).to(device)
    print(f"\nVAE: in_channels={in_channels}, latent_channels={args.latent_channels}, "
          f"hidden_channels={args.hidden_channels}, num_res_blocks={args.num_res_blocks}")
    return vae


def save_checkpoint(path: str, vae: PatchTokenVAE, optimizer: torch.optim.Optimizer,
                     epoch: int, step: int, args: argparse.Namespace):
    torch.save({
        "model": vae.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": vars(args),
    }, path)
    print(f"Saved checkpoint: {path}")


def infer_size(encoder, image_size: int, device: torch.device) -> tuple:
    """Run the encoder once on a dummy image to determine its (H, W, C) spatial token grid shape."""
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        spatial_tokens, _ = encoder(dummy)
    n, c = spatial_tokens.shape[1], spatial_tokens.shape[2]
    h = w = int(round(n ** 0.5))
    return h, w, c


def train_one_epoch(vae: PatchTokenVAE, encoder, dataloader: DataLoader,
                     optimizer: torch.optim.Optimizer, device: torch.device,
                     args: argparse.Namespace, epoch: int, step: int, grid_size: tuple) -> int:
    vae.train()
    amp_dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    use_amp = args.precision == 'bf16'
    h, w, c = grid_size

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
    for images, _ in pbar:
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            with torch.no_grad():
                spatial_tokens, _ = encoder(images)
            patch_grid = spatial_tokens.float().permute(0, 2, 1).reshape(-1, c, h, w)
            loss, logs = vae.loss(patch_grid, kl_weight=args.kl_weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1
        if step % args.log_every == 0:
            pbar.set_postfix(
                loss=f"{logs['loss'].item():.4f}",
                rec=f"{logs['rec_loss'].item():.4f}",
                kl=f"{logs['kl_loss'].item():.4f}",
            )

        if args.save_every_steps and step % args.save_every_steps == 0:
            save_checkpoint(os.path.join(args.output_dir, "vae_latest.pt"), vae, optimizer, epoch + 1, step, args)

    return step


def train(vae: PatchTokenVAE, encoder, dataloader: DataLoader,
          optimizer: torch.optim.Optimizer, device: torch.device, args: argparse.Namespace, grid_size: tuple):
    step = 0
    for epoch in range(args.epochs):
        step = train_one_epoch(vae, encoder, dataloader, optimizer, device, args, epoch, step, grid_size)
        save_checkpoint(os.path.join(args.output_dir, f"vae_epoch{epoch + 1}.pt"), vae, optimizer, epoch + 1, step, args)

    final_path = os.path.join(args.output_dir, "vae_final.pt")
    torch.save({"model": vae.state_dict(), "args": vars(args)}, final_path)
    print(f"\nSaved final VAE model to {final_path}")


def main():
    args = build_arg_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, image_size, proc = load_encoder(args.config, device)
    dataloader = build_dataloader(args, image_size, proc)
    print(f"\nDataset size: {len(dataloader.dataset):,}")

    os.makedirs(args.output_dir, exist_ok=True)

    h, w, c = infer_size(encoder, image_size, device)
    print(f"\nPatch grid: {h}x{w}, channels={c}")

    vae = build_vae(args, c, device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train(vae, encoder, dataloader, optimizer, device, args, (h, w, c))


if __name__ == "__main__":
    main()
