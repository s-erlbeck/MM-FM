#!/usr/bin/env python3

import os
import sys
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import CosineSimilarity, MeanAbsoluteError, MeanMetric, MeanSquaredError
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

# Add src to path
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from stage1.encoders import ARCHS
from transformers import AutoImageProcessor
from utils import wandb_utils
from utils.data_utils import ClassBalancedSubset, ImageNetDataset
from vae import PatchTokenVAE


class SpatialTokenEncoder(nn.Module):
    """Wraps a frozen vision encoder, reshaping its spatial tokens into a (C, H, W) patch grid
    and applying RAE-normalization."""

    def __init__(self, encoder, image_size: int, device: torch.device,
                 normalization_stat_path: Optional[str] = None, eps: float = 1e-5):
        super().__init__()
        self.encoder = encoder
        self.channels, self.h, self.w = self.infer_shape(image_size, device)
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location=device)
            self.latent_mean = stats.get('mean', 0)
            self.latent_var = stats.get('var', 1)
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False

    @torch.no_grad()
    def infer_shape(self, image_size, device):
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        spatial_tokens, _ = self.encoder(dummy)
        _, n, c = spatial_tokens.shape
        h = int(round(n ** 0.5))
        return c, h, h

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        spatial_tokens, _ = self.encoder(images)
        z = spatial_tokens.float().permute(0, 2, 1).reshape(-1, self.channels, self.h, self.w)
        if self.do_normalization:
            z = (z - self.latent_mean) / torch.sqrt(self.latent_var + self.eps)
        return z


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


def load_encoder(config_path: str, normalize: bool, device: torch.device):
    """Load the frozen vision encoder and its preprocessing settings from a training config."""
    from utils.train_utils import parse_configs
    rae_config, *_ = parse_configs(config_path)

    encoder_cls = rae_config.params.encoder_cls
    encoder_config_path = rae_config.params.encoder_config_path
    encoder_params = dict(rae_config.params.get('encoder_params', {}))
    image_size = rae_config.params.get('encoder_input_size', 224)
    # load normalization stats from RAE config as well
    normalization_stat_path = rae_config.params.get('normalization_stat_path', None) if normalize else None
    eps = rae_config.params.get('eps', 1e-5)

    print(f"\nEncoder: {encoder_cls}")
    print(f"Config: {encoder_config_path}")
    print(f"Image size: {image_size}")

    encoder_class = ARCHS[encoder_cls]
    encoder = encoder_class(**encoder_params).to(device)
    encoder.eval()

    proc = AutoImageProcessor.from_pretrained(encoder_config_path)
    print(f"\nEncoder loaded: {encoder_cls}")
    print(f"Hidden size: {encoder.hidden_size}")

    encoder = SpatialTokenEncoder(encoder, image_size, device, normalization_stat_path, eps)
    print(f"Patch grid: {encoder.h}x{encoder.w}, channels={encoder.channels}")

    transform = get_transform(image_size, proc.image_mean, proc.image_std)
    return encoder, transform


def build_dataset(data_path: str, transform: transforms.Compose, args: argparse.Namespace,
                   sample_percentage: float, shuffle: bool) -> DataLoader:
    dataset = ImageNetDataset(data_path, transform=transform)

    if sample_percentage < 1.0 or args.data_limit_class_percentage < 1.0:
        print(f"\nApplying data limit: {sample_percentage*100:.1f}% of samples from "
              f"{args.data_limit_class_percentage*100:.1f}% of classes (seed={args.data_limit_seed})")
        original_size = len(dataset)
        dataset = ClassBalancedSubset(
            dataset,
            sample_percentage=sample_percentage,
            class_percentage=args.data_limit_class_percentage,
            seed=args.data_limit_seed,
        )
        print(f"Dataset reduced from {original_size:,} to {len(dataset):,} samples")
    else:
        print(f"Dataset size: {len(dataset):,}")

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
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


def train_one_epoch(vae: PatchTokenVAE, encoder: SpatialTokenEncoder, dataloader: DataLoader,
                     optimizer: torch.optim.Optimizer, device: torch.device,
                     args: argparse.Namespace, epoch: int, step: int) -> int:
    vae.train()
    amp_dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    use_amp = args.precision == 'bf16'

    metrics = {name: MeanMetric().to(device) for name in ("loss", "rec_loss", "kl_loss")}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
    for images, _ in pbar:
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            patch_grid = encoder(images)
            loss, logs = vae.loss(patch_grid, kl_weight=args.kl_weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1
        for name, metric in metrics.items():
            metric.update(logs[name].expand(images.size(0)))

    epoch_stats = {name: metric.compute().item() for name, metric in metrics.items()}
    if args.wandb:
        wandb_utils.log({f"train/{key}": value for key, value in epoch_stats.items()}, step=epoch + 1)

    return step


@torch.no_grad()
def validate_one_epoch(vae: PatchTokenVAE, encoder: SpatialTokenEncoder, dataloader: DataLoader,
                       device: torch.device, args: argparse.Namespace, epoch: int, step: int) -> int:
    vae.eval()
    amp_dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    use_amp = args.precision == 'bf16'

    loss_metrics = {name: MeanMetric().to(device) for name in ("loss", "rec_loss", "kl_loss")}
    mse_metric = MeanSquaredError().to(device)
    mae_metric = MeanAbsoluteError().to(device)
    cos_sim_metric = CosineSimilarity(reduction="mean").to(device)

    pbar = tqdm(dataloader, desc=f"Validation {epoch + 1}/{args.epochs}")
    for images, _ in pbar:
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            patch_grid = encoder(images)
            # loss terms use a sampled latent, to stay comparable to training
            x_rec, posterior = vae(patch_grid)
            rec_loss = F.mse_loss(x_rec, patch_grid)
            kl_loss = posterior.kl().mean()
            loss = rec_loss + args.kl_weight * kl_loss
            # reconstruction metrics use the posterior mode, i.e. without sampling noise
            x_rec_mode = vae.decode(posterior.mode())

        step += 1
        loss_metrics["loss"].update(loss.expand(images.size(0)))
        loss_metrics["rec_loss"].update(rec_loss.expand(images.size(0)))
        loss_metrics["kl_loss"].update(kl_loss.expand(images.size(0)))
        mse_metric.update(x_rec_mode.contiguous(), patch_grid.contiguous())
        mae_metric.update(x_rec_mode.contiguous(), patch_grid.contiguous())
        cos_sim_metric.update(x_rec_mode.flatten(1), patch_grid.flatten(1))

    epoch_stats = {name: metric.compute().item() for name, metric in loss_metrics.items()}
    epoch_stats["mse"] = mse_metric.compute().item()
    epoch_stats["mae"] = mae_metric.compute().item()
    epoch_stats["cos_sim"] = cos_sim_metric.compute().item()
    if args.wandb:
        wandb_utils.log({f"val/{key}": value for key, value in epoch_stats.items()}, step=epoch + 1)

    return step


def train(vae: PatchTokenVAE, encoder: SpatialTokenEncoder, train_loader: DataLoader, val_loader: DataLoader,
          optimizer: torch.optim.Optimizer, device: torch.device, args: argparse.Namespace):
    step = 0
    val_step = 0
    for epoch in range(args.epochs):
        step = train_one_epoch(vae, encoder, train_loader, optimizer, device, args, epoch, step)
        val_step = validate_one_epoch(vae, encoder, val_loader, device, args, epoch, val_step)
        save_checkpoint(os.path.join(args.output_dir, f"vae_epoch{epoch + 1}.pt"), vae, optimizer, epoch + 1, step, args)

    final_path = os.path.join(args.output_dir, "vae_final.pt")
    torch.save({"model": vae.state_dict(), "args": vars(args)}, final_path)
    print(f"\nSaved final VAE model to {final_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Data settings
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the train.zip archive (see ImageNetDataset)")
    parser.add_argument("--val-data-path", type=str, required=True,
                        help="Path to the val.zip archive")
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
    parser.add_argument("--normalize", action="store_true",
                        help="Train on position-wise normalized RAE patch tokens")

    # Training settings
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="AdamW weight decay (default: 0.0)")
    parser.add_argument("--kl-weight", type=float, default=1e-6,
                        help="KL loss weight (default: 1e-6)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")

    # Data limiting
    parser.add_argument("--data-limit-sample-percentage", type=float, default=1.0,
                        help="Percentage of samples to use per kept class (0.0-1.0, default: 1.0)")
    parser.add_argument("--data-limit-class-percentage", type=float, default=1.0,
                        help="Percentage of classes to keep (0.0-1.0, default: 1.0)")
    parser.add_argument("--data-limit-seed", type=int, default=42,
                        help="Random seed for data limiting (default: 42)")

    return parser


def main():
    args = build_arg_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, transform = load_encoder(args.config, args.normalize, device)
    train_loader = build_dataset(args.data_path, transform, args, sample_percentage=args.data_limit_sample_percentage, shuffle=True)
    val_loader = build_dataset(args.val_data_path, transform, args, sample_percentage=1.0, shuffle=False)

    if isinstance(train_loader.dataset, ClassBalancedSubset):
        assert train_loader.dataset.selected_classes == val_loader.dataset.selected_classes

    os.makedirs(args.output_dir, exist_ok=True)

    vae = build_vae(args, encoder.channels, device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.wandb:
        exp_name = os.path.basename(os.path.normpath(args.output_dir))
        wandb_utils.initialize(args, "plankton-diffusion", exp_name, "mm-fm")
        if "SLURM_JOB_ID" in os.environ:
            print(f"Running Slurm job {os.environ["SLURM_JOB_ID"]}")

    train(vae, encoder, train_loader, val_loader, optimizer, device, args)


if __name__ == "__main__":
    main()
