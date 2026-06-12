#!/usr/bin/env python3
"""
Adaptive class/mode-level reweighting for flow matching training.
Tracks per-class loss and reweights to balance learning across classes/modes.
"""

import torch
from typing import Literal


class AdaptiveClassWeighter:
    """
    Tracks per-class/mode loss statistics and computes adaptive sample weights
    to balance learning.

    Key features:
    - EMA tracking of per-class loss
    - Multiple reweighting strategies (focal, inverse, sqrt, log)
    - Configurable warmup and weight bounds
    """

    def __init__(
        self,
        num_classes: int = 1000,
        ema_decay: float = 0.99,
        warmup_steps: int = 5000,
        min_weight: float = 0.5,
        max_weight: float = 2.0,
        strategy: Literal['focal', 'inverse', 'sqrt', 'log'] = 'focal',
        focal_gamma: float = 2.0,
        normalize: bool = True,
    ):
        """
        Args:
            num_classes: Number of classes/modes
            ema_decay: Decay factor for EMA (0.99 = slow adaptation)
            warmup_steps: Steps before starting reweighting
            min_weight: Minimum sample weight
            max_weight: Maximum sample weight
            strategy: Reweighting strategy
            focal_gamma: Exponent for focal loss style weighting
            normalize: If True, normalize weights to mean=1.0
        """
        self.num_classes = num_classes
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.strategy = strategy
        self.focal_gamma = focal_gamma
        self.normalize = normalize

        # Per-class loss tracking
        self.class_loss_ema = torch.ones(num_classes)
        self.class_counts = torch.zeros(num_classes, dtype=torch.long)
        self.global_step = 0

        # For initialization
        self.warmup_loss_sum = torch.zeros(num_classes)
        self.warmup_sample_count = torch.zeros(num_classes, dtype=torch.long)

    def update(
        self,
        class_labels: torch.Tensor,
        per_sample_loss: torch.Tensor,
    ) -> None:
        """
        Update per-class loss statistics using vectorized operations.

        Args:
            class_labels: (B,) tensor of class/mode indices
            per_sample_loss: (B,) tensor of unreduced loss per sample
        """
        self.global_step += 1

        class_labels_cpu = class_labels.cpu().long()
        per_sample_loss_cpu = per_sample_loss.detach().cpu().float()

        if self.global_step <= self.warmup_steps:
            # Vectorized accumulation using scatter_add
            self.warmup_loss_sum.scatter_add_(0, class_labels_cpu, per_sample_loss_cpu)
            self.warmup_sample_count.scatter_add_(0, class_labels_cpu, torch.ones_like(class_labels_cpu))

            if self.global_step == self.warmup_steps:
                self._initialize_ema()
        else:
            # Vectorized EMA update using scatter operations
            # Compute per-class sum and count
            class_loss_sum = torch.zeros(self.num_classes)
            class_count = torch.zeros(self.num_classes)
            class_loss_sum.scatter_add_(0, class_labels_cpu, per_sample_loss_cpu)
            class_count.scatter_add_(0, class_labels_cpu, torch.ones(len(class_labels_cpu)))

            # Update EMA only for classes that appeared in this batch
            appeared_mask = class_count > 0
            class_loss_mean = torch.where(
                appeared_mask,
                class_loss_sum / class_count.clamp(min=1),
                self.class_loss_ema  # Keep old value for classes not in batch
            )

            # EMA update: new = decay * old + (1-decay) * new_mean
            self.class_loss_ema = torch.where(
                appeared_mask,
                self.ema_decay * self.class_loss_ema + (1 - self.ema_decay) * class_loss_mean,
                self.class_loss_ema
            )

            # Update class counts (vectorized)
            self.class_counts += class_count.long()

    def _initialize_ema(self) -> None:
        """Initialize EMA from warmup statistics (vectorized)."""
        # Vectorized initialization
        valid_mask = self.warmup_sample_count > 0
        self.class_loss_ema = torch.where(
            valid_mask,
            self.warmup_loss_sum / self.warmup_sample_count.float().clamp(min=1),
            self.class_loss_ema
        )
        num_initialized = valid_mask.sum().item()

        print(f"[AdaptiveWeighter] Initialized after {self.warmup_steps} steps")
        print(f"  Classes initialized: {num_initialized}/{self.num_classes}")
        print(f"  Mean loss: {self.class_loss_ema.mean():.4f}")

    def get_weights(
        self,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute adaptive weights for current batch (GPU-optimized).

        Args:
            class_labels: (B,) tensor of class/mode indices

        Returns:
            weights: (B,) tensor of sample weights
        """
        batch_size = class_labels.shape[0]
        device = class_labels.device

        if self.global_step <= self.warmup_steps:
            return torch.ones(batch_size, device=device)

        mean_loss = self.class_loss_ema.mean()

        if self.strategy == 'focal':
            class_weights = (self.class_loss_ema / (mean_loss + 1e-8)) ** self.focal_gamma
        elif self.strategy == 'inverse':
            class_weights = self.class_loss_ema / (mean_loss + 1e-8)
        elif self.strategy == 'sqrt':
            class_weights = torch.sqrt(self.class_loss_ema / (mean_loss + 1e-8))
        elif self.strategy == 'log':
            class_weights = torch.log(1 + self.class_loss_ema) / (torch.log(1 + mean_loss) + 1e-8)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        if self.normalize:
            class_weights = class_weights / (class_weights.mean() + 1e-8)

        class_weights = torch.clamp(class_weights, self.min_weight, self.max_weight)

        # Transfer weights to GPU once, then index with GPU labels (no CPU round-trip)
        class_weights_gpu = class_weights.to(device)
        weights = class_weights_gpu[class_labels]

        return weights

    def get_statistics(self) -> dict:
        """Get current reweighting statistics for logging."""
        return {
            'mean_loss': self.class_loss_ema.mean().item(),
            'std_loss': self.class_loss_ema.std().item(),
            'min_loss': self.class_loss_ema.min().item(),
            'max_loss': self.class_loss_ema.max().item(),
            'hardest_classes': torch.argsort(self.class_loss_ema, descending=True)[:10].tolist(),
            'easiest_classes': torch.argsort(self.class_loss_ema)[:10].tolist(),
            'class_loss_ema': self.class_loss_ema.cpu().numpy(),
        }

    def save_state(self, path: str) -> None:
        """Save weighter state for checkpoint."""
        state = {
            'class_loss_ema': self.class_loss_ema,
            'class_counts': self.class_counts,
            'global_step': self.global_step,
            'warmup_loss_sum': self.warmup_loss_sum,
            'warmup_sample_count': self.warmup_sample_count,
        }
        torch.save(state, path)

    def load_state(self, path: str) -> None:
        """Load weighter state from checkpoint."""
        state = torch.load(path)
        self.class_loss_ema = state['class_loss_ema']
        self.class_counts = state['class_counts']
        self.global_step = state['global_step']
        self.warmup_loss_sum = state['warmup_loss_sum']
        self.warmup_sample_count = state['warmup_sample_count']

    @classmethod
    def from_config(cls, weighting_config: dict, num_classes: int):
        """
        Factory method to create from config dict.

        Args:
            weighting_config: The adaptive_weighting config section
            num_classes: Number of classes/modes

        Returns:
            AdaptiveClassWeighter instance or None if disabled
        """
        if not weighting_config.get('enabled', False):
            return None

        return cls(
            num_classes=num_classes,
            ema_decay=weighting_config.get('ema_decay', 0.99),
            warmup_steps=weighting_config.get('warmup_steps', 5000),
            min_weight=weighting_config.get('min_weight', 0.5),
            max_weight=weighting_config.get('max_weight', 2.0),
            strategy=weighting_config.get('strategy', 'focal'),
            focal_gamma=weighting_config.get('focal_gamma', 2.0),
            normalize=weighting_config.get('normalize', True),
        )
