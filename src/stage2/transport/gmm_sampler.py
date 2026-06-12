#!/usr/bin/env python3
"""
Minimalist GMM Sampler for Class-Conditional/Mode-Conditional Noise Generation.

Supports:
- Vectorized sampling from GMM components
- Mode prediction from CLS tokens (for mode_conditional training)
- Weighted mode sampling (for inference)
"""

import torch
import numpy as np
import pickle
from typing import Optional, Tuple, Union


class GMMSampler:
    """
    Vectorized GMM sampler for spatial tokens.
    Supports both class-conditional and mode-conditional training.
    """
    def __init__(
        self,
        gmm_path: str,
        cls_gmm_path_for_assignment: Optional[str] = None,
        cls_assignment_mode: str = 'soft',
        verbose: bool = True,
        validate_gpu_inference: bool = False,
    ):
        """
        Initialize GMM sampler.

        Args:
            gmm_path: Path to pickled GMM file for spatial tokens
            cls_gmm_path_for_assignment: Path to CLS GMM for mode prediction (mode-conditional only)
            cls_assignment_mode: 'soft' (multinomial) or 'hard' (argmax) for mode prediction
            verbose: Whether to print loading messages (set False for non-rank-0 processes)
            validate_gpu_inference: If True, validate GPU results against sklearn for first few calls
        """
        if cls_assignment_mode not in ['soft', 'hard']:
            raise ValueError(f"cls_assignment_mode must be 'soft' or 'hard', got '{cls_assignment_mode}'")

        self.cls_assignment_mode = cls_assignment_mode
        self.validate_gpu_inference = validate_gpu_inference
        self._validation_count = 0
        self._max_validations = 10  # Only validate first N calls

        # Load spatial GMM
        with open(gmm_path, 'rb') as f:
            gmm_data = pickle.load(f)

        # Handle both wrapped and unwrapped formats
        if isinstance(gmm_data, dict) and 'model' in gmm_data:
            self.gmm = gmm_data['model']
        else:
            self.gmm = gmm_data

        # Infer num_classes from loaded GMM
        self.num_classes = self.gmm.n_components

        # Convert to torch tensors (keep on CPU to save GPU memory)
        self.means = torch.from_numpy(self.gmm.means_)  # (num_classes, D)
        self.variances = torch.from_numpy(self.gmm.covariances_).float()  # (num_classes,)
        self.stds = torch.sqrt(self.variances)
        self.latent_dim = self.gmm.means_.shape[1]

        if verbose:
            print(f"[GMMSampler] Loaded spatial GMM from {gmm_path}")
            print(f"  Components: {self.num_classes}, Latent dim: {self.latent_dim}")
            print(f"  Variance range: [{self.variances.min():.4f}, {self.variances.max():.4f}]")

        # Load CLS GMM for mode assignment (mode-conditional only)
        self.cls_gmm_model = None
        self._cls_means = None
        self._cls_precisions = None
        self._cls_log_weights = None
        self._cls_log_det = None

        if cls_gmm_path_for_assignment is not None:
            with open(cls_gmm_path_for_assignment, 'rb') as f:
                cls_data = pickle.load(f)
            self.cls_gmm_model = cls_data['model']

            if verbose:
                print(f"[GMMSampler] Loaded CLS GMM for mode assignment")
                print(f"  CLS GMM path: {cls_gmm_path_for_assignment}")
                print(f"  CLS GMM components: {self.cls_gmm_model.n_components}")
                print(f"  CLS assignment mode: {cls_assignment_mode}")

            if self.cls_gmm_model.n_components != self.num_classes:
                raise ValueError(
                    f"Component count mismatch!\n"
                    f"  Spatial GMM: {self.num_classes} components\n"
                    f"  CLS GMM: {self.cls_gmm_model.n_components} components"
                )

            # Detect covariance type
            self._cls_cov_type = self.cls_gmm_model.covariance_type
            self._cls_dim = self.cls_gmm_model.means_.shape[1]

            # Only use GPU inference for spherical/diagonal (minimal memory overhead)
            # Fall back to sklearn CPU for full/tied covariance (too much memory)
            self._use_gpu_inference = self._cls_cov_type in ('spherical', 'diag')

            if self._use_gpu_inference:
                self._cls_means = torch.from_numpy(self.cls_gmm_model.means_).float()  # (K, D)
                self._cls_log_weights = torch.from_numpy(np.log(self.cls_gmm_model.weights_)).float()  # (K,)

                if self._cls_cov_type == 'diag':
                    # Diagonal: (K, D) per-dimension variance
                    self._cls_precisions = torch.from_numpy(
                        1.0 / self.cls_gmm_model.covariances_
                    ).float()  # (K, D)
                    self._cls_log_det = torch.from_numpy(
                        np.sum(np.log(self.cls_gmm_model.covariances_), axis=-1)
                    ).float()  # (K,)
                else:  # spherical
                    # Spherical: (K,) single variance per component
                    self._cls_precisions = torch.from_numpy(
                        1.0 / self.cls_gmm_model.covariances_
                    ).float()  # (K,)
                    self._cls_log_det = torch.from_numpy(
                        np.log(self.cls_gmm_model.covariances_) * self._cls_dim
                    ).float()  # (K,)

                if verbose:
                    print(f"  Covariance type: {self._cls_cov_type} -> using GPU inference")
            else:
                if verbose:
                    print(f"  Covariance type: {self._cls_cov_type} -> using sklearn CPU (GPU memory too large)")

    @torch.no_grad()
    def sample(
        self,
        class_labels: torch.Tensor,
        output_shape: Union[Tuple[int, ...], int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized GMM sampling (fully on GPU).

        Args:
            class_labels: (B,) tensor of class/mode indices (can include null label = num_classes)
            output_shape: (C, H, W) for spatial tokens
            device: torch device
            dtype: torch dtype

        Returns:
            noise: (B, C, H, W) sampled noise
            per_sample_variance: (B,) variance used for each sample
        """
        batch_size = class_labels.shape[0]

        # Cache spatial GMM tensors on GPU (lazy initialization)
        if not hasattr(self, '_means_gpu') or self._means_gpu.device != device:
            self._means_gpu = self.means.to(device=device, dtype=dtype)
            self._stds_gpu = self.stds.to(device=device, dtype=dtype)
            self._variances_gpu = self.variances.to(device=device, dtype=dtype)

        # Handle null labels (class_idx == num_classes should get N(0,I))
        mask_valid = class_labels < self.num_classes
        class_labels_clamped = torch.clamp(class_labels, 0, self.num_classes - 1)

        # GPU indexing (no CPU transfer)
        selected_means = self._means_gpu[class_labels_clamped]  # (B, D)
        selected_stds = self._stds_gpu[class_labels_clamped]    # (B,)

        # Sample: N(mu_c, sigma_c^2 * I)
        isotropic_noise = torch.randn(batch_size, self.latent_dim, device=device, dtype=dtype)
        gmm_noise = selected_means + selected_stds.unsqueeze(-1) * isotropic_noise

        # Replace null labels with N(0, I)
        null_noise = torch.randn(batch_size, self.latent_dim, device=device, dtype=dtype)
        noise_flat = torch.where(mask_valid.unsqueeze(-1), gmm_noise, null_noise)

        # Reshape to output shape
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        noise = noise_flat.view(batch_size, *output_shape)

        # Track per-sample variance (GPU indexing)
        selected_variances = self._variances_gpu[class_labels_clamped]
        per_sample_variance = torch.where(
            mask_valid,
            selected_variances,
            torch.ones(batch_size, device=device, dtype=dtype)
        )

        return noise, per_sample_variance

    @torch.no_grad()
    def _gpu_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities for all GMM components on GPU.
        Supports spherical and diagonal covariance types.

        Args:
            x: Input tokens (B, D)

        Returns:
            log_prob: Log probabilities (B, K)
        """
        device = x.device
        dtype = x.dtype

        # Move pre-computed tensors to GPU (cached after first call)
        if not hasattr(self, '_cls_means_gpu') or self._cls_means_gpu.device != device:
            self._cls_means_gpu = self._cls_means.to(device=device, dtype=dtype)
            self._cls_precisions_gpu = self._cls_precisions.to(device=device, dtype=dtype)
            self._cls_log_weights_gpu = self._cls_log_weights.to(device=device, dtype=dtype)
            self._cls_log_det_gpu = self._cls_log_det.to(device=device, dtype=dtype)

        # Compute Mahalanobis distance
        # diff: (B, 1, D) - (1, K, D) = (B, K, D)
        diff = x.unsqueeze(1) - self._cls_means_gpu.unsqueeze(0)

        if self._cls_cov_type == 'spherical':
            # Spherical: precision is (K,) single scalar per component
            # Mahalanobis: ||x - mu||^2 / sigma^2
            mahal = (diff ** 2).sum(dim=-1) * self._cls_precisions_gpu.unsqueeze(0)  # (B, K)
        else:  # 'diag'
            # Diagonal: precision is (K, D) per-dimension
            # Mahalanobis: sum_i (x_i - mu_i)^2 / sigma_i^2
            mahal = ((diff ** 2) * self._cls_precisions_gpu.unsqueeze(0)).sum(dim=-1)  # (B, K)

        # Log probability: log(weight) - 0.5 * (D * log(2π) + log|Σ| + mahal)
        log_prob = (
            self._cls_log_weights_gpu
            - 0.5 * (self._cls_dim * np.log(2 * np.pi) + self._cls_log_det_gpu + mahal)
        )

        return log_prob

    @torch.no_grad()
    def _sklearn_predict_proba(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        """
        Fallback to sklearn CPU inference for full/tied covariance.

        Args:
            cls_tokens: CLS tokens (B, D)

        Returns:
            mode_probs: Mode probabilities (B, K) on CPU
        """
        cls_tokens_np = cls_tokens.cpu().numpy()
        mode_probs_np = self.cls_gmm_model.predict_proba(cls_tokens_np)
        return torch.from_numpy(mode_probs_np).float()

    @torch.no_grad()
    def predict_and_sample_mode(
        self,
        cls_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict cluster modes from CLS tokens.

        Uses GPU-accelerated inference for spherical/diagonal covariance,
        falls back to sklearn CPU for full/tied covariance.

        Args:
            cls_tokens: CLS tokens (B, C) where C=768 for DINOv2-base

        Returns:
            mode_indices: Sampled mode indices (B,) on same device as cls_tokens
            mode_probs: Mode probabilities (B, K) on CPU (memory efficient)
        """
        if self.cls_gmm_model is None:
            raise RuntimeError(
                "predict_and_sample_mode() requires CLS GMM. "
                "Initialize with cls_gmm_path_for_assignment parameter."
            )

        if cls_tokens.dim() != 2:
            raise ValueError(f"CLS tokens must be 2D (B, C), got shape {cls_tokens.shape}")

        if self._use_gpu_inference:
            # GPU path for spherical/diagonal covariance
            log_prob = self._gpu_log_prob(cls_tokens)  # (B, K)
            log_prob_normalized = log_prob - log_prob.logsumexp(dim=-1, keepdim=True)
            mode_probs = log_prob_normalized.exp()

            # Optional validation against sklearn
            if self.validate_gpu_inference and self._validation_count < self._max_validations:
                self._validation_count += 1
                sklearn_probs = self._sklearn_predict_proba(cls_tokens)
                max_diff = (mode_probs.cpu() - sklearn_probs).abs().max().item()
                # Threshold 1e-3: differences up to 0.1% are fine (float32 vs float64 precision)
                if max_diff > 1e-3:
                    print(f"[GMMSampler] WARNING: GPU vs sklearn max prob diff = {max_diff:.6f}")
                elif self._validation_count == 1:
                    print(f"[GMMSampler] GPU validation passed (max diff = {max_diff:.2e})")

            if self.cls_assignment_mode == 'hard':
                mode_indices = log_prob.argmax(dim=-1)
            else:  # 'soft'
                mode_indices = torch.multinomial(mode_probs, num_samples=1).squeeze(1)

            mode_probs_cpu = mode_probs.cpu()
        else:
            # Fallback to sklearn CPU for full/tied covariance
            mode_probs_cpu = self._sklearn_predict_proba(cls_tokens)

            if self.cls_assignment_mode == 'hard':
                mode_indices = mode_probs_cpu.argmax(dim=-1).to(cls_tokens.device)
            else:  # 'soft'
                mode_indices = torch.multinomial(mode_probs_cpu, num_samples=1).squeeze(1)
                mode_indices = mode_indices.to(cls_tokens.device)

        return mode_indices, mode_probs_cpu

    @torch.no_grad()
    def sample_modes_weighted(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample mode indices weighted by GMM component weights (for inference).

        Args:
            batch_size: Number of modes to sample
            device: Device for output tensor
            generator: Optional random generator for reproducibility

        Returns:
            mode_indices: Sampled mode indices (batch_size,)
        """
        weights = torch.from_numpy(self.gmm.weights_).to(device)
        mode_indices = torch.multinomial(
            weights,
            num_samples=batch_size,
            replacement=True,
            generator=generator
        )
        return mode_indices

    @classmethod
    def from_config(cls, gmm_config: dict):
        """
        Factory method to create from config dict.

        Args:
            gmm_config: The gmm config section

        Returns:
            GMMSampler instance or None if disabled
        """
        if not gmm_config.get('enabled', False):
            return None

        gmm_path = gmm_config['path']
        cls_gmm_path = gmm_config.get('cls_path') if gmm_config.get('cls_enabled', False) else None

        return cls(
            gmm_path=gmm_path,
            cls_gmm_path_for_assignment=cls_gmm_path,
            cls_assignment_mode=gmm_config.get('cls_assignment_mode', 'soft'),
        )
