from transformers import Dinov2WithRegistersModel
from torch import nn
import torch
from math import *
from . import register_encoder


@register_encoder()
class Dinov2withNorm(nn.Module):
    def __init__(
        self,
        dinov2_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=False)
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        
    def dinov2_forward(self, x: torch.Tensor):
        """
        Extract spatial and CLS tokens from DINOv2.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            spatial_tokens: (B, N, C) - spatial patch tokens
            cls_token: (B, C) - global CLS token
        """
        x = self.encoder(x, output_hidden_states=True)

        # DINOv2 with registers: [CLS, reg_1, reg_2, reg_3, reg_4, patch_1, ..., patch_N]
        cls_token = x.last_hidden_state[:, 0]  # Extract CLS token
        unused_register_num = 4  # 4 register tokens
        spatial_tokens = x.last_hidden_state[:, 1+unused_register_num:]  # Spatial tokens

        return spatial_tokens, cls_token

    def forward(self, x: torch.Tensor):
        return self.dinov2_forward(x)
