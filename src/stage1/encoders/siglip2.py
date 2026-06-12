from torch import nn
import torch
from math import *
from . import register_encoder
from transformers import SiglipModel

@register_encoder()
class SigLIP2wNorm(nn.Module):
    def __init__(self, model_name:str, num_tokens=256):
        super().__init__()
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.model = SiglipModel.from_pretrained(self.model_name).vision_model
        # remove the affine of final layernorm
        self.model.post_layernorm.elementwise_affine = False
        # remove the param
        self.model.post_layernorm.weight = None
        self.model.post_layernorm.bias = None
        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size

    @torch.no_grad() # encoder is always frozen
    def forward(self, images):
        """
        Extract spatial and pooled tokens from SigLIP2.

        Args:
            images: Input images (B, C, H, W)

        Returns:
            spatial_tokens: (B, N, C) - spatial patch tokens
            pooled_output: (B, C) - pooled features (CLS token equivalent via MAP pooling)
        """
        outputs = self.model(images, output_hidden_states=True, interpolate_pos_encoding=True)
        spatial_tokens = outputs.last_hidden_state  # (B, N, C)
        pooled_output = outputs.pooler_output  # (B, C) - MAP pooled features
        return spatial_tokens, pooled_output