from torch import nn
import torch
from math import *
from . import register_encoder
from transformers import ViTMAEForPreTraining

@register_encoder()
class MAEwNorm(nn.Module):
    def __init__(self, model_name:str):
        super().__init__()
        self.model_name = model_name
        self.model = ViTMAEForPreTraining.from_pretrained(self.model_name).vit
        # remove the affine of final layernorm
        self.model.layernorm.elementwise_affine = False
        # remove the param
        self.model.layernorm.weight = None
        self.model.layernorm.bias = None
        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.model.config.mask_ratio = 0. # no masking
    def forward(self, images):
        """
        Extract spatial and CLS tokens from MAE.

        Args:
            images: Input images of shape (B, C, H, W)

        Returns:
            spatial_tokens: (B, N, C) - spatial patch tokens
            cls_token: (B, C) - global CLS token
        """
        h,w = images.shape[2], images.shape[3]
        patch_num = int(h * w  // self.patch_size ** 2)
        assert patch_num * self.patch_size ** 2 == h * w, 'image size should be divisible by patch size'
        noise = torch.arange(patch_num).unsqueeze(0).expand(images.shape[0],-1).to(images.device).to(images.dtype)
        outputs = self.model(images, noise, interpolate_pos_encoding = True)
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token at index 0
        spatial_tokens = outputs.last_hidden_state[:, 1:]  # spatial tokens
        return spatial_tokens, cls_token