import importlib
from dataclasses import dataclass
from typing import Union, Tuple, Optional
from stage1 import RAE
import torch.nn as nn
from omegaconf import OmegaConf
import torch

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    model = get_obj_from_str(config["target"])(**config.get("params", dict()))
    ckpt_path = config.get("ckpt", None)
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # see if it's a ckpt from training by checking for "model"
        if "ema" in state_dict:
            state_dict = state_dict["ema"]
        elif "model" in state_dict:
            raise NotImplementedError("Loading from 'model' key not implemented yet.")
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        print(f'target {config["target"]} loaded from {ckpt_path}')
    return model


def get_model_attr(model, attr, default=None):
    """
    Get attribute from model, handling DDP wrapping.

    Args:
        model: Model instance (possibly DDP-wrapped)
        attr: Attribute name
        default: Default value if attribute doesn't exist

    Returns:
        Attribute value
    """
    model_module = model.module if hasattr(model, 'module') else model
    return getattr(model_module, attr, default)


def is_model_conditional(model):
    """
    Check if model uses y conditioning (class or mode labels).

    Args:
        model: Model instance (possibly DDP-wrapped)

    Returns:
        True if model uses y conditioning, False otherwise
    """
    return get_model_attr(model, 'use_y_conditioning', True)

