"""
Helper utilities for the Fusion framework.
"""

import random

import numpy as np
import torch
import torch.nn as nn

def flatten_dict(
    data: dict,
    parent_key: str = "",
    sep: str = "."
) -> dict:
    """
    Flatten a nested dictionary using dot notation.

    Args:
        data: Dictionary to flatten.
        parent_key: Parent key for recursion.
        sep: Separator between nested keys.

    Returns:
        Flattened dictionary.
    """
    items = {}

    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep))
        else:
            items[new_key] = value

    return items

def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Total number of trainable parameters.
    """
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

def freeze_model(model: nn.Module) -> None:
    """
    Freeze all model parameters.

    Args:
        model: PyTorch model.
    """
    for param in model.parameters():
        param.requires_grad_(False)

def unfreeze_model(model: nn.Module) -> None:
    """
    Unfreeze all model parameters.

    Args:
        model: PyTorch model.
    """
    for param in model.parameters():
        param.requires_grad_(True)

def seed_everything(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)