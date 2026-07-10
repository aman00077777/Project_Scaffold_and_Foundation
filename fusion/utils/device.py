"""Device detection, placement, and GPU memory utilities.

Provides helpers that abstract away CUDA availability checks and
recursive tensor/module device transfers.

Expected config params: None — all functions accept explicit arguments.
"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


def get_device(device: str = "auto") -> torch.device:
    """Return a ``torch.device`` for the requested target.

    Args:
        device (str): One of ``"auto"``, ``"cpu"``, ``"cuda"``,
            or a specific index like ``"cuda:0"``.  ``"auto"`` selects
            CUDA when available, otherwise CPU.

    Returns:
        torch.device: Resolved device object.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def move_to_device(
    obj: Any,
    device: Union[str, torch.device],
) -> Any:
    """Recursively move tensors, modules, dicts, or lists to *device*.

    Args:
        obj: A ``torch.Tensor``, ``nn.Module``, ``dict``, ``list``,
            or nested combination thereof.
        device: Target device (string or ``torch.device``).

    Returns:
        The input structure with all tensors/modules on *device*.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, nn.Module):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [move_to_device(item, device) for item in obj]
        return type(obj)(moved)
    return obj


def get_available_gpus() -> List[int]:
    """Return indices of all available CUDA GPUs.

    Returns:
        List[int]: GPU indices (empty list if CUDA is unavailable).
    """
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def is_cuda_available() -> bool:
    """Check whether CUDA is available on this system.

    Returns:
        bool: ``True`` if at least one CUDA device is detected.
    """
    return torch.cuda.is_available()


def get_memory_info(device: Union[str, int, torch.device] = 0) -> Dict[str, int]:
    """Return allocated and reserved GPU memory for *device*.

    Args:
        device: CUDA device index, string, or ``torch.device``.

    Returns:
        Dict[str, int]: Dictionary with keys ``allocated``,
            ``reserved``, ``total`` (values in bytes).
            Returns zeros for every key when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}

    if isinstance(device, torch.device):
        device = device.index or 0

    return {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "total": torch.cuda.get_device_properties(device).total_mem,
    }
