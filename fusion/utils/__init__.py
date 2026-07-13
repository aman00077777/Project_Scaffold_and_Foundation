"""Utility modules for the Fusion framework.

Provides configuration loading, device checks, downloads, registries, metrics, logging, and seeds.
"""

from fusion.utils.config import Config
from fusion.utils.device import (
    get_device,
    move_to_device,
    get_available_gpus,
    is_cuda_available,
    get_memory_info,
)
from fusion.utils.download import download_file, download_from_hub
from fusion.utils.helpers import (
    flatten_dict,
    count_parameters,
    freeze_model,
    unfreeze_model,
    seed_everything,
)
from fusion.utils.logging import get_logger
from fusion.utils.metrics import (
    compute_accuracy,
    compute_top_k_accuracy,
    compute_map,
    compute_ndcg,
)
from fusion.utils.registry import Registry

__all__ = [
    "Config",
    "get_device",
    "move_to_device",
    "get_available_gpus",
    "is_cuda_available",
    "get_memory_info",
    "download_file",
    "download_from_hub",
    "flatten_dict",
    "count_parameters",
    "freeze_model",
    "unfreeze_model",
    "seed_everything",
    "get_logger",
    "compute_accuracy",
    "compute_top_k_accuracy",
    "compute_map",
    "compute_ndcg",
    "Registry",
]
