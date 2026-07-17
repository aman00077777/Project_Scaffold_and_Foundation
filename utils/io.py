"""
Input/Output utilities for the Fusion framework.

Provides helper functions for saving and loading checkpoints,
configuration files, and JSON data.
"""

import json
import os

import torch
import yaml

from exceptions import CheckpointError
from utils.config import Config

def save_checkpoint(state: dict, path: str) -> None:
    """
    Save a model checkpoint.

    Args:
        state: Checkpoint dictionary.
        path: Destination file path.
    """
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, path)
    
    torch.save(state, path)

def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    """
    Load a model checkpoint.

    Args:
        path: Path to the checkpoint file.
        map_location: Device to load the checkpoint onto.

    Returns:
        Loaded checkpoint dictionary.

    Raises:
        CheckpointError: If the checkpoint file does not exist.
    """
    if not os.path.exists(path):
        raise CheckpointError(f"Checkpoint file not found: {path}")

    return torch.load(path, map_location=map_location)

def save_config(config: Config, path: str) -> None:
    """
    Save a configuration object as a YAML file.

    Args:
        config: Configuration object.
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(config.to_dict(), file)

def load_json(path: str) -> dict:
    """
    Load data from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Dictionary containing the JSON data.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
    
def save_json(data: dict, path: str) -> None:
    """
    Save a dictionary as a JSON file.

    Args:
        data: Dictionary to save.
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)