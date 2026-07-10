"""
Input/Output utilities for the Fusion framework.

Provides helper functions for saving and loading checkpoints,
configuration files, and JSON data.
"""

import json
import os

import torch
import yaml

from fusion.exceptions import CheckpointError
from fusion.utils.config import Config