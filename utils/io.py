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