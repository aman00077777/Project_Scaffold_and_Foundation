"""Fusion — A modular multimodal fusion framework for deep learning.

Provides core abstractions, schemas, alignment tools, registry structures,
and utilities for multimodal representation learning across vision, language,
and audio.
"""

from fusion.constants import (
    Modality,
    FusionType,
    TaskType,
    DataSplit,
    PrecisionType,
    SUPPORTED_MODALITIES,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_AUDIO_SAMPLE_RATE,
)
from fusion.exceptions import (
    FusionError,
    ModalityError,
    EncoderError,
    FusionStrategyError,
    ConfigError,
    CheckpointError,
    DataError,
    DeploymentError,
    SearchError,
)
from fusion.core.base import FusionComponent
from fusion.core.modal_tensor import ModalTensor
from fusion.core.alignment import ModalityAligner
from fusion.core.schema import ModalitySchema
from fusion.utils.config import Config

__all__ = [
    # Constants
    "Modality",
    "FusionType",
    "TaskType",
    "DataSplit",
    "PrecisionType",
    "SUPPORTED_MODALITIES",
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_AUDIO_SAMPLE_RATE",
    # Exceptions
    "FusionError",
    "ModalityError",
    "EncoderError",
    "FusionStrategyError",
    "ConfigError",
    "CheckpointError",
    "DataError",
    "DeploymentError",
    "SearchError",
    # Core Components
    "FusionComponent",
    "ModalTensor",
    "ModalityAligner",
    "ModalitySchema",
    # Config
    "Config",
]
