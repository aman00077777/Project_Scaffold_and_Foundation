"""Core abstractions and representations for the Fusion framework.

Contains basic components, datatype schemas, alignment wrappers, and typing aliases.
"""

from fusion.core.base import FusionComponent
from fusion.core.modal_tensor import ModalTensor
from fusion.core.alignment import ModalityAligner
from fusion.core.schema import ModalitySchema
from fusion.core.types import (
    TensorDict,
    ModalityDict,
    EmbeddingDict,
    ConfigDict,
)

__all__ = [
    "FusionComponent",
    "ModalTensor",
    "ModalityAligner",
    "ModalitySchema",
    "TensorDict",
    "ModalityDict",
    "EmbeddingDict",
    "ConfigDict",
]
