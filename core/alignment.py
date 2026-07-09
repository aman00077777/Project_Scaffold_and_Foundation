"""Per-modality linear projections into a shared embedding space.

Provides ``ModalityAligner``, an ``nn.Module`` that projects heterogeneous
modality embeddings to a common dimensionality for downstream fusion.

Expected config params:
    input_dims : Dict[Modality, int]
        Mapping of each modality to its native embedding dimension.
    target_dim : int
        The shared output dimensionality all modalities will be projected to.
"""

from typing import Dict

import torch
import torch.nn as nn

from fusion.constants import Modality
from fusion.core.modal_tensor import ModalTensor


class ModalityAligner(nn.Module):
    """Project heterogeneous modality embeddings to a common dimension.

    Each modality gets its own ``nn.Linear`` projection, stored in an
    ``nn.ModuleDict`` (keyed by the modality's string value) so that all
    parameters are properly registered for optimiser discovery and device
    placement.
    """

    def __init__(self, input_dims: Dict[Modality, int], target_dim: int) -> None:
        """Initialise per-modality linear projections.

        Args:
            input_dims (Dict[Modality, int]): Mapping of each ``Modality``
                enum member to its native embedding dimensionality.
            target_dim (int): The shared output dimensionality that every
                modality will be projected to.
        """
        super().__init__()
        self.target_dim = target_dim
        self.projections = nn.ModuleDict(
            {
                modality.value: nn.Linear(input_dim, target_dim)
                for modality, input_dim in input_dims.items()
            }
        )

    def forward(
        self, embeddings: Dict[Modality, torch.Tensor]
    ) -> Dict[Modality, torch.Tensor]:
        """Project each modality embedding to the common space.

        Args:
            embeddings (Dict[Modality, torch.Tensor]): Mapping of ``Modality``
                enum members to their embedding tensors, each of shape
                ``(batch, input_dim)``.

        Returns:
            Dict[Modality, torch.Tensor]: Dictionary with the same ``Modality``
                keys, values projected to shape ``(batch, target_dim)``.

        Raises:
            KeyError: If a modality in *embeddings* has no registered
                projection layer.
        """
        aligned: Dict[Modality, torch.Tensor] = {}

        for modality, tensor in embeddings.items():
            key = modality.value
            if key not in self.projections:
                raise KeyError(
                    f"No projection registered for modality '{key}'. "
                    f"Known modalities: {list(self.projections.keys())}"
                )
            projected = self.projections[key](tensor)
            print(
                f"[ModalityAligner] {modality.value}: "
                f"{tensor.shape} -> {projected.shape}"
            )
            aligned[modality] = projected

        return aligned

    def get_output_dim(self) -> int:
        """Return the common output dimensionality.

        Returns:
            int: The target dimension all modality projections map to.
        """
        return self.target_dim
