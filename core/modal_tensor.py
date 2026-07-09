"""Lightweight tensor wrapper that binds data to its modality metadata.

Provides ``ModalTensor``, a dataclass that pairs a ``torch.Tensor`` with
a ``Modality`` tag, optional embedding / mask tensors, and an arbitrary
metadata dict.  All tensor fields support ``.to(device)`` and
``.detach()`` propagation.

Expected config params: None — this is a pure data container.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from fusion.constants import Modality


@dataclass
class ModalTensor:
    """A tensor annotated with modality information.

    Args:
        data (torch.Tensor): The raw tensor payload.
        modality (Modality): Modality enum tag (e.g. ``Modality.VISION``).
        embedding (Optional[torch.Tensor]): Pre-computed embedding for
            this modality.
        mask (Optional[torch.Tensor]): Attention / padding mask aligned
            with *data*.
        metadata (Dict[str, Any]): Arbitrary extra information
            (sample id, timestamp, etc.).
    """

    data: torch.Tensor
    modality: Modality
    embedding: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Device / gradient helpers
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "ModalTensor":
        """Return a copy of this ``ModalTensor`` moved to *device*.

        Args:
            device (torch.device): Target device (e.g. ``"cuda:0"``).

        Returns:
            ModalTensor: New instance with all tensor fields on *device*.
        """
        return ModalTensor(
            data=self.data.to(device),
            modality=self.modality,
            embedding=self.embedding.to(device) if self.embedding is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None,
            metadata=self.metadata,
        )

    def detach(self) -> "ModalTensor":
        """Return a copy with all tensor fields detached from the graph.

        Returns:
            ModalTensor: New instance with detached tensors.
        """
        return ModalTensor(
            data=self.data.detach(),
            modality=self.modality,
            embedding=self.embedding.detach() if self.embedding is not None else None,
            mask=self.mask.detach() if self.mask is not None else None,
            metadata=self.metadata,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> torch.Size:
        """Shape of the underlying *data* tensor.

        Returns:
            torch.Size: The shape of ``self.data``.
        """
        return self.data.shape

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ModalTensor(modality={self.modality.value!r}, "
            f"shape={self.shape}, "
            f"has_embedding={self.embedding is not None}, "
            f"has_mask={self.mask is not None})"
        )
