"""Schema-level validation for multimodal pipeline inputs.

Provides ``ModalitySchema``, a declarative dataclass that describes the
expected modality contract (required / optional modalities, tensor
shapes, task type) and can validate an incoming dictionary of tensors
against that contract at runtime.

Expected config params:
    input_modalities : List[Modality]
        Modalities the pipeline accepts.
    output_type : TaskType
        The downstream task this schema is designed for.
    expected_shapes : Dict[Modality, Tuple]
        Per-modality shape constraints (excluding batch dimension).
        Use an empty tuple ``()`` for scalar modalities.
    optional_modalities : List[Modality]
        Subset of *input_modalities* that may be omitted at inference.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from fusion.constants import Modality, TaskType
from fusion.exceptions import ModalityError


@dataclass
class ModalitySchema:
    """Declarative schema describing a multimodal pipeline's I/O contract.

    Args:
        input_modalities (List[Modality]): All modalities the component
            can accept.
        output_type (TaskType): The downstream task category.
        expected_shapes (Dict[Modality, Tuple]): Per-modality shape
            constraints **excluding** the batch dimension.  A ``-1`` in
            any position acts as a wildcard (matches any size).
        optional_modalities (List[Modality]): Modalities that may be
            absent at runtime without raising an error.
    """

    input_modalities: List[Modality]
    output_type: TaskType
    expected_shapes: Dict[Modality, Tuple]
    optional_modalities: List[Modality] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, inputs: Dict[Modality, Any]) -> bool:
        """Validate *inputs* against this schema.

        Args:
            inputs (Dict[Modality, Any]): Mapping of ``Modality`` members
                to tensors (or tensor-like objects with a ``.shape``
                attribute).

        Returns:
            bool: ``True`` if all checks pass.

        Raises:
            ModalityError: If a required modality is missing **or** a
                tensor's shape (excluding batch dim) does not conform to
                ``expected_shapes``.
        """
        # --- required-modality presence check ---
        required = [
            m for m in self.input_modalities
            if m not in self.optional_modalities
        ]
        for modality in required:
            if modality not in inputs:
                raise ModalityError(
                    f"Required modality '{modality.value}' is missing from "
                    f"inputs. Provided: {[m.value for m in inputs.keys()]}",
                    details={
                        "missing": modality.value,
                        "provided": [m.value for m in inputs.keys()],
                    },
                )

        # --- per-modality shape conformance (excluding batch dim) ---
        for modality, tensor in inputs.items():
            if modality not in self.expected_shapes:
                continue

            expected = self.expected_shapes[modality]
            # Exclude the leading batch dimension for comparison.
            actual = tuple(tensor.shape[1:])

            if len(expected) != len(actual):
                raise ModalityError(
                    f"Shape mismatch for modality '{modality.value}': "
                    f"expected {len(expected)}D (shape {expected}) excluding "
                    f"batch, got {len(actual)}D (shape {actual})",
                    details={
                        "modality": modality.value,
                        "expected_shape": expected,
                        "actual_shape": actual,
                    },
                )

            for dim_idx, (exp_dim, act_dim) in enumerate(zip(expected, actual)):
                if exp_dim != -1 and exp_dim != act_dim:
                    raise ModalityError(
                        f"Shape mismatch for modality '{modality.value}' at "
                        f"dim {dim_idx + 1} (excluding batch): expected "
                        f"{expected}, got {actual}",
                        details={
                            "modality": modality.value,
                            "dim": dim_idx + 1,
                            "expected_shape": expected,
                            "actual_shape": actual,
                        },
                    )

        return True

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Return a multi-line, human-readable summary of this schema.

        Returns:
            str: Formatted description of required/optional modalities
                and the output task type.
        """
        lines = [
            "ModalitySchema Summary:",
            f"  Output Type: {self.output_type.value}",
        ]

        required = [
            m for m in self.input_modalities
            if m not in self.optional_modalities
        ]
        if required:
            lines.append("  Required Modalities:")
            for mod in required:
                shape_info = self.expected_shapes.get(mod, "any")
                lines.append(f"    - {mod.value} (expected shape: {shape_info})")

        if self.optional_modalities:
            lines.append("  Optional Modalities:")
            for mod in self.optional_modalities:
                shape_info = self.expected_shapes.get(mod, "any")
                lines.append(f"    - {mod.value} (expected shape: {shape_info})")

        return "\n".join(lines)
