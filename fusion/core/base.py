"""Abstract base class for all pluggable Fusion components.

Provides ``FusionComponent``, the root abstraction that all encoders,
fusion strategies, and prediction heads must inherit from.

Expected config params:
    Subclass-dependent.  Each concrete component defines its own
    configuration keys in its ``get_config()`` / ``from_config()``
    implementations.
"""

from abc import ABC, abstractmethod

from fusion.core.types import ConfigDict


class FusionComponent(ABC):
    """Base contract that every Fusion component must satisfy.

    Subclasses **must** implement:

    * ``forward(*args, **kwargs)`` — the main computation entry-point.
    * ``from_config(config)``      — a classmethod that reconstructs the
      component from a configuration dictionary.

    ``get_config()`` returns an empty ``dict`` by default; override it to
    enable round-trip serialisation via ``from_config(get_config())``.
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Execute the component's core computation.

        Args:
            *args: Positional arguments (subclass-defined).
            **kwargs: Keyword arguments (subclass-defined).

        Returns:
            Subclass-defined output.
        """
        ...

    def get_config(self) -> ConfigDict:
        """Serialise the component's configuration.

        Returns:
            ConfigDict: A dictionary representing the component's settings.
                Defaults to an empty dict — override in subclasses to
                support reconstruction via :meth:`from_config`.
        """
        return {}

    @classmethod
    @abstractmethod
    def from_config(cls, config: ConfigDict) -> "FusionComponent":
        """Construct a component instance from a configuration dict.

        Args:
            config (ConfigDict): A dictionary containing the component's
                settings as produced by :meth:`get_config`.

        Returns:
            FusionComponent: A new instance of the component.
        """
        ...
