"""Generic component registry for plugin-style discovery.

Provides ``Registry``, a generic name→class/function mapping that any
phase can instantiate to create domain-specific registries
(``MODEL_REGISTRY``, ``HEAD_REGISTRY``, etc.) without modality- or
model-specific logic.

Expected config params: None — this module defines only the registry
mechanism.
"""

from typing import Any, Callable, Dict, List, Optional, Type

from fusion.exceptions import ConfigError


class Registry:
    """A generic name → component mapping with decorator registration.

    Example::

        STRATEGY_REGISTRY = Registry("strategy")

        @STRATEGY_REGISTRY.register("early")
        class EarlyFusion:
            ...

        strategy = STRATEGY_REGISTRY.build("early", dim=256)

    Args:
        name (str): Human-readable name for this registry
            (used in error messages).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str) -> Callable:
        """Decorator factory that registers a class or function.

        Args:
            name (str): Unique key to register the component under.

        Returns:
            Callable: The original class/function, unmodified.

        Raises:
            ConfigError: If *name* is already registered.
        """

        def decorator(cls_or_fn: Any) -> Any:
            if name in self._registry:
                raise ConfigError(
                    f"'{name}' is already registered in the "
                    f"'{self._name}' registry.",
                    details={
                        "registry": self._name,
                        "duplicate": name,
                        "existing": str(self._registry[name]),
                    },
                )
            self._registry[name] = cls_or_fn
            return cls_or_fn

        return decorator

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Any:
        """Return the component registered under *name*.

        Args:
            name (str): Registered key.

        Returns:
            Any: The registered class or function.

        Raises:
            ConfigError: If *name* is not found.
        """
        if name not in self._registry:
            raise ConfigError(
                f"'{name}' is not registered in the '{self._name}' "
                f"registry. Available: {self.list()}",
                details={
                    "registry": self._name,
                    "missing": name,
                    "available": self.list(),
                },
            )
        return self._registry[name]

    def list(self) -> List[str]:
        """Return all registered component names.

        Returns:
            List[str]: Sorted list of registered keys.
        """
        return sorted(self._registry.keys())

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def build(self, name: str, **kwargs: Any) -> Any:
        """Look up *name* and instantiate / call it with *kwargs*.

        Args:
            name (str): Registered key.
            **kwargs: Arguments forwarded to the component's constructor
                or call signature.

        Returns:
            Any: The instantiated component.

        Raises:
            ConfigError: If *name* is not found.
        """
        cls_or_fn = self.get(name)
        return cls_or_fn(**kwargs)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return (
            f"Registry(name={self._name!r}, "
            f"components={self.list()})"
        )
