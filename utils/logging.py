"""
Logging utilities for the Fusion framework.

Provides helper functions to configure and retrieve loggers.
"""

import logging
from typing import Optional


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name: Name of the logger.
        level: Logging level (e.g. INFO, DEBUG).

    Returns:
        Configured Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper()))
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)
    logger.propagate = False

    return logger


def setup_logging(log_file: Optional[str] = None,
                  level: str = "INFO") -> None:
    """
    Configure application-wide logging.

    Args:
        log_file: Optional log file path.
        level: Logging level.
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )