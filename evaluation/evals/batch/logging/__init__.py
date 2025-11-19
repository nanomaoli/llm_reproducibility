"""Logging."""

import logging
from typing import Optional

from ray._private.ray_logging.filters import CoreContextFilter
from ray._private.ray_logging.formatters import JSONFormatter


def _add_ray_logging(handler: logging.Handler):
    """Add Ray logging to the handler.

    This is not used for now and will be enabled after the Ray Job is supported.

    Args:
        handler: The handler to add Ray logging to.
    """
    handler.addFilter(CoreContextFilter())
    handler.setFormatter(JSONFormatter())


def _setup_logger(logger_name: str):
    """Setup logger given the logger name.

    This function is idempotent and won't set up the same logger multiple times.

    Args:
        logger_name: The name of the logger.
    """
    logger = logging.getLogger(logger_name)

    # Skip setup if the logger already has handlers setup.
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a structured logger.

    Loggers by default are logging to stdout, and are expected to be scraped by an
    external process.

    Args:
        name: The name of the logger.

    Returns:
        A logger instance.
    """
    _setup_logger(name)
    return logging.getLogger(name)
