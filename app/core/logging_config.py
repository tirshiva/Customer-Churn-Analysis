"""Logging configuration for application."""

from ml_pipeline.core.logging_config import setup_logging as _setup_logging


def setup_logging(log_level: str = "INFO") -> None:
    """Configure application logging."""
    _setup_logging(log_level=log_level, component="app")
