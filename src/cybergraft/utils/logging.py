"""Consistent console and experiment-file logging."""

import logging
from pathlib import Path


def configure_logging(log_path: Path, verbose: bool = False) -> logging.Logger:
    """Configure a non-propagating CyberGraft logger for one run."""
    logger = logging.getLogger("cybergraft")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
