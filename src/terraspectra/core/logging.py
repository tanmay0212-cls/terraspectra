"""
Structured logging configuration for TerraSpectra.
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "terraspectra",
    level: str = "INFO",
    log_file: Optional[Path | str] = None
) -> logging.Logger:
    """
    Configure and return a structured logger with console (and optional file) handlers.
    
    Args:
        name: Logger hierarchy name.
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional file path for file logging.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console stdout handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(p), encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
