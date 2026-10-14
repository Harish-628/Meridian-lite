"""
Logging configuration for MERIDIAN-Lite.
Configures dual logging to a file in outputs/reports/meridian_stage1.log
and INFO-level output to the console.
"""

import logging
import sys
from pathlib import Path
from utils.paths import DEFAULT_LOG_FILE, REPORTS_OUTPUT_DIR

_configured = False


def setup_logging(
    log_file: Path | str | None = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up root or package-level logger with file and console handlers.
    Ensures safe formatting with timestamps, levels, and module names.
    """
    global _configured

    REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_log_file = Path(log_file) if log_file else DEFAULT_LOG_FILE

    logger = logging.getLogger("meridian")
    logger.setLevel(log_level)

    # Avoid duplicate handlers if setup is called multiple times
    if not _configured or not logger.handlers:
        logger.handlers.clear()

        # Format: timestamp [level] [module_name]: message
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File Handler (append mode)
        file_handler = logging.FileHandler(target_log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.propagate = False
        _configured = True

    return logger


def get_logger(name: str = "meridian") -> logging.Logger:
    """Retrieve a namespaced child logger or root meridian logger."""
    if not _configured:
        setup_logging()

    if name.startswith("meridian"):
        return logging.getLogger(name)
    return logging.getLogger(f"meridian.{name}")
