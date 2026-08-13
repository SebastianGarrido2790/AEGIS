"""
Centralized logging configuration for the entire project.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__, headline="main.py")
    logger.info("Started data download...")
"""

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

from src.constants import LOGS_DIR

LOG_FILE = LOGS_DIR / "running_logs.log"


def get_logger(name: str | None = None, headline: str | None = None) -> logging.Logger:
    """
    Returns a configured logger with consistent formatting.
    Adds an optional headline section to separate logs per script.

    Ensures that:
        - Only one handler is attached (prevents duplicates)
        - Log messages include timestamps and module names
        - Works safely across multi-module projects

    Args:
        name (Optional[str]): Optional logger name, typically __name__.
        headline (Optional[str]): Optional headline for visual separation
            (e.g., script name).

    Returns:
        logging.Logger: Configured logger instance (using RichHandler if available).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if logger already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M",
        )

        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5_000_000,  # 5 MB per file
            backupCount=5,  # Keep up to 5 old logs
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        try:
            from rich.logging import RichHandler

            console_handler = RichHandler(rich_tracebacks=True, markup=True)
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        except ImportError:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.propagate = False

        # Add a newline separator before each run (for readability)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write("\n\n")

        # Add a visually distinct headline
        if headline:
            headline_text = (
                f"========================= START: {headline} "
                f"({datetime.now():%Y-%m-%d %H:%M}) =========================\n"
            )
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(headline_text)

    return logger


def log_spacer() -> None:
    """
    Appends a raw newline to the log file to provide visual spacing
    without the log formatter prefix (timestamp/levelname).
    """
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write("\n")
