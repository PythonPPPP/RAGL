from __future__ import annotations

import sys
from loguru import logger


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        colorize=True,
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )


__all__ = ["logger", "setup_logging"]
