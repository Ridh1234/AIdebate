from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(logs_dir: str = "logs") -> str:
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"debate_{timestamp}.log")

    logger = logging.getLogger("debate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    ch = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Logger initialized. Writing to %s", log_path)
    return log_path


def get_logger() -> logging.Logger:
    return logging.getLogger("debate")
