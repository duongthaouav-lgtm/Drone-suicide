"""
logger.py - Logging setup for the YOLO Detection & Tracking System
"""
import logging
import os
from datetime import datetime

import config as cfg


def setup_logger() -> logging.Logger | None:
    """
    Configure and return the application logger.
    Returns None when ENABLE_LOGGING is False.
    """
    if not cfg.ENABLE_LOGGING:
        return None

    log_dir = os.path.join(cfg.OUTPUT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"detection_log_{timestamp}.txt")

    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(cfg.LOG_LEVEL)
    handlers.append(console_handler)

    # File handler (UTF-8 to support emoji on Windows)
    if cfg.LOG_TO_FILE:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(cfg.LOG_LEVEL)
        handlers.append(file_handler)

    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger("yolo_detection")
    logger.info("=" * 60)
    logger.info("YOLO Detection & Tracking System Started")
    logger.info("=" * 60)

    return logger


# Module-level logger instance (set after setup_logger() is called)
logger: logging.Logger | None = None