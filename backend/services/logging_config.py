import os
import sys
from loguru import logger


def setup_logging():
    """Configure Loguru with JSON structured output."""
    logger.remove()

    level = os.environ.get("LOG_LEVEL", "INFO")
    logger.add(
        sys.stdout,
        level=level,
        serialize=True,
        backtrace=True,
        diagnose=True,
    )

    log_file = os.environ.get("LOG_FILE")
    log_dir = os.environ.get("LOG_DIR")
    if log_file or log_dir:
        path = log_file or os.path.join(log_dir or "logs", "app.log")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        logger.add(
            path,
            level=os.environ.get("LOG_FILE_LEVEL", "DEBUG"),
            serialize=True,
            rotation=os.environ.get("LOG_ROTATION", "10 MB"),
            retention=os.environ.get("LOG_RETENTION", "10 days"),
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

    logger.info("Structured logging initialized via Loguru.")


# Initialize logging on import
setup_logging()
