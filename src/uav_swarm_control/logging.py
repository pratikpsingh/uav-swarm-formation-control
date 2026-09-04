"""Central logging configuration for command-line applications."""

import logging
import logging.config
import time
from enum import StrEnum


class LogLevel(StrEnum):
    """Supported application logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class UtcFormatter(logging.Formatter):
    """Format timestamps in UTC so logs agree across machines."""

    converter = time.gmtime


def configure_logging(level: LogLevel = LogLevel.INFO) -> None:
    """Configure process-wide console logging for an application entry point."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "console": {
                    "()": UtcFormatter,
                    "format": "%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%dT%H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "console",
                    "level": level.value,
                    "stream": "ext://sys.stderr",
                }
            },
            "root": {"handlers": ["console"], "level": level.value},
        }
    )
