"""Tests for consistent application logging."""

import logging
import time

from uav_swarm_control.logging import LogLevel, UtcFormatter, configure_logging


def test_utc_formatter_uses_utc_clock() -> None:
    """Log timestamps must not depend on a lab machine's local timezone."""
    assert UtcFormatter.converter is time.gmtime


def test_configure_logging_sets_requested_level() -> None:
    """The configured severity should control the root logger and its handler."""
    configure_logging(LogLevel.DEBUG)

    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG
    assert root_logger.handlers[0].level == logging.DEBUG
