"""Command-line entry point for project tools."""

import argparse
import logging
from collections.abc import Sequence

from uav_swarm_control.logging import LogLevel, configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser."""
    parser = argparse.ArgumentParser(
        prog="uav-swarm-control",
        description="Research tools for multi-UAV swarm control.",
    )
    parser.add_argument(
        "--log-level",
        choices=[level.value for level in LogLevel],
        default=LogLevel.INFO.value,
        help="minimum severity written to the console (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the currently available project command."""
    args = build_parser().parse_args(argv)
    configure_logging(LogLevel(args.log_level))
    LOGGER.info("Geometry and multi-agent contracts are ready; no environment exists yet.")
