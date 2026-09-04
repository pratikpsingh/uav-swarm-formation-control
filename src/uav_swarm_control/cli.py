"""Command-line entry point for project tools."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from uav_swarm_control.configuration import ConfigurationError, load_experiment_config
from uav_swarm_control.controllers.proportional import ProportionalPositionController
from uav_swarm_control.environments.kinematic import KinematicSwarmEnvironment
from uav_swarm_control.evaluation.rollout import run_episode
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
    commands = parser.add_subparsers(dest="command")
    run_scripted = commands.add_parser(
        "run-scripted",
        help="run the deterministic proportional-controller baseline",
    )
    run_scripted.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to a validated experiment YAML file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the currently available project command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(LogLevel(args.log_level))
    if args.command == "run-scripted":
        try:
            config = load_experiment_config(args.config)
            environment = KinematicSwarmEnvironment(config)
            controller = ProportionalPositionController(
                gain_per_second=config.controller.gain_per_second,
                max_velocity_component_mps=config.environment.max_velocity_component_mps,
            )
            try:
                result = run_episode(
                    environment,
                    controller,
                    seed=config.seed,
                    safety_step_limit=config.environment.max_episode_steps,
                )
            finally:
                environment.close()
        except (ConfigurationError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        LOGGER.info(
            "Scripted episode complete: success=%s steps=%d position_rmse_m=%.6f",
            result.success,
            result.steps,
            result.final_metrics["position_rmse_m"],
        )
        return
    LOGGER.info("Kinematic control loop is ready; no reinforcement-learning algorithm exists yet.")
