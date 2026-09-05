"""Command-line entry point for project tools."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from uav_swarm_control.algorithms.ppo import (
    evaluate_continuous_bandit,
    load_policy_checkpoint,
    save_policy_checkpoint,
    train_ppo,
)
from uav_swarm_control.configuration import (
    ConfigurationError,
    load_experiment_config,
    load_ppo_experiment_config,
)
from uav_swarm_control.controllers.proportional import ProportionalPositionController
from uav_swarm_control.environments.continuous_bandit import ContinuousTargetBandit
from uav_swarm_control.environments.kinematic import KinematicSwarmEnvironment
from uav_swarm_control.evaluation.rollout import run_episode
from uav_swarm_control.logging import LogLevel, configure_logging
from uav_swarm_control.seeding import RandomStream, derive_seed

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
    train = commands.add_parser(
        "train-ppo",
        help="train PPO on the continuous reference task",
    )
    train.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to a validated PPO experiment YAML file",
    )
    train.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/ppo_continuous_bandit.pt"),
        help="output checkpoint path (default: %(default)s)",
    )
    evaluate = commands.add_parser(
        "evaluate-ppo",
        help="evaluate a saved PPO policy on the reference task",
    )
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
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
    if args.command == "train-ppo":
        try:
            config = load_ppo_experiment_config(args.config)
            result = train_ppo(
                ContinuousTargetBandit(config.task),
                config.algorithm,
                seed=config.seed,
            )
            evaluation = evaluate_continuous_bandit(
                result.model,
                config.task,
                seed=derive_seed(config.seed, RandomStream.EVALUATION),
                device=result.device,
            )
            checkpoint_path = save_policy_checkpoint(
                args.checkpoint,
                result.model,
                metadata={
                    "experiment": config.name,
                    "seed": config.seed,
                    "environment_steps": result.environment_steps,
                },
            )
        except (ConfigurationError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        LOGGER.info(
            "PPO training complete: steps=%d action_mse=%.6f success_rate=%.3f checkpoint=%s",
            result.environment_steps,
            evaluation.action_mse,
            evaluation.success_rate,
            checkpoint_path,
        )
        return
    if args.command == "evaluate-ppo":
        try:
            config = load_ppo_experiment_config(args.config)
            model, metadata = load_policy_checkpoint(args.checkpoint, device="cpu")
            evaluation = evaluate_continuous_bandit(
                model,
                config.task,
                seed=derive_seed(config.seed, RandomStream.EVALUATION),
            )
        except (ConfigurationError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        LOGGER.info(
            "PPO evaluation complete: episodes=%d mean_reward=%.6f action_mse=%.6f "
            "success_rate=%.3f checkpoint_steps=%s",
            evaluation.episodes,
            evaluation.mean_reward,
            evaluation.action_mse,
            evaluation.success_rate,
            metadata.get("environment_steps", "unknown"),
        )
        return
    LOGGER.info("Kinematic control loop is ready; single-agent PPO foundation is also ready.")
