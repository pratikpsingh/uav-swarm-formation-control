"""Command-line entry point for project tools."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from uav_swarm_control.algorithms.mappo import (
    evaluate_mappo,
    load_mappo_checkpoint,
    save_mappo_checkpoint,
    train_mappo,
)
from uav_swarm_control.algorithms.ppo import (
    evaluate_continuous_bandit,
    load_policy_checkpoint,
    save_policy_checkpoint,
    train_ppo,
)
from uav_swarm_control.configuration import (
    ConfigurationError,
    load_experiment_config,
    load_mappo_experiment_config,
    load_ppo_experiment_config,
    load_pybullet_experiment_config,
)
from uav_swarm_control.controllers.proportional import ProportionalPositionController
from uav_swarm_control.environments.continuous_bandit import ContinuousTargetBandit
from uav_swarm_control.environments.kinematic import KinematicSwarmEnvironment
from uav_swarm_control.environments.pybullet import PyBulletSwarmEnvironment
from uav_swarm_control.evaluation.artifacts import pybullet_episode_record, save_json_artifact
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
    run_pybullet = commands.add_parser(
        "run-pybullet",
        help="run the scripted baseline in pinned Crazyflie rigid-body simulation",
    )
    run_pybullet.add_argument("--config", type=Path, required=True)
    run_pybullet.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/pybullet-scripted.json"),
        help="JSON result path (default: %(default)s)",
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
    train_multi_agent = commands.add_parser(
        "train-mappo",
        help="train parameter-shared MAPPO on the kinematic swarm task",
    )
    train_multi_agent.add_argument("--config", type=Path, required=True)
    train_multi_agent.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/mappo_triangle_kinematic.pt"),
        help="output checkpoint path (default: %(default)s)",
    )
    evaluate_multi_agent = commands.add_parser(
        "evaluate-mappo",
        help="evaluate a saved MAPPO actor without its centralized critic",
    )
    evaluate_multi_agent.add_argument("--config", type=Path, required=True)
    evaluate_multi_agent.add_argument("--checkpoint", type=Path, required=True)
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
    if args.command == "run-pybullet":
        try:
            config = load_pybullet_experiment_config(args.config)
            environment = PyBulletSwarmEnvironment(config)
            controller = ProportionalPositionController(
                gain_per_second=config.experiment.controller.gain_per_second,
                max_velocity_component_mps=(
                    config.experiment.environment.max_velocity_component_mps
                ),
            )
            try:
                result = run_episode(
                    environment,
                    controller,
                    seed=config.experiment.seed,
                    safety_step_limit=config.experiment.environment.max_episode_steps,
                )
                record = pybullet_episode_record(
                    config,
                    environment.simulator_metadata,
                    result,
                )
                output_path = save_json_artifact(args.output, record)
            finally:
                environment.close()
        except (ConfigurationError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        LOGGER.info(
            "PyBullet episode complete: success=%s steps=%d position_rmse_m=%.6f result=%s",
            result.success,
            result.steps,
            result.final_metrics["position_rmse_m"],
            output_path,
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
    if args.command == "train-mappo":
        try:
            config = load_mappo_experiment_config(args.config)

            def environment_factory() -> KinematicSwarmEnvironment:
                return KinematicSwarmEnvironment(config.experiment)

            result = train_mappo(environment_factory, config.algorithm, seed=config.experiment.seed)
            evaluation = evaluate_mappo(
                result.model,
                environment_factory,
                episodes=config.evaluation_episodes,
                seed=config.experiment.seed,
                device=result.device,
            )
            checkpoint_path = save_mappo_checkpoint(
                args.checkpoint,
                result.model,
                metadata={
                    "experiment": config.experiment.name,
                    "seed": config.experiment.seed,
                    "environment_steps": result.environment_steps,
                    "agent_samples": result.agent_samples,
                },
            )
        except (ConfigurationError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        LOGGER.info(
            "MAPPO training complete: environment_steps=%d agent_samples=%d success_rate=%.3f "
            "position_rmse_m=%.6f checkpoint=%s",
            result.environment_steps,
            result.agent_samples,
            evaluation.success_rate,
            evaluation.mean_final_position_rmse_m,
            checkpoint_path,
        )
        return
    if args.command == "evaluate-mappo":
        try:
            config = load_mappo_experiment_config(args.config)
            model, metadata = load_mappo_checkpoint(args.checkpoint, device="cpu")
            evaluation = evaluate_mappo(
                model,
                lambda: KinematicSwarmEnvironment(config.experiment),
                episodes=config.evaluation_episodes,
                seed=config.experiment.seed,
            )
        except (ConfigurationError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        LOGGER.info(
            "MAPPO evaluation complete: episodes=%d success_rate=%.3f collision_rate=%.3f "
            "position_rmse_m=%.6f shape_rmse_m=%.6f checkpoint_steps=%s",
            evaluation.episodes,
            evaluation.success_rate,
            evaluation.collision_episode_rate,
            evaluation.mean_final_position_rmse_m,
            evaluation.mean_final_shape_rmse_m,
            metadata.get("environment_steps", "unknown"),
        )
        return
    LOGGER.info("Kinematic control loop is ready; PPO and MAPPO foundations are also ready.")
