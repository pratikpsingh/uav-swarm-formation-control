"""Command-line entry point for project tools."""

import argparse
import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path

import torch

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
    load_communication_experiment_config,
    load_dmpc_config,
    load_experiment_config,
    load_generalization_config,
    load_mappo_experiment_config,
    load_obstacle_experiment_config,
    load_ppo_experiment_config,
    load_pybullet_experiment_config,
)
from uav_swarm_control.configuration.baseline import load_baseline_config
from uav_swarm_control.controllers.proportional import ProportionalPositionController
from uav_swarm_control.environments.continuous_bandit import ContinuousTargetBandit
from uav_swarm_control.environments.kinematic import KinematicSwarmEnvironment
from uav_swarm_control.environments.pybullet import PyBulletSwarmEnvironment
from uav_swarm_control.evaluation.artifacts import pybullet_episode_record, save_json_artifact
from uav_swarm_control.evaluation.baseline import (
    evaluate_saved_baseline,
    run_baseline,
    smoke_config,
)
from uav_swarm_control.evaluation.comparison import compare_controller_results
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
    baseline = commands.add_parser(
        "run-baseline", help="train and evaluate the corrected Paper 04 baseline across seeds"
    )
    baseline.add_argument("--config", type=Path, action="append", required=True)
    baseline.add_argument("--output", type=Path, default=Path("artifacts/baselines"))
    baseline.add_argument("--project-root", type=Path, default=Path.cwd())
    baseline.add_argument(
        "--smoke", action="store_true", help="256 training steps per seed; validates plumbing only"
    )
    baseline.add_argument(
        "--resume",
        action="store_true",
        help="skip matching completed seeds; does not resume optimizer state",
    )
    baseline.add_argument("--torch-threads", type=int, default=1)
    generalization = commands.add_parser(
        "run-generalization",
        help="train and evaluate Stage 9 policies on disjoint 3D pose splits",
    )
    generalization.add_argument("--config", type=Path, action="append", required=True)
    generalization.add_argument("--output", type=Path, default=Path("artifacts/generalization"))
    generalization.add_argument("--project-root", type=Path, default=Path.cwd())
    generalization.add_argument(
        "--smoke",
        action="store_true",
        help="retain all variants and seeds with a bounded plumbing-only budget",
    )
    generalization.add_argument(
        "--resume",
        action="store_true",
        help="skip matching completed seeds; does not resume optimizer state",
    )
    generalization.add_argument("--torch-threads", type=int, default=1)
    obstacles = commands.add_parser(
        "run-obstacle-study",
        help="train matched controls and a curriculum on oracle dynamic obstacles",
    )
    obstacles.add_argument("--config", type=Path, action="append", required=True)
    obstacles.add_argument("--output", type=Path, default=Path("artifacts/obstacles"))
    obstacles.add_argument("--project-root", type=Path, default=Path.cwd())
    obstacles.add_argument(
        "--smoke",
        action="store_true",
        help="retain every regimen, scenario, and seed with a plumbing-only budget",
    )
    obstacles.add_argument(
        "--resume",
        action="store_true",
        help="skip matching completed seeds; does not resume optimizer state",
    )
    obstacles.add_argument("--torch-threads", type=int, default=1)
    communication = commands.add_parser(
        "run-communication-study",
        help="compare fixed and variable neighbor policies across topology conditions",
    )
    communication.add_argument("--config", type=Path, action="append", required=True)
    communication.add_argument("--output", type=Path, default=Path("artifacts/communication"))
    communication.add_argument("--project-root", type=Path, default=Path.cwd())
    communication.add_argument(
        "--smoke",
        action="store_true",
        help="retain every condition, regimen, and seed with a plumbing-only budget",
    )
    communication.add_argument(
        "--resume",
        action="store_true",
        help="skip matching completed seeds; does not resume optimizer state",
    )
    communication.add_argument("--torch-threads", type=int, default=1)
    baseline_evaluate = commands.add_parser(
        "evaluate-baseline", help="evaluate a saved baseline actor on compatible held-out episodes"
    )
    baseline_evaluate.add_argument("--config", type=Path, required=True)
    baseline_evaluate.add_argument("--checkpoint", type=Path, required=True)
    baseline_evaluate.add_argument("--output", type=Path, required=True)
    baseline_evaluate.add_argument("--project-root", type=Path, default=Path.cwd())
    baseline_evaluate.add_argument("--smoke", action="store_true")
    dmpc = commands.add_parser(
        "run-dmpc",
        help="evaluate the classical DMPC adaptation on one or more Paper 04 tasks",
    )
    dmpc.add_argument("--task", type=Path, action="append", required=True)
    dmpc.add_argument("--controller", type=Path, required=True)
    dmpc.add_argument("--output", type=Path, default=Path("artifacts/baselines"))
    dmpc.add_argument("--project-root", type=Path, default=Path.cwd())
    dmpc.add_argument(
        "--smoke",
        action="store_true",
        help="use the same two-episode, 48-step profile as the MAPPO smoke run",
    )
    compare = commands.add_parser(
        "compare-controllers",
        help="compare compatible MAPPO and DMPC artifacts using common metrics",
    )
    compare.add_argument("--mappo-summary", type=Path, required=True)
    compare.add_argument("--dmpc-result", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the currently available project command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(LogLevel(args.log_level))
    if args.command == "compare-controllers":
        try:
            path = compare_controller_results(
                args.mappo_summary,
                args.dmpc_result,
                args.output,
            )
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        LOGGER.info("Controller comparison complete: %s", path)
        return
    if args.command == "run-dmpc":
        try:
            # Keep SciPy optional for users who do not invoke the DMPC command.
            from uav_swarm_control.evaluation.dmpc import dmpc_smoke_config, run_dmpc

            controller_config = load_dmpc_config(args.controller)
            tasks = [load_baseline_config(path) for path in args.task]
            for task in tasks:
                if args.smoke:
                    task = dmpc_smoke_config(task)
                result_path = run_dmpc(
                    task,
                    controller_config,
                    args.output,
                    project_root=args.project_root.resolve(),
                )
                LOGGER.info(
                    "DMPC evaluation complete: profile=%s result=%s",
                    task.profile,
                    result_path,
                )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            parser.error(str(error))
        return
    if args.command == "evaluate-baseline":
        try:
            baseline_config = load_baseline_config(args.config)
            if args.smoke:
                baseline_config = smoke_config(baseline_config)
            path = evaluate_saved_baseline(
                baseline_config,
                args.checkpoint,
                args.output,
                project_root=args.project_root.resolve(),
            )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            parser.error(str(error))
        LOGGER.info("Baseline evaluation complete: %s", path)
        return
    if args.command == "run-baseline":
        if args.torch_threads < 1:
            parser.error("--torch-threads must be positive.")
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(args.torch_threads)
            # Validate the whole requested suite before starting its first training run.
            configurations = [load_baseline_config(path) for path in args.config]
            for baseline_config in configurations:
                if args.smoke:
                    baseline_config = smoke_config(baseline_config)
                summary_path = run_baseline(
                    baseline_config,
                    args.output,
                    project_root=args.project_root.resolve(),
                    resume=args.resume,
                )
                LOGGER.info(
                    "Baseline complete: profile=%s summary=%s",
                    baseline_config.profile,
                    summary_path,
                )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            parser.error(str(error))
        finally:
            torch.set_num_threads(previous_threads)
        return
    if args.command == "run-generalization":
        if args.torch_threads < 1:
            parser.error("--torch-threads must be positive.")
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(args.torch_threads)
            # Validate the entire suite before creating any training artifacts.
            configurations = [load_generalization_config(path) for path in args.config]
            from uav_swarm_control.evaluation.generalization import (
                generalization_smoke_config,
                run_generalization,
            )

            for generalization_config in configurations:
                if args.smoke:
                    generalization_config = generalization_smoke_config(generalization_config)
                summary_path = run_generalization(
                    generalization_config,
                    args.output,
                    project_root=args.project_root.resolve(),
                    resume=args.resume,
                )
                LOGGER.info(
                    "Generalization study complete: profile=%s summary=%s",
                    generalization_config.profile,
                    summary_path,
                )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            parser.error(str(error))
        finally:
            torch.set_num_threads(previous_threads)
        return
    if args.command == "run-obstacle-study":
        if args.torch_threads < 1:
            parser.error("--torch-threads must be positive.")
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(args.torch_threads)
            configurations = [load_obstacle_experiment_config(path) for path in args.config]
            from uav_swarm_control.evaluation.obstacles import (
                obstacle_smoke_config,
                run_obstacle_study,
            )

            for obstacle_config in configurations:
                if args.smoke:
                    obstacle_config = obstacle_smoke_config(obstacle_config)
                summary_path = run_obstacle_study(
                    obstacle_config,
                    args.output,
                    project_root=args.project_root.resolve(),
                    resume=args.resume,
                )
                LOGGER.info(
                    "Obstacle study complete: profile=%s summary=%s",
                    obstacle_config.profile,
                    summary_path,
                )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            parser.error(str(error))
        finally:
            torch.set_num_threads(previous_threads)
        return
    if args.command == "run-communication-study":
        if args.torch_threads < 1:
            parser.error("--torch-threads must be positive.")
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(args.torch_threads)
            configurations = [load_communication_experiment_config(path) for path in args.config]
            from uav_swarm_control.evaluation.communication import (
                communication_smoke_config,
                run_communication_study,
            )

            for communication_config in configurations:
                if args.smoke:
                    communication_config = communication_smoke_config(communication_config)
                summary_path = run_communication_study(
                    communication_config,
                    args.output,
                    project_root=args.project_root.resolve(),
                    resume=args.resume,
                )
                LOGGER.info(
                    "Communication study complete: profile=%s summary=%s",
                    communication_config.profile,
                    summary_path,
                )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            parser.error(str(error))
        finally:
            torch.set_num_threads(previous_threads)
        return
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
