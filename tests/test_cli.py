"""Tests for the project command-line entry point."""

from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch

from uav_swarm_control.cli import main
from uav_swarm_control.configuration import (
    CommunicationExperimentConfig,
    DeploymentStudyConfig,
    GeneralizationConfig,
    ObstacleExperimentConfig,
)


def _write_small_ppo_config(path: Path) -> None:
    path.write_text(
        """\
schema_version: 1
name: ppo-cli-test
seed: 42
task:
  target_low: -0.8
  target_high: 0.8
  success_tolerance: 0.1
  evaluation_episodes: 8
algorithm:
  total_steps: 256
  rollout_steps: 256
  update_epochs: 1
  minibatch_size: 64
  learning_rate: 0.003
  gamma: 0.99
  gae_lambda: 0.95
  clip_coefficient: 0.2
  value_coefficient: 0.5
  entropy_coefficient: 0.0
  max_gradient_norm: 0.5
  hidden_sizes: [8, 8]
  initial_log_standard_deviation: -0.5
  device: cpu
""",
        encoding="utf-8",
    )


def _write_small_mappo_config(path: Path) -> None:
    path.write_text(
        """\
schema_version: 1
name: mappo-cli-test
seed: 42
formation:
  kind: triangle
  num_agents: 3
  spacing_m: 1.0
  center_m: [0.0, 0.0, 1.0]
  euler_radians: [0.0, 0.0, 0.0]
environment:
  time_step_seconds: 0.1
  max_episode_steps: 8
  max_velocity_component_mps: 1.0
observation:
  max_neighbors: 2
  neighbor_radius_m: null
task:
  initial_center_m: [-1.0, 0.0, 1.0]
  initial_position_noise_m: 0.0
  success_tolerance_m: 0.1
  success_hold_steps: 2
  collision_distance_m: 0.2
  terminate_on_collision: false
reward:
  navigation_weight: 1.0
  formation_weight: 0.25
  collision_penalty: 2.0
  smoothness_weight: 0.01
  success_bonus: 5.0
controller:
  gain_per_second: 1.5
algorithm:
  total_steps: 128
  rollout_steps: 64
  num_environments: 1
  update_epochs: 1
  minibatch_size: 64
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_coefficient: 0.2
  value_coefficient: 0.5
  entropy_coefficient: 0.001
  max_gradient_norm: 0.5
  hidden_sizes: [8, 8]
  critic_hidden_sizes: [16, 16]
  initial_log_standard_deviation: -0.5
  device: cpu
evaluation:
  episodes: 1
""",
        encoding="utf-8",
    )


def test_cli_reports_current_project_status(capsys: CaptureFixture[str]) -> None:
    """The CLI should prove that packaging and logging are connected."""
    main(["--log-level", "INFO"])

    captured = capsys.readouterr()
    assert "INFO" in captured.err
    assert "Kinematic control loop is ready" in captured.err


def test_cli_runs_scripted_triangle_episode(capsys: CaptureFixture[str]) -> None:
    """The sample YAML should drive a successful end-to-end rollout."""
    config_path = Path(__file__).parents[1] / "configs" / "experiment" / "triangle_kinematic.yaml"

    main(["--log-level", "INFO", "run-scripted", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert "success=True" in captured.err
    assert "position_rmse_m=" in captured.err


def test_cli_trains_saves_and_evaluates_ppo(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    """The CLI should connect configuration, training, checkpoints, and evaluation."""
    config_path = tmp_path / "ppo.yaml"
    checkpoint_path = tmp_path / "policy.pt"
    _write_small_ppo_config(config_path)

    main(
        [
            "train-ppo",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
    )
    training_output = capsys.readouterr()

    assert checkpoint_path.is_file()
    assert "PPO training complete" in training_output.err
    assert "steps=256" in training_output.err

    main(
        [
            "evaluate-ppo",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
    )
    evaluation_output = capsys.readouterr()

    assert "PPO evaluation complete" in evaluation_output.err
    assert "checkpoint_steps=256" in evaluation_output.err


def test_cli_reports_invalid_ppo_configuration(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    config_path = tmp_path / "ppo.txt"
    config_path.write_text("not yaml", encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["train-ppo", "--config", str(config_path)])

    assert "must use a .yaml or .yml extension" in capsys.readouterr().err


def test_cli_trains_saves_and_evaluates_mappo(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    config_path = tmp_path / "mappo.yaml"
    checkpoint_path = tmp_path / "mappo.pt"
    _write_small_mappo_config(config_path)

    main(
        [
            "train-mappo",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
    )
    training_output = capsys.readouterr()

    assert checkpoint_path.is_file()
    assert "MAPPO training complete" in training_output.err
    assert "environment_steps=128" in training_output.err
    assert "agent_samples=384" in training_output.err

    main(
        [
            "evaluate-mappo",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
    )
    evaluation_output = capsys.readouterr()

    assert "MAPPO evaluation complete" in evaluation_output.err
    assert "checkpoint_steps=128" in evaluation_output.err


def test_cli_validates_and_runs_generalization_smoke(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    """The command should apply explicit smoke overrides before orchestration."""
    from uav_swarm_control.evaluation import generalization

    calls: list[tuple[GeneralizationConfig, Path, Path, bool]] = []

    def fake_run(
        config: GeneralizationConfig,
        output: Path,
        *,
        project_root: Path,
        resume: bool = False,
    ) -> Path:
        calls.append((config, output, project_root, resume))
        return output / "summary.json"

    monkeypatch.setattr(generalization, "run_generalization", fake_run)
    root = Path(__file__).parents[1]
    output = tmp_path / "results"
    main(
        [
            "run-generalization",
            "--config",
            str(root / "configs/experiment/stage9_plane_4uav.yaml"),
            "--smoke",
            "--resume",
            "--output",
            str(output),
            "--project-root",
            str(root),
        ]
    )

    assert len(calls) == 1
    config, called_output, project_root, resume = calls[0]
    assert config.profile == "smoke"
    assert config.mappo.algorithm.ppo.total_steps == 256
    assert called_output == output
    assert project_root == root.resolve()
    assert resume
    assert "Generalization study complete" in capsys.readouterr().err


def test_cli_validates_and_runs_obstacle_smoke(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    from uav_swarm_control.evaluation import obstacles

    calls: list[tuple[ObstacleExperimentConfig, Path, Path, bool]] = []

    def fake_run(
        config: ObstacleExperimentConfig,
        output: Path,
        *,
        project_root: Path,
        resume: bool = False,
    ) -> Path:
        calls.append((config, output, project_root, resume))
        return output / "summary.json"

    monkeypatch.setattr(obstacles, "run_obstacle_study", fake_run)
    root = Path(__file__).parents[1]
    output = tmp_path / "obstacles"
    main(
        [
            "run-obstacle-study",
            "--config",
            str(root / "configs/experiment/stage10_dynamic_obstacles_4uav.yaml"),
            "--smoke",
            "--resume",
            "--output",
            str(output),
            "--project-root",
            str(root),
        ]
    )

    assert len(calls) == 1
    config, called_output, project_root, resume = calls[0]
    assert config.profile == "smoke"
    assert config.mappo.algorithm.ppo.total_steps == 256
    assert len(config.training_regimens) == 3
    assert len(config.evaluation_scenarios) == 4
    assert called_output == output
    assert project_root == root.resolve()
    assert resume
    assert "Obstacle study complete" in capsys.readouterr().err


def test_cli_validates_and_runs_communication_smoke(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    from uav_swarm_control.evaluation import communication

    calls: list[tuple[CommunicationExperimentConfig, Path, Path, bool]] = []

    def fake_run(
        config: CommunicationExperimentConfig,
        output: Path,
        *,
        project_root: Path,
        resume: bool = False,
    ) -> Path:
        calls.append((config, output, project_root, resume))
        return output / "summary.json"

    monkeypatch.setattr(communication, "run_communication_study", fake_run)
    root = Path(__file__).parents[1]
    output = tmp_path / "communication"
    main(
        [
            "run-communication-study",
            "--config",
            str(root / "configs/experiment/stage11_plane_4uav.yaml"),
            "--smoke",
            "--resume",
            "--output",
            str(output),
            "--project-root",
            str(root),
        ]
    )

    assert len(calls) == 1
    config, called_output, project_root, resume = calls[0]
    assert config.profile == "smoke"
    assert config.mappo.algorithm.ppo.total_steps == 256
    assert len(config.training_regimens) == 3
    assert len(config.evaluation_conditions) == 16
    assert called_output == output
    assert project_root == root.resolve()
    assert resume
    assert "Communication study complete" in capsys.readouterr().err


def test_cli_applies_bounded_deployment_smoke_profile(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    """The deployment command must preserve candidates and label smoke as non-scientific."""
    from uav_swarm_control.evaluation import deployment

    calls: list[tuple[CommunicationExperimentConfig, DeploymentStudyConfig]] = []

    def fake_run(
        task: CommunicationExperimentConfig,
        study: DeploymentStudyConfig,
        checkpoint: Path,
        teacher_result: Path,
        output: Path,
        *,
        project_root: Path,
    ) -> Path:
        del checkpoint, teacher_result, project_root
        calls.append((task, study))
        return output / "summary.json"

    monkeypatch.setattr(deployment, "run_deployment_study", fake_run)
    root = Path(__file__).parents[1]
    main(
        [
            "run-deployment-study",
            "--task",
            str(root / "configs/experiment/stage11_plane_4uav.yaml"),
            "--deployment",
            str(root / "configs/deployment/stage12_policy_compression.yaml"),
            "--teacher-checkpoint",
            str(tmp_path / "model.pt"),
            "--teacher-result",
            str(tmp_path / "result.json"),
            "--output",
            str(tmp_path / "deployment"),
            "--project-root",
            str(root),
            "--smoke",
        ]
    )

    assert len(calls) == 1
    task, study = calls[0]
    assert task.profile == study.profile == "smoke"
    assert len(study.candidates) == 7
    assert len(study.distillation_seeds) == 5
    assert "Deployment study complete" in capsys.readouterr().err
