"""Tests for the project command-line entry point."""

from pathlib import Path

import pytest
from pytest import CaptureFixture

from uav_swarm_control.cli import main


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
