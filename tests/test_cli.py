"""Tests for the project command-line entry point."""

from pathlib import Path

from pytest import CaptureFixture

from uav_swarm_control.cli import main


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
