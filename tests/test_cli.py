"""Tests for the project command-line entry point."""

from pytest import CaptureFixture

from uav_swarm_control.cli import main


def test_cli_reports_current_project_status(capsys: CaptureFixture[str]) -> None:
    """The CLI should prove that packaging and logging are connected."""
    main(["--log-level", "INFO"])

    captured = capsys.readouterr()
    assert "INFO" in captured.err
    assert "Geometry and multi-agent contracts are ready" in captured.err
