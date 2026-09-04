"""Tests for the project command-line entry point."""

from pytest import CaptureFixture

from uav_swarm_control.cli import main


def test_cli_reports_foundation_status(capsys: CaptureFixture[str]) -> None:
    """The initial CLI should prove that packaging and logging are connected."""
    main(["--log-level", "INFO"])

    captured = capsys.readouterr()
    assert "INFO" in captured.err
    assert "Project foundation is ready" in captured.err
