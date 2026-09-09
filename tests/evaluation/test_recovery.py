"""Tests for sustained-threshold recovery metrics."""

import pytest

from uav_swarm_control.evaluation.recovery import measure_recovery


def test_recovery_requires_a_sustained_below_threshold_window() -> None:
    result = measure_recovery(
        [0.6, 0.2, 0.4, 0.25, 0.2, 0.1],
        threshold=0.3,
        hold_steps=3,
        time_step_seconds=0.1,
    )
    assert result.peak_error == 0.6
    assert result.recovery_seconds == pytest.approx(0.5)
    assert result.final_error == 0.1
