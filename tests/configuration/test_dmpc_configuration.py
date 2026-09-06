"""Validation checks for explicit DMPC controller configuration."""

from pathlib import Path
from typing import cast

import pytest
import yaml

from uav_swarm_control.configuration import ConfigurationError, load_dmpc_config

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/algorithm/dmpc_native.yaml"


def test_native_reference_parameters_are_explicit() -> None:
    config = load_dmpc_config(CONFIG)
    assert config.prediction_horizon_steps == 15
    assert config.planning_interval_seconds == 0.2
    assert config.weights.terminal_position == 1.0
    assert config.weights.velocity == 0.0001
    assert config.weights.acceleration == 0.005
    assert config.weights.jerk == 0.001
    assert config.limits.minimum_separation_m == 0.4
    assert config.limits.downwash_scaling == 4.0


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("root", "prediction_horizon_steps", 1),
        ("weights", "terminal_position", 0),
        ("limits", "minimum_separation_m", -0.1),
        ("solver", "max_iterations", True),
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, section: str, key: str, value: object) -> None:
    raw = cast(dict[str, object], yaml.safe_load(CONFIG.read_text()))
    if section == "root":
        raw[key] = value
    else:
        cast(dict[str, object], raw[section])[key] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigurationError):
        load_dmpc_config(path)


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    raw = cast(dict[str, object], yaml.safe_load(CONFIG.read_text()))
    raw["hidden_choice"] = 1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigurationError, match="unknown keys"):
        load_dmpc_config(path)
