"""Tests for strict PPO experiment configuration."""

from pathlib import Path

import pytest

from uav_swarm_control.configuration import (
    ConfigurationError,
    load_ppo_experiment_config,
    ppo_experiment_config_from_mapping,
)


def _mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "ppo-test",
        "seed": 42,
        "task": {
            "target_low": -0.8,
            "target_high": 0.8,
            "success_tolerance": 0.1,
            "evaluation_episodes": 64,
        },
        "algorithm": {
            "total_steps": 1024,
            "rollout_steps": 256,
            "update_epochs": 4,
            "minibatch_size": 64,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coefficient": 0.2,
            "value_coefficient": 0.5,
            "entropy_coefficient": 0.0,
            "max_gradient_norm": 0.5,
            "hidden_sizes": [32, 32],
            "initial_log_standard_deviation": -0.5,
            "device": "cpu",
        },
    }


def test_sample_ppo_configuration_loads() -> None:
    path = Path(__file__).parents[2] / "configs" / "experiment" / "ppo_continuous_bandit.yaml"

    config = load_ppo_experiment_config(path)

    assert config.name == "ppo-continuous-bandit"
    assert config.algorithm.num_updates == 16
    assert config.algorithm.hidden_sizes == (32, 32)


def test_ppo_configuration_rejects_unknown_keys() -> None:
    values = _mapping()
    algorithm = values["algorithm"]
    assert isinstance(algorithm, dict)
    algorithm["mystery"] = 1

    with pytest.raises(ConfigurationError, match="unknown keys: mystery"):
        ppo_experiment_config_from_mapping(values)


def test_ppo_configuration_rejects_incomplete_minibatches() -> None:
    values = _mapping()
    algorithm = values["algorithm"]
    assert isinstance(algorithm, dict)
    algorithm["minibatch_size"] = 100

    with pytest.raises(ConfigurationError, match="divisible by minibatch_size"):
        ppo_experiment_config_from_mapping(values)


def test_ppo_configuration_rejects_target_outside_action_bounds() -> None:
    values = _mapping()
    task = values["task"]
    assert isinstance(task, dict)
    task["target_high"] = 1.0

    with pytest.raises(ConfigurationError, match="target bounds"):
        ppo_experiment_config_from_mapping(values)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("algorithm", "learning_rate", 0.0, "greater than zero"),
        ("algorithm", "gamma", 1.1, r"\[0, 1\]"),
        ("algorithm", "clip_coefficient", 0.0, r"\(0, 1\)"),
        ("algorithm", "value_coefficient", -0.1, "non-negative"),
        ("algorithm", "initial_log_standard_deviation", float("inf"), "finite"),
        ("algorithm", "device", "tpu", "one of"),
        ("algorithm", "hidden_sizes", [], "must not be empty"),
        ("algorithm", "hidden_sizes", "32", "YAML list"),
        ("algorithm", "total_steps", 1000, "divisible by rollout_steps"),
        ("task", "success_tolerance", 2.0, "smaller than the target range"),
    ],
)
def test_ppo_configuration_rejects_invalid_scientific_settings(
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    values = _mapping()
    selected = values[section]
    assert isinstance(selected, dict)
    selected[key] = value

    with pytest.raises(ConfigurationError, match=message):
        ppo_experiment_config_from_mapping(values)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("schema_version", 2, "unsupported schema_version"),
        ("name", "Not Kebab Case", "kebab-case"),
        ("seed", -1, "at least 0"),
    ],
)
def test_ppo_configuration_rejects_invalid_experiment_identity(
    key: str,
    value: object,
    message: str,
) -> None:
    values = _mapping()
    values[key] = value

    with pytest.raises(ConfigurationError, match=message):
        ppo_experiment_config_from_mapping(values)


def test_ppo_configuration_requires_every_key() -> None:
    values = _mapping()
    del values["task"]

    with pytest.raises(ConfigurationError, match="missing required keys: task"):
        ppo_experiment_config_from_mapping(values)


def test_ppo_loader_rejects_wrong_extension_and_malformed_yaml(tmp_path: Path) -> None:
    wrong_extension = tmp_path / "ppo.txt"
    wrong_extension.write_text("configuration", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"\.yaml or \.yml"):
        load_ppo_experiment_config(wrong_extension)

    malformed = tmp_path / "ppo.yaml"
    malformed.write_text("root: [", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid YAML"):
        load_ppo_experiment_config(malformed)
