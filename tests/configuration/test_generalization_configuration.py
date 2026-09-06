"""Validation tests for Stage 9 experiment protocols."""

from pathlib import Path
from typing import cast

import pytest
import yaml

from uav_swarm_control.configuration import ConfigurationError, load_generalization_config
from uav_swarm_control.evaluation.generalization import generalization_smoke_config
from uav_swarm_control.formations import FormationKind, pose_ranges_overlap

ROOT = Path(__file__).parents[2]
CONFIG_DIRECTORY = ROOT / "configs/experiment"
CUBE_CONFIG = CONFIG_DIRECTORY / "stage9_cube_8uav.yaml"


@pytest.mark.parametrize(
    ("name", "kind", "agents", "variants"),
    [
        ("stage9_plane_4uav.yaml", FormationKind.PLANE, 4, 1),
        ("stage9_pyramid_5uav.yaml", FormationKind.PYRAMID, 5, 1),
        ("stage9_cube_8uav.yaml", FormationKind.CUBE, 8, 4),
        ("stage9_sphere_8uav.yaml", FormationKind.SPHERE, 8, 1),
    ],
)
def test_research_suite_is_explicit_and_disjoint(
    name: str, kind: FormationKind, agents: int, variants: int
) -> None:
    config = load_generalization_config(CONFIG_DIRECTORY / name)

    assert config.mappo.experiment.formation.kind is kind
    assert config.mappo.experiment.formation.num_agents == agents
    assert len(config.variants) == variants
    assert len(config.training_seeds) == 5
    assert config.profile == "research"
    assert config.mappo.algorithm.ppo.total_steps == 10_000_000
    assert not pose_ranges_overlap(config.training_pose, config.held_out_pose)


def test_smoke_profile_preserves_scientific_factors() -> None:
    config = load_generalization_config(CUBE_CONFIG)
    smoke = generalization_smoke_config(config)

    assert smoke.profile == "smoke"
    assert smoke.variants == config.variants
    assert smoke.training_seeds == config.training_seeds
    assert smoke.training_pose == config.training_pose
    assert smoke.held_out_pose == config.held_out_pose
    assert smoke.mappo.algorithm.ppo.total_steps == 256
    assert smoke.mappo.evaluation_episodes == 2
    assert config.profile == "research"


def _raw_config() -> dict[str, object]:
    return cast(dict[str, object], yaml.safe_load(CUBE_CONFIG.read_text(encoding="utf-8")))


def _write_config(tmp_path: Path, raw: dict[str, object]) -> Path:
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_overlapping_pose_splits_are_rejected(tmp_path: Path) -> None:
    raw = _raw_config()
    generalization = cast(dict[str, object], raw["generalization"])
    generalization["held_out_pose"] = generalization["training_pose"]

    with pytest.raises(ConfigurationError, match="must be disjoint"):
        load_generalization_config(_write_config(tmp_path, raw))


def test_duplicate_experimental_choice_is_rejected(tmp_path: Path) -> None:
    raw = _raw_config()
    generalization = cast(dict[str, object], raw["generalization"])
    variants = cast(list[dict[str, object]], generalization["variants"])
    variants[1]["assignment"] = variants[0]["assignment"]
    variants[1]["coordinate_frame"] = variants[0]["coordinate_frame"]

    with pytest.raises(ConfigurationError, match="unique names and choices"):
        load_generalization_config(_write_config(tmp_path, raw))


def test_nonspatial_formation_is_rejected(tmp_path: Path) -> None:
    raw = _raw_config()
    cast(dict[str, object], raw["formation"])["kind"] = "line"

    with pytest.raises(ConfigurationError, match=r"formation\.kind"):
        load_generalization_config(_write_config(tmp_path, raw))


def test_unknown_protocol_key_is_rejected(tmp_path: Path) -> None:
    raw = _raw_config()
    cast(dict[str, object], raw["protocol"])["hidden_choice"] = True

    with pytest.raises(ConfigurationError, match="unknown keys"):
        load_generalization_config(_write_config(tmp_path, raw))
