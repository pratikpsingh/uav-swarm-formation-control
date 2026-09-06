"""Versioned checkpoint persistence for shared actor and centralized critic."""

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from uav_swarm_control.models import NeighborEncoderSpec, SharedActorCentralCritic

_FORMAT_VERSION = 2
_SUPPORTED_FORMATS = {1, 2}


def _string_mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {name} must be a mapping.")
    result: dict[str, object] = {}
    for key, item in cast(dict[object, object], value).items():
        if not isinstance(key, str):
            raise ValueError(f"checkpoint {name} keys must be strings.")
        result[key] = item
    return result


def _integer(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"checkpoint {key} must be a positive integer.")
    return value


def _sizes(mapping: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = mapping.get(key)
    if not isinstance(value, tuple):
        raise ValueError(f"checkpoint {key} must be a tuple.")
    untyped = cast(tuple[object, ...], value)
    if not untyped or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in untyped
    ):
        raise ValueError(f"checkpoint {key} must contain positive integers.")
    return cast(tuple[int, ...], untyped)


def _neighbor_encoder(value: object) -> NeighborEncoderSpec | None:
    if value is None:
        return None
    specification = _string_mapping(value, name="neighbor_encoder")
    expected = {
        "ego_features",
        "max_neighbors",
        "neighbor_features",
        "embedding_size",
        "hidden_sizes",
    }
    if set(specification) != expected:
        raise ValueError("checkpoint neighbor_encoder has invalid keys.")
    return NeighborEncoderSpec(
        ego_features=_integer(specification, "ego_features"),
        max_neighbors=_integer(specification, "max_neighbors"),
        neighbor_features=_integer(specification, "neighbor_features"),
        embedding_size=_integer(specification, "embedding_size"),
        hidden_sizes=_sizes(specification, "hidden_sizes"),
    )


def save_mappo_checkpoint(
    path: str | Path,
    model: SharedActorCentralCritic,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically save the full training model and provenance metadata."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    torch.save(
        {
            "format_version": _FORMAT_VERSION,
            "model": {
                "local_observation_size": model.local_observation_size,
                "centralized_state_size": model.centralized_state_size,
                "action_size": model.action_size,
                "num_agents": model.num_agents,
                "actor_hidden_sizes": model.actor_hidden_sizes,
                "critic_hidden_sizes": model.critic_hidden_sizes,
                "initial_log_standard_deviation": model.initial_log_standard_deviation,
                "neighbor_encoder": (
                    asdict(model.neighbor_encoder) if model.neighbor_encoder is not None else None
                ),
            },
            "state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def load_mappo_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[SharedActorCentralCritic, dict[str, object]]:
    """Validate and reconstruct a MAPPO model with the weights-only loader."""
    raw = torch.load(Path(path), map_location=device, weights_only=True)
    payload = _string_mapping(cast(object, raw), name="root")
    format_version = payload.get("format_version")
    if format_version not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"unsupported MAPPO checkpoint format; expected one of {sorted(_SUPPORTED_FORMATS)}."
        )
    specification = _string_mapping(payload.get("model"), name="model specification")
    initial_log_standard_deviation = specification.get("initial_log_standard_deviation")
    if not isinstance(initial_log_standard_deviation, float):
        raise ValueError("checkpoint initial_log_standard_deviation must be a float.")
    model = SharedActorCentralCritic(
        local_observation_size=_integer(specification, "local_observation_size"),
        centralized_state_size=_integer(specification, "centralized_state_size"),
        action_size=_integer(specification, "action_size"),
        num_agents=_integer(specification, "num_agents"),
        actor_hidden_sizes=_sizes(specification, "actor_hidden_sizes"),
        critic_hidden_sizes=_sizes(specification, "critic_hidden_sizes"),
        initial_log_standard_deviation=initial_log_standard_deviation,
        neighbor_encoder=(
            _neighbor_encoder(specification.get("neighbor_encoder"))
            if format_version == 2
            else None
        ),
    ).to(device)
    state = _string_mapping(payload.get("state_dict"), name="state_dict")
    if not all(isinstance(value, Tensor) for value in state.values()):
        raise ValueError("checkpoint state_dict values must be tensors.")
    model.load_state_dict(cast(dict[str, Tensor], state))
    metadata = _string_mapping(payload.get("metadata", {}), name="metadata")
    return model, metadata
