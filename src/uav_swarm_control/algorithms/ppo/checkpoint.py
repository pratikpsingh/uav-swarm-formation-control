"""Versioned, atomic policy checkpoint persistence."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from uav_swarm_control.models import ActorCritic

_CHECKPOINT_FORMAT_VERSION = 1


def save_policy_checkpoint(
    path: str | Path,
    model: ActorCritic,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically save model structure, weights, and small provenance metadata."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    payload: dict[str, object] = {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "model": {
            "observation_size": model.observation_size,
            "action_size": model.action_size,
            "hidden_sizes": model.hidden_sizes,
            "initial_log_standard_deviation": model.initial_log_standard_deviation,
        },
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def _required(mapping: Mapping[str, object], key: str) -> object:
    if key not in mapping:
        raise ValueError(f"checkpoint is missing {key!r}.")
    return mapping[key]


def _string_mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {name} must be a mapping.")
    result: dict[str, object] = {}
    for key, item in cast(dict[object, object], value).items():
        if not isinstance(key, str):
            raise ValueError(f"checkpoint {name} keys must be strings.")
        result[key] = item
    return result


def load_policy_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[ActorCritic, dict[str, object]]:
    """Load a project checkpoint using PyTorch's restricted weights-only unpickler."""
    raw = torch.load(Path(path), map_location=device, weights_only=True)
    payload = _string_mapping(cast(object, raw), name="root")
    version = _required(payload, "format_version")
    if version != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {version!r}; expected {_CHECKPOINT_FORMAT_VERSION}."
        )
    raw_specification = _required(payload, "model")
    specification = _string_mapping(raw_specification, name="model specification")
    observation_size = _required(specification, "observation_size")
    action_size = _required(specification, "action_size")
    hidden_sizes = _required(specification, "hidden_sizes")
    initial_log_standard_deviation = _required(
        specification,
        "initial_log_standard_deviation",
    )
    if (
        not isinstance(observation_size, int)
        or not isinstance(action_size, int)
        or not isinstance(hidden_sizes, tuple)
        or not isinstance(initial_log_standard_deviation, float)
    ):
        raise ValueError("checkpoint model specification has invalid types.")
    raw_hidden_sizes = cast(tuple[object, ...], hidden_sizes)
    if not raw_hidden_sizes or not all(isinstance(size, int) for size in raw_hidden_sizes):
        raise ValueError("checkpoint hidden sizes must be positive integers.")
    model = ActorCritic(
        observation_size,
        action_size,
        cast(tuple[int, ...], raw_hidden_sizes),
        initial_log_standard_deviation,
    ).to(device)
    raw_state = _required(payload, "state_dict")
    untyped_state = _string_mapping(raw_state, name="state_dict")
    if not all(isinstance(value, Tensor) for value in untyped_state.values()):
        raise ValueError("checkpoint state_dict values must be tensors.")
    model.load_state_dict(cast(dict[str, Tensor], untyped_state))
    raw_metadata = payload.get("metadata", {})
    return model, _string_mapping(raw_metadata, name="metadata")
