"""Versioned persistence for paper-aligned recurrent MAPPO checkpoints."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from uav_swarm_control.models import NeighborEncoderSpec
from uav_swarm_control.models.recurrent_actor_critic import PaperRecurrentActorCritic

_FORMAT_VERSION = 1


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {name} must be a mapping.")
    result: dict[str, object] = {}
    for key, item in cast(dict[object, object], value).items():
        if not isinstance(key, str):
            raise ValueError(f"checkpoint {name} keys must be strings.")
        result[key] = item
    return result


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"checkpoint {key} must be a positive integer.")
    return value


def _neighbor_spec(value: object) -> NeighborEncoderSpec | None:
    if value is None:
        return None
    values = _mapping(value, name="neighbor_encoder")
    hidden = values.get("hidden_sizes")
    if not isinstance(hidden, tuple):
        raise ValueError("checkpoint neighbor hidden_sizes must be a tuple.")
    sizes = cast(tuple[object, ...], hidden)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in sizes):
        raise ValueError("checkpoint neighbor hidden_sizes must contain integers.")
    return NeighborEncoderSpec(
        _integer(values, "ego_features"),
        _integer(values, "max_neighbors"),
        _integer(values, "neighbor_features"),
        _integer(values, "embedding_size"),
        cast(tuple[int, ...], sizes),
    )


def save_recurrent_mappo_checkpoint(
    path: str | Path,
    model: PaperRecurrentActorCritic,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically save architecture, parameters, and non-executable metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    neighbor = model.neighbor_encoder
    record = {
        "format_version": _FORMAT_VERSION,
        "model_family": model.model_family,
        "model": {
            "local_observation_size": model.local_observation_size,
            "centralized_state_size": model.centralized_state_size,
            "action_size": model.action_size,
            "feature_size": model.feature_size,
            "recurrent_hidden_size": model.recurrent_hidden_size,
            "recurrent_layers": model.recurrent_layers,
            "initial_log_standard_deviation": model.initial_log_standard_deviation,
            "neighbor_encoder": (
                None
                if neighbor is None
                else {
                    "ego_features": neighbor.ego_features,
                    "max_neighbors": neighbor.max_neighbors,
                    "neighbor_features": neighbor.neighbor_features,
                    "embedding_size": neighbor.embedding_size,
                    "hidden_sizes": neighbor.hidden_sizes,
                }
            ),
        },
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    try:
        torch.save(record, temporary)
        temporary.replace(output)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return output


def load_recurrent_mappo_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[PaperRecurrentActorCritic, dict[str, object]]:
    """Load only the declared recurrent model family with weights-only unpickling."""
    try:
        raw: object = torch.load(path, map_location=device, weights_only=True)
    except OSError:
        raise
    record = _mapping(raw, name="root")
    if record.get("format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported recurrent checkpoint format version.")
    if record.get("model_family") != PaperRecurrentActorCritic.model_family:
        raise ValueError("checkpoint is not a paper-recurrent-mappo model.")
    specification = _mapping(record.get("model"), name="model")
    initial_std = specification.get("initial_log_standard_deviation")
    if isinstance(initial_std, bool) or not isinstance(initial_std, (int, float)):
        raise ValueError("checkpoint initial_log_standard_deviation must be numeric.")
    model = PaperRecurrentActorCritic(
        _integer(specification, "local_observation_size"),
        _integer(specification, "centralized_state_size"),
        action_size=_integer(specification, "action_size"),
        feature_size=_integer(specification, "feature_size"),
        recurrent_hidden_size=_integer(specification, "recurrent_hidden_size"),
        recurrent_layers=_integer(specification, "recurrent_layers"),
        initial_log_standard_deviation=float(initial_std),
        neighbor_encoder=_neighbor_spec(specification.get("neighbor_encoder")),
    ).to(device)
    state_dict = record.get("state_dict")
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in cast(dict[object, object], state_dict).items()
    ):
        raise ValueError("checkpoint state_dict is invalid.")
    model.load_state_dict(cast(dict[str, Tensor], state_dict))
    metadata = _mapping(record.get("metadata", {}), name="metadata")
    return model, metadata


__all__ = ["load_recurrent_mappo_checkpoint", "save_recurrent_mappo_checkpoint"]
