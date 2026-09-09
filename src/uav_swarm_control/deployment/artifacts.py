"""Actor-only INT8 conversion, portable export, and isolated host measurement."""

import copy
import hashlib
import importlib
import json
import subprocess
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from uav_swarm_control.deployment.contracts import (
    DeploymentActionProfile,
    DeploymentArchitecture,
)
from uav_swarm_control.evaluation.artifacts import save_json_artifact


def count_parameters(model: nn.Module) -> int:
    """Count only this actor's parameters; callers must not pass the training critic."""
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_storage_bytes(model: nn.Module) -> int:
    """Logical tensor bytes, separate from serialized artifact and runtime overhead."""
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def quantize_int8_dynamic(model: nn.Module) -> nn.Module:
    """Apply TorchAO dynamic activation/per-channel weight INT8 quantization."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Deprecation: PlainLayout is deprecated.*",
                module=r"torchao\.dtypes\.utils",
            )
            module = importlib.import_module("torchao.quantization")
            config_type = cast(
                Callable[[], object],
                module.Int8DynamicActivationInt8WeightConfig,
            )
            quantize = cast(Callable[[nn.Module, object], None], module.quantize_)

            quantized = copy.deepcopy(model).eval()
            quantize(quantized, config_type())
    except ImportError as error:
        raise RuntimeError("INT8 candidates require `uv sync --extra deployment`.") from error
    return quantized


def _example_inputs(
    architecture: DeploymentArchitecture, observation_size: int, recurrent_width: int
) -> tuple[Tensor, ...]:
    observations = (
        torch.zeros(1, observation_size)
        if architecture is DeploymentArchitecture.FEED_FORWARD
        else torch.zeros(1, 1, observation_size)
    )
    if architecture is DeploymentArchitecture.FEED_FORWARD:
        return (observations,)
    hidden = torch.zeros(1, 1, recurrent_width)
    if architecture is DeploymentArchitecture.GRU:
        return observations, hidden
    return observations, hidden, torch.zeros_like(hidden)


def export_actor(
    model: nn.Module,
    directory: Path,
    *,
    architecture: DeploymentArchitecture,
    observation_size: int,
    action_size: int,
    recurrent_width: int,
    candidate: str,
    quantization: str,
    structured_zero_channel_fraction: float,
    action_profile: DeploymentActionProfile = DeploymentActionProfile.NORMALIZED_VELOCITY,
) -> tuple[Path, dict[str, object]]:
    """Export an actor graph and verify the saved graph reproduces the source output."""
    directory.mkdir(parents=True, exist_ok=False)
    inputs = _example_inputs(architecture, observation_size, recurrent_width)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(r"The tensor attributes self\..*_flat_weights.*were assigned during export.*"),
            module="contextlib",
        )
        warnings.filterwarnings(
            "ignore",
            message="Deprecation: .* is deprecated.*",
            module=r"torchao\.dtypes\..*",
        )
        with torch.no_grad():
            source = model(*inputs)
            program = torch.export.export(model.eval(), inputs)
        artifact = directory / "actor.pt2"
        torch.export.save(program, artifact)
        loaded = torch.export.load(artifact).module()
        with torch.no_grad():
            restored = loaded(*inputs)
    source_value = cast(object, source)
    restored_value = cast(object, restored)
    source_action: object = (
        cast(tuple[object, ...], source_value)[0]
        if isinstance(source_value, tuple)
        else source_value
    )
    restored_action: object = (
        cast(tuple[object, ...], restored_value)[0]
        if isinstance(restored_value, tuple)
        else restored_value
    )
    if not isinstance(source_action, Tensor) or not isinstance(restored_action, Tensor):
        raise RuntimeError("actor export must return an action tensor first.")
    maximum_error = float((source_action - restored_action).abs().max().item())
    if maximum_error > 1e-5:
        raise RuntimeError("exported actor does not reproduce the in-memory actor.")
    recurrent_state_tensors = {
        DeploymentArchitecture.FEED_FORWARD: 0,
        DeploymentArchitecture.GRU: 1,
        DeploymentArchitecture.LSTM: 2,
    }[architecture]
    metadata: dict[str, object] = {
        "artifact_schema_version": 1,
        "candidate": candidate,
        "architecture": architecture.value,
        "quantization": quantization,
        "action_profile": DeploymentActionProfile(action_profile).value,
        "structured_zero_channel_fraction": structured_zero_channel_fraction,
        "contains_centralized_critic": False,
        "format": "torch.export ExportedProgram",
        "target_runtime_status": (
            "portable graph only; compile and benchmark with a target-specific ExecuTorch "
            "backend before claiming embedded deployment"
        ),
        "input_contract": {
            "batch_agents": 1,
            "sequence_steps": 1
            if architecture is not DeploymentArchitecture.FEED_FORWARD
            else None,
            "observation_size": observation_size,
            "action_size": action_size,
            "recurrent_width": (
                recurrent_width if architecture is not DeploymentArchitecture.FEED_FORWARD else None
            ),
        },
        "parameter_count": count_parameters(model),
        "logical_parameter_bytes": parameter_storage_bytes(model),
        "logical_recurrent_state_bytes_per_agent": (recurrent_state_tensors * recurrent_width * 4),
        "artifact_bytes": artifact.stat().st_size,
        "flash_proxy_bytes": artifact.stat().st_size,
        "export_maximum_absolute_error": maximum_error,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "pytorch_version": torch.__version__,
    }
    save_json_artifact(directory / "artifact.json", metadata)
    return artifact, metadata


def benchmark_exported_actor(
    artifact: Path,
    metadata_path: Path,
    *,
    warmup_iterations: int,
    measured_iterations: int,
    torch_threads: int,
) -> dict[str, object]:
    """Measure one artifact in a fresh process so peak RSS is not cross-contaminated."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "uav_swarm_control.deployment.benchmark_worker",
            "--artifact",
            str(artifact),
            "--metadata",
            str(metadata_path),
            "--warmup",
            str(warmup_iterations),
            "--iterations",
            str(measured_iterations),
            "--torch-threads",
            str(torch_threads),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, object], json.loads(completed.stdout))


__all__ = [
    "benchmark_exported_actor",
    "count_parameters",
    "export_actor",
    "parameter_storage_bytes",
    "quantize_int8_dynamic",
]
