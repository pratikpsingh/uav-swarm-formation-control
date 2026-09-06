"""Portable actor artifact and isolated host benchmark tests."""

from pathlib import Path

import pytest

from uav_swarm_control.deployment.artifacts import (
    benchmark_exported_actor,
    export_actor,
    quantize_int8_dynamic,
)
from uav_swarm_control.deployment.contracts import DeploymentArchitecture
from uav_swarm_control.deployment.models import (
    CompactFeedForwardActor,
    CompactGRUActor,
    CompactLSTMActor,
)
from uav_swarm_control.models import NeighborEncoderSpec


def _model() -> CompactFeedForwardActor:
    return CompactFeedForwardActor(10, 3, 8, 1, NeighborEncoderSpec(2, 2, 2, 4, (4,))).eval()


def test_export_is_actor_only_and_benchmark_is_labeled_as_host(tmp_path: Path) -> None:
    artifact, metadata = export_actor(
        _model(),
        tmp_path / "actor",
        architecture=DeploymentArchitecture.FEED_FORWARD,
        observation_size=10,
        action_size=3,
        recurrent_width=0,
        candidate="ff-8",
        quantization="none",
        structured_zero_channel_fraction=0.0,
    )
    benchmark = benchmark_exported_actor(
        artifact,
        artifact.parent / "artifact.json",
        warmup_iterations=1,
        measured_iterations=3,
        torch_threads=1,
    )
    assert metadata["contains_centralized_critic"] is False
    assert metadata["artifact_bytes"] == artifact.stat().st_size
    assert benchmark["platform"] == "host-cpu-python-process"
    latency = benchmark["latency_ms_p95"]
    assert isinstance(latency, (int, float))
    assert latency >= 0.0


def test_torchao_int8_actor_can_be_exported(tmp_path: Path) -> None:
    pytest.importorskip("torchao")
    model = quantize_int8_dynamic(_model())
    artifact, metadata = export_actor(
        model,
        tmp_path / "int8",
        architecture=DeploymentArchitecture.FEED_FORWARD,
        observation_size=10,
        action_size=3,
        recurrent_width=0,
        candidate="ff-8-int8",
        quantization="int8-dynamic",
        structured_zero_channel_fraction=0.0,
    )
    assert artifact.is_file()
    assert metadata["quantization"] == "int8-dynamic"


@pytest.mark.parametrize(
    ("architecture", "model"),
    [
        (
            DeploymentArchitecture.GRU,
            CompactGRUActor(10, 3, 6, NeighborEncoderSpec(2, 2, 2, 4, (4,))),
        ),
        (
            DeploymentArchitecture.LSTM,
            CompactLSTMActor(10, 3, 6, NeighborEncoderSpec(2, 2, 2, 4, (4,))),
        ),
    ],
)
def test_recurrent_actor_state_contract_exports(
    tmp_path: Path, architecture: DeploymentArchitecture, model: CompactGRUActor | CompactLSTMActor
) -> None:
    artifact, metadata = export_actor(
        model.eval(),
        tmp_path / architecture.value,
        architecture=architecture,
        observation_size=10,
        action_size=3,
        recurrent_width=6,
        candidate=architecture.value,
        quantization="none",
        structured_zero_channel_fraction=0.0,
    )
    assert artifact.is_file()
    assert metadata["export_maximum_absolute_error"] == 0.0
