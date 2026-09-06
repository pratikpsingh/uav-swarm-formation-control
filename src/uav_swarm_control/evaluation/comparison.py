"""Method-independent protocol fingerprints and guarded result comparison."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

from uav_swarm_control.evaluation.artifacts import save_json_artifact
from uav_swarm_control.evaluation.provenance import stable_digest

if TYPE_CHECKING:
    from uav_swarm_control.configuration.baseline import BaselineConfig

COMMON_METRICS_SCHEMA_VERSION = 1


def comparison_protocol(config: BaselineConfig) -> dict[str, object]:
    """Describe the environment and evaluation schedule shared by all controllers."""
    protocol: dict[str, object] = {
        "common_metrics_schema_version": COMMON_METRICS_SCHEMA_VERSION,
        "physics": asdict(config.physics),
        "evaluation_root_seed": config.evaluation_seed,
        "evaluation_episodes": config.mappo.evaluation_episodes,
        "horizon": config.physics.experiment.environment.max_episode_steps,
    }
    protocol = cast(
        dict[str, object],
        json.loads(json.dumps(protocol, sort_keys=True, allow_nan=False)),
    )
    protocol["fingerprint"] = stable_digest(protocol)
    return protocol


def compare_controller_results(
    mappo_summary_path: Path,
    dmpc_result_path: Path,
    output: Path,
) -> Path:
    """Compare common means only after verifying identical task/evaluation fingerprints."""
    mappo = cast(dict[str, object], json.loads(mappo_summary_path.read_text(encoding="utf-8")))
    dmpc = cast(dict[str, object], json.loads(dmpc_result_path.read_text(encoding="utf-8")))
    fingerprint = mappo.get("comparison_fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != dmpc.get("comparison_fingerprint"):
        raise ValueError("MAPPO and DMPC results use different comparison protocols.")
    mappo_metrics = cast(dict[str, dict[str, float]], mappo.get("metrics"))
    common_summary = cast(dict[str, object], dmpc.get("common_summary"))
    dmpc_metrics = cast(dict[str, dict[str, float]], common_summary.get("metrics"))
    shared = sorted(set(mappo_metrics) & set(dmpc_metrics))
    if not shared:
        raise ValueError("results contain no shared metrics.")
    rows = {
        name: {
            "mappo_mean": mappo_metrics[name]["mean"],
            "mappo_sample_std_across_training_seeds": mappo_metrics[name]["sample_std"],
            "dmpc_mean": dmpc_metrics[name]["mean"],
            "dmpc_sample_std_across_evaluation_episodes": dmpc_metrics[name]["sample_std"],
        }
        for name in shared
    }
    return save_json_artifact(
        output,
        {
            "artifact_schema_version": 1,
            "comparison_fingerprint": fingerprint,
            "warning": (
                "MAPPO uncertainty is across independently trained policies; DMPC uncertainty is "
                "across initial-condition episodes. The standard deviations are not equivalent "
                "sampling units and this artifact makes no significance claim."
            ),
            "mappo_result": str(mappo_summary_path),
            "dmpc_result": str(dmpc_result_path),
            "metrics": rows,
        },
    )
