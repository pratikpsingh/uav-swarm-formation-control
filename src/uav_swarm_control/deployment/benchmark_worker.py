"""Fresh-process benchmark worker; invoke through deployment.artifacts."""

import argparse
import json
import resource
from pathlib import Path
from time import perf_counter_ns
from typing import cast

import numpy as np
import torch

from uav_swarm_control.deployment.contracts import DeploymentArchitecture


def _peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB. Stage 12 records its platform explicitly.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _inputs(metadata: dict[str, object]) -> tuple[torch.Tensor, ...]:
    contract = cast(dict[str, object], metadata["input_contract"])
    architecture = DeploymentArchitecture(cast(str, metadata["architecture"]))
    observation_size = cast(int, contract["observation_size"])
    observations = (
        torch.zeros(1, observation_size)
        if architecture is DeploymentArchitecture.FEED_FORWARD
        else torch.zeros(1, 1, observation_size)
    )
    if architecture is DeploymentArchitecture.FEED_FORWARD:
        return (observations,)
    width = cast(int, contract["recurrent_width"])
    hidden = torch.zeros(1, 1, width)
    if architecture is DeploymentArchitecture.GRU:
        return observations, hidden
    return observations, hidden, torch.zeros_like(hidden)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--torch-threads", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    baseline_rss = _peak_rss_bytes()
    metadata = cast(dict[str, object], json.loads(args.metadata.read_text(encoding="utf-8")))
    model = torch.export.load(args.artifact).module()
    inputs = _inputs(metadata)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(*inputs)
        durations: list[int] = []
        for _ in range(args.iterations):
            started = perf_counter_ns()
            model(*inputs)
            durations.append(perf_counter_ns() - started)
    milliseconds = np.asarray(durations, dtype=np.float64) / 1_000_000.0
    peak_rss = _peak_rss_bytes()
    print(
        json.dumps(
            {
                "platform": "host-cpu-python-process",
                "batch_agents": 1,
                "warmup_iterations": args.warmup,
                "measured_iterations": args.iterations,
                "torch_threads": args.torch_threads,
                "latency_ms_mean": float(milliseconds.mean()),
                "latency_ms_p50": float(np.percentile(milliseconds, 50)),
                "latency_ms_p95": float(np.percentile(milliseconds, 95)),
                "process_rss_before_load_bytes": baseline_rss,
                "peak_process_rss_bytes": peak_rss,
                "incremental_peak_rss_upper_bound_bytes": max(0, peak_rss - baseline_rss),
                "warning": "Host Python RSS and latency are not target-device measurements.",
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
