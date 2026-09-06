# Deployment

Stage 12 writes generated deployment studies under `artifacts/deployment/`; those outputs remain out
of Git. Each candidate seed contains an actor-only `actor.pt2`, an `artifact.json`, and a `result.json`.
The centralized critic, optimizer, replay data, and Python source are not part of the exported graph.

The `.pt2` file is a portable `torch.export` graph, not a Crazyflie firmware image. Its byte count is a
flash-screening proxy. A real deployment must compile it for a selected ExecuTorch backend, include
required operators/runtime memory in the target budget, and repeat latency, peak-RAM, and energy
measurements on the actual flight controller.

Install the optional quantization dependency with `uv sync --extra deployment`. Never commit trained
weights or generated deployment directories. Commit a reviewed, compact report only after a full
research run.
