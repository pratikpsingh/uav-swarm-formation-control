"""Small neural-network construction helpers shared by PPO and MAPPO."""

import math

from torch import nn


def build_mlp(
    input_size: int,
    output_size: int,
    hidden_sizes: tuple[int, ...],
    *,
    output_gain: float,
) -> nn.Sequential:
    """Build an orthogonally initialized tanh multilayer perceptron."""
    if (
        input_size < 1
        or output_size < 1
        or not hidden_sizes
        or any(size < 1 for size in hidden_sizes)
    ):
        raise ValueError("network dimensions must be positive.")
    layers: list[nn.Module] = []
    previous = input_size
    for size in hidden_sizes:
        layer = nn.Linear(previous, size)
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(layer.bias)
        layers.extend((layer, nn.Tanh()))
        previous = size
    output = nn.Linear(previous, output_size)
    nn.init.orthogonal_(output.weight, gain=output_gain)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)
