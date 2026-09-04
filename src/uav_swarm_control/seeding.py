"""Deterministic, independent random-number streams."""

from enum import IntEnum

import numpy as np


class RandomStream(IntEnum):
    """Stable namespaces for independent sources of randomness."""

    ENVIRONMENT = 0
    INITIAL_STATE = 1
    ACTION_SAMPLING = 2
    POLICY = 3


def validate_seed(value: object) -> int:
    """Validate a portable unsigned 64-bit root seed."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("seed must be an integer.")
    if not 0 <= value < 2**64:
        raise ValueError("seed must be in the range [0, 2**64).")
    return value


def derive_seed(root_seed: int, stream: RandomStream) -> int:
    """Derive a repeatable seed whose value depends on the stream namespace."""
    root = validate_seed(root_seed)
    sequence = np.random.SeedSequence([root, int(stream)])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def make_rng(root_seed: int, stream: RandomStream) -> np.random.Generator:
    """Create an isolated NumPy generator for one named stream."""
    return np.random.default_rng(derive_seed(root_seed, stream))
