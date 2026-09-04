"""Tests for deterministic and isolated random streams."""

import numpy as np
import pytest

from uav_swarm_control.seeding import RandomStream, derive_seed, make_rng, validate_seed


def test_same_root_and_stream_reproduce_identical_sequences() -> None:
    first = make_rng(42, RandomStream.ENVIRONMENT).normal(size=5)
    second = make_rng(42, RandomStream.ENVIRONMENT).normal(size=5)

    np.testing.assert_array_equal(first, second)


def test_named_streams_are_deterministic_and_distinct() -> None:
    environment_seed = derive_seed(42, RandomStream.ENVIRONMENT)
    policy_seed = derive_seed(42, RandomStream.POLICY)

    assert environment_seed == derive_seed(42, RandomStream.ENVIRONMENT)
    assert environment_seed != policy_seed


@pytest.mark.parametrize("value", [-1, 2**64, True])
def test_seed_validation_rejects_invalid_values(value: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_seed(value)
