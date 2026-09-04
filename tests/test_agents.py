"""Tests for stable agent identities."""

import pytest

from uav_swarm_control.agents import AgentId, sequential_agent_ids


def test_sequential_agent_ids_are_stable_and_ordered() -> None:
    assert sequential_agent_ids(3) == (AgentId(0), AgentId(1), AgentId(2))
    assert str(AgentId(2)) == "agent_2"


@pytest.mark.parametrize("value", [-1, True])
def test_agent_id_rejects_invalid_values(value: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        AgentId(value)


def test_sequential_agent_ids_requires_an_agent() -> None:
    with pytest.raises(ValueError, match="at least one"):
        sequential_agent_ids(0)
