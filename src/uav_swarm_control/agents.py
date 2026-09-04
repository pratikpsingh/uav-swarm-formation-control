"""Stable identities for agents in a swarm."""

from dataclasses import dataclass


def _non_negative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


@dataclass(frozen=True, slots=True, order=True)
class AgentId:
    """A validated, stable agent identifier."""

    value: int

    def __post_init__(self) -> None:
        value = _non_negative_integer(self.value, name="agent identifier")
        object.__setattr__(self, "value", value)

    def __str__(self) -> str:
        return f"agent_{self.value}"


def sequential_agent_ids(num_agents: int) -> tuple[AgentId, ...]:
    """Create deterministic identifiers in ascending order."""
    count = _non_negative_integer(num_agents, name="num_agents")
    if count < 1:
        raise ValueError("num_agents must be at least one.")
    return tuple(AgentId(index) for index in range(count))
