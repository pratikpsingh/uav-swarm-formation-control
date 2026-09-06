"""Validated experimental conditions for local neighbor communication."""

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from uav_swarm_control.obstacles import ObstacleScenario

_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class CommunicationRegimenKind(StrEnum):
    """Whether training uses one neighbor condition or a distribution."""

    FIXED = "fixed"
    VARIABLE = "variable"


@dataclass(frozen=True, slots=True)
class CommunicationCondition:
    """One requested-degree, sensing-range, and obstacle combination."""

    name: str
    requested_neighbors: int
    sensing_radius_m: float | None
    obstacle_scenario: ObstacleScenario

    def __post_init__(self) -> None:
        if _NAME.fullmatch(self.name) is None:
            raise ValueError("communication condition name must use lowercase kebab-case.")
        if isinstance(self.requested_neighbors, bool) or self.requested_neighbors < 0:
            raise ValueError("requested_neighbors must be a non-negative integer.")
        if self.sensing_radius_m is not None and (
            not math.isfinite(self.sensing_radius_m) or self.sensing_radius_m <= 0.0
        ):
            raise ValueError("sensing_radius_m must be positive when provided.")
        object.__setattr__(self, "obstacle_scenario", ObstacleScenario(self.obstacle_scenario))


@dataclass(frozen=True, slots=True)
class CommunicationRegimen:
    """Named fixed-condition or variable-condition training distribution."""

    name: str
    kind: CommunicationRegimenKind
    conditions: tuple[CommunicationCondition, ...]

    def __post_init__(self) -> None:
        if _NAME.fullmatch(self.name) is None:
            raise ValueError("communication regimen name must use lowercase kebab-case.")
        kind = CommunicationRegimenKind(self.kind)
        if not self.conditions:
            raise ValueError("communication regimen requires at least one condition.")
        if kind is CommunicationRegimenKind.FIXED and len(self.conditions) != 1:
            raise ValueError("a fixed regimen requires exactly one condition.")
        if kind is CommunicationRegimenKind.VARIABLE and len(self.conditions) < 2:
            raise ValueError("a variable regimen requires at least two conditions.")
        if len({condition.name for condition in self.conditions}) != len(self.conditions):
            raise ValueError("regimen condition names must be unique.")
        object.__setattr__(self, "kind", kind)
