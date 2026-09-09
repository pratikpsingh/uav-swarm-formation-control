"""Explicit construction-hold-navigation mission state machine."""

from dataclasses import dataclass
from enum import StrEnum


class MissionPhase(StrEnum):
    CONSTRUCTING = "constructing"
    HOLDING = "holding"
    NAVIGATING = "navigating"
    COMPLETE = "complete"


@dataclass(slots=True)
class ConstructionMission:
    """Track distinct formation construction, stability hold, and navigation phases."""

    construction_hold_steps: int
    phase: MissionPhase = MissionPhase.CONSTRUCTING
    construction_steps: int = 0
    holding_steps: int = 0
    navigation_steps: int = 0

    def __post_init__(self) -> None:
        if self.construction_hold_steps < 1:
            raise ValueError("construction_hold_steps must be positive.")

    def update(self, *, formation_ready: bool, navigation_complete: bool = False) -> MissionPhase:
        """Advance deterministically while retaining per-phase elapsed-step counts."""
        if self.phase is MissionPhase.CONSTRUCTING:
            self.construction_steps += 1
            if formation_ready:
                self.phase = MissionPhase.HOLDING
        elif self.phase is MissionPhase.HOLDING:
            if formation_ready:
                self.holding_steps += 1
            else:
                self.phase = MissionPhase.CONSTRUCTING
                self.holding_steps = 0
            if self.holding_steps >= self.construction_hold_steps:
                self.phase = MissionPhase.NAVIGATING
        elif self.phase is MissionPhase.NAVIGATING:
            self.navigation_steps += 1
            if navigation_complete:
                self.phase = MissionPhase.COMPLETE
        return self.phase
