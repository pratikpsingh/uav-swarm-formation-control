"""Framework-light contracts shared by deployment configuration and models."""

from enum import StrEnum


class DeploymentArchitecture(StrEnum):
    """Actor families compared by the policy-compression study."""

    FEED_FORWARD = "feed-forward"
    GRU = "gru"
    LSTM = "lstm"


class DeploymentActionProfile(StrEnum):
    """How bounded actor outputs are converted to normalized velocity commands."""

    NORMALIZED_VELOCITY = "normalized-velocity"
    DIRECTION_SPEED = "direction-speed"


__all__ = ["DeploymentActionProfile", "DeploymentArchitecture"]
