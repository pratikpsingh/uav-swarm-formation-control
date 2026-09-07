"""Framework-light contracts shared by deployment configuration and models."""

from enum import StrEnum


class DeploymentArchitecture(StrEnum):
    """Actor families compared by the policy-compression study."""

    FEED_FORWARD = "feed-forward"
    GRU = "gru"
    LSTM = "lstm"


__all__ = ["DeploymentArchitecture"]
