"""Framework-light contracts shared by deployment configuration and models."""

from enum import StrEnum


class DeploymentArchitecture(StrEnum):
    """Actor families compared by Stage 12."""

    FEED_FORWARD = "feed-forward"
    GRU = "gru"
    LSTM = "lstm"


__all__ = ["DeploymentArchitecture"]
