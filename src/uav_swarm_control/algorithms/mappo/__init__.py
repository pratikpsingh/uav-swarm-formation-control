"""Multi-Agent PPO with parameter sharing and centralized training."""

from uav_swarm_control.algorithms.mappo.buffer import (
    FlatMAPPOBatch,
    MAPPOBatch,
    MAPPORolloutBuffer,
)
from uav_swarm_control.algorithms.mappo.checkpoint import (
    load_mappo_checkpoint,
    save_mappo_checkpoint,
)
from uav_swarm_control.algorithms.mappo.evaluation import MAPPOEvaluation, evaluate_mappo
from uav_swarm_control.algorithms.mappo.trainer import (
    EnvironmentFactory,
    MAPPOTrainingResult,
    MAPPOUpdateMetrics,
    train_mappo,
)

__all__ = [
    "EnvironmentFactory",
    "FlatMAPPOBatch",
    "MAPPOBatch",
    "MAPPOEvaluation",
    "MAPPORolloutBuffer",
    "MAPPOTrainingResult",
    "MAPPOUpdateMetrics",
    "evaluate_mappo",
    "load_mappo_checkpoint",
    "save_mappo_checkpoint",
    "train_mappo",
]
