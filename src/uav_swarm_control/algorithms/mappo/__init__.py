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
from uav_swarm_control.algorithms.mappo.recurrent_buffer import (
    RecurrentMAPPORolloutBuffer,
    RecurrentSequenceBatch,
)
from uav_swarm_control.algorithms.mappo.recurrent_checkpoint import (
    load_recurrent_mappo_checkpoint,
    save_recurrent_mappo_checkpoint,
)
from uav_swarm_control.algorithms.mappo.recurrent_evaluation import (
    RecurrentMAPPOEvaluation,
    evaluate_recurrent_mappo,
)
from uav_swarm_control.algorithms.mappo.recurrent_trainer import (
    EnvironmentFactory as RecurrentEnvironmentFactory,
)
from uav_swarm_control.algorithms.mappo.recurrent_trainer import (
    RecurrentMAPPOTrainingResult,
    RecurrentMAPPOUpdateMetrics,
    train_recurrent_mappo,
)
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
    "RecurrentEnvironmentFactory",
    "RecurrentMAPPOEvaluation",
    "RecurrentMAPPORolloutBuffer",
    "RecurrentMAPPOTrainingResult",
    "RecurrentMAPPOUpdateMetrics",
    "RecurrentSequenceBatch",
    "evaluate_mappo",
    "evaluate_recurrent_mappo",
    "load_mappo_checkpoint",
    "load_recurrent_mappo_checkpoint",
    "save_mappo_checkpoint",
    "save_recurrent_mappo_checkpoint",
    "train_mappo",
    "train_recurrent_mappo",
]
