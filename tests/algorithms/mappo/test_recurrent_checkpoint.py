"""Round-trip tests for the isolated recurrent checkpoint family."""

from pathlib import Path

import torch

from uav_swarm_control.algorithms.mappo.recurrent_checkpoint import (
    load_recurrent_mappo_checkpoint,
    save_recurrent_mappo_checkpoint,
)
from uav_swarm_control.models.recurrent_actor_critic import PaperRecurrentActorCritic


def test_recurrent_checkpoint_round_trip(tmp_path: Path) -> None:
    model = PaperRecurrentActorCritic(7, 12, feature_size=8, recurrent_hidden_size=6)
    state = model.initial_actor_state(2, device=torch.device("cpu"))
    inputs = torch.randn(3, 2, 7)
    masks = torch.ones(3, 2, dtype=torch.bool)
    expected, _ = model.deterministic_actions(inputs, state, masks)
    path = tmp_path / "recurrent.pt"

    save_recurrent_mappo_checkpoint(path, model, metadata={"steps": 128})
    loaded, metadata = load_recurrent_mappo_checkpoint(path)
    actual, _ = loaded.deterministic_actions(
        inputs, loaded.initial_actor_state(2, device=torch.device("cpu")), masks
    )

    torch.testing.assert_close(actual, expected)
    assert metadata == {"steps": 128}
