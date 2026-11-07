"""Save and load policy, optimizer, configuration, and training progress."""

from pathlib import Path
from typing import Any

import torch

from arena_ai.config import Config
from arena_ai.training.train import TrainingState


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    state: TrainingState,
) -> None:
    """Save all state needed to resume a training run."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "training_state": state,
        },
        destination,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore a checkpoint and return its serialized metadata."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
