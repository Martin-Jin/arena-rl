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
            "signature": getattr(model, "signature", None),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "training_state": state,
        },
        destination,
    )


def _check_compatible(path: str | Path, checkpoint: dict[str, Any], model) -> None:
    """Raise a ``ValueError`` that says what differs if the checkpoint does not fit."""
    expected = getattr(model, "signature", None)
    saved = checkpoint.get("signature")
    if expected is None:
        return
    if saved is None:
        raise ValueError(
            f"{path} was saved before observation spec {expected['spec_version']} "
            "(the attention network). Its weights no longer fit: train a new checkpoint."
        )
    differences = [
        f"{key}: checkpoint {saved.get(key)!r}, current {value!r}"
        for key, value in expected.items()
        if saved.get(key) != value
    ]
    if differences:
        raise ValueError(
            f"{path} does not match the current observation or network ({'; '.join(differences)})."
            " Train a new checkpoint, or restore the old settings in configs/."
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
    _check_compatible(path, checkpoint, model)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
