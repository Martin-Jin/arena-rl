"""AI controllers that pick actions for fighters in the viewer.

The human player is not a controller: the match reads the keyboard directly.
Torch is imported only when a trained agent is requested, so the viewer starts
quickly and still works without torch when scripted opponents are chosen.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from arena_ai.agents.scripted import OPPONENTS
from arena_ai.agents.scripted.view import team_observation
from arena_ai.config import Config
from arena_ai.sim.observation import get_observation
from arena_ai.sim.observation_vector import encode_observation
from arena_ai.sim.state import AGENT_ID, FighterState, enemies_of, is_opponent

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINED = "trained"


class Controller(Protocol):
    """Chooses a flat action index for one fighter from the full game state."""

    def action(self, fighters: Mapping[str, FighterState]) -> int: ...


def checkpoint_path(config: Config) -> Path:
    """Return the configured trained-agent checkpoint location."""
    return REPO_ROOT / config.viewer.opponent.checkpoint_path


def list_checkpoints(config: Config) -> list[Path]:
    """Return the ``.pt`` files next to the default checkpoint.

    The configured default comes first, then the most recently saved, so a
    checkpoint renamed to keep it (for example ``best.pt``) is still found.
    """
    default = checkpoint_path(config)
    if not default.parent.is_dir():
        return []
    files = [path for path in default.parent.glob("*.pt") if path.is_file()]
    return sorted(files, key=lambda path: (path != default, -path.stat().st_mtime))


def resolve_checkpoint(config: Config, name: str = "") -> Path | None:
    """Return the checkpoint to load, or None when there is none.

    ``name`` is a file in the checkpoint folder. An empty or unknown name means
    automatic: the configured default if it exists, else the newest ``.pt`` file.
    """
    default = checkpoint_path(config)
    if name and Path(name).name == name:  # A bare file name, never a path.
        chosen = default.parent / name
        if chosen.is_file():
            return chosen
    files = list_checkpoints(config)
    return files[0] if files else None


def trained_available(config: Config) -> bool:
    """Return whether any trained-agent checkpoint exists."""
    return resolve_checkpoint(config) is not None


class ScriptedController:
    """A scripted opponent. Scripted policies play one of the opponents' sides."""

    def __init__(self, name: str, config: Config, fighter_id: str) -> None:
        self.policy = OPPONENTS[name]
        self.config = config
        self.fighter_id = fighter_id

    def action(self, fighters: Mapping[str, FighterState]) -> int:
        # Scripted policies read the learner-first observation used in training: the agent,
        # this fighter, then its living allies.
        return self.policy(team_observation(fighters, self.fighter_id, self.config), self.config)


class TrainedController:
    """The trained network playing either side, seeing the game from that side."""

    def __init__(
        self, config: Config, greedy: bool, fighter_id: str, checkpoint: str = ""
    ) -> None:
        import torch

        from arena_ai.agents.distribution import (
            FactoredCategorical,
            sample_with_log_prob,
        )
        from arena_ai.agents.policy import ActorCritic
        from arena_ai.training.checkpoints import load_checkpoint

        # A tiny network evaluated once per frame: one thread is fastest.
        torch.set_num_threads(1)
        self._torch = torch
        self._categorical = FactoredCategorical
        self._sample = sample_with_log_prob
        self.config = config
        self.greedy = greedy
        self.fighter_id = fighter_id
        self.model = ActorCritic(config)
        path = resolve_checkpoint(config, checkpoint)
        if path is None:
            raise FileNotFoundError(f"no checkpoint in {checkpoint_path(config).parent}")
        load_checkpoint(path, self.model)
        self.model.eval()

    def action(self, fighters: Mapping[str, FighterState]) -> int:
        enemies = enemies_of(self.fighter_id, fighters)
        structured = get_observation(fighters[self.fighter_id], enemies, self.config)
        observation = self._torch.tensor(
            encode_observation(structured, self.config), dtype=self._torch.float32
        ).unsqueeze(0)
        with self._torch.no_grad():
            logits = self.model.policy_network(observation)
        if self.greedy:
            return int(self._categorical(logits).mode().item())
        return int(self._sample(logits)[0].item())


def make_controller(
    kind: str, config: Config, greedy: bool, fighter_id: str, checkpoint: str = ""
) -> Controller:
    """Build the controller for a scripted opponent name or ``"trained"``.

    ``checkpoint`` names the file the trained agent loads ("" for automatic).
    """
    if kind == TRAINED:
        return TrainedController(config, greedy, fighter_id, checkpoint)
    if kind in OPPONENTS:
        if not is_opponent(fighter_id):
            raise ValueError("scripted policies can only control an opponent")
        return ScriptedController(kind, config, fighter_id)
    raise ValueError(f"Unknown controller: {kind}")
