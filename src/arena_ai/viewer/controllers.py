"""AI controllers that pick actions for fighters in the viewer.

The human player is not a controller: the match reads the keyboard directly.
Torch is imported only when a trained agent is requested, so the viewer starts
quickly and still works without torch when scripted opponents are chosen.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from arena_ai.agents.scripted import OPPONENTS
from arena_ai.config import Config
from arena_ai.sim.observation import get_observation
from arena_ai.sim.observation_vector import encode_observation
from arena_ai.sim.state import FighterState

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINED = "trained"


class Controller(Protocol):
    """Chooses a flat action index for one fighter from the full game state."""

    def action(self, fighters: Mapping[str, FighterState]) -> int: ...


def checkpoint_path(config: Config) -> Path:
    """Return the configured trained-agent checkpoint location."""
    return REPO_ROOT / config.viewer.opponent.checkpoint_path


def trained_available(config: Config) -> bool:
    """Return whether a trained-agent checkpoint exists."""
    return checkpoint_path(config).is_file()


class ScriptedController:
    """A scripted opponent. Scripted policies play the ``"opponent"`` side."""

    def __init__(self, name: str, config: Config) -> None:
        self.policy = OPPONENTS[name]
        self.config = config

    def action(self, fighters: Mapping[str, FighterState]) -> int:
        # Scripted policies read the learner-first observation used in training.
        structured = get_observation(
            fighters["agent"], {"opponent": fighters["opponent"]}, self.config
        )
        return self.policy(encode_observation(structured, self.config), self.config)


class TrainedController:
    """The trained network playing either side, seeing the game from that side."""

    def __init__(self, config: Config, greedy: bool, fighter_id: str) -> None:
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
        load_checkpoint(checkpoint_path(config), self.model)
        self.model.eval()

    def action(self, fighters: Mapping[str, FighterState]) -> int:
        others = {fid: f for fid, f in fighters.items() if fid != self.fighter_id}
        structured = get_observation(fighters[self.fighter_id], others, self.config)
        observation = self._torch.tensor(
            encode_observation(structured, self.config), dtype=self._torch.float32
        ).unsqueeze(0)
        with self._torch.no_grad():
            logits = self.model.policy_network(observation)
        if self.greedy:
            return int(self._categorical(logits).mode().item())
        return int(self._sample(logits)[0].item())


def make_controller(
    kind: str, config: Config, greedy: bool, fighter_id: str
) -> Controller:
    """Build the controller for a scripted opponent name or ``"trained"``."""
    if kind == TRAINED:
        return TrainedController(config, greedy, fighter_id)
    if kind in OPPONENTS:
        if fighter_id != "opponent":
            raise ValueError("scripted policies can only control the opponent")
        return ScriptedController(kind, config)
    raise ValueError(f"Unknown controller: {kind}")
