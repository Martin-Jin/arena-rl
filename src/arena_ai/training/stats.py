"""Cheap rolling training statistics: episode outcomes and per-batch behaviour.

Everything here is designed to cost almost nothing next to a simulation step.
Episode statistics keep running sums over a sliding window, so recording an
episode and reading a summary are O(1) however large the window is. Behaviour
statistics are a handful of vectorized operations on the rollout batch that the
update already holds.
"""

from collections import deque
from dataclasses import dataclass

import torch

from arena_ai.agents.action import MOVE_DIRS, decode_parts


@dataclass(frozen=True)
class EpisodeRecord:
    """What one finished episode contributes to the rolling statistics."""

    won: bool
    lost: bool  # The opponent won. A truncated episode is neither won nor lost.
    truncated: bool
    length: int
    episode_return: float
    damage_dealt: int
    damage_taken: int
    swings: int
    hits: int
    edge_steps: int  # Steps spent within the edge margin of a wall.
    parry_attempts: int = 0
    parries: int = 0
    timeout_win: bool = False  # Truncated with the agent ahead on health.


class _Window:
    """Running sums over the most recent ``size`` episodes."""

    _FIELDS = (
        "wins", "losses", "truncated", "return", "length", "win_length",
        "loss_length", "dealt", "taken", "swings", "hits", "edge_steps",
        "parry_attempts", "parries", "timeout_wins",
    )  # fmt: skip

    def __init__(self, size: int) -> None:
        self._size = size
        self._records: deque[EpisodeRecord] = deque()
        self._sums = dict.fromkeys(self._FIELDS, 0.0)

    def _apply(self, record: EpisodeRecord, sign: int) -> None:
        sums = self._sums
        sums["wins"] += sign * record.won
        sums["losses"] += sign * record.lost
        sums["truncated"] += sign * record.truncated
        sums["return"] += sign * record.episode_return
        sums["length"] += sign * record.length
        sums["win_length"] += sign * record.length * record.won
        sums["loss_length"] += sign * record.length * record.lost
        sums["dealt"] += sign * record.damage_dealt
        sums["taken"] += sign * record.damage_taken
        sums["swings"] += sign * record.swings
        sums["hits"] += sign * record.hits
        sums["edge_steps"] += sign * record.edge_steps
        sums["parry_attempts"] += sign * record.parry_attempts
        sums["parries"] += sign * record.parries
        sums["timeout_wins"] += sign * record.timeout_win

    def add(self, record: EpisodeRecord) -> None:
        self._records.append(record)
        self._apply(record, 1)
        if len(self._records) > self._size:
            self._apply(self._records.popleft(), -1)

    def summary(self) -> dict[str, float]:
        count = len(self._records)
        if count == 0:
            return {}
        s = self._sums
        return {
            "episodes": float(count),
            "win_rate": s["wins"] / count,
            "truncation_rate": s["truncated"] / count,
            "timeout_win_rate": s["timeout_wins"] / count,
            "mean_return": s["return"] / count,
            "mean_length": s["length"] / count,
            # How long wins and losses take: shorter wins mean a more decisive agent.
            "win_length": s["win_length"] / s["wins"] if s["wins"] else 0.0,
            "loss_length": s["loss_length"] / s["losses"] if s["losses"] else 0.0,
            "damage_dealt": s["dealt"] / count,
            "damage_taken": s["taken"] / count,
            # Hits per swing started: low accuracy with many swings is spamming.
            "attack_accuracy": s["hits"] / s["swings"] if s["swings"] else 0.0,
            # Parries that stopped an attack per parry started: low means mistimed spam.
            "parry_accuracy": (
                s["parries"] / s["parry_attempts"] if s["parry_attempts"] else 0.0
            ),
            "parries": s["parries"] / count,
            "wall_fraction": s["edge_steps"] / s["length"] if s["length"] else 0.0,
        }


class EpisodeStats:
    """Rolling episode statistics, overall and for each opponent.

    A high truncation rate against an opponent (for example ``idle``) means the
    episode time limit, not the fight, is ending its episodes. Each opponent
    keeps its own window, so rates stay comparable when opponents are not
    sampled equally.
    """

    def __init__(self, window: int = 100) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window
        self._overall = _Window(window)
        self._per_opponent: dict[str, _Window] = {}

    def record(self, opponent: str, record: EpisodeRecord) -> None:
        """Record one finished episode against ``opponent``."""
        self._overall.add(record)
        if opponent not in self._per_opponent:
            self._per_opponent[opponent] = _Window(self.window)
        self._per_opponent[opponent].add(record)

    def overall(self) -> dict[str, float]:
        """Return statistics over the last ``window`` episodes of any opponent."""
        return self._overall.summary()

    def by_opponent(self) -> dict[str, dict[str, float]]:
        """Return statistics keyed by opponent name."""
        return {
            name: window.summary()
            for name, window in sorted(self._per_opponent.items())
        }


def rollout_metrics(
    actions: torch.Tensor, rewards: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return what the agent did in a rollout, as 0-d tensors.

    Kept as tensors so the caller can read every statistic back from the device
    with one transfer. ``actions`` holds flat action indices.
    """
    move, jump, attack, parry, dodge = decode_parts(actions)
    left, right = MOVE_DIRS.index(-1), MOVE_DIRS.index(1)
    return {
        "reward_per_step": rewards.mean(),
        "attack_rate": attack.float().mean(),
        "jump_rate": jump.float().mean(),
        "parry_rate": parry.float().mean(),
        "dodge_rate": dodge.float().mean(),
        "left_rate": (move == left).float().mean(),
        "right_rate": (move == right).float().mean(),
        "idle_rate": (actions == 0).float().mean(),
    }
