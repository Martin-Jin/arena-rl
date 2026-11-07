"""Rolling per-opponent episode statistics for training diagnostics."""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class _EpisodeResult:
    won: bool
    truncated: bool
    length: int


class OpponentStats:
    """Track win rate, truncation rate and mean length over recent episodes.

    Each opponent keeps its own window of the last ``window`` episodes, so a
    rate is comparable between opponents even when they are not sampled equally.
    A high truncation rate against an opponent (for example ``idle``) means the
    episode time limit, not the fight, is ending its episodes.
    """

    def __init__(self, window: int = 100) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window
        self._results: dict[str, deque[_EpisodeResult]] = {}

    def record(
        self, opponent: str, *, won: bool, truncated: bool, length: int
    ) -> None:
        """Record one finished episode against ``opponent``."""
        results = self._results.setdefault(opponent, deque(maxlen=self.window))
        results.append(_EpisodeResult(won, truncated, length))

    def summary(self) -> dict[str, dict[str, float]]:
        """Return rolling statistics keyed by opponent name."""
        summary: dict[str, dict[str, float]] = {}
        for opponent, results in sorted(self._results.items()):
            count = len(results)
            summary[opponent] = {
                "episodes": float(count),
                "win_rate": sum(r.won for r in results) / count,
                "truncation_rate": sum(r.truncated for r in results) / count,
                "mean_length": sum(r.length for r in results) / count,
            }
        return summary
