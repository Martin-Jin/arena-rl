"""The fade-away of a fighter that has died."""

from collections.abc import Mapping

from arena_ai.config import Config
from arena_ai.sim.state import FighterState


class DeathFade:
    """Times how long each dead fighter has been dead, so the scene can fade it out."""

    def __init__(self, config: Config) -> None:
        self.seconds = config.viewer.ui.death_fade_seconds
        self._age: dict[str, float] = {}

    def reset(self) -> None:
        self._age.clear()

    def update(self, dt: float, fighters: Mapping[str, FighterState]) -> None:
        """Age every dead fighter. A fighter's first update after dying adds no time."""
        for fighter_id, fighter in fighters.items():
            if fighter.health > 0:
                self._age.pop(fighter_id, None)
            elif fighter_id in self._age:
                self._age[fighter_id] += dt
            else:
                self._age[fighter_id] = 0.0

    def strengths(self) -> dict[str, float]:
        """Return each dead fighter's opacity from 1 (just died) to 0 (gone)."""
        return {
            fighter_id: max(0.0, 1 - age / self.seconds) for fighter_id, age in self._age.items()
        }
