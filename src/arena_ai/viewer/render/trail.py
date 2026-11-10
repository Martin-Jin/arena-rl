"""Afterimages left behind by a dodge."""

from collections.abc import Mapping
from dataclasses import dataclass, replace

from arena_ai.config import Config
from arena_ai.sim.state import FighterState


@dataclass
class _Ghost:
    fighter_id: str
    fighter: FighterState
    age: float = 0.0


class DodgeTrail:
    """Remembers where a dodging fighter was, so the scene can draw faint copies of it."""

    def __init__(self, config: Config) -> None:
        self.ui = config.viewer.ui
        self._ghosts: list[_Ghost] = []
        self._since_spawn: dict[str, float] = {}

    def reset(self) -> None:
        self._ghosts.clear()
        self._since_spawn.clear()

    def update(self, dt: float, fighters: Mapping[str, FighterState]) -> None:
        """Age the copies, drop the faded ones and leave a new one while a fighter dodges."""
        for ghost in self._ghosts:
            ghost.age += dt
        self._ghosts = [g for g in self._ghosts if g.age < self.ui.dodge_trail_seconds]
        spacing = self.ui.dodge_trail_spacing_seconds
        for fighter_id, fighter in fighters.items():
            if not fighter.dodge_frames_left:
                self._since_spawn.pop(fighter_id, None)
                continue
            # A fresh dodge leaves its first copy at once.
            since = self._since_spawn.get(fighter_id, spacing) + dt
            if since >= spacing:
                # Only the pose matters, so drop the effects that would draw a bubble.
                copy = replace(fighter, parry_frame=0, parry_pose_frame=0, stagger_frames=0)
                self._ghosts.append(_Ghost(fighter_id, copy))
                since = 0.0
            self._since_spawn[fighter_id] = since

    def ghosts(self) -> list[tuple[str, FighterState, float]]:
        """Return (fighter id, copy of its state, strength) with strength fading 1 to 0."""
        seconds = self.ui.dodge_trail_seconds
        return [(g.fighter_id, g.fighter, 1 - g.age / seconds) for g in self._ghosts]
