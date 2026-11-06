# state.py
# This module defines the sprite states that are updated in the game loop.


from dataclasses import dataclass

from arena_ai.config import GameConfig


@dataclass
class FighterState:
    x: float = 6.0  # meters
    y: float = 0.0
    vx: float = 0.0  # m/s
    vy: float = 0.0
    on_ground: bool = True


def create_initial_fighters(config: GameConfig) -> dict[str, FighterState]:
    """Create the deterministic starting states for the first two-fighter episode."""
    return {
        "agent": FighterState(x=config.world.arena_width_m * 0.25),
        "opponent": FighterState(x=config.world.arena_width_m * 0.75),
    }
