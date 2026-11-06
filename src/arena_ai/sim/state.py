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
    # TODO: Rename or generalize this to a support-contact state when platform
    # collisions exist; it should mean supported by any walkable surface.
    on_ground: bool = True
    facing_dir: int = 1
    health: int = 100
    attack_frame: int = 0
    attack_hit: bool = False


def create_initial_fighters(config: GameConfig) -> dict[str, FighterState]:
    """Create the deterministic starting states for the first two-fighter episode."""
    return {
        "agent": FighterState(
            x=config.world.arena_width_m * 0.25,
            facing_dir=1,
            health=config.fighter.max_health,
        ),
        "opponent": FighterState(
            x=config.world.arena_width_m * 0.75,
            facing_dir=-1,
            health=config.fighter.max_health,
        ),
    }
