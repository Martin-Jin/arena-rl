# state.py
# This module defines the sprite states that are updated in the game loop.


import random
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


def create_initial_fighters(
    config: GameConfig, rng: random.Random | None = None
) -> dict[str, FighterState]:
    """Create the starting states for a two-fighter episode.

    Without ``rng`` the fighters start at fixed positions, which keeps episodes
    deterministic. With ``rng`` they spawn at random positions facing each other,
    which stays reproducible when the generator is seeded.
    """
    fighters = {
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
    if rng is None:
        return fighters
    margin = config.fighter.width_m / 2
    for fighter in fighters.values():
        fighter.x = rng.uniform(margin, config.world.arena_width_m - margin)
    agent, opponent = fighters["agent"], fighters["opponent"]
    agent.facing_dir = 1 if opponent.x >= agent.x else -1
    opponent.facing_dir = -agent.facing_dir
    return fighters
