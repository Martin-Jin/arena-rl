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
    attack_counter: bool = False  # The current swing is a counter (shorter wind-up).
    lunge_dir: int = 0  # Direction of the counter lunge, fixed when the swing starts.
    parry_frame: int = 0  # 0 when not parrying, else 1 at the press and counting up.
    parry_window: int = 0  # Protected frames of the current parry.
    parry_count: int = 0  # Recent parry presses, which shrink the next window.
    parry_idle_frames: int = 0  # Frames since the last parry, capped at the reset time.
    parry_held: bool = False  # Parry input of the previous step, so holding parries once.
    # Visual only: the parry animation keeps playing after a successful parry, which ends
    # the parry itself so the fighter can counter at once. This is the frame it is on.
    parry_pose_frame: int = 0
    stagger_frames: int = 0  # Frames left in which a parried fighter cannot attack.
    # The normal-swing frame the parried swing had reached, for the deflect animation.
    stagger_from_frame: int = 0
    counter_frames: int = 0  # Frames left in which an attack is a counter.
    hover_frames: int = 0  # Frames left of reduced gravity after an air parry.
    # Events of the latest step, cleared at the start of the next one. Rewards read them.
    parried: bool = False  # This fighter parried an attack.
    counter_started: bool = False  # This fighter began a counter attack.
    missed_attack: bool = False  # A swing finished without hitting or being parried.
    missed_parry: bool = False  # A parry ended without parrying anything.


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
