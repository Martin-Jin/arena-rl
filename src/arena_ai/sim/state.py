"""Define the fighter state that the game loop updates, and spawning helpers."""

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from arena_ai.config import AttackType, GameConfig

T = TypeVar("T")


@dataclass
class FighterState:
    """The state of a fighter in the arena, updated every step."""

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
    attack_hit: bool = False  # The current swing hit someone.
    hit_targets: set[str] = field(default_factory=set)  # Who the current swing already hit.
    attack_counter: bool = False  # The current swing is a counter (shorter wind-up).
    lunge_dir: int = 0  # Direction of the counter lunge, fixed when the swing starts.
    parry_frame: int = 0  # 0 when not parrying, else 1 at the press and counting up.
    parry_window: int = 0  # Protected frames of the current parry.
    parry_count: int = 0  # Recent parry presses, which shrink the next window.
    parry_idle_frames: int = 0  # Frames since the last parry, capped at the reset time.
    parry_held: bool = False  # Parry input of the previous step, so holding parries once.
    # Visual only: the parry effects keep playing after a successful parry, which ends the
    # parry itself so the fighter can counter at once. Attacking does not stop them. This
    # is the frame they are on.
    parry_pose_frame: int = 0
    stagger_frames: int = 0  # Frames left in which a parried fighter cannot attack.
    # The normal-swing frame the parried swing had reached, for the deflect animation.
    stagger_from_frame: int = 0
    counter_frames: int = 0  # Frames left in which an attack is a counter.
    # Id of the fighter whose swing this fighter parried. Its counter lunges toward it.
    counter_target: str | None = None
    hover_frames: int = 0  # Frames left of reduced gravity after an air parry.
    # Events of the latest step, cleared at the start of the next one. Rewards read them.
    parried: bool = False  # This fighter parried an attack.
    counter_started: bool = False  # This fighter began a counter attack.
    counter_hit: bool = False  # This fighter's counter attack hit.
    missed_attack: bool = False  # A swing finished without hitting or being parried.
    missed_parry: bool = False  # A parry ended without parrying anything.
    attack_type: AttackType = (
        AttackType.SLASH
    )  # The type of attack the fighter is performing currently.
    # TODO attacks: list[AttackType] = field(default_factory=lambda: [AttackType.SLASH])
    # TODO The types of attacks the fighter can perform.


AGENT_ID = "agent"
_OPPONENT_PREFIX = "opponent"


def opponent_id(index: int) -> str:
    """Return the id of opponent ``index``: "opponent", "opponent_1", "opponent_2", ..."""
    return _OPPONENT_PREFIX if index == 0 else f"{_OPPONENT_PREFIX}_{index}"


def opponent_ids(count: int) -> list[str]:
    """Return the ids of ``count`` opponents, in order."""
    return [opponent_id(index) for index in range(count)]


def is_opponent(fighter_id: str) -> bool:
    """Return whether the id belongs to an opponent of the agent."""
    return fighter_id == _OPPONENT_PREFIX or fighter_id.startswith(_OPPONENT_PREFIX + "_")


def team_of(fighter_id: str) -> int:
    """Return 0 for the agent and 1 for everyone else. Fighters never hit their own team."""
    return 0 if fighter_id == AGENT_ID else 1


def sort_key(fighter_id: str) -> tuple[int, int, str]:
    """Order ids naturally: the agent first, then opponents by index (10 after 2)."""
    if fighter_id == AGENT_ID:
        return (0, 0, "")
    if fighter_id == _OPPONENT_PREFIX:
        return (1, 0, "")
    head, _, tail = fighter_id.rpartition("_")
    if head == _OPPONENT_PREFIX and tail.isdigit():
        return (1, int(tail), "")
    return (2, 0, fighter_id)


def enemies_of(fighter_id: str, fighters: Mapping[str, T]) -> dict[str, T]:
    """Return the fighters on the other team, in natural id order."""
    team = team_of(fighter_id)
    return {
        other_id: fighters[other_id]
        for other_id in sorted(fighters, key=sort_key)
        if team_of(other_id) != team
    }


def _opponent_x_positions(config: GameConfig, count: int) -> list[float]:
    """Spread ``count`` opponents in a row on the right side, centered on 0.75 of the arena."""
    width = config.world.arena_width_m
    step = min(0.15 * width, 0.45 * width / (count - 1)) if count > 1 else 0.0
    return [0.75 * width + (index - (count - 1) / 2) * step for index in range(count)]


def create_initial_fighters(
    config: GameConfig, rng: random.Random | None = None, opponent_count: int = 1
) -> dict[str, FighterState]:
    """Create the starting states for an episode with ``opponent_count`` opponents.

    Without ``rng`` the agent starts on the left and the opponents in a row on the
    right, which keeps episodes deterministic. With ``rng`` every fighter spawns at a
    random position (the agent first, then the opponents in order), which stays
    reproducible when the generator is seeded. Fighters that would overlap are pushed
    apart, which draws no extra random numbers. The agent faces its nearest opponent
    and every opponent faces the agent.
    """
    if opponent_count < 1:
        raise ValueError("opponent_count must be at least 1")
    fighters = {
        AGENT_ID: FighterState(
            x=config.world.arena_width_m * 0.25, health=config.fighter.max_health
        )
    }
    for index, x in enumerate(_opponent_x_positions(config, opponent_count)):
        fighters[opponent_id(index)] = FighterState(x=x, health=config.fighter.max_health)
    if rng is not None:
        margin = config.fighter.width_m / 2
        for fighter in fighters.values():
            fighter.x = rng.uniform(margin, config.world.arena_width_m - margin)
        # Imported here because collision imports rules, which imports this module.
        from arena_ai.sim.collision import resolve_collisions

        resolve_collisions(fighters, config)
    agent = fighters[AGENT_ID]
    opponents = [f for fighter_id, f in fighters.items() if fighter_id != AGENT_ID]
    nearest = min(opponents, key=lambda opponent: abs(opponent.x - agent.x))
    agent.facing_dir = 1 if nearest.x >= agent.x else -1
    for opponent in opponents:
        opponent.facing_dir = 1 if agent.x > opponent.x else -1
    return fighters
