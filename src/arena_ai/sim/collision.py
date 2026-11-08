# collision.py
# This module keeps fighters' bodies from overlapping. Pure simulation: no pygame.

from collections.abc import Mapping

from arena_ai.config import GameConfig
from arena_ai.sim.rules import LEVEL_TOLERANCE_M, is_lunging
from arena_ai.sim.state import FighterState

# Bodies closer than this are touching, not overlapping: it absorbs float rounding after a
# push.
OVERLAP_TOLERANCE_M = 1e-9


def bodies_overlap(a: FighterState, b: FighterState, config: GameConfig) -> bool:
    """Return whether the two fighters' body boxes overlap (touching is not overlap)."""
    width, height = config.fighter.width_m, config.fighter.height_m
    return (
        abs(a.x - b.x) < width - OVERLAP_TOLERANCE_M
        and abs(a.y - b.y) < height
    )


def _left_right(a: FighterState, b: FighterState) -> tuple[FighterState, FighterState]:
    """Order two overlapping fighters by x.

    Fighters exactly level (a spawn, or a lunge that stopped at the opponent's x) are
    ordered so that a lunger ends on the side it came from, in front of the opponent,
    where the opponent can still parry it. Otherwise they are ordered by list position.
    """
    if abs(a.x - b.x) > LEVEL_TOLERANCE_M:
        return (a, b) if a.x < b.x else (b, a)
    if a.lunge_dir:
        return (a, b) if a.lunge_dir > 0 else (b, a)
    if b.lunge_dir:
        return (b, a) if b.lunge_dir > 0 else (a, b)
    return a, b


def resolve_collisions(fighters: Mapping[str, FighterState], config: GameConfig) -> None:
    """Push overlapping fighters apart along x so none stays inside another.

    Every fighter has a body box (``fighter.width_m`` by ``height_m``). Fighters at
    different heights do not collide, so a jump clears an opponent. A fighter that walks into
    another is stopped by it, and the one standing still is not moved. If both walk into
    each other, or neither does, the overlap is split evenly. An arena wall takes the
    push first: a fighter against it stays put and the other moves the full overlap.

    The one exception is a counter's lunge (``is_lunging``), which passes through the
    opponent. Once the lunge ends, so does the exemption, so a fighter left inside the
    opponent is pushed out. This also runs after a parry, which cancels a lunge, so a
    parried fighter stops at the parrier instead of ending up inside it.
    """
    width = config.fighter.width_m
    arena_width = config.world.arena_width_m
    ordered = list(fighters.values())
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if is_lunging(a, config.combat) or is_lunging(b, config.combat):
                continue
            if not bodies_overlap(a, b, config):
                continue
            left, right = _left_right(a, b)
            push = width - (right.x - left.x)
            # Whoever walked into the other undoes the overlap. A fighter that stands
            # still is not moved, and with no walker (or two) the push is split evenly.
            into_left, into_right = max(0.0, left.vx), max(0.0, -right.vx)
            walkers = into_left + into_right
            left_share = into_left / walkers if walkers else 0.5
            left.x = min(max(left.x - push * left_share, 0.0), arena_width)
            right.x = min(max(right.x + push * (1 - left_share), 0.0), arena_width)
            if right.x - left.x < width:  # A wall stopped one of them: the other makes room.
                if right.x >= arena_width:
                    left.x = right.x - width
                else:
                    right.x = left.x + width
