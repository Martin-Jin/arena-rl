"""Keep fighters' bodies from overlapping. Pure simulation: no pygame."""

from collections.abc import Mapping

from arena_ai.config import GameConfig
from arena_ai.sim.rules import LEVEL_TOLERANCE_M, is_lunging
from arena_ai.sim.state import FighterState

# Bodies closer than this are touching, not overlapping: it absorbs float rounding after a
# push.
OVERLAP_TOLERANCE_M = 1e-6
# Each pass halves what a chain of three or more leaves over, so this ends them well inside
# the tolerance. The loop stops at the first pass that finds nothing.
MAX_PASSES = 32


def bodies_overlap(a: FighterState, b: FighterState, config: GameConfig) -> bool:
    """Return whether the two fighters' body boxes overlap (touching is not overlap)."""
    width, height = config.fighter.width_m, config.fighter.height_m
    return abs(a.x - b.x) < width - OVERLAP_TOLERANCE_M and abs(a.y - b.y) < height


def _left_right(
    a: FighterState, b: FighterState
) -> tuple[FighterState, FighterState, FighterState | None]:
    """Order two overlapping fighters by x, and name a lunger that must do the moving.

    Fighters exactly level (a spawn, or a lunge that stopped at the opponent's x) are
    ordered so that a lunger ends on the far side, the side it was heading toward, having
    cut through the opponent. That lunger is returned as the third value, and moves the
    whole push. Otherwise they are ordered by x, or by list position when level.
    """
    if abs(a.x - b.x) > LEVEL_TOLERANCE_M:
        return (a, b, None) if a.x < b.x else (b, a, None)
    for lunger, other in ((a, b), (b, a)):
        if lunger.lunge_dir:
            left, right = (other, lunger) if lunger.lunge_dir > 0 else (lunger, other)
            return left, right, lunger
    return a, b, None


def resolve_collisions(
    fighters: Mapping[str, FighterState], config: GameConfig, hits_checked: bool = False
) -> None:
    """Push overlapping fighters apart along x so none stays inside another.

    Every living fighter has a body box (``fighter.width_m`` by ``height_m``). Fighters at
    different heights do not collide, so a jump clears an opponent. A fighter that walks into
    another is stopped by it, and the one standing still is not moved. If both walk into
    each other, or neither does, the overlap is split evenly. An arena wall takes the
    push first: a fighter against it stays put and the other moves the full overlap.

    The one exception is a counter's lunge (``is_lunging``), which passes through the
    opponent. Once the lunge ends, so does the exemption, so a fighter left inside the
    opponent is pushed out, on the far side, the whole way: the lunger moves and the
    opponent stays put. ``hits_checked`` is set for the run after the hits, which ends the
    lunge's last frame (the one a parry is checked on). That run also follows a parry,
    which cancels a lunge, so a parried fighter is put on its own side of the parrier
    (see ``apply_parry``) instead of ending up inside it.
    """
    ordered = list(fighters.values())
    # Pushing one pair apart can push a fighter into a third, so repeat until none overlap.
    for _ in range(MAX_PASSES):
        if not _resolve_pass(ordered, config, hits_checked):
            break


def _resolve_pass(ordered: list[FighterState], config: GameConfig, hits_checked: bool) -> bool:
    """Push apart every overlapping pair once. Return whether any pair overlapped."""
    width = config.fighter.width_m
    arena_width = config.world.arena_width_m
    found = False
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if is_lunging(a, config.combat, hits_checked) or is_lunging(
                b, config.combat, hits_checked
            ):
                continue
            if a.health <= 0 or b.health <= 0 or not bodies_overlap(a, b, config):
                continue
            found = True
            left, right, lunger = _left_right(a, b)
            # The overlap is the width minus the distance between their centers. The push is
            # the overlap minus the tolerance, so touching fighters are not pushed apart.
            push = width - (right.x - left.x)
            # Whoever walked into the other undoes the overlap. A fighter that stands
            # still is not moved, and with no walker (or two) the push is split evenly.
            # negative vx is left walking left, positive is right, and zero is standing still.
            into_left, into_right = max(0.0, left.vx), max(0.0, -right.vx)
            if lunger is not None:  # The lunger moves the whole push, the opponent stays put.
                into_left, into_right = (0.0, 1.0) if lunger is right else (1.0, 0.0)
            # into left is how much the left fighter walked into the right one, and into right is
            # how much the right fighter walked into the left one.
            walkers = into_left + into_right
            left_share = into_left / walkers if walkers else 0.5
            left.x = min(max(left.x - push * left_share, 0.0), arena_width)
            right.x = min(max(right.x + push * (1 - left_share), 0.0), arena_width)
            if right.x - left.x < width:  # A wall stopped one of them: the other makes room.
                if right.x >= arena_width:
                    left.x = right.x - width
                else:
                    right.x = left.x + width
    return found
