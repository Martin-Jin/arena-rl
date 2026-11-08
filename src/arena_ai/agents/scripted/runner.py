"""The opponent that runs away from the learner."""

from arena_ai.agents.action import Action
from arena_ai.agents.scripted.view import learner_and_me
from arena_ai.config import Config, load_config


def runner(observation: list[float], config: Config | None = None) -> int:
    """Run away from the learner, and go past it by jumping when pinned against a wall.

    It never attacks or parries, so it can only be beaten by catching it. Running at
    the same speed as the learner, it is only caught when it is cornered, which the
    jump delays: it leaps toward the learner and lands behind it with room to run.
    """
    learner, me = learner_and_me(observation)
    config = config or load_config()
    game = config.game
    arena_width = game.world.arena_width_m
    reach = game.fighter.width_m + game.combat.attack_range_m

    dx = (learner["x"] - me["x"]) * arena_width  # Learner relative to this fighter.
    self_x = me["x"] * arena_width
    away = -1 if dx > 0 else 1  # Run left when the learner is on the right.
    wall_gap = self_x if away < 0 else arena_width - self_x
    toward = Action.MOVE_RIGHT if dx > 0 else Action.MOVE_LEFT
    # Near the wall behind it, running away is a dead end, so go past the learner instead:
    # walk toward it and jump over it once close. This fighter's own velocity is the memory
    # that keeps it going. Without it, walking toward the learner moves it out of the zone,
    # fleeing moves it back in, and it shuffles left and right at the wall.
    heading_toward = me["vx"] * dx > 0
    trap_zone = 4 * reach if heading_toward else 2 * reach
    if wall_gap < trap_zone:
        if abs(dx) < 2 * reach and me["on_ground"] == 1.0:
            return int(Action.MOVE_RIGHT_JUMP if dx > 0 else Action.MOVE_LEFT_JUMP)
        return int(toward)
    if abs(dx) > 4 * reach:
        return int(Action.IDLE)  # The learner is far off, so there is nothing to run from.
    return int(Action.MOVE_LEFT if away < 0 else Action.MOVE_RIGHT)
