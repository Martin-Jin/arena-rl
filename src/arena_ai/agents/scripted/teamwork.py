"""Teamwork for scripted opponents that fight side by side."""

from collections.abc import Callable
from functools import wraps

from arena_ai.agents.action import Action
from arena_ai.agents.scripted.view import learner_and_me, living_allies
from arena_ai.config import Config, load_config

# An ally counts as in the way when it is no farther than a body width plus this.
BLOCKED_MARGIN_M = 0.1

Policy = Callable[..., int]


def jumps_over_allies(policy: Policy) -> Policy:
    """Wrap a policy so that it jumps over an ally that blocks its way to the learner.

    **Why:** fighters cannot pass through each other, so opponents that all come from the
    same side queue up behind the first, and only the front one can hit. **What:** when the
    policy walks toward the learner and a living ally stands level with it, in that
    direction and within touching distance, it jumps while walking, and keeps walking toward
    the learner while it is above that ally. A jump clears a body, so it lands past the
    ally, in reach of the learner or even behind it. **Limits:** it only changes a plain walk
    toward the learner on the ground, or the pause in the air above an ally. A one-on-one has
    no allies, so nothing changes there.
    """

    @wraps(policy)
    def wrapped(observation: list[float], config: Config | None = None, **options) -> int:
        action = policy(observation, config, **options)
        config = config or load_config()
        learner, me = learner_and_me(observation)
        allies = living_allies(observation)
        if not allies:
            return action
        arena_width = config.game.world.arena_width_m
        fighter = config.game.fighter
        toward = 1 if learner["x"] >= me["x"] else -1
        walk = Action.MOVE_RIGHT if toward > 0 else Action.MOVE_LEFT
        if me["on_ground"] != 1.0:
            # Over an ally, carry on across it: the policy would stand still in the air, since
            # nothing can be hit from up here, and fall back onto the ally's side.
            height = config.game.observation.max_observed_height_m
            over_ally = any(
                abs(ally["x"] - me["x"]) * arena_width < fighter.width_m
                and (me["y"] - ally["y"]) * height >= fighter.height_m
                for ally in allies
            )
            return int(walk) if over_ally else action
        if action not in (Action.MOVE_LEFT, Action.MOVE_RIGHT):
            return action
        direction = 1 if action == Action.MOVE_RIGHT else -1
        for ally in allies:
            gap = (ally["x"] - me["x"]) * arena_width * direction
            level = abs(ally["y"] - me["y"]) * config.game.observation.max_observed_height_m
            if 0 < gap <= fighter.width_m + BLOCKED_MARGIN_M and level < fighter.height_m:
                return int(Action.MOVE_RIGHT_JUMP if direction > 0 else Action.MOVE_LEFT_JUMP)
        return action

    return wrapped
