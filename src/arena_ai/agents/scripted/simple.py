"""The two trivial opponents: one that does nothing and one that acts at random."""

import random

from arena_ai.agents.action import ACTION_COUNT, Action


def idle(observation, config=None) -> int:
    return int(Action.IDLE)


def random_action(observation, config=None, rng=random) -> int:
    """Pick any action. ``rng`` is anything with ``randrange``; training passes a seeded one."""
    return rng.randrange(ACTION_COUNT)
