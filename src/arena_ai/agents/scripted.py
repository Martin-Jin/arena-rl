"""Provide a deterministic opponent policy for the first training prototype."""

from arena_ai.agents.action import Action
from arena_ai.config import Config, load_config


def approach_and_attack(
    observation: list[float],
    config: Config | None = None,
) -> int:
    """Move toward the learner and attack when within range."""
    if len(observation) < 14:
        raise ValueError("one-versus-one observations require at least 14 values")

    config = config or load_config()
    agent_x = observation[0]
    opponent_x = observation[7]
    distance = abs(opponent_x - agent_x) * config.game.world.arena_width_m
    if distance <= config.game.combat.attack_range_m:
        return int(Action.ATTACK)
    return int(Action.MOVE_LEFT if opponent_x > agent_x else Action.MOVE_RIGHT)
