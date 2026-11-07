"""Provide a deterministic opponent policy for the first training prototype."""

from arena_ai.agents.action import Action
from arena_ai.config import Config, load_config


def approach_and_attack(
    observation: list[float],
    # array with indexes:
    # 0: agent x position
    # 1: agent y position
    # 2: agent x velocity
    # 3: agent y velocity
    # 4: agent on ground
    # 5: agent facing direction
    # 6: agent health
    # 7: agent attack frame
    # 8: opponent x position
    # 9: opponent y position
    # 10: opponent x velocity
    # 11: opponent y velocity
    # 12: opponent on ground
    # 13: opponent facing direction
    # 14: opponent health
    # 15: opponent attack frame
    config: Config | None = None,
) -> int:
    """Move toward the learner and attack when within range.
    Agent is the scripted opponent, and the learner is the agent being trained.
    The x and y axes are flipped: larger x is to the left, and smaller y is higher.
    This policy is deterministic and does not use any randomness."""
    if len(observation) < 16:
        raise ValueError("one-versus-one observations require at least 16 values")

    config = config or load_config()
    agent_x = observation[0]
    agent_y = observation[1]
    agent_on_ground = observation[4] == 1.0
    agent_attack_frame = observation[7]
    opponent_x = observation[8]
    opponent_y = observation[9]

    # Let an attack that is already in progress play out.
    if agent_attack_frame > 0:
        return int(Action.IDLE)

    horizontal_distance = abs(opponent_x - agent_x) * config.game.world.arena_width_m
    vertical_distance = (
        abs(opponent_y - agent_y) * config.game.observation.max_observed_height_m
    )

    if horizontal_distance <= config.game.combat.attack_range_m:
        if vertical_distance <= config.game.fighter.height_m:
            return int(Action.ATTACK)
        # Flipped y axis: the learner is above when its y is smaller.
        if opponent_y < agent_y and agent_on_ground:
            return int(Action.JUMP)
        return int(Action.IDLE)

    # Flipped x axis: a larger opponent x means the learner is to the left.
    return int(Action.MOVE_LEFT if opponent_x > agent_x else Action.MOVE_RIGHT)
