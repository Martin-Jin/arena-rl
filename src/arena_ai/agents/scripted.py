"""Provide a deterministic opponent policy for the first training prototype."""

from arena_ai.agents.action import Action
from arena_ai.config import Config, load_config

# Observation layout (learner-perspective vector from encode_observation):
# 0-7 learner: x, y, vx, vy, on_ground, facing_dir, health, attack_frame
# 8-15 scripted opponent (this policy): same order.


def approach_and_attack(
    observation: list[float],
    config: Config | None = None,
) -> int:
    """Face the learner, approach, jump when it is above, and attack in reach.

    Deterministic. The scripted fighter is the opponent half (indices 8-15)
    of the learner-perspective observation.
    """
    if len(observation) < 16:
        raise ValueError("one-versus-one observations require at least 16 values")

    config = config or load_config()
    arena_width = config.game.world.arena_width_m
    max_height = config.game.observation.max_observed_height_m
    fighter_height = config.game.fighter.height_m
    fighter_width = config.game.fighter.width_m
    # matches sim.simulate.attack_hits
    attack_range = config.game.combat.attack_range_m

    learner_x = observation[0] * arena_width
    learner_y = observation[1] * max_height
    self_x = observation[8] * arena_width
    self_y = observation[9] * max_height
    self_on_ground = observation[12] == 1.0
    self_facing = int(observation[13])
    self_attack_frame = observation[15]

    dx = learner_x - self_x
    dy = learner_y - self_y
    toward = 1 if dx > 0 else -1 if dx < 0 else self_facing
    move_toward = Action.MOVE_RIGHT if toward > 0 else Action.MOVE_LEFT

    # Positive when the learner is in front of the scripted fighter.
    forward = self_facing * dx
    reach = fighter_width + attack_range

    # Mid-swing: facing is locked, so only chase a learner that is in front
    # of us and out of reach. Otherwise let the swing play out.
    if self_attack_frame > 0:
        if forward > reach:
            return int(move_toward)
        return int(Action.IDLE)

    # Learner is past our center (behind us): turn to face it. Moving
    # toward it flips facing, so it cannot hide behind us.
    if forward < 0:
        return int(move_toward)

    # In horizontal reach of the hitbox (center to front edge + range).
    if forward <= fighter_width + attack_range:
        if abs(dy) < fighter_height:
            return int(Action.ATTACK)
        if dy > 0 and self_on_ground:
            return int(Action.JUMP)
        return int(Action.IDLE)

    return int(move_toward)
