"""The approach-and-attack opponent."""

from arena_ai.agents.action import Action
from arena_ai.agents.scripted.teamwork import jumps_over_allies
from arena_ai.agents.scripted.view import is_attacking, learner_and_me
from arena_ai.config import Config, load_config


@jumps_over_allies
def approach_and_attack(
    observation: list[float],
    config: Config | None = None,
    swing_start_offset_m: float = 0.0,
) -> int:
    """Face the learner, approach, jump when it is above, and attack in reach.

    It keeps walking toward a learner in front of it during its swing, including on the
    step it starts the swing. ``swing_start_offset_m`` moves the distance at which it starts
    the swing: positive starts it early, while still out of reach (the walk during the
    wind-up closes the gap), and negative walks closer first. Training draws it per
    episode so no fixed distance can be learned. Deterministic. The scripted fighter is slot 1
    of the learner-perspective observation. It never parries.
    """
    learner, me = learner_and_me(observation)
    config = config or load_config()
    arena_width = config.game.world.arena_width_m
    max_height = config.game.observation.max_observed_height_m
    fighter_height = config.game.fighter.height_m
    fighter_width = config.game.fighter.width_m
    # matches sim.simulate.attack_hits
    attack_range = config.game.combat.attack_range_m

    learner_x = learner["x"] * arena_width
    learner_y = learner["y"] * max_height
    self_x = me["x"] * arena_width
    self_y = me["y"] * max_height
    self_on_ground = me["on_ground"] == 1.0
    self_facing = int(me["facing_dir"])
    self_attacking = is_attacking(me)

    dx = learner_x - self_x
    dy = learner_y - self_y
    toward = 1 if dx > 0 else -1 if dx < 0 else self_facing
    move_toward = Action.MOVE_RIGHT if toward > 0 else Action.MOVE_LEFT

    # Positive when the learner is in front of the scripted fighter.
    forward = self_facing * dx
    reach = fighter_width + attack_range

    # Mid-swing: do not turn (turning would also turn the swing), so only push toward a
    # learner that is in front of us. Pushing on past the edge of reach, up to body contact
    # (bodies block each other), means a learner that steps back a few centimeters on the
    # active frames is still hit. Standing still at the edge let it dodge that way with one
    # tap and punish the recovery.
    if self_attacking:
        if forward > 0:
            return int(move_toward)
        return int(Action.IDLE)

    # Learner is past our center (behind us): turn to face it. Moving
    # toward it flips facing, so it cannot hide behind us.
    if forward < 0:
        return int(move_toward)

    # In horizontal reach of the hitbox (center to front edge + range), moved by the offset.
    if forward <= reach + swing_start_offset_m:
        if abs(dy) < fighter_height:
            return int(Action.MOVE_RIGHT_ATTACK if toward > 0 else Action.MOVE_LEFT_ATTACK)
        if dy > 0 and self_on_ground:
            return int(Action.JUMP)
        return int(Action.IDLE)

    return int(move_toward)
