"""Provide a deterministic opponent policy for the first training prototype."""

from collections.abc import Callable
import random

from arena_ai.agents.action import ACTION_COUNT, Action
from arena_ai.config import Config, load_config

# Observation layout (learner-perspective vector from encode_observation):
# 0-13 learner: x, y, vx, vy, on_ground, facing_dir, health, attack_frame, attack_counter,
#      parry_frame, parry_window, counter_frames, stagger_frames,
#      parry_held
# 14-27 scripted opponent (this policy): same order.


def approach_and_attack(
    observation: list[float],
    config: Config | None = None,
) -> int:
    """Face the learner, approach, jump when it is above, and attack in reach.

    Deterministic. The scripted fighter is the opponent half (indices 14-27)
    of the learner-perspective observation. It never parries.
    """
    if len(observation) < 28:
        raise ValueError("one-versus-one observations require at least 28 values")

    config = config or load_config()
    arena_width = config.game.world.arena_width_m
    max_height = config.game.observation.max_observed_height_m
    fighter_height = config.game.fighter.height_m
    fighter_width = config.game.fighter.width_m
    # matches sim.simulate.attack_hits
    attack_range = config.game.combat.attack_range_m

    learner_x = observation[0] * arena_width
    learner_y = observation[1] * max_height
    self_x = observation[14] * arena_width
    self_y = observation[15] * max_height
    self_on_ground = observation[18] == 1.0
    self_facing = int(observation[19])
    self_attack_frame = observation[21]

    dx = learner_x - self_x
    dy = learner_y - self_y
    toward = 1 if dx > 0 else -1 if dx < 0 else self_facing
    move_toward = Action.MOVE_RIGHT if toward > 0 else Action.MOVE_LEFT

    # Positive when the learner is in front of the scripted fighter.
    forward = self_facing * dx
    reach = fighter_width + attack_range

    # Mid-swing: do not turn (turning would also turn the swing), so only chase a learner
    # that is in front
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
    if forward <= reach:
        if abs(dy) < fighter_height:
            return int(Action.ATTACK)
        if dy > 0 and self_on_ground:
            return int(Action.JUMP)
        return int(Action.IDLE)

    return int(move_toward)


# Chance per step that the perfect-parry opponent attacks a learner that stands still. It cannot
# tell a learner that is passive from one that is spamming attack: a spamming learner
# looks still for the one step between swings. Attacking at once on that step meant it
# swung at the same time as the learner and could never parry, so it only attacks now
# and then. At 0.2 a passive learner is attacked after 5 frames on average (99.9% within
# 30 frames, half a second), so it does not stand there visibly doing nothing, while a
# spamming learner is still parried on about four swings in five. A lower chance made the
# wait before its attack long and uneven, which looked like a bug. A higher one trades
# hits with a spamming learner more often.
HUNTER_ATTACK_CHANCE = 0.2


def parrying_hunter(
    observation: list[float],
    config: Config | None = None,
) -> int:
    """Chase the learner, parry every attack it can, and attack a learner that stands still.

    A swing that is about to hit it from the front is parried on the exact step the swing turns
    active, so the learner cannot win by attacking from the front. It attacks after a
    parry (the counter) and, now and then (``HUNTER_ATTACK_CHANCE`` per step), when the
    learner stands on the ground with no movement input, including one that swings
    away from it, since that swing cannot hit. It never starts a swing against a
    learner that is moving or whose swing points at it, because a swing leaves it
    unable to parry. A learner that swings away and turns back while the swing is
    active can therefore hit it. An attack it starts against a still learner can still trade hits
    with a swing the learner starts right after. It uses the global ``random`` module,
    like ``random_action``, so it is not reproducible.
    It turns to face an attacker that is behind it (for example one that walked past it
    before attacking) and then parries. It only has a chance to beat it while it is busy:
    - It cannot parry or turn while it is attacking or parrying, so a swing that lands
      during its own swing or its parry recovery is not parried.
    - It needs one step to turn, so a swing that is already on its last wind-up step
      when it ends up behind the opponent still lands.
    """
    if len(observation) < 26:
        raise ValueError("one-versus-one observations require at least 26 values")

    config = config or load_config()
    game = config.game
    combat = game.combat
    arena_width = game.world.arena_width_m
    max_height = game.observation.max_observed_height_m

    learner_x = observation[0] * arena_width
    learner_y = observation[1] * max_height
    learner_swing = round(observation[7] * combat.attack_total_frames)
    learner_facing = int(observation[5])
    learner_is_counter = observation[8] > 0.5
    self_x = observation[14] * arena_width
    self_y = observation[15] * max_height
    self_facing = int(observation[19])
    self_busy = observation[21] > 0 or observation[23] > 0  # attack_frame or parry_frame
    counter_ready = observation[25] > 0  # counter_frames

    if learner_swing and not self_busy:
        # Where the learner's swing reaches, widened by one step of the learner's
        # walking, since its input on the step the swing lands is not known. A swing that
        # points away from this fighter cannot hit it, so it is not a threat: the fighter
        # carries on and may attack instead of parrying.
        reach = game.fighter.width_m / 2 + combat.attack_range_m
        slack = game.fighter.move_speed_mps * game.world.dt_seconds
        to_self = (self_x - learner_x) * learner_facing  # Positive: in front of the swing.
        # A counter also reaches behind its attacker.
        behind = combat.counter_hitbox_back_m if learner_is_counter else 0.0
        in_reach = -behind - slack <= to_self <= reach + slack
        level = abs(self_y - learner_y) < game.fighter.height_m
        if in_reach and level:
            learner_in_front = (learner_x - self_x) * self_facing >= 0
            windup = combat.windup_frames(learner_is_counter)
            if not learner_in_front and learner_swing < windup:
                # Moving toward the learner turns this fighter at once, in time to parry.
                return int(Action.MOVE_RIGHT if learner_x > self_x else Action.MOVE_LEFT)
            if learner_in_front and learner_swing == windup:
                return int(Action.PARRY)
            return int(Action.IDLE)  # Waiting to parry, or too late to turn around.
    if counter_ready and not self_busy:
        return int(Action.ATTACK)  # The counter after a parry.
    # Otherwise chase and face the learner, and attack only a learner that stands still.
    action = approach_and_attack(observation, config)
    # A swing that reached the branch above is no threat (it points away, or this fighter is
    # busy), so only movement, not swinging, stops the fighter from attacking.
    learner_still = observation[2] == 0 and observation[4] == 1.0  # vx, on_ground
    if action == Action.ATTACK and not (learner_still and random.random() < HUNTER_ATTACK_CHANCE):
        return int(Action.IDLE)
    return action


def idle(observation, config=None) -> int:
    return int(Action.IDLE)


def runner(observation: list[float], config: Config | None = None) -> int:
    """Run away from the learner, and jump over it when pinned against a wall.

    It never attacks or parries, so it can only be beaten by catching it. Running at
    the same speed as the learner, it is only caught when it is cornered, which the
    jump delays: it leaps toward the learner and lands behind it with room to run.
    """
    if len(observation) < 28:
        raise ValueError("one-versus-one observations require at least 28 values")

    config = config or load_config()
    game = config.game
    arena_width = game.world.arena_width_m
    reach = game.fighter.width_m + game.combat.attack_range_m

    dx = (observation[0] - observation[14]) * arena_width  # Learner relative to this fighter.
    self_x = observation[14] * arena_width
    away = -1 if dx > 0 else 1  # Run left when the learner is on the right.
    wall_gap = self_x if away < 0 else arena_width - self_x
    if wall_gap < game.fighter.width_m:
        if abs(dx) < 1.5 * reach and observation[18] == 1.0:  # Cornered, learner close, grounded.
            return int(Action.MOVE_RIGHT_JUMP if dx > 0 else Action.MOVE_LEFT_JUMP)
        return int(Action.IDLE)
    return int(Action.MOVE_LEFT if away < 0 else Action.MOVE_RIGHT)


def random_action(observation, config=None) -> int:
    return random.randrange(ACTION_COUNT)


# Every scripted opponent, for the training loop (``training.opponents``) and the viewer.
# ``parrier`` parries every attack it can see from the front, so a learner cannot win
# against it by attacking head on. Mix it with easier opponents so there is still a
# winning signal, and so the learner has to find another way past it.
OPPONENTS = {
    "approach": approach_and_attack,
    "parrier": parrying_hunter,
    "idle": idle,
    "runner": runner,
    "random": random_action,
}


def training_opponents(names: list[str], max_envs: int) -> list[tuple[str, Callable]]:
    """Return the (name, policy) of each training environment, in the order of ``names``.

    Raises ``ValueError`` for an empty list, more names than ``max_envs`` or a name
    that is not in ``OPPONENTS``, so a bad ``training.opponents`` stops training
    before it starts.
    """
    if not names:
        raise ValueError("training.opponents must name at least one opponent")
    if len(names) > max_envs:
        raise ValueError(
            f"training.opponents lists {len(names)} opponents but training.num_envs "
            f"allows at most {max_envs} environments"
        )
    unknown = [name for name in names if name not in OPPONENTS]
    if unknown:
        raise ValueError(
            f"Unknown training opponents {unknown}; choose from {sorted(OPPONENTS)}"
        )
    return [(name, OPPONENTS[name]) for name in names]
