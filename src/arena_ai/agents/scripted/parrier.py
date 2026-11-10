"""The perfect-parry opponent: parries every swing it can see, attacks a still learner."""

import random

from arena_ai.agents.action import Action
from arena_ai.agents.scripted.approach import approach_and_attack
from arena_ai.agents.scripted.teamwork import jumps_over_allies
from arena_ai.agents.scripted.view import is_attacking, learner_and_me
from arena_ai.config import Config, load_config

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

# The actions that start a swing.
SWING_STARTS = (
    Action.ATTACK,
    Action.MOVE_LEFT_ATTACK,
    Action.MOVE_RIGHT_ATTACK,
    Action.JUMP_ATTACK,
    Action.MOVE_LEFT_JUMP_ATTACK,
    Action.MOVE_RIGHT_JUMP_ATTACK,
)


@jumps_over_allies
def parrying_hunter(
    observation: list[float],
    config: Config | None = None,
    swing_start_offset_m: float = 0.0,
    attack_chance: float = HUNTER_ATTACK_CHANCE,
    rng=random,
) -> int:
    """Chase the learner, parry every attack it can, and attack a learner that stands still.

    A swing that is about to hit it from the front is parried on the exact step the swing turns
    active, so the learner cannot win by attacking from the front. It attacks after a
    parry (the counter) and, now and then (``HUNTER_ATTACK_CHANCE`` per step), when the
    learner is not walking away from it, including one that swings away from it, since
    that swing cannot hit. It never starts a swing against a learner that is walking
    away or whose swing points at it, because a swing leaves it unable to parry. A learner
    that swings away and turns back while the swing is active can therefore hit it. An
    attack it starts against a still learner can still trade hits with a swing the learner
    starts right after. ``swing_start_offset_m`` is passed to ``approach_and_attack``.
    ``attack_chance`` replaces ``HUNTER_ATTACK_CHANCE``. ``rng`` (anything with ``random()``)
    draws the chance: it defaults to the global ``random`` module, so the viewer is not
    reproducible, and training passes a seeded generator.
    It turns to face an attacker that is behind it (for example one that walked past it
    before attacking) and then parries. It only has a chance to beat it while it is busy:
    - It cannot parry or turn while it is attacking or parrying, so a swing that lands
      during its own swing or its parry recovery is not parried.
    - It needs one step to turn, so a swing that is already on its last wind-up step
      when it ends up behind the opponent still lands.
    """
    learner, me = learner_and_me(observation)
    config = config or load_config()
    game = config.game
    combat = game.combat
    arena_width = game.world.arena_width_m
    max_height = game.observation.max_observed_height_m

    learner_x = learner["x"] * arena_width
    learner_y = learner["y"] * max_height
    learner_facing = int(learner["facing_dir"])
    learner_is_counter = learner["attack_counter"] > 0.5
    learner_swinging = is_attacking(learner)
    # Wind-up frames left: 1 is the last one, so a parry pressed now is up on the first hit.
    windup_left = round(learner["attack_windup_left"] * combat.longest_windup_frames)
    self_x = me["x"] * arena_width
    self_y = me["y"] * max_height
    self_facing = int(me["facing_dir"])
    self_busy = me["busy_frames_left"] > 0
    counter_ready = me["counter_frames"] > 0

    if learner_swinging and not self_busy:
        # Where the learner's swing reaches, widened by one step of the learner's
        # walking, since its input on the step the swing lands is not known. The swing
        # starts at the learner's front edge and hits any body it touches, so it reaches a
        # fighter whose center is a body width plus the range away: the same distance at
        # which this fighter stops to attack. A swing that points away from this fighter
        # cannot hit it, so it is not a threat: the fighter carries on and may attack
        # instead of parrying.
        reach = game.fighter.width_m + combat.attack_range_m
        slack = game.fighter.move_speed_mps * game.world.dt_seconds
        to_self = (self_x - learner_x) * learner_facing  # Positive: in front of the swing.
        # A counter also reaches behind its attacker, as far back as its hitbox touches the
        # body of this fighter.
        behind = (
            combat.counter_hitbox_back_m + game.fighter.width_m / 2 if learner_is_counter else 0.0
        )
        in_reach = -behind - slack <= to_self <= reach + slack
        level = abs(self_y - learner_y) < game.fighter.height_m
        if in_reach and level:
            learner_in_front = (learner_x - self_x) * self_facing >= 0
            if not learner_in_front and windup_left > 1:
                # Moving toward the learner turns this fighter at once, in time to parry.
                return int(Action.MOVE_RIGHT if learner_x > self_x else Action.MOVE_LEFT)
            if learner_in_front and windup_left == 1:
                return int(Action.PARRY)
            return int(Action.IDLE)  # Waiting to parry, or too late to turn around.
    if counter_ready and not self_busy:
        return int(Action.ATTACK)  # The counter after a parry.
    # Otherwise chase and face the learner, and attack only a learner that stands still.
    action = approach_and_attack(observation, config, swing_start_offset_m=swing_start_offset_m)
    # A swing that reached the branch above is no threat (it points away, or this fighter is
    # busy), so only movement, not swinging, stops the fighter from attacking.
    # Only a learner walking away from this fighter can dodge its swing, so only that
    # blocks an attack. Walking toward it, or back and forth in place, does not: this
    # fighter keeps closing in during the swing. Blocking on any movement let a learner
    # that wiggled left and right stop it from ever attacking.
    away = 1 if learner_x > self_x else -1
    learner_still = learner["vx"] * away <= 0
    if action in SWING_STARTS and not (learner_still and rng.random() < attack_chance):
        return int(Action.IDLE)
    return action
