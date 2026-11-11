"""A scripted agent that attacks and parries perfectly but does not dodge."""

import random

from arena_ai.agents.action import Action
from arena_ai.agents.scripted.approach import approach_and_attack
from arena_ai.agents.scripted.teamwork import jumps_over_allies
from arena_ai.agents.scripted.view import is_attacking, learner_and_me, swing_threatens
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
    """Chases and attacks normally. When a swing starts within 4.0 sprite widths, it only turns, jump-parries and counters."""
    learner, me = learner_and_me(observation)
    config = config or load_config()
    combat = config.game.combat
    arena_width = config.game.world.arena_width_m

    learner_x = learner["x"] * arena_width
    windup_left = round(learner["attack_windup_left"] * combat.longest_windup_frames)
    self_x = me["x"] * arena_width
    self_facing = int(me["facing_dir"])
    self_busy = me["busy_frames_left"] > 0
    counter_ready = me["counter_frames"] > 0
    learner_airborne = learner["y"] > 0.0
    learner_in_front = (learner_x - self_x) * self_facing >= 0

    sprite_width_m = config.game.fighter.height_m * config.game.fighter.width_to_height_ratio
    in_focus_zone = abs(learner_x - self_x) <= 4.0 * sprite_width_m

    # Counter after a successful parry.
    if counter_ready and not self_busy:
        return int(Action.ATTACK)

    # Focus mode: a swing is active within 4.0 sprite widths. No walking, no attacking.
    if is_attacking(learner) and in_focus_zone:
        if self_busy:
            return int(Action.IDLE)
        if not learner_in_front:
            return int(Action.MOVE_RIGHT if learner_x > self_x else Action.MOVE_LEFT)  # turn only
        if windup_left == 1:
            return int(
                Action.JUMP_PARRY if learner_airborne else Action.PARRY
            )  # ASSUMPTION: JUMP_PARRY exists
        return int(Action.IDLE)

    # Normal mode: chase and attack, same gating as before.
    action = approach_and_attack(observation, config, swing_start_offset_m=swing_start_offset_m)
    away = 1 if learner_x > self_x else -1
    learner_still = learner["vx"] * away <= 0
    if action in SWING_STARTS and not (learner_still and rng.random() < attack_chance):
        return int(Action.IDLE)
    return action
