"""The dodging opponent: approach-and-attack that rolls through every swing it can see."""

from arena_ai.agents.action import Action
from arena_ai.agents.scripted.approach import approach_and_attack
from arena_ai.agents.scripted.teamwork import jumps_over_allies
from arena_ai.agents.scripted.view import (
    attack_spec,
    is_attacking,
    learner_and_me,
    swing_threatens,
)
from arena_ai.config import Config, load_config


def _can_dodge(me: dict[str, float]) -> bool:
    """Return whether the dodge would start: on the ground, free and with enough stamina."""
    # A swing past its wind-up cannot be dodged out of, and a fighter cannot dodge in the gap
    # after a dodge. Stamina and its cost are both shares of the maximum.
    committed = me["attack_windup_left"] == 0 and (
        me["attack_active_left"] > 0 or me["attack_recovery_left"] > 0
    )
    return (
        me["on_ground"] == 1.0
        and me["stagger_frames"] == 0
        and me["dodge_frames_left"] == 0
        and me["dodge_gap_left"] == 0
        and me["stamina"] >= me["dodge_cost"]
        and not committed
    )


@jumps_over_allies
def dodging_approacher(
    observation: list[float],
    config: Config | None = None,
    swing_start_offset_m: float = 0.0,
) -> int:
    """Approach and attack like ``approach_and_attack``, but dodge every swing it can.

    It dodges toward the learner, so it rolls through it and ends behind it, where the swing
    does not point. It dodges when the swing's wind-up has so few frames left that the whole
    active window falls inside the dodge's invincible frames: with the shipped numbers (an
    8-frame dodge, a 4-frame active window) that is the last 5 wind-up frames, so the swing
    cannot hit it. It reads the swing's range, reach and timing from the attack's own spec, so
    a counter or any later attack type needs no special case. A dodge cancels its own
    wind-up, so it dodges even mid-swing.
    **Limits:** each dodge costs stamina, so a learner that swings often enough runs it out
    (a full bar is two dodges, and it refills slowly), and a dodge leaves a gap in which it
    can be hit. ``swing_start_offset_m`` is passed to ``approach_and_attack``.
    """
    learner, me = learner_and_me(observation)
    config = config or load_config()
    combat = config.game.combat
    # Wind-up frames left: 1 is the last one, so a dodge pressed now is up on the first hit.
    windup_left = round(learner["attack_windup_left"] * combat.longest_windup_frames)
    # The frames its active window needs inside the invincible frames, from the swing's own spec.
    active_frames = attack_spec(learner, combat).attack_active_frames
    lead = max(1, config.game.dodge.duration_frames - active_frames + 1)
    if (
        is_attacking(learner)
        and 1 <= windup_left <= lead
        and swing_threatens(learner, me, config)
        and _can_dodge(me)
    ):
        toward_right = learner["x"] >= me["x"]
        return int(Action.MOVE_RIGHT_DODGE if toward_right else Action.MOVE_LEFT_DODGE)
    return approach_and_attack(observation, config, swing_start_offset_m=swing_start_offset_m)
