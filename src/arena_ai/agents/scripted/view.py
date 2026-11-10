"""What a scripted opponent sees, and how to read it.

A scripted opponent gets the learner-perspective vector from ``encode_observation``: slot 0
is the learner (the agent), slot 1 is the opponent itself and slots 2 and up are its living
allies, each with the features in ``FEATURE_NAMES``. Read them with the helpers here, never
by number.
"""

from collections.abc import Mapping

from arena_ai.config import AttackSpec, AttackType, CombatConfig, Config
from arena_ai.sim.observation_vector import (
    FEATURES_PER_FIGHTER,
    assemble_vector,
    fighter_features,
    fighter_rows,
    opponent_mask,
    slot_count,
)
from arena_ai.sim.state import AGENT_ID, FighterState, enemies_of


def learner_and_me(observation: list[float]) -> tuple[dict[str, float], dict[str, float]]:
    """Return the learner's and this opponent's named features."""
    if len(observation) < 2 * FEATURES_PER_FIGHTER:
        raise ValueError("scripted opponents need the learner and one opponent in the observation")
    return fighter_features(observation, 0), fighter_features(observation, 1)


def is_attacking(fighter: dict[str, float]) -> bool:
    """Return whether the fighter is in any phase of a swing."""
    return (
        fighter["attack_windup_left"]
        + fighter["attack_active_left"]
        + fighter["attack_recovery_left"]
    ) > 0


def attack_spec(fighter: dict[str, float], combat: CombatConfig) -> AttackSpec:
    """Return the spec of the attack the fighter is in (the basic attack when it is not).

    It reads the attack-type one-hot, so a new attack type needs no change in the policies
    that use its range, reach and timing.
    """
    for attack_type in AttackType:
        if fighter[f"attack_type_{attack_type.value}"] > 0.5:
            return combat.spec(attack_type)
    return combat.basic


def swing_threatens(learner: dict[str, float], me: dict[str, float], config: Config) -> bool:
    """Return whether the learner's swing, once active, would reach this opponent.

    It is where the swing reaches, widened by one step of the learner's walking, since its
    input on the step the swing lands is not known. The swing starts at the learner's front
    edge and hits any body it touches, so it reaches a fighter whose center is a body width
    plus the range away. A swing that points away cannot hit, and neither can one more than
    a fighter height above or below. An attack with a ``hitbox_back_m`` (the counter) also
    reaches behind its attacker. The range comes from the attack's own spec.
    """
    game = config.game
    combat = game.combat
    learner_x = learner["x"] * game.world.arena_width_m
    learner_y = learner["y"] * game.observation.max_observed_height_m
    self_x = me["x"] * game.world.arena_width_m
    self_y = me["y"] * game.observation.max_observed_height_m
    spec = attack_spec(learner, combat)
    reach = game.fighter.width_m + spec.range_m
    slack = game.fighter.move_speed_mps * game.world.dt_seconds
    to_self = (self_x - learner_x) * int(learner["facing_dir"])  # Positive: in front of it.
    behind = 0.0 if spec.hitbox_back_m is None else spec.hitbox_back_m + game.fighter.width_m / 2
    in_reach = -behind - slack <= to_self <= reach + slack
    return in_reach and abs(self_y - learner_y) < game.fighter.height_m


def living_allies(observation: list[float]) -> list[dict[str, float]]:
    """Return the named features of this opponent's living allies (empty in a one-on-one)."""
    slots = slot_count(observation)
    mask = opponent_mask(observation, slots)
    return [
        fighter_features(observation, slot)
        for slot in range(2, slots + 1)
        if mask[slot - 1] > 0.5
    ]


def team_observation(
    fighters: Mapping[str, FighterState],
    fighter_id: str,
    config: Config,
    rows: Mapping[str, list[float]] | None = None,
) -> list[float]:
    """Return the vector for the opponent ``fighter_id``: the agent, itself, then its allies.

    The opponent is the first opponent slot, which is where the scripted policies read
    themselves. Dead allies are left out (the mask is 0 for them). Pass ``rows`` (from
    ``fighter_rows``) when building the views of several opponents from the same state, so
    each fighter is encoded once.
    """
    if rows is None:
        rows = fighter_rows(fighters, config)
    allies = [
        other_id
        for other_id, fighter in enemies_of(AGENT_ID, fighters).items()
        if other_id != fighter_id and fighter.health > 0
    ]
    return assemble_vector(
        rows[AGENT_ID],
        [rows.get(other_id) for other_id in [fighter_id, *allies]],
        config.game.observation.max_opponents,
    )
