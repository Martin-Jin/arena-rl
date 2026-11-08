"""What a scripted opponent sees, and how to read it.

A scripted opponent gets the learner-perspective vector from ``encode_observation``: slot 0
is the learner (the agent), slot 1 is the opponent itself and slots 2 and up are its living
allies, each with the features in ``FEATURE_NAMES``. Read them with the helpers here, never
by number.
"""

from collections.abc import Mapping

from arena_ai.config import Config
from arena_ai.sim.observation import get_observation
from arena_ai.sim.observation_vector import (
    FEATURES_PER_FIGHTER,
    encode_observation,
    fighter_features,
    opponent_mask,
    slot_count,
)
from arena_ai.sim.state import AGENT_ID, FighterState, enemies_of, opponent_id


def learner_and_me(observation: list[float]) -> tuple[dict[str, float], dict[str, float]]:
    """Return the learner's and this opponent's named features."""
    if len(observation) < 2 * FEATURES_PER_FIGHTER:
        raise ValueError("scripted opponents need the learner and one opponent in the observation")
    return fighter_features(observation, 0), fighter_features(observation, 1)


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
    fighters: Mapping[str, FighterState], fighter_id: str, config: Config
) -> list[float]:
    """Return the vector for the opponent ``fighter_id``: the agent, itself, then its allies.

    The ids are renumbered so that the opponent is the first opponent slot, which is where
    the scripted policies read themselves. Dead allies are left out (the mask is 0 for them).
    """
    allies = [
        fighter
        for other_id, fighter in enemies_of(AGENT_ID, fighters).items()
        if other_id != fighter_id and fighter.health > 0
    ]
    team = {
        opponent_id(index): fighter
        for index, fighter in enumerate([fighters[fighter_id], *allies])
    }
    return encode_observation(get_observation(fighters[AGENT_ID], team, config), config)
