import unittest

from arena_ai.config import load_config
from arena_ai.sim.observation import get_observation
from arena_ai.sim.observation_vector import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    FEATURES_PER_FIGHTER,
    encode_observation,
    fighter_features,
    observation_size,
    opponent_mask,
)
from arena_ai.sim.state import FighterState, opponent_ids

CONFIG = load_config()
SLOTS = CONFIG.game.observation.max_opponents
F = FEATURES_PER_FIGHTER


def encode(agent: FighterState, opponents: dict[str, FighterState]) -> list[float]:
    return encode_observation(get_observation(agent, opponents, CONFIG), CONFIG)


class ObservationVectorTests(unittest.TestCase):
    def test_the_layout_is_self_then_slots_then_mask(self) -> None:
        self.assertEqual(observation_size(SLOTS), F * (1 + SLOTS) + SLOTS)
        self.assertEqual(len(FEATURE_NAMES), F)
        self.assertEqual(FEATURE_INDEX["x"], 0)
        vector = encode(FighterState(x=3.0), {"opponent": FighterState(x=9.0)})
        self.assertEqual(len(vector), observation_size(SLOTS))
        width = CONFIG.game.world.arena_width_m
        self.assertAlmostEqual(fighter_features(vector, 0)["x"], 3.0 / width)
        self.assertAlmostEqual(fighter_features(vector, 1)["x"], 9.0 / width)

    def test_missing_opponents_are_zero_padded_with_mask_zero(self) -> None:
        vector = encode(FighterState(), {"opponent": FighterState(x=9.0)})
        self.assertEqual(opponent_mask(vector, SLOTS), [1.0] + [0.0] * (SLOTS - 1))
        for slot in range(2, SLOTS + 1):
            self.assertEqual(list(fighter_features(vector, slot).values()), [0.0] * F)

    def test_a_dead_opponent_is_zeroed_and_masked(self) -> None:
        opponents = {
            "opponent": FighterState(x=9.0, health=0),
            "opponent_1": FighterState(x=8.0),
        }
        vector = encode(FighterState(), opponents)
        self.assertEqual(opponent_mask(vector, SLOTS)[:2], [0.0, 1.0])
        self.assertEqual(list(fighter_features(vector, 1).values()), [0.0] * F)
        self.assertGreater(fighter_features(vector, 2)["health"], 0.0)

    def test_opponents_are_sorted_naturally(self) -> None:
        ids = opponent_ids(3)
        opponents = {ids[2]: FighterState(x=1.0), ids[0]: FighterState(x=2.0),
                     ids[1]: FighterState(x=3.0)}
        vector = encode(FighterState(), opponents)
        xs = [fighter_features(vector, slot)["x"] * CONFIG.game.world.arena_width_m
              for slot in (1, 2, 3)]
        self.assertEqual([round(x) for x in xs], [2, 3, 1])

    def test_too_many_opponents_are_refused(self) -> None:
        opponents = {i: FighterState() for i in opponent_ids(SLOTS + 1)}
        with self.assertRaises(ValueError):
            encode(FighterState(), opponents)

    def test_a_row_past_the_end_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            fighter_features([0.0] * F, 1)
