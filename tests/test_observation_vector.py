import unittest

from arena_ai.config import AttackType, load_config
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
from arena_ai.sim.rules import attack_frames_left
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

    def test_attack_countdowns_follow_the_swing_phases(self) -> None:
        spec = CONFIG.game.combat.basic
        windup, active, recovery = (
            spec.attack_windup_frames,
            spec.attack_active_frames,
            spec.attack_recovery_frames,
        )
        combat = CONFIG.game.combat
        cases = {
            0: (0, 0, 0),
            1: (windup, active, recovery),
            windup: (1, active, recovery),  # Last wind-up frame: a parry pressed now is up.
            windup + 1: (0, active, recovery),  # First damaging frame.
            windup + active: (0, 1, recovery),
            windup + active + 1: (0, 0, recovery),
            spec.attack_total_frames: (0, 0, 1),  # Last frame: free to act on the next step.
        }
        for frame, expected in cases.items():
            fighter = FighterState(attack_frame=frame)
            self.assertEqual(attack_frames_left(fighter, combat), expected, frame)

    def test_a_counter_has_a_shorter_windup_but_the_same_recovery(self) -> None:
        combat = CONFIG.game.combat
        counter = FighterState(attack_frame=1, attack_type=AttackType.COUNTER)
        spec = combat.spec(AttackType.COUNTER)
        self.assertEqual(
            attack_frames_left(counter, combat),
            (spec.attack_windup_frames, combat.basic.attack_active_frames,
             combat.basic.attack_recovery_frames),
        )

    def test_the_vector_shows_the_dodge_state_and_cost(self) -> None:
        dodge = CONFIG.game.dodge
        fighter = FighterState(
            dodge_frames_left=dodge.duration_frames,
            dodge_gap_left=dodge.gap_frames,
            dodge_held=True,
            stamina=dodge.cost,
        )
        vector = encode_observation(get_observation(fighter, {"opponent": FighterState()}), CONFIG)
        row = fighter_features(vector, 0)
        self.assertEqual(row["dodge_frames_left"], 1.0)
        self.assertEqual(row["dodge_gap_left"], 1.0)
        self.assertEqual(row["dodge_held"], 1.0)
        self.assertEqual(row["stamina"], dodge.cost / dodge.max_stamina)
        self.assertEqual(row["dodge_cost"], dodge.cost / dodge.max_stamina)
        # A dodge burst blocks attacking and parrying, so it counts as busy time.
        self.assertGreater(row["busy_frames_left"], 0.0)
        # Opponent rows carry the same features, for the attention layer.
        self.assertEqual(fighter_features(vector, 1)["stamina"], 1.0)

    def test_a_counter_is_its_own_attack_type_in_the_vector(self) -> None:
        # The counter used to be a separate attack_counter flag beside the attack type.
        row = fighter_features(
            encode_observation(
                get_observation(
                    FighterState(attack_frame=1, attack_type=AttackType.COUNTER),
                    {"opponent": FighterState(x=3.0)},
                    CONFIG,
                ),
                CONFIG,
            ),
            0,
        )
        self.assertEqual(row[f"attack_type_{AttackType.COUNTER.value}"], 1.0)
        self.assertEqual(row[f"attack_type_{AttackType.SLASH.value}"], 0.0)
        self.assertNotIn("attack_counter", row)

    def test_the_vector_shows_the_countdowns_busy_frames_and_attack_type(self) -> None:
        combat = CONFIG.game.combat
        idle = fighter_features(encode(FighterState(), {"opponent": FighterState()}), 0)
        self.assertEqual(idle["busy_frames_left"], 0.0)
        self.assertEqual(idle[f"attack_type_{AttackType.SLASH.value}"], 0.0)
        swinging = fighter_features(
            encode(FighterState(attack_frame=1), {"opponent": FighterState()}), 0
        )
        self.assertAlmostEqual(
            swinging["attack_windup_left"],
            combat.basic.attack_windup_frames / combat.longest_windup_frames,
        )
        self.assertAlmostEqual(
            swinging["busy_frames_left"],
            combat.attack_total_frames
            / max(combat.longest_attack_frames, combat.parry_total_frames),
        )
        self.assertEqual(swinging[f"attack_type_{AttackType.SLASH.value}"], 1.0)

    def test_relative_features_measure_from_the_acting_fighter(self) -> None:
        combat = CONFIG.game.combat
        reach = CONFIG.game.fighter.width_m + combat.attack_range_m
        agent = FighterState(x=5.0, facing_dir=1)
        near = FighterState(x=5.0 + reach, facing_dir=-1)
        far = FighterState(x=5.0 - 3 * reach, facing_dir=-1)
        vector = encode(agent, {"opponent": near, "opponent_1": far})
        own, near_row, far_row = (fighter_features(vector, slot) for slot in range(3))
        for name in ("rel_x_ahead", "rel_y", "rel_x_aimed"):
            self.assertEqual(own[name], 0.0)
        self.assertAlmostEqual(near_row["rel_x_ahead"], 1 / 3)  # In front, one reach away.
        self.assertAlmostEqual(near_row["rel_x_aimed"], 1 / 3)  # Facing the agent.
        self.assertAlmostEqual(far_row["rel_x_ahead"], -1.0)  # Behind, clamped.
        self.assertAlmostEqual(far_row["rel_x_aimed"], -1.0)  # Faces away (to its left).
        far.facing_dir = 1
        far_row = fighter_features(encode(agent, {"opponent": far}), 1)
        self.assertAlmostEqual(far_row["rel_x_aimed"], 1.0)  # Now facing the agent.

    def test_a_row_past_the_end_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            fighter_features([0.0] * F, 1)
