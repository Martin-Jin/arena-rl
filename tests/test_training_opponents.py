import dataclasses
import random
import unittest

from arena_ai.agents.scripted import OpponentMixture, random_opponent_pool, training_opponents
from arena_ai.config import load_config


SLOTS = load_config().game.observation.max_opponents


def run(entries, max_envs=4, max_opponents=SLOTS):
    return training_opponents(entries, max_envs, max_opponents)


class TrainingOpponentsTests(unittest.TestCase):
    def test_each_entry_becomes_one_environment_in_order(self) -> None:
        entries = [["approach"], ["idle", "random"], "approach", ["random"]]
        result = run(entries)
        self.assertEqual(result, [["approach"], ["idle", "random"], ["approach"], ["random"]])

    def test_more_environments_than_the_maximum_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 2"):
            run([["approach"], ["idle"], ["random"]], max_envs=2)

    def test_an_empty_list_an_empty_entry_or_an_unknown_name_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            run([])
        with self.assertRaises(ValueError):
            run([[]])
        with self.assertRaisesRegex(ValueError, "nonsense"):
            run([["nonsense"]])

    def test_an_entry_with_more_opponents_than_slots_is_an_error(self) -> None:
        run([["idle"] * SLOTS])
        with self.assertRaisesRegex(ValueError, "max_opponents"):
            run([["idle"] * (SLOTS + 1)])
        with self.assertRaises(ValueError):
            run([["idle", "idle"]], max_opponents=1)

    def test_the_perfect_parry_opponent_can_be_trained_against(self) -> None:
        self.assertEqual(run([["parrier"], ["idle"]]), [["parrier"], ["idle"]])

    def test_a_null_slot_is_drawn_and_a_null_entry_is_a_whole_mixture(self) -> None:
        self.assertEqual(run([["parrier", None], None, [None]]), [["parrier", None], None, [None]])
        with self.assertRaisesRegex(ValueError, "nonsense"):
            run([["nonsense", None]])
        with self.assertRaisesRegex(ValueError, "max_opponents"):
            run([[None] * (SLOTS + 1)])

    def test_the_random_pool_defaults_to_every_opponent_and_rejects_unknown_names(self) -> None:
        self.assertIn("dodger", random_opponent_pool([]))
        self.assertEqual(random_opponent_pool(["idle", "approach"]), ["idle", "approach"])
        with self.assertRaisesRegex(ValueError, "nonsense"):
            random_opponent_pool(["nonsense"])

    def test_a_mixture_draws_only_the_empty_slots(self) -> None:
        mixture = OpponentMixture(("parrier", None, None), ("idle", "approach"), SLOTS)
        self.assertEqual(mixture.label, "parrier+*+*")
        draws = [mixture.draw(random.Random(seed)) for seed in range(30)]
        self.assertTrue(all(draw[0] == "parrier" and len(draw) == 3 for draw in draws))
        self.assertEqual({name for draw in draws for name in draw[1:]}, {"idle", "approach"})

    def test_a_whole_mixture_also_draws_how_many_opponents(self) -> None:
        mixture = OpponentMixture(None, ("idle",), SLOTS)
        self.assertEqual(mixture.label, "random")
        counts = {len(mixture.draw(random.Random(seed))) for seed in range(60)}
        self.assertEqual(counts, set(range(1, SLOTS + 1)))

    def test_the_shipped_config_is_valid(self) -> None:
        training = load_config().training
        self.assertEqual(
            len(run(training.opponents, training.num_envs)), len(training.opponents)
        )
        self.assertTrue(dataclasses.is_dataclass(training))
        random_opponent_pool(training.random_opponent_pool)


if __name__ == "__main__":
    unittest.main()
