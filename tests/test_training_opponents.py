import dataclasses
import unittest

from arena_ai.agents.scripted import training_opponents
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

    def test_the_shipped_config_is_valid(self) -> None:
        training = load_config().training
        self.assertEqual(
            len(run(training.opponents, training.num_envs)), len(training.opponents)
        )
        self.assertTrue(dataclasses.is_dataclass(training))


if __name__ == "__main__":
    unittest.main()
