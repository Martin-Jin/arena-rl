import dataclasses
import unittest

from arena_ai.agents.scripted import OPPONENTS, training_opponents
from arena_ai.config import load_config


class TrainingOpponentsTests(unittest.TestCase):
    def test_each_entry_becomes_one_environment_in_order(self) -> None:
        names = ["approach", "idle", "approach", "random"]
        result = training_opponents(names, max_envs=4)
        self.assertEqual([name for name, _ in result], names)
        self.assertIs(result[1][1], OPPONENTS["idle"])

    def test_more_opponents_than_the_maximum_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 2"):
            training_opponents(["approach", "idle", "random"], max_envs=2)

    def test_an_empty_list_or_unknown_name_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            training_opponents([], max_envs=4)
        with self.assertRaisesRegex(ValueError, "nonsense"):
            training_opponents(["nonsense"], max_envs=4)

    def test_the_perfect_parry_opponent_can_be_trained_against(self) -> None:
        names = [name for name, _ in training_opponents(["parrier", "idle"], max_envs=4)]
        self.assertEqual(names, ["parrier", "idle"])

    def test_the_shipped_config_is_valid(self) -> None:
        training = load_config().training
        self.assertEqual(
            len(training_opponents(training.opponents, training.num_envs)),
            len(training.opponents),
        )
        self.assertTrue(dataclasses.is_dataclass(training))


if __name__ == "__main__":
    unittest.main()
