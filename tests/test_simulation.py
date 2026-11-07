import random
import unittest

from arena_ai.agents.action import (
    ACTION_COUNT,
    Action,
    from_parts,
    to_fighter_action,
)
from arena_ai.config import load_config
from arena_ai.sim.observation_vector import observation_size
from arena_ai.envs.training_envs.one_v_one_env import OneVOneEnv


class SimulationTests(unittest.TestCase):
    def test_action_index_round_trips_through_fighter_action(self) -> None:
        for index in range(ACTION_COUNT):
            action = to_fighter_action(index)
            self.assertEqual(
                from_parts(action.move_dir, action.jump, action.attack, action.parry),
                index,
            )
        self.assertEqual(ACTION_COUNT, len(Action))

    def test_same_seed_and_actions_give_identical_episodes(self) -> None:
        config = load_config()

        def run() -> list:
            env = OneVOneEnv(config)
            observation, _ = env.reset(seed=123)
            rng = random.Random(7)
            trace = [observation]
            for _ in range(120):
                observation, reward, terminated, truncated, _ = env.step(
                    rng.randrange(ACTION_COUNT)
                )
                trace.append((observation, reward))
                if terminated or truncated:
                    break
            return trace

        first, second = run(), run()
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), observation_size())


if __name__ == "__main__":
    unittest.main()
