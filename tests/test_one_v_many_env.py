import unittest

from arena_ai.agents.action import Action
from arena_ai.agents.scripted import approach_and_attack, idle, parrying_hunter
from arena_ai.config import load_config
from arena_ai.envs.training_envs.one_v_many_env import OneVManyEnv, matchup_label
from arena_ai.sim.observation_vector import fighter_features, opponent_mask
from arena_ai.sim.state import enemies_of

CONFIG = load_config()
SLOTS = CONFIG.game.observation.max_opponents
WIDTH = CONFIG.game.world.arena_width_m


def env(names, policy=idle, steps=None):
    return OneVManyEnv(CONFIG, [(name, policy) for name in names], max_episode_steps=steps)


class OneVManyEnvTests(unittest.TestCase):
    def test_the_matchup_label_sorts_and_joins_the_names(self) -> None:
        self.assertEqual(matchup_label(["runner", "approach", "idle"]), "approach+idle+runner")
        self.assertEqual(matchup_label(["idle"]), "idle")

    def test_it_creates_one_fighter_per_opponent_and_reports_the_matchup(self) -> None:
        e = env(["idle", "approach", "idle"], steps=2)
        observation, _ = e.reset(seed=1)
        self.assertEqual(len(e.fighters), 4)
        self.assertEqual(opponent_mask(observation, SLOTS), [1.0, 1.0, 1.0])
        _, _, _, _, info = e.step(0)
        self.assertEqual(info["opponent"], "approach+idle+idle")
        self.assertEqual(info["opponent_count"], 3)

    def test_the_episode_ends_only_when_every_opponent_is_dead(self) -> None:
        e = env(["idle", "idle"])
        e.reset(seed=1)
        e.fighters["opponent"].health = 0
        _, _, terminated, _, _ = e.step(0)
        self.assertFalse(terminated)
        e.fighters["opponent_1"].health = 0
        _, _, terminated, _, info = e.step(0)
        self.assertTrue(terminated)
        self.assertEqual(info["winner"], "agent")

    def test_each_opponent_sees_the_agent_then_itself_then_its_allies(self) -> None:
        seen = []

        def spy(observation):
            seen.append(observation)
            return int(Action.IDLE)

        e = env(["idle", "idle", "idle"], policy=spy)
        e.reset(seed=1)
        e.step(0)
        agent = e.fighters["agent"]
        enemies = enemies_of("agent", e.fighters)
        self.assertEqual(len(seen), len(enemies))
        for view, fighter in zip(seen, enemies.values()):
            self.assertAlmostEqual(fighter_features(view, 0)["x"] * WIDTH, agent.x, places=5)
            self.assertAlmostEqual(fighter_features(view, 1)["x"] * WIDTH, fighter.x, places=5)
            self.assertEqual(opponent_mask(view, SLOTS), [1.0, 1.0, 1.0])
            allies = [f for f in enemies.values() if f is not fighter]
            seen_allies = sorted(fighter_features(view, slot)["x"] * WIDTH for slot in (2, 3))
            for got, expected in zip(seen_allies, sorted(f.x for f in allies)):
                self.assertAlmostEqual(got, expected, places=4)

    def test_a_dead_opponent_is_not_asked_for_an_action(self) -> None:
        calls = []

        def policy(observation):
            calls.append(1)
            return int(Action.IDLE)

        e = env(["idle", "idle"], policy=policy)
        e.reset(seed=1)
        e.fighters["opponent"].health = 0
        e.step(0)
        self.assertEqual(len(calls), 1)

    def test_too_many_or_no_opponents_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            env([])
        with self.assertRaises(ValueError):
            env(["idle"] * (SLOTS + 1))

    def test_a_three_opponent_episode_runs_with_real_policies(self) -> None:
        e = OneVManyEnv(
            CONFIG,
            [("a", approach_and_attack), ("p", parrying_hunter), ("i", idle)],
            max_episode_steps=300,
        )
        e.reset(seed=2)
        for _ in range(300):
            _, _, terminated, truncated, _ = e.step(int(Action.IDLE))
            if terminated or truncated:
                break
        self.assertTrue(e.fighters["agent"].health <= 100)
