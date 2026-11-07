import unittest

from arena_ai.agents.scripted import idle
from arena_ai.config import load_config
from arena_ai.envs.training_envs.one_v_one_env import OneVOneEnv
from arena_ai.training.stats import OpponentStats


class OpponentStatsTests(unittest.TestCase):
    def test_rolling_window_keeps_only_recent_episodes(self) -> None:
        stats = OpponentStats(window=2)
        stats.record("idle", won=False, truncated=True, length=100)
        stats.record("idle", won=True, truncated=False, length=40)
        stats.record("idle", won=True, truncated=False, length=60)
        summary = stats.summary()["idle"]
        self.assertEqual(summary["episodes"], 2.0)
        self.assertEqual(summary["win_rate"], 1.0)
        self.assertEqual(summary["truncation_rate"], 0.0)
        self.assertEqual(summary["mean_length"], 50.0)

    def test_env_reports_opponent_and_truncates_at_limit(self) -> None:
        env = OneVOneEnv(
            load_config(), opponent_policy=idle, max_episode_steps=3,
            opponent_name="idle",
        )
        env.reset(seed=1)
        for _ in range(2):
            env.step(0)
        _, _, terminated, truncated, info = env.step(0)
        self.assertTrue(truncated and not terminated)
        self.assertEqual(info["opponent"], "idle")
        self.assertEqual(info["step_count"], 3)


if __name__ == "__main__":
    unittest.main()
