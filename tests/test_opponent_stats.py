import unittest

from arena_ai.agents.action import Action
from arena_ai.agents.scripted import idle
from arena_ai.config import load_config
from arena_ai.envs.training_envs.one_v_one_env import OneVOneEnv
from arena_ai.training.stats import EpisodeRecord, EpisodeStats


def record(**overrides) -> EpisodeRecord:
    values = dict(
        won=False,
        lost=False,
        truncated=False,
        length=100,
        episode_return=0.0,
        damage_dealt=0,
        damage_taken=0,
        swings=0,
        hits=0,
        edge_steps=0,
    )
    values.update(overrides)
    return EpisodeRecord(**values)


class EpisodeStatsTests(unittest.TestCase):
    def test_rolling_window_keeps_only_recent_episodes(self) -> None:
        stats = EpisodeStats(window=2)
        stats.record("idle", record(truncated=True, length=100))
        stats.record("idle", record(won=True, length=40))
        stats.record("idle", record(won=True, length=60))
        summary = stats.by_opponent()["idle"]
        self.assertEqual(summary["episodes"], 2.0)
        self.assertEqual(summary["win_rate"], 1.0)
        self.assertEqual(summary["truncation_rate"], 0.0)
        self.assertEqual(summary["mean_length"], 50.0)

    def test_running_sums_match_a_direct_recomputation(self) -> None:
        stats = EpisodeStats(window=5)
        records = [
            record(
                won=i % 3 == 0,
                lost=i % 3 == 1,
                length=50 + i,
                episode_return=i * 0.5,
                damage_dealt=i,
                damage_taken=2 * i,
                swings=i + 1,
                hits=i // 2,
                edge_steps=i,
            )
            for i in range(23)
        ]
        for item in records:
            stats.record("approach", item)
        recent = records[-5:]
        overall = stats.overall()
        self.assertAlmostEqual(
            overall["mean_return"], sum(r.episode_return for r in recent) / 5
        )
        self.assertAlmostEqual(
            overall["damage_taken"], sum(r.damage_taken for r in recent) / 5
        )
        self.assertAlmostEqual(
            overall["attack_accuracy"],
            sum(r.hits for r in recent) / sum(r.swings for r in recent),
        )
        wins = [r for r in recent if r.won]
        self.assertAlmostEqual(
            overall["win_length"], sum(r.length for r in wins) / len(wins)
        )
        self.assertAlmostEqual(
            overall["wall_fraction"],
            sum(r.edge_steps for r in recent) / sum(r.length for r in recent),
        )

    def test_overall_combines_every_opponent_and_empty_is_blank(self) -> None:
        stats = EpisodeStats(window=10)
        self.assertEqual(stats.overall(), {})
        stats.record("idle", record(won=True))
        stats.record("random", record())
        self.assertEqual(stats.overall()["win_rate"], 0.5)
        self.assertEqual(stats.by_opponent()["idle"]["win_rate"], 1.0)


class EnvInfoTests(unittest.TestCase):
    def test_env_reports_opponent_and_truncates_at_limit(self) -> None:
        env = OneVOneEnv(
            load_config(),
            opponent_policy=idle,
            max_episode_steps=3,
            opponent_name="idle",
        )
        env.reset(seed=1)
        for _ in range(2):
            env.step(0)
        _, _, terminated, truncated, info = env.step(0)
        self.assertTrue(truncated and not terminated)
        self.assertEqual(info["opponent"], "idle")
        self.assertEqual(info["step_count"], 3)

    def test_env_counts_swings_hits_and_damage(self) -> None:
        env = OneVOneEnv(load_config(), opponent_policy=idle, opponent_name="idle")
        env.reset(seed=1)
        env.fighters["agent"].x = 5.5
        env.fighters["agent"].facing_dir = 1
        env.fighters["opponent"].x = 6.0
        info = {}
        for _ in range(17):  # Longer than the wind-up, so the swing connects.
            _, _, _, _, info = env.step(int(Action.ATTACK))
        self.assertGreaterEqual(info["swings"], 1)
        self.assertEqual(info["hits"], 1)
        self.assertEqual(info["damage_dealt"], 10)
        self.assertEqual(info["damage_taken"], 0)


if __name__ == "__main__":
    unittest.main()
