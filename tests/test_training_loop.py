import signal
import unittest

import torch

from arena_ai.config import TrainingConfig, load_config
from arena_ai.envs.training_envs.one_v_one_env import OneVOneEnv
from arena_ai.envs.vector_env import VectorEnv
from arena_ai.training.train import TrainingLoop


class _StubPolicy:
    def act(self, observation):
        return 0, -1.0

    def value(self, observation):
        return 0.0


class _ValuePolicy:
    """Values each observation by its first number, and counts the value calls."""

    def __init__(self) -> None:
        self.value_calls = 0

    def act(self, observation):
        return 0, -1.0

    def value(self, observation):
        self.value_calls += 1
        return float(observation[0])


class _BatchedValuePolicy(_ValuePolicy):
    def act(self, observations):
        return [0] * len(observations), [-1.0] * len(observations)

    def act_and_value(self, observations):
        actions, log_probs = self.act(observations)
        return actions, log_probs, self.value(observations)

    def value(self, observations):
        self.value_calls += 1
        return [float(observation[0]) for observation in observations]


def _short_episodes(count: int) -> OneVOneEnv:
    return OneVOneEnv(load_config(), max_episode_steps=count)


class ValueReuseTests(unittest.TestCase):
    """The value of a step's next observation is reused as the next step's value."""

    def _check(self, batch) -> None:
        # Every stored value must be the value of its own observation, whether it was
        # computed fresh (the first step, after a reset) or carried over.
        torch.testing.assert_close(batch.values, batch.observations[..., 0])
        # And a step that did not end bootstraps from the next observation's value.
        continuing = ~batch.episode_ends[:-1]
        torch.testing.assert_close(
            batch.next_values[:-1][continuing], batch.observations[1:, ..., 0][continuing]
        )

    def test_scalar_rollout_values_stay_correct_across_resets(self) -> None:
        policy = _ValuePolicy()
        loop = TrainingLoop(_short_episodes(4), policy, lambda *a, **k: {}, device="cpu")
        observation, _ = loop.environment.reset(seed=1)
        batch, _, _ = loop.collect_rollout(observation, 11)
        self._check(batch)
        self.assertGreater(batch.episode_ends.sum(), 0)

    def test_scalar_rollout_makes_one_value_pass_per_step_without_resets(self) -> None:
        policy = _ValuePolicy()
        loop = TrainingLoop(_short_episodes(1000), policy, lambda *a, **k: {}, device="cpu")
        observation, _ = loop.environment.reset(seed=1)
        loop.collect_rollout(observation, 20)
        self.assertEqual(policy.value_calls, 20 + 1)  # Was 2 * 20 + 1 before.

    def test_vector_rollout_values_stay_correct_across_resets(self) -> None:
        policy = _BatchedValuePolicy()
        environment = VectorEnv([lambda: _short_episodes(4), lambda: _short_episodes(6)])
        loop = TrainingLoop(environment, policy, lambda *a, **k: {}, device="cpu")
        observations, _ = environment.reset(seed=1)
        batch, _, _ = loop.collect_vector_rollout(observations, 13)
        self._check(batch)
        self.assertGreater(batch.episode_ends.sum(), 0)

    def test_vector_rollout_makes_one_value_pass_per_step_without_resets(self) -> None:
        policy = _BatchedValuePolicy()
        environment = VectorEnv([lambda: _short_episodes(1000)] * 2)
        loop = TrainingLoop(environment, policy, lambda *a, **k: {}, device="cpu")
        observations, _ = environment.reset(seed=1)
        loop.collect_vector_rollout(observations, 20)
        self.assertEqual(policy.value_calls, 20 + 1)

    def test_the_timing_split_adds_up(self) -> None:
        policy = _ValuePolicy()
        loop = TrainingLoop(_short_episodes(1000), policy, lambda *a, **k: {}, device="cpu")
        observation, _ = loop.environment.reset(seed=1)
        loop.collect_rollout(observation, 20)
        self.assertGreater(loop._act_seconds, 0.0)
        self.assertGreater(loop._value_seconds, 0.0)
        self.assertGreater(loop._env_seconds, 0.0)


class InterruptTests(unittest.TestCase):
    def test_ctrl_c_during_an_update_waits_for_the_update_to_finish(self) -> None:
        finished = []

        def update(batch, last_value, **kwargs):
            signal.raise_signal(signal.SIGINT)  # Ctrl+C in the middle of the update.
            finished.append(True)
            return {}

        handler_before = signal.getsignal(signal.SIGINT)
        loop = TrainingLoop(
            OneVOneEnv(load_config()), _StubPolicy(), update, device="cpu"
        )
        config = TrainingConfig(
            iterations=3, rollout_steps=4, checkpoint_interval=10, seed=1
        )
        with self.assertRaises(KeyboardInterrupt):
            loop.run(config)
        self.assertEqual(finished, [True])  # The update ran to its end.
        self.assertEqual(loop.state.iteration, 1)  # And it was counted.
        self.assertIs(signal.getsignal(signal.SIGINT), handler_before)


if __name__ == "__main__":
    unittest.main()
