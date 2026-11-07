import signal
import unittest

from arena_ai.config import TrainingConfig, load_config
from arena_ai.envs.training_envs.one_v_one_env import OneVOneEnv
from arena_ai.training.train import TrainingLoop


class _StubPolicy:
    def act(self, observation):
        return 0, -1.0

    def value(self, observation):
        return 0.0


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
