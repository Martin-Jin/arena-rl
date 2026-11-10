from functools import partial
import multiprocessing
import random
import signal
import unittest

from arena_ai.agents.scripted import EpisodePolicy
from arena_ai.config import load_config
from arena_ai.envs.async_vector_env import AsyncVectorEnv, _worker
from arena_ai.envs.training_envs.one_v_many_env import OneVManyEnv
from arena_ai.envs.vector_env import VectorEnv

CONFIG = load_config()
SHORT_EPISODE = 40  # Steps, so episodes end and auto-reset during the test.
NAMES = ["parrier", "approach", "random", "runner", "idle"]


def make_env(name: str) -> OneVManyEnv:
    return OneVManyEnv(CONFIG, [(name, EpisodePolicy(name, CONFIG))], SHORT_EPISODE)


def factories() -> list:
    # Module-level partials, not lambdas: the workers receive them by pickling.
    return [partial(make_env, name) for name in NAMES]


def broken_factory():
    raise RuntimeError("cannot build this environment")


class AsyncVectorEnvTests(unittest.TestCase):
    def test_a_worker_ignores_ctrl_c_so_only_the_main_process_handles_it(self) -> None:
        # Regression: workers that exited on Ctrl+C broke the main process's pipes, so it
        # failed with a BrokenPipeError instead of saving a checkpoint.
        self.addCleanup(signal.signal, signal.SIGINT, signal.getsignal(signal.SIGINT))
        parent, child = multiprocessing.Pipe()
        parent.send(("close", None))  # The worker answers "ok", then reads this and exits.
        _worker(child, [])
        self.assertEqual(signal.getsignal(signal.SIGINT), signal.SIG_IGN)
        self.assertEqual(parent.recv(), ("ok", None))

    def test_it_matches_the_synchronous_vector_env_step_for_step(self) -> None:
        sync = VectorEnv(factories())
        parallel = AsyncVectorEnv(factories(), num_workers=2)
        self.addCleanup(parallel.close)
        self.assertEqual(sync.reset(seed=3), parallel.reset(seed=3))
        rng = random.Random(0)
        episodes_ended = 0
        for _ in range(150):
            actions = [rng.randrange(24) for _ in NAMES]
            expected = sync.step(actions)
            self.assertEqual(expected, parallel.step(actions))
            episodes_ended += sum(expected[2]) + sum(expected[3])
        self.assertGreater(episodes_ended, len(NAMES))  # The auto-reset path was covered.

    def test_results_stay_in_environment_order_with_uneven_workers(self) -> None:
        sync = VectorEnv(factories())
        parallel = AsyncVectorEnv(factories(), num_workers=3)  # Workers hold 2, 2 and 1.
        self.addCleanup(parallel.close)
        self.assertEqual(parallel.num_envs, len(NAMES))
        self.assertEqual(sync.reset(seed=9), parallel.reset(seed=9))
        self.assertEqual(sync.step([0] * len(NAMES)), parallel.step([0] * len(NAMES)))

    def test_a_failing_environment_raises_in_the_main_process(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cannot build this environment"):
            AsyncVectorEnv([broken_factory], num_workers=1)

    def test_it_checks_the_action_count_and_closes_twice_safely(self) -> None:
        parallel = AsyncVectorEnv(factories()[:2], num_workers=2)
        parallel.reset(seed=1)
        with self.assertRaises(ValueError):
            parallel.step([0])
        parallel.close()
        parallel.close()


if __name__ == "__main__":
    unittest.main()
