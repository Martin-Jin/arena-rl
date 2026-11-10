"""Vectorized environments that step in worker processes, for faster rollouts.

**Why:** stepping the simulation is about half of the rollout time, and ``VectorEnv`` steps
its environments one after another on one core. **What:** ``AsyncVectorEnv`` splits the
environments over worker processes. Each worker owns its environments for the whole run (they
hold the state of a fight), and every ``step`` sends each worker its actions, lets them all
step at once, and collects the results in the original order. **Why processes:** the
simulation is pure Python, so threads would take turns on the interpreter lock.
**Limits:** it is a drop-in for ``VectorEnv`` in training, but its environments live in
other processes, so there is no ``envs`` list for the training visualizer. Factories are
sent to the workers, so they must be picklable (module-level callables and ``partial``, not
lambdas).
"""

from collections.abc import Callable, Sequence
import multiprocessing
import signal
import traceback
from typing import Any

from arena_ai.envs.arena_env import ArenaEnv
from arena_ai.envs.vector_env import VectorEnv, step_and_reset

# Seconds to wait for a worker to exit after it is told to close, before it is killed.
CLOSE_TIMEOUT_S = 5.0


def _worker(connection, factories: Sequence[Callable[[], ArenaEnv]]) -> None:
    """Own some environments and answer the main process's commands until told to close.

    It ignores ``Ctrl+C``: on Windows the console sends it to every process in the group, so
    workers that exited on it closed their pipes under the main process, which then failed
    with a ``BrokenPipeError`` before it could save a checkpoint. Now only the main process
    handles it, and it closes the workers itself.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        envs = [factory() for factory in factories]
        connection.send(("ok", None))
        while True:
            command, payload = connection.recv()
            if command == "reset":
                connection.send(("ok", [env.reset(seed=seed) for env, seed in zip(envs, payload)]))
            elif command == "step":
                connection.send(
                    ("ok", [step_and_reset(env, action) for env, action in zip(envs, payload)])
                )
            else:  # "close"
                break
    except (EOFError, KeyboardInterrupt):
        pass  # The main process went away or was interrupted: just exit.
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


class AsyncVectorEnv(VectorEnv):
    """Run independent environments in worker processes, with the ``VectorEnv`` interface.

    Environment ``i`` goes to worker ``i % workers``, so environments of different cost
    (an opponent list is often ordered from cheap to expensive) spread evenly. Results come
    back in environment order, so a run matches ``VectorEnv`` exactly for the same seed.
    """

    def __init__(
        self, env_factories: Sequence[Callable[[], ArenaEnv]], num_workers: int
    ) -> None:
        if not env_factories:
            raise ValueError("at least one environment factory is required")
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        self._count = len(env_factories)
        workers = min(num_workers, self._count)
        # The environment indices each worker owns, in the order its results come back.
        self._indices = [list(range(worker, self._count, workers)) for worker in range(workers)]
        self._connections = []
        self._processes = []
        context = multiprocessing.get_context("spawn")  # The only start method on Windows.
        try:
            for indices in self._indices:
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker,
                    args=(child, [env_factories[index] for index in indices]),
                    daemon=True,  # Never outlives the main process.
                )
                process.start()
                child.close()  # Only the worker holds its end.
                self._connections.append(parent)
                self._processes.append(process)
            self._gather()  # Each worker reports that its environments were built.
        except BaseException:
            self.close()
            raise

    @property
    def num_envs(self) -> int:
        """Return the number of independent environments."""
        return self._count

    def reset(self, seed: int | None = None):
        """Reset every environment (environment ``i`` gets ``seed + i``)."""
        for connection, indices in zip(self._connections, self._indices):
            seeds = [None if seed is None else seed + index for index in indices]
            connection.send(("reset", seeds))
        results = self._collect()
        return [result[0] for result in results], [result[1] for result in results]

    def step(self, actions: Sequence[int]):
        """Step all environments at once and reset the ones that ended."""
        if len(actions) != self._count:
            raise ValueError("actions must contain one action per environment")
        for connection, indices in zip(self._connections, self._indices):
            connection.send(("step", [actions[index] for index in indices]))
        results = self._collect()
        # Transpose the per-environment tuples into the five parallel lists.
        return tuple(list(column) for column in zip(*results))

    def close(self) -> None:
        """Stop the workers. Safe to call more than once."""
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (OSError, ValueError):
                pass  # The worker is already gone.
        for process in self._processes:
            process.join(CLOSE_TIMEOUT_S)
            if process.is_alive():
                process.terminate()
        for connection in self._connections:
            connection.close()
        self._connections = []
        self._processes = []

    def _gather(self) -> list[Any]:
        """Return each worker's reply, raising the worker's traceback if one failed."""
        replies = [connection.recv() for connection in self._connections]
        for status, payload in replies:
            if status == "error":
                self.close()
                raise RuntimeError(f"environment worker failed:\n{payload}")
        return [payload for _, payload in replies]

    def _collect(self) -> list[Any]:
        """Return the per-environment results of one command, in environment order."""
        results: list[Any] = [None] * self._count
        for indices, worker_results in zip(self._indices, self._gather()):
            for index, result in zip(indices, worker_results):
                results[index] = result
        return results
