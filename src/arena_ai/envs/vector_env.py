"""Synchronous vectorized environment management for on-policy collection."""

from collections.abc import Callable, Sequence
from typing import Any

from arena_ai.envs.arena_env import ArenaEnv


def step_and_reset(env: ArenaEnv, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    """Step one environment and, if its episode ended, reset it at once.

    The last observation and info of the ended episode go in ``info["final_observation"]``
    and ``info["final_info"]``, because PPO needs them to bootstrap, and the reset's info
    goes in ``info["reset_info"]``. Shared by ``VectorEnv`` and the worker processes of
    ``AsyncVectorEnv``, so both end episodes the same way.
    """
    observation, reward, ended, timed_out, info = env.step(action)
    if ended or timed_out:
        info = {**info, "final_observation": observation, "final_info": info.copy()}
        observation, reset_info = env.reset()
        info["reset_info"] = reset_info
    return observation, reward, ended, timed_out, info


class VectorEnv:
    """Run independent environment instances in one synchronous batch, in this process."""

    def __init__(self, env_factories: Sequence[Callable[[], ArenaEnv]]) -> None:
        if not env_factories:
            raise ValueError("at least one environment factory is required")
        self.envs = [factory() for factory in env_factories]

    @property
    def num_envs(self) -> int:
        """Return the number of independent environments."""
        return len(self.envs)

    def reset(
        self, seed: int | None = None
    ) -> tuple[list[list[float]], list[dict[str, Any]]]:
        """Reset every environment and return batched observations."""
        observations = []
        infos = []
        for index, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + index
            observation, info = env.reset(seed=env_seed)
            observations.append(observation)
            infos.append(info)
        return observations, infos

    def step(
        self, actions: Sequence[int]
    ) -> tuple[
        list[list[float]],
        list[float],
        list[bool],
        list[bool],
        list[dict[str, Any]],
    ]:
        """Step all environments and automatically reset completed ones."""
        if len(actions) != self.num_envs:
            raise ValueError("actions must contain one action per environment")

        observations: list[list[float]] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        infos: list[dict[str, Any]] = []
        for env, action in zip(self.envs, actions):
            observation, reward, ended, timed_out, info = step_and_reset(env, action)
            observations.append(observation)
            rewards.append(reward)
            terminated.append(ended)
            truncated.append(timed_out)
            infos.append(info)
        return observations, rewards, terminated, truncated, infos

    def set_opponent_weights(self, weights) -> None:
        """Give the self-play opponents the learner's weights.

        The network is shared by every environment of this process, so the first one loads it.
        """
        self.envs[0].set_opponent_weights(weights)

    def close(self) -> None:
        """Release resources. Nothing to release in this process."""
