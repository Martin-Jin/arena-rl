"""Synchronous vectorized environment management for on-policy collection."""

from collections.abc import Callable, Sequence
from typing import Any

from arena_ai.envs.arena_env import ArenaEnv


class VectorEnv:
    """Run independent environment instances in one synchronous batch."""

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
            observation, reward, ended, timed_out, info = env.step(action)
            if ended or timed_out:
                info = {
                    **info,
                    "final_observation": observation,
                    "final_info": info.copy(),
                }
                observation, reset_info = env.reset()
                info["reset_info"] = reset_info
            observations.append(observation)
            rewards.append(reward)
            terminated.append(ended)
            truncated.append(timed_out)
            infos.append(info)
        return observations, rewards, terminated, truncated, infos
