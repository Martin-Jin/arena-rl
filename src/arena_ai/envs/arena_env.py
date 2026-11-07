"""Define the shared lifecycle for headless reinforcement-learning environments."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from arena_ai.config import Config
from arena_ai.sim.observation import get_observation
from arena_ai.sim.observation_vector import encode_observation
from arena_ai.sim.simulate import FighterAction, step_game
from arena_ai.sim.state import FighterState
from arena_ai.envs.reward import calculate_reward


class ArenaEnv(ABC):
    """Coordinate episode state, simulation, observations, and timing."""

    def __init__(
        self,
        config: Config,
        max_episode_steps: int | None = None,
    ) -> None:
        """Initialize shared environment configuration and episode bookkeeping."""
        self.config = config
        self.max_episode_steps = (
            config.game.world.sim_hz * 10
            if max_episode_steps is None
            else max_episode_steps
        )
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        self.fighters: dict[str, FighterState] | None = None
        self.step_count = 0
        self._episode_done = False

    @property
    def episode_done(self) -> bool:
        """Return whether the current episode must be reset."""
        return self._episode_done

    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        """Reset the scenario and return its initial observation."""
        del seed  # Randomized starts can be added by concrete scenarios later.
        self.fighters = self.create_initial_fighters()
        self.step_count = 0
        self._episode_done = False
        return self.get_observation(), {}

    def step(
        self,
        action: int,
    ) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        """Advance one timestep and return observation, reward, and episode status."""
        if self.fighters is None:
            raise RuntimeError("reset() must be called before step()")
        if self._episode_done:
            raise RuntimeError("reset() must be called after an episode ends")

        current_observation = self.get_observation()
        actions = self.get_actions(action, current_observation)
        previous_health = {
            fighter_id: fighter.health for fighter_id, fighter in self.fighters.items()
        }
        step_game(self.fighters, actions, self.config)
        self.step_count += 1

        terminated = self.is_terminated()
        truncated = self.step_count >= self.max_episode_steps and not terminated
        self._episode_done = terminated or truncated

        reward = self.calculate_reward(previous_health)
        winner = None
        if terminated and self.fighters["agent"].health > 0:
            winner = "agent"
        elif terminated:
            winner = "opponent"
        info = {"step_count": self.step_count, "winner": winner}
        return self.get_observation(), reward, terminated, truncated, info

    def get_observation(self) -> list[float]:
        """Return the current flattened and normalized observation vector."""
        if self.fighters is None:
            raise RuntimeError("reset() must be called before getting an observation")
        agent = self.fighters["agent"]
        opponents = {
            fighter_id: fighter
            for fighter_id, fighter in self.fighters.items()
            if fighter_id != "agent"
        }
        structured = get_observation(agent, opponents, self.config)
        return encode_observation(structured, self.config)

    @abstractmethod
    def create_initial_fighters(self) -> dict[str, FighterState]:
        """Create the initial fighter states for this scenario."""

    @abstractmethod
    def get_actions(
        self,
        agent_action: int,
        observation: list[float],
    ) -> Mapping[str, FighterAction]:
        """Convert scenario controller decisions into ID-keyed simulation actions."""

    def calculate_reward(self, previous_health: Mapping[str, int]) -> float:
        """Delegate to the shared reward module."""
        if self.fighters is None:
            return 0.0
        return calculate_reward(previous_health, self.fighters, self.config)

    def is_terminated(self) -> bool:
        """Return whether the scenario has reached a natural terminal state."""
        if self.fighters is None:
            return False
        return any(fighter.health <= 0 for fighter in self.fighters.values())
