"""Wrap the two-fighter simulation in a reset/step environment interface."""

from collections.abc import Callable
from typing import Any

from arena_ai.agents.action import to_fighter_action
from arena_ai.config import Config
from arena_ai.sim.observation import get_observation
from arena_ai.sim.observation_vector import encode_observation
from arena_ai.sim.simulate import step_game
from arena_ai.sim.state import FighterState, create_initial_fighters


class ArenaEnv:
    """Coordinate episode state, actions, simulation, observations, and timing."""

    def __init__(
        self,
        config: Config,
        opponent_policy: Callable[[list[float]], int] | None = None,
        max_episode_steps: int | None = None,
    ) -> None:
        """Initialize the environment with configuration,
        opponent policy, and episode length."""
        self.config = config
        self.opponent_policy = opponent_policy
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

    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        """Reset the deterministic episode and return its initial observation."""
        del seed  # Initial state is deterministic until randomized starts are added.
        self.fighters = create_initial_fighters(self.config.game)
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

        # Get the opponent's action based on the current observation and step the game.
        current_observation = self.get_observation()
        opponent_action = (
            self.opponent_policy(current_observation)
            if self.opponent_policy is not None
            else 0
        )
        # Get self action and opponent action as FighterAction objects and step the game.
        actions = {
            "agent": to_fighter_action(action),
            "opponent": to_fighter_action(opponent_action),
        }
        step_game(self.fighters, actions, self.config)
        self.step_count += 1

        # Terminate if max steps reached or game ended.
        terminated = False
        truncated = self.step_count >= self.max_episode_steps
        self._episode_done = terminated or truncated

        return self.get_observation(), 0.0, terminated, truncated, {}

    def get_observation(self) -> list[float]:
        """Return the current flattened and normalized observation vector."""
        if self.fighters is None:
            raise RuntimeError("reset() must be called before getting an observation")
        agent = self.fighters["agent"]
        # Create a dictionary of the other fighters (opponents) for the observation function.
        opponents = {
            fighter_id: fighter
            for fighter_id, fighter in self.fighters.items()
            if fighter_id != "agent"
        }
        structured = get_observation(agent, opponents, self.config)
        return encode_observation(structured, self.config)
