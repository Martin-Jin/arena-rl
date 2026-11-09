"""Define the shared lifecycle for headless reinforcement-learning environments."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from arena_ai.config import Config
from arena_ai.sim.observation import get_observation
from arena_ai.sim.observation_vector import encode_observation
from arena_ai.sim.outcome import is_over, winner
from arena_ai.sim.simulate import FighterAction, step_game
from arena_ai.sim.state import AGENT_ID, FighterState, enemies_of
from arena_ai.envs.reward import calculate_reward, wall_distance


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
        self._reset_counters()

    def _reset_counters(self) -> None:
        """Zero the per-episode counters that are reported in ``info``."""
        self.damage_dealt = 0
        self.damage_taken = 0
        self.swings = 0  # Attacks the agent started.
        self.hits = 0  # Steps in which the agent damaged an opponent.
        self.edge_steps = 0  # Steps within the edge margin of a wall.
        self.parry_attempts = 0  # Parries the agent started.
        self.parries = 0  # Parries of the agent that stopped an attack.

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
        self._reset_counters()
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

        actions = self.get_actions(action)
        previous_health = {
            fighter_id: fighter.health for fighter_id, fighter in self.fighters.items()
        }
        attack_frame_before = self.fighters[AGENT_ID].attack_frame
        parry_frame_before = self.fighters[AGENT_ID].parry_frame
        step_game(self.fighters, actions, self.config)
        self.step_count += 1
        self._track_episode(previous_health, attack_frame_before, parry_frame_before)

        terminated = self.is_terminated()
        truncated = self.step_count >= self.max_episode_steps and not terminated
        self._episode_done = terminated or truncated

        reward = self.calculate_reward(previous_health)
        # A time-out is not a kill, so "winner" stays None for it. It still earns a small
        # reward when the agent is ahead on health, and is reported as "timeout_win".
        timeout_win = truncated and winner(self.fighters, timed_out=True) == AGENT_ID
        if timeout_win:
            reward += self.config.scoring.timeout_win_reward
        info = {
            "step_count": self.step_count,
            "winner": winner(self.fighters),
            "timeout_win": timeout_win,
            "opponent_count": len(self.fighters) - 1,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "swings": self.swings,
            "hits": self.hits,
            "edge_steps": self.edge_steps,
            "parry_attempts": self.parry_attempts,
            "parries": self.parries,
        }
        return self.get_observation(), reward, terminated, truncated, info

    def _track_episode(
        self,
        previous_health: Mapping[str, int],
        attack_frame_before: int,
        parry_frame_before: int,
    ) -> None:
        """Update the cheap per-episode counters after one simulation step."""
        assert self.fighters is not None
        agent = self.fighters[AGENT_ID]
        if attack_frame_before == 0 and agent.attack_frame == 1:
            self.swings += 1
        if parry_frame_before == 0 and agent.parry_frame == 1:
            self.parry_attempts += 1
        self.parries += agent.parried
        dealt = sum(
            previous_health[fighter_id] - fighter.health
            for fighter_id, fighter in enemies_of(AGENT_ID, self.fighters).items()
        )
        if dealt > 0:
            self.hits += 1
            self.damage_dealt += dealt
        self.damage_taken += previous_health[AGENT_ID] - agent.health
        if wall_distance(agent, self.config) < self.config.scoring.edge_margin:
            self.edge_steps += 1

    def get_observation(self) -> list[float]:
        """Return the current flattened and normalized observation vector."""
        if self.fighters is None:
            raise RuntimeError("reset() must be called before getting an observation")
        agent = self.fighters[AGENT_ID]
        opponents = enemies_of(AGENT_ID, self.fighters)
        structured = get_observation(agent, opponents, self.config)
        return encode_observation(structured, self.config)

    @abstractmethod
    def create_initial_fighters(self) -> dict[str, FighterState]:
        """Create the initial fighter states for this scenario."""

    @abstractmethod
    def get_actions(
        self,
        agent_action: int,
    ) -> Mapping[str, FighterAction]:
        """Convert scenario controller decisions into ID-keyed simulation actions."""

    def calculate_reward(self, previous_health: Mapping[str, int]) -> float:
        """Delegate to the shared reward module."""
        if self.fighters is None:
            return 0.0
        return calculate_reward(previous_health, self.fighters, self.config)

    def is_terminated(self) -> bool:
        """Return whether the agent or every opponent is dead."""
        if self.fighters is None:
            return False
        return is_over(self.fighters)
