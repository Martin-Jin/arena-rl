"""Orchestrate rollout collection and policy updates.

The policy and optimizer are deliberately injected.  This keeps the training
loop independent from a particular network implementation while the agent and
PPO modules are still being built.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

import torch

from arena_ai.config import TrainingConfig, load_config
from arena_ai.envs.arena_env import ArenaEnv
from arena_ai.training.rollout import Rollout, RolloutBatch


class Policy(Protocol):
    """Act in an environment and estimate values for observations."""

    def act(self, observation: list[float]) -> tuple[int, float]:
        """Return ``(action, log_probability)`` for one observation."""

    def value(self, observation: list[float]) -> float:
        """Return the critic estimate for one observation."""


class UpdatePolicy(Protocol):
    """Apply one PPO update using explicit bootstrap and episode masks."""

    def __call__(
        self,
        batch: RolloutBatch,
        last_value: float,
        *,
        terminated: torch.Tensor,
        episode_ends: torch.Tensor,
        next_values: torch.Tensor,
    ) -> Mapping[str, float]:
        """Compute advantages/returns and update the actor and critic."""
# A function that saves a checkpoint given the current training state.
SaveCheckpoint = Callable[["TrainingState"], None]


def _require_python_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"policy {name} must return a Python int")
    return value


def _require_python_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError(f"policy {name} must return a Python float")
    return value


@dataclass
class TrainingState:
    """Progress and metrics passed to checkpoint callbacks."""

    iteration: int = 0
    environment_steps: int = 0
    episodes: int = 0
    wins: int = 0
    episode_return: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


class TrainingLoop:
    """Collect fresh on-policy data, update the policy, and save progress."""

    def __init__(
        self,
        environment: ArenaEnv,
        policy: Policy,
        update_policy: UpdatePolicy,
        *,
        save_checkpoint: SaveCheckpoint | None = None,
    ) -> None:
        # One training loop consists of the:
        # - the environment in which the agent is learning
        # - the policy being trained
        # - the function that updates the policy given a batch of rollouts
        # - the function that saves a checkpoint given the current training state
        # - the current training state, which is updated after each iteration
        self.environment = environment
        self.policy = policy
        self.update_policy = update_policy
        self.save_checkpoint = save_checkpoint
        self.state = TrainingState()
        self._current_episode_return = 0.0

    def collect_rollout(
        self,
        observation: list[float],
        steps: int,
    ) -> tuple[RolloutBatch, list[float], float]:
        """Collect exactly ``steps`` transitions in the environment without
        changing the policy.

        The step budget is not an episode: the rollout spans as many episodes as
        it takes, resetting the environment after each terminated or truncated
        step and continuing, and it can end mid-episode.
        """
        rollout = Rollout()
        # next_values[t] is the critic's value V(s_{t+1}) of the state reached after step t.
        # Used for the bootstrap term in GAE. At a truncated step it is V(final_observation),
        # and at the end of the rollout it is V(obs) of the observation carried forward.
        next_values: list[float] = []
        with torch.no_grad():
            for _ in range(steps):
                action_result = self.policy.act(observation)
                action = _require_python_int(action_result[0], "act action")
                log_probability = _require_python_float(
                    action_result[1], "act log_probability"
                )
                # Calculated the critic value for the current observation before
                # stepping the environment.
                value = _require_python_float(
                    self.policy.value(observation), "value"
                )
                next_observation, reward, terminated, truncated, info = (
                    self.environment.step(action)
                )
                self._current_episode_return += reward
                final_observation = next_observation
                # Calculate the value of the final observation for bootstrapping. This is
                # done after the step to ensure that the final observation is not
                # overwritten by a subsequent reset.
                next_value = _require_python_float(
                    self.policy.value(final_observation), "value"
                )
                rollout.add(
                    observation,
                    action,
                    reward,
                    terminated,
                    truncated,
                    value,
                    log_probability,
                )
                next_values.append(next_value)
                self.state.environment_steps += 1
                observation = next_observation
                # Evaluate the terminal or truncated observation before reset.
                # The value is retained in next_values for correct bootstrapping.
                if terminated or truncated:
                    self.state.episodes += 1
                    self.state.episode_return = self._current_episode_return
                    if info.get("winner") == "agent":
                        self.state.wins += 1
                    self._current_episode_return = 0.0
                    observation, _ = self.environment.reset()
        batch = rollout.as_batch()
        batch = replace(
            batch,
            next_values=torch.tensor(
                next_values,
                dtype=batch.values.dtype,
                device=batch.values.device,
            ),
        )
        with torch.no_grad():
            last_value = _require_python_float(
                self.policy.value(observation), "value"
            )
        return batch, observation, last_value

    def run(self, config: TrainingConfig) -> TrainingState:
        """Run the training loop: one rollout plus one update per iteration.

        An iteration is not an episode. Each iteration collects exactly
        ``rollout_steps`` environment steps, spanning as many episodes as it
        takes, and then updates the policy once.
        """
        if config.iterations < 1:
            raise ValueError("iterations must be positive")
        if config.rollout_steps < 1:
            raise ValueError("rollout_steps must be positive")
        if config.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        observation, _ = self.environment.reset(seed=config.seed)
        # Each iteration collects exactly rollout_steps environment steps and then
        # updates the policy once. The rollout spans as many episodes as it takes
        # and can end mid-episode, so it is a step budget, not a game.
        for iteration in range(1, config.iterations + 1):
            batch, observation, last_value = self.collect_rollout(
                observation,
                config.rollout_steps,
            )
            self.state.metrics = dict(
                self.update_policy(
                    batch,
                    last_value,
                    terminated=batch.terminated,
                    episode_ends=batch.episode_ends,
                    next_values=batch.next_values,
                )
            )
            # Update the training state with the latest metrics and progress.
            self.state.metrics.update(
                {
                    "episode_return": self.state.episode_return,
                    "win_rate": (
                        self.state.wins / self.state.episodes
                        if self.state.episodes
                        else 0.0
                    ),
                    "episodes": float(self.state.episodes),
                    "wins": float(self.state.wins),
                }
            )
            # update the iteration count in the training state after each iteration
            self.state.iteration = iteration
            # Save a checkpoint if the save_checkpoint callback is provided and
            # the current iteration is a multiple of the checkpoint interval.
            if (
                self.save_checkpoint is not None
                and iteration % config.checkpoint_interval == 0
            ):
                self.save_checkpoint(self.state)
        # Return the final training state after all iterations are complete.
        return self.state


def train(
    environment_factory: Callable[[], ArenaEnv],
    policy: Policy,
    update_policy: UpdatePolicy,
    config: TrainingConfig | None = None,
    *,
    save_checkpoint: SaveCheckpoint | None = None,
) -> TrainingState:
    """Build and run a training loop for one learner-versus-opponent environment."""
    loop = TrainingLoop(
        environment_factory(),
        policy,
        update_policy,
        save_checkpoint=save_checkpoint,
    )
    return loop.run(config or load_config().training)
