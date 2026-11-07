"""Orchestrate rollout collection and policy updates.

The policy and optimizer are deliberately injected. This keeps the training
loop independent from a particular network implementation.
"""

import signal
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Protocol

import torch

from arena_ai.config import TrainingConfig
from arena_ai.envs.arena_env import ArenaEnv
from arena_ai.envs.vector_env import VectorEnv
from arena_ai.training.rollout import Rollout, RolloutBatch
from arena_ai.training.stats import EpisodeRecord, EpisodeStats


class Policy(Protocol):
    """Act in an environment and estimate values for observations."""

    def act(
        self, observation: list[float] | list[list[float]]
    ) -> tuple[int, float] | tuple[list[int], list[float]]:
        """Return actions and log probabilities for one or more observations."""

    def value(self, observation: list[float] | list[list[float]]) -> float | list[float]:
        """Return critic estimates for one or more observations."""


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
StepCallback = Callable[[ArenaEnv], None]
LogCallback = Callable[["TrainingState"], None]


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
    # Rolling statistics over the most recent episodes: overall, and per opponent.
    episode_stats: dict[str, float] = field(default_factory=dict)
    opponent_stats: dict[str, dict[str, float]] = field(default_factory=dict)


class TrainingLoop:
    """Collect fresh on-policy data, update the policy, and save progress."""

    def __init__(
        self,
        environment: ArenaEnv | VectorEnv,
        policy: Policy,
        update_policy: UpdatePolicy,
        *,
        save_checkpoint: SaveCheckpoint | None = None,
        step_callback: StepCallback | None = None,
        log_callback: LogCallback | None = None,
        device: torch.device | str | None = None,
        stats_window: int = 100,
        initial_state: TrainingState | None = None,
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
        self.step_callback = step_callback
        self.log_callback = log_callback
        self.device = device
        # Pass a restored state to continue a previous run's counters.
        self.state = initial_state if initial_state is not None else TrainingState()
        self._episode_stats = EpisodeStats(stats_window)
        self._interrupt_pending = False
        # Seconds the last rollout spent in the environment and in the policy.
        self._env_seconds = 0.0
        self._policy_seconds = 0.0
        self._current_episode_return = 0.0
        self._vector_episode_returns: list[float] = []

    def _remember_interrupt(self, signum: int, frame: object) -> None:
        """Hold the first Ctrl+C until the update ends. A second one stops at once."""
        if self._interrupt_pending:
            raise KeyboardInterrupt
        self._interrupt_pending = True

    @contextmanager
    def _defer_interrupt(self):
        """Delay Ctrl+C while the policy updates, so weights and iteration agree.

        An interrupt in the middle of an update would leave weights that are
        partway through it, saved next to an iteration count that excludes it.
        Only the main thread can install a signal handler, so elsewhere this does
        nothing.
        """
        try:
            previous = signal.signal(signal.SIGINT, self._remember_interrupt)
        except ValueError:
            yield
            return
        try:
            yield
        finally:
            if previous is not None:
                signal.signal(signal.SIGINT, previous)

    def _record_episode(
        self, info: Mapping[str, object], truncated: bool, episode_return: float
    ) -> None:
        """Add a finished episode to the rolling statistics."""

        def count(key: str) -> int:
            value = info.get(key, 0)
            return value if isinstance(value, int) else 0

        winner = info.get("winner")
        self._episode_stats.record(
            str(info.get("opponent", "unknown")),
            EpisodeRecord(
                won=winner == "agent",
                lost=winner == "opponent",
                truncated=truncated,
                length=count("step_count"),
                episode_return=episode_return,
                damage_dealt=count("damage_dealt"),
                damage_taken=count("damage_taken"),
                swings=count("swings"),
                hits=count("hits"),
                edge_steps=count("edge_steps"),
                parry_attempts=count("parry_attempts"),
                parries=count("parries"),
            ),
        )

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
        # Wall-clock time in the policy and the environment, for the timing stats.
        policy_seconds = env_seconds = 0.0
        with torch.no_grad():
            for _ in range(steps):
                started = perf_counter()
                action_result = self.policy.act(observation)
                action = _require_python_int(action_result[0], "act action")
                log_probability = _require_python_float(
                    action_result[1], "act log_probability"
                )
                # Critic value of the current observation, before stepping.
                value = _require_python_float(self.policy.value(observation), "value")
                acted = perf_counter()
                next_observation, reward, terminated, truncated, info = (
                    self.environment.step(action)
                )
                if self.step_callback is not None:
                    self.step_callback(self.environment)
                stepped = perf_counter()
                self._current_episode_return += reward
                # Value of the final observation, for bootstrapping. It is computed
                # before any reset so the reset observation cannot replace it.
                next_value = _require_python_float(
                    self.policy.value(next_observation), "value"
                )
                policy_seconds += (acted - started) + (perf_counter() - stepped)
                env_seconds += stepped - acted
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
                # The terminal or truncated observation was valued above and is
                # kept in next_values, so it is safe to reset now.
                if terminated or truncated:
                    self.state.episodes += 1
                    self.state.episode_return = self._current_episode_return
                    if info.get("winner") == "agent":
                        self.state.wins += 1
                    self._record_episode(info, truncated, self._current_episode_return)
                    self._current_episode_return = 0.0
                    observation, _ = self.environment.reset()
        self._env_seconds, self._policy_seconds = env_seconds, policy_seconds
        batch = rollout.as_batch(device=self.device)
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

    def collect_vector_rollout(
        self,
        observations: list[list[float]],
        steps: int,
    ) -> tuple[RolloutBatch, list[list[float]], list[float]]:
        """Collect ``steps`` synchronous steps from every environment in the batch."""
        if not isinstance(self.environment, VectorEnv):
            raise TypeError("vector rollouts require a VectorEnv")
        environment_count = self.environment.num_envs
        rollout = Rollout(environment_count)
        next_values: list[list[float]] = []
        if len(self._vector_episode_returns) != environment_count:
            self._vector_episode_returns = [0.0] * environment_count

        policy_seconds = env_seconds = 0.0
        with torch.no_grad():
            for _ in range(steps):
                started = perf_counter()
                action_result = self.policy.act(observations)
                if not (
                    isinstance(action_result[0], list)
                    and isinstance(action_result[1], list)
                ):
                    raise TypeError(
                        "batched policy must return lists of actions and log probabilities"
                    )
                actions, log_probabilities = action_result
                values = self.policy.value(observations)
                if not isinstance(values, list):
                    raise TypeError("batched policy must return a list of values")
                acted = perf_counter()
                next_observations, rewards, terminated, truncated, infos = (
                    self.environment.step(actions)
                )
                stepped = perf_counter()
                final_observations = [
                    info.get("final_observation", observation)
                    for info, observation in zip(infos, next_observations)
                ]
                final_values = self.policy.value(final_observations)
                if not isinstance(final_values, list):
                    raise TypeError("batched policy must return a list of values")
                policy_seconds += (acted - started) + (perf_counter() - stepped)
                env_seconds += stepped - acted
                rollout.add(
                    observations,
                    actions,
                    rewards,
                    terminated,
                    truncated,
                    values,
                    log_probabilities,
                )
                next_values.append(final_values)
                for index, (reward, ended, timed_out, info) in enumerate(
                    zip(rewards, terminated, truncated, infos)
                ):
                    self._vector_episode_returns[index] += reward
                    if ended or timed_out:
                        self.state.episodes += 1
                        self.state.episode_return = self._vector_episode_returns[index]
                        if info.get("winner") == "agent":
                            self.state.wins += 1
                        self._record_episode(
                            info, timed_out, self._vector_episode_returns[index]
                        )
                        self._vector_episode_returns[index] = 0.0
                if self.step_callback is not None:
                    best_index = max(
                        range(environment_count),
                        key=lambda index: self._vector_episode_returns[index],
                    )
                    self.step_callback(self.environment.envs[best_index])
                observations = next_observations
                self.state.environment_steps += environment_count
        self._env_seconds, self._policy_seconds = env_seconds, policy_seconds

        batch = rollout.as_batch(device=self.device)
        batch = replace(
            batch,
            next_values=torch.tensor(
                next_values, dtype=batch.values.dtype, device=batch.values.device
            ),
        )
        with torch.no_grad():
            last_value = self.policy.value(observations)
        if not isinstance(last_value, list):
            raise TypeError("batched policy must return a list of values")
        return batch, observations, last_value

    def run(self, config: TrainingConfig) -> TrainingState:
        """Run the training loop: one rollout plus one update per iteration.

        An iteration is not an episode. Each iteration collects exactly
        ``rollout_steps`` environment steps, spanning as many episodes as it
        takes, and then updates the policy once.

        ``config.iterations`` is the number of iterations to run now. When the
        loop was given a restored state, they are added after the iterations it
        already completed, and iteration numbers keep counting from there.
        """
        if config.iterations < 1:
            raise ValueError("iterations must be positive")
        if config.rollout_steps < 1:
            raise ValueError("rollout_steps must be positive")
        if config.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        if config.print_interval < 1:
            raise ValueError("print_interval must be positive")
        self._interrupt_pending = False
        if isinstance(self.environment, VectorEnv):
            if self.environment.num_envs > config.num_envs:
                raise ValueError("the vector environment has more instances than num_envs")
        observation, _ = self.environment.reset(seed=config.seed)
        # Each iteration collects exactly rollout_steps environment steps and then
        # updates the policy once. The rollout spans as many episodes as it takes
        # and can end mid-episode, so it is a step budget, not a game.
        first_iteration = self.state.iteration + 1
        last_iteration = first_iteration + config.iterations - 1
        for iteration in range(first_iteration, last_iteration + 1):
            steps_before = self.state.environment_steps
            started = perf_counter()
            if isinstance(self.environment, VectorEnv):
                batch, observation, last_value = self.collect_vector_rollout(
                    observation, config.rollout_steps
                )
            else:
                batch, observation, last_value = self.collect_rollout(
                    observation, config.rollout_steps
                )
            collected = perf_counter()
            with self._defer_interrupt():
                metrics = dict(
                    self.update_policy(
                        batch,
                        last_value,
                        terminated=batch.terminated,
                        episode_ends=batch.episode_ends,
                        next_values=batch.next_values,
                    )
                )
            finished = perf_counter()
            # Update the training state with the latest metrics and progress.
            self.state.iteration = iteration
            self.state.episode_stats = self._episode_stats.overall()
            self.state.opponent_stats = self._episode_stats.by_opponent()
            seconds = finished - started
            metrics.update(
                {
                    # Rolling over the last stats_window episodes, unlike the
                    # cumulative counts below.
                    "win_rate": self.state.episode_stats.get("win_rate", 0.0),
                    "episodes": float(self.state.episodes),
                    "wins": float(self.state.wins),
                    "rollout_seconds": collected - started,
                    "env_seconds": self._env_seconds,
                    "policy_seconds": self._policy_seconds,
                    "update_seconds": finished - collected,
                    "iteration_seconds": seconds,
                    "steps_per_second": (
                        (self.state.environment_steps - steps_before) / seconds
                        if seconds > 0
                        else 0.0
                    ),
                }
            )
            self.state.metrics = metrics
            if self.log_callback is not None and (
                iteration in (first_iteration, last_iteration)
                or iteration % config.print_interval == 0
            ):
                self.log_callback(self.state)
            # Save a checkpoint if the save_checkpoint callback is provided and
            # the current iteration is a multiple of the checkpoint interval.
            if (
                self.save_checkpoint is not None
                and iteration % config.checkpoint_interval == 0
            ):
                self.save_checkpoint(self.state)
            if self._interrupt_pending:
                # Ctrl+C arrived during the update. The iteration is now complete
                # and recorded, so stop with a consistent state.
                self._interrupt_pending = False
                raise KeyboardInterrupt
        # Return the final training state after all iterations are complete.
        return self.state
