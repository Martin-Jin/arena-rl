"""Collect and store environment transitions for policy updates."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RolloutBatch:
    """Tensor representation of a rollout with shape ``[time, environments]``."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    next_values: torch.Tensor

    @property
    def episode_ends(self) -> torch.Tensor:
        """Return flags for both natural ends and time-limit truncations."""
        return torch.logical_or(self.terminated, self.truncated)

    def flattened(self) -> "RolloutBatch":
        """Flatten time and environment axes for minibatch policy updates."""
        if self.actions.ndim == 1:
            return self
        time_steps, environment_count = self.actions.shape
        flatten = lambda tensor: tensor.reshape(
            time_steps * environment_count, *tensor.shape[2:]
        )
        return RolloutBatch(
            observations=flatten(self.observations),
            actions=flatten(self.actions),
            rewards=flatten(self.rewards),
            terminated=flatten(self.terminated),
            truncated=flatten(self.truncated),
            values=flatten(self.values),
            log_probs=flatten(self.log_probs),
            next_values=flatten(self.next_values),
        )


class Rollout:
    """Store transitions collected with one unchanged policy.
    This on-policy buffer is discarded after each policy update."""

    def __init__(self, environment_count: int = 1) -> None:
        if environment_count < 1:
            raise ValueError("environment_count must be positive")
        self.environment_count = environment_count
        self._observations: list[list[list[float]]] = []
        self._actions: list[list[int]] = []
        self._rewards: list[list[float]] = []
        self._terminated: list[list[bool]] = []
        self._truncated: list[list[bool]] = []
        self._values: list[list[float]] = []
        self._log_probs: list[list[float]] = []

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return len(self._actions) * self.environment_count

    def add(
        self,
        observation: list[float] | list[list[float]],
        action: int | list[int],
        reward: float | list[float],
        terminated: bool | list[bool],
        truncated: bool | list[bool],
        value: float | list[float],
        log_prob: float | list[float],
    ) -> None:
        """Append one transition produced by the current policy.

        Takes either one environment or several.
        """
        if self.environment_count == 1 and isinstance(observation[0], float):
            observation = [observation]  # type: ignore[list-item]
        observations = observation
        # if is a single value, convert to list for consistency
        # else from multiple environments, so already a list
        values = [value] if isinstance(value, float) else value
        actions = [action] if isinstance(action, int) else action
        rewards = [reward] if isinstance(reward, float) else reward
        terminated_flags = [terminated] if isinstance(terminated, bool) else terminated
        truncated_flags = [truncated] if isinstance(truncated, bool) else truncated
        log_probs = [log_prob] if isinstance(log_prob, float) else log_prob
        fields = (
            observations,
            actions,
            rewards,
            terminated_flags,
            truncated_flags,
            values,
            log_probs,
        )
        if any(len(field) != self.environment_count for field in fields):
            raise ValueError("all transition fields must match environment_count")
        # Store the transition data in the rollout buffer as a list of lists,
        # where each inner list corresponds to a time step and contains the data
        # for all environments if training in parallel.
        self._observations.append([list(item) for item in observations])
        self._actions.append(list(actions))
        self._rewards.append(list(rewards))
        self._terminated.append(list(terminated_flags))
        self._truncated.append(list(truncated_flags))
        self._values.append(list(values))
        self._log_probs.append(list(log_probs))
        # shape of:
        # observations: [time, env]
        # actions: [time, env]
        # rewards: [time, env]
        # terminated: [time, env]
        # truncated: [time, env]
        # values: [time, env]
        # log_probs: [time, env]
        # appending will append at the end of the time dimension, so the shape will be
        # [time+1, env] after appending

    def as_batch(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> RolloutBatch:
        """Convert the stored transitions into tensors for PPO calculations.
        Called at the end of the a rollout to convert data to a tensor for ppo
        calculations."""
        if not self._actions:
            raise ValueError("cannot convert an empty rollout to a batch")

        batch = RolloutBatch(
            observations=torch.tensor(self._observations, dtype=dtype, device=device),
            actions=torch.tensor(self._actions, dtype=torch.long, device=device),
            rewards=torch.tensor(self._rewards, dtype=dtype, device=device),
            terminated=torch.tensor(self._terminated, dtype=torch.bool, device=device),
            truncated=torch.tensor(self._truncated, dtype=torch.bool, device=device),
            values=torch.tensor(self._values, dtype=dtype, device=device),
            log_probs=torch.tensor(self._log_probs, dtype=dtype, device=device),
            next_values=torch.zeros(
                len(self._actions),
                self.environment_count,
                dtype=dtype,
                device=device,
            ),
        )
        # If only one environment, return a batch with shape [time]
        # instead of the [time, 1] shape produced before squeezing.
        # Because the rollout is designed to handle multiple environments, the
        # batch will have an extra dimension for the environment count.
        # However, if there is only one environment, it is more convenient to
        # return a batch without that extra dimension.
        if self.environment_count == 1:
            return RolloutBatch(
                observations=batch.observations[:, 0],
                actions=batch.actions[:, 0],
                rewards=batch.rewards[:, 0],
                terminated=batch.terminated[:, 0],
                truncated=batch.truncated[:, 0],
                values=batch.values[:, 0],
                log_probs=batch.log_probs[:, 0],
                next_values=batch.next_values[:, 0],
            )
        return batch

    def clear(self) -> None:
        """Discard the old on-policy data before collecting a new rollout."""
        self._observations.clear()
        self._actions.clear()
        self._rewards.clear()
        self._terminated.clear()
        self._truncated.clear()
        self._values.clear()
        self._log_probs.clear()
