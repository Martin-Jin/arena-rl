"""Collect and store environment transitions for policy updates."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RolloutBatch:
    """Tensor representation of one on-policy rollout. (one episode of transitions)"""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor

    @property
    def dones(self) -> torch.Tensor:
        """Return flags used to stop bootstrapping at episode boundaries."""
        return torch.logical_or(self.terminated, self.truncated)


class Rollout:
    """Store transitions collected with one unchanged policy.
    (for mutliple episodes, if the environment is reset after each episode)"""

    def __init__(self) -> None:
        self._observations: list[list[float]] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._values: list[float] = []
        self._log_probs: list[float] = []

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return len(self._actions)

    def add(
        self,
        observation: list[float],
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Append one transition produced by the current policy."""
        self._observations.append(list(observation))
        self._actions.append(action)
        self._rewards.append(reward)
        self._terminated.append(terminated)
        self._truncated.append(truncated)
        self._values.append(value)
        self._log_probs.append(log_prob)

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

        return RolloutBatch(
            observations=torch.tensor(self._observations, dtype=dtype, device=device),
            actions=torch.tensor(self._actions, dtype=torch.long, device=device),
            rewards=torch.tensor(self._rewards, dtype=dtype, device=device),
            terminated=torch.tensor(self._terminated, dtype=torch.bool, device=device),
            truncated=torch.tensor(self._truncated, dtype=torch.bool, device=device),
            values=torch.tensor(self._values, dtype=dtype, device=device),
            log_probs=torch.tensor(self._log_probs, dtype=dtype, device=device),
        )

    def clear(self) -> None:
        """Discard the old on-policy data before collecting a new rollout."""
        self._observations.clear()
        self._actions.clear()
        self._rewards.clear()
        self._terminated.clear()
        self._truncated.clear()
        self._values.clear()
        self._log_probs.clear()
