"""PPO loss functions and a single actor-critic update step."""

from collections.abc import Callable

import torch
from torch import nn


def probability_ratio(
    new_log_probs: torch.Tensor, old_log_probs: torch.Tensor
) -> torch.Tensor:
    """Return the probability ratio used by the PPO clipped objective."""
    return torch.exp(new_log_probs - old_log_probs)


# Keep the original misspelling as a compatibility alias.
probabilty_ratio = probability_ratio


def surrogate_objective(
    prob_ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    """Return the mean clipped PPO surrogate objective."""
    clipped_ratio = torch.clamp(prob_ratio, 1 - clip_range, 1 + clip_range)
    return torch.minimum(
        prob_ratio * advantages,
        clipped_ratio * advantages,
    ).mean()


def critic_loss(value_preds: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Return the value-function mean squared error."""
    return nn.functional.mse_loss(value_preds, returns)


def advantage_estimation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    discount_factor: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Calculate generalized advantage estimates for a rollout."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(len(rewards))):
        not_done = 1 - dones[t].to(rewards.dtype)
        delta = rewards[t] + discount_factor * next_values[t] * not_done - values[t]
        last_advantage = (
            delta + discount_factor * gae_lambda * not_done * last_advantage
        )
        advantages[t] = last_advantage
    return advantages


def discounted_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    discount_factor: float,
) -> torch.Tensor:
    """Calculate bootstrapped discounted returns for a rollout."""
    returns = torch.zeros_like(rewards)
    running_return = last_value
    for t in reversed(range(len(rewards))):
        not_done = 1 - dones[t].to(rewards.dtype)
        running_return = rewards[t] + discount_factor * running_return * not_done
        returns[t] = running_return
    return returns


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Normalize advantages while avoiding division by zero for one-item batches."""
    standard_deviation = advantages.std(unbiased=False).clamp_min(1e-8)
    return (advantages - advantages.mean()) / standard_deviation


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return mean categorical entropy for a batch of action logits."""
    distribution = torch.distributions.Categorical(logits=logits)
    return distribution.entropy().mean()


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Return the signed entropy loss used in the combined PPO loss."""
    return -entropy_from_logits(logits)


def total_loss(
    prob_ratio: torch.Tensor,
    advantages: torch.Tensor,
    value_preds: torch.Tensor,
    returns: torch.Tensor,
    logits: torch.Tensor,
    clip_range: float,
    value_loss_coef: float,
    entropy_coef: float,
) -> torch.Tensor:
    """Combine policy, value, and entropy terms into the PPO loss."""
    policy_loss = -surrogate_objective(prob_ratio, advantages, clip_range)
    value_loss = critic_loss(value_preds, returns)
    return (
        policy_loss
        + value_loss_coef * value_loss
        + entropy_coef * entropy_loss(logits)
    )


def optimize_policy(
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    max_grad_norm: float | None = 0.5,
) -> None:
    """Backpropagate one PPO loss and update the model parameters."""
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm is not None:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    optimizer.step()


def train_step(
    policy: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float,
    value_loss_coef: float,
    entropy_coef: float,
    max_grad_norm: float | None = 0.5,
) -> dict[str, float]:
    """Run one PPO update for a policy returning ``(logits, values)``."""
    logits, value_preds = policy(observations)
    distribution = torch.distributions.Categorical(logits=logits)
    new_log_probs = distribution.log_prob(actions)
    prob_ratio = probability_ratio(new_log_probs, old_log_probs)
    loss = total_loss(
        prob_ratio,
        advantages,
        value_preds.squeeze(-1),
        returns,
        logits,
        clip_range,
        value_loss_coef,
        entropy_coef,
    )
    optimize_policy(optimizer, loss, max_grad_norm)
    with torch.no_grad():
        return {
            "loss": loss.item(),
            "policy_loss": -surrogate_objective(
                prob_ratio, advantages, clip_range
            ).item(),
            "value_loss": critic_loss(value_preds.squeeze(-1), returns).item(),
            "entropy": distribution.entropy().mean().item(),
        }
