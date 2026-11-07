"""PPO loss functions and a single actor-critic update step."""

from collections.abc import Callable

import torch
from torch import nn


def probability_ratio(
    new_log_probs: torch.Tensor, old_log_probs: torch.Tensor
) -> torch.Tensor:
    """Return the probability ratio used by the PPO clipped objective."""
    # note log probs are used so do ratio in exponential space
    # also is more stable to do it vs using probability that can be very small
    # which will lose precision and cause numerical issues
    return torch.exp(new_log_probs - old_log_probs)


# Keep the original misspelling as a compatibility alias.
probabilty_ratio = probability_ratio


def surrogate_objective(
    prob_ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    """Return the mean clipped PPO surrogate objective over a rollout."""
    clipped_ratio = torch.clamp(prob_ratio, 1 - clip_range, 1 + clip_range)
    return torch.minimum(
        prob_ratio * advantages,
        clipped_ratio * advantages,
    ).mean()


def critic_loss(value_preds: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Return the value-function mean squared error."""
    # value_preds is the predicted value function, returns is the target value function
    return nn.functional.mse_loss(value_preds, returns)


def advantage_estimation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    discount_factor: float,
    gae_lambda: float,
    episode_ends: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate GAE for one or multiple environments.

    ``terminated`` controls value bootstrapping. ``episode_ends`` additionally
    cuts the recursive estimate at time-limit truncations.
    """
    if episode_ends is None:
        episode_ends = terminated
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rewards[0])
    # We iterate backwards through the rollout to compute the advantage estimates
    # recursively. Backward iteration is necessary because the advantage at time t
    # depends on the advantage at time t+1, which is computed in the previous iteration
    # of the loop.
    for t in reversed(range(len(rewards))):
        not_terminated = 1 - terminated[t].to(rewards.dtype)
        not_episode_end = 1 - episode_ends[t].to(rewards.dtype)
        # delta = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        # weighted sum of deltas with discounting and GAE lambda,
        # instead of a one step advantage
        delta = (
            rewards[t] + discount_factor * next_values[t] * not_terminated - values[t]
        )
        last_advantage = (
            delta + discount_factor * gae_lambda * not_episode_end * last_advantage
        )
        advantages[t] = last_advantage
    # tensor of advantages for each time step per episode in the rollout
    return advantages


def discounted_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    last_value: torch.Tensor,
    discount_factor: float,
    episode_ends: torch.Tensor | None = None,
    next_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate returns (critic targets) for one or multiple environments. The return is
    the discounted sum of all future rewards.
    This is used by the critic loss function, which is the mean squared error between the
    predicted value function and the return.

    The ``last_value`` is the value function of the last observation in the rollout,
    which is used for bootstrapping the return.

    ``terminated`` controls value bootstrapping. ``episode_ends`` additionally
    cuts the recursive return at time-limit truncations. Pass ``next_values``
    when truncations can occur before the final rollout timestep.
    """
    if episode_ends is None:
        episode_ends = terminated
    returns = torch.zeros_like(rewards)
    running_return = last_value
    for t in reversed(range(len(rewards))):
        not_terminated = 1 - terminated[t].to(rewards.dtype)
        truncated_now = episode_ends[t] & ~terminated[t]
        # A terminated state has no bootstrap value. At a truncation, use the
        # value of the final observation rather than the reset episode.
        if truncated_now.any() and next_values is None:
            raise ValueError(
                "next_values is required when returns include episode truncations"
            )
        if next_values is not None and truncated_now.any():
            # continuation is the value to use for bootstrapping the return. If the episode
            # was truncated, use the value of the next state. If it was terminated, use
            # the running return (which is zero for a terminated state). If it was neither,
            # use the running return (which is the discounted sum of future rewards).
            continuation = torch.where(
                truncated_now,
                next_values[t],
                running_return,
            )
        else:
            continuation = running_return
        running_return = rewards[t] + discount_factor * continuation * not_terminated
        returns[t] = running_return
    return returns


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Normalize advantages while avoiding division by zero for one-item batches.
    This is so that the ppo loss is invariant to the scale of the advantages,
    which can vary a lot between environments."""
    standard_deviation = advantages.std(unbiased=False).clamp_min(1e-8)
    return (advantages - advantages.mean()) / standard_deviation


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return mean categorical entropy for a batch of action logits.
    Measures how spread out the action distribution is, which encourages exploration."""
    distribution = torch.distributions.Categorical(logits=logits)
    return distribution.entropy().mean()


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Return the signed entropy loss used in the combined PPO loss.
    Negative because we want to maximize entropy, but optimizers minimize loss."""
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
        policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss(logits)
    )


def optimize_policy(
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    max_grad_norm: float | None = 0.5,
) -> None:
    """Backpropagate one PPO loss and update the model parameters."""
    optimizer.zero_grad(set_to_none=True)  # clear gradients before backward pass
    loss.backward()  # back prop from total loss to compute gradients for all parameters
    if max_grad_norm is not None:
        # flatten all parameters from all parameter groups into a single list
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        # clip gradients to avoid exploding gradients
        nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    # single step of the optimizer to update the model parameters
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
    """Run one PPO update for a policy returning ``(logits, values)``. Wrapper for
    the loss functions and the optimizer step. Returns a dictionary of loss values
    for logging."""
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
    # Return loss values for logging, without tracking gradients.
    with torch.no_grad():
        return {
            "loss": loss.item(),
            "policy_loss": -surrogate_objective(
                prob_ratio, advantages, clip_range
            ).item(),
            "value_loss": critic_loss(value_preds.squeeze(-1), returns).item(),
            "entropy": distribution.entropy().mean().item(),
        }
