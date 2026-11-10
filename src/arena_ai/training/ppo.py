"""PPO loss functions and a single actor-critic update step."""

from collections.abc import Callable

import torch
from torch import nn
from arena_ai.agents.action import HEAD_NAMES
from arena_ai.agents.distribution import FactoredCategorical


def probability_ratio(
    new_log_probs: torch.Tensor, old_log_probs: torch.Tensor
) -> torch.Tensor:
    """Return the probability ratio used by the PPO clipped objective."""
    # note log probs are used so do ratio in exponential space
    # also is more stable to do it vs using probability that can be very small
    # which will lose precision and cause numerical issues
    return torch.exp(new_log_probs - old_log_probs)


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
    # tensor of advantages for each time step of the rollout, shape [time, environments]
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
    truncated = episode_ends & ~terminated
    # Checked once up front: calling .any() inside the loop forces a device sync
    # on every step when the tensors live on the GPU.
    has_truncation = bool(truncated.any())
    if has_truncation and next_values is None:
        raise ValueError(
            "next_values is required when returns include episode truncations"
        )
    for t in reversed(range(len(rewards))):
        not_terminated = 1 - terminated[t].to(rewards.dtype)
        truncated_now = truncated[t]
        # A terminated state has no bootstrap value. At a truncation, use the
        # value of the final observation rather than the reset episode.
        if has_truncation:
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


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Return how much of the return variance the critic predicts.

    ``1 - Var(returns - values) / Var(returns)``: near 1 is a good critic, near 0
    predicts nothing and below 0 is worse than a constant. Unlike the value loss
    its scale does not depend on the reward. Returns a 0-d tensor, and 0 when the
    returns do not vary.
    """
    variance = returns.var(unbiased=False)
    ratio = (returns - values).var(unbiased=False) / variance.clamp_min(1e-8)
    return torch.where(variance > 1e-8, 1 - ratio, torch.zeros_like(ratio))


def total_loss(
    prob_ratio: torch.Tensor,
    advantages: torch.Tensor,
    value_preds: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_range: float,
    value_loss_coef: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine the policy, value and entropy terms into the PPO loss.

    Returns ``(loss, policy_loss, value_loss)``. The entropy is subtracted
    because the optimizer minimizes, and more entropy encourages exploration.
    """
    policy_loss = -surrogate_objective(prob_ratio, advantages, clip_range)
    value_loss = critic_loss(value_preds, returns)
    loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
    return loss, policy_loss, value_loss


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
) -> dict[str, torch.Tensor]:
    """Run one PPO update for a policy returning ``(logits, values)``.

    Returns detached 0-d tensors, not Python floats. Reading a GPU tensor back
    forces a device sync, so the caller averages the tensors over its minibatches
    and reads them once. The statistics reuse values the loss already computed:

    - ``approx_kl``: how far the policy moved from the one that collected the
      data (a non-negative estimate). Large means unstable, near 0 means the
      policy barely changes.
    - ``clip_fraction``: the share of samples whose probability ratio left the
      clip range. Around 0.1 to 0.2 is typical, 0 means updates barely move the
      policy and a persistently high value means they are too aggressive.
    - ``entropy_<head>``: entropy of each action head, which shows if one head
      (for example jump) collapses early.
    """
    logits, value_preds = policy(observations)
    distribution = FactoredCategorical(logits)
    new_log_probs = distribution.log_prob(actions)
    prob_ratio = probability_ratio(new_log_probs, old_log_probs)
    head_entropies = [entropy.mean() for entropy in distribution.head_entropies()]
    entropy = sum(head_entropies)
    loss, policy_loss, value_loss = total_loss(
        prob_ratio,
        advantages,
        value_preds.squeeze(-1),
        returns,
        entropy,
        clip_range,
        value_loss_coef,
        entropy_coef,
    )
    optimize_policy(optimizer, loss, max_grad_norm)
    with torch.no_grad():
        ratio = prob_ratio.detach()
        stats = {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": ((ratio - 1) - (new_log_probs.detach() - old_log_probs)).mean(),
            "clip_fraction": ((ratio - 1).abs() > clip_range).float().mean(),
        }
        for name, head_entropy in zip(HEAD_NAMES, head_entropies):
            stats[f"entropy_{name}"] = head_entropy
        return {name: value.detach() for name, value in stats.items()}


class GraphedTrainStep:
    """Run ``train_step`` for a fixed minibatch size as one replayed CUDA graph.

    **Why:** a PPO step on this small network is a thousand tiny GPU launches, and the
    time is the CPU queuing them one by one (about 27 ms), not the GPU running them (a few
    ms). **What:** the first call records one whole step (forward, loss, backward, gradient
    clipping, Adam step) against fixed input buffers. After that each step copies a
    minibatch into those buffers and replays the recording with one call. **How:** the
    recording remembers memory addresses, so the buffers, the weights and the optimizer
    state are only ever updated in place. The statistics are summed inside the graph into
    fixed tensors, and ``means`` returns them. **Limits:** CUDA only. A minibatch of another
    size (the short last one) runs as an ordinary ``train_step``, which gives the same
    result. The learning rate is recorded as a constant, so it cannot change between steps.
    Building it runs three warm-up steps, then puts the weights and optimizer state back as
    they were, so it can be built after a checkpoint was loaded.
    """

    def __init__(
        self,
        policy: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        example: tuple[torch.Tensor, ...],
        clip_range: float,
        value_loss_coef: float,
        entropy_coef: float,
        max_grad_norm: float | None = 0.5,
    ) -> None:
        """``example`` is one full minibatch: its tensors give the shapes the graph is for."""
        self.policy = policy
        self.optimizer = optimizer
        self.settings = (clip_range, value_loss_coef, entropy_coef, max_grad_norm)
        self.batch_size = example[0].shape[0]
        self.inputs = [tensor.clone() for tensor in example]
        parameters = [
            parameter for group in optimizer.param_groups for parameter in group["params"]
        ]
        device = parameters[0].device
        # A graph needs the optimizer's step counters on the GPU.
        for group in optimizer.param_groups:
            group["capturable"] = True
        for state in optimizer.state.values():
            if "step" in state:
                state["step"] = state["step"].to(device=device, dtype=torch.float32)
        saved_parameters = [parameter.detach().clone() for parameter in parameters]
        saved_state = {
            parameter: {
                name: value.clone() if torch.is_tensor(value) else value
                for name, value in optimizer.state[parameter].items()
            }
            for parameter in parameters
            if parameter in optimizer.state
        }

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                stats = self._step()
        torch.cuda.current_stream().wait_stream(side_stream)
        self.sums = {name: torch.zeros_like(value) for name, value in stats.items()}
        self.count = 0
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            captured = self._step()
            for name, value in captured.items():
                self.sums[name].add_(value)

        # Undo the warm-up, in place: the graph remembers these tensors' addresses.
        with torch.no_grad():
            for parameter, saved in zip(parameters, saved_parameters):
                parameter.copy_(saved)
            for parameter in parameters:
                state = optimizer.state[parameter]
                for name, value in state.items():
                    if not torch.is_tensor(value):
                        continue
                    if parameter in saved_state:
                        value.copy_(saved_state[parameter][name])
                    else:
                        value.zero_()  # The optimizer had no state before the warm-up.
        for value in self.sums.values():
            value.zero_()

    def _step(self) -> dict[str, torch.Tensor]:
        clip_range, value_loss_coef, entropy_coef, max_grad_norm = self.settings
        return train_step(
            self.policy,
            self.optimizer,
            *self.inputs,
            clip_range,
            value_loss_coef,
            entropy_coef,
            max_grad_norm,
        )

    def step(self, tensors: tuple[torch.Tensor, ...], indices: torch.Tensor) -> None:
        """Train on ``tensor[indices]`` for each of the five minibatch tensors.

        The five are observations, actions, old log-probabilities, advantages and returns,
        in the order ``train_step`` takes them.
        """
        self.count += 1
        if len(indices) == self.batch_size:
            for tensor, buffer in zip(tensors, self.inputs):
                torch.index_select(tensor, 0, indices, out=buffer)
            self.graph.replay()
            return
        clip_range, value_loss_coef, entropy_coef, max_grad_norm = self.settings
        stats = train_step(
            self.policy,
            self.optimizer,
            *(tensor[indices] for tensor in tensors),
            clip_range,
            value_loss_coef,
            entropy_coef,
            max_grad_norm,
        )
        for name, value in stats.items():
            self.sums[name].add_(value)

    def means(self) -> dict[str, torch.Tensor]:
        """Return each statistic averaged over the steps since the last call, and reset."""
        means = {name: value / self.count for name, value in self.sums.items()}
        for value in self.sums.values():
            value.zero_()
        self.count = 0
        return means
