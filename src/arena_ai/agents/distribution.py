"""Independent categorical heads over (move, jump, attack)."""

import torch
from torch.distributions import Categorical

from arena_ai.agents.action import (
    HEAD_SIZES,
    HEAD_STRIDES,
    decode_parts,
    encode_parts,
)


class FactoredCategorical:
    """Drop-in for Categorical: flat action index in, flat index out."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.heads = [
            Categorical(logits=part) for part in torch.split(logits, HEAD_SIZES, dim=-1)
        ]

    def sample(self) -> torch.Tensor:
        """Sample a flat action index from the factored distribution."""
        return encode_parts([head.sample() for head in self.heads])

    def mode(self) -> torch.Tensor:
        """Return the flat action index with the highest probability."""
        return encode_parts([head.probs.argmax(-1) for head in self.heads])

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Return the log probability of a flat action index."""
        parts = decode_parts(action)
        return sum(head.log_prob(part) for head, part in zip(self.heads, parts))

    def head_entropies(self) -> list[torch.Tensor]:
        """Return each head's entropy, in the order of ``HEAD_NAMES``."""
        return [head.entropy() for head in self.heads]

    def entropy(self) -> torch.Tensor:
        """Return the joint entropy (sum over the independent heads)."""
        return sum(self.head_entropies())


def sample_with_log_prob(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample flat actions and their log-probabilities from ``[batch, logits]``.

    Equivalent to ``FactoredCategorical.sample`` followed by ``log_prob``, but
    without building ``Categorical`` objects. For the tiny batches used during
    rollouts, the per-operation overhead of those objects (argument validation,
    repeated normalization) costs more than the sampling itself. Gradients are
    not tracked, so use ``FactoredCategorical`` for the PPO update.
    """
    action = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    log_prob = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    start = 0
    for size, stride in zip(HEAD_SIZES, HEAD_STRIDES):
        head_log_probs = torch.log_softmax(logits[:, start : start + size], dim=-1)
        part = torch.multinomial(head_log_probs.exp(), 1)
        log_prob += head_log_probs.gather(-1, part).squeeze(-1)
        action += part.squeeze(-1) * stride
        start += size
    return action, log_prob
