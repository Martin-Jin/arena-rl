"""Independent categorical heads over (move, jump, attack)."""

import torch
from torch.distributions import Categorical

from arena_ai.agents.action import HEAD_SIZES, decode_parts, encode_parts


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

    def entropy(self) -> torch.Tensor:
        """Return the mean entropy of the factored distribution."""
        return sum(head.entropy() for head in self.heads)
