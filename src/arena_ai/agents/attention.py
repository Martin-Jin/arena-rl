"""Attention input block that lets one network face any number of opponents."""

import math

import torch
from torch import nn

from arena_ai.sim.observation_vector import FEATURES_PER_FIGHTER


def activation(name: str) -> nn.Module:
    """Return the activation module called ``name`` ("linear", "relu" or "tanh")."""
    activations: dict[str, type[nn.Module]] = {
        "linear": nn.Identity,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as error:
        raise ValueError(f"Unsupported network activation: {name}") from error


def _entity_encoder(embed_dim: int, activation_name: str) -> nn.Sequential:
    """Return Linear(F, d) -> activation -> Linear(d, d), applied to one fighter's features."""
    return nn.Sequential(
        nn.Linear(FEATURES_PER_FIGHTER, embed_dim),
        activation(activation_name),
        nn.Linear(embed_dim, embed_dim),
    )


class EntityAttention(nn.Module):
    """Turn the flat observation into ``[e_self ; context]``, a vector of ``2 * embed_dim``.

    **Why:** the observation holds the acting fighter, ``max_opponents`` opponent slots and
    a mask. A plain network would learn a separate weight set per slot and care about slot
    order. This block treats the opponents as a set, so what it learns about one opponent
    applies to all, and it works for any count up to ``max_opponents``.

    **How:**
    1. Split the vector into the self row, the opponent rows ``[B, K, F]`` and the mask.
    2. Encode the self row with ``phi_self`` and every opponent row with one shared
       ``phi_opp`` (each Linear -> activation -> Linear), giving the entities
       ``E = [e_self; e_1 .. e_K]``.
    3. Only the acting fighter asks a question: ``q = W_Q e_self``. Keys and values come from
       every entity, the fighter itself included: ``K = W_K E``, ``V = W_V E``. Scores are
       ``q . K / sqrt(d_k)``. Masked opponent slots (absent or dead) get ``-inf``, so their
       weight is exactly 0. The self slot is never masked, so no softmax row is empty.
    4. The context is the softmax-weighted sum of ``V``. With more than one head the heads
       are joined and mixed by an output projection.

    The hidden layers get ``e_self`` next to the context so that the fighter's own state
    reaches them directly, not only as one entry in a weighted average.

    **Limits:** it has no positions between opponents beyond what the features carry (x and
    y, and each row's position relative to the acting fighter), and each head's context is
    a blend, so the network sees a summary of the opponents, not each one separately.
    """

    def __init__(
        self, max_opponents: int, embed_dim: int, num_heads: int, activation_name: str
    ) -> None:
        super().__init__()
        if max_opponents < 1:
            raise ValueError("max_opponents must be at least 1")
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        self.max_opponents = max_opponents
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.phi_self = _entity_encoder(embed_dim, activation_name)
        self.phi_opp = _entity_encoder(embed_dim, activation_name)
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim) if num_heads > 1 else nn.Identity()

    @property
    def output_size(self) -> int:
        """Return the size of the vector handed to the hidden layers."""
        return 2 * self.embed_dim

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        batch = observation.shape[0]
        f, k = FEATURES_PER_FIGHTER, self.max_opponents
        self_row = observation[:, :f]
        opponent_rows = observation[:, f : f * (k + 1)].reshape(batch, k, f)
        mask = observation[:, f * (k + 1) : f * (k + 1) + k] > 0.5  # [B, K]

        e_self = self.phi_self(self_row)  # [B, d]
        e_opp = self.phi_opp(opponent_rows)  # [B, K, d]
        entities = torch.cat([e_self.unsqueeze(1), e_opp], dim=1)  # [B, 1 + K, d]
        visible = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)  # [B, 1 + K]

        h, dk = self.num_heads, self.head_dim
        query = self.w_q(e_self).reshape(batch, h, 1, dk)
        keys = self.w_k(entities).reshape(batch, k + 1, h, dk).transpose(1, 2)
        values = self.w_v(entities).reshape(batch, k + 1, h, dk).transpose(1, 2)
        scores = query @ keys.transpose(-1, -2) / math.sqrt(dk)  # [B, H, 1, 1 + K]
        scores = scores.masked_fill(~visible[:, None, None, :], float("-inf"))
        context = (torch.softmax(scores, dim=-1) @ values).reshape(batch, h * dk)
        context = self.out_proj(context)
        return torch.cat([e_self, context], dim=-1)  # [B, 2d]
