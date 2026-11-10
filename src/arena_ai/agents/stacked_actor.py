"""Inference-only copy of the actor and critic that runs both in one pass.

**Why:** a rollout step runs the network twice (action logits, then values), and with a batch
of a few observations the time is the per-operation overhead of about 230 small tensor
operations per network, not the arithmetic. **What:** ``StackedActor`` copies the weights of
the two networks of an ``ActorCritic`` into tensors with a leading dimension of 2 (slot 0
the actor, slot 1 the critic) and runs both with batched matrix multiplies, so one pass costs
about as many operations as one network. **How it differs from the networks:**
- The two networks have the same shapes except the output layer (``LOGIT_COUNT`` outputs for
  the actor, 1 for the critic), so the critic's output layer is padded with zero columns and
  its value is read from column 0.
- The key and value projections are joined into one layer.
**Limits:** it only reads weights, never trains, and must be rebuilt with ``load`` after every
update (``PolicyAdapter.sync`` does). Results match the networks to float rounding, not bit for
bit, because a batched multiply adds in a different order.
"""

import math

import torch
from torch import nn

from arena_ai.agents.action import LOGIT_COUNT
from arena_ai.agents.attention import activation
from arena_ai.sim.observation_vector import FEATURES_PER_FIGHTER

_Layer = tuple[torch.Tensor, torch.Tensor]  # Stacked weight [2, in, out] and bias [2, 1, out].


def _stack(actor: nn.Linear, critic: nn.Linear, device, out_pad: int = 0) -> _Layer:
    """Stack one layer of each network, zero-padding the output to ``out_pad`` columns."""
    weights = [layer.weight.detach().T for layer in (actor, critic)]
    biases = [layer.bias.detach() for layer in (actor, critic)]
    if out_pad:
        weights = [nn.functional.pad(w, (0, out_pad - w.shape[-1])) for w in weights]
        biases = [nn.functional.pad(b, (0, out_pad - b.shape[-1])) for b in biases]
    weight, bias = torch.stack(weights), torch.stack(biases).unsqueeze(1)
    return weight.contiguous().to(device), bias.contiguous().to(device)


def _stack_kv(actor, critic, device) -> _Layer:
    """Join the key and value projections into one layer: output is ``[keys ; values]``."""
    keys = _stack(actor.w_k, critic.w_k, device)
    values = _stack(actor.w_v, critic.w_v, device)
    return torch.cat([keys[0], values[0]], dim=-1), torch.cat([keys[1], values[1]], dim=-1)


def _linear(x: torch.Tensor, layer: _Layer) -> torch.Tensor:
    weight, bias = layer
    return torch.baddbmm(bias, x, weight)


class StackedActor:
    """Run an ``ActorCritic``'s policy and value networks together, for rollouts only."""

    def __init__(self, model, device: torch.device) -> None:
        self.device = device
        self.load(model)

    def load(self, model) -> None:
        """Copy the current weights of ``model`` (an ``ActorCritic``) into stacked tensors."""
        actor, critic = model.policy_network, model.value_network
        encoder_a, encoder_c = actor.encoder, critic.encoder
        device = self.device
        self.max_opponents = encoder_a.max_opponents
        self.num_heads = encoder_a.num_heads
        self.embed_dim = encoder_a.embed_dim
        self.head_dim = encoder_a.head_dim
        self.activation = activation(model.signature["activation"])
        self.output_activation = activation(model.signature["output_activation"])

        def pair(path: str) -> _Layer:
            first, second = (self._get(net.encoder, path) for net in (actor, critic))
            return _stack(first, second, device)

        self.phi_self = [pair("phi_self.0"), pair("phi_self.2")]
        self.phi_opp = [pair("phi_opp.0"), pair("phi_opp.2")]
        self.w_q = pair("w_q")
        self.w_kv = _stack_kv(encoder_a, encoder_c, device)
        self.out_proj = None if isinstance(encoder_a.out_proj, nn.Identity) else pair("out_proj")
        hidden_a = [m for m in actor.hidden if isinstance(m, nn.Linear)]
        hidden_c = [m for m in critic.hidden if isinstance(m, nn.Linear)]
        self.hidden = [_stack(a, c, device) for a, c in zip(hidden_a, hidden_c)]
        self.output = _stack(actor.output_layer, critic.output_layer, device, LOGIT_COUNT)
        self.matchup_weight = model.matchup.weight.detach().T.contiguous().to(device)  # [T, 1]
        self.matchup_bias = model.matchup.bias.detach().to(device)
        self.policy_input_size = f_size = FEATURES_PER_FIGHTER * (self.max_opponents + 1) + (
            self.max_opponents
        )
        del f_size

    @staticmethod
    def _get(module: nn.Module, path: str):
        for part in path.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module

    def _mlp(self, x: torch.Tensor, layers: list[_Layer]) -> torch.Tensor:
        return _linear(self.activation(_linear(x, layers[0])), layers[1])

    def __call__(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action logits ``[B, LOGIT_COUNT]`` and values ``[B]`` for ``[B, D]`` input."""
        batch = observation.shape[0]
        f, k = FEATURES_PER_FIGHTER, self.max_opponents
        d, h, dk = self.embed_dim, self.num_heads, self.head_dim
        self_row = observation[:, :f]
        opponent_rows = observation[:, f : f * (k + 1)].reshape(batch * k, f)
        mask = observation[:, f * (k + 1) : f * (k + 1) + k] > 0.5  # [B, K]
        visible = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)  # [B, 1 + K]

        e_self = self._mlp(self_row.unsqueeze(0).expand(2, -1, -1), self.phi_self)  # [2, B, d]
        e_opp = self._mlp(opponent_rows.unsqueeze(0).expand(2, -1, -1), self.phi_opp)
        entities = torch.cat([e_self.unsqueeze(2), e_opp.reshape(2, batch, k, d)], dim=2)

        query = _linear(e_self, self.w_q).reshape(2, batch, h, 1, dk)
        kv = _linear(entities.reshape(2, batch * (k + 1), d), self.w_kv)
        # [keys or values, network, B, H, 1 + K, dk]
        kv = kv.reshape(2, batch, k + 1, 2, h, dk).permute(3, 0, 1, 4, 2, 5)
        keys, values = kv[0], kv[1]
        scores = query @ keys.transpose(-1, -2) / math.sqrt(dk)  # [2, B, H, 1, 1 + K]
        scores = scores.masked_fill(~visible[None, :, None, None, :], float("-inf"))
        context = (torch.softmax(scores, dim=-1) @ values).reshape(2, batch, h * dk)
        if self.out_proj is not None:
            context = _linear(context, self.out_proj)

        x = torch.cat([e_self, context], dim=-1)  # [2, B, 2d]
        for layer in self.hidden:
            x = self.activation(_linear(x, layer))
        out = self.output_activation(_linear(x, self.output))  # [2, B, LOGIT_COUNT]
        value = out[1, :, 0]
        context = observation[:, self.policy_input_size :]  # The critic-only opponent types.
        if context.shape[1]:
            value = value + (context @ self.matchup_weight)[:, 0] + self.matchup_bias
        return out[0], value
