"""Actor-critic policy used by the training entry point."""

import torch
from torch import nn

from arena_ai.agents.action import LOGIT_COUNT
from arena_ai.agents.distribution import sample_with_log_prob
from arena_ai.agents.stacked_actor import StackedActor
from arena_ai.config import Config
from arena_ai.agents.attention import EntityAttention, activation
from arena_ai.agents.scripted.registry import CONTEXT_SIZE
from arena_ai.sim.observation_vector import FEATURES_PER_FIGHTER, observation_size


class AttentionNetwork(nn.Module):
    """One head: the attention input block, the hidden layers and an output layer.

    The attention block (``EntityAttention``) replaces the first layer of a plain MLP. The
    policy and the value network each own one, with no shared parameters.
    """

    def __init__(self, config: Config, output_size: int) -> None:
        super().__init__()
        network = config.agent.network
        self.encoder = EntityAttention(
            config.game.observation.max_opponents,
            network.embed_dim,
            network.num_heads,
            network.activation,
        )
        layers: list[nn.Module] = []
        input_size = self.encoder.output_size
        for hidden_size in network.hidden_layers:
            layers.extend((nn.Linear(input_size, hidden_size), activation(network.activation)))
            input_size = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, output_size)
        self.output_activation = activation(network.output_activation)
        # Orthogonal weights with the hidden activation's gain and zero biases everywhere,
        # except the output layer, whose gain ActorCritic sets for its head.
        gain = nn.init.calculate_gain(network.activation.lower())
        for module in (*self.encoder.modules(), *self.hidden.modules()):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.output_layer(self.hidden(self.encoder(observation))))


class ActorCritic(nn.Module):
    """Creates two attention networks, one for action logits and one for value estimates."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        # Build both the policy network (for action selection) and the value network
        # (for state value estimation).
        self.policy_network = AttentionNetwork(config, LOGIT_COUNT)
        self.value_network = AttentionNetwork(config, 1)
        nn.init.orthogonal_(self.policy_network.output_layer.weight, gain=0.01)
        nn.init.orthogonal_(self.value_network.output_layer.weight)
        # The critic's offset for the opponent types in a fight (critic-only input that follows
        # the observation, see ``opponent_context``). It starts at zero, so a checkpoint saved
        # before it existed gives the same values (see ``training/checkpoints.py``).
        self.policy_input_size = observation_size(config.game.observation.max_opponents)
        self.matchup = nn.Linear(CONTEXT_SIZE, 1)
        nn.init.zeros_(self.matchup.weight)
        nn.init.zeros_(self.matchup.bias)
        network = config.agent.network
        # What a checkpoint must match to load: the same observation layout and network
        # shape. ``max_opponents`` is not part of it, because the weights do not depend on it.
        self.signature = {
            "spec_version": config.game.observation.spec_version,
            "features_per_fighter": FEATURES_PER_FIGHTER,
            "hidden_layers": list(network.hidden_layers),
            "embed_dim": network.embed_dim,
            "num_heads": network.num_heads,
            "activation": network.activation,
            "output_activation": network.output_activation,
        }

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and critic values for a tensor batch.

        A batch may carry the opponent types after the observation (the training environments
        append them). The policy ignores them, and the critic adds ``matchup`` of them to its
        value. A batch without them (the viewer, tests) simply gets no offset.
        """
        values = self.value_network(observations)
        context = observations[:, self.policy_input_size :]
        if context.shape[1]:
            values = values + self.matchup(context)
        return self.policy_network(observations), values


class PolicyAdapter:
    """Bridge the actor-critic module to environment action selection."""

    def __init__(
        self, model: ActorCritic, device: torch.device, act_device: str = "cpu"
    ) -> None:
        # Initiates the network of the agent and sets the device to be used for
        # computations (CPU or GPU).
        self.model = model
        self.device = device
        self.act_device = torch.device(act_device)
        # The acting copy runs the policy and value networks together, for rollouts only.
        self.actor = StackedActor(model, self.act_device)

    def sync(self) -> None:
        """Copy updated weights into the acting copy."""
        self.actor.load(self.model)

    def _to_tensor(
        self, observation: list[float] | list[list[float]]
    ) -> tuple[torch.Tensor, bool]:
        """Return a ``[batch, features]`` tensor and whether the input was batched."""
        batched = bool(observation) and isinstance(observation[0], list)
        tensor = torch.tensor(observation, dtype=torch.float32, device=self.act_device)
        return (tensor if batched else tensor.unsqueeze(0)), batched

    def act(
        self, observation: list[float] | list[list[float]]
    ) -> tuple[int, float] | tuple[list[int], list[float]]:
        """Sample one action or a batch of actions for rollout storage."""
        observation_tensor, batched = self._to_tensor(observation)
        with torch.no_grad():
            # use the policy network to compute the action logits for the given observation.
            logits, _ = self.actor(observation_tensor)
            action, log_prob = sample_with_log_prob(logits)
        if batched:
            return (
                [int(item) for item in action.tolist()],
                [float(item) for item in log_prob.tolist()],
            )
        return int(action.item()), float(log_prob.item())

    def value(
        self, observation: list[float] | list[list[float]]
    ) -> float | list[float]:
        """Estimate one observation or a batch of observations."""
        observation_tensor, batched = self._to_tensor(observation)
        with torch.no_grad():
            # uses the value network that is part of the actor-critic model to estimate
            # the value of the given observation.
            _, value = self.actor(observation_tensor)
        if batched:
            return [float(item) for item in value.tolist()]
        return float(value.squeeze().item())

    def act_and_value(
        self, observations: list[list[float]]
    ) -> tuple[list[int], list[float], list[float]]:
        """Sample actions and estimate values for a batch with one network pass."""
        observation_tensor, _ = self._to_tensor(observations)
        with torch.no_grad():
            logits, values = self.actor(observation_tensor)
            action, log_prob = sample_with_log_prob(logits)
        return (
            [int(item) for item in action.tolist()],
            [float(item) for item in log_prob.tolist()],
            [float(item) for item in values.tolist()],
        )

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the gradient-tracked actor-critic forward pass for PPO."""
        return self.model(observations.to(self.device))
