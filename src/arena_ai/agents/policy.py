"""Actor-critic policy used by the training entry point."""

import torch
from torch import nn

from arena_ai.agents.action import ACTION_COUNT
from arena_ai.config import Config
from arena_ai.sim.observation_vector import observation_size
from arena_ai.agents.action import LOGIT_COUNT
from arena_ai.agents.distribution import FactoredCategorical


class ActorCritic(nn.Module):
    """Creates two MLPs, one for action logits and one for value estimates."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        def activation(name: str) -> nn.Module:
            activations: dict[str, type[nn.Module]] = {
                "linear": nn.Identity,
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
            }
            try:
                return activations[name.lower()]()
            except KeyError as error:
                raise ValueError(f"Unsupported network activation: {name}") from error

        def build_network(output_size: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            input_size = observation_size()
            for hidden_size in config.agent.network.hidden_layers:
                layer = nn.Linear(input_size, hidden_size)
                hidden_gain = nn.init.calculate_gain(
                    config.agent.network.activation.lower()
                )
                nn.init.orthogonal_(layer.weight, gain=hidden_gain)
                nn.init.zeros_(layer.bias)
                layers.extend((layer, activation(config.agent.network.activation)))
                input_size = hidden_size
            output_layer = nn.Linear(input_size, output_size)
            nn.init.zeros_(output_layer.bias)
            layers.extend(
                (output_layer, activation(config.agent.network.output_activation))
            )
            return nn.Sequential(*layers)

        # Build both the policy network (for action selection) and the value network (for state value estimation).
        self.policy_network = build_network(LOGIT_COUNT)
        self.value_network = build_network(1)
        policy_output = self.policy_network[-2]
        value_output = self.value_network[-2]
        if not isinstance(policy_output, nn.Linear) or not isinstance(
            value_output, nn.Linear
        ):
            raise RuntimeError("Network output layers must be linear")
        nn.init.orthogonal_(policy_output.weight, gain=0.01)
        nn.init.orthogonal_(value_output.weight)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and critic values for a tensor batch."""
        return self.policy_network(observations), self.value_network(observations)


class PolicyAdapter:
    """Bridge the actor-critic module to environment action selection."""

    def __init__(self, model: ActorCritic, device: torch.device) -> None:
        # Initiates the network of the agent and sets the device to be used for computations (CPU or GPU).
        self.model = model
        self.device = device

    def act(
        self, observation: list[float] | list[list[float]]
    ) -> tuple[int, float] | tuple[list[int], list[float]]:
        """Sample one action or a batch of actions for rollout storage."""
        batched = bool(observation) and isinstance(observation[0], list)
        observation_tensor = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if batched:
            observation_tensor = observation_tensor.squeeze(0)
        with torch.no_grad():
            # use the policy network to compute the action logits for the given observation.
            logits = self.model.policy_network(observation_tensor)
        distribution = FactoredCategorical(logits=logits)
        action = distribution.sample()
        if batched:
            return (
                [int(item) for item in action.tolist()],
                [float(item) for item in distribution.log_prob(action).tolist()],
            )
        return int(action.item()), float(distribution.log_prob(action).item())

    def value(
        self, observation: list[float] | list[list[float]]
    ) -> float | list[float]:
        """Estimate one observation or a batch of observations."""
        batched = bool(observation) and isinstance(observation[0], list)
        observation_tensor = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if batched:
            observation_tensor = observation_tensor.squeeze(0)
        with torch.no_grad():
            # uses the value network that is part of the actor-critic model to estimate
            # the value of the given observation.
            value = self.model.value_network(observation_tensor)
        if batched:
            return [float(item) for item in value.squeeze(-1).tolist()]
        return float(value.squeeze().item())

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the gradient-tracked actor-critic forward pass for PPO."""
        return self.model(observations.to(self.device))
