"""Actor-critic policy used by the training entry point."""

import torch
from torch import nn

from arena_ai.agents.action import ACTION_COUNT
from arena_ai.config import Config
from arena_ai.sim.observation_vector import observation_size


class ActorCritic(nn.Module):
    """Separate MLPs for action logits and value estimates."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        def build_network(output_size: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            input_size = observation_size()
            for hidden_size in config.agent.network.hidden_layers:
                layer = nn.Linear(input_size, hidden_size)
                nn.init.orthogonal_(layer.weight, gain=2**0.5)
                nn.init.zeros_(layer.bias)
                layers.extend((layer, nn.ReLU()))
                input_size = hidden_size
            layers.append(nn.Linear(input_size, output_size))
            return nn.Sequential(*layers)

        # Build both the policy network (for action selection) and the value network (for state value estimation).
        self.policy_network = build_network(ACTION_COUNT)
        self.value_network = build_network(1)
        nn.init.orthogonal_(self.policy_network[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy_network[-1].bias)
        nn.init.orthogonal_(self.value_network[-1].weight)
        nn.init.zeros_(self.value_network[-1].bias)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and critic values for a tensor batch."""
        return self.policy_network(observations), self.value_network(observations)


class PolicyAdapter:
    """Bridge the actor-critic module to environment action selection."""

    def __init__(self, model: ActorCritic, device: torch.device) -> None:
        # Initiates the network of the agent and sets the device to be used for computations (CPU or GPU).
        self.model = model
        self.device = device

    def act(self, observation: list[float]) -> tuple[int, float]:
        """Sample an action and return Python scalars for rollout storage."""
        observation_tensor = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            # use the policy network to compute the action logits for the given observation,
            # then sample an action from the resulting categorical distribution and return both
            # the action and its log probability.
            logits = self.model.policy_network(observation_tensor)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item())

    def value(self, observation: list[float]) -> float:
        """Estimate one observation's value as a Python float."""
        observation_tensor = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            # uses the value network that is part of the actor-critic model to estimate
            # the value of the given observation.
            value = self.model.value_network(observation_tensor)
        return float(value.squeeze().item())

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the gradient-tracked actor-critic forward pass for PPO."""
        return self.model(observations.to(self.device))
