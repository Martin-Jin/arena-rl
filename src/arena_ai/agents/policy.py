"""Actor-critic policy used by the training entry point."""

import copy

import torch
from torch import nn

from arena_ai.agents.action import LOGIT_COUNT
from arena_ai.agents.distribution import sample_with_log_prob
from arena_ai.config import Config
from arena_ai.sim.observation_vector import observation_size


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

        # Build both the policy network (for action selection) and the value network
        # (for state value estimation).
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

    def __init__(
        self, model: ActorCritic, device: torch.device, act_device: str = "cpu"
    ) -> None:
        # Initiates the network of the agent and sets the device to be used for
        # computations (CPU or GPU).
        self.model = model
        self.device = device
        self.act_device = torch.device(act_device)
        self.actor = copy.deepcopy(model).to(self.act_device).eval()

    def sync(self) -> None:
        """Copy updated weights into the acting copy."""
        self.actor.load_state_dict(self.model.state_dict())

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
            logits = self.actor.policy_network(observation_tensor)
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
            value = self.actor.value_network(observation_tensor)
        if batched:
            return [float(item) for item in value.squeeze(-1).tolist()]
        return float(value.squeeze().item())

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the gradient-tracked actor-critic forward pass for PPO."""
        return self.model(observations.to(self.device))
