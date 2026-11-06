"""Define the neural network architecture for the AI agent."""

import torch
from torch import nn

from arena_ai.config import load_config
from arena_ai.agents.action import ACTION_COUNT
from arena_ai.sim.observation_vector import observation_size

config = load_config()
network_config = config.agent.network

device = "cuda" if torch.cuda.is_available() else "cpu"


class Network(nn.Module):
    """A multilayer network whose dimensions come from action and observation codecs."""

    def __init__(self, input_size, hidden_layers, output_size):
        super().__init__()
        self.layers = []
        # define all the layers
        layer_sizes = [input_size, *hidden_layers, output_size]
        modules = []
        # pairs the input and output for each layer to create the network
        # e.g. [(input, hidden1), (hidden1, hidden2), (hidden2, output)]
        for layer_input, layer_output in zip(layer_sizes, layer_sizes[1:]):
            modules.append(nn.Linear(layer_input, layer_output))
            # non linear activation for all but the last layer (RELU)
            if layer_output != output_size:
                modules.append(nn.ReLU())
        # add to layers as a single sequential module
        # so that the parameters are registered
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


network = Network(
    # Input order: agent [x, y, vx, vy, on_ground], then each opponent in
    # sorted ID order with the same five features.
    input_size=observation_size(),
    hidden_layers=network_config.hidden_layers,
    # Output order: IDLE, MOVE_LEFT, MOVE_RIGHT, JUMP, MOVE_LEFT_JUMP,
    # MOVE_RIGHT_JUMP. See agents/action.py.
    output_size=ACTION_COUNT,
)
