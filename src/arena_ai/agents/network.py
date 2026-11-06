# network.py
# This module defines the neural network architecture for the AI agent.

import torch
import torch.nn.functional as F
from arena_ai.config import load_config

config = load_config()
network_config = config.agent.network

device = "cuda" if torch.cuda.is_available() else "cpu"


# One layer of the neural network, consisting of weights and biases.
class Layer:
    def __init__(self, input_size, output_size):
        self.weights = torch.randn(
            input_size, output_size, device=device, requires_grad=True
        )
        self.biases = torch.zeros(output_size, device=device, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.weights) + self.biases


# The neural network class that consists of multiple layers.
class Network:
    def __init__(self, input_size, hidden_layers, output_size):
        self.layers = []
        layer_sizes = [input_size, *hidden_layers, output_size]
        for input_size, output_size in zip(layer_sizes, layer_sizes[1:]):
            self.layers.append(Layer(input_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer.forward(x))
        return x


network = Network(
    input_size=network_config.input_size,
    hidden_layers=network_config.hidden_layers,
    output_size=network_config.output_size,
)
