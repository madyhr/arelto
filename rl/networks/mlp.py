import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: int,
        activation_func_class: type[nn.Module] = nn.Tanh,
    ) -> None:
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.activation_func_class = activation_func_class

        self._verify_arg_validity()
        self._setup_network()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def _setup_network(self) -> None:
        current_dim = self.input_dim
        layers = []

        for hidden_dim in self.hidden_size:
            layers.append(layer_init(nn.Linear(current_dim, hidden_dim)))
            layers.append(self.activation_func_class())
            current_dim = hidden_dim

        layers.append(layer_init(nn.Linear(current_dim, self.output_dim)))

        self.network = nn.Sequential(*layers)

    def _verify_arg_validity(self) -> None:

        if self.input_dim <= 0:
            raise ValueError("Input dim must be a positive integer.")

        if self.output_dim <= 0:
            raise ValueError("Output dim must be a positive integer.")

        if not self.hidden_size:
            raise ValueError("Hidden size cannot be empty.")

        if any(h <= 0 for h in self.hidden_size):
            raise ValueError("Hidden dims must be positive.")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # Adapted from https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer
