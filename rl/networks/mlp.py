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
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(self.activation_func_class())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, self.output_dim))

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


class GaussianActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: int,
        activation_func_class: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.mean_network = MLP(
            input_dim, hidden_size, output_dim, activation_func_class
        )
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_network(x)
        std = torch.exp(self.log_std)

        return mean, std.expand_as(mean)

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self(x)
        dist = torch.distributions.Normal(mean, std)
        sample = dist.sample()
        log_prob = dist.log_prob(sample).sum(dim=-1)

        return sample, log_prob


class ValueNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        activation_func_class: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        self.network = MLP(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_dim=1,
            activation_func_class=activation_func_class,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
