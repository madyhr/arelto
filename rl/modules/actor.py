import torch
import torch.nn as nn

from rl.networks import MLP


class BaseActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: int,
        activation_func_class: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.network = MLP(input_dim, hidden_size, output_dim, activation_func_class)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def get_action(self, x: torch.Tensor):
        raise NotImplementedError


class GaussianActor(BaseActor):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: int,
        activation_func_class: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__(input_dim, hidden_size, output_dim, activation_func_class)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.network(x)
        std = torch.exp(self.log_std)

        return mean, std.expand_as(mean)

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self(x)
        dist = torch.distributions.Normal(mean, std)
        sample = dist.sample()
        log_prob = dist.log_prob(sample).sum(dim=-1)

        return sample, log_prob


class DiscreteActor(BaseActor):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: int,
        activation_func_class: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__(input_dim, hidden_size, output_dim, activation_func_class)

    def forward(self, x) -> torch.Tensor:
        return self.network(x)

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self(x)
        dist = torch.distributions.Categorical(logits)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)
        return sample, log_prob
