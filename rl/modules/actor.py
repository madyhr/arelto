import torch
import torch.nn as nn
from networks import MLP


class BaseActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: int | list[int],
        activation_func_class: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        # As the MLP class expects a one-dimensional output dim, we have to sum
        # the output dimensions in case it is not one-dimensional.
        if isinstance(output_dim, list):
            mlp_output_dim = sum(output_dim)
        else:
            mlp_output_dim = output_dim
        self.network = MLP(
            input_dim, hidden_size, mlp_output_dim, activation_func_class
        )

    def forward(self, obs: torch.Tensor):
        raise NotImplementedError

    def get_action(self, obs: torch.Tensor, action: torch.Tensor | None = None):
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

    def forward(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.network(obs)
        std = torch.exp(self.log_std)

        return mean, std.expand_as(mean)

    def get_action(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self(obs)
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

    def forward(self, obs) -> torch.Tensor:
        return self.network(obs)

    def get_action(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)
        return sample, log_prob


class MultiDiscreteActor(BaseActor):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: list[int],
        activation_func_class: type[nn.Module] = nn.Tanh,
    ) -> None:

        self.output_dim = output_dim
        super().__init__(input_dim, hidden_size, self.output_dim, activation_func_class)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        flat_logits = self.network(obs)
        split_logits = torch.split(flat_logits, self.output_dim, dim=-1)
        return split_logits

    def get_action(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        split_logits = self(obs)

        multi_categoricals = [
            torch.distributions.Categorical(logits=logits) for logits in split_logits
        ]

        if action is None:
            action = torch.stack([dist.sample() for dist in multi_categoricals], dim=-1)

        log_prob = torch.stack(
            [
                dist.log_prob(sample)
                for dist, sample in zip(multi_categoricals, action.T)
            ],
            dim=-1,
        ).sum(dim=-1)

        entropy = torch.stack(
            [dist.entropy() for dist in multi_categoricals],
            dim=-1,
        ).sum(dim=-1)

        return action, log_prob, entropy
