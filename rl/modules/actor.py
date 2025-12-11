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

    def forward(self, x) -> list[torch.Tensor]:
        flat_logits = self.network(x)
        split_logits = torch.split(flat_logits, self.output_dim, dim=-1)
        return list(split_logits)

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        split_logits = self(x)

        actions = []
        log_probs = []

        for logits in split_logits:
            dist = torch.distributions.Categorical(logits=logits)
            sample = dist.sample()
            actions.append(sample)
            log_probs.append(dist.log_prob(sample))

        action_tensor = torch.stack(actions, dim=-1)
        log_prob_tensor = torch.stack(log_probs, dim=-1).sum(dim=-1)

        return action_tensor, log_prob_tensor
