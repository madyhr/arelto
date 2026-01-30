import torch
import torch.nn as nn

from rl.modules.ray_encoder import RayEncoder
from rl.networks import MLP


class BaseActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: int | list[int],
        encoder: RayEncoder,
        activation_func_class: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        mlp_input_dim = encoder.output_dim

        # As the MLP class expects a one-dimensional output dim, we have to sum
        # the output dimensions in case it is not one-dimensional.
        if isinstance(output_dim, list):
            mlp_output_dim = sum(output_dim)
        else:
            mlp_output_dim = output_dim
        self.network = MLP(
            mlp_input_dim, hidden_size, mlp_output_dim, activation_func_class
        )

        self.encoder = encoder

    def forward(self, obs: torch.Tensor):
        raise NotImplementedError

    def get_action(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        raise NotImplementedError


class MultiDiscreteActor(BaseActor):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: list[int],
        encoder: RayEncoder,
        activation_func_class: type[nn.Module] = nn.Tanh,
    ) -> None:

        self.output_dim = output_dim
        super().__init__(
            input_dim,
            hidden_size,
            self.output_dim,
            encoder,
            activation_func_class,
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        obs = self.encoder(obs)
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
        ).sum(dim=-1, keepdim=True)

        entropy = torch.stack(
            [dist.entropy() for dist in multi_categoricals],
            dim=-1,
        ).sum(dim=-1, keepdim=True)

        return action, log_prob, entropy
