import torch
import torch.nn as nn

from rl.modules.ray_encoder import RayEncoder
from rl.modules.mlp import MLP
from rl.modules.rnn import RNN


class BaseActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: int | list[int],
        activation_func_class: type[nn.Module] = nn.Tanh,
        is_recurrent: bool = False,
    ) -> None:
        super().__init__()

        self.is_recurrent = is_recurrent
        self.memory = RNN(input_dim) if self.is_recurrent else None
        self.mlp_input_dim = self.memory.hidden_dim if self.is_recurrent else input_dim

        # As the MLP class expects a one-dimensional output dim, we have to sum
        # the output dimensions in case it is not one-dimensional.
        if isinstance(output_dim, list):
            mlp_output_dim = sum(output_dim)
        else:
            mlp_output_dim = output_dim

        self.network = MLP(
            self.mlp_input_dim, hidden_size, mlp_output_dim, activation_func_class
        )

    def forward(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor | None = None,
        prev_hidden_state: torch.Tensor | None = None,
    ):
        raise NotImplementedError

    def get_action(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        prev_hidden_state: torch.Tensor | None = None,
    ):
        raise NotImplementedError

    def get_hidden_state(self) -> torch.Tensor | None:
        return self.memory.hidden_state  # type: ignore

    def reset_memory(self, dones: torch.Tensor) -> None:
        if self.memory is None:
            return

        self.memory.reset(dones)


class MultiDiscreteActor(BaseActor):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: list[int],
        activation_func_class: type[nn.Module] = nn.Tanh,
        is_recurrent: bool = False,
    ) -> None:

        self.output_dim = output_dim
        super().__init__(
            input_dim,
            hidden_size,
            self.output_dim,
            activation_func_class,
            is_recurrent,
        )

    def forward(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor | None = None,
        prev_hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.is_recurrent:
            obs = self.memory(obs, masks, prev_hidden_state)
        flat_logits = self.network(obs)
        split_logits = torch.split(flat_logits, self.output_dim, dim=-1)
        return split_logits

    def get_action(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        prev_hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        split_logits = self(obs, masks, prev_hidden_state)

        multi_categoricals = [
            torch.distributions.Categorical(logits=logits) for logits in split_logits
        ]

        if action is None:
            action = torch.stack([dist.sample() for dist in multi_categoricals], dim=-1)

        log_prob = torch.stack(
            [
                dist.log_prob(sample)
                for dist, sample in zip(multi_categoricals, action.unbind(-1))
            ],
            dim=-1,
        ).sum(dim=-1, keepdim=True)

        entropy = torch.stack(
            [dist.entropy() for dist in multi_categoricals],
            dim=-1,
        ).sum(dim=-1, keepdim=True)

        return action, log_prob, entropy
