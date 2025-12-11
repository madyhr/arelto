import typing
from dataclasses import MISSING

import torch
import torch.nn as nn

from rl.networks.mlp import DiscreteActor, GaussianActor, ValueNetwork

if typing:
    from rl.networks.mlp import Actor


class BaseActorCritic(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: int,
        activation_func_class: type[nn.Module],
    ) -> None:
        super().__init__()

        self.critic: ValueNetwork = ValueNetwork(
            input_dim, hidden_size, activation_func_class
        )

        self.actor: Actor = MISSING

    def forward(self, state: torch.Tensor):
        action, log_prob = self.actor.get_action(state)
        value = self.critic(state)
        return action, log_prob, value

    def get_value(self, state: torch.Tensor):
        return self.critic(state)


class DiscreteActorCritic(BaseActorCritic):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: int,
        activation_func_class: type[nn.Module],
    ) -> None:
        super().__init__(input_dim, hidden_size, output_dim, activation_func_class)

        self.actor: DiscreteActor = DiscreteActor(
            input_dim, hidden_size, output_dim, activation_func_class
        )


class ContinuousActorCritic(BaseActorCritic):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: int,
        activation_func_class: type[nn.Module],
    ) -> None:
        super().__init__(input_dim, hidden_size, output_dim, activation_func_class)

        self.actor: GaussianActor = GaussianActor(
            input_dim, hidden_size, output_dim, activation_func_class
        )
