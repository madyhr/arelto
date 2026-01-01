from __future__ import annotations

import typing

import torch
import torch.nn as nn

if typing.TYPE_CHECKING:
    from rl.modules import BaseActor, ValueCritic


class ActorCritic(nn.Module):

    def __init__(
        self,
        actor_class: type[BaseActor],
        critic_class: type[ValueCritic],
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: int | list[int],
        activation_func_class: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        self.critic: ValueCritic = critic_class(
            input_dim, hidden_size, activation_func_class
        )

        self.actor: BaseActor = actor_class(
            input_dim,
            hidden_size,
            output_dim,
            activation_func_class,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        action, log_prob, entropy = self.actor.get_action(obs, action)
        value = self.critic(obs)
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor):
        return self.critic(obs)
