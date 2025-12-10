import torch
import torch.nn as nn

from rl.networks.mlp import GaussianActor, ValueNetwork


class ActorCritic(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: int,
        activation_func_class: type[nn.Module],
    ) -> None:
        super().__init__()
        self.actor: GaussianActor = GaussianActor(
            input_dim, hidden_size, output_dim, activation_func_class
        )

        self.critic: ValueNetwork = ValueNetwork(
            input_dim, hidden_size, activation_func_class
        )

    def forward(self, state: torch.Tensor):
        action, log_prob = self.actor.get_action(state)
        value = self.critic(state)
        return action, log_prob, value

    def get_value(self, state: torch.Tensor):
        return self.critic(state)
