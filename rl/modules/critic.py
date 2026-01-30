import torch
import torch.nn as nn

from rl.modules.ray_encoder import RayEncoder
from rl.networks import MLP


class ValueCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        encoder: RayEncoder,
        activation_func_class: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        mlp_input_dim = encoder.output_dim

        self.network = MLP(
            input_dim=mlp_input_dim,
            hidden_size=hidden_size,
            output_dim=1,
            activation_func_class=activation_func_class,
        )

        self.encoder = encoder

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.encoder(obs)
        return self.network(obs)
