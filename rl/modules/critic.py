import torch
import torch.nn as nn
from modules.ray_encoder import RayEncoder
from networks import MLP


class ValueCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        activation_func_class: type[nn.Module] = nn.Tanh,
        encoder: RayEncoder | None = None,
    ) -> None:
        super().__init__()

        mlp_input_dim = encoder.output_dim if encoder else input_dim

        self.network = MLP(
            input_dim=mlp_input_dim,
            hidden_size=hidden_size,
            output_dim=1,
            activation_func_class=activation_func_class,
        )

        self.encoder = encoder

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.encoder:
            obs = self.encoder(obs)
        return self.network(obs)
