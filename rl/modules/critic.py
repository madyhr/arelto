import torch
import torch.nn as nn
from networks import MLP


class ValueCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        activation_func_class: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        self.network = MLP(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_dim=1,
            activation_func_class=activation_func_class,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
