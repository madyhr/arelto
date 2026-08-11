import contextlib

import torch
import torch.nn as nn

from rl.modules.mlp import MLP
from rl.modules.rnn import RNN


class ValueCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        activation_func_class: type[nn.Module] = nn.Tanh,
        is_recurrent: bool = False,
    ) -> None:
        super().__init__()

        self.is_recurrent = is_recurrent
        self.memory = RNN(input_dim) if self.is_recurrent else None
        self.mlp_input_dim = self.memory.hidden_dim if self.is_recurrent else input_dim

        self.network = MLP(
            input_dim=self.mlp_input_dim,
            hidden_size=hidden_size,
            output_dim=1,
            activation_func_class=activation_func_class,
        )

    def forward(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor | None = None,
        prev_hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.is_recurrent:
            obs = self.memory(obs, masks, prev_hidden_state)
        return self.network(obs)

    def get_hidden_state(self) -> torch.Tensor | None:
        if self.memory is None:
            return None

        return self.memory.hidden_state

    def reset_memory(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> None:
        if self.memory is None:
            return

        self.memory.reset(dones, hidden_state)

    @contextlib.contextmanager
    def preserved_hidden_state(self):
        hidden_state = self.get_hidden_state()
        hidden_state_clone = hidden_state.clone() if hidden_state is not None else None
        try:
            yield
        finally:
            self.reset_memory(hidden_state=hidden_state_clone)
