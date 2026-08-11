import torch
import torch.nn as nn

from rl.utils.trajectory_padding import combine_and_unpad_trajectories


class RNN(nn.Module):
    def __init__(
        self, input_size: int, hidden_dim: int = 128, num_layers: int = 1
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(self.input_size, self.hidden_dim, self.num_layers)
        self.hidden_state: torch.Tensor | None = None

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor | None,
        hidden_state: torch.Tensor | None,
    ) -> torch.Tensor:

        is_batch = masks is not None

        if is_batch:
            out, _ = self.rnn(input, hidden_state)
            out = combine_and_unpad_trajectories(out, masks)
        else:
            out, self.hidden_state = self.rnn(input.unsqueeze(0), self.hidden_state)
            out = out.squeeze(0)

        return out

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> None:
        if dones is None:
            self.hidden_state = None if hidden_state is None else hidden_state
        elif self.hidden_state is not None:
            if hidden_state is None:
                done_mask = dones.reshape(-1).bool()
                self.hidden_state[:, done_mask, :] = 0.0

    def detach_hidden_state(self):
        raise NotImplementedError
