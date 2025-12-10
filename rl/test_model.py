# python/test_model.py

import torch


class TestModel:
    def __init__(self, num_envs: int, action_dim: int) -> None:
        self.num_envs: int = num_envs
        self.action_dim: int = action_dim

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:

        action = (
            2
            * torch.rand(
                size=(self.num_envs, self.action_dim),
                dtype=torch.float32,
            )
            - 1
        )
        return action
