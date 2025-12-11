# python/test_model.py

import torch


class TestModel:
    def __init__(self, num_envs: int, action_dim: int) -> None:
        self.num_envs: int = num_envs
        self.action_dim: int = action_dim

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        # 0 = left, 1 = noop, 2 = right
        action = torch.randint(0, 3, (self.num_envs, self.action_dim))
        return action
