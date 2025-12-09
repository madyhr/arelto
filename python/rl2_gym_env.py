# python/rl2_gym_env.py

import os
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))

import rl2_py


class RL2Env(gym.Env):

    def __init__(self, step_dt=0.02) -> None:
        self.step_dt = step_dt

        self.game = rl2_py.Game()
        self.num_envs: int = self.game.num_enemies

        self._obs_size: int = self.game.get_observation_size()
        self._act_size: int = self.game.get_action_size()
        self._rew_size: int = self.game.get_reward_size()
        self._np_obs_buf = np.zeros(self._obs_size, dtype=np.float32)
        self._np_rew_buf = np.zeros(self._rew_size, dtype=np.float32)
        self._np_done_buf = np.zeros(self.num_envs, dtype=bool)

        super().__init__()

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.game.apply_action(action.detach().numpy())
        self.game.step(self.step_dt)
        self.game.fill_observation_buffer(self._np_obs_buf)
        self.game.fill_reward_buffer(self._np_rew_buf)
        self.game.fill_dones_buffer(self._np_done_buf)

        obs = torch.from_numpy(self._np_obs_buf)
        reward = torch.from_numpy(self._np_rew_buf)
        dones = torch.from_numpy(self._np_done_buf)
        info = torch.zeros((1,))

        return obs, reward, info, dones

    def reset(self, seed: int | None = 42) -> tuple[torch.Tensor, torch.Tensor]:
        super().reset(seed=seed)

        obs = torch.from_numpy(self._np_obs_buf)
        info = torch.zeros((1,))

        return obs, info
