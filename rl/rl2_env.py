# python/rl2_env.py

import numpy as np
import rl2_py
import torch


class RL2Env:

    def __init__(self, step_dt: float = 0.02) -> None:
        self.game = rl2_py.Game()

        self.step_dt: float = step_dt
        self.num_envs: int = self.game.num_enemies

        self._obs_size: int = self.game.get_observation_size()
        self._act_size: int = self.game.get_action_size()
        self._rew_size: int = self.game.get_reward_size()
        self._np_obs_buf = np.zeros((self._obs_size, self.num_envs), dtype=np.float32)
        self._np_rew_buf = np.zeros((self.num_envs,), dtype=np.float32)
        self._np_terminated_buf = np.zeros((self.num_envs,), dtype=bool)
        self._np_truncated_buf = np.zeros((self.num_envs,), dtype=bool)

    def step(
        self, action: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert action is not None
        self.game.apply_action(action.t().detach().cpu().numpy())
        self.game.step(self.step_dt)
        self.game.fill_observation_buffer(self._np_obs_buf)
        self.game.fill_reward_buffer(self._np_rew_buf)
        self.game.fill_terminated_buffer(self._np_terminated_buf)
        self.game.fill_truncated_buffer(self._np_truncated_buf)

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def reset(self, seed: int | None = 42) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _get_obs(self) -> torch.Tensor:
        return torch.from_numpy(self._np_obs_buf).t()

    def _get_reward(self) -> torch.Tensor:
        return torch.from_numpy(self._np_rew_buf)

    def _get_info(self) -> torch.Tensor:
        return torch.zeros((1,))

    def _get_terminated(self) -> torch.Tensor:
        return torch.from_numpy(self._np_terminated_buf)

    def _get_truncated(self) -> torch.Tensor:
        return torch.from_numpy(self._np_truncated_buf)
