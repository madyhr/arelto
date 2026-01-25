from __future__ import annotations

import typing

import torch
import torch.nn as nn

from rl.modules.ray_encoder import RayEncoder
from rl.networks.normalization import EmpiricalNormalization

if typing.TYPE_CHECKING:
    from rl.modules import BaseActor, ValueCritic


class ActorCritic(nn.Module):

    def __init__(
        self,
        actor_class: type[BaseActor],
        critic_class: type[ValueCritic],
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: int | list[int],
        activation_func_class: type[nn.Module] = nn.Tanh,
        encoder: RayEncoder | None = None,
    ) -> None:
        super().__init__()

        self.encoder = encoder

        self.critic: ValueCritic = critic_class(
            input_dim, hidden_size, activation_func_class, self.encoder
        )

        self.actor: BaseActor = actor_class(
            input_dim, hidden_size, output_dim, activation_func_class, self.encoder
        )

        # The observation space contains total_rays number of ray distances
        # then total_rays number of ray types. We only want to normalize
        # the distances as the types are categorical.
        self.norm_dim = self.encoder.total_rays if self.encoder else input_dim
        self.obs_normalizer = EmpiricalNormalization(self.norm_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        obs = self._normalize_obs(obs)
        action, log_prob, entropy = self.actor.get_action(obs, action)
        value = self.critic(obs)
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor):
        obs = self._normalize_obs(obs)
        return self.critic(obs)

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.encoder:
            continuous = obs[:, : self.norm_dim]
            discrete = obs[:, self.norm_dim :]
            norm_continuous = self.obs_normalizer(continuous)
            return torch.cat([norm_continuous, discrete], dim=-1)
        else:
            return self.obs_normalizer(obs)

    def update_normalization(self, obs: torch.Tensor):
        if self.encoder:
            self.obs_normalizer.update(obs[:, : self.norm_dim])
        else:
            self.obs_normalizer.update(obs)
