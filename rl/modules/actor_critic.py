from __future__ import annotations

import typing

import torch
import torch.nn as nn

from rl.modules.ray_encoder import RayEncoder
from rl.modules.rnn import RNN

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
        encoder: RayEncoder,
        activation_func_class: type[nn.Module] = nn.Tanh,
        is_recurrent: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.is_recurrent = is_recurrent
        self.feature_dim = encoder.output_dim

        self.critic: ValueCritic = critic_class(
            self.feature_dim,
            hidden_size,
            activation_func_class,
            is_recurrent,
        )

        self.actor: BaseActor = actor_class(
            self.feature_dim,
            hidden_size,
            output_dim,
            activation_func_class,
            is_recurrent,
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        prev_hidden_states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        # When using recurrent mini batches, the shape differs between training
        # and inference, so these reshapes ensure the final shape is preserved
        batch_shape_obs = obs.shape[:-1]
        latent_obs = self.encoder(obs.reshape(-1, obs.shape[-1]))
        latent_obs = latent_obs.reshape(*batch_shape_obs, self.feature_dim)

        prev_hidden_state_a = (
            prev_hidden_states[0] if prev_hidden_states is not None else None
        )
        action, log_prob, entropy = self.actor.get_action(
            latent_obs, action, masks, prev_hidden_state_a
        )

        prev_hidden_state_c = (
            prev_hidden_states[1] if prev_hidden_states is not None else None
        )
        value = self.critic(latent_obs, masks, prev_hidden_state_c)

        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor):
        latent_obs = self.encoder(obs)
        return self.critic(latent_obs)

    @torch.no_grad()
    def get_bootstrap_value(self, obs: torch.Tensor) -> torch.Tensor:
        latent_obs = self.encoder(obs)

        if not self.is_recurrent:
            return self.critic(latent_obs)

        with self.critic.preserved_hidden_state():
            return self.critic(latent_obs)
