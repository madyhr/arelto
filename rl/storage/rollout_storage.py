from collections.abc import Generator
from dataclasses import dataclass

import torch

from rl.utils.trajectory_padding import split_and_pad_trajectories


@dataclass
class Transition:
    """Storage for a single state transition."""

    observation: torch.Tensor | None = None
    action: torch.Tensor | None = None
    reward: torch.Tensor | None = None
    done: torch.Tensor | None = None
    value: torch.Tensor | None = None
    action_log_prob: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)

    def clear(self) -> None:
        """Resets the dataclass to its default values."""
        self.__init__()


class RolloutStorage:
    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        observations,
        actions,
        hidden_states: tuple[torch.Tensor, torch.Tensor] | None = None,
        device="cpu",
    ) -> None:

        self.transition = Transition()
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.device = device

        self.observations = torch.zeros(
            (self.num_transitions_per_env, *observations.shape), device=self.device
        )
        self.actions = torch.zeros(
            (self.num_transitions_per_env, *actions.shape), device=self.device
        )
        self.rewards = torch.zeros(
            (num_transitions_per_env, self.num_envs, 1), device=self.device
        )
        self.dones = torch.zeros(
            (num_transitions_per_env, self.num_envs, 1), device=self.device
        )
        self.values = torch.zeros(
            (num_transitions_per_env, self.num_envs, 1), device=self.device
        )
        self.action_log_probs = torch.zeros(
            (num_transitions_per_env, self.num_envs, 1), device=self.device
        )
        self.advantages = torch.zeros(
            (num_transitions_per_env, self.num_envs, 1), device=self.device
        )
        self.returns = torch.zeros(
            (num_transitions_per_env, self.num_envs, 1), device=self.device
        )
        self.saved_hidden_state_a: torch.Tensor | None = None
        self.saved_hidden_state_c: torch.Tensor | None = None

        self.step = 0
        self.batch_size = self.num_envs * self.num_transitions_per_env

    def add_transition(self, transition: Transition) -> None:
        assert transition.observation is not None
        assert transition.action is not None
        assert transition.reward is not None
        assert transition.done is not None
        assert transition.value is not None
        assert transition.action_log_prob is not None

        # Define the idx for the current transition using mod to create a circular buffer
        # that allows continuous collection of data.
        idx = self.step % self.num_transitions_per_env

        self.observations[idx].copy_(transition.observation)
        self.actions[idx].copy_(transition.action)
        self.rewards[idx].copy_(transition.reward.view(-1, 1))
        self.dones[idx].copy_(transition.done.view(-1, 1))
        self.values[idx].copy_(transition.value)
        self.action_log_probs[idx].copy_(transition.action_log_prob.view(-1, 1))
        self._save_hidden_states(transition.hidden_states)

        self.step += 1

    def _save_hidden_states(
        self, hidden_states: tuple[torch.Tensor | None, torch.Tensor | None]
    ) -> None:

        if hidden_states == (None, None) or hidden_states is None:
            return

        actor_hidden, critic_hidden = hidden_states
        # Using index for current transition in the circular buffer
        idx = self.step % self.num_transitions_per_env

        if actor_hidden is not None:
            if self.saved_hidden_state_a is None:
                self.saved_hidden_state_a = actor_hidden.new_zeros(
                    self.num_transitions_per_env, *actor_hidden.shape
                )
            self.saved_hidden_state_a[idx].copy_(actor_hidden)

        if critic_hidden is not None:
            if self.saved_hidden_state_c is None:
                self.saved_hidden_state_c = critic_hidden.new_zeros(
                    self.num_transitions_per_env, *critic_hidden.shape
                )
            self.saved_hidden_state_c[idx].copy_(critic_hidden)

    def clear(self):
        self.step = 0

    def get_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int
    ) -> Generator:
        mini_batch_size = self.batch_size // num_mini_batches
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        values = self.values.flatten(0, 1)
        old_action_log_probs = self.action_log_probs.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)

        for epoch in range(num_epochs):
            indices = torch.randperm(
                num_mini_batches * mini_batch_size,
                requires_grad=False,
                device=self.device,
            )
            for i in range(num_mini_batches):
                begin = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[begin:end]

                yield (
                    observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_action_log_probs[batch_idx],
                )

    def recurrent_get_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int
    ) -> Generator:
        """Get mini-batch generator when using RNNs, where mini batches must be
        chronological.

        Legend:
            - B: Number of envs
            - T: Number of transitions
            - K: Number of transition fragments (transitions split by dones)
            - H: Hidden dim
            - L: Number of layers
        """
        if not 1 <= num_mini_batches <= self.num_envs:
            raise ValueError("num_mini_batches must be between 1 and num_envs")

        observations = self.to_chronological(self.observations)
        dones = self.to_chronological(self.dones)
        actions = self.to_chronological(self.actions)
        values = self.to_chronological(self.values)
        advantages = self.to_chronological(self.advantages)
        returns = self.to_chronological(self.returns)
        old_action_log_probs = self.to_chronological(self.action_log_probs)
        actor_hidden = self.to_chronological(self.saved_hidden_state_a)
        critic_hidden = self.to_chronological(self.saved_hidden_state_c)
        padded_obs, traj_masks = split_and_pad_trajectories(observations, dones)

        done_flags = dones.squeeze(-1).bool()
        traj_starts = torch.zeros_like(done_flags)
        traj_starts[0] = True
        traj_starts[1:] = done_flags[:-1]

        # [T, L, B, H] -> [B, T, L, H]
        actor_hidden_by_env = actor_hidden.permute(2, 0, 1, 3)
        critic_hidden_by_env = critic_hidden.permute(2, 0, 1, 3)

        # [T, B] -> [B, T]
        traj_starts_by_env = traj_starts.transpose(0, 1)

        actor_hidden_initial = actor_hidden_by_env[traj_starts_by_env]
        critic_hidden_initial = critic_hidden_by_env[traj_starts_by_env]

        # `nn.GRU` expects the hidden state in format [layers, traj, hidden_dim]
        # [K, L, H] -> [L, K, H]
        actor_hidden_initial = actor_hidden_initial.transpose(0, 1).contiguous()
        critic_hidden_initial = critic_hidden_initial.transpose(0, 1).contiguous()

        num_fragments_per_env = traj_starts.sum(dim=0)

        traj_offsets = torch.cat(
            [num_fragments_per_env.new_zeros(1), num_fragments_per_env.cumsum(dim=0)]
        )

        # This is in order to keep every fragment belonging to an env in the
        # same minibatch and to distribute remainder environments so each env
        # appears once per epoch.
        envs_per_batch, remainder = divmod(self.num_envs, num_mini_batches)

        for epoch in range(num_epochs):
            env_start = 0

            for mini_batch_idx in range(num_mini_batches):
                num_envs_in_batch = envs_per_batch + (mini_batch_idx < remainder)
                env_stop = env_start + num_envs_in_batch

                traj_start = int(traj_offsets[env_start].item())
                traj_stop = int(traj_offsets[env_stop].item())

                yield (
                    padded_obs[:, traj_start:traj_stop],
                    actions[:, env_start:env_stop],
                    values[:, env_start:env_stop],
                    advantages[:, env_start:env_stop],
                    returns[:, env_start:env_stop],
                    old_action_log_probs[:, env_start:env_stop],
                    (
                        actor_hidden_initial[:, traj_start:traj_stop],
                        critic_hidden_initial[:, traj_start:traj_stop],
                    ),
                    traj_masks[:, traj_start:traj_stop],
                )

                env_start = env_stop

    def to_chronological(self, batch: torch.Tensor) -> torch.Tensor:
        num_valid_transitions = min(self.step, self.num_transitions_per_env)
        start_idx = (self.step - num_valid_transitions) % self.num_transitions_per_env
        offsets = torch.arange(num_valid_transitions, device=batch.device)
        indices = (start_idx + offsets) % self.num_transitions_per_env
        return batch.index_select(0, indices)
