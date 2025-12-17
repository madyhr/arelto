from collections.abc import Generator
from dataclasses import dataclass

import torch


@dataclass
class Transition:
    """Storage for a single state transition."""

    observation: torch.Tensor | None = None
    action: torch.Tensor | None = None
    reward: torch.Tensor | None = None
    done: torch.Tensor | None = None
    value: torch.Tensor | None = None
    action_log_prob: torch.Tensor | None = None

    def clear(self) -> None:
        """Resets the dataclass to its default values."""
        self.__init__()


class RolloutStorage:
    def __init__(
        self, num_envs, num_transitions_per_env, observations, actions, device="cpu"
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

        self.step = 0
        self.batch_size = self.num_envs * self.num_transitions_per_env

    def add_transition(self, transition: Transition) -> None:
        assert transition.observation is not None
        assert transition.action is not None
        assert transition.reward is not None
        assert transition.done is not None
        assert transition.value is not None
        assert transition.action_log_prob is not None
        assert self.step < self.num_transitions_per_env

        self.observations[self.step].copy_(transition.observation)
        self.actions[self.step].copy_(transition.action)
        self.rewards[self.step].copy_(transition.reward.view(-1, 1))
        self.dones[self.step].copy_(transition.done.view(-1, 1))
        self.values[self.step].copy_(transition.value)
        self.action_log_probs[self.step].copy_(transition.action_log_prob.view(-1, 1))

        self.step += 1

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
