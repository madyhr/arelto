# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from RSL-RL:
# https://github.com/leggedrobotics/rsl_rl/blob/<commit>/tests/algorithms/test_ppo.py
# Modifications Copyright (c) 2026 Marcus Dyhr

from __future__ import annotations
from collections.abc import Iterator

import pytest
import torch

from rl.algorithms.ppo import PPO
from rl.storage.rollout_storage import Batch


NUM_ENVS = 2
NUM_STEPS = 4
NUM_RAYS = 8
NUM_ENTITY_TYPES = 3
INPUT_DIM = 4 * NUM_RAYS
OUTPUT_DIM = [2, 3]


@pytest.fixture
def mock_env_config() -> dict[str, object]:
    return {
        "num_envs": NUM_ENVS,
        "input_dim": INPUT_DIM,
        "num_rays": NUM_RAYS,
        "num_entity_types": NUM_ENTITY_TYPES,
        "ray_history_length": 1,
        "output_dim": OUTPUT_DIM,
        "device": "cpu",
        "is_recurrent": True,
    }


def _make_obs(num_envs: int = NUM_ENVS) -> torch.Tensor:
    """Build observations that satisfy the distance/type layout of RayEncoder."""
    distances = torch.randn(num_envs, 2 * NUM_RAYS)
    entity_types = torch.randint(NUM_ENTITY_TYPES, (num_envs, 2 * NUM_RAYS)).float()
    return torch.cat((distances, entity_types), dim=-1)


def _build_ppo(**overrides: object) -> PPO:
    """Build a small PPO instance using the project's real policy and storage."""
    config: dict[str, object] = {
        "num_envs": NUM_ENVS,
        "input_dim": INPUT_DIM,
        "hidden_size": [16],
        "output_dim": OUTPUT_DIM,
        "num_transitions_per_env": NUM_STEPS,
        "num_mini_batches": 2,
        "num_epochs": 1,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 1e-3,
        "clip_coef": 0.2,
        "entropy_loss_coef": 0.01,
        "value_loss_coef": 0.5,
        "max_grad_norm": 1.0,
        "num_rays": NUM_RAYS,
        "num_entity_types": NUM_ENTITY_TYPES,
        "encoder_output_dim": 8,
        "is_recurrent": False,
        "device": "cpu",
    }
    config.update(overrides)
    return PPO(**config)  # type: ignore


def test_ppo_valid_initialization(mock_env_config: dict[str, object]) -> None:
    ppo = PPO(
        **mock_env_config,  # type: ignore
        hidden_size=[16],
        encoder_output_dim=8,
        num_transitions_per_env=NUM_STEPS,
    )

    assert ppo.input_dim == INPUT_DIM
    assert ppo.encoder.num_rays == NUM_RAYS
    assert ppo.storage.observations.shape == (NUM_STEPS, NUM_ENVS, INPUT_DIM)
    assert ppo.policy.is_recurrent


def test_ppo_invalid_shape_mismatch(mock_env_config: dict[str, object]) -> None:
    mock_env_config["input_dim"] = INPUT_DIM + 1

    with pytest.raises(ValueError, match="Shape mismatch: Environment obs_size"):
        PPO(**mock_env_config)  # type: ignore


class TestGAEComputation:
    """Generalized advantage estimation through PPO.compute_returns."""

    def test_gae_returns_match_hand_computed_values(self) -> None:
        gamma, lam = 0.99, 0.95
        ppo = _build_ppo(
            num_envs=1,
            num_transitions_per_env=3,
            num_mini_batches=1,
            gamma=gamma,
            lam=lam,
        )
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 1.0, 1.5])
        ppo.storage.rewards[:, 0, 0] = rewards
        ppo.storage.values[:, 0, 0] = values
        ppo.storage.dones.zero_()

        ppo.compute_returns(_make_obs(1), bootstrap_value=torch.tensor([[2.0]]))

        expected_advantages = torch.tensor(
            [
                1.49 + gamma * lam * (2.485 + gamma * lam * 3.48),
                2.485 + gamma * lam * 3.48,
                3.48,
            ]
        )
        expected_returns = expected_advantages + values
        torch.testing.assert_close(ppo.storage.advantages[:, 0, 0], expected_advantages)
        torch.testing.assert_close(ppo.storage.returns[:, 0, 0], expected_returns)

    def test_terminal_state_cuts_bootstrap(self) -> None:
        ppo = _build_ppo(
            num_envs=1,
            num_transitions_per_env=2,
            num_mini_batches=1,
            gamma=0.99,
            lam=0.95,
        )
        ppo.storage.rewards[:, 0, 0] = torch.tensor([1.0, 2.0])
        ppo.storage.values[:, 0, 0] = torch.tensor([0.5, 1.0])
        ppo.storage.dones[:, 0, 0] = torch.tensor([1.0, 0.0])

        ppo.compute_returns(_make_obs(1), bootstrap_value=torch.tensor([[3.0]]))

        # At step 0, done=1 removes both the value bootstrap and recursive GAE.
        torch.testing.assert_close(
            ppo.storage.advantages[:, 0, 0], torch.tensor([0.5, 3.97])
        )
        torch.testing.assert_close(
            ppo.storage.returns[:, 0, 0], torch.tensor([1.0, 4.97])
        )

    def test_gae_uses_circular_buffer_chronology(self) -> None:
        ppo = _build_ppo(
            num_envs=1,
            num_transitions_per_env=3,
            num_mini_batches=1,
            gamma=0.9,
            lam=1.0,
        )
        # Five collected transitions leave absolute steps 2, 3, and 4 in
        # physical slots 2, 0, and 1 respectively.
        ppo.storage.rewards[:, 0, 0] = torch.tensor([4.0, 5.0, 3.0])
        ppo.storage.values.zero_()
        ppo.storage.dones.zero_()

        ppo.compute_returns(
            _make_obs(1),
            total_steps_collected=5,
            bootstrap_value=torch.tensor([[10.0]]),
        )

        expected_chronological_returns = torch.tensor(
            [
                3.0 + 0.9 * (4.0 + 0.9 * (5.0 + 0.9 * 10.0)),
                4.0 + 0.9 * (5.0 + 0.9 * 10.0),
                5.0 + 0.9 * 10.0,
            ]
        )
        physical_returns = ppo.storage.returns[:, 0, 0]
        chronological_returns = physical_returns[torch.tensor([2, 0, 1])]
        torch.testing.assert_close(
            chronological_returns, expected_chronological_returns
        )


def test_act_and_process_env_step_store_a_transition() -> None:
    torch.manual_seed(0)
    ppo = _build_ppo()
    obs = _make_obs()
    rewards = torch.tensor([1.0, -0.5])
    dones = torch.tensor([False, True])

    actions = ppo.act(obs)
    assert actions is not None
    normalized_obs = ppo.transition.observation.clone()  # type: ignore
    ppo.process_env_step(obs, rewards, dones)

    assert actions.shape == (NUM_ENVS, len(OUTPUT_DIM))
    assert ppo.storage.step == 1
    torch.testing.assert_close(ppo.storage.observations[0], normalized_obs)
    torch.testing.assert_close(ppo.storage.actions[0], actions.float())
    torch.testing.assert_close(ppo.storage.rewards[0, :, 0], rewards)
    torch.testing.assert_close(ppo.storage.dones[0, :, 0], dones.float())
    assert ppo.obs_normalizer.count.item() == NUM_ENVS
    assert ppo.transition.observation is None


class TestPPOLosses:
    """Loss checks that execute PPO.update rather than duplicate its equations."""

    @staticmethod
    def _replace_single_batch(
        ppo: PPO,
        monkeypatch: pytest.MonkeyPatch,
        *,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        new_log_probs: torch.Tensor,
        new_values: torch.Tensor,
    ) -> None:
        batch_size = advantages.shape[0]
        batch = Batch(
            observations=_make_obs(batch_size),
            actions=torch.zeros(batch_size, len(OUTPUT_DIM)),
            returns=returns,
            values=old_values,
            advantages=advantages,
            action_log_prob=old_log_probs,
        )

        def one_batch(*_args: object) -> Iterator[Batch]:
            return iter((batch,))

        probe = next(ppo.policy.parameters()).reshape(-1)[0]

        def fixed_policy(*_args: object, **_kwargs: object):
            # Keep the synthetic outputs connected to the optimizer parameters
            # so PPO.update can perform its real backward pass.
            zero_grad = probe * 0.0
            return (
                batch.actions,
                new_log_probs + zero_grad,
                torch.zeros_like(new_log_probs) + zero_grad,
                new_values + zero_grad,
            )

        monkeypatch.setattr(ppo.storage, "get_mini_batch_generator", one_batch)
        monkeypatch.setattr(ppo.policy, "forward", fixed_policy)

    def test_surrogate_loss_clips_both_ratio_directions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ppo = _build_ppo(num_envs=3, num_transitions_per_env=1, num_mini_batches=1)
        advantages = torch.tensor([[-1.0], [0.0], [1.0]])
        old_log_probs = torch.zeros(3, 1)
        new_log_probs = torch.tensor([[-0.5], [0.0], [0.5]])
        zeros = torch.zeros(3, 1)
        self._replace_single_batch(
            ppo,
            monkeypatch,
            advantages=advantages,
            old_log_probs=old_log_probs,
            returns=zeros,
            old_values=zeros,
            new_log_probs=new_log_probs,
            new_values=zeros,
        )

        metrics = ppo.update()

        # The normalized advantages remain [-1, 0, 1]. Both non-zero samples
        # exceed the 0.2 ratio clip, producing [0.8, 0, -1.2].
        assert metrics["loss/policy"] == pytest.approx(-0.4 / 3)
        assert metrics["tech/clip_fraction"] == pytest.approx(2 / 3)

    def test_value_loss_uses_clipped_reference_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ppo = _build_ppo(num_envs=2, num_transitions_per_env=1, num_mini_batches=1)
        old_values = torch.tensor([[1.0], [1.0]])
        new_values = torch.tensor([[2.0], [1.1]])
        returns = torch.tensor([[1.5], [1.5]])
        self._replace_single_batch(
            ppo,
            monkeypatch,
            advantages=torch.tensor([[-1.0], [1.0]]),
            old_log_probs=torch.zeros(2, 1),
            returns=returns,
            old_values=old_values,
            new_log_probs=torch.zeros(2, 1),
            new_values=new_values,
        )

        metrics = ppo.update()

        # PPO applies the conventional 0.5 factor to mean(max(0.25, 0.09),
        # max(0.16, 0.16)).
        assert metrics["loss/value"] == pytest.approx(0.5 * (0.25 + 0.16) / 2)


@pytest.mark.parametrize("is_recurrent", [False, True])
def test_update_runs_end_to_end_and_reports_finite_metrics(
    is_recurrent: bool,
) -> None:
    torch.manual_seed(1)
    ppo = _build_ppo(is_recurrent=is_recurrent)

    for step in range(NUM_STEPS):
        obs = _make_obs()
        ppo.act(obs)
        ppo.process_env_step(
            obs,
            torch.tensor([1.0 + step, -0.25 + step]),
            torch.zeros(NUM_ENVS, dtype=torch.bool),
        )

    ppo.compute_returns(_make_obs(), bootstrap_value=torch.zeros(NUM_ENVS, 1))
    parameters_before = [
        parameter.detach().clone() for parameter in ppo.policy.parameters()
    ]

    metrics = ppo.update()

    assert set(metrics) == {
        "loss/policy",
        "loss/value",
        "loss/entropy",
        "loss/total",
        "tech/kl_divergence",
        "tech/clip_fraction",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, ppo.policy.parameters())
    )
    assert ppo.storage.step == 0
