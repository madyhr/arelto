from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import nn

from rl.algorithms.async_ppo import AsyncPPO


NUM_ENVS = 2
NUM_RAYS = 4
INPUT_DIM = 4 * NUM_RAYS
OUTPUT_DIM = [2, 3]
PPO_CONFIG = {
    "num_envs": NUM_ENVS,
    "input_dim": INPUT_DIM,
    "hidden_size": [8],
    "output_dim": OUTPUT_DIM,
    "num_transitions_per_env": 1,
    "num_mini_batches": 1,
    "num_epochs": 1,
    "num_rays": NUM_RAYS,
    "num_entity_types": 3,
    "encoder_output_dim": 4,
    "is_recurrent": False,
    "device": "cuda",
}

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="AsyncPPO publication tests require CUDA",
)


@pytest.fixture
def async_ppo() -> AsyncPPO:
    ppo = AsyncPPO(**PPO_CONFIG)  # type: ignore
    torch.cuda.synchronize()
    return ppo


@pytest.fixture
def observation() -> torch.Tensor:
    return torch.zeros(NUM_ENVS, INPUT_DIM, device="cuda")


@torch.no_grad()
def _fill_parameters(module: nn.Module, value: float) -> None:
    for parameter in module.parameters():
        parameter.fill_(value)


def _assert_parameters_equal(module: nn.Module, value: float) -> None:
    for parameter in module.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, value))


@pytest.fixture
def ppo_with_unpublished_update(
    async_ppo: AsyncPPO,
    observation: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[AsyncPPO]:
    """Complete training while leaving its policy waiting for publication."""

    _fill_parameters(async_ppo.inference_policy, 0.0)
    _fill_parameters(async_ppo.learner.policy, 1.0)
    torch.cuda.synchronize()

    monkeypatch.setattr(async_ppo.learner, "compute_returns", lambda *_args: None)
    monkeypatch.setattr(async_ppo.learner, "update", lambda: {})

    bootstrap_value = torch.zeros(NUM_ENVS, 1, device="cuda")

    # Run the worker body directly so completion is deterministic and does not
    # depend on thread scheduling.
    with torch.cuda.stream(async_ppo.training_stream):
        async_ppo._run_training(observation, 1, bootstrap_value)
    torch.cuda.synchronize()

    yield async_ppo

    # An implementation that incorrectly accepts a second update may have
    # started a worker; do not leak it into another test.
    if async_ppo.training_thread is not None:
        async_ppo.training_thread.join(timeout=5)
        assert not async_ppo.training_thread.is_alive()
        torch.cuda.synchronize()


def test_completed_update_is_published_at_the_next_inference_boundary(
    ppo_with_unpublished_update: AsyncPPO,
    observation: torch.Tensor,
) -> None:
    """A completed update remains invisible until the next call to act()."""

    _assert_parameters_equal(ppo_with_unpublished_update.inference_policy, 0.0)

    ppo_with_unpublished_update.act(observation)
    torch.cuda.synchronize()

    _assert_parameters_equal(ppo_with_unpublished_update.inference_policy, 1.0)


def test_new_update_is_rejected_until_the_previous_update_is_published(
    ppo_with_unpublished_update: AsyncPPO,
    observation: torch.Tensor,
) -> None:
    """A completed policy update must be published before another update can start."""
    assert not ppo_with_unpublished_update.async_update(observation)


def test_training_metrics_report_rollout_reward_and_losses(
    observation: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async_ppo = AsyncPPO(**PPO_CONFIG, track_metrics=True)  # type: ignore
    async_ppo.training_storage.rewards[0, :, 0] = torch.tensor(
        [2.0, 4.0], device="cuda"
    )
    async_ppo.training_storage.step = 1

    monkeypatch.setattr(async_ppo.learner, "compute_returns", lambda *_args: None)
    monkeypatch.setattr(
        async_ppo.learner,
        "update",
        lambda: {"loss/policy": 1.25, "loss/value": 2.5},
    )

    bootstrap_value = torch.zeros(NUM_ENVS, 1, device="cuda")
    with torch.cuda.stream(async_ppo.training_stream):
        async_ppo._run_training(observation, 1, bootstrap_value)
    torch.cuda.synchronize()

    metrics = async_ppo.drain_metrics()
    assert len(metrics) == 1
    assert metrics[0].trained_samples == (
        NUM_ENVS * PPO_CONFIG["num_transitions_per_env"]
    )
    assert metrics[0].policy_loss == pytest.approx(1.25)
    assert metrics[0].value_loss == pytest.approx(2.5)
    assert metrics[0].mean_total_reward == pytest.approx(3.0)
    assert async_ppo.drain_metrics() == []


def test_policy_publication_is_ordered_across_cuda_streams() -> None:
    """CUDA sanitizer must see an ordered copy-to-inference handoff."""

    sanitizer_scenario = Path(__file__).with_name("async_ppo_cuda_sanitizer.py")

    result = subprocess.run(
        [sys.executable, str(sanitizer_scenario)],
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ | {"TORCH_CUDA_SANITIZER": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, (
        "CUDA sanitizer scenario failed:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
