import torch

from rl.utils.trajectory_padding import (
    split_and_pad_trajectories,
    combine_and_unpad_trajectories,
)

observations = torch.tensor(
    [
        [[0.0], [10.0]],
        [[1.0], [11.0]],
        [[2.0], [12.0]],
        [[3.0], [13.0]],
    ]
)

dones = torch.tensor(
    [
        [[False], [False]],
        [[True], [False]],
        [[False], [False]],
        [[False], [False]],
    ]
)

expected_padded = torch.tensor(
    [
        [[0.0], [2.0], [10.0]],
        [[1.0], [3.0], [11.0]],
        [[0.0], [0.0], [12.0]],
        [[0.0], [0.0], [13.0]],
    ]
)

expected_mask = torch.tensor(
    [
        [True, True, True],
        [True, True, True],
        [False, False, True],
        [False, False, True],
    ]
)


def test_split_and_pad_trajectories():
    padded, mask = split_and_pad_trajectories(observations, dones)

    assert torch.equal(padded, expected_padded)
    assert torch.equal(mask, expected_mask)


def test_combine_and_unpad_trajectories():
    padded, mask = split_and_pad_trajectories(observations, dones)
    restored = combine_and_unpad_trajectories(padded, mask)

    assert restored.shape == observations.shape
    assert torch.equal(restored, observations)


def test_combine_and_unpad_trajectories_multi_dim():
    multi_dim_observations = torch.arange(5 * 3 * 2).reshape(5, 3, 2).float()

    multi_dim_dones = torch.zeros(5, 3, 1, dtype=torch.bool)
    multi_dim_dones[1, 0] = True
    multi_dim_dones[2, 1] = True
    multi_dim_dones[0, 2] = True
    multi_dim_dones[3, 2] = True

    padded, mask = split_and_pad_trajectories(
        multi_dim_observations,
        multi_dim_dones,
    )
    restored = combine_and_unpad_trajectories(padded, mask)

    assert restored.shape == multi_dim_observations.shape
    assert torch.equal(restored, multi_dim_observations)
