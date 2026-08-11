import torch


def split_and_pad_trajectories(
    input_tensor: torch.Tensor,
    dones: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Splits a time-major rollout at dones and pads each
    trajectory according to the rollout length.

    Args:
        input_tensor: Tensor shaped [time, environments, ...].
        dones: Terminal flags shaped [time, environments, ...].

    Returns:
        Padded fragments shaped [time, trajectories, ...] and a boolean
        mask shaped [time, trajectories].
    """

    dones = dones.clone()
    dones[-1, :] = 1.0
    flat_dones = dones.transpose(0, 1).reshape(-1)
    padded_dones = torch.nn.functional.pad(flat_dones.bool(), (1, 0), value=True)
    done_indices = torch.where(padded_dones)[0]
    traj_lengths = torch.diff(done_indices)

    num_transitions = input_tensor.shape[0]
    num_traj = traj_lengths.numel()
    feature_shape = input_tensor.shape[2:]
    padded_traj = input_tensor.new_zeros(num_transitions, num_traj, *feature_shape)

    range_matrix = torch.arange(num_transitions, device=input_tensor.device).unsqueeze(
        1
    )
    mask = range_matrix < traj_lengths.unsqueeze(0)

    input_flat = input_tensor.transpose(0, 1).flatten(0, 1)
    # Use trajectory-major views so their valid entries follow the same
    # [trajectory, time] ordering as input_flat.
    padded_by_traj = padded_traj.transpose(0, 1)
    valid_steps_by_traj = mask.transpose(0, 1)
    padded_by_traj[valid_steps_by_traj] = input_flat

    return padded_traj, mask


def combine_and_unpad_trajectories(
    padded_traj: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Performs the inverse operation of `split_and_pad_trajectories`."""
    padded_by_traj = padded_traj.transpose(0, 1)
    valid_steps_by_traj = mask.transpose(0, 1)

    valid_steps = padded_by_traj[valid_steps_by_traj]

    num_transitions = padded_traj.shape[0]
    feature_shape = padded_traj.shape[2:]

    traj_by_env = valid_steps.view(-1, num_transitions, *feature_shape)
    traj_by_transition = traj_by_env.transpose(0, 1)

    return traj_by_transition
