import tempfile

import pytest
import torch
import torch.nn as nn
from modules.ray_encoder import RayEncoder


@pytest.fixture
def base_config():
    return {
        "num_rays": 64,
        "num_ray_types": 5,
        "output_dim": 32,
    }


@pytest.fixture
def encoder(base_config):
    return RayEncoder(**base_config)


def test_initialization_assertions():
    """Test that __init__ raises an assertion error if list lengths mismatch."""
    with pytest.raises(AssertionError):
        RayEncoder(
            num_rays=10,
            num_ray_types=2,
            output_dim=10,
            out_channels_list=[16, 32],
            kernel_sizes=[3],  # Mismatch length
            strides=[1, 1],
        )


def test_convolutional_ray_encoder(encoder):
    batch_size = 4

    # num_rays (distances) + num_rays (types)
    obs_dim = encoder.num_rays + encoder.num_rays

    # Create dummy observation
    # Distances: random
    # Types: random integers [0, num_ray_types)
    obs = torch.randn(batch_size, obs_dim)

    # Force types to be valid integers
    obs[:, encoder.num_rays :] = torch.randint(
        0, encoder.num_ray_types, (batch_size, encoder.num_rays)
    ).float()

    # Forward pass
    output = encoder(obs)

    # Check shape
    assert output.shape == (batch_size, encoder.output_dim)

    # Check backward pass
    loss = output.sum()
    loss.backward()

    # Check if gradients exist
    for param in encoder.parameters():
        assert param.grad is not None


def test_custom_architecture():
    num_rays = 32
    num_ray_types = 3
    output_dim = 64
    out_channels_list = [8, 16, 32]
    kernel_sizes = [3, 3, 3]
    strides = [1, 2, 1]

    encoder = RayEncoder(
        num_rays=num_rays,
        num_ray_types=num_ray_types,
        output_dim=output_dim,
        out_channels_list=out_channels_list,
        kernel_sizes=kernel_sizes,
        strides=strides,
    )

    obs_dim = num_rays + num_rays
    obs = torch.randn(2, obs_dim)

    # Force types to be valid integers
    obs[:, num_rays:] = torch.randint(0, num_ray_types, (2, num_rays)).float()

    output = encoder(obs)
    assert output.shape == (2, output_dim)


def test_input_dimension_mismatch(encoder, base_config):
    """Test behavior when input observation size does not match expected size."""
    # Expected obs_dim = num_rays (dist) + num_rays (types) = 128
    # Pass a tensor that is too small
    wrong_obs = torch.randn(4, base_config["num_rays"])  # Missing types part

    # Depending on implementation, this usually crashes at the slicing or embedding step.
    # We just want to ensure it fails and doesn't silently produce garbage.
    with pytest.raises(Exception):  # Catching general runtime/index errors
        encoder(wrong_obs)


def test_out_of_bounds_ray_types(encoder, base_config):
    """Test that passing ray types >= num_ray_types triggers an index error."""
    num_rays = base_config["num_rays"]
    obs = torch.randn(2, num_rays * 2)

    # Set a type index that is out of bounds (e.g., 5 if num_types is 5, since 0-4 are valid)
    obs[:, num_rays:] = base_config["num_ray_types"]

    with pytest.raises(IndexError):
        encoder(obs)


def test_batch_independence(encoder, base_config):
    """
    Crucial for CNNs/Padding: Ensure modifying batch element 1 does not change output of batch element 0.
    """
    num_rays = base_config["num_rays"]
    obs_dim = num_rays * 2

    # Create two identical inputs
    input_a = torch.randn(1, obs_dim)
    # Ensure types are valid
    input_a[:, num_rays:] = torch.zeros(1, num_rays)

    input_b = input_a.clone()

    # Construct batch
    batch_input = torch.cat([input_a, input_b], dim=0)

    # Get initial output
    encoder.eval()
    with torch.no_grad():
        out_initial = encoder(batch_input)

    # Modify the second element in the batch significantly
    batch_input[1, :num_rays] += 5.0

    with torch.no_grad():
        out_modified = encoder(batch_input)

    # The output for index 0 should be EXACTLY the same in both passes
    assert torch.allclose(
        out_initial[0], out_modified[0], atol=1e-6
    ), "Batch element 0 output changed when element 1 was modified! Check padding/batchnorm logic."

    # The output for index 1 should be different
    assert not torch.allclose(
        out_initial[1], out_modified[1]
    ), "Batch element 1 output did not change despite input changing."


# --- Edge Case Tests ---


@pytest.mark.parametrize("num_rays", [1, 2, 3])
def test_small_ray_counts(num_rays):
    """Test if the CNN arithmetic holds up for very small numbers of rays."""
    # This often breaks if kernel sizes > num_rays without proper padding
    encoder = RayEncoder(
        num_rays=num_rays,
        num_ray_types=2,
        output_dim=16,
        kernel_sizes=[3, 3],  # Kernel larger than input width if rays=1,2
        out_channels_list=[4, 4],
        strides=[1, 1],
    )

    obs = torch.randn(1, num_rays * 2)
    obs[:, num_rays:] = 0  # valid types

    try:
        out = encoder(obs)
        assert out.shape == (1, 16)
    except RuntimeError as e:
        pytest.fail(f"Failed with num_rays={num_rays}: {e}")


def test_circular_padding_continuity(base_config):
    """
    Verify circular padding logic roughly works.
    If we roll the input (shift rays), the output feature map (before FC) should roughly roll.
    Note: FC layer breaks translation invariance, so we check the conv_net output directly.
    """
    num_rays = base_config["num_rays"]
    encoder = RayEncoder(**base_config)

    # Create input
    dists = torch.randn(1, num_rays)
    types = torch.zeros(1, num_rays).long()

    # Function to run just the conv part (mimicking forward)
    def get_conv_features(d, t):
        emb = encoder.type_embed(t)
        d_uns = d.unsqueeze(-1)
        combined = torch.cat([d_uns, emb], dim=-1).permute(0, 2, 1)
        return encoder.conv_net(combined)

    # Original features
    feat_orig = get_conv_features(dists, types)

    # Shift input by 1 (Roll rays)
    dists_rolled = torch.roll(dists, shifts=1, dims=1)
    types_rolled = torch.roll(types, shifts=1, dims=1)

    feat_rolled = get_conv_features(dists_rolled, types_rolled)

    # The output features should also be rolled by 1 (due to stride 1 and circular padding)
    # Note: This assumes strides are all 1.
    feat_expected_roll = torch.roll(feat_orig, shifts=1, dims=2)

    assert torch.allclose(
        feat_rolled, feat_expected_roll, atol=1e-5
    ), "Circular padding did not result in translation equivariance."


# --- Training Integration Test ---


def test_overfitting_sanity_check(encoder, base_config):
    """
    Verify that the model can overfit a single batch.
    This confirms gradients flow through: Embedding -> Concat -> Conv -> Flatten -> Linear.
    """
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    num_rays = base_config["num_rays"]
    obs = torch.randn(2, num_rays * 2)
    obs[:, num_rays:] = torch.randint(
        0, base_config["num_ray_types"], (2, num_rays)
    ).float()

    target = torch.randn(2, base_config["output_dim"])

    initial_loss = criterion(encoder(obs), target).item()

    # Train for a few steps
    for _ in range(20):
        optimizer.zero_grad()
        output = encoder(obs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    final_loss = criterion(encoder(obs), target).item()

    # Loss should decrease significantly
    assert (
        final_loss < initial_loss * 0.1
    ), f"Model failed to overfit: {initial_loss} -> {final_loss}"


# --- System Tests ---


import os


def test_model_save_load(encoder):
    """Test that the model can be pickled and unpickled (saved/loaded)."""
    # Use TemporaryDirectory to avoid file locking issues on Windows
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "model.pth")
        torch.save(encoder.state_dict(), tmp_path)

        # specific reload
        new_encoder = RayEncoder(
            num_rays=encoder.num_rays,
            num_ray_types=encoder.num_ray_types,
            output_dim=encoder.output_dim,
        )
        new_encoder.load_state_dict(torch.load(tmp_path))

        # Verify weights match
        for p1, p2 in zip(encoder.parameters(), new_encoder.parameters()):
            assert torch.equal(p1, p2)
