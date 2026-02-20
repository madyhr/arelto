import pytest
from rl.algorithms.ppo import PPO


@pytest.fixture
def mock_env_config():
    return {
        "num_envs": 2,
        "input_dim": 288,  # (4 blocks * 72 rays * 1 history)
        "num_rays": 72,
        "ray_history_length": 1,
        "output_dim": [2],
        "device": "cpu",
    }


def test_ppo_valid_initialization(mock_env_config):
    """
    Test that when input_dim mathematically matches the parameters
    in RayEncoder (4 * num_rays * history_length), PPO initializes without error.
    """
    ppo = PPO(
        input_dim=mock_env_config["input_dim"],
        output_dim=mock_env_config["output_dim"],
        num_envs=mock_env_config["num_envs"],
        num_rays=mock_env_config["num_rays"],
        ray_history_length=mock_env_config["ray_history_length"],
        device=mock_env_config["device"],
    )

    # Assert successful configuration
    assert ppo.input_dim == 288
    assert ppo.encoder.num_rays == 72


def test_ppo_invalid_shape_mismatch(mock_env_config):
    """
    Test that supplying an obs_size (input_dim) that conflicts with
    the RayEncoder configuration intentionally raises a ValueError.
    """
    invalid_input_dim = 1152

    with pytest.raises(ValueError, match="Shape mismatch: Environment obs_size"):
        PPO(
            input_dim=invalid_input_dim,
            output_dim=mock_env_config["output_dim"],
            num_envs=mock_env_config["num_envs"],
            num_rays=mock_env_config["num_rays"],
            ray_history_length=mock_env_config["ray_history_length"],
            device=mock_env_config["device"],
        )
