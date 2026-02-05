# tests/conftest.py
import pytest
import torch
from modules.ray_encoder import RayEncoder


@pytest.fixture
def input_dim():
    return 20


@pytest.fixture
def output_dim():
    return 4


@pytest.fixture
def multi_discrete_output_dims():
    return [2, 3]


@pytest.fixture
def hidden_size():
    return [32, 32]


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def num_ray_types():
    return 3


@pytest.fixture
def dummy_input(batch_size, input_dim, num_ray_types):
    distances = torch.randn(batch_size, input_dim)
    # Types are categorical, but their expected type is float.
    types = torch.randint(0, num_ray_types, (batch_size, input_dim)).float()
    return torch.cat([distances, types], dim=1)


@pytest.fixture
def dummy_encoder(input_dim, output_dim, num_ray_types):
    return RayEncoder(input_dim, num_ray_types, output_dim)
