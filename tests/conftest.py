# tests/conftest.py
import pytest
import torch


@pytest.fixture
def input_dim():
    return 10


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
def dummy_input(batch_size, input_dim):
    return torch.randn(batch_size, input_dim)
