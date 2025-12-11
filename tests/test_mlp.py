import pytest
import torch
import torch.nn as nn

from rl.networks.mlp import MLP


def test_mlp_initialization(input_dim, output_dim, hidden_size):
    model = MLP(input_dim, hidden_size, output_dim)
    assert isinstance(model.network, nn.Sequential)
    # 2 layers * 2 (linear+act) + 1 final linear = 5
    assert len(model.network) == 5
    assert isinstance(model.network[0], nn.Linear)
    assert isinstance(model.network[1], model.activation_func_class)
    assert isinstance(model.network[-1], nn.Linear)


def test_mlp_forward_shape(dummy_input, input_dim, output_dim, hidden_size, batch_size):
    model = MLP(input_dim, hidden_size, output_dim)
    output = model(dummy_input)
    assert output.shape == (batch_size, output_dim)
    assert not torch.isnan(output).any()


def test_mlp_invalid_args():
    with pytest.raises(ValueError, match="Input dim must be a positive integer"):
        MLP(input_dim=0, hidden_size=[32], output_dim=4)

    with pytest.raises(ValueError, match="Hidden size cannot be empty"):
        MLP(input_dim=10, hidden_size=[], output_dim=4)
