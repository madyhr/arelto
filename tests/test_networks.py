import pytest
import torch
import torch.nn as nn

from rl.modules.actor_critic import ActorCritic
from rl.networks.mlp import MLP, GaussianActor, ValueNetwork


@pytest.fixture
def input_dim():
    return 10


@pytest.fixture
def output_dim():
    return 4


@pytest.fixture
def hidden_size():
    return [32, 32]


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def dummy_input(batch_size, input_dim):
    return torch.randn(batch_size, input_dim)


def test_mlp_initialization(input_dim, output_dim, hidden_size):
    model = MLP(input_dim, hidden_size, output_dim)

    assert isinstance(model.network, nn.Sequential)

    assert len(model.network) == 5

    assert isinstance(model.network[0], nn.Linear)
    assert isinstance(model.network[1], nn.Tanh)
    assert isinstance(model.network[-1], nn.Linear)


def test_mlp_forward_shape(dummy_input, input_dim, output_dim, hidden_size, batch_size):
    model = MLP(input_dim, hidden_size, output_dim)
    output = model(dummy_input)

    assert output.shape == (batch_size, output_dim)
    assert not torch.isnan(output).any()


def test_mlp_invalid_arguments():
    with pytest.raises(ValueError, match="Hidden size cannot be empty"):
        MLP(10, [], 5)

    with pytest.raises(ValueError, match="Input dim must be a positive"):
        MLP(0, [10], 5)

    with pytest.raises(ValueError, match="Hidden dims must be positive"):
        MLP(10, [10, -5], 5)


def test_gaussian_actor_forward_shape(
    dummy_input, input_dim, output_dim, hidden_size, batch_size
):
    actor = GaussianActor(input_dim, hidden_size, output_dim)

    mean, std = actor(dummy_input)

    assert mean.shape == (batch_size, output_dim)
    assert std.shape == (batch_size, output_dim)

    assert (std > 0).all()


def test_gaussian_actor_get_action(
    dummy_input, input_dim, output_dim, hidden_size, batch_size
):
    actor = GaussianActor(input_dim, hidden_size, output_dim)

    action, log_prob = actor.get_action(dummy_input)

    assert action.shape == (batch_size, output_dim)

    assert log_prob.shape == (batch_size,)

    loss = action.sum() + log_prob.sum()
    loss.backward()

    assert actor.mean_network.network[0].weight.grad is not None
    assert actor.log_std.grad is not None


def test_value_network_shape(dummy_input, input_dim, hidden_size, batch_size):
    critic = ValueNetwork(input_dim, hidden_size)
    value = critic(dummy_input)

    assert value.shape == (batch_size, 1)


def test_actor_critic_forward_flow(
    dummy_input, input_dim, output_dim, hidden_size, batch_size
):
    ac = ActorCritic(input_dim, hidden_size, output_dim, activation_func_class=nn.Tanh)

    action, log_prob, value = ac(dummy_input)

    assert action.shape == (batch_size, output_dim)
    assert log_prob.shape == (batch_size,)
    assert value.shape == (batch_size, 1)


def test_actor_critic_get_value(
    dummy_input, input_dim, output_dim, hidden_size, batch_size
):
    ac = ActorCritic(input_dim, hidden_size, output_dim, activation_func_class=nn.Tanh)
    value = ac.get_value(dummy_input)

    assert value.shape == (batch_size, 1)


def test_actor_critic_gradient_flow(dummy_input, input_dim, output_dim, hidden_size):
    ac = ActorCritic(input_dim, hidden_size, output_dim, activation_func_class=nn.Tanh)

    action, log_prob, value = ac(dummy_input)

    loss = log_prob.sum() + value.sum()

    ac.zero_grad()
    loss.backward()

    actor_first_layer = ac.actor.mean_network.network[0]
    assert actor_first_layer.weight.grad is not None
    assert torch.norm(actor_first_layer.weight.grad) > 0

    critic_first_layer = ac.critic.network.network[0]
    assert critic_first_layer.weight.grad is not None
    assert torch.norm(critic_first_layer.weight.grad) > 0
