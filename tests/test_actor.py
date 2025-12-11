import torch

from rl.modules.actor import DiscreteActor, GaussianActor, MultiDiscreteActor


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

    assert actor.network.network[0].weight.grad is not None
    assert actor.log_std.grad is not None


def test_discrete_actor_forward_shape(
    dummy_input, input_dim, output_dim, hidden_size, batch_size
):
    actor = DiscreteActor(input_dim, hidden_size, output_dim)
    logits = actor(dummy_input)

    assert logits.shape == (batch_size, output_dim)
    assert not torch.isnan(logits).any()


def test_discrete_actor_get_action(
    dummy_input, input_dim, output_dim, hidden_size, batch_size
):
    actor = DiscreteActor(input_dim, hidden_size, output_dim)
    action, log_prob = actor.get_action(dummy_input)

    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)

    assert not action.is_floating_point()
    assert (action >= 0).all() and (action < output_dim).all()


def test_multi_discrete_actor_forward_shape(
    dummy_input, input_dim, multi_discrete_output_dims, hidden_size, batch_size
):
    actor = MultiDiscreteActor(input_dim, hidden_size, multi_discrete_output_dims)
    logits_list = actor(dummy_input)

    assert isinstance(logits_list, list)
    assert len(logits_list) == len(multi_discrete_output_dims)

    for logits, dim in zip(logits_list, multi_discrete_output_dims):
        assert logits.shape == (batch_size, dim)
        assert not torch.isnan(logits).any()


def test_multi_discrete_actor_get_action(
    dummy_input, input_dim, multi_discrete_output_dims, hidden_size, batch_size
):
    actor = MultiDiscreteActor(input_dim, hidden_size, multi_discrete_output_dims)
    action, log_prob = actor.get_action(dummy_input)

    assert action.shape == (batch_size, len(multi_discrete_output_dims))
    assert log_prob.shape == (batch_size,)

    for i, dim in enumerate(multi_discrete_output_dims):
        assert (action[:, i] >= 0).all()
        assert (action[:, i] < dim).all()
