import torch
from modules.actor import MultiDiscreteActor


def test_multi_discrete_actor_forward_shape(
    dummy_input, input_dim, multi_discrete_output_dims, hidden_size, batch_size
):
    actor = MultiDiscreteActor(input_dim, hidden_size, multi_discrete_output_dims)
    logits_tensor_tuple = actor(dummy_input)

    assert isinstance(logits_tensor_tuple, tuple)
    assert len(logits_tensor_tuple) == len(multi_discrete_output_dims)

    for logits, dim in zip(logits_tensor_tuple, multi_discrete_output_dims):
        assert logits.shape == (batch_size, dim)
        assert not torch.isnan(logits).any()


def test_multi_discrete_actor_get_action(
    dummy_input, input_dim, multi_discrete_output_dims, hidden_size, batch_size
):
    actor = MultiDiscreteActor(input_dim, hidden_size, multi_discrete_output_dims)
    action, log_prob, entropy = actor.get_action(dummy_input)

    assert action.shape == (batch_size, len(multi_discrete_output_dims))
    assert log_prob.shape == (batch_size, 1)
    assert entropy.shape == (batch_size, 1)

    for i, dim in enumerate(multi_discrete_output_dims):
        assert (action[:, i] >= 0).all()
        assert (action[:, i] < dim).all()
