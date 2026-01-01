import pytest
import torch
import torch.nn as nn
from modules.actor import MultiDiscreteActor
from modules.actor_critic import ActorCritic
from modules.critic import ValueCritic


@pytest.mark.parametrize("actor_class", [MultiDiscreteActor])
def test_actor_critic_forward_flow(
    actor_class,
    dummy_input,
    input_dim,
    output_dim,
    multi_discrete_output_dims,
    hidden_size,
    batch_size,
):
    if actor_class == MultiDiscreteActor:
        current_output_dim = multi_discrete_output_dims
    else:
        current_output_dim = output_dim

    ac = ActorCritic(
        actor_class=actor_class,
        critic_class=ValueCritic,
        input_dim=input_dim,
        hidden_size=hidden_size,
        output_dim=current_output_dim,
        activation_func_class=nn.Tanh,
    )

    action, log_prob, entropy, value = ac(dummy_input)

    if actor_class == MultiDiscreteActor:
        assert action.shape == (batch_size, len(current_output_dim))

    assert log_prob.shape == (batch_size, 1)
    assert entropy.shape == (batch_size, 1)
    assert value.shape == (batch_size, 1)


@pytest.mark.parametrize("actor_class", [MultiDiscreteActor])
def test_actor_critic_gradient_flow(
    actor_class,
    dummy_input,
    input_dim,
    output_dim,
    multi_discrete_output_dims,
    hidden_size,
):
    if actor_class == MultiDiscreteActor:
        current_output_dim = multi_discrete_output_dims
    else:
        current_output_dim = output_dim

    ac = ActorCritic(
        actor_class=actor_class,
        critic_class=ValueCritic,
        input_dim=input_dim,
        hidden_size=hidden_size,
        output_dim=current_output_dim,
        activation_func_class=nn.Tanh,
    )

    action, log_prob, entropy, value = ac(dummy_input)

    action_loss = action.sum()

    loss = log_prob.sum() + entropy.sum() + value.sum() + action_loss

    ac.zero_grad()
    loss.backward()

    actor_first_layer = ac.actor.network.network[0]
    assert actor_first_layer.weight.grad is not None
    assert torch.norm(actor_first_layer.weight.grad) > 0

    critic_first_layer = ac.critic.network.network[0]
    assert critic_first_layer.weight.grad is not None
    assert torch.norm(critic_first_layer.weight.grad) > 0
