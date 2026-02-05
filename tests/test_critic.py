from modules.critic import ValueCritic


def test_value_critic_shape(dummy_input, input_dim, hidden_size, batch_size, dummy_encoder):
    critic = ValueCritic(input_dim, hidden_size, encoder=dummy_encoder)
    value = critic(dummy_input)
    assert value.shape == (batch_size, 1)
