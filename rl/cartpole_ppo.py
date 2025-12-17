from collections import deque

import gymnasium as gym
import numpy as np
import torch
from algorithms.ppo import PPO


def test_on_cartpole_v2():
    # 1. Hyperparameters
    NUM_ENVS = 8
    TRANSITIONS_PER_ENV = 128
    TOTAL_TIMESTEPS = 1000000  # Increased slightly to ensure convergence
    INPUT_DIM = 4
    HIDDEN_SIZE = [64]
    OUTPUT_DIM = [2]
    DEVICE = "cpu"

    # 2. Setup Vectorized Environment
    envs = gym.make_vec("CartPole-v1", num_envs=NUM_ENVS, vectorization_mode="sync")

    # 3. Initialize PPO
    ppo = PPO(
        num_envs=NUM_ENVS,
        num_transitions_per_env=TRANSITIONS_PER_ENV,
        input_dim=INPUT_DIM,
        hidden_size=HIDDEN_SIZE,
        output_dim=OUTPUT_DIM,
        device=DEVICE,
        learning_rate=2.5e-4,  # Standard PPO learning rate
        gamma=0.99,
        entropy_loss_coef=0.01,
    )

    print("Starting Training on CartPole-v1...")

    # Tracking variables
    global_step = 0
    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(DEVICE)

    # Track rewards for each environment
    current_episode_rewards = np.zeros(NUM_ENVS)
    # Store completed episode rewards to calculate average
    completed_episode_rewards = deque(maxlen=20)

    while global_step < TOTAL_TIMESTEPS:

        # --- Collection Phase ---
        for _ in range(TRANSITIONS_PER_ENV):
            action = ppo.act(obs)
            action_np = action.cpu().numpy().squeeze(-1)

            next_obs, rewards, terminated, truncated, infos = envs.step(action_np)

            # Accumulate rewards
            current_episode_rewards += rewards

            # Identify finished episodes
            dones = terminated | truncated

            # If an env is done, record the final reward and reset its tracker
            for i, done in enumerate(dones):
                if done:
                    completed_episode_rewards.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0

            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(DEVICE)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(DEVICE)

            ppo.process_env_step(rewards_tensor, dones_tensor)
            obs = next_obs_tensor
            global_step += NUM_ENVS

        # --- Update Phase ---
        ppo.compute_returns(obs)
        train_metrics = ppo.update()

        # --- Logging ---
        if len(completed_episode_rewards) > 0:
            avg_reward = np.mean(completed_episode_rewards)
            print(
                f"Step {global_step}: Avg Episode Reward: {avg_reward:.1f} / 500.0 | Value Loss: {train_metrics['loss/value']:.3f}"
            )
        else:
            print(f"Step {global_step}: collecting initial episodes...")

    envs.close()


if __name__ == "__main__":
    test_on_cartpole_v2()
