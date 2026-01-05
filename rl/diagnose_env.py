import time

import numpy as np
import torch
from rl2_env import RL2Env

game_state = {
    "in_start_screen": 0,
    "in_main_menu": 1,
    "is_running": 2,
    "is_gameover": 3,
    "in_shutdown": 4,
}


def diagnose_custom_env():
    env = RL2Env()
    print(f"\n=== Diagnostics for RL2Env (Baseline Mode) ===")

    is_initialized = env.game.initialize()
    if not is_initialized:
        print("CRITICAL: Game was not initialized properly. Exiting...")
        return

    print(f"Num Envs: {env.num_envs}")

    N_STEPS = 10000
    print(f"Running {N_STEPS} steps (or until shutdown) to gather stats...")

    obs_history = []
    rew_history = []

    ACTION_DIM_COUNT = 2
    ACTION_DIM_SIZE = 3

    counter = 0
    next_tick = time.perf_counter()

    obs, _ = env.reset()

    try:
        while (
            counter < N_STEPS and env.game.get_game_state() != game_state["in_shutdown"]
        ):

            env.game.process_input()

            action = torch.randint(
                0, ACTION_DIM_SIZE, (env.num_envs, ACTION_DIM_COUNT)
            ).float()

            next_obs, reward, terminated, truncated, _ = env.step(action)

            obs_history.append(next_obs.numpy())
            rew_history.append(reward.numpy())

            env.game.render(1.0)

            obs = next_obs
            counter += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")

    if counter == 0:
        print("No steps collected.")
        return

    obs_data = np.concatenate(obs_history, axis=0)
    rew_data = np.concatenate(rew_history, axis=0)

    print(f"\n=== Statistics over {counter} steps ===")

    print(f"\n--- Observation Stats (Input Normalization Check) ---")
    print(f"{'Dim':<5} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}")
    print("-" * 55)
    for i in range(obs_data.shape[1]):
        col = obs_data[:, i]
        print(
            f"{i:<5} | {col.mean():<10.3f} | {col.std():<10.3f} | {col.min():<10.3f} | {col.max():<10.3f}"
        )

    print(f"\n--- Reward Stats (Reward Scaling Check) ---")
    print(f"Mean: {rew_data.mean():.4f}")
    print(f"Std:  {rew_data.std():.4f}")
    print(f"Min:  {rew_data.min():.4f}")
    print(f"Max:  {rew_data.max():.4f}")


if __name__ == "__main__":
    diagnose_custom_env()
