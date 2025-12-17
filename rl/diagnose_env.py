import time

import numpy as np
import torch
from rl2_env import RL2Env

# Copied from rl/run_game.py
game_state = {
    "in_start_screen": 0,
    "in_main_menu": 1,
    "is_running": 2,
    "is_gameover": 3,
    "in_shutdown": 4,
}


def diagnose_custom_env():
    # 1. Initialize Environment
    env = RL2Env()
    print(f"\n=== Diagnostics for RL2Env (Baseline Mode) ===")

    # 2. explicit Game Initialization (from run_game.py)
    is_initialized = env.game.initialize()
    if not is_initialized:
        print("CRITICAL: Game was not initialized properly. Exiting...")
        return

    print(f"Num Envs: {env.num_envs}")

    # 3. Setup Stats Collection
    N_STEPS = 1000
    print(f"Running {N_STEPS} steps (or until shutdown) to gather stats...")

    obs_history = []
    rew_history = []

    # Based on run_game.py: output_dim=[3, 3] implies 2 discrete actions with 3 options each.
    # We will generate random actions in this range.
    # Adjust this if your actual action space is different!
    ACTION_DIM_COUNT = 2
    ACTION_DIM_SIZE = 3

    counter = 0
    next_tick = time.perf_counter()

    # Initial Reset
    obs, _ = env.reset()

    try:
        while (
            counter < N_STEPS and env.game.get_game_state() != game_state["in_shutdown"]
        ):

            # A. Process Input (Crucial for OS event handling)
            env.game.process_input()

            # B. Generate Random Action matching 'output_dim=[3, 3]' structure
            # Shape: (Num_Envs, 2) with values 0, 1, or 2
            action = torch.randint(
                0, ACTION_DIM_SIZE, (env.num_envs, ACTION_DIM_COUNT)
            ).float()

            # C. Step
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # D. Record Data
            obs_history.append(next_obs.numpy())
            rew_history.append(reward.numpy())

            # E. Render (Keep this to ensure visual pipeline works, or comment out for speed)
            env.game.render(1.0)

            # F. Time/Sleep Management (from run_game.py)
            next_tick += env.step_dt
            sleep_needed = next_tick - time.perf_counter()
            if sleep_needed > 0:
                time.sleep(sleep_needed)
            else:
                next_tick = time.perf_counter()

            obs = next_obs
            counter += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")

    # 4. Analyze Data
    if counter == 0:
        print("No steps collected.")
        return

    obs_data = np.concatenate(obs_history, axis=0)  # (Total Steps * Envs, Obs Dim)
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

    # 5. Heuristic Checks
    print(f"\n=== Suggestions ===")

    # Check Observation Magnitude
    max_obs = np.max(np.abs(obs_data))
    if max_obs > 100:
        print(f"CRITICAL: Observations contain large values (Max: {max_obs:.2f}).")
        print("          Neural networks struggle with inputs > 1.0 or < -1.0.")
        print("          ACTION: Normalize observations in C++ or Python wrapper.")
    elif max_obs > 10:
        print(f"WARNING:  Observations are somewhat large (Max: {max_obs:.2f}).")
    else:
        print("OK:       Observation scaling looks safe.")

    # Check Reward Magnitude
    max_rew = np.max(np.abs(rew_data))
    if max_rew > 10:
        print(f"CRITICAL: Rewards are large (Max: {max_rew:.2f}).")
        print("          PPO is unstable with large rewards.")
        print(
            "          ACTION: Scale rewards (e.g. * 0.01) so they are roughly -1 to 1."
        )
    elif np.std(rew_data) < 1e-6:
        print("WARNING:  Rewards are constant/zero. Agent cannot learn.")
    else:
        print("OK:       Reward scaling looks reasonable.")


if __name__ == "__main__":
    diagnose_custom_env()
