# python/run_game.py
import os
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))

import rl2_py

from rl2_gym_env import RL2Env
from test_model import TestModel

# This comes from the order of the game states as defined in the C++ source code.
game_state = {
    "in_start_screen": 0,
    "in_main_menu": 1,
    "is_running": 2,
    "is_gameover": 3,
    "in_shutdown": 4,
}


def run_game():

    env = RL2Env()
    model = TestModel(num_envs=env._num_envs, action_dim=env.game.get_action_size())

    is_initialized = env.game.initialize()

    if not is_initialized:
        print("Game was not initialized properly. Exiting...")
        return

    current_time = time.time()

    counter = 0
    next_tick = time.perf_counter()

    obs, _ = env.reset()

    while env.game.get_game_state() != game_state["in_shutdown"]:
        env.game.process_input()

        action = model(obs)
        obs, reward, _, terminated, truncated = env.step(action)

        dones = terminated | truncated

        env.game.render(1.0)

        next_tick += env._step_dt

        if torch.any(dones) == 1:
            print(f"Terminated or truncated envs detected at count: {counter}")

        if torch.any(terminated) == 1:
            print(f"Terminated envs detected at count: {counter}")

        if torch.any(truncated) == 1:
            print(f"Truncated envs detected at count: {counter}")

        if counter % 50 == 0:
            print(f"Rew buffer: {reward}")

        sleep_needed = next_tick - time.perf_counter()

        if sleep_needed > 0:
            time.sleep(sleep_needed)
        else:
            next_tick = time.perf_counter()

        counter += 1


def main():
    run_game()


if __name__ == "__main__":
    main()
