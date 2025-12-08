# python/run_game.py
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))

import rl2_py
from test_model import TestModel

# This comes from the order of the game states as defined in the C++ source code.
game_state = {
    "in_start_screen": 0,
    "in_main_menu": 1,
    "is_running": 2,
    "is_gameover": 3,
    "in_shutdown": 4,
}

game = rl2_py.Game()
model = TestModel(num_envs=game.num_enemies, action_dim=game.get_action_size())

is_initialized = game.initialize()

obs_size = game.get_observation_size()
numpy_obs_buffer = np.zeros(obs_size, dtype=np.float32)
action_size = game.get_action_size()
numpy_action_buffer = np.zeros(action_size, dtype=np.float32)
print(f"{numpy_obs_buffer}")

torch_obs = torch.from_numpy(numpy_obs_buffer)

print("Game initialized!")

physics_dt = 0.001
step_dt = 0.02
current_time = time.time()

next_tick = time.perf_counter()
while game.get_game_state() != game_state["in_shutdown"]:
    game.process_input()

    if game.get_game_state() == game_state["is_running"]:
        game.fill_observation_buffer(numpy_obs_buffer)
        action = model(torch_obs)
        game.apply_action(action.detach().numpy())
        game.step(step_dt)

    game.render(1.0)

    next_tick += step_dt

    # Calculate how long we need to sleep to hit 60 FPS
    sleep_needed = next_tick - time.perf_counter()

    if sleep_needed > 0:
        time.sleep(sleep_needed)
    else:
        next_tick = time.perf_counter()
