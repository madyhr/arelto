# python/rl2_env.py
import sys
import os
import numpy as np
import torch
import timeit
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

import rl2_py

game = rl2_py.Game()

is_initialized = game.initialize()

obs_size = game.get_observation_size()
numpy_buffer = np.zeros(obs_size, dtype=np.float32) 
print(f"{numpy_buffer}")

torch_obs = torch.from_numpy(numpy_buffer)

print("Game initialized!")

start = timeit.timeit()
dt = 0.001
accumulator = 0.0
current_time = time.time()

while game.get_game_state() == 2:
    new_time = time.time()
    frame_time = new_time - current_time
    # Cap frame_time to prevent spiral of death
    if frame_time > 0.1:
        frame_time = 0.1
    
    current_time = new_time
    accumulator += frame_time

    while accumulator >= dt:
        game.step() 
        accumulator -= dt

    game.fill_observation_buffer(numpy_buffer)
    
    alpha = accumulator / dt
    game.render(alpha)
    
game.shutdown()

