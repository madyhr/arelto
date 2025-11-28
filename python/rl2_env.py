# python/rl2_env.py
import sys
import os
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

import rl2_py

game = rl2_py.Game()

is_initialized = game.initialize()

obs_size = game.get_observation_size()
# Having dtype np.float32 is important as we need to match the C++ type (float)
numpy_buffer = np.zeros(obs_size, dtype=np.float32) 
print(f"{numpy_buffer}")

# Using torch.from_numpy ensures that it views the same memory, 
# so we can write to this buffer directly.
torch_obs = torch.from_numpy(numpy_buffer)

if is_initialized:
    while game.get_game_state() == 2:
        game.run()

        game.fill_observation_buffer(numpy_buffer)

