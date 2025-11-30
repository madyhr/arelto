# python/rl2_env.py
import sys
import os
import numpy as np
import torch
import time
from diagnostics import GameDiagnostics # Import the class we just made

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

import rl2_py

game = rl2_py.Game()

is_initialized = game.initialize()

obs_size = game.get_observation_size()
numpy_buffer = np.zeros(obs_size, dtype=np.float32) 
print(f"{numpy_buffer}")

torch_obs = torch.from_numpy(numpy_buffer)

print("Game initialized!")

dt = 0.001
accumulator = 0.0
current_time = time.time()
# --- Constants ---
TARGET_FPS = 1000.0
DT = 1.0 / TARGET_FPS
MAX_FRAME_TIME = 0.25
MS_PER_FRAME_TARGET = 1000.0 / TARGET_FPS  # ~16.66ms
previous_frame_start = time.perf_counter()
# 1. Start the UI
with GameDiagnostics(target_fps=TARGET_FPS) as monitor:
    
    # 2. RESET THE CLOCK HERE (Critical Fix)
    # This ensures the startup time of 'GameDiagnostics' doesn't break the first frame.
    current_time = time.perf_counter() 

    while game.get_game_state() == 2:
        loop_start = time.perf_counter()
        
        new_time = time.perf_counter()
        frame_time = new_time - current_time
        
        # Spiral of death protection
        if frame_time > 0.25:
            frame_time = 0.25
        
        current_time = new_time
        accumulator += frame_time

        # --- Physics Phase ---
        t0 = time.perf_counter()
        while accumulator >= DT:
            game.step()
            accumulator -= DT
        monitor.record('physics', time.perf_counter() - t0)

        # --- Buffer Phase ---
        t0 = time.perf_counter()
        game.fill_observation_buffer(numpy_buffer)
        monitor.record('buffer', time.perf_counter() - t0)
        
        # --- Render Phase ---
        t0 = time.perf_counter()
        alpha = accumulator / DT
        game.render(alpha)
        monitor.record('render', time.perf_counter() - t0)

        # --- 5. FRAME CAP (The Fix) ---
        # Calculate how long the work took
        work_time = time.perf_counter() - loop_start
        monitor.record('total', work_time)
        
        # Calculate how long we need to sleep to hit 60 FPS
        sleep_needed = DT - work_time
        
        if sleep_needed > 0:
            time.sleep(sleep_needed)


        # 6. Stats Update (Real FPS)
        now = time.perf_counter()
        actual_delta = now - previous_frame_start
        previous_frame_start = now # Update for next loop
        
        if actual_delta > 0:
            monitor.record('fps', 1.0 / actual_delta)
        
        monitor.tick()
