import argparse
import datetime
import os
import time

import torch

from rl.algorithms.async_ppo import AsyncPPO
from rl.rl2_env import RL2Env

# TODO: Refactor this into a pybinding that gets a dict directly from RL2Env.
# Map C++ game states
game_state = {
    "in_start_screen": 0,
    "in_main_menu": 1,
    "is_running": 2,
    "is_gameover": 3,
    "in_shutdown": 4,
    "is_paused": 5,
    "is_level_up": 6,
}

TARGET_FPS = 60
TARGET_FRAME_TIME = 1 / TARGET_FPS


def start_game(args):
    checkpoint_path: str = args.load_checkpoint
    device: str = "cuda"
    env = RL2Env(step_dt=0.02)
    num_envs = env.num_envs
    obs_size = env.game.get_observation_size()
    num_rays = env.game.get_enemy_num_rays()
    ray_history_length = env.game.get_enemy_ray_history_length()

    def create_agent() -> AsyncPPO:
        return AsyncPPO(
            input_dim=obs_size,
            num_envs=num_envs,
            num_rays=num_rays,
            ray_history_length=ray_history_length,
            device=device,
        )

    ppo = create_agent()

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        ppo.learner.policy.load_state_dict(state_dict)
        ppo.inference_policy.load_state_dict(state_dict)

    elif checkpoint_path:
        print(
            f"Checkpoint path {checkpoint_path} provided but file not found. Starting from scratch."
        )

    if not env.game.initialize():
        return

    env.game.set_game_state(game_state["in_start_screen"])

    # We need to get the initial obs to infer first action.
    obs, _ = env.reset()

    # Main Game Loop
    while env.game.get_game_state() != game_state["in_shutdown"]:

        if env.game.get_game_state() == game_state["in_start_screen"]:
            env.game.process_input()
            env.game.render(1.0)
            if env.game.get_game_state() == game_state["is_running"]:
                print("Transitioning to Training Loop...")
                # As we might have transitions stored in the rollout storage
                # when we enter the start screen, we clear the storage screen
                # to not go out of bounds when we begin filling it again.
                ppo.inference_storage.clear()

        elif (
            env.game.get_game_state() == game_state["is_running"]
            or env.game.get_game_state() == game_state["is_paused"]
            or env.game.get_game_state() == game_state["is_level_up"]
        ):
            # We keep track of the number of steps to handle pauses correctly.
            step = 0
            while step < ppo.num_transitions_per_env:
                frame_start = time.perf_counter()
                env.game.process_input()
                state = env.game.get_game_state()
                if state == game_state["in_shutdown"]:
                    break
                if state == game_state["in_start_screen"]:
                    print("Returned to Menu")
                    print("Resetting policy parameters...")
                    ppo = create_agent()
                    break
                if (
                    state == game_state["is_paused"]
                    or state == game_state["is_level_up"]
                ):
                    env.game.render(1.0)
                    continue

                with torch.inference_mode():
                    env.game.process_input()

                    if env.game.get_game_state() != game_state["is_running"]:
                        continue

                    action = ppo.act(obs.to(device))
                    obs, reward, terminated, truncated, _ = env.step(action)
                    dones = terminated | truncated
                    ppo.process_env_step(reward.to(device), dones.to(device))
                    env.game.render(1.0)

                step += 1

                elapsed_time = time.perf_counter() - frame_start
                sleep_time = TARGET_FRAME_TIME - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if step >= ppo.num_transitions_per_env:
                ppo.async_update(obs.to(device))

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_filename = f"{timestamp}_ppo_policy.pt"
    save_path = os.path.join(save_dir, save_filename)
    torch.save(ppo.learner.policy.state_dict(), save_path)
    print(f"Final policy saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL2 Start Menu Game")
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to a specific .pt file to load weights from",
    )
    args = parser.parse_args()
    start_game(args)
