import argparse
import datetime
import os
import time

import torch

from rl.algorithms.async_ppo import AsyncPPO
from rl.arelto_env import AreltoEnv

TARGET_FPS = 60
TARGET_FRAME_TIME = 1 / TARGET_FPS


def start_game(args):
    checkpoint_path: str = args.load_checkpoint
    device: str = "cuda"
    env = AreltoEnv(step_dt=0.02)
    num_envs = env.num_envs
    obs_size = env.game.get_observation_size()
    num_rays = env.game.get_enemy_num_rays()
    ray_history_length = env.game.get_enemy_ray_history_length()
    game_state_dict = env.game_state_dict

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

    env.game.set_game_state(game_state_dict["in_start_screen"])

    # We need to get the initial obs to infer first action.
    obs, _ = env.reset()

    # Main Game Loop
    while env.game.get_game_state() != game_state_dict["in_shutdown"]:
        if env.game.get_game_state() == game_state_dict["in_start_screen"]:
            env.game.process_input()
            env.game.render(1.0)
            if env.game.get_game_state() == game_state_dict["is_running"]:
                print("Transitioning to Training Loop...")
                # As we might have transitions stored in the rollout storage
                # when we enter the start screen, we clear the storage screen
                # to not go out of bounds when we begin filling it again.
                ppo.inference_storage.clear()

        elif (
            env.game.get_game_state() == game_state_dict["is_running"]
            or env.game.get_game_state() == game_state_dict["in_settings_menu"]
            or env.game.get_game_state() == game_state_dict["in_level_up"]
            or env.game.get_game_state() == game_state_dict["in_quit_confirm"]
        ):
            # We keep track of the number of steps to handle pauses correctly.
            step = 0
            while True:
                frame_start = time.perf_counter()
                env.game.process_input()
                state = env.game.get_game_state()
                if state == game_state_dict["in_shutdown"]:
                    break
                if state == game_state_dict["in_start_screen"]:
                    print("Returned to Menu")
                    print("Resetting policy parameters...")
                    ppo = create_agent()
                    break
                if (
                    state == game_state_dict["in_settings_menu"]
                    or state == game_state_dict["in_level_up"]
                    or state == game_state_dict["in_quit_confirm"]
                ):
                    env.game.render(1.0)
                    continue

                with torch.inference_mode():
                    env.game.process_input()

                    if env.game.get_game_state() != game_state_dict["is_running"]:
                        continue

                    action = ppo.act(obs.to(device))
                    obs, reward, terminated, truncated, _ = env.step(action)
                    dones = terminated | truncated
                    ppo.process_env_step(
                        obs.to(device), reward.to(device), dones.to(device)
                    )
                    env.game.render(1.0)

                step += 1

                # We only want to update the policy once we have collected enough data.
                if step >= ppo.num_transitions_per_env:
                    if ppo.async_update(obs.to(device)):
                        step = 0

                elapsed_time = time.perf_counter() - frame_start
                sleep_time = TARGET_FRAME_TIME - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_filename = f"{timestamp}_ppo_policy.pt"
    save_path = os.path.join(save_dir, save_filename)
    torch.save(ppo.learner.policy.state_dict(), save_path)
    print(f"Final policy saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arelto Start Menu Game")
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to a specific .pt file to load weights from",
    )
    args = parser.parse_args()
    start_game(args)
