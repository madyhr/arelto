import argparse
import datetime
import os
import time

import torch

from rl.algorithms.async_ppo import AsyncPPO
from rl.arelto_env import (
    PAUSE_STATES,
    AreltoEnv,
    GameState,
)

TARGET_FPS = 60
TARGET_FRAME_TIME = 1 / TARGET_FPS


def start_game(args):
    checkpoint_path: str = args.load_checkpoint
    live_metrics_enabled: bool = args.live_metrics
    device: str = "cuda"
    env = AreltoEnv(step_dt=TARGET_FRAME_TIME)
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
            track_metrics=live_metrics_enabled,
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

    plotter = None
    if live_metrics_enabled:
        from rl.utils.live_metrics_plotter import LiveMetricsPlotter

        plotter = LiveMetricsPlotter()

    def update_live_metrics() -> None:
        if plotter is None:
            return
        for metrics in ppo.drain_metrics():
            plotter.add_metrics(metrics)
        plotter.process_events()

    env.game.set_game_state(GameState.IN_START_SCREEN)

    # We need to get the initial obs to infer first action.
    obs, _ = env.reset()

    # Main Game Loop
    while env.game.get_game_state() != GameState.IN_SHUTDOWN:
        update_live_metrics()
        state = env.game.get_game_state()

        if state == GameState.IN_START_SCREEN:
            env.game.process_input()
            env.game.render(1.0)
            if env.game.get_game_state() == GameState.IS_RUNNING:
                print("Transitioning to Training Loop...")
                # As we might have transitions stored in the rollout storage
                # when we enter the start screen, we clear the storage screen
                # to not go out of bounds when we begin filling it again.
                ppo.inference_storage.clear()

        elif state == GameState.IS_RUNNING or state in PAUSE_STATES:
            while True:
                frame_start = time.perf_counter()
                update_live_metrics()
                env.game.process_input()
                state = env.game.get_game_state()

                if state == GameState.IN_SHUTDOWN:
                    break

                if state == GameState.IN_START_SCREEN:
                    print("Returned to Menu")
                    print("Resetting policy parameters...")
                    if plotter is not None:
                        plotter.reset()
                    ppo = create_agent()
                    break

                if state in PAUSE_STATES:
                    env.game.step(TARGET_FRAME_TIME)
                    env.game.render(1.0)

                    elapsed_time = time.perf_counter() - frame_start
                    sleep_time = TARGET_FRAME_TIME - elapsed_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

                with torch.inference_mode():
                    # We might have transitioned state during process_input
                    if env.game.get_game_state() != GameState.IS_RUNNING:
                        continue

                    action = ppo.act(obs.to(device))
                    obs, reward, terminated, truncated, _ = env.step(action)
                    dones = terminated | truncated
                    ppo.process_env_step(
                        obs.to(device), reward.to(device), dones.to(device)
                    )
                    env.game.render(1.0)

                # We only want to update the policy once we have collected enough data.
                if ppo.inference_storage.step >= ppo.num_transitions_per_env:
                    ppo.async_update(obs.to(device))

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
    if plotter is not None:
        plotter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arelto Start Menu Game")
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to a specific .pt file to load weights from",
    )
    parser.add_argument(
        "--live-metrics",
        action="store_true",
        help="Show live policy loss, value loss, and reward graphs",
    )
    args = parser.parse_args()
    start_game(args)
