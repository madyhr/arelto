import argparse
import datetime
import os

import torch
from algorithms.ppo import PPO
from rl2_env import RL2Env

# Map C++ game states
game_state = {
    "in_start_screen": 0,
    "in_main_menu": 1,
    "is_running": 2,
    "is_gameover": 3,
    "in_shutdown": 4,
}


def run_learner(args):
    checkpoint_path: str = args.load_checkpoint
    device: str = args.device

    env = RL2Env(step_dt=0.02)
    num_envs = env.num_envs

    obs_size = env.game.get_observation_size()
    num_rays = env.game.get_enemy_num_rays()
    ray_history_length = env.game.get_enemy_ray_history_length()

    ppo = PPO(
        input_dim=obs_size,
        num_envs=num_envs,
        num_rays=num_rays,
        ray_history_length=ray_history_length,
        device=device,
    )
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        ppo.policy.load_state_dict(state_dict)
    elif checkpoint_path:
        print(
            f"Checkpoint path {checkpoint_path} provided but file not found. Starting from scratch."
        )

    if not env.game.initialize():
        return

    obs, _ = env.reset()

    while env.game.get_game_state() != 4:
        for _ in range(ppo.num_transitions_per_env):

            env.game.process_input()
            if env.game.get_game_state() == game_state["in_shutdown"]:
                break

            with torch.inference_mode():

                env.game.process_input()
                action = ppo.act(obs.to(device))
                obs, reward, terminated, truncated, _ = env.step(action)
                dones = terminated | truncated
                ppo.process_env_step(reward.to(device), dones.to(device))

                env.game.render(1.0)

        with torch.inference_mode():
            ppo.compute_returns(obs.to(device))

        train_metrics = ppo.update()

        print(train_metrics)

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_filename = f"{timestamp}_ppo_policy.pt"
    save_path = os.path.join(save_dir, save_filename)

    torch.save(ppo.policy.state_dict(), save_path)
    print(f"Final policy saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL2 PPO Learner")

    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to a specific .pt file to load weights from",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Path to a specific .pt file to load weights from",
    )

    args = parser.parse_args()
    run_learner(args)
