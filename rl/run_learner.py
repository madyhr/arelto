import torch
from algorithms.ppo import PPO
from rl2_env import RL2Env

import wandb

TOTAL_TIMESTEPS = 1_000_000
TRANSITIONS_PER_ENV = 500
INPUT_DIM = 10
HIDDEN_SIZE = [32, 32, 32]
OUTPUT_DIM = [3, 3]
EXP_NAME = "rl2_ppo_interactive"
DEVICE = "cuda"

# Map C++ game states
game_state = {
    "in_start_screen": 0,
    "in_main_menu": 1,
    "is_running": 2,
    "is_gameover": 3,
    "in_shutdown": 4,
}


def run_learner():
    env = RL2Env()  #
    num_envs = env.num_envs

    ppo = PPO(
        num_envs=num_envs,
        num_transitions_per_env=TRANSITIONS_PER_ENV,
        input_dim=INPUT_DIM,
        hidden_size=HIDDEN_SIZE,
        output_dim=OUTPUT_DIM,
        device=DEVICE,
    )  #

    if not env.game.initialize():
        return

    obs, _ = env.reset()

    while env.game.get_game_state != 4:
        for _ in range(TRANSITIONS_PER_ENV):

            env.game.process_input()
            if env.game.get_game_state() == game_state["in_shutdown"]:
                wandb.finish()
                return

            with torch.inference_mode():

                env.game.process_input()
                action = ppo.act(obs.to(DEVICE))
                obs, reward, terminated, truncated, _ = env.step(action)
                dones = terminated | truncated
                ppo.process_env_step(reward.to(DEVICE), dones.to(DEVICE))

                env.game.render(1.0)

        with torch.inference_mode():
            ppo.compute_returns(obs.to(DEVICE))

        train_metrics = ppo.update()

        print(train_metrics)


if __name__ == "__main__":
    run_learner()
