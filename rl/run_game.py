# rl/run_game.py
import time

from modules.actor import MultiDiscreteActor
from modules.actor_critic import ActorCritic
from modules.critic import ValueCritic
from rl2_env import RL2Env

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
    model = ActorCritic(
        MultiDiscreteActor,
        ValueCritic,
        input_dim=10,
        hidden_size=[16, 16],
        output_dim=[3, 3],
    )

    is_initialized = env.game.initialize()

    if not is_initialized:
        print("Game was not initialized properly. Exiting...")
        return

    counter = 0
    next_tick = time.perf_counter()

    obs, _ = env.reset()

    while env.game.get_game_state() != game_state["in_shutdown"]:
        env.game.process_input()

        action, _, _ = model(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        dones = terminated | truncated

        env.game.render(1.0)

        next_tick += env.step_dt
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
