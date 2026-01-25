import copy
import threading

import torch

from rl.algorithms.ppo import PPO
from rl.storage.rollout_storage import RolloutStorage, Transition


class AsyncPPO:
    def __init__(self, *args, **kwargs):
        self.learner = PPO(*args, **kwargs)

        self.device = self.learner.device
        self.num_envs = self.learner.num_envs
        self.num_transitions_per_env = self.learner.num_transitions_per_env

        self.input_dim = self.learner.input_dim
        self.output_dim = self.learner.output_dim

        self.inference_policy = copy.deepcopy(self.learner.policy)
        self.inference_policy.eval()
        self.inference_policy.to(self.device)

        # We define two rollout storages to be able to train with one while
        # populating the other.
        self.inference_storage = RolloutStorage(
            self.num_envs,
            self.num_transitions_per_env,
            torch.zeros(self.num_envs, self.input_dim),
            torch.zeros(self.num_envs, len(self.output_dim)),
            device=self.device,
        )

        self.training_storage = RolloutStorage(
            self.num_envs,
            self.num_transitions_per_env,
            torch.zeros(self.num_envs, self.input_dim),
            torch.zeros(self.num_envs, len(self.output_dim)),
            device=self.device,
        )

        self.transition = Transition()

        self.training_thread = None
        self.training_lock = threading.Lock()
        self.training_in_progress = False

        # The training stream has priority 1 so that GPU inference is always
        # prioritized as a stable game loop is more important than a shorter
        # training time.
        self.training_stream = torch.cuda.Stream(device=self.device, priority=1)

    def act(self, obs: torch.Tensor) -> torch.Tensor | None:
        self.transition.observation = obs
        with torch.no_grad():
            action, log_prob, _, value = self.inference_policy(obs)

        self.transition.action = action.detach()
        self.transition.action_log_prob = log_prob.detach()
        self.transition.value = value.detach()
        return self.transition.action

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        self.transition.reward = rewards
        self.transition.done = dones
        self.inference_storage.add_transition(self.transition)
        self.transition.clear()

    def async_update(self, obs: torch.Tensor) -> None:

        with self.training_lock:
            assert (
                not self.training_in_progress
            ), "Training was still in progress when a new update started."

            full_storage = self.inference_storage
            empty_storage = self.training_storage

            self.training_storage = full_storage
            self.inference_storage = empty_storage

            self.inference_storage.clear()

            self.training_in_progress = True

            self.training_thread = threading.Thread(
                target=self._training_worker, args=(obs.clone(),)
            )
            self.training_thread.start()

            return None

    def _training_worker(self, obs: torch.Tensor) -> None:
        try:
            with torch.cuda.stream(self.training_stream):
                self._run_training(obs)

        except Exception as e:
            print(f"Exception in training thread: {e}")
            import traceback

            traceback.print_exc()

        finally:
            with self.training_lock:
                self.training_in_progress = False

    def _run_training(self, obs: torch.Tensor) -> None:
        self.learner.storage = self.training_storage

        with torch.inference_mode():
            self.learner.compute_returns(obs)

        metrics = self.learner.update()

        self.training_stream.synchronize()

        self.inference_policy.load_state_dict(self.learner.policy.state_dict())

        print(f"Async Training Step Done. Loss: {metrics.get('loss/total', 0):.4f}")
