import copy
import threading

import torch

from rl.modules.actor_critic import ActorCritic
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

        self.queued_policy = copy.deepcopy(self.inference_policy).to(self.device)
        self.inference_obs_normalizer = copy.deepcopy(self.learner.obs_normalizer)
        self.queued_obs_normalizer = copy.deepcopy(self.inference_obs_normalizer)
        self.raw_obs_inference = torch.zeros(
            (self.num_transitions_per_env, self.num_envs, self.input_dim),
            device=self.device,
        )
        self.raw_obs_learner = torch.zeros(
            (self.num_transitions_per_env, self.num_envs, self.input_dim),
            device=self.device,
        )

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
        self.is_new_policy_available = False
        self.new_policy_ready_event = torch.cuda.Event()

        # The training stream has priority 1 so that GPU inference is always
        # prioritized as a stable game loop is more important than a shorter
        # training time.
        self.training_stream = torch.cuda.Stream(device=self.device, priority=1)

        # Similarly, the copy stream that copies params from old to new policy
        # is set to an even lower priority.
        self.copy_stream = torch.cuda.Stream(device=self.device, priority=2)

    def act(self, obs: torch.Tensor) -> torch.Tensor | None:
        if self.is_new_policy_available:
            self._publish_new_policy()

        if self.inference_policy.is_recurrent:
            actor_hidden = self.inference_policy.actor.get_hidden_state()
            critic_hidden = self.inference_policy.critic.get_hidden_state()
            self.transition.hidden_states = (
                None if actor_hidden is None else actor_hidden.detach(),
                None if critic_hidden is None else critic_hidden.detach(),
            )

        normalized_obs = self._normalize_obs(obs)
        self.transition.observation = normalized_obs
        with torch.no_grad():
            action, log_prob, _, value = self.inference_policy(normalized_obs)

        self.transition.action = action.detach()
        self.transition.action_log_prob = log_prob.detach()
        self.transition.value = value.detach()
        return self.transition.action

    def process_env_step(
        self, obs: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor
    ) -> None:
        # Save raw obs in separate buffer for updating normalizer
        idx = self.inference_storage.step % self.num_transitions_per_env
        self.raw_obs_inference[idx].copy_(obs)

        self.transition.reward = rewards
        self.transition.done = dones
        self.inference_storage.add_transition(self.transition)
        if self.inference_policy.is_recurrent:
            self.inference_policy.actor.reset_memory(dones)
            self.inference_policy.critic.reset_memory(dones)

        self.transition.clear()

    def async_update(self, obs: torch.Tensor) -> bool:
        """Begins an asynchronous PPO update.
        Returns early if training is already in progress."""

        with self.training_lock:
            if self.training_in_progress or self.is_new_policy_available:
                return False

            total_steps_collected = self.inference_storage.step

            self.inference_storage, self.training_storage = (
                self.training_storage,
                self.inference_storage,
            )

            self.raw_obs_inference, self.raw_obs_learner = (
                self.raw_obs_learner,
                self.raw_obs_inference,
            )

            self.inference_storage.clear()

            self.training_in_progress = True

            obs = obs.clone()
            bootstrap_value = self.inference_policy.get_bootstrap_value(
                self._normalize_obs(obs)
            ).clone()

            # We create a synchronization event to mark the completion of any buffer writes
            # (e.g., storage.add_transition()) that occurred on the inference CUDA stream
            # to account for the potential read/write race condition.
            sync_event = torch.cuda.Event()
            sync_event.record()

            self.training_thread = threading.Thread(
                target=self._training_worker,
                args=(
                    obs,
                    sync_event,
                    total_steps_collected,
                    bootstrap_value,
                ),
            )
            self.training_thread.start()

            return True

    def _training_worker(
        self,
        obs: torch.Tensor,
        sync_event: torch.cuda.Event,
        total_steps_collected: int,
        bootstrap_value: torch.Tensor,
    ) -> None:
        try:
            with torch.cuda.stream(self.training_stream):
                self.training_stream.wait_event(sync_event)  # pyright: ignore[reportArgumentType]
                self._run_training(obs, total_steps_collected, bootstrap_value)

        except Exception as e:
            print(f"Exception in training thread: {e}")
            import traceback

            traceback.print_exc()

        finally:
            with self.training_lock:
                self.training_in_progress = False

    def _run_training(
        self,
        obs: torch.Tensor,
        total_steps_collected: int,
        bootstrap_value: torch.Tensor,
    ) -> None:
        self.learner.storage = self.training_storage
        self._update_learner_normalization(self.raw_obs_learner)

        with torch.inference_mode():
            self.learner.compute_returns(obs, total_steps_collected, bootstrap_value)

        metrics = self.learner.update()

        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_stream(self.training_stream)
            self.queued_policy.load_state_dict(self.learner.policy.state_dict())
            self.queued_obs_normalizer.load_state_dict(
                self.learner.obs_normalizer.state_dict()
            )
            self.new_policy_ready_event.record(self.copy_stream)

        with self.training_lock:
            self.is_new_policy_available = True

    def _publish_new_policy(self) -> None:
        with self.training_lock:
            if not self.is_new_policy_available:
                return

            torch.cuda.current_stream(self.device).wait_event(
                self.new_policy_ready_event
            )

            old_policy = self.inference_policy
            new_policy = self.queued_policy

            # `load_state_dict()` does not include the memory's states,
            # so in case of a recurrent policy, these must be loaded separately.
            if old_policy.is_recurrent:
                self._load_recurrent_state(old_policy, new_policy)

            self.inference_policy = new_policy
            self.queued_policy = old_policy

            self.inference_obs_normalizer, self.queued_obs_normalizer = (
                self.queued_obs_normalizer,
                self.inference_obs_normalizer,
            )

            self.is_new_policy_available = False

    @staticmethod
    def _load_recurrent_state(old_policy: ActorCritic, new_policy: ActorCritic) -> None:
        for old_module, new_module in (
            (old_policy.actor, new_policy.actor),
            (old_policy.critic, new_policy.critic),
        ):
            old_memory = old_module.memory
            new_memory = new_module.memory

            if old_memory is None or new_memory is None:
                return

            new_memory.hidden_state = old_memory.hidden_state
            old_memory.hidden_state = None

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        continuous = obs[:, : self.learner.norm_dim]
        categorical = obs[:, self.learner.norm_dim :]
        continuous_normalized = self.inference_obs_normalizer(continuous)
        return torch.cat([continuous_normalized, categorical], dim=-1)

    def _update_learner_normalization(self, raw_obs: torch.Tensor) -> None:
        # As the raw obs has shape (num_transitions, num_envs, num_inputs)
        # we need to flatten to use the learner's `update_normalization` method
        obs_flat = raw_obs.flatten(0, 1)
        self.learner.update_normalization(obs_flat)
