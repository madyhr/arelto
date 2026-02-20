import torch

from rl.modules.actor import MultiDiscreteActor
from rl.modules.actor_critic import ActorCritic
from rl.modules.critic import ValueCritic
from rl.modules.normalization import EmpiricalNormalization
from rl.modules.ray_encoder import RayEncoder
from rl.storage.rollout_storage import RolloutStorage, Transition


class PPO:
    def __init__(
        self,
        num_envs: int,
        input_dim: int,
        hidden_size: tuple[int] | list[int] = [512, 256, 128],
        output_dim: list[int] = [3, 3],
        num_transitions_per_env: int = 50,
        num_mini_batches: int = 8,
        num_epochs: int = 2,
        gamma: float = 0.99,
        lam: float = 0.95,
        learning_rate: float = 3e-4,
        clip_coef: float = 0.2,
        entropy_loss_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        num_rays: int = 72,
        num_entity_types: int = 6,
        ray_history_length: int = 1,
        encoder_output_dim: int = 128,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.encoder = RayEncoder(
            num_rays=num_rays,
            num_entity_types=num_entity_types,
            history_length=ray_history_length,
            output_dim=encoder_output_dim,
        )
        # The observation space contains 2 * 'total_rays' number of ray distances
        # then 2 * 'total_rays' number of entity types.
        expected_input_dim = 4 * num_rays * ray_history_length
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"Shape mismatch: Environment obs_size ({self.input_dim=}) "
                f"does not match computed RayEncoder expected dimensions ({expected_input_dim=}). "
                f"Check C++ bindings and Python RayEncoder definitions."
            )

        #  We only want to normalize the distances as the types are categorical.
        self.norm_dim = 2 * self.encoder.total_rays
        self.obs_normalizer = EmpiricalNormalization(self.norm_dim).to(self.device)

        self.policy = ActorCritic(
            MultiDiscreteActor,
            ValueCritic,
            self.input_dim,
            hidden_size,
            self.output_dim,
            self.encoder,
            activation_func_class=torch.nn.ReLU,
        )

        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.transition = Transition()
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            torch.zeros(num_envs, self.input_dim),
            torch.zeros(num_envs, len(self.output_dim)),
            device=self.device,
        )

        self.num_envs = num_envs

        self.gamma = gamma
        self.lam = lam
        self.clip_coef = clip_coef
        self.entropy_loss_coef = entropy_loss_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        self.num_transitions_per_env = num_transitions_per_env
        self.num_mini_batches = num_mini_batches
        self.num_epochs = num_epochs

        print(" --- PPO Params --- ")
        print(f"Hidden shape: {hidden_size}")
        print(f"Num epochs: {num_epochs}")
        print(f"Num mini batches: {num_mini_batches}")

    def act(self, obs: torch.Tensor) -> torch.Tensor | None:
        obs = self._normalize_obs(obs)
        self.transition.observation = obs
        action, log_prob, _, value = self.policy(obs)
        self.transition.action = action.detach()
        self.transition.action_log_prob = log_prob.detach()
        self.transition.value = value.detach()
        return self.transition.action

    def process_env_step(
        self,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.update_normalization(obs)
        self.transition.reward = rewards
        self.transition.done = dones
        self.storage.add_transition(self.transition)

        self.transition.clear()

    def compute_returns(self, obs: torch.Tensor) -> None:
        obs = self._normalize_obs(obs)
        last_value = self.policy.get_value(obs).detach()
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_value = (
                last_value
                if step == self.storage.num_transitions_per_env - 1
                else self.storage.values[step + 1]
            )
            next_is_nonterminal = 1.0 - self.storage.dones[step].float()
            delta = (
                self.storage.rewards[step]
                + next_is_nonterminal * self.gamma * next_value
                - self.storage.values[step]
            )
            advantage = delta + next_is_nonterminal * self.gamma * self.lam * advantage

            self.storage.returns[step] = advantage + self.storage.values[step]

        self.storage.advantages = self.storage.returns - self.storage.values

    def update(self) -> dict[str, float]:
        generator = self.storage.get_mini_batch_generator(
            self.num_mini_batches, self.num_epochs
        )
        metrics = {
            "loss/policy": [],
            "loss/value": [],
            "loss/entropy": [],
            "loss/total": [],
            "tech/kl_divergence": [],
            "tech/clip_fraction": [],
        }

        for (
            obs_batch,
            action_batch,
            value_batch,
            advantage_batch,
            return_batch,
            old_action_log_prob_batch,
        ) in generator:
            advantage_batch = (advantage_batch - advantage_batch.mean()) / (
                advantage_batch.std() + 1e-8
            )

            _, logprob, entropy, value = self.policy(obs_batch, action_batch)
            logratio = logprob - old_action_log_prob_batch
            ratio = logratio.exp()
            with torch.no_grad():
                # k3 estimation as described in http://joschu.net/blog/kl-approx.html
                # only used for diagnostics
                approx_kl = ((ratio - 1) - logratio).mean()
                metrics["tech/kl_divergence"].append(approx_kl)
                # calculate how often we are clipping (useful debug stat)
                clip_frac = (torch.abs(ratio - 1.0) > self.clip_coef).float().mean()
                metrics["tech/clip_fraction"].append(clip_frac)

            pg_loss1 = -advantage_batch * ratio
            pg_loss2 = -advantage_batch * torch.clamp(
                ratio, 1 - self.clip_coef, 1 + self.clip_coef
            )

            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            value_loss_unclipped = ((value - return_batch) ** 2).mean()
            value_clipped = value_batch + torch.clamp(
                value - value_batch, -self.clip_coef, self.clip_coef
            )

            value_loss_clipped = ((value_clipped - return_batch) ** 2).mean()
            value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)

            value_loss = 0.5 * value_loss_max.mean()

            entropy_loss = entropy.mean()

            loss = (
                pg_loss
                - self.entropy_loss_coef * entropy_loss
                + self.value_loss_coef * value_loss
            )

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            metrics["loss/policy"].append(pg_loss.detach())
            metrics["loss/value"].append(value_loss.detach())
            metrics["loss/entropy"].append(entropy_loss.detach())
            metrics["loss/total"].append(loss.detach())

        self.storage.clear()

        return {k: torch.stack(v).mean().item() for k, v in metrics.items()}

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        continuous = obs[:, : self.norm_dim]
        categorical = obs[:, self.norm_dim :]
        continuous_normalized = self.obs_normalizer(continuous)
        return torch.cat([continuous_normalized, categorical], dim=-1)

    def update_normalization(self, obs: torch.Tensor):
        self.obs_normalizer.update(obs[:, : self.norm_dim])
