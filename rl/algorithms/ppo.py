import torch
from modules.actor import MultiDiscreteActor
from modules.actor_critic import ActorCritic
from modules.critic import ValueCritic
from storage.rollout_storage import RolloutStorage, Transition


class PPO:

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        input_dim: int,
        hidden_size: tuple[int] | list[int],
        output_dim: list[int],
        num_mini_batches: int = 4,
        num_epochs: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95,
        learning_rate: float = 3e-4,
        device: str = "cpu",
        clip_coef: float = 0.2,
        entropy_loss_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.device = device
        self.policy = ActorCritic(
            MultiDiscreteActor,
            ValueCritic,
            input_dim,
            hidden_size,
            output_dim,
            activation_func_class=torch.nn.ReLU,
        )

        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.transition = Transition()
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            torch.zeros(num_envs, input_dim),
            torch.zeros(num_envs, len(output_dim)),
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
        self.batch_size = self.num_envs * self.num_transitions_per_env

    def act(self, obs: torch.Tensor) -> torch.Tensor | None:
        self.transition.observation = obs
        action, log_prob, _, value = self.policy(obs)
        self.transition.action = action.detach()
        self.transition.action_log_prob = log_prob.detach()
        self.transition.value = value.detach()
        return self.transition.action

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.transition.reward = rewards
        self.transition.done = dones
        self.storage.add_transition(self.transition)

        self.transition.clear()

    def compute_returns(self, obs: torch.Tensor) -> None:
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
                metrics["tech/kl_divergence"].append(approx_kl.item())
                # calculate how often we are clipping (useful debug stat)
                clip_frac = (torch.abs(ratio - 1.0) > self.clip_coef).float().mean()
                metrics["tech/clip_fraction"].append(clip_frac.item())

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

            metrics["loss/policy"].append(pg_loss.item())
            metrics["loss/value"].append(value_loss.item())
            metrics["loss/entropy"].append(entropy_loss.item())
            metrics["loss/total"].append(loss.item())
            self.storage.clear()

        return {k: sum(v) / len(v) for k, v in metrics.items()}
