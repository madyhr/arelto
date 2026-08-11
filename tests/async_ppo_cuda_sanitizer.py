"""Standalone CUDA sanitizer scenario for AsyncPPO policy publication."""

import torch

from rl.algorithms.async_ppo import AsyncPPO


NUM_ENVS = 2
NUM_RAYS = 4
INPUT_DIM = 4 * NUM_RAYS


def main() -> None:
    ppo = AsyncPPO(
        num_envs=NUM_ENVS,
        input_dim=INPUT_DIM,
        hidden_size=[8],
        output_dim=[2, 3],
        num_transitions_per_env=1,
        num_mini_batches=1,
        num_epochs=1,
        num_rays=NUM_RAYS,
        num_entity_types=3,
        encoder_output_dim=4,
        is_recurrent=False,
        device="cuda",
    )
    torch.cuda.synchronize()

    with torch.no_grad():
        for parameter in ppo.learner.policy.parameters():
            parameter.add_(1.0)
    torch.cuda.synchronize()

    ppo.learner.compute_returns = lambda *_args: None
    ppo.learner.update = lambda: {}

    observation = torch.zeros(NUM_ENVS, INPUT_DIM, device="cuda")
    bootstrap_value = torch.zeros(NUM_ENVS, 1, device="cuda")

    with torch.cuda.stream(ppo.training_stream):
        ppo._run_training(observation, 1, bootstrap_value)

    # Inference immediately consumes the published policy on the default
    # stream. The sanitizer raises if publication omitted the required handoff.
    ppo.act(observation)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
