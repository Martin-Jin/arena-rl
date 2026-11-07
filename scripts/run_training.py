"""Run the Arena AI PPO training workflow."""

from functools import partial
from pathlib import Path

import torch

from arena_ai.agents.policy import ActorCritic, PolicyAdapter
from arena_ai.config import Config, TrainingConfig, load_config
from arena_ai.envs.training_envs.one_v_one_env import OneVOneEnv
from arena_ai.training.checkpoints import save_checkpoint
from arena_ai.training.ppo import (
    advantage_estimation,
    normalize_advantages,
    train_step,
)
from arena_ai.training.train import TrainingLoop, TrainingState


def build_update_callback(
    policy: PolicyAdapter,
    optimizer: torch.optim.Optimizer,
    config: Config,
):
    """Build the PPO callback consumed by the outer training loop."""
    policy_config = config.agent.policy

    def update_policy(
        batch,
        last_value: float,
        *,
        terminated: torch.Tensor,
        episode_ends: torch.Tensor,
        next_values: torch.Tensor,
    ) -> dict[str, float]:
        values = batch.values
        del last_value
        advantages = advantage_estimation(
            batch.rewards,
            values,
            next_values,
            terminated,
            policy_config.discount_factor,
            policy_config.gae_lambda,
            episode_ends,
        )
        returns = advantages + values
        advantages = normalize_advantages(advantages)
        flat = batch.flattened()
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        batch_size = min(policy_config.batch_size, len(flat.actions))
        metrics: dict[str, float] = {}
        for _ in range(policy_config.num_epochs):
            permutation = torch.randperm(
                len(flat.actions), device=flat.actions.device
            )
            for indices in permutation.split(batch_size):
                metrics = train_step(
                    policy.forward,
                    optimizer,
                    flat.observations[indices],
                    flat.actions[indices],
                    flat.log_probs[indices],
                    flat_advantages[indices],
                    flat_returns[indices],
                    policy_config.clip_range,
                    policy_config.value_loss_coef,
                    policy_config.entropy_coef,
                )
        return metrics

    return update_policy


def run_training(config: Config | None = None) -> TrainingState:
    """Construct all training components and run the configured workflow."""
    config = config or load_config()
    if config.training.seed is not None:
        torch.manual_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.agent.policy.learning_rate
    )
    policy = PolicyAdapter(model, device)
    environment_factory = partial(OneVOneEnv, config)
    update_policy = build_update_callback(policy, optimizer, config)
    checkpoint_path = Path("checkpoints") / "latest.pt"
    checkpoint = partial(
        save_checkpoint,
        checkpoint_path,
        model,
        optimizer,
        config,
    )
    loop = TrainingLoop(
        environment_factory(),
        policy,
        update_policy,
        save_checkpoint=checkpoint,
    )
    state = loop.run(
        TrainingConfig(
            iterations=config.training.iterations,
            rollout_steps=config.training.rollout_steps,
            checkpoint_interval=config.training.checkpoint_interval,
            seed=config.training.seed,
        )
    )
    checkpoint(state)
    return state


if __name__ == "__main__":
    run_training()
