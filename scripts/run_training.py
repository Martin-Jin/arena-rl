"""Run the Arena AI PPO training workflow."""

from functools import partial
from pathlib import Path

import torch

from arena_ai.agents.policy import ActorCritic, PolicyAdapter
from arena_ai.agents.scripted import OPPONENTS
from arena_ai.config import Config, load_config
from arena_ai.envs.training_envs.one_v_one_env import OneVOneEnv
from arena_ai.envs.vector_env import VectorEnv
from arena_ai.training.checkpoints import save_checkpoint
from arena_ai.training.ppo import (
    advantage_estimation,
    discounted_returns,
    normalize_advantages,
    train_step,
)
from arena_ai.training.train import TrainingLoop, TrainingState

# Switch this to True to render every simulation step during training.
VISUALIZE_TRAINING = False
TRAINING_DEVICE = "auto"


def select_training_device() -> torch.device:
    """Select CUDA by default when available, otherwise use the CPU."""
    if TRAINING_DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(TRAINING_DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TRAINING_DEVICE is set to CUDA, but CUDA is unavailable")
    return device


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
        device = values.device
        # 1. Compute advantages and returns for the batch, using GAE and discounting.
        # These loops run one tiny operation per step, which is much faster on the
        # CPU than on a GPU, so compute them there and move the results back.
        cpu_values = values.cpu()
        advantages = advantage_estimation(
            batch.rewards.cpu(),
            cpu_values,
            next_values.cpu(),
            terminated.cpu(),
            policy_config.discount_factor,
            policy_config.gae_lambda,
            episode_ends.cpu(),
        ).to(device)
        returns = discounted_returns(
            batch.rewards.cpu(),
            terminated.cpu(),
            torch.as_tensor(last_value, dtype=cpu_values.dtype),
            policy_config.discount_factor,
            episode_ends.cpu(),
            next_values.cpu(),
        ).to(device)
        # 2. Normalize advantages to stabilize training.
        advantages = normalize_advantages(advantages)
        # 3. Flatten the batch to prepare for mini-batch training.
        flat = batch.flattened()
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        # 4. Split the batch into mini-batches and perform multiple epochs of training.
        batch_size = min(policy_config.batch_size, len(flat.actions))
        metrics: dict[str, float] = {}
        for _ in range(policy_config.num_epochs):
            # Randomly permute the batch indices to create mini-batches for training.
            permutation = torch.randperm(len(flat.actions), device=flat.actions.device)
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
        policy.sync()
        return metrics

    return update_policy


def run_training(config: Config | None = None) -> TrainingState:
    """Construct all training components and run the configured workflow."""
    config = config or load_config()
    # Set the random seed for reproducibility if specified in the config.
    if config.training.seed is not None:
        torch.manual_seed(config.training.seed)
    device = select_training_device()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"Training device: {device}")
    # Build the model, optimizer, policy, environment, and training loop.
    model = ActorCritic(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.agent.policy.learning_rate
    )
    policy = PolicyAdapter(model, device, "cpu")
    environment_factory = partial(OneVOneEnv, config)
    update_policy = build_update_callback(policy, optimizer, config)
    checkpoint_path = Path("checkpoints") / "latest.pt"

    def checkpoint(state: TrainingState) -> None:
        """Save a checkpoint, then print the latest training metrics."""
        save_checkpoint(checkpoint_path, model, optimizer, config, state)
        m = state.metrics
        print(
            f"[iter {state.iteration}] steps={state.environment_steps} "
            f"episodes={int(m.get('episodes', 0))} "
            f"win_rate={m.get('win_rate', 0.0):.2%} "
            f"episode_return={m.get('episode_return', 0.0):.3f} "
            f"loss={m.get('loss', 0.0):.4f} "
            f"policy_loss={m.get('policy_loss', 0.0):.4f} "
            f"value_loss={m.get('value_loss', 0.0):.4f} "
            f"entropy={m.get('entropy', 0.0):.3f}"
        )
        for name, stats in state.opponent_stats.items():
            print(
                f"    vs {name}: episodes={int(stats['episodes'])} "
                f"win_rate={stats['win_rate']:.0%} "
                f"truncation_rate={stats['truncation_rate']:.0%} "
                f"mean_length={stats['mean_length']:.0f}"
            )

    visualizer = None
    if VISUALIZE_TRAINING:
        from arena_ai.viewer.training import TrainingVisualizer

        visualizer = TrainingVisualizer(config)

    # Build the environment(s) and run the training loop.
    factories = make_factories(config)
    environment = (
        VectorEnv(factories) if config.training.num_envs > 1 else factories[0]()
    )
    loop = TrainingLoop(
        environment,
        policy,
        update_policy,
        save_checkpoint=checkpoint,
        step_callback=visualizer,
        device=device,
        stats_window=config.training.stats_window,
    )
    try:
        state = loop.run(config.training)
        # The loop already saved this iteration if it fell on the interval.
        if state.iteration % config.training.checkpoint_interval:
            checkpoint(state)
    except KeyboardInterrupt:
        # Preserve the latest model even when training is stopped between
        # configured checkpoint intervals.
        checkpoint(loop.state)
        print(
            f"\nTraining interrupted at iteration {loop.state.iteration}; "
            f"checkpoint saved to {checkpoint_path}."
        )
        state = loop.state
    finally:
        if visualizer is not None:
            visualizer.close()
    return state


def make_factories(config: Config):
    """Create a list of environment factories for the training loop."""
    opponents = list(OPPONENTS.items())
    return [
        partial(
            OneVOneEnv,
            config,
            opponent_policy=partial(policy, config=config),
            opponent_name=name,
            max_episode_steps=config.training.max_episode_steps,
        )
        for name, policy in (
            opponents[i % len(opponents)] for i in range(config.training.num_envs)
        )
    ]


if __name__ == "__main__":
    # Since relatively small MLP, we can limit PyTorch to a single thread for training,
    # which can reduce overhead and improves performance.
    torch.set_num_threads(1)
    run_training()
