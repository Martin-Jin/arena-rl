"""Run the Arena AI PPO training workflow."""

from functools import partial
from datetime import datetime
from pathlib import Path

import torch

from arena_ai.agents.policy import ActorCritic, PolicyAdapter
from arena_ai.agents.scripted import EpisodePolicy, training_opponents
from arena_ai.config import Config, load_config
from arena_ai.envs.training_envs.one_v_many_env import OneVManyEnv
from arena_ai.envs.async_vector_env import AsyncVectorEnv
from arena_ai.envs.vector_env import VectorEnv
from arena_ai.training.checkpoints import load_checkpoint, save_checkpoint
from arena_ai.training.ppo import (
    advantage_estimation,
    discounted_returns,
    GraphedTrainStep,
    explained_variance,
    normalize_advantages,
    train_step,
)
from arena_ai.training.report import format_stats
from arena_ai.training.stats import rollout_metrics
from arena_ai.training.train import TrainingLoop, TrainingState

# Switch this to True to render every simulation step during training.
VISUALIZE_TRAINING = False
TRAINING_DEVICE = "auto"
# Run each PPO minibatch step as one replayed CUDA graph (CUDA only; see GraphedTrainStep).
# Set to False to use ordinary steps, for example to compare results or to debug.
USE_CUDA_GRAPH = True
# Set to a checkpoint path, for example "checkpoints/latest.pt", to continue that
# run. The model, optimizer and progress counters are restored and
# training.iterations more iterations are run on top of the saved ones. Leave as
# None to start a new run.
# A checkpoint only loads if it was saved with the current observation layout and network
# shape, so train a new one after changing either. checkpoints/best.pt currently loads.
# Opponent behavior is read from the code, not the checkpoint, so a resumed run fights the
# current scripted opponents.
# RESUME_FROM: str | None = None
RESUME_FROM = "checkpoints/latest.pt"


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
    graphed: GraphedTrainStep | None = None

    def update_policy(
        batch,
        last_value: float,
        *,
        terminated: torch.Tensor,
        episode_ends: torch.Tensor,
        next_values: torch.Tensor,
    ) -> dict[str, float]:
        nonlocal graphed
        values = batch.values
        device = values.device
        use_graph = USE_CUDA_GRAPH and device.type == "cuda"
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
        cpu_returns = discounted_returns(
            batch.rewards.cpu(),
            terminated.cpu(),
            torch.as_tensor(last_value, dtype=cpu_values.dtype),
            policy_config.discount_factor,
            episode_ends.cpu(),
            next_values.cpu(),
        )
        returns = cpu_returns.to(device)
        # 2. Normalize advantages to stabilize training.
        advantages = normalize_advantages(advantages)
        # 3. Flatten the batch to prepare for mini-batch training.
        flat = batch.flattened()
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        # 4. Split the batch into mini-batches and perform multiple epochs of training.
        batch_size = min(policy_config.batch_size, len(flat.actions))
        totals: dict[str, torch.Tensor] = {}
        minibatches = 0
        minibatch_tensors = (
            flat.observations,
            flat.actions,
            flat.log_probs,
            flat_advantages,
            flat_returns,
        )
        for _ in range(policy_config.num_epochs):
            # Randomly permute the batch indices to create mini-batches for training.
            permutation = torch.randperm(len(flat.actions), device=flat.actions.device)
            for indices in permutation.split(batch_size):
                if use_graph:
                    if graphed is None and len(indices) == batch_size:
                        graphed = GraphedTrainStep(
                            policy.forward,
                            optimizer,
                            tuple(tensor[indices] for tensor in minibatch_tensors),
                            policy_config.clip_range,
                            policy_config.value_loss_coef,
                            policy_config.entropy_coef,
                        )
                    graphed.step(minibatch_tensors, indices)
                    continue
                stats = train_step(
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
                for name, value in stats.items():
                    totals[name] = totals[name] + value if name in totals else value
                minibatches += 1
        policy.sync()
        # Average over every minibatch (the last one alone is noisy), add the
        # batch-level statistics, and read everything back with one device
        # transfer. Reading each value separately would sync the GPU every time.
        if use_graph:
            tensors = graphed.means()
        else:
            tensors = {name: value / minibatches for name, value in totals.items()}
        tensors["explained_variance"] = explained_variance(
            cpu_values.reshape(-1), cpu_returns.reshape(-1)
        ).to(device)
        tensors.update(rollout_metrics(flat.actions, flat.rewards))
        return dict(zip(tensors, torch.stack(list(tensors.values())).tolist()))

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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.agent.policy.learning_rate)
    policy = PolicyAdapter(model, device, "cpu")
    resumed_state: TrainingState | None = None
    if RESUME_FROM is not None:
        saved = load_checkpoint(RESUME_FROM, model, optimizer, map_location=device)
        resumed_state = saved["training_state"]
        policy.sync()  # the CPU acting copy must start from the restored weights
        # A graph run saves its optimizer as "capturable", which only works on CUDA, so set
        # the flag for this run (GraphedTrainStep sets it again when it builds the graph).
        for group in optimizer.param_groups:
            group["capturable"] = USE_CUDA_GRAPH and device.type == "cuda"
        if config.training.seed is not None:
            # Offset so the continued run does not replay the first run's stream.
            torch.manual_seed(config.training.seed + resumed_state.iteration)
        print(
            f"Resuming from {RESUME_FROM} at iteration {resumed_state.iteration}; "
            f"running {config.training.iterations} more."
        )
    update_policy = build_update_callback(policy, optimizer, config)
    checkpoint_path = Path("checkpoints") / "latest.pt"
    # Each run also keeps numbered checkpoints in its own folder, so starting a
    # new run overwrites latest.pt but never an earlier run's history.
    run_dir = Path("checkpoints") / datetime.now().strftime("run-%Y%m%d-%H%M%S")

    def checkpoint(state: TrainingState) -> None:
        """Save the latest model and training progress."""
        save_checkpoint(checkpoint_path, model, optimizer, config, state)
        save_checkpoint(run_dir / f"iter_{state.iteration:06d}.pt", model, optimizer, config, state)
        if config.training.print_stats:
            print(f"[iter {state.iteration}] checkpoint saved to {checkpoint_path}")

    visualizer = None
    if VISUALIZE_TRAINING:
        from arena_ai.viewer.training import TrainingVisualizer

        visualizer = TrainingVisualizer(config)

    # Build the environment(s) and run the training loop.
    factories = make_factories(config)
    workers = config.training.env_workers
    if len(factories) == 1:
        environment = factories[0]()
    elif workers > 1 and visualizer is None:  # The visualizer needs the environments here.
        environment = AsyncVectorEnv(factories, workers)
    else:
        environment = VectorEnv(factories)
    loop = TrainingLoop(
        environment,
        policy,
        update_policy,
        save_checkpoint=checkpoint,
        step_callback=visualizer,
        log_callback=(
            (lambda state: print(format_stats(state))) if config.training.print_stats else None
        ),
        device=device,
        stats_window=config.training.stats_window,
        initial_state=resumed_state,
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
        if isinstance(environment, VectorEnv):
            environment.close()  # Stops the worker processes.
        if visualizer is not None:
            visualizer.close()
    return state


def make_factories(config: Config):
    """Create a list of environment factories for the training loop."""
    opponent_names = training_opponents(
        config.training.opponents,
        config.training.num_envs,
        config.game.observation.max_opponents,
    )
    return [
        partial(
            OneVManyEnv,
            config,
            [(name, EpisodePolicy(name, config)) for name in names],
            max_episode_steps=config.training.max_episode_steps,
        )
        for names in opponent_names
    ]


if __name__ == "__main__":
    # Since relatively small MLP, we can limit PyTorch to a single thread for training,
    # which can reduce overhead and improves performance.
    torch.set_num_threads(1)
    run_training()
