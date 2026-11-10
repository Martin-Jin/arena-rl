# Hardware use: what runs on the CPU and the GPU

Training uses the GPU only for the work that benefits from it. The network is small:
an attention input block over the fighters (64-wide embeddings, four heads) that feeds two
hidden layers of 128 units, about 123,000 parameters in total. The hidden layers receive
`[e_self ; context]`, the acting fighter's own embedding joined to the attention context
(see [network.md](network.md)). Most of the work is many tiny
steps, so a GPU is often slower than the CPU. Each GPU operation has a fixed
launch cost of a few microseconds, and moving data between devices costs more.
The GPU pays off only when one operation processes a lot of data at once.

## Choosing the device

[`scripts/run_training.py`](../scripts/run_training.py) selects the training
device from `TRAINING_DEVICE`:

- `"auto"` (default) uses CUDA when PyTorch can see a GPU, otherwise the CPU.
- `"cpu"` or `"cuda"` forces a device. `"cuda"` raises an error if CUDA is not
  available.

On CUDA the script also sets `torch.set_float32_matmul_precision("high")`, which
lets matrix multiplications use faster TensorFloat-32 on supported GPUs at
slightly lower precision. This only affects training on the GPU.

The script sets `torch.set_num_threads(1)` when run directly. The network is
small, so splitting each operation across CPU threads costs more in
synchronization than it saves. It also stops PyTorch from competing with the
simulation for cores.

## Where each step runs

| Step                                          | Device         | Why                                                                                                          |
| --------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------ |
| Game simulation, rewards, scripted opponents  | CPU            | Pure Python, deterministic, one fighter at a time. It must not depend on a GPU or on pygame. `training.env_workers` worker processes step the environments in parallel. |
| Observation encoding                          | CPU            | Plain Python lists, built for each step.                                                                     |
| Action selection and value estimates (rollout) | CPU            | Batches are one to a few observations. Launch and transfer costs would exceed the arithmetic.                |
| Rollout storage                               | CPU, then GPU  | Transitions are stored as Python lists. `Rollout.as_batch` converts them to tensors on the training device once per iteration. |
| GAE advantages and discounted returns         | CPU            | Sequential: 512 dependent steps, each doing tiny operations. See below.                                       |
| Advantage normalization, minibatch indexing   | Training device | One operation over the whole batch.                                                                          |
| PPO forward pass, loss, backward, Adam step   | Training device | Minibatches of 512 samples. On CUDA each step is one replayed CUDA graph (see below).                         |
| Checkpoint saving                             | Training device state | Tensors are saved as they are on the device. Loading uses `map_location="cpu"` by default.             |
| Viewer inference (playing against the agent)  | CPU            | One observation per frame, and the viewer must run without a GPU.                                            |

## The CPU copy used for acting

`PolicyAdapter` keeps two copies of the network:

- `model` is on the training device and is the one PPO updates.
- `actor` is a CPU copy used by `act`, `value` and `act_and_value` during rollouts. It is a
  [`StackedActor`](../src/arena_ai/agents/stacked_actor.py), not a second `ActorCritic`:
  the policy and value networks have the same shapes except the output layer, so their
  weights are stacked into tensors with a leading dimension of 2 and one pass runs both
  (the critic's output layer is padded with zeros). With a batch of 16 the time is the
  overhead of about 230 small operations per network, not the arithmetic, so one stacked pass
  costs about the same as one network. The stacked copy only reads weights and matches the
  networks to float rounding.

After every PPO update, `PolicyAdapter.sync()` rebuilds the stacked tensors from the new
weights of `model`. The rollout therefore always acts with the same weights the update
started from, which PPO requires, while avoiding a GPU round trip for every
simulation step. Without the copy, each step would move an observation to the
GPU, run the network, and move the result back. That is thousands of transfers
per iteration for a few microseconds of arithmetic each.

`act` also samples with `sample_with_log_prob` instead of
`torch.distributions.Categorical`. The tensors are tiny, so the objects'
validation and bookkeeping would cost more than the sampling. The PPO update
still uses `FactoredCategorical`, which needs gradients and entropy.

## Why GAE and returns run on the CPU

`advantage_estimation` and `discounted_returns` loop backwards over every step
of the rollout, because each step depends on the one after it. The loop cannot
be run in parallel, and each pass does a handful of operations on a few elements.
On a GPU each of those operations is a separate kernel launch, and checking
`.any()` forces the CPU to wait for the GPU. On the CPU the same operations run
directly.

`update_policy` in `run_training.py` therefore moves rewards, values, next
values and the episode flags to the CPU, computes advantages and returns, and
moves the results back to the training device before the PPO epochs. The numbers
are identical either way. `discounted_returns` also checks for truncations once
before its loop, not on every step.

## The CUDA graph for the PPO update

**Why it exists.** One PPO minibatch step on this network is about a thousand tiny GPU
operations. The GPU runs each in microseconds, but the CPU needs about 27 ms to queue them
one by one from Python, so the GPU mostly waits. **What it does.**
[`GraphedTrainStep`](../src/arena_ai/training/ppo.py) records one whole step (forward, loss,
backward, gradient clipping, Adam step) as a CUDA graph (a recording of the GPU work that is
replayed with one call). Each minibatch is copied into fixed buffers and the recording is
replayed, which takes about 3.7 ms. **How it works in code.** The recording stores memory
addresses, so the buffers, weights and optimizer state are only updated in place. Building it
runs three warm-up steps and then restores the weights and optimizer state, so it can be built
after a checkpoint was loaded. Statistics are summed inside the graph and read once per
update. **Limits.** CUDA only (`USE_CUDA_GRAPH` in `run_training.py` turns it off). A minibatch
of another size, such as a short last one, runs as an ordinary step with the same result. The
learning rate is recorded as a constant. The Adam optimizer is switched to its `capturable`
mode, so `run_training.py` sets that flag again after loading a checkpoint. The first update
after the graph is built matches the ordinary step to about four digits, and 20 iterations
took 48.7 s instead of 78.2 s.

`FactoredCategorical` skips argument validation (`validate_args=False`) because the checks
read tensors back to Python, which waits for the GPU and is not allowed while a graph is
being recorded.

## Measured effect

A profiled run of 4 iterations (4 environments, 512 steps each, 1 PyTorch
thread, CUDA available) went from about 7.6 s to about 4.8 s after moving GAE
and returns to the CPU and replacing `Categorical` sampling in the rollout.
These figures include one-off start-up costs (imports and CUDA initialization),
so a long run will differ. The remaining rollout time is mostly the network
forward passes and the Python simulation, which are CPU-bound.

## Things not yet measured

- Whether running the PPO update on the CPU is as fast as on the GPU for this
  network size. Try `TRAINING_DEVICE = "cpu"` and compare the iteration time.
  A larger network or a larger `batch_size` favors the GPU.
- Whether more parallel environments (`num_envs`) help. The simulation runs
  sequentially in one Python process, so more environments add CPU time per step
  even though they give larger batches.
