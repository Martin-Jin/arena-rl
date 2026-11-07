# Training process

Arena AI uses an on-policy PPO workflow. The current implementation trains one
learner in a headless environment against the environment's configured
opponent. The training loop is reusable: it receives an environment factory, a
policy adapter, an update callback, and an optional checkpoint callback.

## Configuration

Training defaults are loaded from [`configs/agent.yaml`](../configs/agent.yaml)
through [`load_config`](../src/arena_ai/config.py). The `training` section
defines:

- `iterations` — number of rollout-and-update cycles.
- `rollout_steps` — environment transitions collected in each cycle.
- `checkpoint_interval` — cycles between checkpoint callbacks.
- `seed` — optional seed for the initial environment reset.

An iteration is not an episode. A rollout collects exactly `rollout_steps`
transitions, resetting the environment whenever an episode terminates or is
truncated. A rollout can therefore contain multiple episodes or end in the
middle of an episode.

## Current loop

[`TrainingLoop.run`](../src/arena_ai/training/train.py) performs this sequence:

1. Reset the environment.
2. Collect one fixed-size rollout with the unchanged policy.
3. Pass the rollout to the update callback.
4. Record training metrics.
5. Save a checkpoint when the configured interval is reached.
6. Repeat for the configured number of iterations.

[`TrainingLoop.collect_rollout`](../src/arena_ai/training/train.py) runs policy
inference under `torch.no_grad()`. The acting interface is intentionally
separate from the PPO forward interface:

- `policy.act(observation)` returns a Python `int` action and Python `float`
  log-probability.
- `policy.value(observation)` returns a Python `float` critic estimate.
- The PPO update callback can use its own callable network interface returning
  `(logits, values)` for [`train_step`](../src/arena_ai/training/ppo.py).

The scalar boundary is checked before data enters
[`Rollout.add`](../src/arena_ai/training/rollout.py), where observations,
actions, rewards, terminal flags, values, and old log-probabilities are stored.
The rollout is converted to a [`RolloutBatch`](../src/arena_ai/training/rollout.py)
after collection and is discarded after the update.

## Bootstrapping and episode boundaries

Each transition stores `next_values[t]`, the critic value for the observation
reached after step `t`. This is evaluated before an environment reset, so a
truncated episode retains the value of its final observation rather than the
initial observation of the next episode.

The update callback receives:

- `last_value` — the value of the observation carried into the next rollout.
- `terminated` — natural terminal flags. These prevent value bootstrapping.
- `episode_ends` — natural terminals or time-limit truncations. These stop
  recursive GAE accumulation across episode boundaries.
- `next_values` — per-transition values used when a truncation requires a final
  observation bootstrap.

The callback should pass these values to
[`advantage_estimation`](../src/arena_ai/training/ppo.py) and use the matching
episode-boundary arguments when calculating discounted returns. This distinction
allows a time-limit truncation to bootstrap from its final observation while
still preventing advantages from leaking into the reset episode.

## Environment and rewards

[`ArenaEnv.step`](../src/arena_ai/envs/arena_env.py) returns the next
observation, reward, `terminated`, `truncated`, and an info mapping. The
environment adds `info["winner"]` at natural episode termination:

- `"agent"` when the learner survives and an opponent is defeated.
- `"opponent"` when the learner is defeated.
- `None` for non-terminal steps and truncations.

The loop accumulates rewards for the current episode. When an episode ends, it
records the completed return and increments the episode count. It increments
the win count only when `info["winner"] == "agent"`. The metrics exposed after
each update include:

- `episode_return` — the most recently completed episode return.
- `win_rate` — cumulative learner wins divided by completed episodes.
- `episodes` and `wins` — cumulative counts.
- PPO loss metrics returned by the update callback.

An iteration may end mid-episode, so an incomplete episode is not counted until
the environment reports termination or truncation.

## PPO update boundary

PPO is on-policy. The batch collected before an update belongs to the old
policy and must not be reused indefinitely after the policy changes. The
intended update callback computes advantages and returns, runs the configured
PPO epochs/minibatches, and returns scalar metrics. The next iteration then
creates a fresh [`Rollout`](../src/arena_ai/training/rollout.py).

The current repository exposes the reusable loop and PPO building blocks;
[`scripts/run_training.py`](../scripts/run_training.py) now wires those components into a
single runnable workflow. It creates the actor-critic model and optimizer,
constructs the one-versus-one environment, computes PPO advantages and
returns, performs minibatch updates, and saves `checkpoints/latest.pt` at the
configured interval. Run it from the repository root with:

```powershell
py scripts/run_training.py
```

The script uses the values in `configs/agent.yaml`; reduce `training.iterations`
and `training.rollout_steps` there for a short smoke run.
