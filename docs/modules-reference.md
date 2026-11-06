# Arena AI Modules Reference

This document describes the intended purpose of each module and project directory.

## Project configuration

- `pyproject.toml` — Defines project metadata, Python dependencies, and tool configuration.
- `README.md` — Provides the project overview, setup instructions, and usage guidance.
- `.gitignore` — Excludes generated and local-only content such as `data/`, `checkpoints/`, `runs/`, `.venv/`, and `__pycache__/`.

## Configuration

- `configs/` — Stores the YAML files read by the global configuration loader.
- `configs/game.yaml` — Defines gameplay and observation constants, including damage, cooldowns, parry timing, and arena dimensions.
- `configs/agent.yaml` — Defines neural-network settings for the agent, including input/output sizes and the size of each hidden layer.
- `configs/scoring.yaml` — Defines reward scales and terminal win/loss rewards
  used by training environments.
- `src/arena_ai/config.py` — Loads all YAML files from `configs/` and exposes the typed game settings plus the agent settings.

## Core package

- `src/arena_ai/` — Contains the main Arena AI Python package.

### Simulation

The simulation layer is deterministic and independent of rendering. It must not import `pygame`.

- `src/arena_ai/sim/` — Provides pure game simulation logic.
- `src/arena_ai/sim/state.py` — Defines data structures such as `FighterState` and `GameState`.
- `src/arena_ai/sim/simulate.py` — Advances independently controlled fighters by one timestep using ID-keyed actions.
- `src/arena_ai/sim/observation.py` — Builds nested observations with one agent and a keyed collection of opponents.
- `src/arena_ai/sim/observation_vector.py` — Flattens observations into a
  fixed ordered and normalized vector whose layout matches the network input.
  It exposes the derived observation size so the network does not duplicate it
  in configuration. Position, velocity, and health features are scaled using
  the game configuration's fixed observation limits. Each fighter also
  contributes grounded and facing-direction features.
- `src/arena_ai/sim/rules.py` — Implements state transitions, damage calculation, and collision handling.

### Environments

- `src/arena_ai/envs/` — Contains reinforcement-learning environment adapters.
- `src/arena_ai/envs/arena_env.py` — Defines the shared abstract environment
  lifecycle: reset, step, observations, episode limits, and terminal status.
  It must remain headless and must not import `pygame`.
- `src/arena_ai/envs/training_envs/` — Contains concrete environments for
  training and scenario testing.
- `src/arena_ai/envs/training_envs/one_v_one_env.py` — Implements the current
  one-learning-agent versus one-opponent scenario, including its initial
  fighters and opponent policy.

### Agents

- `src/arena_ai/agents/` — Contains agent implementations and shared agent interfaces.
- `src/arena_ai/agents/action.py` — Defines the fixed discrete action encoding,
  including `ATTACK`, derives its action count, and converts a policy index
  into a `FighterAction`.
- `src/arena_ai/agents/scripted.py` — Controls the non-learning opponent with a
  deterministic baseline policy.
- `src/arena_ai/agents/network.py` — Defines the current PyTorch network and its
  configurable hidden-layer structure. Input and output sizes are derived from
  the observation and action codecs.
- `src/arena_ai/agents/policy.py` — Converts fixed observation vectors into
  action probabilities, samples actions during training, and selects greedy
  actions during evaluation.
- `src/arena_ai/agents/training.py` — Reserved for the agent training workflow; implementation is not yet complete.

### Training

- `src/arena_ai/training/` — Contains the first learner-versus-scripted-opponent
  workflow and later self-play, fine-tuning, curriculum, and opponent-pool
  workflows.
- `src/arena_ai/training/rewards.py` — Defines terminal and shaping rewards
  from the learner's perspective.
- `src/arena_ai/training/rollout.py` — Stores observations, actions, rewards,
  termination flags, and policy values/log probabilities collected from episodes.
- `src/arena_ai/training/ppo.py` — Calculates PPO policy, value, and entropy
  losses and applies optimizer updates.
- `src/arena_ai/training/checkpoints.py` — Saves and restores the policy,
  optimizer, configuration, and training progress.
- `src/arena_ai/training/train.py` — Owns episode creation, scripted-opponent
  selection, rollout collection, updates, checkpointing, and evaluation.

### Viewer

The viewer layer is the only layer intended to depend on `pygame`.

- `src/arena_ai/viewer/` — Provides rendering and interactive display functionality.
- `src/arena_ai/viewer/game.py` — Runs the interactive pygame game loop.
- The renderer, input, and UI modules are planned as the viewer grows.

### Utilities

- `src/arena_ai/utils/` — Contains shared supporting functionality for logging, random seeding, and replay recording.

## Assets

- `assets/` — Stores non-code resources used by the viewer.
- `assets/sprites/` — Stores fighter sprite sheets and visual effects.
- `assets/sprites/fighter/` — Stores ordered idle, run, attack, and turn frames
  extracted from the credited Knight Hero Platformer sprite pack. The viewer
  scales the source cells to the 0.9 m fighter hitbox. The attack action uses
  only the pack's single-slash `Attack 1` sequence until combo mechanics exist.
- `assets/sprites/fighter/deferred/` — Preserves source sheets and extracted
  combo/recovery frames that are not loaded until their actions exist.
- `assets/audio/` — Stores music and sound effects.
- `assets/fonts/` — Stores fonts used by the user interface.

## Entry points

- `src/arena_ai/viewer/game.py` — Current interactive pygame game loop.
- Command-line scripts for training and evaluation are planned under `scripts/`.

## Tests

- `tests/` — Reserved for automated tests organized to mirror the source package.
- Determinism and simulation-rule tests are planned as the simulation grows.

## Documentation and experiments

- `notebooks/` — Contains exploratory analysis, experiments, and visualizations.
- `docs/` — Contains project specifications and reference documentation.
- `docs/modules-reference.md` — Describes the current and planned responsibility of each project module and directory.

## Generated and runtime data

- `data/` — Stores logged fights and other runtime-generated data; gitignored.
- `checkpoints/` — Stores saved model checkpoints; gitignored.
- `runs/` — Stores TensorBoard logs and other training-run outputs; gitignored.
