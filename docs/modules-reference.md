# Arena AI Modules Reference

This document describes the intended purpose of each module and project directory.

## Project configuration

- `pyproject.toml` — Defines project metadata, Python dependencies, and tool configuration.
- `README.md` — Provides the project overview, setup instructions, and usage guidance.
- `.gitignore` — Excludes generated and local-only content such as `data/`, `checkpoints/`, `runs/`, `.venv/`, and `__pycache__/`.

## Configuration

- `configs/` — Stores the YAML files read by the global configuration loader.
- `configs/game.yaml` — Defines gameplay and observation constants, including damage, cooldowns, parry timing, and arena dimensions.
- `configs/agent.yaml` — Defines neural-network settings for the agent.
- `src/arena_ai/config.py` — Loads all YAML files from `configs/` and exposes the typed game settings plus the agent settings.

## Core package

- `src/arena_ai/` — Contains the main Arena AI Python package.

### Simulation

The simulation layer is deterministic and independent of rendering. It must not import `pygame`.

- `src/arena_ai/sim/` — Provides pure game simulation logic.
- `src/arena_ai/sim/state.py` — Defines data structures such as `FighterState` and `GameState`.
- `src/arena_ai/sim/actions.py` — Defines action enums and the `MultiDiscrete` action-space layout.
- `src/arena_ai/sim/rules.py` — Implements state transitions, damage calculation, and collision handling.
- `src/arena_ai/sim/observation.py` — Builds observations from the current game state.
- `src/arena_ai/sim/rewards.py` — Defines reward functions used by training environments.

### Environments

- `src/arena_ai/envs/` — Contains reinforcement-learning environment adapters.
- `src/arena_ai/envs/arena_env.py` — Wraps the simulation for Gymnasium and/or PettingZoo interfaces.

### Agents

- `src/arena_ai/agents/` — Contains agent implementations and shared agent interfaces.
- `src/arena_ai/agents/base.py` — Defines common agent abstractions and interfaces.
- `src/arena_ai/agents/scripted.py` — Implements deterministic or rule-based scripted opponents.
- `src/arena_ai/agents/ppo.py` — Implements or integrates a Proximal Policy Optimization agent.
- `src/arena_ai/agents/clone.py` — Implements behavior cloning or policy cloning functionality.

### Training

- `src/arena_ai/training/` — Contains training workflows and supporting utilities.
- `src/arena_ai/training/selfplay.py` — Runs self-play training between agents.
- `src/arena_ai/training/finetune.py` — Fine-tunes an existing model.
- `src/arena_ai/training/curriculum.py` — Defines staged training difficulty or curriculum progression.
- `src/arena_ai/training/opponent_pool.py` — Manages the collection and selection of training opponents.

### Viewer

The viewer layer is the only layer intended to depend on `pygame`.

- `src/arena_ai/viewer/` — Provides rendering and interactive display functionality.
- `src/arena_ai/viewer/renderer.py` — Renders the arena, fighters, effects, and other visual elements.
- `src/arena_ai/viewer/human_input.py` — Translates keyboard or controller input into game actions.
- `src/arena_ai/viewer/ui.py` — Implements menus, overlays, status displays, and other interface elements.

### Utilities

- `src/arena_ai/utils/` — Contains shared supporting functionality for logging, random seeding, and replay recording.

## Assets

- `assets/` — Stores non-code resources used by the viewer.
- `assets/sprites/` — Stores fighter sprite sheets and visual effects.
- `assets/audio/` — Stores music and sound effects.
- `assets/fonts/` — Stores fonts used by the user interface.

## Scripts

- `scripts/` — Contains executable project entry points.
- `scripts/train.py` — Starts a training run.
- `scripts/play.py` — Starts an interactive play session.
- `scripts/evaluate.py` — Evaluates agents or saved models.

## Tests

- `tests/` — Contains automated tests organized to mirror the source package.
- `tests/test_rules.py` — Tests simulation rules, including damage and collision behavior.
- `tests/test_determinism.py` — Verifies reproducible simulation behavior when using the same seed and inputs.

## Documentation and experiments

- `notebooks/` — Contains exploratory analysis, experiments, and visualizations.
- `docs/` — Contains project specifications and reference documentation.
- `docs/spec.md` — Documents the observation, action, and reward specifications.
- `docs/modules-reference.md` — Describes the intended responsibility of each project module and directory.

## Generated and runtime data

- `data/` — Stores logged fights and other runtime-generated data; gitignored.
- `checkpoints/` — Stores saved model checkpoints; gitignored.
- `runs/` — Stores TensorBoard logs and other training-run outputs; gitignored.
