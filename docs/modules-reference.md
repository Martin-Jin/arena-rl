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
- `src/arena_ai/config.py` — Loads all YAML files from `configs/` and exposes the typed game settings plus the agent settings.

## Core package

- `src/arena_ai/` — Contains the main Arena AI Python package.

### Simulation

The simulation layer is deterministic and independent of rendering. It must not import `pygame`.

- `src/arena_ai/sim/` — Provides pure game simulation logic.
- `src/arena_ai/sim/state.py` — Defines data structures such as `FighterState` and `GameState`.
- `src/arena_ai/sim/simulate.py` — Advances the headless simulation by one timestep for training and evaluation.
- `src/arena_ai/sim/observation.py` — Builds nested observations with one agent and a keyed collection of opponents.
- `src/arena_ai/sim/rules.py` — Implements state transitions, damage calculation, and collision handling.

### Environments

- `src/arena_ai/envs/` — Contains reinforcement-learning environment adapters.
- `src/arena_ai/envs/arena_env.py` — Wraps the simulation for Gymnasium and/or PettingZoo interfaces.

### Agents

- `src/arena_ai/agents/` — Contains agent implementations and shared agent interfaces.
- `src/arena_ai/agents/network.py` — Defines the current PyTorch network and its configurable layer structure.
- `src/arena_ai/agents/training.py` — Reserved for the agent training workflow; implementation is not yet complete.

### Training

- `src/arena_ai/training/` — Reserved for self-play, fine-tuning, curriculum, and opponent-pool workflows.

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
