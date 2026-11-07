# Arena AI

Arena AI is a Python 2D arena-fighting prototype with a deterministic fighter
simulation, reinforcement-learning components, and a pygame viewer.

The project is under active development. The simulation, manual viewer, and
initial headless PPO training workflow are available now; Gymnasium/PettingZoo
compatibility and more advanced training workflows remain future work.

## Current features

- Fixed-timestep fighter simulation with movement, jumping, attacks, health,
  damage, and hit detection.
- Configurable game, agent, and scoring values in [`configs/`](configs/).
- Observation encoding and discrete action definitions for future agents.
- Initial agent, rollout, PPO, checkpoint, and environment modules.
- Pygame viewer with configurable idle, running, turning, single-slash attack,
  and airborne animation fallback, plus battle music and fighter sound effects.
- Premade Knight Hero Platformer sprite assets, scaled to the configured
  fighter hitbox.

## Run the viewer

Follow the [setup guide](docs/setup.md) to install the project and its
dependencies.

The setup guide includes separate CPU and NVIDIA CUDA installation commands.
Use the CUDA command before running training if you want PyTorch to use your
GPU; the training script automatically selects CUDA when it is available.

Start the viewer from the repository root:

```powershell
py -m arena_ai.viewer.game
```

The viewer opens on a start menu with **Play**, **Settings** and **Quit**. Use the
mouse, or the arrow keys with `Enter`. The settings menu has:

- **Audio** — music and sound-effect volume.
- **Player** — you, or the trained agent (the start button then reads **Watch**
  and you spectate the agent fight the chosen opponent).
- **Opponent** — a dropdown with the trained agent, the scripted
  approach-and-attack fighter, an idle dummy, a random fighter, or solo practice
  (not available while watching). The trained agent needs
  `checkpoints/latest.pt`, so run training first. It can play its best move or
  sample varied moves.
- **Random starting positions** and whether to show the controls hint.

Choices are saved to `data/viewer_settings.json` (gitignored) and restored on the
next launch. In a match, `Esc` opens a pause menu with a way back to Settings,
and the result screen offers to play again. The opponent uses the same animated
sprites as the player with a light-red tint.

Controls:

| Key                                | Action     |
| ---------------------------------- | ---------- |
| `A` or `Left`                      | Move left  |
| `D` or `Right`                     | Move right |
| `W` or `Up`                        | Jump       |
| `Space` or left mouse button       | Attack     |
| `Esc` or `P`                       | Pause      |

## Documentation

- [`docs/setup.md`](docs/setup.md) — install the project, with CPU and NVIDIA
  CUDA options, and run the viewer. Open it first on a new machine.
- [`docs/simulation-spec.md`](docs/simulation-spec.md) — the action set,
  observation layout, rewards and timing. Open it before changing the
  simulation or anything a checkpoint depends on.
- [`docs/training-process.md`](docs/training-process.md) — the training loop, PPO
  data flow, bootstrapping, resuming a run and the printed training statistics.
- [`docs/hardware.md`](docs/hardware.md) — which training steps run on the CPU or
  GPU, and why.
- [`docs/modules-reference.md`](docs/modules-reference.md) — what every module and
  folder is for. Open it to find where code belongs.

## Project layout

- [`src/arena_ai/sim/`](src/arena_ai/sim/) — deterministic simulation,
  fighter state, rules, observations, and stepping.
- [`src/arena_ai/viewer/`](src/arena_ai/viewer/) — pygame rendering, input,
  and animation loading.
- [`scripts/`](scripts/) — runnable entry points (currently training).
- [`tests/`](tests/) — automated tests.
- [`configs/viewer.yaml`](configs/viewer.yaml) — viewer display, animation,
  audio volume, normalization, and fade settings.
- [`src/arena_ai/agents/`](src/arena_ai/agents/) — agent and policy modules.
- [`src/arena_ai/envs/`](src/arena_ai/envs/) — environment wrappers.
- [`src/arena_ai/training/`](src/arena_ai/training/) — rollout, PPO,
  checkpoint, and training-loop modules. Rewards live in
  [`src/arena_ai/envs/reward.py`](src/arena_ai/envs/reward.py).
- [`assets/sprites/fighter/`](assets/sprites/fighter/) — active animation
  frames.
- [`assets/sprites/fighter/deferred/`](assets/sprites/fighter/deferred/) —
  preserved source sheets and animations not yet connected to game actions.
- [`docs/`](docs/) — project documentation, indexed below.

## Training

See the [training process guide](docs/training-process.md) for the current
rollout/update loop, PPO bootstrapping and episode-boundary handling, policy
interfaces, and progress metrics.

Run training from the repository root with:

```powershell
py scripts/run_training.py
```

Press `Ctrl+C` to stop training. The latest in-memory model, optimizer, and
completed training state are saved to `checkpoints/latest.pt` before the script
exits. A checkpoint is also saved every `checkpoint_interval` iterations.

Training prints a statistics block every `training.print_interval` iterations
(win rate, approximate KL, clip fraction, explained variance, per-head entropy,
action shares, combat figures and timing). Set `training.print_stats: false` in
[`configs/agent.yaml`](configs/agent.yaml) for silent training. See
[Training statistics](docs/training-process.md#training-statistics).

To continue a stopped run, set `RESUME_FROM = "checkpoints/latest.pt"` near the
top of [`scripts/run_training.py`](scripts/run_training.py). It restores the
model, optimizer and progress counters and runs `training.iterations` more
iterations on top of the saved ones. Leave it as `None` to start a new run. Each
run also keeps numbered checkpoints in `checkpoints/run-<timestamp>/`, so a new
run overwrites `latest.pt` but never an earlier run's history. See
[Resuming a run](docs/training-process.md#resuming-a-run).

Training automatically uses CUDA when PyTorch detects a CUDA-capable GPU and
falls back to the CPU otherwise. Set `TRAINING_DEVICE = "cpu"` or
`TRAINING_DEVICE = "cuda"` in [`scripts/run_training.py`](scripts/run_training.py)
to override the default. PPO tensors and optimizer updates run on the selected
device. Per-step action selection and value estimates during rollouts run on a
CPU copy of the network that is synchronized after each update, because the
network is small and per-step GPU transfers would be slower. The deterministic
game simulation and pygame visualization are also CPU-bound. See the
[hardware guide](docs/hardware.md) for what runs where and why.

To visualize training, set `VISUALIZE_TRAINING = True` near the top of
[`scripts/run_training.py`](scripts/run_training.py). The viewer renders every
fixed simulation step during every rollout without adding a real-time delay, so
training runs at the simulation stepping rate. Close the window to stop the
training run.

## Animation assets

The active attack animation uses only the single-slash `Attack 1` sequence.
Combo and recovery animations are preserved for later mechanics but are not
played because the game does not have combo actions yet.

Animation credit: [Knight Hero Platformer by
Pixivan](https://pixivan.itch.io/knight-hero-platfor).

## Audio assets

Sound effects in [`assets/audio/effects/`](assets/audio/effects/) are royalty
free and do not require attribution. The battle music in
[`assets/audio/music/`](assets/audio/music/) is **“Senkae” by Hideyuki
Fukasawa**, from the *TSUKIHIME -A piece of blue glass moon- Original
Soundtrack*. Attribution is required. Source:
[official YouTube Music track](https://www.youtube.com/watch?v=qpElSu-nRRk).

Sound effects are normalized in memory using configurable RMS and peak targets
before the master effects volume is applied. Running audio fades on start,
stop, and direction changes, and is only active while the fighter is supported.
The source audio files are not modified by normalization or fades.

## Development

The project requires Python 3.10 or newer. The current correctness check is:

```powershell
py -m compileall src tests scripts
py -m unittest discover -s tests
```

The simulation is intentionally independent of pygame so it can be used
headlessly by future training environments. Keep pygame-dependent code inside
[`src/arena_ai/viewer/`](src/arena_ai/viewer/).

## License

No project license has been selected yet. The third-party sprite asset is
credited above and remains subject to its creator's license and terms.
