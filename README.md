# Arena AI

Arena AI is a Python 2D arena-fighting prototype with a deterministic fighter
simulation, reinforcement-learning components, and a pygame viewer.

The project is under active development. The simulation, manual viewer, and
initial headless PPO training workflow are available now; Gymnasium/PettingZoo
compatibility and more advanced training workflows remain future work.

## Current features

- Fixed-timestep fighter simulation with movement, jumping, attacks, parrying,
  counter attacks, health, damage, and hit detection.
- Configurable game, agent, and scoring values in [`configs/`](configs/).
- Observation encoding and discrete action definitions used by the learning agent.
- **Training works today:** hand-written PPO, rollout, checkpoint, and environment
  modules, run with `py scripts/run_training.py`. Choose the opponents of each parallel
  environment (one to three at once) with `training.opponents` in
  [`configs/agent.yaml`](configs/agent.yaml).
  See [Training](#training) and the [training guide](docs/training-process.md).
- Scripted opponents for the viewer and training, including one that hunts you and
  parries every attack it can see, and one that runs away from you.
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
- **Opponents** — how many opponents you face (none for solo practice, up to three;
  solo is not available while watching), then **Opponent 1** to **3**, one dropdown
  each: the trained agent, the scripted approach-and-attack fighter, a perfect-parry fighter (it parries every attack it can
  see coming, so hit it while it is busy swinging or recovering), an idle dummy, a fighter that runs away, or a random fighter. Opponents are allies and never hurt
  each other, and you win when all of them are dead. The trained agent needs a checkpoint, so run
  training first. It can play its best move or sample varied moves.
- **Checkpoint** — which file in `checkpoints/` the trained agent loads. Automatic
  uses `latest.pt` if it exists, otherwise the newest `.pt` file, so a checkpoint
  you renamed to keep it (for example `best.pt`) is found. The list shows up to 8
  files.
- **Random starting positions** and whether to show the controls hint.

Choices are saved to `data/viewer_settings.json` (gitignored) and restored on the
next launch. Text that is too long for its box, such as a long checkpoint name,
scrolls sideways by itself instead of running outside it. In a match, `Esc` opens a pause menu with a way back to Settings,
and the result screen offers to play again. The opponent uses the same animated
sprites as the player with a light-red tint.

A match ends when a fighter is defeated, or after `match.time_limit_seconds` in
[`configs/viewer.yaml`](configs/viewer.yaml) (60 by default; 0 removes the limit). When
time runs out the healthier fighter wins, and equal health is a draw.

Controls:

| Key                                | Action     |
| ---------------------------------- | ---------- |
| `A` or `Left`                      | Move left  |
| `D` or `Right`                     | Move right |
| `W`, `Up` or `Space`               | Jump       |
| Left mouse button                  | Attack     |
| `Q`                                | Parry      |
| `Esc` or `P`                       | Pause      |
| `F11`                              | Fullscreen (any screen) |

## Documentation

- [`docs/setup.md`](docs/setup.md) — install the project, with CPU and NVIDIA
  CUDA options, and run the viewer. Open it first on a new machine.
- [`docs/simulation-spec.md`](docs/simulation-spec.md) — the action set,
  observation layout, rewards and timing. Open it before changing the
  simulation or anything a checkpoint depends on.
- [`docs/training-process.md`](docs/training-process.md) — the training loop, PPO
  data flow, bootstrapping, resuming a run and the printed training statistics.
- [`docs/network.md`](docs/network.md) — diagrams of the neural network, how it
  decides, and how the PPO loss trains each part.
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

## Trained agent

The repository includes the best weights so far, `checkpoints/best.pt`, so the viewer's
trained opponent works straight after a clone with no training. Every other file in
`checkpoints/` is gitignored. These weights come from a run of 1,702 iterations (about
13.9 million steps). Over the whole run the agent won 97% of its 27,982 episodes. In the last 100 episodes it won
every one against the approach, idle and perfect-parry opponents and 99 against the
random one. It was trained before the runner opponent existed, so it has not been
measured against it. **Retrain it:** the observation now has opponent slots and a mask
(spec version 2) and the network has an attention input block, so this checkpoint no
longer loads and the viewer's trained agent needs a new one. Its figures above are for the
old network. `checkpoints/best.pt` is left as it was until a new run replaces it. The viewer picks it up automatically when there is no `latest.pt`,
or you can choose it in Settings. Replace `best.pt` only when a run is clearly better,
and update these figures when you do.

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
exits. If you press it during a PPO update, training finishes that update first, so the
saved weights always match the saved iteration. Pressing it a second time stops at once.
A checkpoint is also saved every `checkpoint_interval` iterations.

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
before the master effects volume is applied. Every fighter, including the
opponent, plays attack, jump, landing and running sounds, and a hit sound when it
takes damage. Each fighter's running audio fades on start, stop, and direction
changes, and is only active while that fighter is supported. The source audio
files are not modified by normalization or fades.

## Development

The project requires Python 3.10 or newer. The current correctness check is:

```powershell
py -m compileall src tests scripts
py -m unittest discover -s tests
```

The simulation is intentionally independent of pygame so it can be used
headlessly by the training environments. Keep pygame-dependent code inside
[`src/arena_ai/viewer/`](src/arena_ai/viewer/).

## License

No project license has been selected yet. The third-party sprite asset is
credited above and remains subject to its creator's license and terms.
