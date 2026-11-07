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

Start the manual viewer from the repository root:

```powershell
py -m arena_ai.viewer.game
```

To play against the trained agent, first run training so
`checkpoints/latest.pt` exists, then set `PLAY_AGAINST_TRAINED_AGENT = True`
near the top of [`src/arena_ai/viewer/game.py`](src/arena_ai/viewer/game.py).
Leave `VIEW_OPPONENT = True`. The trained opponent uses greedy action
selection from the latest checkpoint.

The viewer starts with a scripted opponent enabled. To run the single-fighter
practice mode instead, set `VIEW_OPPONENT = False` near the top of
[`src/arena_ai/viewer/game.py`](src/arena_ai/viewer/game.py). The opponent uses
the same animated sprite frames as the learner with a light-red tint.

Controls:

| Key | Action |
|---|---|
| `A` | Move left |
| `D` | Move right |
| `W` | Jump |
| `Space` | Attack |

## Project layout

- [`src/arena_ai/sim/`](src/arena_ai/sim/) — deterministic simulation,
  fighter state, rules, observations, and stepping.
- [`src/arena_ai/viewer/`](src/arena_ai/viewer/) — pygame rendering, input,
  and animation loading.
- [`configs/viewer.yaml`](configs/viewer.yaml) — viewer display, animation,
  audio volume, normalization, and fade settings.
- [`src/arena_ai/agents/`](src/arena_ai/agents/) — agent and policy modules.
- [`src/arena_ai/envs/`](src/arena_ai/envs/) — environment wrappers.
- [`src/arena_ai/training/`](src/arena_ai/training/) — rollout, PPO,
  checkpoint, reward, and training modules.
- [`docs/training-process.md`](docs/training-process.md) — current training
  loop, PPO data flow, bootstrapping, and training metrics.
- [`assets/sprites/fighter/`](assets/sprites/fighter/) — active animation
  frames.
- [`assets/sprites/fighter/deferred/`](assets/sprites/fighter/deferred/) —
  preserved source sheets and animations not yet connected to game actions.
- [`docs/`](docs/) — module and design reference documentation.

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
exits.

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
py -m compileall assets src tests
```

The simulation is intentionally independent of pygame so it can be used
headlessly by future training environments. Keep pygame-dependent code inside
[`src/arena_ai/viewer/`](src/arena_ai/viewer/).

## License

No project license has been selected yet. The third-party sprite asset is
credited above and remains subject to its creator's license and terms.
