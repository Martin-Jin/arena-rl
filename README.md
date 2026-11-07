# Arena AI

Arena AI is a Python 2D arena-fighting prototype with a deterministic fighter
simulation, reinforcement-learning components, and a pygame viewer.

The project is under active development. The simulation and manual viewer are
available now; Gymnasium/PettingZoo integration and full training workflows
are still being built.

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
- [`assets/sprites/fighter/`](assets/sprites/fighter/) — active animation
  frames.
- [`assets/sprites/fighter/deferred/`](assets/sprites/fighter/deferred/) —
  preserved source sheets and animations not yet connected to game actions.
- [`docs/`](docs/) — module and design reference documentation.

## PPO training cycle

PPO training alternates simulation and optimization; they do not run at the
same time. One PPO iteration follows this cycle:

1. **Collect:** run the simulation with the current (old) policy for `N`
   timesteps. Store the observations/states, actions, rewards, termination
   flags, and the old policy's log-probabilities. Do not change model weights
   during collection.
2. **Compute:** evaluate the critic on the stored states, then calculate
   generalized advantage estimates (GAE) and the critic's target returns.
3. **Train:** perform several epochs of minibatch updates on that stored
   rollout. Update the actor with the clipped surrogate objective and entropy
   bonus, and update the critic with its value loss.
4. **Discard:** discard the rollout after those updates and return to step 1
   with the updated policy.

The rollout is on-policy: it was collected by the old policy and is only
useful for a limited number of update epochs before the policy has drifted too
far. PPO therefore collects fresh data for every iteration instead of
reusing an old batch indefinitely.

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
