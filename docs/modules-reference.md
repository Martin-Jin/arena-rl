# Arena AI Modules Reference

This document describes the purpose of each module and project directory.

## Project configuration

- `pyproject.toml` — Defines project metadata, Python dependencies, and tool configuration.
- `README.md` — Provides the project overview, setup instructions, and usage guidance.
- `.gitignore` — Excludes generated and local-only content such as `data/`, `checkpoints/`, `runs/`, `.venv/`, and `__pycache__/`.

## Configuration

- `configs/` — Stores the YAML files read by the global configuration loader.
- `configs/game.yaml` — Defines gameplay and observation constants, including
  attack timing and damage, fighter dimensions, arena dimensions and the
  observation scaling limits, and the parry and counter timings. The observation
  `spec_version` is checked when a checkpoint loads. `reaction_delay_frames` is loaded
  but has no effect: observations are never delayed.
- `configs/agent.yaml` — Defines neural-network settings, PPO hyperparameters
  and training-loop defaults, including hidden-layer sizes, rollout length,
  checkpoint interval, number of parallel environments and spawn
  randomization.
- `configs/scoring.yaml` — Defines reward scales, terminal win/loss rewards and
  the wall-proximity penalty used by training environments.
- `configs/viewer.yaml` — Defines viewer display, animation timing, audio,
  health UI, end-screen, and asset settings.
- `src/arena_ai/config.py` — Loads all YAML files from `configs/` into one typed
  `Config` object (game, agent, training, scoring and viewer settings).

## Core package

- `src/arena_ai/` — Contains the main Arena AI Python package.

### Simulation

The simulation layer is deterministic and independent of rendering. It must not import `pygame`.

- `src/arena_ai/sim/` — Provides pure game simulation logic.
- `src/arena_ai/sim/state.py` — Defines `FighterState`, the fighter ids and teams (`AGENT_ID`,
  `opponent_id`, `team_of`, `enemies_of`, the natural `sort_key`) and the starting positions.
- `src/arena_ai/sim/simulate.py` — Advances independently controlled fighters by one timestep using ID-keyed actions.
- `src/arena_ai/sim/observation.py` — Builds nested observations with one agent and a keyed collection of opponents.
- `src/arena_ai/sim/observation_vector.py` — Flattens observations into a
  fixed ordered and normalized vector whose layout matches the network input.
  It exposes the derived observation size and reads by feature name
  (`FEATURE_NAMES`, `fighter_features`, `opponent_mask`), so nothing indexes a vector by
  number. The layout is the acting fighter, `max_opponents` padded opponent slots and a
  mask. Each fighter contributes fourteen features, scaled using the game
  configuration's fixed observation limits. See
  [`docs/simulation-spec.md`](simulation-spec.md) for the exact order.
- `src/arena_ai/sim/collision.py` — Keeps fighters' body boxes from overlapping; a counter's
  lunge passes through.
- `src/arena_ai/sim/outcome.py` — When an episode or match is over and who won it
  (agent dead or every opponent dead; the time-out health rule).
- `src/arena_ai/sim/rules.py` — Implements one fighter's step: movement, attack and
  parry timing, the spam-shrunk parry window, the counter lunge and hover.

### Environments

- `src/arena_ai/envs/` — Contains reinforcement-learning environment adapters.
- `src/arena_ai/envs/arena_env.py` — Defines the shared abstract environment
  lifecycle: reset, step, observations, episode limits, and terminal status.
  It must remain headless and must not import `pygame`.
- `src/arena_ai/envs/reward.py` — Calculates the learner's per-step reward
  (damage, terminal outcomes, and wall-proximity penalty; the victory bonus needs every
  opponent dead) as a pure function of
  fighter state and scoring config.
- `src/arena_ai/envs/vector_env.py` — Runs independent headless environments
  synchronously and auto-resets completed instances for batched collection.
- `src/arena_ai/envs/async_vector_env.py` — `AsyncVectorEnv`: the same interface, with the
  environments stepped in worker processes (`training.env_workers`).
- `src/arena_ai/envs/training_envs/` — Contains concrete environments for
  training and scenario testing.
- `src/arena_ai/envs/training_envs/one_v_many_env.py` — `OneVManyEnv`: one learning
  agent against one or more allied scripted opponents (up to `max_opponents`), with its
  initial fighters (optionally randomized spawn positions), each opponent's view and
  the matchup label reported in `info`.
- `src/arena_ai/envs/training_envs/one_v_one_env.py` — `OneVOneEnv`, `OneVManyEnv` with
  a single opponent and the signature the one-on-one tests use.

### Agents

- `src/arena_ai/agents/` — Contains agent implementations and shared agent interfaces.
- `src/arena_ai/agents/action.py` — Defines the fixed discrete action encoding
  (movement, jump, and attack combinations), derives its action count, converts
  a policy index into a `FighterAction`, and maps input parts back to an index.
  It also defines the factored move/jump/attack head layout used by the
  network.
- `src/arena_ai/agents/distribution.py` — Implements `FactoredCategorical`,
  independent move, jump and attack distributions that sample, score and
  report the entropy of a flat action index. `sample_with_log_prob` is a
  faster no-gradient version of sampling plus `log_prob` used for rollouts.
- `src/arena_ai/agents/scripted/` — The non-learning opponents, one module each:
  `approach.py` (deterministic approach-and-attack), `parrier.py` (the perfect-parry
  hunter: it parries every attack it can see from the front, so mix it with easier
  opponents in training), `dodger.py` (approach-and-attack that rolls through every swing
  it can see, until its stamina runs out), `runner.py` (flees the learner), `simple.py` (idle and uniformly
  random), `registry.py` (the `OPPONENTS` registry, `SELF_PLAY` and `training_opponents`, which checks a
  `training.opponents` list), `episode.py` (`EpisodePolicy`, which redraws an opponent's
  options each training episode), `view.py` (how a policy reads its observation by name, and
  `team_observation`, which builds it: the agent, the opponent, then its allies, from
  `fighter_rows` so each fighter is encoded once per step) and
  `teamwork.py` (`jumps_over_allies`, which lets opponents queued behind an ally hop over
  it).
- `src/arena_ai/agents/selfplay.py` — `SelfPlayPolicy`, the `agent` training opponent: a copy
  of the learner's network that plays an opponent from its own side. It keeps one network per
  process, loaded from a flat weight vector (`weights_of`, `load_weights`) that training sends
  after every update. `make_opponent_policy` builds the policy for a training opponent name.
  Torch is imported only when it is used.
- `src/arena_ai/agents/attention.py` — `EntityAttention`, the input block of both
  networks: it encodes the acting fighter and each opponent, attends over the opponents
  with a mask and returns `[e_self ; context]`.
- `src/arena_ai/agents/policy.py` — Defines the PyTorch `ActorCritic` module
  with separate, configurable policy and value networks (an attention block, hidden
  layers and an output layer each, with a `signature` that checkpoints must match), and `PolicyAdapter`, which
  samples actions and estimates values for rollouts from a CPU copy of the
  network that is synchronized after each update.
- `src/arena_ai/agents/stacked_actor.py` — `StackedActor`: the inference-only copy that runs
  the policy and value networks together in one pass, built from the `ActorCritic` weights.

### Training

- `src/arena_ai/training/` — Contains the learner-versus-scripted-opponent
  workflow. It has no self-play, fine-tuning, curriculum or opponent-pool workflow.
- `src/arena_ai/training/rollout.py` — Stores observations, actions, rewards,
  termination flags, and policy values/log probabilities collected from rollouts.
- `src/arena_ai/training/ppo.py` — Calculates GAE and bootstrapped critic
  targets, then calculates PPO policy, value, and entropy losses and applies
  optimizer updates. `GraphedTrainStep` replays one PPO minibatch step as a CUDA graph.
- `src/arena_ai/training/stats.py` — Cheap training statistics: rolling episode
  statistics overall and per opponent (running sums, so O(1) to update) and
  behaviour statistics for a rollout batch (action shares, reward per step).
- `src/arena_ai/training/report.py` — Formats the latest statistics for printing.
- `src/arena_ai/training/checkpoints.py` — Saves and restores the policy,
  optimizer, configuration, and training progress.
- `src/arena_ai/training/train.py` — Owns rollout collection across episodes,
  updates, checkpoint callbacks, and training metrics. See
  [`docs/training-process.md`](training-process.md) for the complete current
  process and data flow.
- `scripts/run_training.py` — Wires the model, optimizer, environments (one per
  entry of `training.opponents`), PPO
  update callback, and checkpoint callback into the runnable training command.
  It also restores a saved run when `RESUME_FROM` is set.

### Viewer

The viewer layer is the only layer that depends on `pygame`.

- `src/arena_ai/viewer/` — Provides rendering and interactive display functionality.
- `src/arena_ai/viewer/game.py` — Entry point (`py -m arena_ai.viewer.game`); opens the app.
- `src/arena_ai/viewer/app.py` — `App` owns the window, audio, settings and a
  stack of scenes, and runs the main loop.
- `src/arena_ai/viewer/settings.py` — The settings the menu edits (audio, player, the
  opponent count and the kind of each opponent slot, spawns, randomized opponent behavior,
  hints), their labels, and
  loading/saving them to `data/viewer_settings.json`. A saved single `opponent` key from an
  older version is moved to the first slot. The player can also be a scripted opponent kind
  (to watch it fight); that forces a one-on-one.
- `src/arena_ai/viewer/matchup.py` — Names the fighters of a match ("Scripted: approach #2")
  and writes the matchup line of the main menu.
- `src/arena_ai/viewer/controllers.py` — AI controllers for scripted and trained
  fighters, one per fighter id (a trained one sees only its enemies; a scripted one on the
  agent's side sees the two fighters swapped, so it works only one-on-one; a scripted one is
  an `EpisodePolicy` that redraws its behavior each round when randomizing is on), and the
  discovery of checkpoints in the checkpoint
  folder (`latest.pt` first, then the newest). Torch is imported only when the
  trained agent is used.
- `src/arena_ai/viewer/audio.py` — Loads configured music and sound effects
  and plays them from viewer state transitions. Effects are normalized in
  memory using configurable RMS and peak targets. It plays sounds for every
  fighter, including the opponent: attack, dodge, jump, landing, running, a hit
  sound when a fighter loses health and a death sound when it dies. It reserves two mixer channels per fighter
  (`max_opponents + 1` fighters) and refuses a count the mixer cannot cover. Each fighter has its own running loop,
  which fades on start, stop, and direction changes and is gated by support
  state. Volumes can be changed while running.
- `src/arena_ai/viewer/training.py` — Renders accelerated training simulation
  steps when training visualization is enabled.
- `src/arena_ai/viewer/scenes/` — Full-screen scenes sharing a small `Scene`
  interface: `main_menu.py` (start menu), `settings_menu.py` and `match.py` (intro,
  fight, pause and result screens).
- `src/arena_ai/viewer/ui/` — Reusable UI pieces: `theme.py` (colors, fonts,
  text and panel drawing, and `fit_text`, which scrolls text that is too wide for
  its box), `widgets.py` (buttons, sliders, toggles, selectors, dropdowns and the
  focus-handling `Menu`) and `hud.py` (health bars, swing and stamina meters and damage
  numbers).
- `src/arena_ai/viewer/render/` — Drawing helpers: `layout.py` (meters to pixels),
  `animation.py` (sprite animations), `trail.py` (the dodge's afterimages), `fade.py` (`DeathFade`, which times how long a dead fighter has been fading) and `scene.py` (arena, shadows and
  fighters, shared by the menus, matches and the training view).

## Assets

- `assets/` — Stores non-code resources used by the viewer.
- `assets/sprites/` — Stores fighter sprite sheets and visual effects.
- `assets/sprites/fighter/` — Stores ordered idle, run, attack, and turn frames
  extracted from the credited Knight Hero Platformer sprite pack. The viewer
  scales the source cells to the 0.9 m fighter hitbox. The attack action uses
  only the pack's single-slash `Attack 1` sequence, since the game has no combo actions.
- `assets/sprites/fighter/deferred/` — Preserves source sheets and extracted
  combo/recovery frames that the viewer does not load, since no action uses them.
- `assets/audio/` — Stores music and sound effects.
- `assets/audio/effects/` — Stores one-shot and looping fighter sound effects
  for attacks, movement, jumping, landing, and hits.
- `assets/audio/music/` — Stores looping battle music.
- `assets/fonts/` — Stores fonts used by the user interface.

## Entry points

- `src/arena_ai/viewer/game.py` — Interactive pygame viewer with menus.
- `scripts/run_training.py` — Runs headless PPO training.
- `scripts/evaluate_agent.py` — Plays a checkpoint against each scripted opponent and prints
  win count, remaining health and fight length.

## Tests

- `tests/` — Automated `unittest` tests. Run them with
  `py -m unittest discover -s tests`.
- The simulation tests include deterministic same-seed runs and the parry, counter
  and stagger rules. Add a test with every rule change.

## Documentation and experiments

- `docs/` — Contains project specifications and reference documentation.
- `docs/modules-reference.md` — Describes the responsibility of each project module and directory.
- `docs/simulation-spec.md` — Documents the action set, observation layout, rewards and timing.
- `docs/training-process.md` — Documents the training loop and PPO data flow.
- `docs/setup.md` — Documents installation.
- `docs/hardware.md` — Documents which training steps run on the CPU or GPU and why.

## Generated and runtime data

- `data/` — Stores logged fights and other runtime-generated data; gitignored.
- `checkpoints/` — Stores saved model checkpoints. Gitignored, except `best.pt`, the best
  trained weights, which is committed so the viewer works after a clone.
- `runs/` — Stores TensorBoard logs and other training-run outputs; gitignored.
