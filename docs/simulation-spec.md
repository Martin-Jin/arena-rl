# Simulation specification

This document records the action set, observation layout, rewards and timing
that the simulation and training code share. Changing any of them invalidates
saved checkpoints, so update this file, the tests and the affected agents in the
same change. Numeric values come from [`configs/`](../configs/) and are quoted
here as the current defaults.

## Timing

- The simulation uses a fixed timestep of `1 / sim_hz` seconds (60 Hz by
  default). Rendering speed never changes simulation results.
- One call to `step_game` advances every fighter by one step, then resolves
  hits.
- A training episode is truncated after `max_episode_steps` steps unless the agent
  or every opponent reaches 0 health first (see Teams and termination). `training.max_episode_steps` in
  `agent.yaml` sets it (1500 steps, 25 s); if unset, the environment defaults to
  `sim_hz * 10` steps.

## Actions

Each step a fighter takes one of 24 discrete actions, defined in
[`agents/action.py`](../src/arena_ai/agents/action.py). The index is a
mixed-radix combination of four independent choices:

```
index = move + 3 * jump + 6 * attack + 12 * parry
move: 0 = none, 1 = left, 2 = right    jump, attack, parry: 0 or 1
```

The policy has four output heads (3 + 2 + 2 + 2 = 9 logits) and samples each head
independently. Append new actions only; reordering invalidates checkpoints.
Indices 0-11 are the pre-parry actions, so adding parry did not renumber them.
Attack and parry cannot overlap (see Combat), so a parry press wins over an
attack press made at the same time. The viewer maps `Q` to parry.

## Observation

`encode_observation` returns one flat vector from the acting fighter's point of view
(`spec_version` 2):

```
[acting fighter: F values] [opponent slot 0: F] ... [opponent slot K-1: F] [mask: K values]
```

`F` is `FEATURES_PER_FIGHTER` (14) and `K` is `observation.max_opponents` in
[`configs/game.yaml`](../configs/game.yaml) (3), so the vector has `F * (1 + K) + K` = 59
values. `K` is the only place the number of opponents appears: raising it needs only that
config value (and a retrain).

- Opponents fill the slots in the natural order of their ids (`opponent_2` before
  `opponent_10`), from slot 0.
- A slot with no opponent, or a dead one, is all zeros. Its **mask** value is 0.0 (1.0 for
  a living opponent), so a network can tell an absent opponent from one standing at x = 0.
- Fewer opponents than slots is allowed, so one network plays against any number up to
  `K`. More than `K` raises an error.

Every fighter contributes these 14 values, each clamped to `[-1, 1]`. Their order is
`FEATURE_NAMES` in [`sim/observation_vector.py`](../src/arena_ai/sim/observation_vector.py),
the single source for this table:

| Offset | Feature        | Scaling                                          |
| ------ | -------------- | ------------------------------------------------ |
| 0      | `x`            | divided by `arena_width_m`                       |
| 1      | `y`            | height above the floor / `max_observed_height_m` |
| 2      | `vx`           | divided by `move_speed_mps`                      |
| 3      | `vy`           | divided by `jump_velocity_mps`                   |
| 4      | `on_ground`    | 1.0 or 0.0                                       |
| 5      | `facing_dir`   | -1.0 or 1.0                                      |
| 6      | `health`       | divided by `max_health`                          |
| 7      | `attack_frame` | divided by the total attack length (34 frames)   |
| 8      | `attack_counter` | 1.0 while the swing is a counter, else 0.0     |
| 9      | `parry_frame`  | divided by `parry_window_frames + parry_recovery_frames` (0 when not parrying) |
| 10     | `parry_window` | divided by `parry_window_frames`: the protected frames of the running parry, or of the next one when idle. Below 1.0 means spam is shrinking it |
| 11     | `counter_frames` | frames left to counter, divided by `counter_window_frames` |
| 12     | `stagger_frames` | frames left in which a parried fighter cannot attack, divided by `parried_stagger_frames` |
| 13     | `parry_held` | 1.0 if the fighter's parry input was held on the previous step. Only a new press starts a parry, so a policy that keeps parry on never starts another one, and needs this to know it must release first |

Read a vector by name with `fighter_features(vector, slot)` (slot 0 is the acting fighter,
1 and up are opponents), `opponent_mask(vector, K)` and `slot_count(vector)`. Never index it
by number: add a feature at the end of `FEATURE_NAMES` and bump `spec_version`, and the
checkpoint check refuses old weights.

A scripted opponent sees the vector built from the agent's side: slot 0 is the agent, slot
1 is the opponent itself and slots 2 and up are its living allies (see
[`agents/scripted/view.py`](../src/arena_ai/agents/scripted/view.py)).

## Combat

- A fighter starts an attack with 1 as `attack_frame`; the frame then counts up
  by one each step. Frames 1-16 are wind-up, 17-20 are active and 21-34 are
  recovery. Only the active frames can hit, so a swing can be seen coming for 16
  frames (0.27 s). The first active frame is where the animation shows the sword
  raised high and starting to swing. A fighter can turn mid-swing by moving the other way, and the swing's hitbox
  turns with it. Facing is locked only while parrying.
- During the active frames the attack hits every enemy that overlaps a
  box extending from the attacker's front edge to `width / 2 + attack_range_m` in
  front of its center, and that vertically overlaps by less than one fighter height. Each
  swing hits each enemy at most once, nearest first, and deals `attack_damage` (10) to
  each, or `counter_damage` (15) if it is a counter attack. An enemy that parries the swing cancels it for everyone not yet hit. The box never reaches behind the
  attacker, except for a counter (see Counter).
- Fighters have `max_health` (100) health. Fighters on the same team never hit each other,
  and a dead fighter cannot attack or be hit. Who is alive is fixed at the start of the
  step, for attackers and for targets, so the result does not depend on fighter order:
  if two fighters are hit in the same step, both lose health.
- A fighter starts an attack or a parry only when it was doing neither at the
  start of the step, so the two never overlap.
- The total attack length is derived as wind-up + active + recovery frames
  (`CombatConfig.attack_total_frames`), so the observed attack progress always
  ends at 1.0.

## Parry and counter

All values are in the `combat` section of [`configs/game.yaml`](../configs/game.yaml).

**Parry.** A new press of the parry button starts a parry. Holding the button
parries once, because only a press (not the held state) starts one. The parry has
a protected window followed by a fixed recovery (`parry_recovery_frames`). While
it runs the fighter cannot walk, jump or attack. The parry is not a block: an
attack that lands outside the window deals damage as usual.

**What it stops.** An attack is parried when its hit would land on the parrying
fighter during the protected window and the attacker's center is on the side the
parrying fighter faces (level counts as in front). The hit test already allows
attackers up to one fighter height above or below, so a dive from above is
parried. An attacker whose center is behind the parrying fighter, even one
dropping from above, still deals damage.

**Spam penalty.** Each press made within `parry_spam_reset_frames` (30 frames,
500 ms) of the last parry ending multiplies the window by `parry_window_decay`
(0.5), down to `parry_min_window_frames`: 8, 4, 3, 3 frames. A pause of that
length, or a successful parry, restores the full window. The recovery never
changes, so spamming only makes the timing harder. The next window is observable
(feature 10).

**On success.** The attacker's swing is cancelled (`attack_frame` back to 0) and
the attacker is staggered (below). The parrying fighter's parry ends
and its spam count resets. For `counter_window_frames` (24 frames) its next attack
is a counter.

**Stagger.** The attacker's recovery is replaced by `parried_stagger_frames` (20,
against a normal recovery of 14), during which it cannot start an attack. It can
still move and parry, so it can answer the counter. The viewer plays the swing
backwards over those frames (the sword looks deflected), and the parrying fighter's
shield effects play on to the end of the protected window after the parry, even while
it counters. Only a new parry restarts them, and its sprite pose changes to the attack. The shield is drawn only during the protected window, so
it is never visible while the fighter cannot parry, and a spam-shrunk window gives a
shorter animation. Feature 12 shows the stagger.

**Viewer effects.** Both fighters show them. While the protected window is open the
fighter is surrounded by a bubble outline (no fill, no fading) on the side it faces, an
arc of 200 degrees, for exactly those frames and no others. It burns out over the
window: the edge is uneven and flickers like a flame (more toward the ends), the arc
shrinks from both ends, and particles shed from the ends flow backward along the
bubble's surface and fade. When the window ends the outline is gone at once. Every
effect uses its fighter's side color: blue for the agent and red for the opponent. A
fighter that takes damage flashes: it fades down toward red and back to normal over
`hurt_flash_seconds`. A successful parry adds a ring with sparks and a "Parry" text at
the point of contact. Starting a counter attack adds a "Counter" text and cuts a
straight line that follows the fighter. The line carries on through the opponent if the counter hits, and stops at the parrying fighter
if the counter is parried. Their size, opacity and duration are the `parry_*` and
`counter_*` values in [`configs/viewer.yaml`](../configs/viewer.yaml). Attack and parry cannot cancel each
other: a fighter can only attack once its parry is over, which a successful parry
ends at once.

**Counter.** A counter has a shorter wind-up (`counter_windup_frames`, 12 frames
instead of 16), so it is harder to react to but still possible to parry. During the
wind-up the fighter lunges at `counter_lunge_speed_mps` toward the fighter whose
swing it parried (the nearest other fighter if that one is gone). The direction is fixed
when the swing starts, so the lunge does not follow an opponent who moves away or jumps
over. It cuts through that fighter and goes on to the farthest living enemy in that
direction that the lunge can still reach (`counter_lunge_speed_mps` times
`counter_windup_frames`, 1.2 m), stopping at its x, so enemies in a row are all passed
through and, as the swing hits every enemy it reaches, all damaged. An enemy farther than
the lunge is not reached. The lunge stays stopped if the opponent walks through it. The counter's hitbox also
reaches `counter_hitbox_back_m` (0.8 m) behind the attacker's center, so walking
through the lunging attacker does not dodge it. A normal swing starts at the attacker's
front edge and never reaches behind it. An opponent who runs away is still hit. One
who times a jump as the lunge begins is not. A counter can be parried like any attack
while the attacker is still in front of the parrier, which stops it cutting through and
starts the exchange again.

**Body collision.** Every fighter has a body box (`width_m` by `height_m`) and two
fighters cannot overlap, so neither can walk through or get stuck inside the other.
After the fighters move, and again after hits and parries, `resolve_collisions`
([`sim/collision.py`](../src/arena_ai/sim/collision.py)) pushes any overlapping pair
apart along x. The overlap is split evenly, so a walker nudges a fighter that stands
still, and a wall takes the push first. Fighters at different heights do not collide,
so a jump clears an opponent. The only exception is a counter's lunge (its wind-up and the first active frame),
which passes through the opponent's body. The lunge stops at the opponent's x, so the
two are level on that frame, and the opponent can still parry it then. If it is not
parried, the lunger is pushed out on the far side, the side it was heading toward, and
the opponent is not moved. A parry stops the lunge, so a parried counter is put back
on the side it came from and never ends inside the parrier. Once the lunger is past, it
can no longer be parried, because a parry only blocks an attacker in front of the parrier.
A fighter that walks into another or into a wall is stopped, and the one standing
still is not pushed. Its velocity (`vx`) is what it actually moved, so a fighter held
against a wall or an opponent has `vx` 0: it plays the idle animation and no footsteps,
and scripted opponents see it as standing still. Collisions change no observation,
action or reward layout.

**Air hover.** A successful parry while airborne lifts the fighter by
`parry_hover_impulse_mps`, never above `parry_hover_max_vy_mps` (a faster climb is
kept, not reduced), and scales gravity by `parry_hover_gravity_scale` for
`parry_hover_frames`. A parry that does not stop an attack never hovers, so spamming
the button does not keep a fighter in the air.

## Teams and termination

- **Ids.** [`sim/state.py`](../src/arena_ai/sim/state.py) is the single source: the agent
  is `"agent"` and opponent `i` is `opponent_id(i)`: `"opponent"`, then `"opponent_1"`,
  `"opponent_2"`, ... `team_of` gives the agent team 0 and everyone else team 1, derived
  from the id, so a bare `FighterState` needs no team field. `enemies_of(id, fighters)`
  returns the other team in natural id order.
- **Spawns.** `create_initial_fighters(config, rng, opponent_count)` puts the agent at
  0.25 of the arena width and the opponents in a row on the right, centered on 0.75 and
  one body width or more apart (a single opponent is at 0.75, as before). With an `rng`
  every fighter spawns at a random position, the agent first and then the opponents in
  order, and overlapping fighters are pushed apart without drawing more random numbers.
  The agent faces its nearest opponent, and every opponent faces the agent.
- **Episode end.** The episode terminates when the agent is dead or every opponent is dead
  (`sim/outcome.py`). `info["winner"]` is `"agent"` if the agent is alive and every
  opponent is dead, `"opponent"` if the agent is dead and at least one opponent is alive,
  and `None` for a draw (the agent and the last opponent fall in the same step).
- **Time-out.** A truncated episode has no winner. If the agent is ahead on health (above
  the mean opponent health, dead opponents counting as 0), `info["timeout_win"]` is true
  and the agent earns `timeout_win_reward`, much less than a kill. The viewer decides a
  timed-out match the same way and calls an equal result a draw.

## Reward

[`envs/reward.py`](../src/arena_ai/envs/reward.py) returns the learner's reward
for each step:

```
reward = damage_reward_scale * (damage dealt - damage taken)
         - edge_penalty
         - time_penalty_per_step  (every step)
         + kill_reward         (per opponent that reached 0 health this step; 0 by default)
         + victory_reward      (every opponent reached 0 health)
         - defeat_penalty      (the learner reached 0 health)
         + parry_reward        (the learner's parry stopped an attack)
         + counter_reward      (the learner's counter attack hit)
         - missed_parry_penalty  (a parry ended without stopping an attack)
         - missed_attack_penalty (a swing finished without hitting or being parried)
```

On a time-out the environment adds `timeout_win_reward` (0.25) when the agent is ahead
on health. It is positive but well below the victory reward, so ending a fight by
stalling is better than losing and worse than winning.

The time penalty is small on purpose: it only decides between fights the agent
already wins or parries well, so it pushes the agent to finish them sooner. A
full-length episode costs 0.3, less than a victory (1.0).

The counter reward pays for finishing a parry with its counter attack, so the agent
learns the parry is worth more than a block. It pays only when the counter hits, so
spamming counters earns nothing.

The parry reward is needed because a random policy rarely finds a parry timed
inside an 8-frame window. The two miss penalties push the agent toward accurate
attacks and parries instead of spamming. The simulation reports each as a one-step
event on `FighterState` (`parried`, `counter_hit`, `missed_parry`, `missed_attack`).

The edge penalty applies when the learner is closer than `edge_margin` meters to
a wall and grows linearly from 0 at the margin to `edge_penalty_scale` at the
wall. Current values are in [`configs/scoring.yaml`](../configs/scoring.yaml).
If the learner is defeated the defeat penalty applies even if an opponent also
reached 0 health on the same step.

`info["winner"]` is `"agent"` or `"opponent"` on natural termination (`None` for a draw)
and `None` otherwise. `info["timeout_win"]` and `info["opponent_count"]` are also set. Truncated episodes have no winner and do not stop value
bootstrapping (see [training-process.md](training-process.md)).
