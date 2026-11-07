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
- A training episode is truncated after `max_episode_steps` steps unless a
  fighter reaches 0 health first. `training.max_episode_steps` in
  `agent.yaml` sets it (1500 steps, 25 s); if unset, the environment defaults to
  `sim_hz * 10` steps.

## Actions

Each step a fighter takes one of 12 discrete actions, defined in
[`agents/action.py`](../src/arena_ai/agents/action.py). The index is a
mixed-radix combination of three independent choices:

```
index = move + 3 * jump + 6 * attack
move: 0 = none, 1 = left, 2 = right      jump: 0 or 1      attack: 0 or 1
```

The policy has three output heads (3 + 2 + 2 = 7 logits) and samples each head
independently. Append new actions only; reordering invalidates checkpoints.

## Observation

`encode_observation` returns one vector from the acting fighter's point of
view: the acting fighter first, then each opponent sorted by ID. A one-versus-one
game therefore has 16 values. Every fighter contributes these 8 values, each
clamped to `[-1, 1]`:

| Offset | Feature        | Scaling                                          |
| ------ | -------------- | ------------------------------------------------ |
| 0      | `x`            | divided by `arena_width_m`                       |
| 1      | `y`            | height above the floor / `max_observed_height_m` |
| 2      | `vx`           | divided by `move_speed_mps`                      |
| 3      | `vy`           | divided by `jump_velocity_mps`                   |
| 4      | `on_ground`    | 1.0 or 0.0                                       |
| 5      | `facing_dir`   | -1.0 or 1.0                                      |
| 6      | `health`       | divided by `max_health`                          |
| 7      | `attack_frame` | divided by the total attack length (30 frames)   |

Update `FEATURES_PER_FIGHTER` in
[`sim/observation_vector.py`](../src/arena_ai/sim/observation_vector.py) when
this table changes. The scripted opponent in
[`agents/scripted.py`](../src/arena_ai/agents/scripted.py) reads the vector by
index, with the learner at 0-7 and itself at 8-15.

## Combat

- A fighter starts an attack with 1 as `attack_frame`; the frame then counts up
  by one each step. Frames 1-12 are wind-up, 13-16 are active and 17-30 are
  recovery. Facing is locked for the whole swing.
- During the active frames the attack hits the first opponent that overlaps a
  box extending from the attacker's center to `width / 2 + attack_range_m` in
  front, and that vertically overlaps by less than one fighter height. Each
  swing hits at most once and deals `attack_damage` (10).
- Fighters have `max_health` (100) health. A fighter at 0 health ends the
  episode. If both fighters are hit in the same step, both lose health.
- The total attack length is derived as wind-up + active + recovery frames
  (`CombatConfig.attack_total_frames`), so the observed attack progress always
  ends at 1.0.

## Reward

[`envs/reward.py`](../src/arena_ai/envs/reward.py) returns the learner's reward
for each step:

```
reward = damage_reward_scale * (damage dealt - damage taken)
         - edge_penalty
         + victory_reward      (an opponent reached 0 health)
         - defeat_penalty      (the learner reached 0 health)
```

The edge penalty applies when the learner is closer than `edge_margin` meters to
a wall and grows linearly from 0 at the margin to `edge_penalty_scale` at the
wall. Current values are in [`configs/scoring.yaml`](../configs/scoring.yaml).
If the learner is defeated the defeat penalty applies even if an opponent also
reached 0 health on the same step.

`info["winner"]` is `"agent"` or `"opponent"` on natural termination and `None`
otherwise. Truncated episodes have no winner and do not stop value
bootstrapping (see [training-process.md](training-process.md)).
