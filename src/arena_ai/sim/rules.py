# rules.py
# This module defines the game rules and physics, including player behavior and game-state updates.

from arena_ai.config import GameConfig
from arena_ai.sim.state import FighterState


def step_fighter(
    f: FighterState, move_dir: int, jump: bool, config: GameConfig
) -> None:
    w, fc = config.world, config.fighter
    dt = w.dt_seconds

    # Update the fighter's horizontal velocity.
    f.vx = move_dir * fc.move_speed_mps  # -1, 0, or 1
    # Apply gravity and jump mechanics.
    if jump and f.on_ground:
        f.vy = fc.jump_velocity_mps
        f.on_ground = False

    f.vy -= w.gravity_mps2 * dt  # gravity changes velocity
    f.x += f.vx * dt  # velocity changes position
    f.y += f.vy * dt

    # Apply collision mechanics.
    if f.y <= w.floor_y_m:  # floor
        f.y, f.vy, f.on_ground = w.floor_y_m, 0.0, True
    f.x = min(max(f.x, 0.0), w.arena_width_m)  # arena walls
