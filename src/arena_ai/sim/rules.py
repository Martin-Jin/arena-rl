# rules.py
# This module defines the game rules and physics, including player behavior and game-state updates.

from arena_ai.config import GameConfig
from arena_ai.sim.state import FighterState


def step_fighter(
    f: FighterState,
    move_dir: int,
    jump: bool,
    config: GameConfig,
    attack: bool = False,
) -> None:
    w, fc = config.world, config.fighter
    dt = w.dt_seconds

    if move_dir and not f.attack_frame:
        f.facing_dir = move_dir

    combat = config.combat
    # Update the fighter's attack state.
    if f.attack_frame:
        f.attack_frame += 1
        # Reset the attack state after the full attack animation has completed.
        if f.attack_frame > (
            combat.attack_windup_frames
            + combat.attack_active_frames
            + combat.attack_recovery_frames
        ):
            f.attack_frame = 0
            f.attack_hit = False
    elif attack:
        f.attack_frame = 1
        f.attack_hit = False

    # Update the fighter's horizontal velocity.
    f.vx = move_dir * fc.move_speed_mps  # -1, 0, or 1
    # Apply gravity and jump mechanics.
    if jump and f.on_ground:
        f.vy = fc.jump_velocity_mps
        f.on_ground = False

    f.vy -= w.gravity_mps2 * dt  # gravity changes velocity
    f.x += f.vx * dt  # velocity changes position
    f.y += f.vy * dt

    # TODO: Replace this floor-only check with collision resolution against the
    # arena floor and platforms, then derive on_ground from the resulting
    # walkable support contact rather than from one fixed y level.
    # Apply collision mechanics.
    if f.y <= w.floor_y_m:  # floor
        f.y, f.vy, f.on_ground = w.floor_y_m, 0.0, True
    f.x = min(max(f.x, 0.0), w.arena_width_m)  # arena walls
