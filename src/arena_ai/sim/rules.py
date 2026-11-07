# rules.py
# This module defines the game rules and physics, including player behavior and game-state updates.

from arena_ai.config import CombatConfig, GameConfig
from arena_ai.sim.state import FighterState

# Positions that differ by less than this are level: it absorbs float rounding after
# a lunge stops exactly at the opponent's x.
LEVEL_TOLERANCE_M = 1e-9


def next_parry_window(f: FighterState, combat: CombatConfig) -> int:
    """Return the protected frames a parry started now would have.

    The window halves (``parry_window_decay``) for each recent press, down to
    ``parry_min_window_frames``. A pause of ``parry_spam_reset_frames`` restores it.
    """
    count = 0 if f.parry_idle_frames >= combat.parry_spam_reset_frames else f.parry_count
    window = int(combat.parry_window_frames * combat.parry_window_decay**count)
    return max(combat.parry_min_window_frames, window)


def parry_window_for_observation(f: FighterState, combat: CombatConfig) -> int:
    """Return the window of the running parry, or of the next one when not parrying."""
    return f.parry_window if f.parry_frame else next_parry_window(f, combat)


def end_attack(f: FighterState) -> None:
    """Return a fighter to not attacking."""
    f.attack_frame = 0
    f.attack_hit = False
    f.attack_counter = False
    f.lunge_dir = 0


def step_fighter(
    f: FighterState,
    move_dir: int,
    jump: bool,
    config: GameConfig,
    attack: bool = False,
    parry: bool = False,
    target_x: float | None = None,
) -> None:
    """Advance one fighter by one step.

    ``target_x`` is the x position of the nearest opponent. A counter attack lunges
    toward it. The direction is fixed when the swing starts, so the lunge does not
    follow an opponent that moves away or jumps over. It stops at the opponent's x, and
    stays stopped if the opponent walks through it, so the lunge never carries the
    fighter past or away from an opponent that is no longer ahead.
    """
    w, fc = config.world, config.fighter
    dt = w.dt_seconds
    combat = config.combat

    f.parried = f.missed_attack = f.missed_parry = f.counter_started = False
    f.counter_hit = False
    staggered = f.stagger_frames > 0

    # A fighter that was busy at the start of the step cannot start anything new, so
    # attacking and parrying never overlap.
    was_busy = bool(f.attack_frame or f.parry_frame)
    # Holding the parry button parries once: only a new press starts a parry.
    parry_pressed = parry and not f.parry_held
    f.parry_held = parry
    # A fighter can turn mid-attack, and the swing's hitbox follows. Facing is locked while
    # parrying, and by the press that starts a parry. Otherwise pressing a direction with
    # parry would turn the fighter toward an attacker behind it in the same step, and the
    # parry would stop a hit from behind.
    starting_parry = parry_pressed and not was_busy
    if move_dir and not f.parry_frame and not starting_parry:
        f.facing_dir = move_dir

    if f.attack_frame:
        f.attack_frame += 1
        # Reset the attack state after the full attack animation has completed.
        if f.attack_frame > combat.swing_total_frames(f.attack_counter):
            f.missed_attack = not f.attack_hit
            end_attack(f)
    if f.parry_frame:
        f.parry_frame += 1
        if f.parry_frame > f.parry_window + combat.parry_recovery_frames:
            f.parry_frame = 0
            f.parry_idle_frames = 0
            f.missed_parry = True
    else:
        f.parry_idle_frames = min(f.parry_idle_frames + 1, combat.parry_spam_reset_frames)
    if f.parry_pose_frame:
        f.parry_pose_frame += 1
        if f.parry_pose_frame > f.parry_window + combat.parry_recovery_frames:
            f.parry_pose_frame = 0
    f.stagger_frames = max(0, f.stagger_frames - 1)
    f.counter_frames = max(0, f.counter_frames - 1)
    f.hover_frames = max(0, f.hover_frames - 1)

    if not was_busy:
        if parry_pressed:
            f.parry_window = next_parry_window(f, combat)
            f.parry_count = (
                1
                if f.parry_idle_frames >= combat.parry_spam_reset_frames
                else f.parry_count + 1
            )
            f.parry_frame = 1
            f.parry_pose_frame = 0
        elif attack and not staggered:
            f.attack_frame = 1
            f.attack_hit = False
            if f.counter_frames:
                f.attack_counter = True
                f.counter_started = True
                f.counter_frames = 0
                if target_x is not None:
                    f.lunge_dir = (target_x > f.x) - (target_x < f.x)
                if f.lunge_dir:
                    f.facing_dir = f.lunge_dir

    # A parrying fighter cannot walk or jump.
    if f.parry_frame:
        f.vx = 0.0
    elif f.attack_counter and f.attack_frame <= combat.counter_windup_frames:
        f.vx = f.lunge_dir * combat.counter_lunge_speed_mps
        if target_x is not None:
            ahead = (target_x - f.x) * f.lunge_dir  # Not positive once the target is reached.
            f.vx = f.lunge_dir * max(0.0, min(combat.counter_lunge_speed_mps, ahead / dt))
    else:
        f.vx = move_dir * fc.move_speed_mps  # -1, 0, or 1
    # Apply gravity and jump mechanics.
    if jump and f.on_ground and not f.parry_frame:
        f.vy = fc.jump_velocity_mps
        f.on_ground = False

    gravity = w.gravity_mps2
    if f.hover_frames:
        gravity *= combat.parry_hover_gravity_scale
    f.vy -= gravity * dt  # gravity changes velocity
    f.x += f.vx * dt  # velocity changes position
    f.y += f.vy * dt

    # TODO: Replace this floor-only check with collision resolution against the
    # arena floor and platforms, then derive on_ground from the resulting
    # walkable support contact rather than from one fixed y level.
    # Apply collision mechanics.
    if f.y <= w.floor_y_m:  # floor
        f.y, f.vy, f.on_ground = w.floor_y_m, 0.0, True
        f.hover_frames = 0
    f.x = min(max(f.x, 0.0), w.arena_width_m)  # arena walls
