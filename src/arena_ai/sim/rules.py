"""Define the game rules and physics: fighter behavior and game-state updates."""

from arena_ai.config import AttackType, CombatConfig, GameConfig
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


def _frames_left(phase_length: int, phase_end: int, frame: int) -> int:
    """Frames of a phase still to run on ``frame``, counting this one.

    ``phase_end`` is the last frame of the phase. A phase that has not started yet counts
    in full, and one that is over counts 0.
    """
    return max(0, min(phase_length, phase_end - frame + 1))


def attack_frames_left(f: FighterState, combat: CombatConfig) -> tuple[int, int, int]:
    """Return the (wind-up, active, recovery) frames of the running swing still to run.

    All 0 when not attacking. They are countdowns of the swing the fighter has already
    started, so they follow that attack's own timing (a counter has a shorter wind-up).
    """
    if not f.attack_frame:
        return 0, 0, 0
    spec = combat.spec(f.attack_type)
    windup = spec.attack_windup_frames
    active_end = windup + spec.attack_active_frames
    recovery_end = active_end + spec.attack_recovery_frames
    return (
        _frames_left(windup, windup, f.attack_frame),
        _frames_left(spec.attack_active_frames, active_end, f.attack_frame),
        _frames_left(spec.attack_recovery_frames, recovery_end, f.attack_frame),
    )


def parry_frames_left(f: FighterState, combat: CombatConfig) -> tuple[int, int]:
    """Return the (protected window, recovery) frames of the running parry still to run."""
    if not f.parry_frame:
        return 0, 0
    recovery = combat.parry_recovery_frames
    return (
        _frames_left(f.parry_window, f.parry_window, f.parry_frame),
        _frames_left(recovery, f.parry_window + recovery, f.parry_frame),
    )


def is_lunging(f: FighterState, combat: CombatConfig, hits_checked: bool = False) -> bool:
    """Return whether a swing is lunging, so it moves by the lunge and passes through bodies.

    Only an attack type with a ``lunge_speed_mps`` lunges (the counter).

    The lunge lasts the wind-up plus the first active frame, the one the hit and any parry
    are checked on, so the lunger is still level with the opponent (in front of it) then.
    ``hits_checked`` is for after that check, when only the wind-up counts: the lunger left
    inside an opponent is pushed out the far side.
    """
    spec = combat.spec(f.attack_type)
    last = spec.attack_windup_frames + (0 if hits_checked else 1)
    return bool(f.attack_frame) and spec.lunge_speed_mps > 0 and f.attack_frame <= last


def passes_through_bodies(f: FighterState, hits_checked: bool = False) -> bool:
    """Return whether a dodging fighter passes through other fighters' bodies.

    The whole burst passes through, except that after the hits are checked the last burst
    frame does not: a dodger left inside a body is pushed out the far side (see
    ``collision._left_right``).
    """
    return f.dodge_frames_left > (1 if hits_checked else 0)


def end_attack(f: FighterState) -> None:
    """Return a fighter to not attacking."""
    f.attack_frame = 0
    f.attack_hit = False
    f.hit_targets = set()
    f.attack_type = AttackType.SLASH
    f.lunge_dir = 0
    f.counter_target = None


def step_fighter(
    f: FighterState,
    move_dir: int,
    jump: bool,
    config: GameConfig,
    attack: bool = False,
    parry: bool = False,
    target_x: float | None = None,
    dodge: bool = False,
) -> None:
    """Advance one fighter by one step.

    ``target_x`` is the x position of the fighter the counter lunges toward (the one this
    fighter parried). A counter attack lunges toward it. The direction is fixed when the
    swing starts, so the lunge does not follow an opponent that moves away or jumps over.
    It passes through the opponent's body (see ``resolve_collisions``) and stops at the
    opponent's x, and stays stopped if the opponent walks through it, so the lunge never
    carries the fighter past or away from an opponent that is no longer ahead. The
    collision step after the hits then puts it on the far side.
    """
    w, fc = config.world, config.fighter
    dt = w.dt_seconds
    combat = config.combat
    dodge_config = config.dodge

    f.parried = f.missed_attack = f.missed_parry = f.counter_started = False
    f.counter_hit = False
    # Dodge countdowns: the burst, then the gap. Stamina refills only when neither runs.
    if f.dodge_frames_left:
        f.dodge_frames_left -= 1
        if not f.dodge_frames_left:
            f.dodge_gap_left = dodge_config.gap_frames
    elif f.dodge_gap_left:
        f.dodge_gap_left -= 1
    else:
        f.stamina = min(dodge_config.max_stamina, f.stamina + dodge_config.regen_per_second * dt)

    staggered = f.stagger_frames > 0

    # A fighter that was busy at the start of the step cannot start anything new, so
    # attacking and parrying never overlap.
    was_busy = bool(f.attack_frame or f.parry_frame)
    # Holding the parry button parries once: only a new press starts a parry.
    parry_pressed = parry and not f.parry_held
    f.parry_held = parry
    # Holding the dodge button dodges once: only a new press starts a dodge.
    dodge_pressed = dodge and not f.dodge_held
    f.dodge_held = dodge
    # A fighter can turn mid-attack, and the swing's hitbox follows. Facing is locked while
    # parrying, and by the press that starts a parry. Otherwise pressing a direction with
    # parry would turn the fighter toward an attacker behind it in the same step, and the
    # parry would stop a hit from behind.
    starting_parry = parry_pressed and not was_busy
    # Facing is also locked during a dodge, so the burst keeps the direction it started in.
    if move_dir and not f.parry_frame and not starting_parry and not f.dodge_frames_left:
        f.facing_dir = move_dir

    if f.attack_frame:
        f.attack_frame += 1
        # Reset the attack state after the full attack animation has completed.
        if f.attack_frame > combat.swing_total_frames(f.attack_type):
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
    if not f.counter_frames and f.attack_type is not AttackType.COUNTER:
        f.counter_target = None  # The window passed without a counter.
    f.hover_frames = max(0, f.hover_frames - 1)

    # A dodge can start unless the fighter is in the air, parrying, staggered, dodging or in the
    # gap after one, or short of stamina. A swing still winding up can be dodged out of (it is cancelled).
    # Once the swing is active or recovering the fighter is committed.
    winding_up = f.attack_frame <= combat.windup_frames(f.attack_type)
    if (
        dodge_pressed
        and f.on_ground
        and not staggered
        and not f.parry_frame
        and not f.dodge_frames_left
        and not f.dodge_gap_left
        and f.stamina >= dodge_config.cost
        and winding_up
    ):
        f.stamina -= dodge_config.cost
        f.dodge_frames_left = dodge_config.duration_frames
        f.dodge_dir = f.facing_dir
        if f.attack_frame:
            end_attack(f)

    # Attacking and parrying are not possible during the burst.
    if not was_busy and not f.dodge_frames_left:
        if parry_pressed:
            f.parry_window = next_parry_window(f, combat)
            f.parry_count = (
                1 if f.parry_idle_frames >= combat.parry_spam_reset_frames else f.parry_count + 1
            )
            f.parry_frame = 1
            f.parry_pose_frame = 0
        elif attack and not staggered:
            f.attack_frame = 1
            f.attack_hit = False
            f.hit_targets = set()
            if f.counter_frames:
                f.attack_type = AttackType.COUNTER
                f.counter_started = True
                f.counter_frames = 0
                if target_x is not None:
                    f.lunge_dir = (target_x > f.x) - (target_x < f.x)
                if f.lunge_dir:
                    f.facing_dir = f.lunge_dir

    # A parrying fighter cannot walk or jump.
    if f.parry_frame:
        f.vx = 0.0
    elif f.dodge_frames_left:
        f.vx = f.dodge_dir * dodge_config.speed_mps
    elif is_lunging(f, combat):
        lunge_speed = combat.spec(f.attack_type).lunge_speed_mps
        f.vx = f.lunge_dir * lunge_speed
        if target_x is not None:
            ahead = (target_x - f.x) * f.lunge_dir  # Not positive once the target is reached.
            f.vx = f.lunge_dir * max(0.0, min(lunge_speed, ahead / dt))
    else:
        f.vx = move_dir * fc.move_speed_mps  # -1, 0, or 1
    # Apply gravity and jump mechanics.
    # No jumping while parrying or dodging: a dodge stays on the ground.
    if jump and f.on_ground and not f.parry_frame and not f.dodge_frames_left:
        f.vy = fc.jump_velocity_mps
        f.on_ground = False

    gravity = w.gravity_mps2
    if f.hover_frames:
        gravity *= combat.parry_hover_gravity_scale
    f.vy -= gravity * dt  # gravity changes velocity
    f.x += f.vx * dt  # velocity changes position
    f.y += f.vy * dt

    # The only surface is the floor, so on_ground is "at the floor's y level".
    # Apply collision mechanics.
    if f.y <= w.floor_y_m:  # floor
        f.y, f.vy, f.on_ground = w.floor_y_m, 0.0, True
        f.hover_frames = 0
    f.x = min(max(f.x, 0.0), w.arena_width_m)  # arena walls
