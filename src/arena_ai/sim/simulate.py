"""Advance the headless simulation for multiple independently controlled fighters."""

from dataclasses import dataclass
from collections.abc import Mapping

from arena_ai.config import AttackType, Config, GameConfig
from arena_ai.sim.collision import resolve_collisions
from arena_ai.sim.rules import LEVEL_TOLERANCE_M, end_attack, step_fighter
from arena_ai.sim.state import FighterState, enemies_of


# A difference below this between the intended and the actual velocity is float rounding.
VELOCITY_TOLERANCE_MPS = 1e-6


@dataclass(frozen=True)
class FighterAction:
    """Input applied to one fighter during one simulation timestep."""

    move_dir: int = 0
    jump: bool = False
    attack: bool = False
    parry: bool = False
    dodge: bool = False


def _validate_actions(
    fighters: Mapping[str, FighterState],
    actions: Mapping[str, FighterAction],
) -> None:
    """Ensure that every fighter has a valid
    action and no unknown fighters are present."""
    fighter_ids = set(fighters)
    action_ids = set(actions)
    missing_ids = fighter_ids - action_ids
    unknown_ids = action_ids - fighter_ids

    if missing_ids:
        raise ValueError(f"Missing actions for fighters: {sorted(missing_ids)}")
    if unknown_ids:
        raise ValueError(f"Actions provided for unknown fighters: {sorted(unknown_ids)}")

    for fighter_id, action in actions.items():
        if not isinstance(action, FighterAction):
            raise TypeError(f"Action for {fighter_id!r} must be a FighterAction")
        if action.move_dir not in (-1, 0, 1):
            raise ValueError(
                f"move_dir for {fighter_id!r} must be -1, 0, or 1; got {action.move_dir}"
            )
        flags = (action.jump, action.attack, action.parry, action.dodge)
        if not all(isinstance(flag, bool) for flag in flags):
            raise TypeError(f"jump, attack, parry and dodge for {fighter_id!r} must be bools")


def step_game(
    fighters: Mapping[str, FighterState],
    actions: Mapping[str, FighterAction],
    config: Config,
) -> None:
    """Advance every fighter by one fixed timestep using its own action.

    ``fighters`` and ``actions`` use the same stable IDs. A complete action
    mapping is required so a missing controller cannot silently stop a fighter.
    """
    _validate_actions(fighters, actions)
    start_x = {fighter_id: fighter.x for fighter_id, fighter in fighters.items()}

    # Who is alive is fixed at the start of the step, for attackers and for targets, so that
    # the result does not depend on the order of the fighters: hits that land on the same
    # step all land.
    alive = {fighter_id for fighter_id, fighter in fighters.items() if fighter.health > 0}

    for fighter_id, fighter in fighters.items():
        # A dead fighter does nothing, but still falls and stops.
        action = actions[fighter_id] if fighter_id in alive else FighterAction()
        step_fighter(
            fighter,
            action.move_dir,
            action.jump,
            config.game,
            attack=action.attack,
            parry=action.parry,
            dodge=action.dodge,
            target_x=_lunge_target_x(fighter_id, fighters, config.game),
        )

    resolve_collisions(fighters, config.game)

    for fighter_id, attacker in fighters.items():
        if fighter_id not in alive or not attacker_is_active(attacker, config.game):
            continue
        # Fighters only hit the other team. A swing hits every enemy it reaches, each at most
        # once, nearest first. An enemy that parries it cancels the whole swing.
        enemies = enemies_of(fighter_id, fighters)
        for target_id, target in sorted(
            enemies.items(), key=lambda item: abs(item[1].x - attacker.x)
        ):
            if (
                target_id not in alive
                or target_id in attacker.hit_targets
                or target.dodge_frames_left
                or not attack_hits(attacker, target, config.game)
            ):
                continue
            if parry_blocks(target, attacker):
                apply_parry(target, attacker, fighter_id, config.game)
                break
            damage = config.game.combat.damage(attacker.attack_type)
            target.health = max(0, target.health - damage)
            attacker.hit_targets.add(target_id)
            attacker.attack_hit = True
            attacker.counter_hit = attacker.attack_type is AttackType.COUNTER

    # A counter that was not parried ends its lunge here, so it is pushed out the far side. A
    # parry cancels it, so a parried fighter is pushed out the near side, not left inside.
    resolve_collisions(fighters, config.game, hits_checked=True)

    # A fighter that was stopped by a wall or another fighter is not moving: its velocity is
    # what it actually moved, so the animation, the footsteps and the observation agree.
    dt = config.game.world.dt_seconds
    for fighter_id, fighter in fighters.items():
        moved = (fighter.x - start_x[fighter_id]) / dt
        if abs(moved - fighter.vx) > VELOCITY_TOLERANCE_MPS:
            fighter.vx = moved


def _lunge_target_x(
    fighter_id: str, fighters: Mapping[str, FighterState], config: GameConfig
) -> float | None:
    """Return the x a counter lunges toward, so that it cuts through every enemy it can reach.

    It starts from the parried fighter (the nearest enemy if that one is gone). The lunge
    then goes on to the farthest living enemy in that direction that the lunge can still
    reach, so enemies in a row are all passed through and hit.
    """
    me = fighters[fighter_id]
    parried = fighters.get(me.counter_target or "")
    if parried is not None and parried.health > 0:
        base = parried.x
    else:
        base = _nearest_opponent_x(fighter_id, fighters)
    if base is None:
        return None
    direction = me.lunge_dir or (base > me.x) - (base < me.x)
    if direction == 0:
        return base
    counter = config.combat.spec(AttackType.COUNTER)
    reach = counter.lunge_speed_mps * counter.attack_windup_frames * config.world.dt_seconds
    ahead = [
        (enemy.x - me.x) * direction
        for enemy in enemies_of(fighter_id, fighters).values()
        if enemy.health > 0 and 0 < (enemy.x - me.x) * direction <= reach
    ]
    base_distance = (base - me.x) * direction
    return me.x + direction * max([base_distance, *ahead])


def _nearest_opponent_x(fighter_id: str, fighters: Mapping[str, FighterState]) -> float | None:
    """Return the x position of the nearest living enemy, or None when there is none."""
    me = fighters[fighter_id]
    others = [f for f in enemies_of(fighter_id, fighters).values() if f.health > 0]
    if not others:
        return None
    return min(others, key=lambda other: abs(other.x - me.x)).x


def parry_blocks(target: FighterState, attacker: FighterState) -> bool:
    """Return whether ``target`` parries a hit from ``attacker``.

    The target must be inside the protected frames of its parry and the attacker's
    center must be on the side it faces (or level with it). The hit already
    overlaps the target, so attackers diving from above are covered, and an
    attacker whose center is behind the target still lands the hit.
    """
    in_window = 0 < target.parry_frame <= target.parry_window
    in_front = (attacker.x - target.x) * target.facing_dir >= -LEVEL_TOLERANCE_M
    return in_window and in_front


def apply_parry(
    target: FighterState, attacker: FighterState, attacker_id: str, config: GameConfig
) -> None:
    """Cancel the swing, stagger the attacker and give the parrier its counter window."""
    combat = config.combat
    if attacker.lunge_dir and abs(attacker.x - target.x) < config.fighter.width_m:
        # A parried lunge stops on the side it came from. The lunge direction is cleared
        # below, so put it there now, or the collision step could put it on the far side.
        attacker.x = target.x - attacker.lunge_dir * config.fighter.width_m
    attacker.stagger_from_frame = combat.normal_swing_frame(
        attacker.attack_frame, attacker.attack_type
    )
    end_attack(attacker)
    attacker.stagger_frames = combat.parried_stagger_frames
    target.parry_pose_frame = target.parry_frame
    target.parry_frame = 0
    target.parry_count = 0
    target.parry_idle_frames = 0
    target.parried = True
    target.counter_frames = combat.counter_window_frames
    target.counter_target = attacker_id
    if not target.on_ground:
        # Hover: lift to at most the cap (never reduce a faster climb) and fall slowly.
        target.vy = max(
            target.vy,
            min(target.vy + combat.parry_hover_impulse_mps, combat.parry_hover_max_vy_mps),
        )
        target.hover_frames = combat.parry_hover_frames


def attacker_is_active(fighter: FighterState, config: GameConfig) -> bool:
    """Return whether a fighter is in the active attack frames."""
    spec = config.combat.spec(fighter.attack_type)
    windup = spec.attack_windup_frames
    return windup < fighter.attack_frame <= windup + spec.attack_active_frames


def attack_hits(
    attacker: FighterState,
    target: FighterState,
    config: GameConfig,
) -> bool:
    """Return whether an attacker's forward hitbox overlaps a target."""
    fighter_width = config.fighter.width_m
    spec = config.combat.spec(attacker.attack_type)
    attack_range = spec.range_m
    # Bodies cannot overlap, so the hitbox starts at the attacker's front edge and a swing
    # never reaches a fighter behind it. Only an attack with a ``hitbox_back_m`` (the
    # counter) also reaches back, for an opponent that walked through its lunge.
    behind = -fighter_width / 2 if spec.hitbox_back_m is None else spec.hitbox_back_m
    attack_start = attacker.x - attacker.facing_dir * behind
    attack_end = attacker.x + attacker.facing_dir * (fighter_width / 2 + attack_range)
    hitbox_left = min(attack_start, attack_end)
    hitbox_right = max(attack_start, attack_end)
    target_left = target.x - fighter_width / 2
    target_right = target.x + fighter_width / 2
    horizontal_overlap = hitbox_left <= target_right and target_left <= hitbox_right
    fighter_height = config.fighter.height_m
    vertical_overlap = (
        target.y < attacker.y + fighter_height and attacker.y < target.y + fighter_height
    )
    return horizontal_overlap and vertical_overlap
