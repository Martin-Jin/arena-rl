"""Advance the headless simulation for multiple independently controlled fighters."""

from dataclasses import dataclass
from collections.abc import Mapping

from arena_ai.config import Config, GameConfig
from arena_ai.sim.rules import LEVEL_TOLERANCE_M, end_attack, step_fighter
from arena_ai.sim.state import FighterState


@dataclass(frozen=True)
class FighterAction:
    """Input applied to one fighter during one simulation timestep."""

    move_dir: int = 0
    jump: bool = False
    attack: bool = False
    parry: bool = False


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
        raise ValueError(
            f"Actions provided for unknown fighters: {sorted(unknown_ids)}"
        )

    for fighter_id, action in actions.items():
        if not isinstance(action, FighterAction):
            raise TypeError(f"Action for {fighter_id!r} must be a FighterAction")
        if action.move_dir not in (-1, 0, 1):
            raise ValueError(
                f"move_dir for {fighter_id!r} must be -1, 0, or 1; "
                f"got {action.move_dir}"
            )
        flags = (action.jump, action.attack, action.parry)
        if not all(isinstance(flag, bool) for flag in flags):
            raise TypeError(f"jump, attack and parry for {fighter_id!r} must be bools")


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

    for fighter_id, fighter in fighters.items():
        action = actions[fighter_id]
        step_fighter(
            fighter,
            action.move_dir,
            action.jump,
            config.game,
            attack=action.attack,
            parry=action.parry,
            target_x=_nearest_opponent_x(fighter_id, fighters),
        )

    for fighter_id, attacker in fighters.items():
        if not attacker_is_active(attacker, config.game):
            continue
        if attacker.attack_hit:
            continue
        for target_id, target in fighters.items():
            if target_id == fighter_id or not attack_hits(attacker, target, config.game):
                continue
            if parry_blocks(target, attacker):
                apply_parry(target, attacker, config.game)
            else:
                target.health = max(0, target.health - config.game.combat.attack_damage)
                attacker.attack_hit = True
            break


def _nearest_opponent_x(
    fighter_id: str, fighters: Mapping[str, FighterState]
) -> float | None:
    """Return the x position of the nearest other fighter, or None when alone."""
    me = fighters[fighter_id]
    others = [f for other_id, f in fighters.items() if other_id != fighter_id]
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


def apply_parry(target: FighterState, attacker: FighterState, config: GameConfig) -> None:
    """Cancel the swing, stagger the attacker and give the parrier its counter window."""
    combat = config.combat
    attacker.stagger_from_frame = combat.normal_swing_frame(
        attacker.attack_frame, attacker.attack_counter
    )
    end_attack(attacker)
    attacker.stagger_frames = combat.parried_stagger_frames
    target.parry_pose_frame = target.parry_frame
    target.parry_frame = 0
    target.parry_count = 0
    target.parry_idle_frames = 0
    target.parried = True
    target.counter_frames = combat.counter_window_frames
    if not target.on_ground:
        # Hover: lift to at most the cap (never reduce a faster climb) and fall slowly.
        target.vy = max(
            target.vy,
            min(target.vy + combat.parry_hover_impulse_mps, combat.parry_hover_max_vy_mps),
        )
        target.hover_frames = combat.parry_hover_frames


def attacker_is_active(fighter: FighterState, config: GameConfig) -> bool:
    """Return whether a fighter is in the active attack frames."""
    combat = config.combat
    return (
        combat.windup_frames(fighter.attack_counter)
        < fighter.attack_frame
        <= combat.windup_frames(fighter.attack_counter) + combat.attack_active_frames
    )


def attack_hits(
    attacker: FighterState,
    target: FighterState,
    config: GameConfig,
) -> bool:
    """Return whether an attacker's forward hitbox overlaps a target."""
    fighter_width = config.fighter.width_m
    attack_range = config.combat.attack_range_m
    # Hitbox starts at the attacker's center, so overlapping targets are hit.
    # A counter also reaches behind the attacker, so walking through it does not dodge it.
    behind = config.combat.counter_hitbox_back_m if attacker.attack_counter else 0.0
    attack_start = attacker.x - attacker.facing_dir * behind
    attack_end = attacker.x + attacker.facing_dir * (fighter_width / 2 + attack_range)
    hitbox_left = min(attack_start, attack_end)
    hitbox_right = max(attack_start, attack_end)
    target_left = target.x - fighter_width / 2
    target_right = target.x + fighter_width / 2
    horizontal_overlap = hitbox_left <= target_right and target_left <= hitbox_right
    fighter_height = config.fighter.height_m
    vertical_overlap = (
        target.y < attacker.y + fighter_height
        and attacker.y < target.y + fighter_height
    )
    return horizontal_overlap and vertical_overlap and target.health > 0
