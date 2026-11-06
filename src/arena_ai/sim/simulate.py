"""Advance the headless simulation for multiple independently controlled fighters."""

from dataclasses import dataclass
from collections.abc import Mapping

from arena_ai.config import Config, GameConfig
from arena_ai.sim.rules import step_fighter
from arena_ai.sim.state import FighterState


@dataclass(frozen=True)
class FighterAction:
    """Input applied to one fighter during one simulation timestep."""

    move_dir: int = 0
    jump: bool = False
    attack: bool = False


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
        if not isinstance(action.jump, bool) or not isinstance(action.attack, bool):
            raise TypeError(f"jump and attack for {fighter_id!r} must be bools")


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
        )

    for fighter_id, attacker in fighters.items():
        if not attacker_is_active(attacker, config.game):
            continue
        if attacker.attack_hit:
            continue
        for target_id, target in fighters.items():
            if target_id != fighter_id and attack_hits(attacker, target, config.game):
                target.health = max(0, target.health - config.game.combat.attack_damage)
                attacker.attack_hit = True
                break


def attacker_is_active(fighter: FighterState, config: GameConfig) -> bool:
    """Return whether a fighter is in the active attack frames."""
    combat = config.combat
    return combat.attack_windup_frames < fighter.attack_frame <= (
        combat.attack_windup_frames + combat.attack_active_frames
    )


def attack_hits(
    attacker: FighterState,
    target: FighterState,
    config: GameConfig,
) -> bool:
    """Return whether an attacker's forward hitbox overlaps a target."""
    fighter_width = config.fighter.height_m * 0.3
    attack_range = config.combat.attack_range_m
    attack_start = attacker.x + attacker.facing_dir * fighter_width / 2
    attack_end = attack_start + attacker.facing_dir * attack_range
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
