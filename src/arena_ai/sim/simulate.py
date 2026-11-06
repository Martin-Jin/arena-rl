"""Advance the headless simulation for multiple independently controlled fighters."""

from dataclasses import dataclass
from collections.abc import Mapping

from arena_ai.config import Config
from arena_ai.sim.rules import step_fighter
from arena_ai.sim.state import FighterState


@dataclass(frozen=True)
class FighterAction:
    """Input applied to one fighter during one simulation timestep."""

    move_dir: int = 0
    jump: bool = False


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
        step_fighter(fighter, action.move_dir, action.jump, config.game)
