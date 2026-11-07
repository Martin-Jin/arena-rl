"""Define the discrete action encoding used by the learning agent."""

from enum import IntEnum

from arena_ai.sim.simulate import FighterAction


class Action(IntEnum):
    """Complete actions available to a fighter in one timestep."""

    # Each index represents a position in the neural network output vector,
    # which is used to select the action to take in the simulation.
    # Append new actions only; reordering invalidates saved checkpoints.
    IDLE = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    JUMP = 3
    MOVE_LEFT_JUMP = 4
    MOVE_RIGHT_JUMP = 5
    ATTACK = 6
    MOVE_LEFT_ATTACK = 7
    MOVE_RIGHT_ATTACK = 8
    JUMP_ATTACK = 9
    MOVE_LEFT_JUMP_ATTACK = 10
    MOVE_RIGHT_JUMP_ATTACK = 11


ACTION_COUNT = len(Action)

_FIGHTER_ACTIONS: dict[Action, FighterAction] = {
    Action.IDLE: FighterAction(),
    Action.MOVE_LEFT: FighterAction(move_dir=-1),
    Action.MOVE_RIGHT: FighterAction(move_dir=1),
    Action.JUMP: FighterAction(jump=True),
    Action.MOVE_LEFT_JUMP: FighterAction(move_dir=-1, jump=True),
    Action.MOVE_RIGHT_JUMP: FighterAction(move_dir=1, jump=True),
    Action.ATTACK: FighterAction(attack=True),
    Action.MOVE_LEFT_ATTACK: FighterAction(move_dir=-1, attack=True),
    Action.MOVE_RIGHT_ATTACK: FighterAction(move_dir=1, attack=True),
    Action.JUMP_ATTACK: FighterAction(jump=True, attack=True),
    Action.MOVE_LEFT_JUMP_ATTACK: FighterAction(move_dir=-1, jump=True, attack=True),
    Action.MOVE_RIGHT_JUMP_ATTACK: FighterAction(move_dir=1, jump=True, attack=True),
}

# Reverse lookup so human input maps to the same indices the policy uses.
_ACTION_FROM_PARTS: dict[tuple[int, bool, bool], int] = {
    (fa.move_dir, fa.jump, fa.attack): int(action)
    for action, fa in _FIGHTER_ACTIONS.items()
}


def to_fighter_action(action: Action | int) -> FighterAction:
    """Convert a policy action index into the simulation action type."""
    try:
        selected_action = Action(action)
    except ValueError as error:
        raise ValueError(f"Unknown action index: {action}") from error
    return _FIGHTER_ACTIONS[selected_action]


def from_parts(move_dir: int, jump: bool, attack: bool) -> int:
    """Return the action index for a movement, jump, and attack combination."""
    return _ACTION_FROM_PARTS[(move_dir, bool(jump), bool(attack))]
