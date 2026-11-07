"""Define the discrete action encoding used by the learning agent."""

from dataclasses import replace
from enum import IntEnum
import math

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
    # Parry versions of the twelve actions above: index + 12. Attack and parry
    # cannot overlap, so the simulation lets a parry press win over an attack press.
    PARRY = 12
    MOVE_LEFT_PARRY = 13
    MOVE_RIGHT_PARRY = 14
    JUMP_PARRY = 15
    MOVE_LEFT_JUMP_PARRY = 16
    MOVE_RIGHT_JUMP_PARRY = 17
    ATTACK_PARRY = 18
    MOVE_LEFT_ATTACK_PARRY = 19
    MOVE_RIGHT_ATTACK_PARRY = 20
    JUMP_ATTACK_PARRY = 21
    MOVE_LEFT_JUMP_ATTACK_PARRY = 22
    MOVE_RIGHT_JUMP_ATTACK_PARRY = 23


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
# The parry half of the table is the first half with parry set.
_FIGHTER_ACTIONS.update(
    {
        Action(int(action) + 12): replace(fighter_action, parry=True)
        for action, fighter_action in list(_FIGHTER_ACTIONS.items())
    }
)

# Head 0: index -> move_dir. The only place the move ordering is defined.
MOVE_DIRS = (0, -1, 1)
# Head order: move (none/left/right), jump, attack, parry.
HEAD_SIZES = (len(MOVE_DIRS), 2, 2, 2)
HEAD_NAMES = ("move", "jump", "attack", "parry")
LOGIT_COUNT = sum(HEAD_SIZES)
# Mixed-radix strides, e.g. (3, 2, 2, 2) -> (1, 3, 6, 12).
HEAD_STRIDES = tuple(math.prod(HEAD_SIZES[:i]) for i in range(len(HEAD_SIZES)))


def encode_parts(parts):
    """Pack per-head indices into the flat action index."""
    return sum(part * stride for part, stride in zip(parts, HEAD_STRIDES))


def decode_parts(action):
    """Unpack a flat action index into per-head indices (ints or tensors)."""
    return tuple(
        (action // stride) % size for stride, size in zip(HEAD_STRIDES, HEAD_SIZES)
    )


# Guard: the enum must match the factored encoding.
assert ACTION_COUNT == math.prod(HEAD_SIZES)
assert all(
    int(a)
    == encode_parts(
        (MOVE_DIRS.index(fa.move_dir), int(fa.jump), int(fa.attack), int(fa.parry))
    )
    for a, fa in _FIGHTER_ACTIONS.items()
)
# Reverse lookup so human input maps to the same indices the policy uses.
_ACTION_FROM_PARTS: dict[tuple[int, bool, bool, bool], int] = {
    (fa.move_dir, fa.jump, fa.attack, fa.parry): int(action)
    for action, fa in _FIGHTER_ACTIONS.items()
}


def to_fighter_action(action: Action | int) -> FighterAction:
    """Convert a policy action index into the simulation action type."""
    try:
        selected_action = Action(action)
    except ValueError as error:
        raise ValueError(f"Unknown action index: {action}") from error
    return _FIGHTER_ACTIONS[selected_action]


def from_parts(move_dir: int, jump: bool, attack: bool, parry: bool = False) -> int:
    """Return the action index for a move, jump, attack and parry combination."""
    return _ACTION_FROM_PARTS[(move_dir, bool(jump), bool(attack), bool(parry))]
