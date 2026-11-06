"""Define the discrete action encoding used by the learning agent."""

from enum import IntEnum

from arena_ai.sim.simulate import FighterAction


class Action(IntEnum):
    """Complete actions available to a fighter in one timestep."""

    # Each index represents a position in the neural network output vector,
    # which is used to select the action to take in the simulation.
    # i.e. The index of the value with the highest
    # probability is used to select the action.
    IDLE = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    JUMP = 3
    MOVE_LEFT_JUMP = 4
    MOVE_RIGHT_JUMP = 5


ACTION_COUNT = len(Action)


def to_fighter_action(action: Action | int) -> FighterAction:
    """Convert a policy action index into the simulation action type."""
    try:
        selected_action = Action(action)
    except ValueError as error:
        raise ValueError(f"Unknown action index: {action}") from error

    # Create a mapping to each action first and then returns selected action index.
    return {
        Action.IDLE: FighterAction(),
        Action.MOVE_LEFT: FighterAction(move_dir=-1),
        Action.MOVE_RIGHT: FighterAction(move_dir=1),
        Action.JUMP: FighterAction(jump=True),
        Action.MOVE_LEFT_JUMP: FighterAction(move_dir=-1, jump=True),
        Action.MOVE_RIGHT_JUMP: FighterAction(move_dir=1, jump=True),
    }[selected_action]
