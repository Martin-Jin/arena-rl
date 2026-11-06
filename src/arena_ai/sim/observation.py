"""Build structured observations from simulation state."""

from collections.abc import Mapping, Sequence
from typing import TypedDict

from arena_ai.sim.state import FighterState


class VectorObservation(TypedDict):
    x: float
    y: float
    """Stores the position and velocity of a fighter."""


class FighterObservation(TypedDict):
    position: VectorObservation
    velocity: VectorObservation
    on_ground: bool
    """Stores the observation of a single fighter."""


class Observation(TypedDict):
    agent: FighterObservation
    opponents: dict[str, FighterObservation]
    """Stores the observation of the game state."""


def _fighter_observation(fighter: FighterState) -> FighterObservation:
    """Return the observation of a single fighter."""
    return {
        "position": {"x": fighter.x, "y": fighter.y},
        "velocity": {"x": fighter.vx, "y": fighter.vy},
        "on_ground": fighter.on_ground,
    }


def get_observation(
    fighter: FighterState,
    opponents: Mapping[str, FighterState] | Sequence[FighterState] | FighterState,
    _config: object | None = None,
) -> Observation:
    """Return one agent's state and all opponents as a nested dictionary.

    A mapping preserves stable fighter identifiers. A sequence is assigned
    identifiers in order for callers that do not have IDs yet. A single
    ``FighterState`` is accepted as a convenience for two-fighter games.
    """
    if isinstance(opponents, FighterState):
        opponent_states = {"opponent_0": opponents}
    elif isinstance(opponents, Mapping):
        opponent_states = dict(opponents)
    else:
        opponent_states = {
            f"opponent_{index}": opponent for index, opponent in enumerate(opponents)
        }

    return {
        "agent": _fighter_observation(fighter),
        "opponents": {
            fighter_id: _fighter_observation(opponent)
            for fighter_id, opponent in opponent_states.items()
        },
    }
