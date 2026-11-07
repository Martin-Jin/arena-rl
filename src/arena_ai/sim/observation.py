"""Build structured observations from simulation state."""

from collections.abc import Mapping, Sequence
from typing import TypedDict

from arena_ai.config import CombatConfig, Config, load_config
from arena_ai.sim.rules import parry_window_for_observation
from arena_ai.sim.state import FighterState


class VectorObservation(TypedDict):
    """Stores the position or velocity of a fighter."""

    x: float
    y: float


class FighterObservation(TypedDict):
    """Stores the observation of a single fighter."""

    position: VectorObservation
    velocity: VectorObservation
    on_ground: bool
    facing_dir: int
    health: int
    attack_frame: int
    attack_counter: bool
    parry_frame: int
    parry_window: int
    counter_frames: int
    stagger_frames: int
    parry_held: bool


class Observation(TypedDict):
    """Stores the observation of the game state."""

    agent: FighterObservation
    opponents: dict[str, FighterObservation]


def _fighter_observation(
    fighter: FighterState, combat: CombatConfig
) -> FighterObservation:
    """Return the observation of a single fighter."""
    return {
        "position": {"x": fighter.x, "y": fighter.y},
        "velocity": {"x": fighter.vx, "y": fighter.vy},
        "on_ground": fighter.on_ground,
        "facing_dir": fighter.facing_dir,
        "health": fighter.health,
        "attack_frame": fighter.attack_frame,
        "attack_counter": fighter.attack_counter,
        "parry_frame": fighter.parry_frame,
        # Protected frames of the running parry, or of the next one: shows the spam penalty.
        "parry_window": parry_window_for_observation(fighter, combat),
        "counter_frames": fighter.counter_frames,
        "stagger_frames": fighter.stagger_frames,
        "parry_held": fighter.parry_held,
    }


def get_observation(
    fighter: FighterState,
    opponents: Mapping[str, FighterState] | Sequence[FighterState] | FighterState,
    config: Config | None = None,
) -> Observation:
    """Return one agent's state and all opponents as a nested dictionary.

    A mapping preserves stable fighter identifiers. A sequence is assigned
    identifiers in order for callers that do not have IDs yet. A single
    ``FighterState`` is accepted as a convenience for two-fighter games.
    """
    combat = (config or load_config()).game.combat
    if isinstance(opponents, FighterState):
        opponent_states = {"opponent_0": opponents}
    elif isinstance(opponents, Mapping):
        opponent_states = dict(opponents)
    else:
        opponent_states = {
            f"opponent_{index}": opponent for index, opponent in enumerate(opponents)
        }

    return {
        "agent": _fighter_observation(fighter, combat),
        "opponents": {
            fighter_id: _fighter_observation(opponent, combat)
            for fighter_id, opponent in opponent_states.items()
        },
    }
