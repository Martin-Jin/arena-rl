"""Flatten and normalize structured simulation observations for neural networks."""

from collections.abc import Mapping, Sequence

from arena_ai.config import Config, load_config
from arena_ai.sim.observation import Observation
from arena_ai.sim.state import sort_key

# The features of one fighter, in vector order. Add new features at the end, and bump
# ``observation.spec_version`` in configs/game.yaml: old checkpoints no longer fit.
FEATURE_NAMES = (
    "x",
    "y",
    "vx",
    "vy",
    "on_ground",
    "facing_dir",
    "health",
    "attack_frame",
    "attack_counter",
    "parry_frame",
    "parry_window",
    "counter_frames",
    "stagger_frames",
    "parry_held",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
FEATURES_PER_FIGHTER = len(FEATURE_NAMES)


def observation_size(max_opponents: int) -> int:
    """Return the vector size: the acting fighter, ``max_opponents`` slots and the mask."""
    if max_opponents < 1:
        raise ValueError("max_opponents must be at least 1")
    return FEATURES_PER_FIGHTER * (max_opponents + 1) + max_opponents


def fighter_features(vector: Sequence[float], slot: int) -> dict[str, float]:
    """Return the named features of one row: slot 0 is the acting fighter, 1.. opponents."""
    start = slot * FEATURES_PER_FIGHTER
    row = vector[start : start + FEATURES_PER_FIGHTER]
    if len(row) != FEATURES_PER_FIGHTER:
        raise ValueError(f"observation has no fighter slot {slot}")
    return dict(zip(FEATURE_NAMES, row))


def slot_count(vector: Sequence[float]) -> int:
    """Return how many opponent slots a vector has (its ``max_opponents``)."""
    slots, remainder = divmod(len(vector) - FEATURES_PER_FIGHTER, FEATURES_PER_FIGHTER + 1)
    if slots < 1 or remainder:
        raise ValueError(f"{len(vector)} values is not a valid observation size")
    return slots


def opponent_mask(vector: Sequence[float], max_opponents: int) -> list[float]:
    """Return the mask: 1.0 for an opponent slot that holds a living opponent, else 0.0."""
    start = FEATURES_PER_FIGHTER * (max_opponents + 1)
    return list(vector[start : start + max_opponents])


def _normalize(value: float, maximum: float) -> float:
    """Scale a value by a fixed positive maximum and clamp it to [-1, 1]."""
    if maximum <= 0:
        raise ValueError(f"Normalization maximum must be positive, got {maximum}")
    return max(-1.0, min(1.0, value / maximum))


def _fighter_vector(
    fighter: Mapping[str, object],
    config: Config,
) -> list[float]:
    """Encode one fighter's state as ``FEATURES_PER_FIGHTER`` normalized values."""
    position = fighter["position"]
    velocity = fighter["velocity"]
    if not isinstance(position, Mapping) or not isinstance(velocity, Mapping):
        raise TypeError("Fighter position and velocity must be mappings")
    world = config.game.world
    fighter_config = config.game.fighter
    observation_config = config.game.observation

    combat = config.game.combat
    values = {
        "x": _normalize(float(position["x"]), world.arena_width_m),
        "y": _normalize(
            float(position["y"]) - world.floor_y_m,
            observation_config.max_observed_height_m,
        ),
        "vx": _normalize(float(velocity["x"]), fighter_config.move_speed_mps),
        "vy": _normalize(float(velocity["y"]), fighter_config.jump_velocity_mps),
        "on_ground": float(bool(fighter["on_ground"])),
        "facing_dir": float(fighter["facing_dir"]),
        "health": _normalize(float(fighter["health"]), fighter_config.max_health),
        "attack_frame": _normalize(float(fighter["attack_frame"]), combat.attack_total_frames),
        "attack_counter": float(bool(fighter["attack_counter"])),
        "parry_frame": _normalize(float(fighter["parry_frame"]), combat.parry_total_frames),
        "parry_window": _normalize(float(fighter["parry_window"]), combat.parry_window_frames),
        "counter_frames": _normalize(
            float(fighter["counter_frames"]), combat.counter_window_frames
        ),
        "stagger_frames": _normalize(
            float(fighter["stagger_frames"]), combat.parried_stagger_frames
        ),
        "parry_held": float(bool(fighter["parry_held"])),
    }
    return [values[name] for name in FEATURE_NAMES]


def encode_observation(
    observation: Observation,
    config: Config | None = None,
) -> list[float]:
    """Return the vector ``[acting fighter][opponent slot 0..K-1][mask 0..K-1]``.

    ``K`` is ``observation.max_opponents``. Opponents are sorted by their id's natural
    order (``opponent_10`` after ``opponent_2``) and fill the slots from the first. A slot
    with no opponent, or a dead one, is all zeros with mask 0.0, so the network can tell
    an absent opponent from one standing at x = 0. Fewer opponents than slots is fine, so
    one network plays against any number up to ``K``.
    """
    if config is None:
        config = load_config()
    max_opponents = config.game.observation.max_opponents
    opponents = observation["opponents"]
    if len(opponents) > max_opponents:
        raise ValueError(
            f"{len(opponents)} opponents do not fit observation.max_opponents={max_opponents}"
        )
    vector = _fighter_vector(observation["agent"], config)
    mask = []
    for opponent_id in sorted(opponents, key=sort_key):
        alive = opponents[opponent_id]["health"] > 0
        vector.extend(
            _fighter_vector(opponents[opponent_id], config)
            if alive
            else [0.0] * FEATURES_PER_FIGHTER
        )
        mask.append(1.0 if alive else 0.0)
    padding = max_opponents - len(opponents)
    vector.extend([0.0] * (padding * FEATURES_PER_FIGHTER))
    return vector + mask + [0.0] * padding
