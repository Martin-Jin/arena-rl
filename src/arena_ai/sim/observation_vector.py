"""Flatten and normalize structured simulation observations for neural networks."""

from collections.abc import Mapping

from arena_ai.sim.observation import Observation


FEATURES_PER_FIGHTER = 5


# TODO: Normalize positions, velocities, and future health/combat features.
def observation_size(opponent_count: int = 1) -> int:
    """Return the encoded vector size for the configured opponent count."""
    if opponent_count < 1:
        raise ValueError("opponent_count must be at least 1")
    return FEATURES_PER_FIGHTER * (opponent_count + 1)


def _fighter_vector(fighter: Mapping[str, object]) -> list[float]:
    """Encode one fighter's position, velocity, and grounded state."""
    position = fighter["position"]
    velocity = fighter["velocity"]
    if not isinstance(position, Mapping) or not isinstance(velocity, Mapping):
        raise TypeError("Fighter position and velocity must be mappings")
    # Order: x, y, vx, vy, on_ground.
    return [
        float(position["x"]),
        float(position["y"]),
        float(velocity["x"]),
        float(velocity["y"]),
        float(bool(fighter["on_ground"])),
    ]


def encode_observation(observation: Observation) -> list[float]:
    """Return a fixed-order vector containing the agent and its opponents."""
    vector = _fighter_vector(observation["agent"])
    opponents = observation["opponents"]
    for opponent_id in sorted(opponents):
        vector.extend(_fighter_vector(opponents[opponent_id]))
    return vector
