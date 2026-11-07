"""Flatten and normalize structured simulation observations for neural networks."""

from collections.abc import Mapping

from arena_ai.config import Config, load_config
from arena_ai.sim.observation import Observation

# Update when new features are added to the fighter observation structure.
FEATURES_PER_FIGHTER = 8


def observation_size(opponent_count: int = 1) -> int:
    """Return the encoded vector size for the configured opponent count."""
    if opponent_count < 1:
        raise ValueError("opponent_count must be at least 1")
    return FEATURES_PER_FIGHTER * (opponent_count + 1)


def _normalize(value: float, maximum: float) -> float:
    """Scale a value by a fixed positive maximum and clamp it to [-1, 1]."""
    if maximum <= 0:
        raise ValueError(f"Normalization maximum must be positive, got {maximum}")
    return max(-1.0, min(1.0, value / maximum))


def _fighter_vector(
    fighter: Mapping[str, object],
    config: Config,
) -> list[float]:
    """Encode one fighter's position, velocity, and grounded state."""
    position = fighter["position"]
    velocity = fighter["velocity"]
    if not isinstance(position, Mapping) or not isinstance(velocity, Mapping):
        raise TypeError("Fighter position and velocity must be mappings")
    world = config.game.world
    fighter_config = config.game.fighter
    observation_config = config.game.observation

    # Order: normalized x, y, vx, vy, on_ground, facing_dir, health.
    return [
        _normalize(float(position["x"]), world.arena_width_m),
        _normalize(
            float(position["y"]) - world.floor_y_m,
            observation_config.max_observed_height_m,
        ),
        _normalize(float(velocity["x"]), fighter_config.move_speed_mps),
        _normalize(float(velocity["y"]), fighter_config.jump_velocity_mps),
        float(bool(fighter["on_ground"])),
        float(fighter["facing_dir"]),
        _normalize(float(fighter["health"]), fighter_config.max_health),
        _normalize(float(fighter["attack_frame"]), fighter_config.attack_max_frame),
    ]


def encode_observation(
    observation: Observation,
    config: Config | None = None,
) -> list[float]:
    """Return a fixed-order vector containing the agent and its opponents."""
    if config is None:
        config = load_config()
    vector = _fighter_vector(observation["agent"], config)
    opponents = observation["opponents"]
    for opponent_id in sorted(opponents):
        vector.extend(_fighter_vector(opponents[opponent_id], config))
    return vector
