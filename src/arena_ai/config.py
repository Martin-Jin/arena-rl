"""Load the application's configuration from the YAML files in ``configs/``."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# CONFIGURATION FOR THE GAME SIMULATION
@dataclass(frozen=True)
class WorldConfig:
    sim_hz: int
    gravity_mps2: float
    arena_width_m: float
    floor_y_m: float

    @property
    def dt_seconds(self) -> float:
        return 1 / self.sim_hz


@dataclass(frozen=True)
class FighterConfig:
    max_health: int
    move_speed_mps: float
    jump_velocity_mps: float
    height_m: float


@dataclass(frozen=True)
class CombatConfig:
    attack_windup_frames: int
    attack_active_frames: int
    attack_recovery_frames: int
    attack_damage: int
    parry_window_frames: int
    parry_recovery_frames: int


@dataclass(frozen=True)
class GameConfig:
    world: WorldConfig
    fighter: FighterConfig
    combat: CombatConfig
    observation: ObservationConfig


# CONFIGURATION FOR THE AI AGENT
@dataclass(frozen=True)
class ObservationConfig:
    spec_version: int
    reaction_delay_frames: int


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for the AI agent."""

    model_path: str
    input_size: int
    hidden_size: int
    output_size: int
    num_layers: int


# CONFIGURATION FOR THE APPLICATION
@dataclass(frozen=True)
class Config:
    """All configuration loaded from the files in the configured directory."""

    game: GameConfig
    agent: NetworkConfig


# Helper functions to load the configuration from YAML files in the configs/ directory.
def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return raw


def _build_game_config(raw: dict[str, Any]) -> GameConfig:
    return GameConfig(
        world=WorldConfig(**raw["world"]),
        fighter=FighterConfig(**raw["fighter"]),
        combat=CombatConfig(**raw["combat"]),
        observation=ObservationConfig(**raw["observation"]),
    )


def _build_network_config(raw: dict[str, Any]) -> NetworkConfig:
    return NetworkConfig(**raw)


def load_config(config_dir: str | Path | None = None) -> Config:
    """Load the game and agent configuration files from one directory.

    The YAML files are loaded together so callers use one configuration
    object rather than loading individual files in each subsystem.
    """
    if config_dir is None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
    config_path = Path(config_dir)
    files = sorted(config_path.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"No YAML configuration files found in {config_path}")

    raw_configs = {path.stem: _load_yaml(path) for path in files}
    if "game" not in raw_configs:
        raise FileNotFoundError(f"Missing game.yaml in {config_path}")

    return Config(
        game=_build_game_config(raw_configs["game"]),
        agent=_build_network_config(raw_configs.get("agent", {})),
    )
