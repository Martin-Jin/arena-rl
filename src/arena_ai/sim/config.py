# config.py
# This module defines all available configurable parameters for the game loop and its entities.

from dataclasses import dataclass
from pathlib import Path

import yaml


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


# Store all configurable game data.
@dataclass(frozen=True)
class GameConfig:
    world: WorldConfig
    fighter: FighterConfig
    combat: CombatConfig


# Load the game configuration from YAML and unpack it into the corresponding data classes.
def load_config(path: str | Path) -> GameConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return GameConfig(
        world=WorldConfig(**raw["world"]),
        fighter=FighterConfig(**raw["fighter"]),
        combat=CombatConfig(**raw["combat"]),
    )
