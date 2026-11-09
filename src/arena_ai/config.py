"""Load the application's configuration from the YAML files in ``configs/``."""

from dataclasses import dataclass, field
from enum import Enum
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
    width_to_height_ratio: float

    @property
    def width_m(self) -> float:
        """Fighter body width, derived from height."""
        return self.height_m * self.width_to_height_ratio


@dataclass(frozen=True)
class AttackSpec:
    """Attack timing and damage values for a single attack."""

    attack_windup_frames: int
    attack_active_frames: int
    attack_recovery_frames: int
    damage: int
    range_m: float

    @property
    def attack_total_frames(self) -> int:
        """Length of a full swing: wind-up, active and recovery frames."""
        return self.attack_windup_frames + self.attack_active_frames + self.attack_recovery_frames


class AttackType(str, Enum):
    """The type of attack a fighter is performing."""

    SLASH = "slash"
    # HEAVY = "heavy"


DEFAULT_ATTACK = AttackType.SLASH.value


@dataclass(frozen=True)
class CombatConfig:
    """Attack, parry and counter timing and damage values."""

    # Dictionary of attacks and their specs
    attacks: dict[str, AttackSpec]

    # For parries
    parry_window_frames: int
    parry_recovery_frames: int
    parry_min_window_frames: int
    parry_window_decay: float
    parry_spam_reset_frames: int
    parry_hover_impulse_mps: float
    parry_hover_max_vy_mps: float
    parry_hover_gravity_scale: float
    parry_hover_frames: int
    parried_stagger_frames: int
    # For counters
    counter_damage: int
    counter_window_frames: int
    counter_windup_frames: int
    counter_lunge_speed_mps: float
    counter_hitbox_back_m: float

    def spec(self, attack_type: AttackType) -> AttackSpec:
        """Return the attack spec for this type of attack."""
        return self.attacks[attack_type.value]

    @property
    def basic(self) -> AttackSpec:
        """Return the basic attack spec."""
        return self.attacks[DEFAULT_ATTACK]

    def damage(self, counter: bool) -> int:
        return self.counter_damage if counter else self.basic.damage

    def windup_frames(self, counter: bool) -> int:
        return self.counter_windup_frames if counter else self.basic.attack_windup_frames

    def normal_swing_frame(self, frame: int, counter: bool) -> int:
        if counter:
            return frame + self.basic.attack_windup_frames - self.counter_windup_frames
        return frame

    def swing_total_frames(
        self,
        counter: bool,
    ) -> int:
        basic = self.basic
        return (
            self.windup_frames(counter) + basic.attack_active_frames + basic.attack_recovery_frames
        )

    @property
    def attack_total_frames(self) -> int:
        return self.basic.attack_total_frames

    @property
    def longest_attack_frames(self) -> int:
        """Length of the longest attack: the scale of the observed busy frames."""
        return max(spec.attack_total_frames for spec in self.attacks.values())

    # The longest phase over all attack types: the scale of each observed phase countdown.
    @property
    def longest_windup_frames(self) -> int:
        return max(spec.attack_windup_frames for spec in self.attacks.values())

    @property
    def longest_active_frames(self) -> int:
        return max(spec.attack_active_frames for spec in self.attacks.values())

    @property
    def longest_recovery_frames(self) -> int:
        return max(spec.attack_recovery_frames for spec in self.attacks.values())

    @property
    def parry_total_frames(self) -> int:
        """Length of a rested parry: the longest window plus the recovery."""
        return self.parry_window_frames + self.parry_recovery_frames

    # Backwards compatibility for the old config names, which are used in the game code.
    @property
    def attack_windup_frames(self) -> int:
        return self.basic.attack_windup_frames

    @property
    def attack_damage(self) -> int:
        return self.basic.damage

    @property
    def attack_range_m(self) -> float:
        return self.basic.range_m

    @property
    def attack_active_frames(self) -> int:
        return self.basic.attack_active_frames

    @property
    def attack_recovery_frames(self) -> int:
        return self.basic.attack_recovery_frames


@dataclass(frozen=True)
class ObservationConfig:
    spec_version: int
    reaction_delay_frames: int
    max_observed_height_m: float
    # Opponent slots in the observation. This is the only place the number appears: raising
    # it needs only this value, and a network works for any opponent count up to it.
    max_opponents: int = 3


@dataclass(frozen=True)
class GameConfig:
    world: WorldConfig
    fighter: FighterConfig
    combat: CombatConfig
    observation: ObservationConfig


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for the AI agent."""

    hidden_layers: list[int]
    activation: str
    output_activation: str
    embed_dim: int = 64  # Width of each fighter's embedding in the attention block.
    num_heads: int = 1  # Attention heads. Must divide embed_dim.


@dataclass(frozen=True)
class PolicyConfig:
    """Hyperparameters used by the PPO learner."""

    clip_range: float
    learning_rate: float
    discount_factor: float
    gae_lambda: float
    num_epochs: int
    batch_size: int
    value_loss_coef: float
    entropy_coef: float


@dataclass(frozen=True)
class TrainingConfig:
    """Settings controlling the outer on-policy training loop."""

    iterations: int
    rollout_steps: int
    checkpoint_interval: int
    seed: int | None
    # One entry per parallel environment, each a list of that environment's opponent names
    # (a bare name means one opponent).
    opponents: list = field(default_factory=lambda: [["approach"]])
    # The most environments training allows; ``opponents`` sets how many run.
    num_envs: int = 1
    randomize_spawns: bool = False
    # None keeps the environment default of ten simulated seconds.
    max_episode_steps: int | None = None
    # Number of recent episodes used for the rolling statistics.
    stats_window: int = 100
    # Print the training statistics, and how many iterations apart.
    print_stats: bool = True
    print_interval: int = 10


@dataclass(frozen=True)
class AgentConfig:
    """All configurations for the AI agent."""

    network: NetworkConfig
    policy: PolicyConfig


@dataclass(frozen=True)
class ScoringConfig:
    """Reward values used by training environments."""

    damage_reward_scale: float
    victory_reward: float
    defeat_penalty: float
    edge_margin: float = 1.0
    edge_penalty_scale: float = 0.0
    time_penalty_per_step: float = 0.0
    parry_reward: float = 0.0
    counter_reward: float = 0.0
    missed_parry_penalty: float = 0.0
    missed_attack_penalty: float = 0.0
    timeout_win_reward: float = 0.0
    kill_reward: float = 0.0


@dataclass(frozen=True)
class RenderingConfig:
    """Display and world-to-screen settings used by the pygame viewer."""

    screen_width: int
    screen_height: int
    pixels_per_meter: int
    floor_y_px: int


@dataclass(frozen=True)
class AnimationConfig:
    """Sprite animation timing and source dimensions."""

    source_body_height_px: dict[str, int]
    frame_duration: dict[str, int]
    loop: dict[str, bool]


@dataclass(frozen=True)
class AudioConfig:
    """Viewer audio levels and asset filenames."""

    music_volume: float
    effects_volume: float
    normalization_rms: float
    normalization_peak: float
    running_fade_ms: int
    match_end_fade_ms: int
    effects: dict[str, str]
    music: str


@dataclass(frozen=True)
class ViewerMatchConfig:
    """Rules of a viewer match that are not part of the simulation."""

    time_limit_seconds: float


@dataclass(frozen=True)
class ViewerOpponentConfig:
    """Where the viewer finds the trained agent."""

    checkpoint_path: str


@dataclass(frozen=True)
class ViewerUiConfig:
    """Fonts, colors, HUD layout and animation timings used by the viewer."""

    font_names: list[str]
    font_sizes: dict[str, int]
    colors: dict[str, tuple[int, int, int]]
    overlay_alpha: int
    health_bar_width: int
    health_bar_height: int
    health_bar_margin: int
    health_bar_y: int
    health_trail_delay_seconds: float
    health_trail_speed: float
    ready_seconds: float
    fight_banner_seconds: float
    controls_hint_seconds: float
    damage_text_seconds: float
    hurt_flash_seconds: float
    hurt_flash_min_alpha: int
    parry_bubble_border_px: int
    parry_bubble_radius_scale: float
    parry_bubble_arc_degrees: float
    parry_bubble_burn: float
    parry_flame_amplitude: float
    parry_flame_speed: float
    parry_particle_life_frames: int
    parry_particle_speed_deg: float
    parry_particle_drift: float
    parry_particle_size_px: int
    parry_flash_seconds: float
    parry_flash_radius: int
    counter_effect_seconds: float
    counter_line_width_px: int
    counter_line_carry_m: float


@dataclass(frozen=True)
class ViewerConfig:
    """Configuration for pygame rendering and audio."""

    rendering: RenderingConfig
    animation: AnimationConfig
    audio: AudioConfig
    match: ViewerMatchConfig
    opponent: ViewerOpponentConfig
    ui: ViewerUiConfig


# CONFIGURATION FOR THE APPLICATION
@dataclass(frozen=True)
class Config:
    """All configuration loaded from the files in the configured directory."""

    game: GameConfig
    agent: AgentConfig
    training: TrainingConfig
    scoring: ScoringConfig
    viewer: ViewerConfig


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
        combat=_build_combat_config(raw["combat"]),
        observation=ObservationConfig(**raw["observation"]),
    )


def _build_combat_config(raw: dict[str, Any]) -> CombatConfig:
    raw = dict(raw)
    expected = {t.value for t in AttackType}
    found = set(raw["attacks"])
    if found != expected:
        raise ValueError(
            f"combat.attacks keys must match AttackType: "
            f"unknown {found - expected}, missing {expected - found}"
        )
    raw["attacks"] = {name: AttackSpec(**spec) for name, spec in raw["attacks"].items()}
    return CombatConfig(**raw)


def _build_agent_config(raw: dict[str, Any]) -> AgentConfig:
    return AgentConfig(
        network=NetworkConfig(**raw["network"]),
        policy=PolicyConfig(**raw["policy"]),
    )


def _build_training_config(raw: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(**raw["training"])


def _build_scoring_config(raw: dict[str, Any]) -> ScoringConfig:
    return ScoringConfig(**raw["scoring"])


def _build_viewer_config(raw: dict[str, Any]) -> ViewerConfig:
    ui = dict(raw["ui"])
    ui["colors"] = {name: tuple(value) for name, value in ui["colors"].items()}
    return ViewerConfig(
        rendering=RenderingConfig(**raw["rendering"]),
        animation=AnimationConfig(**raw["animation"]),
        audio=AudioConfig(**raw["audio"]),
        match=ViewerMatchConfig(**raw["match"]),
        opponent=ViewerOpponentConfig(**raw["opponent"]),
        ui=ViewerUiConfig(**ui),
    )


def load_config(config_dir: str | Path | None = None) -> Config:
    """Load the game and agent configuration files from one directory.

    The YAML files are loaded together so callers use one configuration
    object rather than loading individual files in each subsystem. This
    includes the game, agent, and training-scoring settings.
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
    if "agent" not in raw_configs:
        raise FileNotFoundError(f"Missing agent.yaml in {config_path}")
    if "scoring" not in raw_configs:
        raise FileNotFoundError(f"Missing scoring.yaml in {config_path}")
    if "viewer" not in raw_configs:
        raise FileNotFoundError(f"Missing viewer.yaml in {config_path}")

    return Config(
        game=_build_game_config(raw_configs["game"]),
        agent=_build_agent_config(raw_configs["agent"]),
        training=_build_training_config(raw_configs["agent"]),
        scoring=_build_scoring_config(raw_configs["scoring"]),
        viewer=_build_viewer_config(raw_configs["viewer"]),
    )
