"""Viewer settings chosen in the menu and remembered between sessions."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from arena_ai.config import Config
from arena_ai.viewer.controllers import (
    REPO_ROOT,
    TRAINED,
    list_checkpoints,
    trained_available,
)

# (stored value, label shown in the menu)
OPPONENT_OPTIONS = (
    (TRAINED, "Trained agent"),
    ("approach", "Scripted: approach"),
    ("parrier", "Scripted: perfect parry"),
    ("idle", "Idle dummy"),
    ("runner", "Scripted: runs away"),
    ("random", "Random"),
)
# The player's side can be you, or any opponent kind. Scripted policies there fight one-on-one.
PLAYER_OPTIONS = (("human", "You"), *OPPONENT_OPTIONS)
DEFAULT_OPPONENT = "approach"

# Generated runtime data lives in data/, which is gitignored.
SETTINGS_PATH = REPO_ROOT / "data" / "viewer_settings.json"


def label_for(options: tuple[tuple[str, str], ...], value: str) -> str:
    """Return the menu label for a stored option value."""
    return dict(options).get(value, value)


@dataclass
class ViewerSettings:
    """Everything the settings menu can change."""

    music_volume: float
    effects_volume: float
    player: str = "human"  # "human", or an opponent kind to watch it play (scripted: 1v1 only)
    opponent_count: int = 1  # 0 is solo practice, with no opponent and no time limit.
    # The kind of each opponent slot, ``observation.max_opponents`` long. Slots at or above
    # ``opponent_count`` are remembered but not used.
    opponents: list[str] = field(default_factory=list)
    trained_greedy: bool = True  # False samples from the policy for variety
    checkpoint: str = ""  # File the trained agent loads. "" means automatic.
    random_spawns: bool = False
    show_controls_hint: bool = True


def _valid(
    settings: ViewerSettings, trained_ok: bool, checkpoints: list[str], max_opponents: int
) -> ViewerSettings:
    """Clamp loaded values and replace choices that cannot be used."""
    settings.music_volume = min(1.0, max(0.0, float(settings.music_volume)))
    settings.effects_volume = min(1.0, max(0.0, float(settings.effects_volume)))
    if settings.player not in dict(PLAYER_OPTIONS) or (
        settings.player == TRAINED and not trained_ok
    ):
        settings.player = "human"
    settings.opponent_count = min(max_opponents, max(0, settings.opponent_count))
    slots = [*settings.opponents[:max_opponents]]
    slots += [DEFAULT_OPPONENT] * (max_opponents - len(slots))
    settings.opponents = [
        kind if kind in dict(OPPONENT_OPTIONS) and (kind != TRAINED or trained_ok)
        else DEFAULT_OPPONENT
        for kind in slots
    ]
    if settings.checkpoint and settings.checkpoint not in checkpoints:
        settings.checkpoint = ""  # The file was renamed or deleted: choose automatically.
    if settings.player != "human" and settings.opponent_count == 0:
        settings.opponent_count = 1  # an AI needs someone to fight
    if settings.player not in ("human", TRAINED):
        settings.opponent_count = 1  # scripted policies only play one-on-ones
    return settings


def load_settings(config: Config, path: Path = SETTINGS_PATH) -> ViewerSettings:
    """Load saved settings over the defaults from ``viewer.yaml``."""
    trained_ok = trained_available(config)
    max_opponents = config.game.observation.max_opponents
    settings = ViewerSettings(
        music_volume=config.viewer.audio.music_volume,
        effects_volume=config.viewer.audio.effects_volume,
        opponents=[TRAINED if trained_ok else DEFAULT_OPPONENT] * max_opponents,
    )
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        # Settings saved before several opponents existed had a single "opponent" key.
        # TODO: remove this migration once no saved settings file has the key.
        if isinstance(saved.get("opponent"), str) and "opponents" not in saved:
            legacy = saved.pop("opponent")
            if legacy == "none":
                saved["opponent_count"] = 0
            else:
                saved["opponents"] = [legacy, *settings.opponents[1:]]
        for name, value in saved.items():
            if hasattr(settings, name) and type(value) is type(getattr(settings, name)):
                setattr(settings, name, value)
    except (OSError, ValueError, AttributeError):
        pass  # No saved file, or an unreadable one: keep the defaults.
    names = [path.name for path in list_checkpoints(config)]
    return _valid(settings, trained_ok, names, max_opponents)


def save_settings(settings: ViewerSettings, path: Path = SETTINGS_PATH) -> None:
    """Write settings to disk; failing to save never interrupts the viewer."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    except OSError:
        pass
