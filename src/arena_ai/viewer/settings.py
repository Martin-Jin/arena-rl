"""Viewer settings chosen in the menu and remembered between sessions."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from arena_ai.config import Config
from arena_ai.viewer.controllers import REPO_ROOT, TRAINED, trained_available

# (stored value, label shown in the menu)
PLAYER_OPTIONS = (("human", "You"), (TRAINED, "Trained agent"))
OPPONENT_OPTIONS = (
    (TRAINED, "Trained agent"),
    ("approach", "Scripted: approach"),
    ("idle", "Idle dummy"),
    ("random", "Random"),
    ("none", "Practice (solo)"),
)

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
    player: str = "human"  # "human" or "trained" (watch the agent play)
    opponent: str = "approach"
    trained_greedy: bool = True  # False samples from the policy for variety
    random_spawns: bool = False
    show_controls_hint: bool = True


def _valid(settings: ViewerSettings, trained_ok: bool) -> ViewerSettings:
    """Clamp loaded values and replace choices that cannot be used."""
    settings.music_volume = min(1.0, max(0.0, float(settings.music_volume)))
    settings.effects_volume = min(1.0, max(0.0, float(settings.effects_volume)))
    if settings.player not in dict(PLAYER_OPTIONS) or (
        settings.player == TRAINED and not trained_ok
    ):
        settings.player = "human"
    if settings.opponent not in dict(OPPONENT_OPTIONS) or (
        settings.opponent == TRAINED and not trained_ok
    ):
        settings.opponent = "approach"
    if settings.player != "human" and settings.opponent == "none":
        settings.opponent = "approach"  # an AI needs someone to fight
    return settings


def load_settings(config: Config, path: Path = SETTINGS_PATH) -> ViewerSettings:
    """Load saved settings over the defaults from ``viewer.yaml``."""
    trained_ok = trained_available(config)
    settings = ViewerSettings(
        music_volume=config.viewer.audio.music_volume,
        effects_volume=config.viewer.audio.effects_volume,
        opponent=TRAINED if trained_ok else "approach",
    )
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        for name, value in saved.items():
            if hasattr(settings, name) and type(value) is type(getattr(settings, name)):
                setattr(settings, name, value)
    except (OSError, ValueError, AttributeError):
        pass  # No saved file, or an unreadable one: keep the defaults.
    return _valid(settings, trained_ok)


def save_settings(settings: ViewerSettings, path: Path = SETTINGS_PATH) -> None:
    """Write settings to disk; failing to save never interrupts the viewer."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    except OSError:
        pass
