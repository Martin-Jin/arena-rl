"""The settings menu: audio, opponent and display options."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from arena_ai.viewer.controllers import (
    TRAINED,
    list_checkpoints,
    resolve_checkpoint,
    trained_available,
)
from arena_ai.viewer.settings import OPPONENT_OPTIONS, PLAYER_OPTIONS, save_settings
from arena_ai.viewer.ui.widgets import (
    Button,
    Dropdown,
    Menu,
    Selector,
    Slider,
    Toggle,
)

if TYPE_CHECKING:
    from arena_ai.viewer.app import App

PANEL_WIDTH = 640
ROW_HEIGHT = 40
ROW_GAP = 6
SECTION_HEIGHT = 30
SECTION_GAP = 8
MAX_CHECKPOINTS_SHOWN = 8  # Keeps the dropdown list on the screen.


class SettingsScene:
    """Edits ``app.settings`` live and saves them when the menu is closed."""

    def __init__(self, app: App) -> None:
        self.app = app
        settings = app.settings
        trained_ok = trained_available(app.config)

        def set_music(value: float) -> None:
            settings.music_volume = value
            app.audio.set_music_volume(value)

        def set_effects(value: float) -> None:
            settings.effects_volume = value
            app.audio.set_effects_volume(value)

        def set_attr(name: str):
            return lambda value: setattr(settings, name, value)

        def with_note(options, suffix: str = ""):
            """Label the trained-agent choice, and say why if it is unavailable."""
            return [
                (
                    value,
                    label
                    + (suffix if value == TRAINED else "")
                    + (" (no checkpoint)" if value == TRAINED and not trained_ok else ""),
                )
                for value, label in options
            ]

        def set_player(value: str) -> None:
            settings.player = value
            if value != "human" and settings.opponent == "none":
                settings.opponent = "approach"  # An AI needs someone to fight.

        def trained_plays() -> bool:
            return trained_ok and TRAINED in (settings.player, settings.opponent)

        automatic = resolve_checkpoint(app.config)
        shown = [path.name for path in list_checkpoints(app.config)[:MAX_CHECKPOINTS_SHOWN]]
        if settings.checkpoint and settings.checkpoint not in shown:
            shown.append(settings.checkpoint)  # Always list the file in use.
        checkpoint_options = [
            ("", f"Automatic ({automatic.name})" if automatic else "Automatic"),
            *[(name, name) for name in shown],
        ]
        rows = [
            Slider("Music", lambda: settings.music_volume, set_music),
            Slider(
                "Sound effects",
                lambda: settings.effects_volume,
                set_effects,
                on_release=lambda: app.audio.play("hit"),
            ),
            Dropdown(
                "Player",
                with_note(PLAYER_OPTIONS, " (watch)"),
                lambda: settings.player,
                set_player,
                option_enabled=lambda value: value != TRAINED or trained_ok,
            ),
            Dropdown(
                "Opponent",
                with_note(OPPONENT_OPTIONS),
                lambda: settings.opponent,
                set_attr("opponent"),
                option_enabled=lambda value: (value != TRAINED or trained_ok)
                and not (value == "none" and settings.player != "human"),
            ),
            Dropdown(
                "Checkpoint",
                checkpoint_options,
                lambda: settings.checkpoint,
                set_attr("checkpoint"),
                enabled_fn=trained_plays,
            ),
            Selector(
                "Trained agent plays",
                (("greedy", "Best move"), ("sampled", "Varied")),
                lambda: "greedy" if settings.trained_greedy else "sampled",
                lambda value: setattr(settings, "trained_greedy", value == "greedy"),
                enabled_fn=trained_plays,
            ),
            Toggle(
                "Random starting positions",
                lambda: settings.random_spawns,
                set_attr("random_spawns"),
            ),
            Toggle(
                "Show controls hint",
                lambda: settings.show_controls_hint,
                set_attr("show_controls_hint"),
            ),
        ]
        self.back = Button("Back", self.close, primary=True)
        self.menu = Menu([*rows, self.back])
        self._sections = [("Audio", rows[0:2]), ("Match", rows[2:7]), ("Display", rows[7:8])]
        self._layout()

    def _layout(self) -> None:
        """Place section headers, rows and the back button inside the panel."""
        center_x = self.app.screen.get_width() // 2
        x = center_x - PANEL_WIDTH // 2 + 24
        width = PANEL_WIDTH - 48
        y = 102
        self._headers: list[tuple[str, int]] = []
        for title, rows in self._sections:
            self._headers.append((title.upper(), y))
            y += SECTION_HEIGHT
            for row in rows:
                row.rect = pygame.Rect(x, y, width, ROW_HEIGHT)
                y += ROW_HEIGHT + ROW_GAP
            y += SECTION_GAP
        self.back.rect = pygame.Rect(0, 0, 200, 50)
        self.back.rect.midtop = (center_x, y + 4)
        self.panel = pygame.Rect(
            center_x - PANEL_WIDTH // 2, 84, PANEL_WIDTH, self.back.rect.bottom + 24 - 84
        )

    def close(self) -> None:
        save_settings(self.app.settings)
        self.app.pop()

    def handle_event(self, event: pygame.event.Event) -> None:
        escape = event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        if escape and not self.menu.captures_input:  # Escape first closes a dropdown.
            self.close()
            return
        self.menu.handle_event(event)

    def update(self, dt: float) -> None:
        pass

    def draw(self, screen: pygame.Surface) -> None:
        theme = self.app.theme
        screen.fill(theme.color("background"))
        center_x = screen.get_width() // 2
        theme.text(screen, "Settings", "heading", theme.color("text"), (center_x, 44), "center")
        theme.draw_panel(screen, self.panel)
        for title, y in self._headers:
            theme.text(
                screen, title, "small", theme.color("text_muted"), (self.panel.x + 28, y + 6)
            )
        self.menu.draw(screen, theme)
