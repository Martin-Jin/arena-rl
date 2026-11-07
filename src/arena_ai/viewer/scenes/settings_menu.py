"""The settings menu: audio, opponent and display options."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from arena_ai.viewer.controllers import TRAINED, trained_available
from arena_ai.viewer.settings import (
    OPPONENT_OPTIONS,
    save_settings,
)
from arena_ai.viewer.ui.widgets import Button, Menu, Selector, Slider, Toggle

if TYPE_CHECKING:
    from arena_ai.viewer.app import App

PANEL_WIDTH = 640
ROW_HEIGHT = 44
ROW_GAP = 8
SECTION_HEIGHT = 34
SECTION_GAP = 10


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

        # Make it obvious why the trained agent cannot be chosen.
        opponent_options = [
            (value, label + ("" if value != TRAINED or trained_ok else " (no checkpoint)"))
            for value, label in OPPONENT_OPTIONS
        ]
        rows = [
            Slider("Music", lambda: settings.music_volume, set_music),
            Slider(
                "Sound effects",
                lambda: settings.effects_volume,
                set_effects,
                on_release=lambda: app.audio.play("hit"),
            ),
            Selector(
                "Opponent",
                opponent_options,
                lambda: settings.opponent,
                set_attr("opponent"),
                option_enabled=lambda value: value != TRAINED or trained_ok,
            ),
            Selector(
                "Trained agent plays",
                (("greedy", "Best move"), ("sampled", "Varied")),
                lambda: "greedy" if settings.trained_greedy else "sampled",
                lambda value: setattr(settings, "trained_greedy", value == "greedy"),
                enabled_fn=lambda: settings.opponent == TRAINED and trained_ok,
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
        self._sections = [("Audio", rows[0:2]), ("Opponent", rows[2:5]), ("Display", rows[5:6])]
        self._layout()

    def _layout(self) -> None:
        """Place section headers, rows and the back button inside the panel."""
        center_x = self.app.screen.get_width() // 2
        x = center_x - PANEL_WIDTH // 2 + 24
        width = PANEL_WIDTH - 48
        y = 130
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
            center_x - PANEL_WIDTH // 2, 112, PANEL_WIDTH, self.back.rect.bottom + 24 - 112
        )

    def close(self) -> None:
        save_settings(self.app.settings)
        self.app.pop()

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.close()
            return
        self.menu.handle_event(event)

    def update(self, dt: float) -> None:
        pass

    def draw(self, screen: pygame.Surface) -> None:
        theme = self.app.theme
        screen.fill(theme.color("background"))
        center_x = screen.get_width() // 2
        theme.text(screen, "Settings", "heading", theme.color("text"), (center_x, 62), "center")
        theme.draw_panel(screen, self.panel)
        for title, y in self._headers:
            theme.text(
                screen, title, "small", theme.color("text_muted"), (self.panel.x + 28, y + 6)
            )
        self.menu.draw(screen, theme)
