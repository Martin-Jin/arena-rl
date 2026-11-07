"""The start menu."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from arena_ai import __version__
from arena_ai.sim.state import FighterState
from arena_ai.viewer.render.scene import draw_scene
from arena_ai.viewer.scenes.settings_menu import SettingsScene
from arena_ai.viewer.settings import OPPONENT_OPTIONS, PLAYER_OPTIONS, label_for
from arena_ai.viewer.ui.widgets import Button, Menu, stack

if TYPE_CHECKING:
    from arena_ai.viewer.app import App

BUTTON_WIDTH = 300
BUTTON_HEIGHT = 56


class MainMenuScene:
    """Title, three buttons and two idle fighters facing off."""

    def __init__(self, app: App) -> None:
        self.app = app
        self.play = Button("Play", app.start_match, primary=True)
        self.menu = Menu(
            [
                self.play,
                Button("Settings", lambda: app.push(SettingsScene(app))),
                Button("Quit", app.quit),
            ]
        )
        center_x = app.screen.get_width() // 2
        self._buttons_bottom = stack(
            self.menu.widgets, center_x, 290, BUTTON_WIDTH, BUTTON_HEIGHT, 14
        )
        # Two fighters standing off, drawn with the same idle animation as a match.
        self.fighters = {
            "agent": FighterState(x=3.0, facing_dir=1),
            "opponent": FighterState(x=9.0, facing_dir=-1),
        }
        self.tick = 0

    def handle_event(self, event: pygame.event.Event) -> None:
        self.menu.handle_event(event)

    def update(self, dt: float) -> None:
        self.tick += 1
        # Settings can change while this scene sits underneath them.
        self.play.label = "Play" if self.app.settings.player == "human" else "Watch"

    def _matchup_text(self) -> str:
        settings = self.app.settings
        player = label_for(PLAYER_OPTIONS, settings.player)
        opponent = label_for(OPPONENT_OPTIONS, settings.opponent)
        if settings.opponent == "none":
            return f"{player}  ·  {opponent}"
        return f"{player}  vs  {opponent}"

    def draw(self, screen: pygame.Surface) -> None:
        theme = self.app.theme
        draw_scene(
            screen, self.app.config, self.app.to_screen, self.fighters, self.tick
        )
        center_x = screen.get_width() // 2
        theme.text(screen, "ARENA AI", "title", theme.color("text"), (center_x, 150), "center")
        theme.text(
            screen,
            "A 2D duel against a reinforcement-learning opponent",
            "small",
            theme.color("text_muted"),
            (center_x, 212),
            "center",
        )
        self.menu.draw(screen, theme)
        theme.text(
            screen,
            self._matchup_text(),
            "small",
            theme.color("text_muted"),
            (center_x, self._buttons_bottom + 28),
            "center",
        )
        theme.text(
            screen,
            f"v{__version__}",
            "small",
            theme.color("text_muted"),
            (screen.get_width() - 16, screen.get_height() - 12),
            "bottomright",
        )
