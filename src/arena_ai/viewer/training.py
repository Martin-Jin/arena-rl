"""Accelerated pygame visualization for the headless training loop."""

import pygame

from arena_ai.config import Config
from arena_ai.envs.arena_env import ArenaEnv
from arena_ai.viewer.animation import draw_fighter


class TrainingVisualizer:
    """Render each training simulation step without a real-time frame cap."""

    def __init__(self, config: Config) -> None:
        pygame.init()
        self.config = config
        self.screen = pygame.display.set_mode(
            (
                config.viewer.rendering.screen_width,
                config.viewer.rendering.screen_height,
            )
        )
        self.pixels_per_meter = config.viewer.rendering.pixels_per_meter
        self.floor_y = config.viewer.rendering.floor_y_px
        self.tick = 0
        self.closed = False

    def _to_screen(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Convert simulation meters to screen pixels."""
        return (
            x_m * self.pixels_per_meter,
            self.floor_y - y_m * self.pixels_per_meter,
        )

    def __call__(self, environment: ArenaEnv) -> None:
        """Render one step; terminal episodes are not held on an end screen."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
        if self.closed:
            raise KeyboardInterrupt("Training visualization window was closed")
        self.screen.fill(self.config.viewer.ui.background_color)
        if environment.fighters is None:
            return
        for fighter_id, fighter in environment.fighters.items():
            draw_fighter(
                self.screen,
                fighter,
                self.tick,
                self._to_screen,
                shade=(
                    self.config.viewer.ui.opponent_tint
                    if fighter_id == "opponent"
                    else None
                ),
            )
        pygame.display.flip()
        self.tick += 1
        if environment.episode_done:
            # TrainingLoop resets the environment immediately after this
            # callback, so the next callback always renders a fresh episode.
            self.tick = 0

    def close(self) -> None:
        """Close the visualization window."""
        pygame.quit()
