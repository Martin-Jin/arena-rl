"""Accelerated pygame visualization for the headless training loop."""

import pygame

from arena_ai.config import Config
from arena_ai.envs.arena_env import ArenaEnv
from arena_ai.viewer.render.layout import make_to_screen
from arena_ai.viewer.render.scene import draw_scene


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
        self.to_screen = make_to_screen(config)
        self.tick = 0
        self.closed = False

    def __call__(self, environment: ArenaEnv) -> None:
        """Render one step; terminal episodes are not held on an end screen."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
        if self.closed:
            raise KeyboardInterrupt("Training visualization window was closed")
        if environment.fighters is None:
            return
        draw_scene(
            self.screen, self.config, self.to_screen, environment.fighters, self.tick
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
