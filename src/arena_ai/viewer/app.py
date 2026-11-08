"""The viewer application: window, shared services and the scene stack."""

import pygame

from arena_ai.config import Config
from arena_ai.viewer.audio import AudioManager
from arena_ai.viewer.render.layout import make_to_screen
from arena_ai.viewer.scenes.base import Scene
from arena_ai.viewer.scenes.main_menu import MainMenuScene
from arena_ai.viewer.scenes.match import MatchScene
from arena_ai.viewer.settings import load_settings, save_settings
from arena_ai.viewer.ui.theme import Theme

FRAME_RATE = 60


class App:
    """Owns the window and audio, and forwards each frame to the top scene."""

    def __init__(self, config: Config) -> None:
        pygame.init()
        rendering = config.viewer.rendering
        self.config = config
        self.screen = pygame.display.set_mode(
            (rendering.screen_width, rendering.screen_height)
        )
        pygame.display.set_caption("Arena AI")
        self.clock = pygame.time.Clock()
        self.theme = Theme(config.viewer.ui)
        self.to_screen = make_to_screen(config)
        self.settings = load_settings(config)
        self.audio = AudioManager(
            config.viewer.audio, config.game.observation.max_opponents + 1
        )
        self.audio.set_music_volume(self.settings.music_volume)
        self.audio.set_effects_volume(self.settings.effects_volume)
        self.audio.start_music()
        self.running = True
        self.scenes: list[Scene] = [MainMenuScene(self)]

    @property
    def scene(self) -> Scene:
        return self.scenes[-1]

    def push(self, scene: Scene) -> None:
        """Show a scene on top of the current one (for example settings)."""
        self.scenes.append(scene)

    def pop(self) -> None:
        """Return to the scene underneath."""
        if len(self.scenes) > 1:
            self.scenes.pop()

    def to_main_menu(self) -> None:
        self.audio.cut_effects()
        self.audio.ensure_music()
        self.scenes = [MainMenuScene(self)]

    def start_match(self) -> None:
        self.audio.cut_effects()
        self.audio.ensure_music()
        self.scenes = [MatchScene(self)]

    def quit(self) -> None:
        self.running = False

    def run(self) -> None:
        """Run the main loop until the window is closed or Quit is chosen."""
        try:
            while self.running:
                dt = self.clock.tick(FRAME_RATE) / 1000
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.quit()
                    else:
                        self.scene.handle_event(event)
                self.scene.update(dt)
                self.scene.draw(self.screen)
                pygame.display.flip()
        finally:
            save_settings(self.settings)
            self.audio.close()
            pygame.quit()
