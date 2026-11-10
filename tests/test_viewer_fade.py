import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from arena_ai.config import load_config
from arena_ai.sim.state import FighterState
from arena_ai.viewer.render.fade import DeathFade
from arena_ai.viewer.render.layout import make_to_screen
from arena_ai.viewer.render.scene import draw_scene

CONFIG = load_config()
SECONDS = CONFIG.viewer.ui.death_fade_seconds


class DeathFadeTests(unittest.TestCase):
    def test_a_living_fighter_is_not_fading(self) -> None:
        fade = DeathFade(CONFIG)
        fade.update(1.0, {"agent": FighterState()})
        self.assertEqual(fade.strengths(), {})

    def test_a_dead_fighter_starts_opaque_and_fades_to_nothing(self) -> None:
        fade = DeathFade(CONFIG)
        fighters = {"agent": FighterState(health=0)}
        fade.update(0.5, fighters)
        self.assertEqual(fade.strengths()["agent"], 1.0)
        fade.update(SECONDS / 2, fighters)
        self.assertAlmostEqual(fade.strengths()["agent"], 0.5)
        fade.update(SECONDS, fighters)
        self.assertEqual(fade.strengths()["agent"], 0.0)

    def test_reset_forgets_the_dead(self) -> None:
        fade = DeathFade(CONFIG)
        fade.update(0.1, {"agent": FighterState(health=0)})
        fade.reset()
        self.assertEqual(fade.strengths(), {})


class DrawDeadFighterTests(unittest.TestCase):
    def setUp(self) -> None:
        pygame.init()
        self.screen = pygame.Surface((1280, 720))
        self.to_screen = make_to_screen(CONFIG)

    def tearDown(self) -> None:
        pygame.quit()

    def _frame(self, strength: float | None) -> bytes:
        fighters = {"agent": FighterState(health=0)}
        fade = None if strength is None else {"agent": strength}
        draw_scene(self.screen, CONFIG, self.to_screen, fighters, 0, death_fade=fade)
        return pygame.image.tobytes(self.screen, "RGB")

    def test_a_dead_fighter_is_drawn_fainter_as_it_fades_and_then_not_at_all(self) -> None:
        empty = self._frame(None)
        full, half, faint = self._frame(1.0), self._frame(0.5), self._frame(0.1)
        self.assertNotEqual(full, empty)
        self.assertEqual(self._frame(0.0), empty)

        def difference(frame: bytes) -> int:
            return sum(abs(a - b) for a, b in zip(frame, empty))

        self.assertGreater(difference(full), difference(half))
        self.assertGreater(difference(half), difference(faint))
        self.assertGreater(difference(faint), 0)


if __name__ == "__main__":
    unittest.main()
