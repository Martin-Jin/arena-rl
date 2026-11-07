import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from arena_ai.config import load_config
from arena_ai.viewer.ui.theme import SCROLL_PAUSE_MS, Theme, marquee_offset


class MarqueeTests(unittest.TestCase):
    def test_text_that_fits_does_not_move(self) -> None:
        self.assertEqual(marquee_offset(0, 12345), 0.0)
        self.assertEqual(marquee_offset(-5, 12345), 0.0)

    def test_overflowing_text_rests_scrolls_to_the_end_and_comes_back(self) -> None:
        overflow = 80.0  # Two seconds of travel at 40 px/s.
        self.assertEqual(marquee_offset(overflow, 0), 0.0)
        self.assertEqual(marquee_offset(overflow, SCROLL_PAUSE_MS - 1), 0.0)
        halfway = SCROLL_PAUSE_MS + 1000
        self.assertAlmostEqual(marquee_offset(overflow, halfway), 40.0)
        at_end = SCROLL_PAUSE_MS + 2000 + 100
        self.assertEqual(marquee_offset(overflow, at_end), overflow)
        cycle = 2 * (2000 + SCROLL_PAUSE_MS)
        self.assertEqual(marquee_offset(overflow, cycle), 0.0)  # Then it repeats.
        for ms in range(0, cycle, 50):
            self.assertTrue(0.0 <= marquee_offset(overflow, ms) <= overflow)


class FitTextTests(unittest.TestCase):
    def setUp(self) -> None:
        pygame.init()
        self.theme = Theme(load_config().viewer.ui)

    def tearDown(self) -> None:
        pygame.quit()

    def _outside_untouched(self, text: str, rect: pygame.Rect, align: str) -> bool:
        surface = pygame.Surface((500, 120))
        surface.fill((255, 255, 255))
        self.theme.fit_text(surface, text, "body", (0, 0, 0), rect, align)
        for x in range(surface.get_width()):
            for y in range(surface.get_height()):
                if not rect.collidepoint(x, y) and surface.get_at((x, y))[:3] != (255, 255, 255):
                    return False
        return True

    def test_long_text_is_clipped_to_its_container(self) -> None:
        rect = pygame.Rect(150, 40, 80, 40)
        long_text = "A very long checkpoint name that cannot fit in eighty pixels"
        for align in ("left", "center", "right"):
            self.assertTrue(self._outside_untouched(long_text, rect, align))

    def test_short_text_is_drawn_inside_and_the_clip_is_restored(self) -> None:
        surface = pygame.Surface((500, 120))
        surface.fill((255, 255, 255))
        clip_before = surface.get_clip()
        rect = pygame.Rect(100, 40, 300, 40)
        self.theme.fit_text(surface, "Short", "body", (0, 0, 0), rect, "center")
        self.theme.fit_text(surface, "x" * 200, "body", (0, 0, 0), rect)
        self.assertEqual(surface.get_clip(), clip_before)


if __name__ == "__main__":
    unittest.main()
