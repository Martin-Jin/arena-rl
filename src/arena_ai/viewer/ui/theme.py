"""Fonts, colors and text/panel drawing helpers shared by the viewer screens."""

import pygame

from arena_ai.config import ViewerUiConfig

Color = tuple[int, int, int]

SCROLL_SPEED = 40.0  # Pixels per second that too-long text scrolls sideways.
SCROLL_PAUSE_MS = 900  # How long scrolling text rests at each end.


def marquee_offset(overflow: float, elapsed_ms: int) -> float:
    """Return how far to shift text that is ``overflow`` pixels too wide.

    The text rests at the start, scrolls to its end, rests, and scrolls back, so
    both ends can be read. ``elapsed_ms`` is any steadily increasing clock.
    """
    if overflow <= 0:
        return 0.0
    travel_ms = overflow / SCROLL_SPEED * 1000
    cycle = 2 * (travel_ms + SCROLL_PAUSE_MS)
    t = elapsed_ms % cycle
    if t < SCROLL_PAUSE_MS:
        return 0.0
    if t < SCROLL_PAUSE_MS + travel_ms:
        return (t - SCROLL_PAUSE_MS) / 1000 * SCROLL_SPEED
    if t < 2 * SCROLL_PAUSE_MS + travel_ms:
        return overflow
    return overflow - (t - 2 * SCROLL_PAUSE_MS - travel_ms) / 1000 * SCROLL_SPEED


class Theme:
    """Look up configured colors and fonts and draw themed text and panels."""

    # Large fonts are bold so titles stay legible against the light background.
    _BOLD_FONTS = ("title", "heading")

    def __init__(self, ui: ViewerUiConfig) -> None:
        self.ui = ui
        names = ",".join(ui.font_names)
        self._fonts = {
            name: pygame.font.SysFont(names, size, bold=name in self._BOLD_FONTS)
            for name, size in ui.font_sizes.items()
        }
        self._overlays: dict[tuple[int, int], pygame.Surface] = {}

    def color(self, name: str) -> Color:
        """Return a named color from ``viewer.yaml``."""
        return self.ui.colors[name]

    def text(
        self,
        surface: pygame.Surface,
        string: str,
        font: str,
        color: Color,
        pos: tuple[float, float],
        anchor: str = "topleft",
        alpha: int = 255,
    ) -> pygame.Rect:
        """Draw anti-aliased text placed by ``anchor`` (a ``pygame.Rect`` attribute)."""
        image = self._fonts[font].render(string, True, color)
        if alpha < 255:
            image.set_alpha(max(0, alpha))
        rect = image.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
        surface.blit(image, rect)
        return rect

    def fit_text(
        self,
        surface: pygame.Surface,
        string: str,
        font: str,
        color: Color,
        rect: pygame.Rect,
        align: str = "left",
        alpha: int = 255,
    ) -> None:
        """Draw one line of text inside ``rect`` so that it never leaves it.

        Text that fits is placed by ``align`` ("left", "center" or "right"). Text
        that is too wide is clipped to ``rect`` and scrolls sideways by itself, so
        every part of it can be read. Use this wherever the text length is not
        under our control (names, labels, messages).
        """
        image = self._fonts[font].render(string, True, color)
        if alpha < 255:
            image.set_alpha(max(0, alpha))
        y = rect.centery - image.get_height() // 2
        overflow = image.get_width() - rect.width
        if overflow <= 0:
            x = {
                "left": rect.x,
                "center": rect.centerx - image.get_width() // 2,
                "right": rect.right - image.get_width(),
            }[align]
            surface.blit(image, (x, y))
            return
        previous = surface.get_clip()
        surface.set_clip(rect.clip(previous))
        offset = marquee_offset(overflow, pygame.time.get_ticks())
        surface.blit(image, (rect.x - round(offset), y))
        surface.set_clip(previous)

    def draw_overlay(self, surface: pygame.Surface) -> None:
        """Dim whatever is behind a pause or result panel."""
        size = surface.get_size()
        if size not in self._overlays:
            overlay = pygame.Surface(size, pygame.SRCALPHA)
            overlay.fill((*self.color("overlay"), self.ui.overlay_alpha))
            self._overlays[size] = overlay
        surface.blit(self._overlays[size], (0, 0))

    def draw_panel(
        self, surface: pygame.Surface, rect: pygame.Rect, radius: int = 18
    ) -> None:
        """Draw a white rounded panel with a thin border."""
        pygame.draw.rect(surface, self.color("panel"), rect, border_radius=radius)
        pygame.draw.rect(
            surface, self.color("panel_border"), rect, width=1, border_radius=radius
        )
