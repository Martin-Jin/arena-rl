"""Fonts, colors and text/panel drawing helpers shared by the viewer screens."""

import pygame

from arena_ai.config import ViewerUiConfig

Color = tuple[int, int, int]


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
