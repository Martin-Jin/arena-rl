"""World-to-screen conversion shared by every viewer screen."""

from collections.abc import Callable

from arena_ai.config import Config

ToScreen = Callable[[float, float], tuple[float, float]]


def make_to_screen(config: Config) -> ToScreen:
    """Return a function converting meters to pixels, with the arena centered."""
    rendering = config.viewer.rendering
    pixels_per_meter = rendering.pixels_per_meter
    arena_px = config.game.world.arena_width_m * pixels_per_meter
    offset_x = (rendering.screen_width - arena_px) / 2
    floor_y = rendering.floor_y_px

    def to_screen(x_m: float, y_m: float) -> tuple[float, float]:
        return offset_x + x_m * pixels_per_meter, floor_y - y_m * pixels_per_meter

    return to_screen
