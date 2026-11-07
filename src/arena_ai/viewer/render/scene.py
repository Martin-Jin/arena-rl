"""Draw the arena and its fighters. Used by the match, menus and training view."""

from collections.abc import Mapping

import pygame

from arena_ai.config import Config
from arena_ai.sim.state import FighterState
from arena_ai.viewer.render.animation import draw_fighter
from arena_ai.viewer.render.layout import ToScreen


def draw_arena(screen: pygame.Surface, config: Config, to_screen: ToScreen) -> None:
    """Fill the background and draw the floor with the arena's end markers."""
    colors = config.viewer.ui.colors
    screen.fill(colors["background"])
    left, floor_y = (round(value) for value in to_screen(0.0, 0.0))
    right = round(to_screen(config.game.world.arena_width_m, 0.0)[0])
    pygame.draw.rect(
        screen,
        colors["floor"],
        (0, floor_y, screen.get_width(), screen.get_height() - floor_y),
    )
    pygame.draw.line(screen, colors["floor_line"], (left, floor_y), (right, floor_y), 3)
    for x in (left, right):
        pygame.draw.line(screen, colors["floor_line"], (x, floor_y - 14), (x, floor_y), 3)


def _draw_shadow(
    screen: pygame.Surface,
    config: Config,
    fighter: FighterState,
    to_screen: ToScreen,
) -> None:
    """Draw a soft shadow that shrinks and fades as the fighter jumps."""
    pixels_per_meter = config.viewer.rendering.pixels_per_meter
    lift = min(max(fighter.y, 0.0) / 1.2, 1.0)
    width = max(8, round(config.game.fighter.width_m * pixels_per_meter * 1.8 * (1 - 0.45 * lift)))
    height = max(4, round(width * 0.22))
    alpha = round(70 * (1 - 0.6 * lift))
    shadow = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.ellipse(
        shadow, (*config.viewer.ui.colors["shadow"], alpha), shadow.get_rect()
    )
    x, y = to_screen(fighter.x, 0.0)
    screen.blit(shadow, shadow.get_rect(center=(round(x), round(y))))


def draw_scene(
    screen: pygame.Surface,
    config: Config,
    to_screen: ToScreen,
    fighters: Mapping[str, FighterState],
    tick: int,
    turning: bool = False,
) -> None:
    """Draw the arena and every fighter. Only the agent can show a turn frame."""
    colors = config.viewer.ui.colors
    draw_arena(screen, config, to_screen)
    for fighter in fighters.values():
        _draw_shadow(screen, config, fighter, to_screen)
    pixels_per_meter = config.viewer.rendering.pixels_per_meter
    for fighter_id, fighter in fighters.items():
        is_opponent = fighter_id == "opponent"
        drawn = draw_fighter(
            screen,
            fighter,
            tick,
            to_screen,
            turning=turning and not is_opponent,
            shade=colors["opponent_tint"] if is_opponent else None,
        )
        if not drawn:  # No sprite frames available: draw a plain box instead.
            height = config.game.fighter.height_m * pixels_per_meter
            width = config.game.fighter.width_m * pixels_per_meter
            x, y = to_screen(fighter.x, fighter.y)
            color = colors["fallback_opponent" if is_opponent else "fallback_agent"]
            pygame.draw.rect(screen, color, (x - width / 2, y - height, width, height))
