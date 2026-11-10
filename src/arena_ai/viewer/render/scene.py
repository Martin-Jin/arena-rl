"""Draw the arena and its fighters. Used by the match, menus and training view."""

from collections.abc import Mapping, Sequence

import pygame

from arena_ai.config import Config
from arena_ai.sim.state import FighterState, is_opponent
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
    strength: float = 1.0,
) -> None:
    """Draw a soft shadow that shrinks and fades as the fighter jumps.

    ``strength`` (0 to 1) fades it further, with a dying fighter.
    """
    pixels_per_meter = config.viewer.rendering.pixels_per_meter
    lift = min(max(fighter.y, 0.0) / 1.2, 1.0)
    width = max(8, round(config.game.fighter.width_m * pixels_per_meter * 1.8 * (1 - 0.45 * lift)))
    height = max(4, round(width * 0.22))
    alpha = round(70 * (1 - 0.6 * lift) * strength)
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
    hurt: Mapping[str, float] | None = None,
    ghosts: Sequence[tuple[str, FighterState, float]] = (),
    death_fade: Mapping[str, float] | None = None,
) -> None:
    """Draw the arena and every fighter. Only the agent can show a turn frame.

    ``hurt`` gives each fighter's hurt flash strength (0 to 1), as ``Hud.hurt_strength``.
    ``ghosts`` are the dodge afterimages (see ``DodgeTrail.ghosts``), drawn under the fighters.
    ``death_fade`` gives each dead fighter's opacity, as ``DeathFade.strengths``. A dead fighter
    missing from it is not drawn, so a view without the fade drops the dead at once.
    """
    colors = config.viewer.ui.colors
    draw_arena(screen, config, to_screen)
    # A dead fighter fades away and is then gone, so the fight is not crowded by bodies.
    fade = death_fade or {}
    shown: dict[str, tuple[FighterState, float]] = {}
    for fighter_id, fighter in fighters.items():
        strength = fade.get(fighter_id, 0.0) if fighter.health <= 0 else 1.0
        if strength > 0:
            shown[fighter_id] = (fighter, strength)
    for fighter, strength in shown.values():
        _draw_shadow(screen, config, fighter, to_screen, strength)
    pixels_per_meter = config.viewer.rendering.pixels_per_meter
    for fighter_id, ghost, strength in ghosts:
        draw_fighter(
            screen,
            ghost,
            tick,
            to_screen,
            shade=colors["opponent_tint"] if is_opponent(fighter_id) else None,
            alpha=round(config.viewer.ui.dodge_trail_alpha * strength),
        )
    for fighter_id, (fighter, strength) in shown.items():
        opponent = is_opponent(fighter_id)
        drawn = draw_fighter(
            screen,
            fighter,
            tick,
            to_screen,
            turning=turning and not opponent,
            shade=colors["opponent_tint"] if opponent else None,
            hurt=(hurt or {}).get(fighter_id, 0.0),
            effect_color=colors["opponent_health"] if opponent else None,
            alpha=round(255 * strength),
        )
        if not drawn:  # No sprite frames available: draw a plain box instead.
            height = config.game.fighter.height_m * pixels_per_meter
            width = config.game.fighter.width_m * pixels_per_meter
            x, y = to_screen(fighter.x, fighter.y)
            color = colors["fallback_opponent" if opponent else "fallback_agent"]
            box = pygame.Surface((round(width), round(height)))
            box.fill(color)
            box.set_alpha(round(255 * strength))
            screen.blit(box, (x - width / 2, y - height))
