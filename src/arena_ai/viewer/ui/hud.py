"""The in-match heads-up display: health bars, swing meters and damage numbers."""

from collections.abc import Mapping
from dataclasses import dataclass

import pygame

from arena_ai.config import Config
from arena_ai.sim.state import FighterState
from arena_ai.viewer.render.layout import ToScreen
from arena_ai.viewer.ui.theme import Color, Theme

BAR_RADIUS = 6
SWING_HEIGHT = 4


@dataclass
class _Floater:
    """A "-10" that rises above a fighter that was just hit."""

    text: str
    x: float
    y: float
    color: Color
    age: float = 0.0


class Hud:
    """Tracks health changes between frames and draws the match overlay.

    The HUD is kept deliberately light: one health bar per fighter with a small
    name, a thin swing meter that only appears during an attack, the match
    timer, and short-lived damage numbers.
    """

    def __init__(self, config: Config, theme: Theme, to_screen: ToScreen) -> None:
        self.config = config
        self.theme = theme
        self.to_screen = to_screen
        self._health: dict[str, int] = {}
        self._trail: dict[str, float] = {}
        self._hold: dict[str, float] = {}
        self._floaters: list[_Floater] = []

    def reset(self, fighters: Mapping[str, FighterState]) -> None:
        """Start tracking a fresh round."""
        self._health = {fid: f.health for fid, f in fighters.items()}
        self._trail = {fid: float(f.health) for fid, f in fighters.items()}
        self._hold = {fid: 0.0 for fid in fighters}
        self._floaters.clear()

    def update(self, dt: float, fighters: Mapping[str, FighterState]) -> None:
        """Advance the damage trail and floating numbers by ``dt`` seconds."""
        ui = self.config.viewer.ui
        max_health = self.config.game.fighter.max_health
        height = self.config.game.fighter.height_m
        for fighter_id, fighter in fighters.items():
            previous = self._health.get(fighter_id, fighter.health)
            if fighter.health < previous:
                self._hold[fighter_id] = ui.health_trail_delay_seconds
                color = "damage_taken" if fighter_id == "agent" else "damage_dealt"
                self._floaters.append(
                    _Floater(
                        f"-{previous - fighter.health}",
                        fighter.x,
                        fighter.y + height,
                        self.theme.color(color),
                    )
                )
            self._health[fighter_id] = fighter.health
            trail = max(self._trail.get(fighter_id, fighter.health), fighter.health)
            self._hold[fighter_id] = self._hold.get(fighter_id, 0.0) - dt
            if self._hold[fighter_id] <= 0 and trail > fighter.health:
                trail = max(fighter.health, trail - ui.health_trail_speed * max_health * dt)
            self._trail[fighter_id] = trail
        for floater in self._floaters:
            floater.age += dt
        self._floaters = [
            f for f in self._floaters if f.age < ui.damage_text_seconds
        ]

    def _bar_rect(self, mirrored: bool) -> pygame.Rect:
        ui = self.config.viewer.ui
        width = self.config.viewer.rendering.screen_width
        x = width - ui.health_bar_margin - ui.health_bar_width if mirrored else ui.health_bar_margin
        return pygame.Rect(x, ui.health_bar_y + 24, ui.health_bar_width, ui.health_bar_height)

    def _draw_bar(
        self,
        screen: pygame.Surface,
        fighter_id: str,
        fighter: FighterState,
        label: str,
        mirrored: bool,
    ) -> None:
        theme = self.theme
        max_health = self.config.game.fighter.max_health
        rect = self._bar_rect(mirrored)
        pygame.draw.rect(
            screen, theme.color("health_background"), rect, border_radius=BAR_RADIUS
        )

        def span(health: float) -> pygame.Rect:
            width = round(rect.width * max(0.0, min(1.0, health / max_health)))
            x = rect.right - width if mirrored else rect.x
            return pygame.Rect(x, rect.y, width, rect.height)

        trail = span(self._trail.get(fighter_id, fighter.health))
        if trail.width > 0:
            pygame.draw.rect(screen, theme.color("health_trail"), trail, border_radius=BAR_RADIUS)
        fill = span(fighter.health)
        if fill.width > 0:
            pygame.draw.rect(
                screen,
                theme.color("agent_health" if fighter_id == "agent" else "opponent_health"),
                fill,
                border_radius=BAR_RADIUS,
            )

        # Name on the outer edge side of the bar, health value on the inner side.
        top = self.config.viewer.ui.health_bar_y
        name_pos, name_anchor = (rect.right, "topright") if mirrored else (rect.x, "topleft")
        value_pos, value_anchor = (rect.x, "topleft") if mirrored else (rect.right, "topright")
        theme.text(screen, label, "small", theme.color("text"), (name_pos, top), name_anchor)
        theme.text(
            screen,
            str(fighter.health),
            "small",
            theme.color("text_muted"),
            (value_pos, top),
            value_anchor,
        )

        self._draw_swing(screen, fighter, rect, mirrored)

    def _draw_swing(
        self,
        screen: pygame.Surface,
        fighter: FighterState,
        bar: pygame.Rect,
        mirrored: bool,
    ) -> None:
        """Draw a thin meter under the bar while the fighter is mid-attack."""
        if not fighter.attack_frame:
            return
        combat = self.config.game.combat
        if fighter.attack_frame <= combat.attack_windup_frames:
            color = "swing_windup"
        elif fighter.attack_frame <= combat.attack_windup_frames + combat.attack_active_frames:
            color = "swing_active"
        else:
            color = "swing_recovery"
        progress = fighter.attack_frame / combat.attack_total_frames
        width = round(bar.width * min(1.0, progress))
        x = bar.right - width if mirrored else bar.x
        meter = pygame.Rect(x, bar.bottom + 5, width, SWING_HEIGHT)
        pygame.draw.rect(screen, self.theme.color(color), meter, border_radius=2)

    def draw(
        self,
        screen: pygame.Surface,
        fighters: Mapping[str, FighterState],
        labels: Mapping[str, str],
        timer_seconds: int,
        urgent: bool = False,
    ) -> None:
        """Draw bars, the timer and damage numbers for the current frame.

        ``urgent`` colors the timer, for the last seconds of a timed match.
        """
        agent = fighters.get("agent")
        opponent = fighters.get("opponent")
        if agent is not None:
            self._draw_bar(screen, "agent", agent, labels["agent"], mirrored=False)
        if opponent is not None:
            self._draw_bar(screen, "opponent", opponent, labels["opponent"], mirrored=True)
        if opponent is not None:
            minutes, seconds = divmod(timer_seconds, 60)
            bar = self._bar_rect(False)
            self.theme.text(
                screen,
                f"{minutes}:{seconds:02d}",
                "body",
                self.theme.color("damage_taken" if urgent else "text_muted"),
                (screen.get_width() / 2, bar.centery),
                "center",
            )
        self._draw_floaters(screen)

    def _draw_floaters(self, screen: pygame.Surface) -> None:
        duration = self.config.viewer.ui.damage_text_seconds
        for floater in self._floaters:
            progress = floater.age / duration
            x, y = self.to_screen(floater.x, floater.y)
            self.theme.text(
                screen,
                floater.text,
                "body",
                floater.color,
                (x, y - 12 - 38 * progress),
                "midbottom",
                alpha=round(255 * (1 - progress**2)),
            )

    def draw_banner(self, screen: pygame.Surface, text: str, alpha: int) -> None:
        """Draw a large centered message such as "Ready" or "Fight!"."""
        self.theme.text(
            screen,
            text,
            "title",
            self.theme.color("text"),
            (screen.get_width() / 2, screen.get_height() * 0.36),
            "center",
            alpha=alpha,
        )

    def draw_hint(self, screen: pygame.Surface, text: str, alpha: int) -> None:
        """Draw a faint one-line hint on the floor strip."""
        rendering = self.config.viewer.rendering
        y = (rendering.floor_y_px + rendering.screen_height) / 2
        self.theme.text(
            screen,
            text,
            "small",
            self.theme.color("text_muted"),
            (screen.get_width() / 2, y),
            "center",
            alpha=alpha,
        )
