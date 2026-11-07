"""The in-match heads-up display: health bars, swing meters, damage and parry text."""

from collections.abc import Mapping
from dataclasses import dataclass
import math

import pygame

from arena_ai.config import Config
from arena_ai.sim.state import FighterState
from arena_ai.viewer.render.layout import ToScreen
from arena_ai.viewer.ui.theme import Color, Theme

BAR_RADIUS = 6
SWING_HEIGHT = 4
SPARK_COUNT = 8


@dataclass
class _Floater:
    """A "-10" (or "Parry") that rises above a fighter that was just hit or parried."""

    text: str
    x: float
    y: float
    color: Color
    age: float = 0.0


@dataclass
class _Flash:
    """A ring and sparks at the point where a parry stopped an attack."""

    x: float
    y: float
    color: Color
    age: float = 0.0


@dataclass
class _Slash:
    """A straight line cut through the arena by a fighter's counter attack."""

    fighter_id: str
    direction: int  # Which way the fighter lunges.
    start_x: float
    y: float
    end_x: float
    color: Color
    ended_for: float | None = None  # Seconds since the line stopped growing, once it has.


class Hud:
    """Tracks health changes between frames and draws the match overlay.

    The HUD is kept deliberately light: one health bar per fighter with a small
    name, a thin swing meter that only appears during an attack, the match
    timer, and short-lived damage numbers and parry text.
    """

    def __init__(self, config: Config, theme: Theme, to_screen: ToScreen) -> None:
        self.config = config
        self.theme = theme
        self.to_screen = to_screen
        self._health: dict[str, int] = {}
        self._trail: dict[str, float] = {}
        self._hold: dict[str, float] = {}
        self._hurt: dict[str, float] = {}  # Seconds since each fighter was last hurt.
        self._floaters: list[_Floater] = []
        self._flashes: list[_Flash] = []
        self._slashes: list[_Slash] = []

    def reset(self, fighters: Mapping[str, FighterState]) -> None:
        """Start tracking a fresh round."""
        self._health = {fid: f.health for fid, f in fighters.items()}
        self._trail = {fid: float(f.health) for fid, f in fighters.items()}
        self._hold = {fid: 0.0 for fid in fighters}
        self._hurt.clear()
        self._floaters.clear()
        self._flashes.clear()
        self._slashes.clear()

    def update(self, dt: float, fighters: Mapping[str, FighterState]) -> None:
        """Advance the damage trail and floating numbers by ``dt`` seconds."""
        ui = self.config.viewer.ui
        max_health = self.config.game.fighter.max_health
        height = self.config.game.fighter.height_m
        damaged: set[str] = set()
        for fighter_id, fighter in fighters.items():
            previous = self._health.get(fighter_id, fighter.health)
            if fighter.health < previous:
                damaged.add(fighter_id)
                self._hurt[fighter_id] = 0.0
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
        for flash in self._flashes:
            flash.age += dt
        self._flashes = [f for f in self._flashes if f.age < ui.parry_flash_seconds]
        self._update_slashes(dt, fighters, damaged)
        for fighter_id in list(self._hurt):
            self._hurt[fighter_id] += dt
            if self._hurt[fighter_id] >= ui.hurt_flash_seconds:
                del self._hurt[fighter_id]

    def hurt_strength(self, fighter_id: str) -> float:
        """Return how strongly a fighter flashes from a recent hit: 0, up to 1, then 0.

        It rises and falls smoothly over ``hurt_flash_seconds``, so the fighter fades
        down and back up.
        """
        age = self._hurt.get(fighter_id)
        if age is None:
            return 0.0
        return math.sin(math.pi * age / self.config.viewer.ui.hurt_flash_seconds)

    def _side_color(self, fighter_id: str) -> Color:
        """Return the color of a fighter's side: blue for the agent, red for the opponent."""
        return self.theme.color("opponent_health" if fighter_id == "opponent" else "accent")

    def parried(self, fighter_id: str, fighter: FighterState) -> None:
        """Show "Parry" above a fighter that just parried an attack.

        Called once per simulation step, because the parry event lasts one step and
        a frame can span several.
        """
        fighter_config = self.config.game.fighter
        self._flashes.append(
            _Flash(
                fighter.x + fighter.facing_dir * (fighter_config.width_m / 2 + 0.2),
                fighter.y + fighter_config.height_m / 2,
                self._side_color(fighter_id),
            )
        )
        self._floaters.append(
            _Floater(
                "Parry",
                fighter.x,
                fighter.y + self.config.game.fighter.height_m,
                self._side_color(fighter_id),
            )
        )

    def countered(self, fighter_id: str, fighter: FighterState) -> None:
        """Show "Counter" and start the cut line for a fighter that began a counter attack.

        Called once per simulation step, like ``parried``. The text starts above the
        "Parry" text that usually appears just before it, so the two do not overlap.
        """
        height = self.config.game.fighter.height_m
        self._slashes.append(
            _Slash(
                fighter_id,
                fighter.facing_dir,
                fighter.x,
                fighter.y + 0.55 * height,
                fighter.x,
                self._side_color(fighter_id),
            )
        )
        self._floaters.append(
            _Floater("Counter", fighter.x, fighter.y + 1.3 * height, self._side_color(fighter_id))
        )

    def _update_slashes(
        self, dt: float, fighters: Mapping[str, FighterState], damaged: set[str]
    ) -> None:
        """Follow each counter's lunge, then end its line where the counter ends.

        The line follows the fighter. If the counter hits, it carries on through the
        opponent. If the counter is parried (the attacker is staggered) it stops at the
        parrying fighter. If the counter misses it stops where the swing ended.
        """
        ui = self.config.viewer.ui
        fighter_config = self.config.game.fighter
        for slash in self._slashes:
            if slash.ended_for is not None:
                slash.ended_for += dt
                continue
            attacker = fighters.get(slash.fighter_id)
            if attacker is None:
                slash.ended_for = 0.0
                continue
            others = [f for fid, f in fighters.items() if fid != slash.fighter_id]
            target = min(others, key=lambda f: abs(f.x - attacker.x), default=None)
            hit = any(fid in damaged for fid in fighters if fid != slash.fighter_id)
            if target is not None and hit:
                slash.end_x = target.x + slash.direction * ui.counter_line_carry_m
                slash.ended_for = 0.0
            elif attacker.stagger_frames > 0:
                if target is not None:
                    slash.end_x = target.x - slash.direction * fighter_config.width_m / 2
                slash.ended_for = 0.0
            elif not attacker.attack_counter:
                slash.ended_for = 0.0
            else:
                slash.end_x = attacker.x
        self._slashes = [
            s for s in self._slashes
            if s.ended_for is None or s.ended_for < ui.counter_effect_seconds
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
        value_pos, value_anchor = (rect.x, "topleft") if mirrored else (rect.right, "topright")
        # The name gets the bar's width minus room for the health value opposite it.
        name_width = rect.width - 70
        name_area = pygame.Rect(
            rect.right - name_width if mirrored else rect.x, top, name_width, 24
        )
        theme.fit_text(
            screen, label, "small", theme.color("text"), name_area, "right" if mirrored else "left"
        )
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
        self._draw_flashes(screen)
        self._draw_slashes(screen)
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

    def _draw_flashes(self, screen: pygame.Surface) -> None:
        """Draw the ring and sparks of each recent parry, growing and fading out."""
        ui = self.config.viewer.ui
        for flash in self._flashes:
            color = flash.color
            progress = flash.age / ui.parry_flash_seconds
            radius = ui.parry_flash_radius * (0.25 + 0.75 * progress)
            alpha = round(255 * (1 - progress))
            x, y = (round(value) for value in self.to_screen(flash.x, flash.y))
            size = round(2 * radius) + 8
            layer = pygame.Surface((size, size), pygame.SRCALPHA)
            middle = size // 2
            thickness = max(1, round(5 * (1 - progress)))
            pygame.draw.circle(layer, (*color, alpha), (middle, middle), round(radius), thickness)
            for index in range(SPARK_COUNT):
                angle = 2 * math.pi * index / SPARK_COUNT
                inner, outer = radius * 0.55, radius * (0.8 + 0.2 * progress)
                pygame.draw.line(
                    layer,
                    (*color, alpha),
                    (middle + inner * math.cos(angle), middle + inner * math.sin(angle)),
                    (middle + outer * math.cos(angle), middle + outer * math.sin(angle)),
                    3,
                )
            screen.blit(layer, layer.get_rect(center=(x, y)))

    def _draw_slashes(self, screen: pygame.Surface) -> None:
        """Draw each counter's straight cut line, fading out once it has ended."""
        ui = self.config.viewer.ui
        for slash in self._slashes:
            color = slash.color
            fade = 0.0 if slash.ended_for is None else slash.ended_for / ui.counter_effect_seconds
            x_start, y = self.to_screen(slash.start_x, slash.y)
            x_end, _ = self.to_screen(slash.end_x, slash.y)
            left, right = round(min(x_start, x_end)), round(max(x_start, x_end))
            size = (max(1, right - left), ui.counter_line_width_px)
            layer = pygame.Surface(size, pygame.SRCALPHA)
            layer.fill((*color, round(255 * (1 - fade))))
            screen.blit(layer, layer.get_rect(midleft=(left, round(y))))

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
        self.theme.fit_text(
            screen,
            text,
            "small",
            self.theme.color("text_muted"),
            pygame.Rect(40, round(y) - 14, screen.get_width() - 80, 28),
            "center",
            alpha=alpha,
        )
