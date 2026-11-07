"""The match: a short intro, the fight, a pause menu and the result screen."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import pygame

from arena_ai.agents.action import from_parts, to_fighter_action
from arena_ai.sim.simulate import FighterAction, step_game
from arena_ai.sim.state import FighterState, create_initial_fighters
from arena_ai.viewer.audio import FighterSnapshot
from arena_ai.viewer.controllers import Controller, make_controller
from arena_ai.viewer.render.scene import draw_scene
from arena_ai.viewer.scenes.settings_menu import SettingsScene
from arena_ai.viewer.settings import OPPONENT_OPTIONS, PLAYER_OPTIONS, label_for
from arena_ai.viewer.ui.hud import Hud
from arena_ai.viewer.ui.widgets import Button, Menu, stack

if TYPE_CHECKING:
    from arena_ai.viewer.app import App

INTRO, PLAYING, PAUSED, OVER = "intro", "playing", "paused", "over"
PANEL_WIDTH = 520
BUTTON_WIDTH = 300
BUTTON_HEIGHT = 56
TURN_FRAMES = 6  # Render frames the turn animation is shown after a direction change.
NOTICE_SECONDS = 5.0
URGENT_SECONDS = 10  # The timer turns red for the last seconds of a timed match.
HUMAN_HINT = "A / D  move     W / Space  jump     Click  attack     Q  parry     Esc  pause"
WATCH_HINT = "Esc  pause"


class MatchScene:
    """Runs one fighter against another until someone is defeated.

    Each fighter is either the human (keyboard) or an AI controller. By default
    the human plays the agent and an AI plays the opponent, but the settings can
    hand the agent to the trained network to watch it fight.
    """

    def __init__(self, app: App) -> None:
        self.app = app
        self.config = app.config
        self.dt = self.config.game.world.dt_seconds
        # None when the limit is 0, which lets a match run until someone is defeated.
        self.time_limit = self.config.viewer.match.time_limit_seconds or None
        self.rng = random.Random()
        self.hud = Hud(self.config, app.theme, app.to_screen)
        self.notice = ""
        self.age = 0.0  # Real seconds since the first round, for the controls hint.
        self.controllers: dict[str, Controller] = {}
        self.player_kind = "human"
        self.opponent_kind = "none"
        self._build_controllers()
        self.restart()

    # -- setup -------------------------------------------------------------

    def _build_controllers(self) -> None:
        """Create the AI controllers, falling back to scripted play on failure."""
        settings = self.app.settings
        greedy = settings.trained_greedy
        self.player_kind = settings.player
        self.opponent_kind = settings.opponent
        if self.player_kind != "human":
            try:
                self.controllers["agent"] = make_controller(
                    self.player_kind, self.config, greedy, "agent", settings.checkpoint
                )
            except Exception as error:  # Missing checkpoint, torch not installed...
                self.notice = f"Could not load the trained agent ({error}). You play instead."
                self.player_kind = "human"
        if self.opponent_kind != "none":
            try:
                self.controllers["opponent"] = make_controller(
                    self.opponent_kind, self.config, greedy, "opponent", settings.checkpoint
                )
            except Exception as error:
                self.notice = f"Could not load the trained agent ({error}). Using scripted."
                self.opponent_kind = "approach"
                self.controllers["opponent"] = make_controller(
                    "approach", self.config, greedy, "opponent"
                )

    def _new_fighters(self) -> dict[str, FighterState]:
        if self.opponent_kind == "none":
            return {"agent": FighterState()}  # Practice: a single fighter.
        rng = self.rng if self.app.settings.random_spawns else None
        return create_initial_fighters(self.config.game, rng)

    def restart(self) -> None:
        """Start a fresh round with the same opponents."""
        self.fighters = self._new_fighters()
        self.hud.reset(self.fighters)
        self.phase = INTRO
        self.menu: Menu | None = None
        self.intro_elapsed = 0.0
        self.fight_elapsed = 0.0
        self.steps = 0
        self.accumulator = 0.0
        self.render_tick = 0
        self.turn_timer = 0
        self.last_move_dir = 0
        self.winner = ""  # Title of the result screen.
        self.subtitle = ""
        self.outcome = ""  # "agent", "opponent" or "draw" once the match is over.
        self.frozen = False  # The scene stops animating once the result screen is up.
        self.app.audio.cut_effects()
        self.app.audio.ensure_music()

    @property
    def labels(self) -> dict[str, str]:
        return {
            "agent": label_for(PLAYER_OPTIONS, self.player_kind),
            "opponent": label_for(OPPONENT_OPTIONS, self.opponent_kind),
        }

    @property
    def watching(self) -> bool:
        """True when the agent is AI-controlled and the human only watches."""
        return "agent" in self.controllers

    # -- state changes -----------------------------------------------------

    def pause(self) -> None:
        self._resume_phase = self.phase
        self.phase = PAUSED
        self.app.audio.cut_footsteps()
        self.menu = Menu(
            [
                Button("Resume", self.resume, primary=True),
                Button("Settings", lambda: self.app.push(SettingsScene(self.app))),
                Button("Main menu", self.app.to_main_menu),
            ]
        )
        stack(
            self.menu.widgets,
            self.app.screen.get_width() // 2,
            262,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            14,
        )

    def resume(self) -> None:
        self.phase = self._resume_phase
        self.menu = None

    def _decide(self, timed_out: bool) -> str:
        """Return "agent", "opponent" or "draw" for a finished match."""
        agent, opponent = self.fighters["agent"], self.fighters["opponent"]
        if timed_out:  # Nobody was defeated, so the healthier fighter wins.
            if agent.health == opponent.health:
                return "draw"
            return "agent" if agent.health > opponent.health else "opponent"
        if agent.health <= 0 and opponent.health <= 0:
            return "draw"  # Both fell in the same step.
        return "agent" if agent.health > 0 else "opponent"

    def _finish(self, timed_out: bool = False) -> None:
        self.phase = OVER
        self.app.audio.silence()
        self.outcome = self._decide(timed_out)
        # Whoever wins, the music stops at once and the effects still playing fade out.
        self.app.audio.stop_music()
        self.app.audio.fade_out_effects()
        if self.outcome == "draw":
            self.winner = "Draw"
        elif self.watching:
            self.winner = f"{self.labels[self.outcome]} wins"
        else:
            self.winner = "Victory" if self.outcome == "agent" else "Defeat"
        self.subtitle = ""
        if timed_out:
            reason = "equal health" if self.outcome == "draw" else "decided on health"
            self.subtitle = f"Time's up  ·  {reason}"
        self.menu = Menu(
            [
                Button("Play again", self.restart, primary=True),
                Button("Main menu", self.app.to_main_menu),
            ]
        )
        stack(
            self.menu.widgets,
            self.app.screen.get_width() // 2,
            372,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            14,
        )

    # -- events ------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        if self.menu is not None:
            self.menu.handle_event(event)
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_p):
            if self.phase == PAUSED:
                self.resume()
            elif self.phase in (INTRO, PLAYING):
                self.pause()
            elif self.phase == OVER:
                self.app.to_main_menu()
        elif event.type == pygame.WINDOWFOCUSLOST and self.phase in (INTRO, PLAYING):
            self.pause()

    # -- simulation --------------------------------------------------------

    def _human_action(self) -> FighterAction:
        keys = pygame.key.get_pressed()
        move_dir = (
            int(keys[pygame.K_d] or keys[pygame.K_RIGHT])
            - int(keys[pygame.K_a] or keys[pygame.K_LEFT])
        )
        jump = bool(keys[pygame.K_w] or keys[pygame.K_UP] or keys[pygame.K_SPACE])
        attack = bool(pygame.mouse.get_pressed()[0])
        parry = bool(keys[pygame.K_q])
        return to_fighter_action(from_parts(move_dir, jump, attack, parry))

    def _simulate(self, dt: float) -> None:
        """Step the simulation in fixed timesteps, however long the frame took."""
        # Cap the frame time so a lag spike cannot trigger a spiral of catch-up steps.
        self.accumulator += min(dt, 0.25)
        human_action = None if self.watching else self._human_action()
        while self.accumulator >= self.dt and self.phase == PLAYING:
            actions = {
                fighter_id: to_fighter_action(self.controllers[fighter_id].action(self.fighters))
                if fighter_id in self.controllers
                else human_action
                for fighter_id in self.fighters
            }
            before = {
                fighter_id: FighterSnapshot.of(fighter)
                for fighter_id, fighter in self.fighters.items()
            }
            step_game(self.fighters, actions, self.config)
            self.steps += 1
            move_dir = actions["agent"].move_dir
            if move_dir and move_dir != self.last_move_dir:
                self.turn_timer = TURN_FRAMES
            if move_dir:
                self.last_move_dir = move_dir
            for fighter_id, fighter in self.fighters.items():
                if fighter.parried:
                    self.hud.parried(fighter_id, fighter)
                if fighter.counter_started:
                    self.hud.countered(fighter_id, fighter)
                self.app.audio.update_fighter(fighter_id, fighter, before[fighter_id])
            self.accumulator -= self.dt
            if self.opponent_kind == "none":
                continue  # Practice has no opponent to defeat and no time limit.
            if any(f.health <= 0 for f in self.fighters.values()):
                self._finish()
            elif self.time_limit and self.steps * self.dt >= self.time_limit:
                self._finish(timed_out=True)

    def update(self, dt: float) -> None:
        ui = self.config.viewer.ui
        self.age += dt
        if self.phase == INTRO:
            self.intro_elapsed += dt
            if self.intro_elapsed >= ui.ready_seconds:
                self.phase = PLAYING
        elif self.phase == PLAYING:
            self.fight_elapsed += dt
            self._simulate(dt)
        if self.phase == OVER:
            # Freeze every animation behind the result screen. One zero-length update
            # first shows the final hit (its number and the empty health bar).
            if not self.frozen:
                self.hud.update(0.0, self.fighters)
                self.frozen = True
            return
        if self.phase != PAUSED:
            self.render_tick += 1
            self.turn_timer = max(0, self.turn_timer - 1)
        self.hud.update(dt, self.fighters)

    # -- drawing -----------------------------------------------------------

    def _timer(self) -> tuple[int, bool]:
        """Return the seconds to show, and whether the match is nearly out of time."""
        elapsed = self.steps * self.dt
        if not self.time_limit:
            return int(elapsed), False
        remaining = max(0, math.ceil(self.time_limit - elapsed))
        return remaining, remaining <= URGENT_SECONDS

    def draw(self, screen: pygame.Surface) -> None:
        theme = self.app.theme
        ui = self.config.viewer.ui
        draw_scene(
            screen,
            self.config,
            self.app.to_screen,
            self.fighters,
            self.render_tick,
            turning=self.turn_timer > 0,
            hurt={fid: self.hud.hurt_strength(fid) for fid in self.fighters},
        )
        timer_seconds, urgent = self._timer()
        self.hud.draw(screen, self.fighters, self.labels, timer_seconds, urgent)
        if self.phase == INTRO:
            self.hud.draw_banner(screen, "Ready", 255)
        elif self.phase == PLAYING and self.fight_elapsed < ui.fight_banner_seconds:
            fade = 1 - self.fight_elapsed / ui.fight_banner_seconds
            self.hud.draw_banner(screen, "Fight!", round(255 * fade))
        self._draw_hint_and_notice(screen)
        if self.phase == PAUSED:
            self._draw_pause(screen)
        elif self.phase == OVER:
            self._draw_result(screen)

    def _draw_hint_and_notice(self, screen: pygame.Surface) -> None:
        ui = self.config.viewer.ui
        settings = self.app.settings
        if settings.show_controls_hint and self.age < ui.controls_hint_seconds:
            fade = min(1.0, (ui.controls_hint_seconds - self.age) / 1.0)
            text = WATCH_HINT if self.watching else HUMAN_HINT
            self.hud.draw_hint(screen, text, round(255 * fade))
        if self.notice and self.age < NOTICE_SECONDS:
            self.app.theme.fit_text(
                screen,
                self.notice,
                "small",
                self.app.theme.color("damage_taken"),
                pygame.Rect(80, 84, screen.get_width() - 160, 24),
                "center",
            )

    def _draw_pause(self, screen: pygame.Surface) -> None:
        theme = self.app.theme
        theme.draw_overlay(screen)
        center_x = screen.get_width() // 2
        panel = pygame.Rect(0, 0, PANEL_WIDTH, 380)
        panel.midtop = (center_x, 150)
        theme.draw_panel(screen, panel)
        theme.text(screen, "Paused", "heading", theme.color("text"), (center_x, 208), "center")
        if self.menu:
            self.menu.draw(screen, theme)

    def _draw_result(self, screen: pygame.Surface) -> None:
        theme = self.app.theme
        theme.draw_overlay(screen)
        center_x = screen.get_width() // 2
        panel = pygame.Rect(0, 0, PANEL_WIDTH, 390)
        panel.midtop = (center_x, 150)
        theme.draw_panel(screen, panel)
        if self.watching or self.outcome == "draw":
            title_color = theme.color("text")
        else:
            title_color = theme.color("victory" if self.outcome == "agent" else "defeat")
        inner = panel.inflate(-48, 0)  # Text stays inside the panel's padding.
        theme.fit_text(
            screen,
            self.winner,
            "heading",
            title_color,
            pygame.Rect(inner.x, 176, inner.width, 48),
            "center",
        )
        if self.subtitle:
            theme.fit_text(
                screen,
                self.subtitle,
                "small",
                theme.color("text_muted"),
                pygame.Rect(inner.x, 228, inner.width, 24),
                "center",
            )

        max_health = self.config.game.fighter.max_health
        opponent = self.fighters.get("opponent")
        dealt = max_health - opponent.health if opponent else 0
        taken = max_health - self.fighters["agent"].health
        minutes, seconds = divmod(int(self.steps * self.dt), 60)
        stats = (
            ("Time", f"{minutes}:{seconds:02d}"),
            ("Damage dealt", str(dealt)),
            ("Damage taken", str(taken)),
        )
        for index, (label, value) in enumerate(stats):
            x = center_x + (index - 1) * 150
            value_area = pygame.Rect(x - 70, 266, 140, 48)
            theme.fit_text(screen, value, "heading", theme.color("text"), value_area, "center")
            label_area = pygame.Rect(x - 70, 316, 140, 24)
            theme.fit_text(screen, label, "small", theme.color("text_muted"), label_area, "center")
        if self.menu:
            self.menu.draw(screen, theme)
