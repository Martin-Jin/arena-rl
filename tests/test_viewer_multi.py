import dataclasses
import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from arena_ai.config import load_config
from arena_ai.sim.state import AGENT_ID, FighterState, opponent_ids
from arena_ai.viewer.app import App
from arena_ai.viewer.audio import mixer_channels
from arena_ai.viewer.controllers import ScriptedController, make_controller
from arena_ai.viewer.matchup import fighter_labels, matchup_text, opponent_labels
from arena_ai.viewer.scenes.match import MatchScene
from arena_ai.viewer.scenes.settings_menu import SettingsScene

CONFIG = load_config()
SLOTS = CONFIG.game.observation.max_opponents


def make_app(config=CONFIG) -> App:
    app = App(config)
    app.settings.player = "human"
    app.settings.random_spawns = False
    return app


class LabelTests(unittest.TestCase):
    def test_duplicate_opponents_are_numbered(self) -> None:
        self.assertEqual(
            opponent_labels(["approach", "idle", "approach"]),
            ["Scripted: approach #1", "Idle dummy", "Scripted: approach #2"],
        )
        self.assertEqual(opponent_labels(["idle"]), ["Idle dummy"])

    def test_fighter_labels_and_matchup_text(self) -> None:
        labels = fighter_labels("human", ["idle", "idle"])
        self.assertEqual(labels[AGENT_ID], "You")
        self.assertEqual(labels["opponent_1"], "Idle dummy #2")
        self.assertEqual(matchup_text("human", ["idle", "random"]), "You  vs  Idle dummy, Random")
        self.assertIn("Practice", matchup_text("human", []))


class ControllerTests(unittest.TestCase):
    def test_a_scripted_controller_plays_its_own_fighter(self) -> None:
        fighters = {
            AGENT_ID: FighterState(x=3.0),
            "opponent": FighterState(x=9.0, facing_dir=-1),
            "opponent_1": FighterState(x=6.0, facing_dir=-1),
        }
        for fighter_id in ("opponent", "opponent_1"):
            controller = make_controller("approach", CONFIG, True, fighter_id)
            self.assertIsInstance(controller, ScriptedController)
            self.assertIsInstance(controller.action(fighters), int)
        with self.assertRaises(ValueError):
            make_controller("approach", CONFIG, True, AGENT_ID)

    def test_a_scripted_opponent_walks_toward_the_agent_not_its_ally(self) -> None:
        fighters = {
            AGENT_ID: FighterState(x=1.0),
            "opponent": FighterState(x=9.0, facing_dir=-1),
            "opponent_1": FighterState(x=8.0, facing_dir=-1),
        }
        action = make_controller("approach", CONFIG, True, "opponent").action(fighters)
        from arena_ai.agents.action import to_fighter_action

        self.assertEqual(to_fighter_action(action).move_dir, -1)


class SettingsMenuTests(unittest.TestCase):
    def tearDown(self) -> None:
        pygame.quit()

    def test_the_panel_fits_the_screen_at_the_configured_opponent_count(self) -> None:
        app = make_app()
        scene = SettingsScene(app)
        self.assertLessEqual(scene.panel.bottom, app.screen.get_height())
        self.assertGreaterEqual(scene.panel.left, 0)
        self.assertLessEqual(scene.panel.right, app.screen.get_width())

    def test_the_panel_still_fits_with_more_opponent_slots(self) -> None:
        game = dataclasses.replace(
            CONFIG.game, observation=dataclasses.replace(CONFIG.game.observation, max_opponents=5)
        )
        app = make_app(dataclasses.replace(CONFIG, game=game))
        scene = SettingsScene(app)
        self.assertEqual(len(app.settings.opponents), 5)
        self.assertLessEqual(scene.panel.bottom, app.screen.get_height())

    def test_every_row_sits_inside_the_panel(self) -> None:
        scene = SettingsScene(make_app())
        for widget in scene.menu.widgets:
            self.assertTrue(scene.panel.contains(widget.rect), widget.label)

    def test_slots_at_or_above_the_count_are_disabled(self) -> None:
        app = make_app()
        app.settings.opponent_count = 1
        scene = SettingsScene(app)
        slots = [w for w in scene.menu.widgets if getattr(w, "label", "").startswith("Opponent ")]
        self.assertEqual(len(slots), SLOTS)
        self.assertEqual([w.enabled for w in slots], [True] + [False] * (SLOTS - 1))

    def test_the_menu_draws(self) -> None:
        app = make_app()
        SettingsScene(app).draw(app.screen)


class MatchTests(unittest.TestCase):
    def tearDown(self) -> None:
        pygame.quit()

    def _scene(self, kinds: list[str]) -> MatchScene:
        app = make_app()
        app.settings.opponent_count = len(kinds)
        app.settings.opponents = kinds + ["idle"] * (SLOTS - len(kinds))
        return MatchScene(app)

    def test_a_match_has_one_fighter_and_controller_per_opponent(self) -> None:
        scene = self._scene(["idle", "approach", "idle"])
        self.assertEqual(list(scene.fighters), [AGENT_ID, *opponent_ids(3)])
        self.assertEqual(sorted(scene.controllers), opponent_ids(3))
        self.assertEqual(scene.labels["opponent_2"], "Idle dummy #2")

    def test_it_ends_only_when_every_opponent_is_dead(self) -> None:
        scene = self._scene(["idle", "idle"])
        scene.phase = "playing"
        scene.fighters["opponent"].health = 0
        scene.update(1 / 60)
        self.assertEqual(scene.phase, "playing")
        scene.fighters["opponent_1"].health = 0
        scene.update(1 / 60)
        self.assertEqual(scene.phase, "over")
        self.assertEqual(scene.outcome, "agent")

    def test_the_agent_dying_ends_it_with_the_opponents_winning(self) -> None:
        scene = self._scene(["idle", "idle"])
        scene.phase = "playing"
        scene.fighters[AGENT_ID].health = 0
        scene.update(1 / 60)
        self.assertEqual(scene.outcome, "opponent")

    def test_the_last_opponent_and_the_agent_falling_together_is_a_draw(self) -> None:
        scene = self._scene(["idle", "idle"])
        scene.phase = "playing"
        scene.fighters["opponent"].health = 0
        scene.fighters[AGENT_ID].health = 0
        scene.fighters["opponent_1"].health = 0
        scene.update(1 / 60)
        self.assertEqual(scene.outcome, "draw")

    def test_the_result_screen_sums_damage_over_every_opponent(self) -> None:
        scene = self._scene(["idle", "idle"])
        scene.phase = "playing"
        scene.fighters["opponent"].health = 0
        scene.fighters["opponent_1"].health = 30
        scene.fighters[AGENT_ID].health = 90
        scene._finish(timed_out=True)
        scene.draw(scene.app.screen)  # Draws the result screen without error.
        self.assertEqual(scene.outcome, "agent")  # 90 beats the mean opponent health of 15.

    def test_a_match_draws_every_phase_with_three_opponents(self) -> None:
        scene = self._scene(["approach", "idle", "random"])
        for _ in range(5):
            scene.update(1 / 60)
            scene.draw(scene.app.screen)
        scene.phase = "playing"
        for _ in range(120):
            scene.update(1 / 60)
        scene.draw(scene.app.screen)

    def test_health_bars_stack_without_overlapping(self) -> None:
        scene = self._scene(["idle", "idle", "idle"])
        hud = scene.hud
        rects = [hud._bar_rect(True, row, 3) for row in range(3)]
        for a, b in zip(rects, rects[1:]):
            self.assertFalse(a.inflate(0, 4 + 4).colliderect(b.move(0, -24)))
        self.assertEqual(hud._bar_rect(True), hud._bar_rect(True, 0, 1))
        self.assertLess(rects[0].height, hud._bar_rect(True).height)

    def test_solo_practice_has_no_opponent_and_never_ends(self) -> None:
        app = make_app()
        app.settings.opponent_count = 0
        scene = MatchScene(app)
        self.assertEqual(list(scene.fighters), [AGENT_ID])
        scene.phase = "playing"
        for _ in range(60):
            scene.update(1 / 60)
        self.assertEqual(scene.phase, "playing")


class AudioChannelTests(unittest.TestCase):
    def test_the_channels_cover_every_fighter(self) -> None:
        self.assertEqual(mixer_channels(2), 16)
        self.assertGreater(mixer_channels(SLOTS + 1), mixer_channels(2) - 1)

    def test_an_impossible_fighter_count_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_opponents"):
            mixer_channels(100)
        with self.assertRaises(ValueError):
            mixer_channels(0)
