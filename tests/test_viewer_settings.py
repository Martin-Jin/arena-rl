import json
import os
import random
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from arena_ai.config import load_config
from arena_ai.sim.state import create_initial_fighters
from arena_ai.viewer.settings import load_settings, save_settings
from arena_ai.viewer.ui.widgets import Button, Dropdown, Menu


class ViewerSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "settings.json"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_missing_file_uses_config_defaults(self) -> None:
        settings = load_settings(self.config, self.path)
        audio = self.config.viewer.audio
        self.assertEqual(settings.music_volume, audio.music_volume)
        self.assertEqual(settings.player, "human")

    def test_save_and_load_round_trip(self) -> None:
        settings = load_settings(self.config, self.path)
        settings.music_volume = 0.75
        settings.opponents[0] = "idle"
        settings.opponent_count = 2
        settings.random_spawns = True
        save_settings(settings, self.path)
        loaded = load_settings(self.config, self.path)
        self.assertEqual(loaded, settings)

    def test_invalid_saved_values_are_repaired(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "music_volume": 7.0,
                    "opponents": ["nonsense", "idle"],
                    "opponent_count": 99,
                    "player": 3,
                }
            ),
            encoding="utf-8",
        )
        settings = load_settings(self.config, self.path)
        slots = self.config.game.observation.max_opponents
        self.assertEqual(settings.music_volume, 1.0)
        self.assertEqual(settings.opponents, ["approach", "idle"] + ["approach"] * (slots - 2))
        self.assertEqual(settings.opponent_count, slots)
        self.assertEqual(settings.player, "human")

    def test_a_saved_opponent_count_below_zero_or_an_ai_player_alone_is_repaired(self) -> None:
        self.path.write_text(
            json.dumps({"opponent_count": 0, "player": "trained"}), encoding="utf-8"
        )
        settings = load_settings(self.config, self.path)
        self.assertGreaterEqual(settings.opponent_count, 0)
        if settings.player != "human":
            self.assertGreaterEqual(settings.opponent_count, 1)

    def test_the_legacy_single_opponent_key_moves_to_the_first_slot(self) -> None:
        self.path.write_text(json.dumps({"opponent": "idle"}), encoding="utf-8")
        settings = load_settings(self.config, self.path)
        self.assertEqual(settings.opponents[0], "idle")
        self.assertEqual(settings.opponent_count, 1)
        self.assertEqual(len(settings.opponents), self.config.game.observation.max_opponents)

    def test_the_legacy_practice_choice_becomes_zero_opponents(self) -> None:
        self.path.write_text(json.dumps({"opponent": "none"}), encoding="utf-8")
        self.assertEqual(load_settings(self.config, self.path).opponent_count, 0)

    def test_corrupt_file_falls_back_to_defaults(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(load_settings(self.config, self.path).player, "human")


class SpawnTests(unittest.TestCase):
    def test_seeded_random_spawns_repeat_and_face_each_other(self) -> None:
        game = load_config().game
        first = create_initial_fighters(game, random.Random(5))
        second = create_initial_fighters(game, random.Random(5))
        self.assertEqual(first, second)
        agent, opponent = first["agent"], first["opponent"]
        self.assertEqual(agent.facing_dir, 1 if opponent.x >= agent.x else -1)
        self.assertEqual(opponent.facing_dir, -agent.facing_dir)

    def test_default_spawns_are_fixed(self) -> None:
        game = load_config().game
        self.assertEqual(
            create_initial_fighters(game), create_initial_fighters(game, None)
        )


class MenuTests(unittest.TestCase):
    def test_focus_skips_disabled_widgets_and_enter_clicks(self) -> None:
        clicked = []
        menu = Menu(
            [
                Button("a", lambda: clicked.append("a")),
                Button("b", lambda: clicked.append("b"), enabled_fn=lambda: False),
                Button("c", lambda: clicked.append("c")),
            ]
        )
        down = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN)
        enter = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)
        menu.handle_event(down)
        menu.handle_event(enter)
        self.assertEqual(clicked, ["c"])


    def test_dropdown_opens_skips_disabled_options_and_selects(self) -> None:
        chosen = {"value": "a"}
        dropdown = Dropdown(
            "Pick",
            [("a", "A"), ("b", "B"), ("c", "C")],
            lambda: chosen["value"],
            lambda value: chosen.update(value=value),
            option_enabled=lambda value: value != "b",
        )
        menu = Menu([dropdown])
        press = lambda key: menu.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
        press(pygame.K_RETURN)  # Open the list.
        self.assertTrue(menu.captures_input)
        press(pygame.K_DOWN)  # "b" is disabled, so the highlight lands on "c".
        press(pygame.K_RETURN)
        self.assertEqual(chosen["value"], "c")
        self.assertFalse(menu.captures_input)

    def test_escape_closes_an_open_dropdown_without_choosing(self) -> None:
        chosen = {"value": "a"}
        dropdown = Dropdown(
            "Pick",
            [("a", "A"), ("b", "B")],
            lambda: chosen["value"],
            lambda value: chosen.update(value=value),
        )
        menu = Menu([dropdown])
        press = lambda key: menu.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
        press(pygame.K_RETURN)
        press(pygame.K_DOWN)
        press(pygame.K_ESCAPE)
        self.assertFalse(menu.captures_input)
        self.assertEqual(chosen["value"], "a")


if __name__ == "__main__":
    unittest.main()
