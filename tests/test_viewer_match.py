import dataclasses
import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from arena_ai.config import load_config
from arena_ai.viewer.app import App
from arena_ai.viewer.scenes.match import MatchScene


def _app(time_limit: float) -> App:
    config = load_config()
    match = dataclasses.replace(config.viewer.match, time_limit_seconds=time_limit)
    config = dataclasses.replace(
        config, viewer=dataclasses.replace(config.viewer, match=match)
    )
    app = App(config)
    app.settings.player = "human"
    app.settings.opponent_count = 1
    app.settings.opponents = ["idle"] * app.config.game.observation.max_opponents
    app.settings.random_spawns = False
    return app


def _play(scene: MatchScene, frames: int = 600) -> None:
    for _ in range(frames):
        scene.update(1 / 60)
        if scene.phase == "over":
            return


class MatchEndTests(unittest.TestCase):
    def tearDown(self) -> None:
        pygame.quit()

    def test_every_animation_freezes_once_the_match_is_over(self) -> None:
        scene = MatchScene(_app(time_limit=0.0))
        scene.update(1 / 60)
        scene.phase = "playing"
        scene.fighters["opponent"].health = 10
        scene.fighters["agent"].x = scene.fighters["opponent"].x - 0.3
        scene.fighters["agent"].facing_dir = 1
        scene.fighters["agent"].attack_frame = scene.config.game.combat.attack_windup_frames
        scene.update(1 / 60)  # The swing lands and the match ends.
        self.assertEqual(scene.phase, "over")
        scene.update(1 / 60)  # Registers the final hit once, then everything stops.
        tick, hud_state = scene.render_tick, dict(scene.hud._health)
        for _ in range(30):
            scene.update(1 / 60)
        self.assertEqual(scene.render_tick, tick)
        self.assertEqual(scene.hud._health, hud_state)
        self.assertEqual(scene.hud._health["opponent"], 0)  # The final hit was shown.


class MatchTimeLimitTests(unittest.TestCase):
    def tearDown(self) -> None:
        pygame.quit()

    def test_a_match_with_no_winner_ends_as_a_draw_at_the_limit(self) -> None:
        scene = MatchScene(_app(time_limit=1.0))  # Nobody fights, so health is equal.
        _play(scene)
        self.assertEqual(scene.phase, "over")
        self.assertEqual(scene.outcome, "draw")
        self.assertEqual(scene.winner, "Draw")
        self.assertIn("Time's up", scene.subtitle)

    def test_the_healthier_fighter_wins_on_time(self) -> None:
        scene = MatchScene(_app(time_limit=1.0))
        scene.fighters["opponent"].health = 40
        _play(scene)
        self.assertEqual(scene.outcome, "agent")
        self.assertEqual(scene.winner, "Victory")

    def test_a_zero_limit_removes_the_time_limit(self) -> None:
        scene = MatchScene(_app(time_limit=0))
        _play(scene, frames=300)
        self.assertEqual(scene.phase, "playing")
        self.assertFalse(scene.time_limit)


if __name__ == "__main__":
    unittest.main()
