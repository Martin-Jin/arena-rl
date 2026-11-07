import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from arena_ai.config import load_config
from arena_ai.sim.state import FighterState
from arena_ai.viewer.audio import AudioManager, FighterSnapshot


class FighterSoundTests(unittest.TestCase):
    def setUp(self) -> None:
        pygame.init()
        self.audio = AudioManager(load_config().viewer.audio)
        self.played: list[str] = []
        self.audio.play = self.played.append  # Record instead of playing.

    def tearDown(self) -> None:
        self.audio.close()
        pygame.quit()

    def test_the_opponent_makes_sounds_too(self) -> None:
        opponent = FighterState()
        before = FighterSnapshot.of(opponent)
        opponent.attack_frame = 1
        opponent.on_ground = False
        self.audio.update_fighter("opponent", opponent, 0, before)
        self.assertEqual(self.played, ["attack", "jump"])
        before = FighterSnapshot.of(opponent)
        opponent.on_ground = True
        self.audio.update_fighter("opponent", opponent, 0, before)
        self.assertEqual(self.played[-1], "landing")

    def test_a_fighter_that_loses_health_plays_the_hit_sound(self) -> None:
        agent = FighterState()
        before = FighterSnapshot.of(agent)
        self.audio.update_fighter("agent", agent, 0, before)
        self.assertNotIn("hit", self.played)
        agent.health -= 10
        self.audio.update_fighter("agent", agent, 0, before)
        self.assertEqual(self.played, ["hit"])

    def test_each_fighter_has_its_own_running_loop(self) -> None:
        agent, opponent = FighterState(), FighterState()
        self.audio.update_fighter("agent", agent, 1, FighterSnapshot.of(agent))
        self.audio.update_fighter("opponent", opponent, 0, FighterSnapshot.of(opponent))
        loops = self.audio._loops
        self.assertTrue(loops["agent"].running)
        self.assertFalse(loops["opponent"].running)
        self.audio.update_fighter("opponent", opponent, -1, FighterSnapshot.of(opponent))
        self.assertTrue(loops["opponent"].running)
        self.assertTrue(loops["agent"].running)  # The opponent did not stop it.
        self.audio.silence()
        self.assertFalse(loops["agent"].running or loops["opponent"].running)


if __name__ == "__main__":
    unittest.main()
