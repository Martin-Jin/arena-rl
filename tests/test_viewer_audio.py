import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from arena_ai.config import load_config
from arena_ai.sim.state import FighterState
from arena_ai.viewer.audio import AudioManager, FighterSnapshot


class FootstepLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        pygame.init()
        self.audio = AudioManager(load_config().viewer.audio, 2)

    def tearDown(self) -> None:
        self.audio.close()
        pygame.quit()

    def test_footsteps_that_change_direction_every_step_stop_for_good(self) -> None:
        # Regression: fading a loop out the moment it started left its channel playing
        # forever, so the walking sound carried on after the match ended. Two fighters
        # turning every step (a chase, or a dodge) do exactly that.
        fighter = FighterState(on_ground=True)
        snapshot = FighterSnapshot.of(fighter)
        for step in range(12):
            fighter.vx = 4.0 if step % 2 else -4.0
            self.audio.update_fighter("agent", fighter, snapshot)
        self.audio.silence()
        pygame.mixer.fadeout(self.audio.match_end_fade_ms)
        pygame.time.wait(self.audio.match_end_fade_ms + 300)
        channels = self.audio._loops["agent"]._channels
        self.assertFalse(any(channel.get_busy() for channel in channels))


class FighterSoundTests(unittest.TestCase):
    def setUp(self) -> None:
        pygame.init()
        self.audio = AudioManager(load_config().viewer.audio, 2)
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
        self.audio.update_fighter("opponent", opponent, before)
        self.assertEqual(self.played, ["attack", "jump"])
        before = FighterSnapshot.of(opponent)
        opponent.on_ground = True
        self.audio.update_fighter("opponent", opponent, before)
        self.assertEqual(self.played[-1], "landing")

    def test_starting_a_parry_plays_the_shield_raise_sound_once(self) -> None:
        fighter = FighterState()
        before = FighterSnapshot.of(fighter)
        fighter.parry_frame = 1
        self.audio.update_fighter("agent", fighter, before)
        self.assertEqual(self.played, ["shield_raise"])
        before = FighterSnapshot.of(fighter)
        fighter.parry_frame = 2  # Still parrying: no new sound.
        self.audio.update_fighter("agent", fighter, before)
        self.assertEqual(self.played, ["shield_raise"])

    def test_a_counter_plays_its_own_sound_instead_of_the_attack_sound(self) -> None:
        fighter = FighterState()
        before = FighterSnapshot.of(fighter)
        fighter.attack_frame = 1
        fighter.counter_started = True
        self.audio.update_fighter("agent", fighter, before)
        self.assertEqual(self.played, ["counter"])

    def test_parries_alternate_between_the_two_sounds(self) -> None:
        fighter = FighterState(parried=True)
        before = FighterSnapshot.of(fighter)
        for _ in range(4):
            self.audio.update_fighter("agent", fighter, before)
        self.assertEqual(len(self.played), 4)
        self.assertEqual(set(self.played), {"parry_1", "parry_2"})
        self.assertEqual(self.played[0::2], [self.played[0]] * 2)
        self.assertNotEqual(self.played[0], self.played[1])

    def test_a_fighter_that_loses_health_plays_the_hit_sound(self) -> None:
        agent = FighterState()
        before = FighterSnapshot.of(agent)
        self.audio.update_fighter("agent", agent, before)
        self.assertNotIn("hit", self.played)
        agent.health -= 10
        self.audio.update_fighter("agent", agent, before)
        self.assertEqual(self.played, ["hit"])

    def test_music_can_be_cut_and_restarted(self) -> None:
        self.audio.start_music()
        self.assertTrue(pygame.mixer.music.get_busy())
        self.audio.stop_music()
        self.assertFalse(pygame.mixer.music.get_busy())
        self.audio.ensure_music()
        self.assertTrue(pygame.mixer.music.get_busy())
        self.audio.ensure_music()  # Already playing: it is not restarted.
        self.assertTrue(pygame.mixer.music.get_busy())

    def test_fading_out_the_effects_lets_them_finish_fading(self) -> None:
        del self.audio.play
        self.audio.play("parry_1")
        self.audio.fade_out_effects()
        busy = sum(pygame.mixer.Channel(i).get_busy() for i in range(16))
        self.assertGreaterEqual(busy, 1)  # Still fading, not cut off at once.
        pygame.time.wait(self.audio.match_end_fade_ms + 300)
        busy = sum(pygame.mixer.Channel(i).get_busy() for i in range(16))
        self.assertEqual(busy, 0)

    def test_cutting_the_effects_stops_one_shots_and_footsteps(self) -> None:
        del self.audio.play  # Use the real player for this test.
        fighter = FighterState(vx=4.0)
        self.audio.update_fighter(
            "agent", fighter, FighterSnapshot.of(fighter)
        )  # Starts the footsteps.
        self.audio.play("parry_1")
        busy = lambda: sum(pygame.mixer.Channel(i).get_busy() for i in range(16))
        self.assertGreaterEqual(busy(), 2)
        self.audio.cut_effects()
        self.assertEqual(busy(), 0)
        # The footsteps start again afterwards, as the state was reset with the channels.
        self.audio.update_fighter("agent", fighter, FighterSnapshot.of(fighter))
        self.assertGreaterEqual(busy(), 1)

    def test_no_footsteps_while_parrying_even_with_a_direction_held(self) -> None:
        fighter = FighterState(vx=4.0)
        self.audio.update_fighter("agent", fighter, FighterSnapshot.of(fighter))
        self.assertTrue(self.audio._loops["agent"].running)
        fighter.vx = 0.0  # A parrying fighter cannot walk, whatever key is held.
        fighter.parry_frame = 3
        self.audio.update_fighter("agent", fighter, FighterSnapshot.of(fighter))
        self.assertFalse(self.audio._loops["agent"].running)

    def test_each_fighter_has_its_own_running_loop(self) -> None:
        agent, opponent = FighterState(vx=4.0), FighterState()
        self.audio.update_fighter("agent", agent, FighterSnapshot.of(agent))
        self.audio.update_fighter("opponent", opponent, FighterSnapshot.of(opponent))
        loops = self.audio._loops
        self.assertTrue(loops["agent"].running)
        self.assertFalse(loops["opponent"].running)
        opponent.vx = -4.0
        self.audio.update_fighter("opponent", opponent, FighterSnapshot.of(opponent))
        self.assertTrue(loops["opponent"].running)
        self.assertTrue(loops["agent"].running)  # The opponent did not stop it.
        self.audio.silence()
        self.assertFalse(loops["agent"].running or loops["opponent"].running)


if __name__ == "__main__":
    unittest.main()
