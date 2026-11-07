import random
import unittest
from unittest.mock import patch

from arena_ai.agents.action import to_fighter_action
from arena_ai.agents.scripted import OPPONENTS, SCRIPTED_OPPONENTS, parrying_hunter
from arena_ai.config import load_config
from arena_ai.sim.observation import get_observation
from arena_ai.sim.observation_vector import encode_observation
from arena_ai.sim.simulate import FighterAction, step_game
from arena_ai.sim.state import FighterState
from arena_ai.viewer.controllers import make_controller

CONFIG = load_config()
ATTACK = FighterAction(attack=True)


def play(fighters, learner_action, steps: int) -> int:
    """Step with the hunter as the opponent and return how often it parried."""
    parries = 0
    for _ in range(steps):
        observation = encode_observation(
            get_observation(fighters["agent"], {"opponent": fighters["opponent"]}, CONFIG),
            CONFIG,
        )
        step_game(
            fighters,
            {
                "agent": learner_action,
                "opponent": to_fighter_action(parrying_hunter(observation, CONFIG)),
            },
            CONFIG,
        )
        parries += fighters["opponent"].parried
    return parries


class ParryingHunterTests(unittest.TestCase):
    def setUp(self) -> None:
        # By default the hunter never takes its random chance to attack a still learner.
        patcher = patch("arena_ai.agents.scripted.random.random", return_value=0.99)
        self.chance = patcher.start()
        self.addCleanup(patcher.stop)

    def test_it_parries_a_learner_that_spams_attack_from_a_standstill(self) -> None:
        fighters = {
            "agent": FighterState(x=5.0, facing_dir=1),
            "opponent": FighterState(x=5.5, facing_dir=-1),
        }
        parries = play(fighters, ATTACK, 400)
        self.assertGreaterEqual(parries, 5)
        self.assertEqual(fighters["opponent"].health, 100)

    def test_it_is_for_practice_only(self) -> None:
        self.assertNotIn("parrier", OPPONENTS)
        self.assertIn("parrier", SCRIPTED_OPPONENTS)
        controller = make_controller("parrier", CONFIG, True, "opponent")
        fighters = {"agent": FighterState(x=3.0), "opponent": FighterState(x=6.0)}
        self.assertIsInstance(controller.action(fighters), int)

    def test_it_parries_every_attack_from_the_front(self) -> None:
        fighters = {
            "agent": FighterState(x=5.0, facing_dir=1, attack_frame=1),
            "opponent": FighterState(x=5.5, facing_dir=-1),
        }
        parries = play(fighters, ATTACK, 600)
        self.assertGreaterEqual(parries, 5)
        self.assertEqual(fighters["opponent"].health, 100)

    def test_it_attacks_a_learner_that_stands_still(self) -> None:
        self.chance.return_value = 0.0
        fighters = {
            "agent": FighterState(x=5.0, facing_dir=1),
            "opponent": FighterState(x=5.5, facing_dir=-1),
        }
        attacked = False
        for _ in range(60):
            play(fighters, FighterAction(), 1)
            attacked = attacked or fighters["opponent"].attack_frame > 0
        self.assertTrue(attacked)
        self.assertLess(fighters["agent"].health, 100)

    def test_it_attacks_a_still_learner_without_a_long_wait(self) -> None:
        rng = random.Random(1)  # A real, seeded source instead of the default fixed one.
        with patch("arena_ai.agents.scripted.random.random", rng.random):
            waits = []
            for _ in range(200):
                fighters = {
                    "agent": FighterState(x=5.0, facing_dir=1),
                    "opponent": FighterState(x=5.5, facing_dir=-1),
                }
                wait = 0
                while fighters["opponent"].attack_frame == 0 and wait < 120:
                    play(fighters, FighterAction(), 1)
                    wait += 1
                waits.append(wait)
        self.assertLess(sum(waits) / len(waits), 10)  # About 5 frames on average.
        self.assertLessEqual(max(waits), 40)

    def test_it_does_not_attack_a_learner_that_is_moving(self) -> None:
        fighters = {
            "agent": FighterState(x=5.0, facing_dir=1, vx=4.0),
            "opponent": FighterState(x=5.5, facing_dir=-1),
        }
        # Walking back and forth keeps the learner's velocity non-zero but in reach.
        for step_index in range(60):
            direction = 1 if (step_index // 5) % 2 == 0 else -1
            play(fighters, FighterAction(move_dir=direction), 1)
            self.assertEqual(fighters["opponent"].attack_frame, 0)

    def test_it_turns_to_parry_a_swing_from_behind(self) -> None:
        fighters = {  # The learner walked past: it is behind the hunter, which faces away.
            "agent": FighterState(x=5.0, facing_dir=1, attack_frame=1),
            "opponent": FighterState(x=5.5, facing_dir=1),
        }
        self.assertGreaterEqual(play(fighters, ATTACK, 20), 1)
        self.assertEqual(fighters["opponent"].health, 100)

    def test_it_parries_a_swing_from_above(self) -> None:
        fighters = {
            "agent": FighterState(
                x=5.2, y=0.8, on_ground=False, facing_dir=1, attack_frame=1
            ),
            "opponent": FighterState(x=5.5, facing_dir=-1),
        }
        self.assertGreaterEqual(play(fighters, ATTACK, 20), 1)
        self.assertEqual(fighters["opponent"].health, 100)

    def test_a_swing_from_behind_while_it_is_attacking_lands(self) -> None:
        fighters = {
            "agent": FighterState(x=5.4, facing_dir=-1),
            "opponent": FighterState(x=5.0, facing_dir=-1, attack_frame=1),
        }
        play(fighters, ATTACK, 20)
        self.assertEqual(fighters["opponent"].health, 100 - CONFIG.game.combat.attack_damage)

    def test_it_does_not_start_a_swing_while_the_learner_swings_in_reach(self) -> None:
        fighters = {
            "agent": FighterState(x=5.0, facing_dir=1, attack_frame=3),
            "opponent": FighterState(x=5.5, facing_dir=-1),
        }
        observation = encode_observation(
            get_observation(fighters["agent"], {"opponent": fighters["opponent"]}, CONFIG),
            CONFIG,
        )
        self.assertFalse(to_fighter_action(parrying_hunter(observation, CONFIG)).attack)


if __name__ == "__main__":
    unittest.main()
