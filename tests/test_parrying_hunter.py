import random
import unittest
from unittest.mock import patch

from arena_ai.agents.action import Action, to_fighter_action
from arena_ai.agents.scripted import OPPONENTS, parrying_hunter, runner
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

    def test_it_is_a_scripted_opponent_for_training_and_the_viewer(self) -> None:
        self.assertIs(OPPONENTS["parrier"], parrying_hunter)
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

    def _hunter_attacks(self, learner_x: float, hunter_x: float, action: FighterAction) -> bool:
        """Return whether the hunter ever damages a learner that keeps doing ``action``."""
        self.chance.return_value = 0.0
        fighters = {
            "agent": FighterState(x=learner_x, facing_dir=1 if hunter_x > learner_x else -1),
            "opponent": FighterState(x=hunter_x, facing_dir=-1 if hunter_x > learner_x else 1),
        }
        play(fighters, action, 300)
        return fighters["agent"].health < 100

    def test_it_attacks_a_learner_that_keeps_walking_into_it(self) -> None:
        self.assertTrue(self._hunter_attacks(5.0, 5.5, FighterAction(move_dir=1)))

    def test_it_attacks_a_learner_that_keeps_walking_into_a_wall(self) -> None:
        wall = CONFIG.game.world.arena_width_m
        self.assertTrue(self._hunter_attacks(wall, wall - 3.0, FighterAction(move_dir=1)))

    def test_it_attacks_a_learner_that_keeps_jumping(self) -> None:
        self.assertTrue(self._hunter_attacks(5.0, 5.5, FighterAction(jump=True)))

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

    def test_it_attacks_a_learner_that_swings_away_from_it(self) -> None:
        # Regression: the hunter treated every swing in reach as a threat even when it
        # pointed away, so it kept waiting and almost never attacked a spamming learner.
        self.chance.return_value = 0.0
        fighters = {
            "agent": FighterState(x=5.0, facing_dir=-1, attack_frame=3),
            "opponent": FighterState(x=5.5, facing_dir=-1),
        }
        observation = encode_observation(
            get_observation(fighters["agent"], {"opponent": fighters["opponent"]}, CONFIG),
            CONFIG,
        )
        self.assertTrue(to_fighter_action(parrying_hunter(observation, CONFIG)).attack)


class RunnerTests(unittest.TestCase):
    def observe(self, learner_x: float, runner_x: float, **runner_state) -> list[float]:
        fighters = {
            "agent": FighterState(x=learner_x),
            "opponent": FighterState(x=runner_x, **runner_state),
        }
        return encode_observation(
            get_observation(fighters["agent"], {"opponent": fighters["opponent"]}, CONFIG),
            CONFIG,
        )

    def test_it_runs_away_from_the_learner(self) -> None:
        self.assertEqual(runner(self.observe(3.0, 5.0), CONFIG), int(Action.MOVE_RIGHT))
        self.assertEqual(runner(self.observe(7.0, 5.0), CONFIG), int(Action.MOVE_LEFT))

    def test_it_jumps_over_the_learner_when_cornered(self) -> None:
        wall = CONFIG.game.world.arena_width_m
        action = runner(self.observe(wall - 0.8, wall - 0.1), CONFIG)
        self.assertEqual(action, int(Action.MOVE_LEFT_JUMP))

    def test_it_jumps_over_the_learner_before_it_reaches_the_wall(self) -> None:
        wall = CONFIG.game.world.arena_width_m
        action = runner(self.observe(wall - 3.0, wall - 1.5), CONFIG)
        self.assertEqual(action, int(Action.MOVE_LEFT_JUMP))

    def test_it_does_not_shuffle_at_the_wall_while_a_learner_walks_slowly_at_it(self) -> None:
        # Regression: walking toward the learner left the wall zone, fleeing re-entered it,
        # so the runner switched direction every few frames and stayed pinned.
        wall = CONFIG.game.world.arena_width_m
        fighters = {
            "agent": FighterState(x=wall - 6.0),
            "opponent": FighterState(x=wall - 0.1, facing_dir=-1),
        }
        switches, last = 0, 0
        for step_index in range(300):
            observation = encode_observation(
                get_observation(fighters["agent"], {"opponent": fighters["opponent"]}, CONFIG),
                CONFIG,
            )
            action = to_fighter_action(runner(observation, CONFIG))
            if action.move_dir:
                switches += bool(last) and action.move_dir != last
                last = action.move_dir
            walk = 1 if step_index % 3 == 0 else 0
            step_game(fighters, {"agent": FighterAction(move_dir=walk), "opponent": action}, CONFIG)
        self.assertLessEqual(switches, 3)

    def test_it_heads_back_past_a_distant_learner_instead_of_waiting_at_the_wall(self) -> None:
        # Regression: cornered with the learner not yet close, it used to stand still.
        wall = CONFIG.game.world.arena_width_m
        action = runner(self.observe(wall - 5.0, wall - 0.1), CONFIG)
        self.assertEqual(action, int(Action.MOVE_LEFT))

    def test_it_is_a_scripted_opponent_for_training(self) -> None:
        self.assertIs(OPPONENTS["runner"], runner)


if __name__ == "__main__":
    unittest.main()
