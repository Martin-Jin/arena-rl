import unittest

from arena_ai.config import load_config
from arena_ai.sim.collision import bodies_overlap
from arena_ai.sim.simulate import FighterAction, step_game
from arena_ai.sim.state import FighterState

CONFIG = load_config()
GAME = CONFIG.game
COMBAT = GAME.combat
WIDTH = GAME.fighter.width_m
ATTACK = FighterAction(attack=True)
PARRY = FighterAction(parry=True)
RIGHT = FighterAction(move_dir=1)
LEFT = FighterAction(move_dir=-1)


def step(fighters, agent=FighterAction(), opponent=FighterAction()) -> None:
    step_game(fighters, {"agent": agent, "opponent": opponent}, CONFIG)


def duel(gap: float) -> dict[str, FighterState]:
    return {
        "agent": FighterState(x=5.0, facing_dir=1),
        "opponent": FighterState(x=5.0 + gap, facing_dir=-1),
    }


class CollisionTests(unittest.TestCase):
    def test_fighters_cannot_walk_through_each_other(self) -> None:
        fighters = duel(gap=1.0)
        for _ in range(120):
            step(fighters, agent=RIGHT, opponent=LEFT)
            self.assertLess(fighters["agent"].x, fighters["opponent"].x)
            self.assertFalse(bodies_overlap(fighters["agent"], fighters["opponent"], GAME))
        self.assertGreaterEqual(fighters["opponent"].x - fighters["agent"].x, WIDTH - 1e-6)

    def test_a_walker_pushes_a_fighter_standing_still(self) -> None:
        fighters = duel(gap=1.0)
        for _ in range(120):
            step(fighters, agent=RIGHT)
        self.assertGreater(fighters["opponent"].x, 6.0)
        self.assertFalse(bodies_overlap(fighters["agent"], fighters["opponent"], GAME))

    def test_fighters_that_start_on_top_of_each_other_are_separated(self) -> None:
        fighters = duel(gap=0.0)
        step(fighters)
        self.assertGreaterEqual(fighters["opponent"].x - fighters["agent"].x, WIDTH - 1e-6)

    def test_a_fighter_against_a_wall_is_not_pushed_through_it(self) -> None:
        fighters = duel(gap=0.1)
        fighters["agent"].x = 0.0
        fighters["opponent"].x = 0.1
        step(fighters, opponent=LEFT)
        self.assertEqual(fighters["agent"].x, 0.0)
        self.assertGreaterEqual(fighters["opponent"].x, WIDTH - 1e-6)

    def test_a_jump_clears_the_opponent(self) -> None:
        fighters = duel(gap=0.5)
        fighters["agent"].facing_dir = 1
        crossed = False
        for i in range(60):
            step(fighters, agent=FighterAction(move_dir=1, jump=(i == 0)))
            crossed = crossed or fighters["agent"].x > fighters["opponent"].x
        self.assertTrue(crossed)
        self.assertFalse(bodies_overlap(fighters["agent"], fighters["opponent"], GAME))

    def _counter(self):
        """Agent parries an attack and is ready to counter. Returns the fighters."""
        fighters = duel(gap=0.5)
        while fighters["opponent"].attack_frame < COMBAT.attack_windup_frames:
            step(fighters, opponent=ATTACK)
        step(fighters, agent=PARRY, opponent=ATTACK)
        return fighters

    def test_a_counter_lunges_through_the_opponent_and_is_pushed_out_in_front(self) -> None:
        fighters = self._counter()
        agent, opponent = fighters["agent"], fighters["opponent"]
        overlapped = False
        for _ in range(COMBAT.counter_windup_frames + 1):
            step(fighters, agent=ATTACK)
            overlapped = overlapped or bodies_overlap(agent, opponent, GAME)
        self.assertTrue(overlapped)  # The lunge passed through the opponent's body.
        self.assertFalse(bodies_overlap(agent, opponent, GAME))  # The wind-up is over.
        self.assertLess(agent.x, opponent.x)  # Back in front, where it can still be parried.

    def test_a_parried_counter_stops_at_the_parrier_and_does_not_end_inside_it(self) -> None:
        fighters = self._counter()
        for _ in range(COMBAT.counter_windup_frames):
            step(fighters, agent=ATTACK)
        step(fighters, agent=ATTACK, opponent=PARRY)
        agent, opponent = fighters["agent"], fighters["opponent"]
        self.assertTrue(opponent.parried)
        self.assertEqual(agent.attack_frame, 0)
        self.assertFalse(bodies_overlap(agent, opponent, GAME))
        self.assertLess(agent.x, opponent.x)

    def test_a_counter_that_hits_does_not_leave_the_fighters_stuck(self) -> None:
        fighters = self._counter()
        for _ in range(COMBAT.attack_total_frames):
            step(fighters, agent=ATTACK)
        self.assertFalse(bodies_overlap(fighters["agent"], fighters["opponent"], GAME))
