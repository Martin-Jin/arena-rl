import unittest

from arena_ai.config import AttackType, load_config
from arena_ai.sim.collision import bodies_overlap
from arena_ai.sim.simulate import FighterAction, step_game
from arena_ai.sim.state import FighterState

CONFIG = load_config()
GAME = CONFIG.game
COMBAT = GAME.combat
COUNTER = COMBAT.spec(AttackType.COUNTER)
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

    def test_a_walker_is_stopped_by_a_fighter_standing_still(self) -> None:
        fighters = duel(gap=1.0)
        for _ in range(120):
            step(fighters, agent=RIGHT)
        self.assertEqual(fighters["opponent"].x, 6.0)  # Not pushed.
        self.assertFalse(bodies_overlap(fighters["agent"], fighters["opponent"], GAME))

    def test_a_fighter_blocked_by_another_or_a_wall_has_no_velocity(self) -> None:
        fighters = duel(gap=1.0)
        for _ in range(120):
            step(fighters, agent=RIGHT)
        self.assertEqual(fighters["agent"].vx, 0.0)
        self.assertEqual(fighters["agent"].facing_dir, 1)  # Still facing what it pushes.
        step(fighters, agent=RIGHT, opponent=FighterAction(move_dir=1))
        self.assertGreater(fighters["agent"].vx, 0.0)  # Moving again once the way is free.
        wall = {"agent": FighterState(x=GAME.world.arena_width_m, facing_dir=1),
                "opponent": FighterState(x=1.0)}
        step(wall, agent=RIGHT)
        self.assertEqual(wall["agent"].vx, 0.0)
        self.assertEqual(wall["agent"].facing_dir, 1)

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

    def test_a_counter_lunges_through_the_opponent_and_ends_on_the_far_side(self) -> None:
        fighters = self._counter()
        agent, opponent = fighters["agent"], fighters["opponent"]
        overlapped = False
        for _ in range(COUNTER.lunge_last_frame):
            step(fighters, agent=ATTACK)
            overlapped = overlapped or bodies_overlap(agent, opponent, GAME)
        self.assertTrue(overlapped)  # The lunge passed through the opponent's body.
        self.assertFalse(bodies_overlap(agent, opponent, GAME))  # The lunge is over.
        self.assertGreater(agent.x, opponent.x)  # Cut through: ends beyond the opponent.

    def test_a_counter_carries_the_fighter_out_of_the_opponents_swing_range(self) -> None:
        # The opponent cannot turn round and hit the counterer at once: it has to walk first.
        fighters = self._counter()
        agent, opponent = fighters["agent"], fighters["opponent"]
        reach = WIDTH + COUNTER.range_m  # The farthest a swing hits, center to center.
        health = opponent.health
        for _ in range(COUNTER.lunge_last_frame):
            step(fighters, agent=ATTACK)
        self.assertLess(opponent.health, health)  # The counter still hit on the way through.
        clearance = COUNTER.lunge_clearance_reaches * reach
        self.assertGreaterEqual(agent.x - opponent.x, clearance - 1e-6)
        self.assertGreater(agent.x - opponent.x, reach)  # Out of range, not just apart.

    def test_a_counter_still_clears_an_opponent_that_chases_the_lunger(self) -> None:
        # Regression: a parried opponent can still walk, and at the lunge's own speed the
        # lunger gained so little on one chasing it that it ended 0.5 m away, in range.
        fighters = self._counter()
        agent, opponent = fighters["agent"], fighters["opponent"]
        reach = WIDTH + COUNTER.range_m
        for _ in range(COUNTER.attack_windup_frames + 1):
            step(fighters, agent=ATTACK)  # Up to the hit, with the opponent standing.
        for _ in range(COUNTER.lunge_last_frame - COUNTER.attack_windup_frames - 1):
            step(fighters, agent=ATTACK, opponent=RIGHT)  # Now it walks after the lunger.
        self.assertGreaterEqual(
            agent.x - opponent.x, COUNTER.lunge_clearance_reaches * reach - 1e-6
        )

    def test_a_counter_stays_level_with_the_opponent_on_the_hit_frame(self) -> None:
        # The carry starts after the hit is checked, so it does not make the counter miss.
        fighters = self._counter()
        agent, opponent = fighters["agent"], fighters["opponent"]
        for _ in range(COUNTER.attack_windup_frames + 1):
            step(fighters, agent=ATTACK)
        self.assertAlmostEqual(agent.x, opponent.x)
        self.assertTrue(agent.attack_hit)

    def test_a_counter_is_pushed_out_by_the_lunger_and_never_snaps_back(self) -> None:
        fighters = self._counter()
        agent, opponent = fighters["agent"], fighters["opponent"]
        opponent_x = opponent.x
        for _ in range(COUNTER.lunge_last_frame):
            step(fighters, agent=ATTACK)
        self.assertGreaterEqual(agent.x - opponent.x, WIDTH - 1e-6)
        self.assertEqual(opponent.x, opponent_x)  # The opponent was not shoved.
        for _ in range(COMBAT.attack_total_frames):
            step(fighters, agent=ATTACK)
            self.assertGreater(agent.x, opponent.x)

    def test_a_counter_can_still_be_parried_when_it_reaches_the_opponent(self) -> None:
        fighters = self._counter()
        for _ in range(COUNTER.attack_windup_frames):
            step(fighters, agent=ATTACK)
        health = fighters["opponent"].health
        step(fighters, agent=ATTACK, opponent=PARRY)  # The lunger is level with the opponent.
        self.assertTrue(fighters["opponent"].parried)
        self.assertEqual(fighters["opponent"].health, health)

    def test_a_parried_counter_stops_at_the_parrier_and_does_not_end_inside_it(self) -> None:
        fighters = self._counter()
        for _ in range(COUNTER.attack_windup_frames):
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


class HitboxTests(unittest.TestCase):
    def test_a_swing_never_hits_a_fighter_behind_it(self) -> None:
        fighters = duel(gap=WIDTH)  # Touching: as close as bodies get.
        fighters["agent"].facing_dir = -1  # Facing away from the opponent.
        for _ in range(COMBAT.attack_total_frames + 1):
            step(fighters, agent=ATTACK)
        self.assertEqual(fighters["opponent"].health, 100)

    def test_a_swing_hits_a_touching_fighter_in_front(self) -> None:
        fighters = duel(gap=WIDTH)
        for _ in range(COMBAT.attack_total_frames + 1):
            step(fighters, agent=ATTACK)
        self.assertEqual(fighters["opponent"].health, 100 - COMBAT.attack_damage)

    def test_a_counter_lunges_toward_the_fighter_it_parried(self) -> None:
        fighters = duel(gap=0.5)
        fighters["third"] = FighterState(x=4.0, facing_dir=1)  # Nearer on the other side.
        while fighters["opponent"].attack_frame < COMBAT.attack_windup_frames:
            step_game(
                fighters,
                {"agent": FighterAction(), "opponent": ATTACK, "third": FighterAction()},
                CONFIG,
            )
        step_game(
            fighters,
            {"agent": PARRY, "opponent": ATTACK, "third": FighterAction()},
            CONFIG,
        )
        agent = fighters["agent"]
        self.assertEqual(agent.counter_target, "opponent")
        fighters["third"].x = 4.9  # Now the nearest fighter, but not the one it parried.
        step_game(fighters, {"agent": ATTACK, "opponent": FighterAction(),
                             "third": FighterAction()}, CONFIG)
        self.assertEqual(agent.lunge_dir, 1)
