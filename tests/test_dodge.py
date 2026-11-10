import unittest

from arena_ai.config import load_config
from arena_ai.sim.simulate import FighterAction, step_game
from arena_ai.sim.state import FighterState

CONFIG = load_config()
COMBAT = CONFIG.game.combat
DODGE = CONFIG.game.dodge
NONE = FighterAction()
ATTACK = FighterAction(attack=True)
PARRY = FighterAction(parry=True)
DODGE_ONLY = FighterAction(dodge=True)


def duel(gap: float = 0.5) -> dict[str, FighterState]:
    """Agent on the left facing right, opponent on the right facing left."""
    return {
        "agent": FighterState(x=5.0, facing_dir=1),
        "opponent": FighterState(x=5.0 + gap, facing_dir=-1),
    }


def step(fighters, agent=NONE, opponent=NONE) -> None:
    step_game(fighters, {"agent": agent, "opponent": opponent}, CONFIG)


def run(fighters, steps: int, agent=NONE, opponent=NONE) -> None:
    for _ in range(steps):
        step(fighters, agent, opponent)


class DodgeTests(unittest.TestCase):
    def test_a_dodge_bursts_the_way_the_fighter_faces_and_costs_stamina(self) -> None:
        fighters = duel(gap=5.0)
        step(fighters, DODGE_ONLY)
        agent = fighters["agent"]
        self.assertEqual(agent.stamina, DODGE.max_stamina - DODGE.cost)
        self.assertEqual(agent.dodge_frames_left, DODGE.duration_frames)
        run(fighters, DODGE.duration_frames - 1)
        distance = DODGE.speed_mps * DODGE.duration_frames / CONFIG.game.world.sim_hz
        self.assertAlmostEqual(agent.x, 5.0 + distance)

    def test_a_held_direction_turns_the_fighter_first_and_then_locks_its_facing(self) -> None:
        fighters = duel(gap=5.0)
        step(fighters, FighterAction(move_dir=-1, dodge=True))
        self.assertEqual(fighters["agent"].facing_dir, -1)
        self.assertLess(fighters["agent"].x, 5.0)
        step(fighters, FighterAction(move_dir=1))  # Mid-burst: neither turns nor changes course.
        self.assertEqual(fighters["agent"].facing_dir, -1)
        self.assertLess(fighters["agent"].vx, 0)

    def test_it_takes_no_damage_during_the_burst(self) -> None:
        fighters = duel(gap=0.4)
        fighters["opponent"].attack_frame = COMBAT.attack_windup_frames  # Active next step.
        step(fighters, DODGE_ONLY)
        self.assertEqual(fighters["agent"].health, 100)

    def test_it_can_be_hit_in_the_gap_after_the_burst(self) -> None:
        fighters = duel(gap=5.0)
        run(fighters, DODGE.duration_frames + 1, DODGE_ONLY)  # The extra step ends the burst.
        self.assertGreater(fighters["agent"].dodge_gap_left, 0)
        fighters["agent"].x = fighters["opponent"].x - 0.4
        fighters["opponent"].attack_frame = COMBAT.attack_windup_frames
        step(fighters)
        self.assertLess(fighters["agent"].health, 100)

    def test_a_full_bar_gives_two_dodges_with_a_gap_between(self) -> None:
        fighters = duel(gap=8.0)
        agent = fighters["agent"]
        step(fighters, DODGE_ONLY)
        run(fighters, DODGE.duration_frames - 1)
        step(fighters, DODGE_ONLY)  # A press in the gap does nothing.
        self.assertEqual(agent.dodge_frames_left, 0)
        run(fighters, DODGE.gap_frames - 1)
        step(fighters, DODGE_ONLY)
        self.assertEqual(agent.dodge_frames_left, DODGE.duration_frames)
        run(fighters, DODGE.duration_frames + DODGE.gap_frames)
        step(fighters, DODGE_ONLY)
        self.assertEqual(agent.dodge_frames_left, 0)  # No stamina for a third.

    def test_holding_the_button_dodges_once(self) -> None:
        fighters = duel(gap=8.0)
        # Long enough for the stamina to be back above the cost, so only the hold stops a repeat.
        run(fighters, 3 * DODGE.duration_frames, DODGE_ONLY)
        self.assertEqual(fighters["agent"].dodge_frames_left, 0)
        self.assertGreaterEqual(fighters["agent"].stamina, DODGE.cost)

    def test_stamina_refills_only_when_not_dodging(self) -> None:
        fighters = duel(gap=8.0)
        step(fighters, DODGE_ONLY)
        spent = fighters["agent"].stamina
        run(fighters, DODGE.duration_frames + DODGE.gap_frames - 1)
        self.assertEqual(fighters["agent"].stamina, spent)
        run(fighters, 61)  # The first step only ends the gap, then a second of refilling.
        self.assertAlmostEqual(fighters["agent"].stamina, spent + DODGE.regen_per_second)

    def test_it_passes_through_an_opponent_and_ends_on_its_far_side(self) -> None:
        fighters = duel(gap=0.3)
        run(fighters, DODGE.duration_frames, DODGE_ONLY)
        self.assertGreater(fighters["agent"].x, fighters["opponent"].x)
        self.assertEqual(fighters["opponent"].x, 5.3)

    def test_a_dodge_in_the_windup_cancels_the_swing(self) -> None:
        fighters = duel(gap=5.0)
        run(fighters, 3, ATTACK)
        step(fighters, FighterAction(attack=True, dodge=True))
        self.assertEqual(fighters["agent"].attack_frame, 0)
        self.assertGreater(fighters["agent"].dodge_frames_left, 0)

    def test_a_dodge_is_refused_once_the_swing_is_active(self) -> None:
        fighters = duel(gap=5.0)
        run(fighters, COMBAT.attack_windup_frames + 1, ATTACK)
        step(fighters, DODGE_ONLY)
        self.assertEqual(fighters["agent"].dodge_frames_left, 0)
        self.assertEqual(fighters["agent"].stamina, DODGE.max_stamina)

    def test_a_dodge_is_refused_while_parrying_or_staggered(self) -> None:
        fighters = duel(gap=5.0)
        step(fighters, PARRY)
        step(fighters, DODGE_ONLY)
        self.assertEqual(fighters["agent"].dodge_frames_left, 0)
        fighters = duel(gap=5.0)
        fighters["agent"].stagger_frames = 5
        step(fighters, DODGE_ONLY)
        self.assertEqual(fighters["agent"].dodge_frames_left, 0)

    def test_a_dodge_is_refused_in_the_air(self) -> None:
        fighters = duel(gap=5.0)
        step(fighters, FighterAction(jump=True))
        self.assertFalse(fighters["agent"].on_ground)
        step(fighters, DODGE_ONLY)
        self.assertEqual(fighters["agent"].dodge_frames_left, 0)
        self.assertEqual(fighters["agent"].stamina, DODGE.max_stamina)

    def test_it_cannot_jump_out_of_a_dodge(self) -> None:
        fighters = duel(gap=5.0)
        step(fighters, FighterAction(dodge=True, jump=True))
        self.assertTrue(fighters["agent"].on_ground)
        self.assertGreater(fighters["agent"].dodge_frames_left, 0)

    def test_it_cannot_attack_during_the_burst(self) -> None:
        fighters = duel(gap=5.0)
        step(fighters, FighterAction(dodge=True, attack=True))
        step(fighters, ATTACK)
        self.assertEqual(fighters["agent"].attack_frame, 0)


if __name__ == "__main__":
    unittest.main()
