import unittest

from arena_ai.agents.action import Action, to_fighter_action
from arena_ai.config import load_config
from arena_ai.envs.reward import calculate_reward
from arena_ai.sim.observation import get_observation
from arena_ai.sim.observation_vector import (
    FEATURE_INDEX,
    encode_observation,
    observation_size,
)
from arena_ai.sim.rules import parry_frames_left
from arena_ai.sim.simulate import FighterAction, step_game
from arena_ai.sim.state import FighterState

CONFIG = load_config()
COMBAT = CONFIG.game.combat
NONE = FighterAction()
ATTACK = FighterAction(attack=True)
PARRY = FighterAction(parry=True)
FULL_HEALTH = {"agent": 100, "opponent": 100}


def duel(gap: float = 0.5) -> dict[str, FighterState]:
    """Agent on the left facing right, opponent on the right facing left."""
    return {
        "agent": FighterState(x=5.0, facing_dir=1),
        "opponent": FighterState(x=5.0 + gap, facing_dir=-1),
    }


def step(fighters, agent=NONE, opponent=NONE) -> None:
    step_game(fighters, {"agent": agent, "opponent": opponent}, CONFIG)


def wind_up_opponent(fighters) -> None:
    """Step until the opponent's swing is one step away from its active frames."""
    while fighters["opponent"].attack_frame < COMBAT.attack_windup_frames:
        step(fighters, opponent=ATTACK)


class ParryTests(unittest.TestCase):
    def test_attack_does_no_damage_during_windup(self) -> None:
        fighters = duel()
        for _ in range(COMBAT.attack_windup_frames):
            step(fighters, opponent=ATTACK)
        self.assertEqual(fighters["agent"].health, 100)
        step(fighters, opponent=ATTACK)
        self.assertEqual(fighters["agent"].health, 100 - COMBAT.attack_damage)

    def test_a_timed_parry_cancels_the_attack_and_opens_a_counter(self) -> None:
        fighters = duel()
        wind_up_opponent(fighters)
        step(fighters, agent=PARRY, opponent=ATTACK)  # The hit lands inside the window.
        agent, opponent = fighters["agent"], fighters["opponent"]
        self.assertEqual(agent.health, 100)
        self.assertTrue(agent.parried)
        self.assertEqual((agent.parry_frame, agent.parry_count), (0, 0))
        self.assertEqual(agent.counter_frames, COMBAT.counter_window_frames)
        self.assertEqual((opponent.attack_frame, opponent.attack_hit), (0, False))
        self.assertAlmostEqual(
            calculate_reward(FULL_HEALTH, fighters, CONFIG),
            CONFIG.scoring.parry_reward,
        )

    def test_a_late_parry_takes_the_hit(self) -> None:
        fighters = duel()
        step(fighters, agent=PARRY)
        for _ in range(fighters["agent"].parry_window):
            step(fighters)
        wind_up_opponent(fighters)
        step(fighters, opponent=ATTACK)
        self.assertEqual(fighters["agent"].health, 100 - COMBAT.attack_damage)

    def test_an_attack_from_behind_is_not_parried(self) -> None:
        fighters = duel()
        fighters["agent"].facing_dir = -1  # Facing away from the opponent.
        wind_up_opponent(fighters)
        step(fighters, agent=PARRY, opponent=ATTACK)
        self.assertEqual(fighters["agent"].health, 100 - COMBAT.attack_damage)

    def test_an_attack_from_above_is_parried(self) -> None:
        fighters = duel(gap=0.1)
        fighters["opponent"].y = 0.4
        fighters["opponent"].on_ground = False
        fighters["opponent"].attack_frame = COMBAT.attack_windup_frames
        step(fighters, agent=PARRY, opponent=ATTACK)
        self.assertTrue(fighters["agent"].parried)
        self.assertEqual(fighters["agent"].health, 100)

    def test_holding_parry_parries_once_and_blocks_movement(self) -> None:
        fighters = duel(gap=5.0)
        start_x = fighters["agent"].x
        step(fighters, agent=FighterAction(move_dir=1, jump=True, parry=True))
        self.assertTrue(fighters["agent"].on_ground)  # No jumping while parrying.
        self.assertEqual(fighters["agent"].x, start_x)  # No walking while parrying.
        hold = FighterAction(move_dir=1, parry=True)
        total = fighters["agent"].parry_window + COMBAT.parry_recovery_frames
        for _ in range(total + 2):
            step(fighters, agent=hold)
        agent = fighters["agent"]
        self.assertEqual(agent.parry_count, 1)  # One press, however long it was held.
        self.assertEqual(agent.parry_frame, 0)
        # Walked on the last three steps only, once the parry had ended.
        self.assertAlmostEqual(
            agent.x - start_x,
            3 * CONFIG.game.fighter.move_speed_mps * CONFIG.game.world.dt_seconds,
        )

    def test_spamming_halves_the_window_and_a_pause_restores_it(self) -> None:
        fighters = duel(gap=5.0)
        windows = []
        for _ in range(4):
            step(fighters, agent=PARRY)
            windows.append(fighters["agent"].parry_window)
            step(fighters)  # Release so that the next press is a new one.
            while fighters["agent"].parry_frame:
                step(fighters)
        self.assertEqual(windows[0], COMBAT.parry_window_frames)
        halved = int(COMBAT.parry_window_frames * COMBAT.parry_window_decay)
        self.assertEqual(windows[1], max(COMBAT.parry_min_window_frames, halved))
        self.assertEqual(windows[-1], COMBAT.parry_min_window_frames)
        for _ in range(COMBAT.parry_spam_reset_frames):
            step(fighters)
        step(fighters, agent=PARRY)
        self.assertEqual(fighters["agent"].parry_window, COMBAT.parry_window_frames)

    def test_pressing_a_direction_with_parry_does_not_turn_toward_an_attacker(self) -> None:
        fighters = duel()
        fighters["agent"].facing_dir = -1  # The opponent is behind.
        wind_up_opponent(fighters)
        step(fighters, agent=FighterAction(move_dir=1, parry=True), opponent=ATTACK)
        self.assertEqual(fighters["agent"].facing_dir, -1)
        self.assertEqual(fighters["agent"].health, 100 - COMBAT.attack_damage)

    def test_a_fighter_can_turn_mid_attack_and_the_swing_follows(self) -> None:
        fighters = duel()
        fighters["opponent"].x = 4.5  # Behind the agent, which faces right.
        for _ in range(COMBAT.attack_windup_frames):
            step(fighters, agent=ATTACK)
        self.assertEqual(fighters["opponent"].health, 100)
        step(fighters, agent=FighterAction(attack=True, move_dir=-1))
        self.assertEqual(fighters["agent"].facing_dir, -1)
        self.assertEqual(fighters["opponent"].health, 100 - COMBAT.attack_damage)

    def test_a_fighter_can_jump_while_attacking(self) -> None:
        fighters = duel(gap=5.0)
        step(fighters, agent=ATTACK)
        step(fighters, agent=FighterAction(attack=True, jump=True))
        agent = fighters["agent"]
        self.assertFalse(agent.on_ground)
        self.assertEqual(agent.attack_frame, 2)  # The swing carried on.

    def test_attack_and_parry_never_overlap(self) -> None:
        fighters = duel(gap=5.0)
        step(fighters, agent=ATTACK)
        step(fighters, agent=PARRY)
        self.assertEqual(fighters["agent"].parry_frame, 0)
        fighters = duel(gap=5.0)
        step(fighters, agent=FighterAction(attack=True, parry=True))
        agent = fighters["agent"]
        self.assertEqual((agent.parry_frame, agent.attack_frame), (1, 0))

    def test_counter_has_a_short_windup_and_lunges_toward_the_opponent(self) -> None:
        fighters = duel()
        wind_up_opponent(fighters)
        step(fighters, agent=PARRY, opponent=ATTACK)
        fighters["opponent"].x += 0.9  # Out of normal reach: only the lunge can hit.
        start_x = fighters["agent"].x
        for _ in range(COMBAT.counter_windup_frames):
            step(fighters, agent=ATTACK)
            self.assertEqual(fighters["opponent"].health, 100)
        step(fighters, agent=ATTACK)
        self.assertTrue(fighters["agent"].attack_counter)
        self.assertGreater(fighters["agent"].x, start_x)
        self.assertEqual(fighters["opponent"].health, 100 - COMBAT.counter_damage)

    def test_starting_a_counter_sets_a_one_step_event(self) -> None:
        fighters = duel()
        wind_up_opponent(fighters)
        step(fighters, agent=PARRY, opponent=ATTACK)
        step(fighters, agent=ATTACK)
        self.assertTrue(fighters["agent"].counter_started)
        step(fighters, agent=ATTACK)
        self.assertFalse(fighters["agent"].counter_started)
        normal = duel(gap=5.0)
        step(normal, agent=ATTACK)
        self.assertFalse(normal["agent"].counter_started)  # A normal swing is not one.

    def test_a_counter_that_hits_is_rewarded_once_and_a_normal_hit_is_not(self) -> None:
        fighters = duel()
        wind_up_opponent(fighters)
        step(fighters, agent=PARRY, opponent=ATTACK)
        hit_steps = 0
        for _ in range(COMBAT.counter_windup_frames + 3):
            step(fighters, agent=ATTACK)
            if fighters["agent"].counter_hit:
                hit_steps += 1
                self.assertAlmostEqual(
                    calculate_reward({"agent": 100, "opponent": 100}, fighters, CONFIG)
                    - CONFIG.scoring.damage_reward_scale * COMBAT.counter_damage,
                    CONFIG.scoring.counter_reward,
                )
        self.assertEqual(hit_steps, 1)
        normal = duel()
        for _ in range(COMBAT.attack_windup_frames + 2):
            step(normal, agent=ATTACK)
        self.assertEqual(normal["opponent"].health, 100 - COMBAT.attack_damage)
        self.assertFalse(normal["agent"].counter_hit)

    def test_a_fleeing_opponent_is_still_hit_by_the_counter(self) -> None:
        fighters = duel()
        wind_up_opponent(fighters)
        step(fighters, agent=PARRY, opponent=ATTACK)
        flee = FighterAction(move_dir=1)
        for _ in range(COMBAT.counter_windup_frames + 1):
            step(fighters, agent=ATTACK, opponent=flee)
        self.assertEqual(fighters["opponent"].health, 100 - COMBAT.counter_damage)

    def test_walking_through_the_attacker_does_not_dodge_the_counter(self) -> None:
        for gap in (0.1, 0.5, 0.9):
            fighters = duel(gap=gap)
            wind_up_opponent(fighters)
            step(fighters, agent=PARRY, opponent=ATTACK)
            fighters["opponent"].x = fighters["agent"].x + gap
            walk = FighterAction(move_dir=-1)  # Toward the attacker, and then past it.
            for _ in range(COMBAT.counter_windup_frames + 1):
                step(fighters, agent=ATTACK, opponent=walk)
            self.assertEqual(
                fighters["opponent"].health, 100 - COMBAT.counter_damage, f"gap {gap}"
            )

    def test_a_counter_can_be_parried_back(self) -> None:
        fighters = duel()
        wind_up_opponent(fighters)
        step(fighters, agent=PARRY, opponent=ATTACK)
        for _ in range(COMBAT.counter_windup_frames):
            step(fighters, agent=ATTACK)
        step(fighters, agent=ATTACK, opponent=PARRY)  # The first active frame.
        self.assertTrue(fighters["opponent"].parried)
        self.assertEqual(fighters["opponent"].health, 100)
        self.assertEqual(fighters["agent"].attack_frame, 0)

    def test_only_a_successful_air_parry_hovers(self) -> None:
        fighters = duel()
        for fighter in fighters.values():
            fighter.y, fighter.on_ground = 1.0, False
        fighters["agent"].vy = -4.0
        fighters["opponent"].attack_frame = COMBAT.attack_windup_frames
        step(fighters, agent=PARRY, opponent=ATTACK)
        agent = fighters["agent"]
        self.assertTrue(agent.parried)
        self.assertGreater(agent.vy, -4.0)  # Lifted by the impulse...
        self.assertLessEqual(agent.vy, COMBAT.parry_hover_max_vy_mps)  # ...never past the cap.
        self.assertEqual(agent.hover_frames, COMBAT.parry_hover_frames)

        spam = duel(gap=5.0)
        spam["agent"].y, spam["agent"].on_ground = 1.0, False
        step(spam, agent=PARRY)
        self.assertEqual(spam["agent"].hover_frames, 0)

    def test_misses_are_penalized(self) -> None:
        scoring = CONFIG.scoring
        fighters = duel(gap=5.0)
        for _ in range(COMBAT.swing_total_frames(False) + 1):
            step(fighters, agent=ATTACK)
            if fighters["agent"].missed_attack:
                break
        self.assertTrue(fighters["agent"].missed_attack)
        self.assertAlmostEqual(
            calculate_reward(FULL_HEALTH, fighters, CONFIG),
            -scoring.missed_attack_penalty,
        )
        fighters = duel(gap=5.0)
        step(fighters, agent=PARRY)
        while not fighters["agent"].missed_parry:
            step(fighters)
        self.assertAlmostEqual(
            calculate_reward(FULL_HEALTH, fighters, CONFIG),
            -scoring.missed_parry_penalty,
        )

    def test_a_parried_attacker_cannot_attack_until_the_stagger_ends(self) -> None:
        fighters = duel()
        wind_up_opponent(fighters)
        step(fighters, agent=PARRY, opponent=ATTACK)
        opponent = fighters["opponent"]
        self.assertEqual(opponent.stagger_frames, COMBAT.parried_stagger_frames)
        move = FighterAction(move_dir=1, attack=True)
        for _ in range(COMBAT.parried_stagger_frames):
            step(fighters, opponent=move)
            self.assertEqual(opponent.attack_frame, 0)
        self.assertGreater(opponent.x, 5.5)  # It could still move.
        step(fighters, opponent=ATTACK)
        self.assertEqual(opponent.attack_frame, 1)

    def test_a_parried_attacker_can_still_parry(self) -> None:
        fighters = duel()
        wind_up_opponent(fighters)
        step(fighters, agent=PARRY, opponent=ATTACK)
        step(fighters, opponent=PARRY)
        self.assertEqual(fighters["opponent"].parry_frame, 1)

    def test_the_parry_animation_plays_on_after_a_success(self) -> None:
        fighters = duel()
        wind_up_opponent(fighters)
        step(fighters, agent=PARRY, opponent=ATTACK)
        agent = fighters["agent"]
        self.assertEqual((agent.parry_frame, agent.parry_pose_frame), (0, 1))
        step(fighters)
        self.assertEqual(agent.parry_pose_frame, 2)
        step(fighters, agent=ATTACK)  # A counter does not cut the shield effects short.
        self.assertEqual(agent.parry_pose_frame, 3)
        step(fighters, agent=ATTACK)
        self.assertEqual(agent.parry_pose_frame, 4)
        for _ in range(20):  # It runs out by itself, a little after the window ends.
            step(fighters)
        self.assertEqual(agent.parry_pose_frame, 0)

    def test_an_uneventful_step_costs_nothing(self) -> None:
        # There is no per-step time cost: it made losing sooner cheaper than a long fight.
        fighters = duel(gap=5.0)
        step(fighters)
        self.assertAlmostEqual(calculate_reward(FULL_HEALTH, fighters, CONFIG), 0.0)

    def test_observation_exposes_the_parry_state(self) -> None:
        fighters = duel()
        step(fighters, agent=PARRY)
        structured = get_observation(
            fighters["agent"], {"opponent": fighters["opponent"]}, CONFIG
        )
        vector = encode_observation(structured, CONFIG)
        self.assertEqual(len(vector), observation_size(CONFIG.game.observation.max_opponents))
        window = fighters["agent"].parry_window
        self.assertAlmostEqual(
            vector[FEATURE_INDEX["parry_window_left"]], window / COMBAT.parry_window_frames
        )
        self.assertAlmostEqual(vector[FEATURE_INDEX["parry_recovery_left"]], 1.0)
        self.assertAlmostEqual(
            vector[FEATURE_INDEX["busy_frames_left"]],
            (window + COMBAT.parry_recovery_frames)
            / max(COMBAT.longest_attack_frames, COMBAT.parry_total_frames),
        )

    def test_parry_countdowns_end_at_one_on_the_last_frame(self) -> None:
        fighter = FighterState(parry_frame=1, parry_window=6)
        recovery = COMBAT.parry_recovery_frames
        self.assertEqual(parry_frames_left(fighter, COMBAT), (6, recovery))
        fighter.parry_frame = 7  # First recovery frame.
        self.assertEqual(parry_frames_left(fighter, COMBAT), (0, recovery))
        fighter.parry_frame = 6 + recovery  # Last frame.
        self.assertEqual(parry_frames_left(fighter, COMBAT), (0, 1))
        fighter.parry_frame = 0
        self.assertEqual(parry_frames_left(fighter, COMBAT), (0, 0))

    def test_observation_shows_whether_parry_is_still_held(self) -> None:
        fighters = duel(gap=5.0)

        def held() -> float:
            structured = get_observation(
                fighters["agent"], {"opponent": fighters["opponent"]}, CONFIG
            )
            return encode_observation(structured, CONFIG)[FEATURE_INDEX["parry_held"]]

        self.assertEqual(held(), 0.0)
        step(fighters, agent=PARRY)
        self.assertEqual(held(), 1.0)
        step(fighters)
        self.assertEqual(held(), 0.0)

    def test_parry_actions_decode(self) -> None:
        self.assertTrue(to_fighter_action(Action.PARRY).parry)
        self.assertFalse(to_fighter_action(Action.ATTACK).parry)


if __name__ == "__main__":
    unittest.main()
