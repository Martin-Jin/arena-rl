import dataclasses
import random
import unittest

from arena_ai.config import AttackType, ScoringConfig, load_config
from arena_ai.envs.reward import calculate_reward
from arena_ai.envs.training_envs.one_v_one_env import OneVOneEnv
from arena_ai.sim.collision import bodies_overlap
from arena_ai.sim.outcome import is_over, winner
from arena_ai.sim.simulate import FighterAction, step_game
from arena_ai.sim.state import (
    AGENT_ID,
    FighterState,
    create_initial_fighters,
    enemies_of,
    is_opponent,
    opponent_id,
    opponent_ids,
    sort_key,
    team_of,
)

CONFIG = load_config()
GAME = CONFIG.game
COMBAT = GAME.combat
COUNTER = COMBAT.spec(AttackType.COUNTER)
ATTACK = FighterAction(attack=True)
NONE = FighterAction()


def step(fighters, **actions) -> None:
    step_game(fighters, {fid: actions.get(fid, NONE) for fid in fighters}, CONFIG)


def swing(fighters, attacker_id: str) -> None:
    for _ in range(COMBAT.attack_total_frames + 1):
        step(fighters, **{attacker_id: ATTACK})


class IdTests(unittest.TestCase):
    def test_ids_and_teams(self) -> None:
        self.assertEqual(opponent_ids(3), ["opponent", "opponent_1", "opponent_2"])
        self.assertEqual(opponent_id(0), "opponent")
        self.assertTrue(all(is_opponent(i) for i in opponent_ids(12)))
        self.assertFalse(is_opponent(AGENT_ID))
        self.assertEqual(team_of(AGENT_ID), 0)
        self.assertEqual({team_of(i) for i in opponent_ids(3)}, {1})

    def test_the_sort_key_is_natural(self) -> None:
        ids = [opponent_id(i) for i in (10, 2, 0, 1)] + [AGENT_ID]
        self.assertEqual(
            sorted(ids, key=sort_key),
            [AGENT_ID, "opponent", "opponent_1", "opponent_2", "opponent_10"],
        )

    def test_enemies_of_returns_the_other_team_in_order(self) -> None:
        fighters = create_initial_fighters(GAME, opponent_count=3)
        self.assertEqual(list(enemies_of(AGENT_ID, fighters)), opponent_ids(3))
        self.assertEqual(list(enemies_of("opponent_1", fighters)), [AGENT_ID])


class SpawnTests(unittest.TestCase):
    def test_one_opponent_keeps_the_fixed_spawn(self) -> None:
        fighters = create_initial_fighters(GAME)
        self.assertEqual(list(fighters), [AGENT_ID, "opponent"])
        self.assertEqual(fighters[AGENT_ID].x, GAME.world.arena_width_m * 0.25)
        self.assertEqual(fighters["opponent"].x, GAME.world.arena_width_m * 0.75)
        self.assertEqual(
            (fighters[AGENT_ID].facing_dir, fighters["opponent"].facing_dir), (1, -1)
        )

    def test_fixed_spawns_form_a_row_on_the_right_without_overlap(self) -> None:
        for count in (2, 3, 5):
            fighters = create_initial_fighters(GAME, opponent_count=count)
            xs = [fighters[i].x for i in opponent_ids(count)]
            self.assertEqual(xs, sorted(xs))
            self.assertGreater(min(xs), fighters[AGENT_ID].x)
            self.assertLess(max(xs), GAME.world.arena_width_m)
            for a, b in zip(xs, xs[1:]):
                self.assertGreaterEqual(b - a, GAME.fighter.width_m)
            self.assertTrue(all(fighters[i].facing_dir == -1 for i in opponent_ids(count)))

    def test_random_spawns_never_overlap_and_are_reproducible(self) -> None:
        for seed in range(200):
            fighters = create_initial_fighters(GAME, random.Random(seed), opponent_count=3)
            again = create_initial_fighters(GAME, random.Random(seed), opponent_count=3)
            self.assertEqual(fighters, again)
            ordered = list(fighters.values())
            for i, a in enumerate(ordered):
                for b in ordered[i + 1:]:
                    self.assertFalse(bodies_overlap(a, b, GAME), f"seed {seed}")

    def test_the_agent_faces_the_nearest_opponent_and_opponents_face_the_agent(self) -> None:
        fighters = create_initial_fighters(GAME, random.Random(3), opponent_count=3)
        agent = fighters[AGENT_ID]
        enemies = enemies_of(AGENT_ID, fighters).values()
        nearest = min(enemies, key=lambda f: abs(f.x - agent.x))
        self.assertEqual(agent.facing_dir, 1 if nearest.x >= agent.x else -1)
        for opponent in enemies:
            self.assertEqual(opponent.facing_dir, 1 if agent.x > opponent.x else -1)

    def test_one_opponent_random_spawn_keeps_the_rng_call_order(self) -> None:
        expected = random.Random(9)
        margin = GAME.fighter.width_m / 2
        agent_x = expected.uniform(margin, GAME.world.arena_width_m - margin)
        opponent_x = expected.uniform(margin, GAME.world.arena_width_m - margin)
        fighters = create_initial_fighters(GAME, random.Random(9))
        self.assertGreaterEqual(abs(agent_x - opponent_x), GAME.fighter.width_m)
        self.assertEqual((fighters[AGENT_ID].x, fighters["opponent"].x), (agent_x, opponent_x))


class TeamCombatTests(unittest.TestCase):
    def _line(self):
        return {
            AGENT_ID: FighterState(x=2.0, facing_dir=1),
            "opponent": FighterState(x=5.0, facing_dir=1),
            "opponent_1": FighterState(x=5.5, facing_dir=1),
        }

    def test_opponents_never_damage_each_other(self) -> None:
        fighters = self._line()
        swing(fighters, "opponent")
        self.assertEqual(fighters["opponent_1"].health, 100)
        self.assertEqual(fighters[AGENT_ID].health, 100)

    def test_a_swing_hits_every_enemy_it_reaches_once_each(self) -> None:
        fighters = {
            AGENT_ID: FighterState(x=5.0, facing_dir=1),
            "opponent": FighterState(x=5.7, facing_dir=-1),
            "opponent_1": FighterState(x=5.4, facing_dir=-1),
            "opponent_2": FighterState(x=8.0, facing_dir=-1),  # Out of reach.
        }
        swing(fighters, AGENT_ID)
        self.assertEqual(fighters["opponent"].health, 100 - COMBAT.attack_damage)
        self.assertEqual(fighters["opponent_1"].health, 100 - COMBAT.attack_damage)
        self.assertEqual(fighters["opponent_2"].health, 100)

    def test_a_parry_by_one_enemy_cancels_the_swing_for_the_rest(self) -> None:
        fighters = {
            AGENT_ID: FighterState(x=5.0, facing_dir=1),
            "opponent": FighterState(x=5.4, facing_dir=-1),
            "opponent_1": FighterState(x=5.7, facing_dir=-1),
        }
        for _ in range(COMBAT.attack_windup_frames - 1):
            step(fighters, **{AGENT_ID: ATTACK})
        step(fighters, **{AGENT_ID: ATTACK, "opponent_1": FighterAction(parry=True)})
        step(fighters, **{AGENT_ID: ATTACK})  # The first active step.
        self.assertTrue(fighters["opponent_1"].parried)
        self.assertEqual(fighters[AGENT_ID].attack_frame, 0)
        self.assertEqual(fighters["opponent"].health, 100 - COMBAT.attack_damage)

    def test_a_counter_cuts_through_opponents_in_a_row_and_hits_them_all(self) -> None:
        fighters = {
            AGENT_ID: FighterState(x=5.0, facing_dir=1),
            "opponent": FighterState(x=5.9, facing_dir=-1),
            "opponent_1": FighterState(x=6.4, facing_dir=-1),
        }
        fighters[AGENT_ID].counter_target = "opponent"
        fighters[AGENT_ID].counter_frames = COMBAT.counter_window_frames
        start = fighters[AGENT_ID].x
        passed = False
        for _ in range(COUNTER.attack_windup_frames + COMBAT.attack_active_frames + 1):
            step(fighters, **{AGENT_ID: ATTACK})
            passed = passed or fighters[AGENT_ID].x > fighters["opponent"].x
        self.assertTrue(passed)  # Through the first opponent...
        self.assertGreater(fighters[AGENT_ID].x, 6.0)
        self.assertGreater(fighters[AGENT_ID].x, start)
        for opponent in ("opponent", "opponent_1"):  # ...and both are hit.
            self.assertEqual(fighters[opponent].health, 100 - COUNTER.damage, opponent)
        self.assertEqual(fighters[AGENT_ID].health, 100)

    def test_a_counter_cannot_cut_through_an_opponent_beyond_its_lunge(self) -> None:
        fighters = {
            AGENT_ID: FighterState(x=2.0, facing_dir=1),
            "opponent": FighterState(x=2.9, facing_dir=-1),
            "opponent_1": FighterState(x=8.0, facing_dir=-1),
        }
        fighters[AGENT_ID].counter_target = "opponent"
        fighters[AGENT_ID].counter_frames = COMBAT.counter_window_frames
        for _ in range(COUNTER.attack_windup_frames + COMBAT.attack_active_frames + 1):
            step(fighters, **{AGENT_ID: ATTACK})
        self.assertLess(fighters[AGENT_ID].x, 6.0)  # Nowhere near the far opponent.
        self.assertEqual(fighters["opponent_1"].health, 100)

    def test_a_dead_fighter_cannot_attack_or_be_hit(self) -> None:
        fighters = {
            AGENT_ID: FighterState(x=5.0, facing_dir=1),
            "opponent": FighterState(x=5.4, facing_dir=-1, health=0),
        }
        swing(fighters, AGENT_ID)
        self.assertEqual(fighters["opponent"].health, 0)
        swing(fighters, "opponent")
        self.assertEqual(fighters[AGENT_ID].health, 100)
        self.assertEqual(fighters["opponent"].attack_frame, 0)

    def test_a_dead_fighter_does_not_block_movement(self) -> None:
        fighters = {
            AGENT_ID: FighterState(x=5.0, facing_dir=1),
            "opponent": FighterState(x=5.4, facing_dir=-1, health=0),
        }
        for _ in range(30):
            step(fighters, **{AGENT_ID: FighterAction(move_dir=1)})
        self.assertGreater(fighters[AGENT_ID].x, 5.4)

    def test_simultaneous_hits_both_land_whatever_the_order(self) -> None:
        for order in ((AGENT_ID, "opponent"), ("opponent", AGENT_ID)):
            fighters = {
                order[0]: FighterState(x=5.0, facing_dir=1, health=10),
                order[1]: FighterState(x=5.4, facing_dir=-1, health=10),
            }
            for _ in range(COMBAT.attack_windup_frames + 1):
                step(fighters, **{AGENT_ID: ATTACK, "opponent": ATTACK})
            self.assertEqual(fighters[AGENT_ID].health, 0)
            self.assertEqual(fighters["opponent"].health, 0)

    def test_a_counter_falls_back_to_the_nearest_enemy_if_the_parried_one_is_dead(self) -> None:
        fighters = self._line()
        fighters[AGENT_ID].counter_target = "opponent_1"
        fighters[AGENT_ID].counter_frames = COMBAT.counter_window_frames
        fighters["opponent_1"].health = 0
        step(fighters, **{AGENT_ID: ATTACK})
        self.assertEqual(fighters[AGENT_ID].lunge_dir, 1)


class CounterDamageTests(unittest.TestCase):
    def test_a_counter_deals_counter_damage_and_a_normal_swing_attack_damage(self) -> None:
        self.assertEqual(COUNTER.damage, 15)
        normal = {
            AGENT_ID: FighterState(x=5.0, facing_dir=1),
            "opponent": FighterState(x=5.4, facing_dir=-1),
        }
        swing(normal, AGENT_ID)
        self.assertEqual(normal["opponent"].health, 100 - COMBAT.attack_damage)
        counter = {
            AGENT_ID: FighterState(
                x=5.0, facing_dir=1, counter_frames=COMBAT.counter_window_frames
            ),
            "opponent": FighterState(x=5.4, facing_dir=-1),
        }
        swing(counter, AGENT_ID)
        self.assertEqual(counter["opponent"].health, 100 - COUNTER.damage)


class OutcomeTests(unittest.TestCase):
    def _fighters(self, agent: int, *opponents: int):
        fighters = {AGENT_ID: FighterState(health=agent)}
        for i, health in enumerate(opponents):
            fighters[opponent_id(i)] = FighterState(health=health)
        return fighters

    def test_the_agent_wins_only_when_every_opponent_is_dead(self) -> None:
        self.assertFalse(is_over(self._fighters(100, 0, 50)))
        self.assertIsNone(winner(self._fighters(100, 0, 50)))
        self.assertTrue(is_over(self._fighters(100, 0, 0)))
        self.assertEqual(winner(self._fighters(100, 0, 0)), AGENT_ID)

    def test_the_opponents_win_when_the_agent_dies_with_one_left(self) -> None:
        fighters = self._fighters(0, 0, 40)
        self.assertTrue(is_over(fighters))
        self.assertEqual(winner(fighters), "opponent")

    def test_the_last_opponent_and_the_agent_dying_together_is_a_draw(self) -> None:
        fighters = self._fighters(0, 0, 0)
        self.assertTrue(is_over(fighters))
        self.assertIsNone(winner(fighters))

    def test_a_time_out_compares_health_with_the_mean_opponent_health(self) -> None:
        self.assertEqual(winner(self._fighters(60, 100, 0, 0), timed_out=True), AGENT_ID)
        self.assertEqual(winner(self._fighters(30, 100, 0, 0), timed_out=True), "opponent")
        self.assertIsNone(winner(self._fighters(50, 100, 0), timed_out=True))
        self.assertIsNone(winner(self._fighters(100, 50)))

    def test_solo_practice_never_ends_by_victory(self) -> None:
        self.assertFalse(is_over({AGENT_ID: FighterState()}))

    def test_victory_reward_needs_every_opponent_dead(self) -> None:
        previous = {AGENT_ID: 100, "opponent": 10, "opponent_1": 10}
        some = self._fighters(100, 0, 10)
        done = self._fighters(100, 0, 0)
        gap = calculate_reward(previous, done, CONFIG) - calculate_reward(previous, some, CONFIG)
        scoring = CONFIG.scoring
        # The step that finishes the last opponent also kills it and pays the win bonuses
        # (this step ends the fight with time_left 0).
        expected = (
            scoring.victory_reward
            + scoring.damage_reward_scale * 10
            + scoring.kill_reward
            + scoring.health_reward
        )
        self.assertAlmostEqual(gap, expected)


class TimeoutRewardTests(unittest.TestCase):
    def test_a_time_out_win_pays_less_than_a_kill_and_is_reported(self) -> None:
        env = OneVOneEnv(CONFIG, max_episode_steps=2)
        env.reset(seed=1)
        env.fighters["opponent"].health = 50
        env.step(0)
        _, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(truncated and not terminated)
        self.assertTrue(info["timeout_win"])
        self.assertIsNone(info["winner"])
        self.assertGreater(reward, 0.0)
        self.assertLess(CONFIG.scoring.timeout_win_reward, CONFIG.scoring.victory_reward)


class WinBonusTests(unittest.TestCase):
    """The health and time bonuses are paid on a win only."""

    def _reward(self, agent_health: int, opponent_health: int, time_left: float) -> float:
        scoring = dataclasses.replace(
            CONFIG.scoring, health_reward=1.0, victory_time_bonus=2.0, kill_reward=0.0
        )
        config = dataclasses.replace(CONFIG, scoring=scoring)
        fighters = {
            AGENT_ID: FighterState(health=agent_health),
            "opponent": FighterState(health=opponent_health),
        }
        previous = {AGENT_ID: agent_health, "opponent": opponent_health}
        return calculate_reward(previous, fighters, config, time_left)

    def test_a_win_pays_more_the_more_health_the_agent_has(self) -> None:
        full, half = self._reward(100, 0, 0.0), self._reward(50, 0, 0.0)
        self.assertAlmostEqual(full - half, 0.5)  # health_reward 1.0 times 50 / 100.

    def test_a_win_pays_more_the_sooner_it_comes(self) -> None:
        early, late = self._reward(100, 0, 0.75), self._reward(100, 0, 0.25)
        self.assertAlmostEqual(early - late, 1.0)  # victory_time_bonus 2.0 times 0.5.

    def test_a_loss_never_pays_more_for_losing_sooner_or_with_health(self) -> None:
        # Regression: a per-step time cost made a quick loss cheaper than a long one. An agent
        # that is dead has no health to be paid for, and time_left must not matter.
        self.assertEqual(self._reward(0, 100, 0.9), self._reward(0, 100, 0.0))

    def test_a_fight_that_is_not_won_pays_no_bonus(self) -> None:
        self.assertEqual(self._reward(100, 50, 0.9), self._reward(100, 50, 0.0))
        self.assertEqual(self._reward(40, 50, 0.9), self._reward(40, 50, 0.0))

    def test_the_environment_pays_the_time_bonus_by_the_share_of_the_limit_left(self) -> None:
        def win_reward(limit: int, steps_before_win: int) -> float:
            env = OneVOneEnv(CONFIG, max_episode_steps=limit)
            env.reset(seed=1)
            for _ in range(steps_before_win):
                env.step(0)
            env.fighters["opponent"].health = 0
            _, reward, terminated, _, info = env.step(0)
            self.assertTrue(terminated)
            self.assertEqual(info["winner"], "agent")
            return reward

        scoring = CONFIG.scoring
        early, late = win_reward(100, 9), win_reward(100, 59)
        self.assertAlmostEqual(early - late, scoring.victory_time_bonus * 0.5, places=6)


class KillRewardTests(unittest.TestCase):
    def _reward(self, kill_reward: float, healths: tuple[int, int]) -> float:
        scoring = dataclasses.replace(CONFIG.scoring, kill_reward=kill_reward)
        config = dataclasses.replace(CONFIG, scoring=scoring)
        fighters = {
            AGENT_ID: FighterState(),
            "opponent": FighterState(health=healths[0]),
            "opponent_1": FighterState(health=healths[1]),
        }
        previous = {AGENT_ID: 100, "opponent": 10, "opponent_1": 100}
        return calculate_reward(previous, fighters, config)

    def test_it_is_off_unless_configured(self) -> None:
        self.assertEqual(ScoringConfig.kill_reward, 0.0)  # The dataclass default.

    def test_each_opponent_finished_off_pays_once_however_the_damage_is_spread(self) -> None:
        base = self._reward(0.0, (0, 100))
        self.assertAlmostEqual(self._reward(0.5, (0, 100)) - base, 0.5)  # One kill.
        # The same 10 damage that does not kill (10 damage to the other) pays no bonus.
        self.assertAlmostEqual(self._reward(0.5, (10, 90)), self._reward(0.0, (10, 90)))

    def test_a_fighter_that_was_already_dead_pays_nothing_more(self) -> None:
        scoring = dataclasses.replace(CONFIG.scoring, kill_reward=0.5)
        config = dataclasses.replace(CONFIG, scoring=scoring)
        fighters = {AGENT_ID: FighterState(), "opponent": FighterState(health=0)}
        previous = {AGENT_ID: 100, "opponent": 0}
        plain = calculate_reward(previous, fighters, CONFIG)
        self.assertAlmostEqual(calculate_reward(previous, fighters, config), plain)
