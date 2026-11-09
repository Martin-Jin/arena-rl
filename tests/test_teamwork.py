import unittest
from functools import partial

from arena_ai.agents.action import Action
from arena_ai.agents.scripted import approach_and_attack, parrying_hunter
from arena_ai.agents.scripted.teamwork import jumps_over_allies
from arena_ai.agents.scripted.view import living_allies, team_observation
from arena_ai.config import load_config
from arena_ai.envs.training_envs.one_v_many_env import OneVManyEnv
from arena_ai.sim.observation_vector import fighter_rows
from arena_ai.sim.state import AGENT_ID, FighterState

CONFIG = load_config()
WIDTH = CONFIG.game.fighter.width_m


def walk_right(observation, config=None) -> int:
    return int(Action.MOVE_RIGHT)


def team(agent_x, me_x, *allies_x):
    fighters = {AGENT_ID: FighterState(x=agent_x), "opponent": FighterState(x=me_x)}
    for index, x in enumerate(allies_x, start=1):
        fighters[f"opponent_{index}"] = FighterState(x=x)
    return fighters


def observation(fighters, fighter_id="opponent"):
    return team_observation(fighters, fighter_id, CONFIG)


class TeamObservationTests(unittest.TestCase):
    def test_views_built_from_shared_rows_match_views_built_alone(self) -> None:
        fighters = team(2.0, 9.0, 8.0, 7.0)
        fighters["opponent_1"].health = 0  # A dead ally is left out of every view.
        rows = fighter_rows(fighters, CONFIG)
        self.assertNotIn("opponent_1", rows)
        for fighter_id in ("opponent", "opponent_2"):
            self.assertEqual(
                team_observation(fighters, fighter_id, CONFIG, rows),
                team_observation(fighters, fighter_id, CONFIG),
            )
        self.assertEqual(len(rows), 3)

    def test_the_opponent_is_slot_one_and_the_allies_follow(self) -> None:
        fighters = team(2.0, 9.0, 8.0, 7.0)
        view = observation(fighters, "opponent_1")
        from arena_ai.sim.observation_vector import fighter_features

        width = CONFIG.game.world.arena_width_m
        self.assertAlmostEqual(fighter_features(view, 1)["x"] * width, 8.0)
        allies = sorted(a["x"] * width for a in living_allies(view))
        self.assertAlmostEqual(allies[0], 7.0)
        self.assertAlmostEqual(allies[1], 9.0)

    def test_a_one_on_one_has_no_allies_and_a_dead_ally_is_not_one(self) -> None:
        self.assertEqual(living_allies(observation(team(2.0, 9.0))), [])
        fighters = team(2.0, 9.0, 8.0)
        fighters["opponent_1"].health = 0
        self.assertEqual(living_allies(observation(fighters)), [])


class JumpOverAlliesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = jumps_over_allies(walk_right)

    def action(self, fighters, fighter_id="opponent") -> int:
        return self.policy(observation(fighters, fighter_id), CONFIG)

    def test_it_jumps_when_an_ally_blocks_the_way_to_the_learner(self) -> None:
        fighters = team(9.0, 5.0, 5.0 + WIDTH)
        self.assertEqual(self.action(fighters), Action.MOVE_RIGHT_JUMP)

    def test_it_walks_when_the_way_is_clear(self) -> None:
        self.assertEqual(self.action(team(9.0, 5.0)), Action.MOVE_RIGHT)  # No ally.
        self.assertEqual(self.action(team(9.0, 5.0, 6.5)), Action.MOVE_RIGHT)  # Far off.
        self.assertEqual(self.action(team(9.0, 5.0, 5.0 - WIDTH)), Action.MOVE_RIGHT)  # Behind.

    def test_it_does_not_jump_in_the_air_or_over_a_dead_ally(self) -> None:
        fighters = team(9.0, 5.0, 5.0 + WIDTH)
        fighters["opponent"].on_ground = False
        self.assertEqual(self.action(fighters), Action.MOVE_RIGHT)
        fighters = team(9.0, 5.0, 5.0 + WIDTH)
        fighters["opponent_1"].health = 0
        self.assertEqual(self.action(fighters), Action.MOVE_RIGHT)

    def test_it_does_not_jump_over_an_ally_that_is_in_the_air_above_it(self) -> None:
        fighters = team(9.0, 5.0, 5.0 + WIDTH)
        fighters["opponent_1"].y = CONFIG.game.fighter.height_m + 0.1
        self.assertEqual(self.action(fighters), Action.MOVE_RIGHT)

    def test_it_leaves_other_actions_alone(self) -> None:
        attack = jumps_over_allies(lambda obs, config=None: int(Action.ATTACK))
        fighters = team(9.0, 5.0, 5.0 + WIDTH)
        self.assertEqual(attack(observation(fighters), CONFIG), Action.ATTACK)

    def test_the_real_opponents_use_it(self) -> None:
        self.assertTrue(hasattr(approach_and_attack, "__wrapped__"))
        self.assertTrue(hasattr(parrying_hunter, "__wrapped__"))


class GangUpTests(unittest.TestCase):
    def test_opponents_queued_behind_an_ally_hop_over_it_and_all_get_to_swing(self) -> None:
        policy = partial(approach_and_attack, config=CONFIG)
        env = OneVManyEnv(CONFIG, [("approach", policy)] * 3, max_episode_steps=600)
        env.reset(seed=1)
        agent = env.fighters[AGENT_ID]
        opponents = {k: f for k, f in env.fighters.items() if k != AGENT_ID}
        start_order = sorted(opponents, key=lambda k: opponents[k].x)
        hopped, swung = set(), set()
        for _ in range(600):
            env.step(int(Action.IDLE))
            agent.health = 10_000  # Stand and take it, so the fight lasts.
            for fighter_id, fighter in opponents.items():
                if fighter.y > CONFIG.game.fighter.height_m:
                    hopped.add(fighter_id)  # Clear of every body: over an ally.
                if fighter.attack_frame:
                    swung.add(fighter_id)
        end_order = sorted(opponents, key=lambda k: opponents[k].x)
        self.assertTrue(hopped)
        self.assertNotEqual(start_order, end_order)  # Somebody got past the front one.
        self.assertEqual(swung, set(opponents))
