import dataclasses
import random
import unittest

from arena_ai.agents.action import Action
from arena_ai.agents.scripted import EpisodePolicy, approach_and_attack, parrying_hunter
from arena_ai.config import load_config
from arena_ai.envs.training_envs.one_v_many_env import OneVManyEnv
from arena_ai.sim.observation import get_observation
from arena_ai.sim.observation_vector import encode_observation
from arena_ai.sim.state import FighterState

BASE = load_config()
EDGE = BASE.game.fighter.width_m + BASE.game.combat.attack_range_m
SWINGS = (Action.ATTACK, Action.MOVE_LEFT_ATTACK, Action.MOVE_RIGHT_ATTACK)


class FixedRandom:
    """A stand-in generator whose ``random()`` always returns the same number."""

    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def config_with(**training):
    return dataclasses.replace(BASE, training=dataclasses.replace(BASE.training, **training))


def observe(gap: float) -> list[float]:
    """The opponent stands ``gap`` meters to the right of the learner, facing it."""
    learner = FighterState(x=5.0, facing_dir=1)
    opponent = FighterState(x=5.0 + gap, facing_dir=-1)
    return encode_observation(get_observation(learner, {"opponent": opponent}, BASE), BASE)


class SwingStartTests(unittest.TestCase):
    def test_it_moves_while_it_starts_the_swing(self) -> None:
        action = approach_and_attack(observe(EDGE - 0.01), BASE)
        self.assertEqual(action, Action.MOVE_LEFT_ATTACK)

    def test_a_positive_offset_starts_the_swing_out_of_reach(self) -> None:
        self.assertEqual(approach_and_attack(observe(EDGE + 0.2), BASE), Action.MOVE_LEFT)
        action = approach_and_attack(observe(EDGE + 0.2), BASE, swing_start_offset_m=0.3)
        self.assertIn(action, SWINGS)

    def test_a_negative_offset_walks_closer_before_the_swing(self) -> None:
        action = approach_and_attack(observe(EDGE - 0.1), BASE, swing_start_offset_m=-0.2)
        self.assertEqual(action, Action.MOVE_LEFT)

    def test_the_hunter_passes_the_offset_on(self) -> None:
        action = parrying_hunter(
            observe(EDGE + 0.2), BASE, swing_start_offset_m=0.3, rng=FixedRandom(0.0)
        )
        self.assertIn(action, SWINGS)

    def test_the_hunter_uses_the_given_attack_chance_and_generator(self) -> None:
        rng = FixedRandom(0.5)
        low = parrying_hunter(observe(EDGE - 0.01), BASE, attack_chance=0.3, rng=rng)
        high = parrying_hunter(observe(EDGE - 0.01), BASE, attack_chance=0.6, rng=rng)
        self.assertEqual(low, Action.IDLE)
        self.assertIn(high, SWINGS)


class EpisodePolicyTests(unittest.TestCase):
    def test_it_draws_options_inside_the_configured_ranges(self) -> None:
        config = config_with(
            randomize_opponents=True,
            swing_start_offset_m=(-0.1, 0.1),
            hunter_attack_chance=(0.3, 0.4),
        )
        policy = EpisodePolicy("parrier", config)
        rng = random.Random(1)
        seen = set()
        for _ in range(50):
            policy.resample(rng)
            self.assertTrue(-0.1 <= policy.options["swing_start_offset_m"] <= 0.1)
            self.assertTrue(0.3 <= policy.options["attack_chance"] <= 0.4)
            self.assertIs(policy.options["rng"], rng)
            seen.add(policy.options["swing_start_offset_m"])
        self.assertGreater(len(seen), 40)

    def test_the_same_seed_draws_the_same_options(self) -> None:
        config = config_with(randomize_opponents=True)
        first, second = EpisodePolicy("approach", config), EpisodePolicy("approach", config)
        first.resample(random.Random(7))
        second.resample(random.Random(7))
        self.assertEqual(first.options, second.options)

    def test_off_it_keeps_the_defaults_but_still_seeds_the_dice(self) -> None:
        config = config_with(randomize_opponents=False)
        rng = random.Random(1)
        approach, parrier = EpisodePolicy("approach", config), EpisodePolicy("parrier", config)
        approach.resample(rng)
        parrier.resample(rng)
        self.assertEqual(approach.options, {})
        self.assertEqual(parrier.options, {"rng": rng})

    def test_opponents_without_options_are_unchanged(self) -> None:
        policy = EpisodePolicy("idle", config_with(randomize_opponents=True))
        policy.resample(random.Random(1))
        self.assertEqual(policy.options, {})
        self.assertEqual(policy(observe(EDGE)), Action.IDLE)

    def test_random_opponent_is_reproducible_with_the_same_seed(self) -> None:
        def actions(seed: int) -> list[int]:
            policy = EpisodePolicy("random", BASE)
            policy.resample(random.Random(seed))
            return [policy(observe(EDGE)) for _ in range(30)]

        self.assertEqual(actions(3), actions(3))
        self.assertNotEqual(actions(3), actions(4))

    def test_the_environment_redraws_the_options_every_episode(self) -> None:
        config = config_with(randomize_opponents=True)
        policy = EpisodePolicy("approach", config)
        env = OneVManyEnv(config, [("approach", policy)])
        env.reset(seed=5)
        first = policy.options["swing_start_offset_m"]
        env.reset()
        self.assertNotEqual(policy.options["swing_start_offset_m"], first)
        env.reset(seed=5)
        self.assertEqual(policy.options["swing_start_offset_m"], first)


if __name__ == "__main__":
    unittest.main()
