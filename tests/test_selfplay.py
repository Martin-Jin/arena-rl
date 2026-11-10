from functools import partial
import random
import unittest

import torch

from arena_ai.agents import selfplay
from arena_ai.agents.action import ACTION_COUNT, LOGIT_COUNT
from arena_ai.agents.distribution import sample_with_log_prob
from arena_ai.agents.policy import ActorCritic
from arena_ai.agents.scripted import (
    SELF_PLAY,
    OpponentMixture,
    random_opponent_pool,
    training_opponents,
    uses_self_play,
)
from arena_ai.agents.selfplay import (
    SelfPlayPolicy,
    load_weights,
    make_opponent_policy,
    own_observation,
    weights_of,
)
from arena_ai.config import load_config
from arena_ai.envs.async_vector_env import AsyncVectorEnv
from arena_ai.envs.training_envs.one_v_many_env import OneVManyEnv
from arena_ai.envs.vector_env import VectorEnv
from arena_ai.sim.observation_vector import FEATURES_PER_FIGHTER, fighter_features, opponent_mask
from arena_ai.sim.state import AGENT_ID, create_initial_fighters

CONFIG = load_config()
SLOTS = CONFIG.game.observation.max_opponents
SHORT_EPISODE = 40


def _reset_shared_network() -> None:
    selfplay._model = None
    selfplay._actor = None


def _weights(seed: int):
    torch.manual_seed(seed)
    return weights_of(ActorCritic(CONFIG))


class SelfPlayNamesTests(unittest.TestCase):
    def test_agent_is_accepted_as_a_training_opponent(self) -> None:
        self.assertEqual(SELF_PLAY, "agent")
        self.assertEqual(
            training_opponents([["agent", "approach"], ["agent", None], None], 4, SLOTS),
            [["agent", "approach"], ["agent", None], None],
        )

    def test_a_drawn_slot_is_never_the_agent(self) -> None:
        # The exception to the random draw: the pool refuses it, and an empty pool leaves it out.
        with self.assertRaisesRegex(ValueError, "cannot be in"):
            random_opponent_pool(["agent", "idle"])
        self.assertNotIn("agent", random_opponent_pool([]))
        mixture = OpponentMixture((None, None), tuple(random_opponent_pool([])), SLOTS)
        rng = random.Random(0)
        self.assertTrue(all("agent" not in mixture.draw(rng) for _ in range(200)))
        everything = OpponentMixture(None, tuple(random_opponent_pool([])), SLOTS)
        self.assertTrue(all("agent" not in everything.draw(rng) for _ in range(200)))

    def test_an_unknown_name_is_still_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "agent"):
            training_opponents([["nonsense"]], 4, SLOTS)
        with self.assertRaisesRegex(ValueError, "Unknown opponents"):
            random_opponent_pool(["nonsense"])

    def test_it_knows_when_self_play_is_in_use(self) -> None:
        self.assertTrue(uses_self_play([["approach"], "agent"]))
        self.assertTrue(uses_self_play([["approach", "agent"]]))
        self.assertTrue(uses_self_play([["agent", None]]))
        self.assertFalse(uses_self_play([["approach"], None]))

    def test_the_policy_factory_makes_a_self_play_policy_only_for_agent(self) -> None:
        self.assertIsInstance(make_opponent_policy("agent", CONFIG), SelfPlayPolicy)
        self.assertEqual(make_opponent_policy("approach", CONFIG).name, "approach")


class OwnObservationTests(unittest.TestCase):
    def test_an_opponent_sees_itself_first_and_the_agent_as_its_only_enemy(self) -> None:
        fighters = create_initial_fighters(CONFIG.game, None, 2)
        view = own_observation(fighters, "opponent_1", CONFIG)
        width = CONFIG.game.world.arena_width_m
        self.assertAlmostEqual(fighter_features(view, 0)["x"], fighters["opponent_1"].x / width)
        self.assertEqual(fighter_features(view, 0)["rel_x_ahead"], 0.0)  # Its own row.
        self.assertAlmostEqual(fighter_features(view, 1)["x"], fighters[AGENT_ID].x / width)
        self.assertEqual(opponent_mask(view, SLOTS), [1.0] + [0.0] * (SLOTS - 1))

    def test_the_agents_own_view_is_the_learners_view(self) -> None:
        env = OneVManyEnv(CONFIG, [("idle", make_opponent_policy("idle", CONFIG))])
        observation, _ = env.reset(seed=1)
        self.assertEqual(own_observation(env.fighters, AGENT_ID, CONFIG), observation)


class SelfPlayPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_shared_network()
        self.addCleanup(_reset_shared_network)

    def test_the_loaded_network_gives_the_logits_of_the_source_model(self) -> None:
        torch.manual_seed(0)
        model = ActorCritic(CONFIG)
        load_weights(weights_of(model), CONFIG)
        observations = torch.rand(5, FEATURES_PER_FIGHTER * (SLOTS + 1) + SLOTS)
        observations[:, -SLOTS:] = (observations[:, -SLOTS:] > 0.5).float()
        with torch.no_grad():
            expected = model.policy_network(observations)
            logits, _ = selfplay._actor(observations)
        self.assertTrue(torch.allclose(logits, expected, atol=1e-5))

    def test_loading_again_replaces_the_weights(self) -> None:
        observation = torch.rand(1, FEATURES_PER_FIGHTER * (SLOTS + 1) + SLOTS)
        load_weights(_weights(1), CONFIG)
        first = selfplay._actor(observation)[0]
        load_weights(_weights(2), CONFIG)
        self.assertFalse(torch.allclose(first, selfplay._actor(observation)[0]))

    def test_it_cannot_act_before_it_has_weights(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "weights"):
            SelfPlayPolicy(CONFIG)([0.0] * 10)

    def test_it_returns_a_legal_action_and_repeats_with_the_same_seed(self) -> None:
        load_weights(_weights(3), CONFIG)
        fighters = create_initial_fighters(CONFIG.game, None, 1)
        view = own_observation(fighters, "opponent", CONFIG)

        def play(seed: int) -> list[int]:
            policy = SelfPlayPolicy(CONFIG)
            policy.resample(random.Random(seed))
            return [policy(view) for _ in range(20)]

        first = play(7)
        self.assertTrue(all(0 <= action < ACTION_COUNT for action in first))
        self.assertEqual(first, play(7))
        self.assertNotEqual(first, play(8))

    def test_a_generator_makes_sampling_repeatable(self) -> None:
        logits = torch.zeros(4, LOGIT_COUNT)
        first = sample_with_log_prob(logits, torch.Generator().manual_seed(1))
        second = sample_with_log_prob(logits, torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(first[0], second[0]))


def make_env(names: tuple[str, ...]) -> OneVManyEnv:
    opponents = [(name, make_opponent_policy(name, CONFIG)) for name in names]
    return OneVManyEnv(CONFIG, opponents, SHORT_EPISODE)


def make_mixture_env() -> OneVManyEnv:
    return OneVManyEnv(
        CONFIG, [], SHORT_EPISODE, mixture=OpponentMixture((None, "agent"), ("idle",), SLOTS)
    )


def factories() -> list:
    # Module-level partials, not lambdas: the workers receive them by pickling.
    return [
        partial(make_env, ("agent",)),
        partial(make_env, ("agent", "approach")),
        partial(make_env, ("idle",)),
        partial(make_mixture_env),
    ]


class SelfPlayEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_shared_network()
        self.addCleanup(_reset_shared_network)

    def test_an_environment_with_an_agent_opponent_runs_episodes(self) -> None:
        env = make_env(("agent",))
        env.set_opponent_weights(_weights(4))
        env.reset(seed=1)
        ended = 0
        for _ in range(3 * SHORT_EPISODE):
            _, _, terminated, truncated, info = env.step(0)
            self.assertEqual(info["opponent"], "agent")
            if terminated or truncated:
                ended += 1
                env.reset()
        self.assertGreaterEqual(ended, 3)

    def test_stepping_without_weights_raises(self) -> None:
        env = make_env(("agent",))
        env.reset(seed=1)
        with self.assertRaisesRegex(RuntimeError, "weights"):
            env.step(0)

    def test_an_opponent_that_plays_the_network_acts_unlike_idle(self) -> None:
        # The opponent really moves: a policy that always returns 0 would stand still.
        env = make_env(("agent",))
        env.set_opponent_weights(_weights(5))
        env.reset(seed=2)
        start = env.fighters["opponent"].x
        positions = set()
        for _ in range(30):
            env.step(0)
            positions.add(round(env.fighters["opponent"].x, 3))
        self.assertGreater(len(positions), 1)
        self.assertNotEqual(start, env.fighters["opponent"].x)

    def test_workers_play_the_same_game_as_the_synchronous_environment(self) -> None:
        weights = _weights(6)
        sync = VectorEnv(factories())
        sync.set_opponent_weights(weights)
        parallel = AsyncVectorEnv(factories(), num_workers=2)
        self.addCleanup(parallel.close)
        parallel.set_opponent_weights(weights)
        self.assertEqual(sync.reset(seed=3), parallel.reset(seed=3))
        rng = random.Random(0)
        ended = 0
        for _ in range(120):
            actions = [rng.randrange(ACTION_COUNT) for _ in range(4)]
            expected = sync.step(actions)
            self.assertEqual(expected, parallel.step(actions))
            ended += sum(expected[2]) + sum(expected[3])
        self.assertGreater(ended, 4)  # Auto-resets with a redrawn mixture were covered.


if __name__ == "__main__":
    unittest.main()
