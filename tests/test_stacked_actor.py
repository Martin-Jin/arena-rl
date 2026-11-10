import unittest

import torch

from arena_ai.agents.policy import ActorCritic, PolicyAdapter
from arena_ai.agents.stacked_actor import StackedActor
from arena_ai.config import load_config
from arena_ai.sim.observation_vector import FEATURES_PER_FIGHTER

CONFIG = load_config()
OPPONENTS = CONFIG.game.observation.max_opponents
OBSERVATION_SIZE = FEATURES_PER_FIGHTER * (OPPONENTS + 1) + OPPONENTS


def distinct_model(seed: int) -> ActorCritic:
    """A model whose two networks differ from each other, so a mix-up would show."""
    torch.manual_seed(seed)
    model = ActorCritic(CONFIG)
    for parameter in model.parameters():
        parameter.data += 0.05 * torch.randn_like(parameter)
    return model.eval()


def observations(count: int = 16) -> torch.Tensor:
    torch.manual_seed(0)
    batch = torch.rand(count, OBSERVATION_SIZE)
    batch[:, -OPPONENTS:] = (torch.rand(count, OPPONENTS) > 0.5).float()  # Some slots masked.
    return batch


class StackedActorTests(unittest.TestCase):
    def test_it_matches_the_two_networks(self) -> None:
        model = distinct_model(1)
        logits, values = StackedActor(model, torch.device("cpu"))(observations())
        with torch.no_grad():
            torch.testing.assert_close(
                logits, model.policy_network(observations()), atol=1e-5, rtol=0
            )
            torch.testing.assert_close(
                values, model.value_network(observations()).squeeze(-1), atol=1e-5, rtol=0
            )

    def test_a_single_observation_and_an_all_masked_one_match(self) -> None:
        model = distinct_model(2)
        actor = StackedActor(model, torch.device("cpu"))
        batch = observations(3)
        batch[1, -OPPONENTS:] = 0.0  # No living opponent: only the fighter itself is attended.
        for row in (batch[:1], batch[1:2], batch):
            logits, values = actor(row)
            torch.testing.assert_close(logits, model.policy_network(row), atol=1e-5, rtol=0)
            torch.testing.assert_close(
                values, model.value_network(row).squeeze(-1), atol=1e-5, rtol=0
            )

    def test_load_picks_up_new_weights(self) -> None:
        old, new = distinct_model(3), distinct_model(4)
        actor = StackedActor(old, torch.device("cpu"))
        actor.load(new)
        logits, values = actor(observations())
        torch.testing.assert_close(logits, new.policy_network(observations()), atol=1e-5, rtol=0)
        torch.testing.assert_close(
            values, new.value_network(observations()).squeeze(-1), atol=1e-5, rtol=0
        )


class PolicyAdapterTests(unittest.TestCase):
    def test_sync_makes_the_acting_copy_follow_the_trained_model(self) -> None:
        model = distinct_model(5)
        adapter = PolicyAdapter(model, torch.device("cpu"))
        observation = observations(4).tolist()
        before = adapter.value(observation)
        for parameter in model.parameters():
            parameter.data += 0.1
        self.assertEqual(adapter.value(observation), before)  # Not synced yet.
        adapter.sync()
        with torch.no_grad():
            expected = model.value_network(torch.tensor(observation)).squeeze(-1).tolist()
        torch.testing.assert_close(torch.tensor(adapter.value(observation)), torch.tensor(expected))

    def test_act_and_value_equals_act_then_value(self) -> None:
        adapter = PolicyAdapter(distinct_model(6), torch.device("cpu"))
        observation = observations(8).tolist()
        torch.manual_seed(7)
        actions, log_probs = adapter.act(observation)
        values = adapter.value(observation)
        torch.manual_seed(7)
        combined = adapter.act_and_value(observation)
        self.assertEqual(combined[0], actions)
        self.assertEqual(combined[1], log_probs)
        self.assertEqual(combined[2], values)

    def test_a_single_observation_still_returns_python_scalars(self) -> None:
        adapter = PolicyAdapter(distinct_model(8), torch.device("cpu"))
        observation = observations(1)[0].tolist()
        action, log_prob = adapter.act(observation)
        self.assertIsInstance(action, int)
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(adapter.value(observation), float)


if __name__ == "__main__":
    unittest.main()
