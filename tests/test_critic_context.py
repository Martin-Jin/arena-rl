from pathlib import Path
import tempfile
import unittest

import torch

from arena_ai.agents.policy import ActorCritic
from arena_ai.agents.scripted import CONTEXT_SIZE, OPPONENT_TYPES, opponent_context
from arena_ai.agents.selfplay import make_opponent_policy
from arena_ai.agents.stacked_actor import StackedActor
from arena_ai.config import load_config
from arena_ai.envs.training_envs.one_v_many_env import OneVManyEnv
from arena_ai.sim.observation_vector import FEATURES_PER_FIGHTER, observation_size
from arena_ai.training.checkpoints import load_checkpoint, save_checkpoint
from arena_ai.training.train import TrainingState

CONFIG = load_config()
SLOTS = CONFIG.game.observation.max_opponents
SIZE = observation_size(SLOTS)


def batch(count: int = 4, context: bool = True) -> torch.Tensor:
    torch.manual_seed(3)
    observations = torch.rand(count, SIZE)
    observations[:, -SLOTS:] = (observations[:, -SLOTS:] > 0.5).float()
    if not context:
        return observations
    return torch.cat([observations, torch.rand(count, CONTEXT_SIZE)], dim=1)


class OpponentContextTests(unittest.TestCase):
    def test_it_counts_each_type_then_the_number_of_opponents_and_its_square(self) -> None:
        context = opponent_context(["parrier", "parrier", "agent"])
        self.assertEqual(len(context), CONTEXT_SIZE)
        self.assertEqual(context[OPPONENT_TYPES.index("parrier")], 2.0)
        self.assertEqual(context[OPPONENT_TYPES.index("agent")], 1.0)
        self.assertEqual(context[OPPONENT_TYPES.index("idle")], 0.0)
        self.assertEqual(context[-2:], [3.0, 9.0])

    def test_the_environment_appends_the_context_to_its_observation(self) -> None:
        opponents = [(name, make_opponent_policy(name, CONFIG)) for name in ("idle", "approach")]
        env = OneVManyEnv(CONFIG, opponents)
        observation, _ = env.reset(seed=1)
        self.assertEqual(len(observation), SIZE + CONTEXT_SIZE)
        self.assertEqual(observation[SIZE:], opponent_context(["idle", "approach"]))


class CriticOffsetTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = ActorCritic(CONFIG).eval()

    def test_it_starts_at_zero_so_the_values_are_those_of_the_value_network(self) -> None:
        observations = batch()
        with torch.no_grad():
            _, values = self.model(observations)
            expected = self.model.value_network(observations[:, :SIZE])
        self.assertTrue(torch.equal(values, expected))

    def test_the_context_shifts_the_value_and_never_the_policy(self) -> None:
        torch.nn.init.normal_(self.model.matchup.weight)
        observations = batch()
        with torch.no_grad():
            logits, values = self.model(observations)
            plain_logits, plain_values = self.model(observations[:, :SIZE])
        self.assertTrue(torch.equal(logits, plain_logits))  # The policy never sees it.
        self.assertFalse(torch.allclose(values, plain_values))
        offset = self.model.matchup(observations[:, SIZE:])
        torch.testing.assert_close(values - plain_values, offset)

    def test_the_stacked_actor_matches_the_model_with_the_context(self) -> None:
        torch.nn.init.normal_(self.model.matchup.weight)
        torch.nn.init.normal_(self.model.matchup.bias)
        observations = batch()
        actor = StackedActor(self.model, torch.device("cpu"))
        with torch.no_grad():
            logits, values = actor(observations)
            expected_logits, expected_values = self.model(observations)
        torch.testing.assert_close(logits, expected_logits, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(values, expected_values.squeeze(-1), atol=1e-5, rtol=1e-4)

    def test_without_the_context_nothing_changes(self) -> None:
        # The viewer and the tests feed the observation alone.
        torch.nn.init.normal_(self.model.matchup.weight)
        observations = batch(context=False)
        actor = StackedActor(self.model, torch.device("cpu"))
        with torch.no_grad():
            _, values = actor(observations)
            _, expected = self.model(observations)
        torch.testing.assert_close(values, expected.squeeze(-1), atol=1e-5, rtol=1e-4)


class OldCheckpointTests(unittest.TestCase):
    def test_a_checkpoint_from_before_the_critic_offset_still_loads_and_resumes(self) -> None:
        torch.manual_seed(0)
        old = ActorCritic(CONFIG)
        old_parameters = [p for name, p in old.named_parameters() if not name.startswith("matchup")]
        old_optimizer = torch.optim.Adam(old_parameters, lr=1e-3)
        loss = sum(p.sum() for p in old_parameters)  # Give Adam some state to restore.
        loss.backward()
        old_optimizer.step()
        saved = {
            "signature": old.signature,
            "model": {k: v for k, v in old.state_dict().items() if not k.startswith("matchup")},
            "optimizer": old_optimizer.state_dict(),
            "config": CONFIG,
            "training_state": TrainingState(),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.pt"
            torch.save(saved, path)
            model = ActorCritic(CONFIG)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            load_checkpoint(path, model, optimizer)
        observations = batch(context=False)
        with torch.no_grad():
            torch.testing.assert_close(model(observations)[1], old(observations)[1])
        self.assertEqual(float(model.matchup.weight.abs().sum()), 0.0)  # Starts from zero.
        first = next(iter(model.parameters()))
        self.assertGreater(len(optimizer.state[first]), 0)  # The old Adam state came back.
        # The restored optimizer still steps, with the new parameters included.
        model(batch())[1].sum().backward()
        optimizer.step()

    def test_a_checkpoint_saved_now_round_trips(self) -> None:
        model = ActorCritic(CONFIG)
        torch.nn.init.normal_(model.matchup.weight)
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new.pt"
            save_checkpoint(path, model, optimizer, CONFIG, TrainingState())
            restored = ActorCritic(CONFIG)
            load_checkpoint(path, restored, torch.optim.Adam(restored.parameters()))
        self.assertTrue(torch.equal(restored.matchup.weight, model.matchup.weight))


if __name__ == "__main__":
    unittest.main()
