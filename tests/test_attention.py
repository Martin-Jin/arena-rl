import dataclasses
import tempfile
import unittest
from pathlib import Path

import torch

from arena_ai.agents.attention import EntityAttention
from arena_ai.agents.policy import ActorCritic
from arena_ai.config import load_config
from arena_ai.sim.observation_vector import FEATURES_PER_FIGHTER as F
from arena_ai.sim.observation_vector import observation_size
from arena_ai.training.checkpoints import load_checkpoint, save_checkpoint
from arena_ai.training.train import TrainingState

CONFIG = load_config()
SLOTS = CONFIG.game.observation.max_opponents
EMBED = CONFIG.agent.network.embed_dim


def config_with(slots: int = SLOTS, **network):
    observation = dataclasses.replace(CONFIG.game.observation, max_opponents=slots)
    game = dataclasses.replace(CONFIG.game, observation=observation)
    agent = dataclasses.replace(
        CONFIG.agent, network=dataclasses.replace(CONFIG.agent.network, **network)
    )
    return dataclasses.replace(CONFIG, game=game, agent=agent)


def batch(real: int, slots: int = SLOTS, size: int = 5, seed: int = 0) -> torch.Tensor:
    """Observations with ``real`` living opponents in the first slots."""
    generator = torch.Generator().manual_seed(seed)
    out = torch.zeros(size, observation_size(slots))
    out[:, : F * (1 + real)] = torch.rand(size, F * (1 + real), generator=generator)
    out[:, F * (1 + slots) : F * (1 + slots) + real] = 1.0
    return out


def same_weights(source: ActorCritic, target: ActorCritic) -> None:
    target.load_state_dict(source.state_dict())


class AttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = ActorCritic(CONFIG).eval()

    def test_shapes_for_a_batch(self) -> None:
        x = batch(2, size=7)
        logits = self.model.policy_network(x)
        value = self.model.value_network(x)
        self.assertEqual(tuple(logits.shape), (7, 9))
        self.assertEqual(tuple(value.shape), (7, 1))
        combined = self.model.policy_network.encoder(x)
        self.assertEqual(tuple(combined.shape), (7, 2 * EMBED))

    def test_the_hidden_layers_get_self_embedding_then_context(self) -> None:
        x = batch(3, size=4)
        encoder = self.model.policy_network.encoder
        combined = encoder(x)
        e_self = encoder.phi_self(x[:, :F])
        self.assertTrue(torch.allclose(combined[:, :EMBED], e_self))

    def test_the_output_does_not_depend_on_opponent_order(self) -> None:
        x = batch(3)
        shuffled = x.clone()
        rows = x[:, F : F * 4].reshape(-1, 3, F)
        shuffled[:, F : F * 4] = rows[:, [2, 0, 1]].reshape(-1, 3 * F)
        for network in (self.model.policy_network, self.model.value_network):
            self.assertTrue(torch.allclose(network(x), network(shuffled), atol=1e-5))

    def test_the_output_is_the_same_for_any_padding(self) -> None:
        small = ActorCritic(config_with(slots=3)).eval()
        large = ActorCritic(config_with(slots=5)).eval()
        same_weights(small, large)
        real = 2
        a = batch(real, slots=3)
        b = torch.zeros(a.shape[0], observation_size(5))
        b[:, : F * (1 + real)] = a[:, : F * (1 + real)]
        b[:, F * 6 : F * 6 + real] = 1.0
        self.assertTrue(
            torch.allclose(small.policy_network(a), large.policy_network(b), atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(small.value_network(a), large.value_network(b), atol=1e-5)
        )

    def test_masked_slots_have_no_influence(self) -> None:
        x = batch(1)
        noisy = x.clone()
        noisy[:, F * 2 : F * 4] = torch.rand(x.shape[0], 2 * F) * 5  # Dead slots' rows.
        for network in (self.model.policy_network, self.model.value_network):
            self.assertTrue(torch.equal(network(x), network(noisy)))

    def test_a_living_opponent_does_change_the_output(self) -> None:
        with_one, with_two = batch(1), batch(1)
        with_two[:, F * 2 : F * 3] = torch.rand(with_two.shape[0], F)
        with_two[:, F * (1 + SLOTS) + 1] = 1.0  # Unmask the second opponent slot.
        self.assertFalse(
            torch.allclose(
                self.model.policy_network(with_one), self.model.policy_network(with_two)
            )
        )

    def test_gradients_reach_every_part_of_the_input_block(self) -> None:
        network = self.model.policy_network
        network.zero_grad()
        network(batch(3)).sum().backward()
        encoder = network.encoder
        for name, module in (
            ("phi_self", encoder.phi_self),
            ("phi_opp", encoder.phi_opp),
            ("w_q", encoder.w_q),
            ("w_k", encoder.w_k),
            ("w_v", encoder.w_v),
        ):
            grads = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and g.abs().sum() > 0 for g in grads), name)

    def test_the_policy_and_value_networks_share_no_parameters(self) -> None:
        policy = {id(p) for p in self.model.policy_network.parameters()}
        value = {id(p) for p in self.model.value_network.parameters()}
        self.assertFalse(policy & value)

    def test_several_heads_work_and_need_an_output_projection(self) -> None:
        model = ActorCritic(config_with(num_heads=4)).eval()
        self.assertEqual(tuple(model.policy_network(batch(2)).shape), (5, 9))
        self.assertNotIsInstance(model.policy_network.encoder.out_proj, torch.nn.Identity)
        with self.assertRaises(ValueError):
            EntityAttention(3, 64, 5, "relu")

    def test_output_layer_gains_and_biases(self) -> None:
        policy_out = self.model.policy_network.output_layer
        self.assertLess(policy_out.weight.abs().max().item(), 0.2)
        self.assertEqual(policy_out.bias.abs().sum().item(), 0.0)


class CheckpointCompatibilityTests(unittest.TestCase):
    def _save(self, model, folder: Path) -> Path:
        path = folder / "model.pt"
        optimizer = torch.optim.Adam(model.parameters())
        save_checkpoint(path, model, optimizer, CONFIG, TrainingState())
        return path

    def test_a_checkpoint_loads_at_another_max_opponents(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = self._save(ActorCritic(config_with(slots=3)), Path(folder))
            load_checkpoint(path, ActorCritic(config_with(slots=5)))

    def test_a_different_network_shape_is_refused_with_a_clear_message(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = self._save(ActorCritic(CONFIG), Path(folder))
            with self.assertRaisesRegex(ValueError, "embed_dim: checkpoint 64, current 32"):
                load_checkpoint(path, ActorCritic(config_with(embed_dim=32)))

    def test_a_different_spec_version_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = self._save(ActorCritic(CONFIG), Path(folder))
            observation = dataclasses.replace(CONFIG.game.observation, spec_version=99)
            game = dataclasses.replace(CONFIG.game, observation=observation)
            other = dataclasses.replace(CONFIG, game=game)
            with self.assertRaisesRegex(ValueError, "spec_version"):
                load_checkpoint(path, ActorCritic(other))

    def test_an_old_checkpoint_without_a_signature_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.pt"
            model = ActorCritic(CONFIG)
            torch.save({"model": model.state_dict(), "config": CONFIG}, path)
            with self.assertRaisesRegex(ValueError, "train a new checkpoint"):
                load_checkpoint(path, model)
