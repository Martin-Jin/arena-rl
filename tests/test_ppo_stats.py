import unittest

import torch
from torch import nn

from arena_ai.agents.action import HEAD_NAMES, LOGIT_COUNT
from arena_ai.agents.distribution import FactoredCategorical
from arena_ai.training.ppo import explained_variance, train_step
from arena_ai.training.stats import rollout_metrics


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Linear(4, LOGIT_COUNT)
        self.critic = nn.Linear(4, 1)

    def forward(self, observations):
        return self.actor(observations), self.critic(observations)


class PpoStatsTests(unittest.TestCase):
    def test_train_step_returns_detached_tensors_and_zero_kl_at_start(self) -> None:
        torch.manual_seed(0)
        policy = _TinyPolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)  # No movement.
        observations = torch.randn(32, 4)
        with torch.no_grad():
            logits, _ = policy(observations)
        distribution = FactoredCategorical(logits)
        actions = distribution.sample()
        stats = train_step(
            policy,
            optimizer,
            observations,
            actions,
            distribution.log_prob(actions),
            torch.randn(32),
            torch.randn(32),
            clip_range=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
        )
        for name in ("loss", "approx_kl", "clip_fraction", "entropy"):
            self.assertIn(name, stats)
        for name in HEAD_NAMES:
            self.assertIn(f"entropy_{name}", stats)
        for value in stats.values():
            self.assertFalse(value.requires_grad)
            self.assertEqual(value.ndim, 0)
        self.assertAlmostEqual(stats["approx_kl"].item(), 0.0, places=6)
        self.assertEqual(stats["clip_fraction"].item(), 0.0)
        head_total = sum(stats[f"entropy_{name}"] for name in HEAD_NAMES)
        torch.testing.assert_close(stats["entropy"], head_total)

    def test_explained_variance(self) -> None:
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(explained_variance(returns, returns).item(), 1.0)
        constant = torch.full_like(returns, returns.mean().item())
        self.assertAlmostEqual(
            explained_variance(constant, returns).item(), 0.0, places=6
        )
        self.assertEqual(explained_variance(returns, torch.ones(4)).item(), 0.0)

    def test_rollout_metrics_count_each_action_part(self) -> None:
        # Flat indices: 0 idle, 1 move left, 2 move right, 6 attack, 3 jump.
        actions = torch.tensor([0, 1, 2, 6, 3])
        metrics = rollout_metrics(actions, torch.tensor([1.0, 0.0, 0.0, 0.0, 2.0]))
        self.assertAlmostEqual(metrics["idle_rate"].item(), 0.2)
        self.assertAlmostEqual(metrics["left_rate"].item(), 0.2)
        self.assertAlmostEqual(metrics["right_rate"].item(), 0.2)
        self.assertAlmostEqual(metrics["attack_rate"].item(), 0.2)
        self.assertAlmostEqual(metrics["jump_rate"].item(), 0.2)
        self.assertAlmostEqual(metrics["reward_per_step"].item(), 0.6)
        self.assertEqual(metrics["dodge_rate"].item(), 0.0)

    def test_rollout_metrics_count_the_dodge_head(self) -> None:
        # Flat index 24 is dodge alone, 25 is dodge with a move left.
        metrics = rollout_metrics(torch.tensor([0, 24, 25, 3]), torch.zeros(4))
        self.assertAlmostEqual(metrics["dodge_rate"].item(), 0.5)


if __name__ == "__main__":
    unittest.main()
