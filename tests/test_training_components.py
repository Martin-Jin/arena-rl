import unittest

import torch

from arena_ai.training.ppo import advantage_estimation
from arena_ai.training.rollout import Rollout


class TrainingComponentTests(unittest.TestCase):
    def test_gae_three_step_example(self) -> None:
        advantages = advantage_estimation(
            rewards=torch.tensor([1.0, 2.0, 3.0]),
            values=torch.tensor([0.5, 0.5, 0.5]),
            next_values=torch.tensor([0.5, 0.5, 0.0]),
            terminated=torch.tensor([False, False, True]),
            discount_factor=0.9,
            gae_lambda=0.8,
        )
        torch.testing.assert_close(advantages, torch.tensor([3.65, 3.75, 2.5]))

    def test_rollout_and_flattened_batch_keep_alignment(self) -> None:
        rollout = Rollout(environment_count=2)
        rollout.add(
            [[1.0], [2.0]],
            [3, 4],
            [0.1, 0.2],
            [False, True],
            [False, False],
            [0.5, 0.6],
            [-0.1, -0.2],
        )
        rollout.add(
            [[5.0], [6.0]],
            [7, 8],
            [0.3, 0.4],
            [True, False],
            [False, True],
            [0.7, 0.8],
            [-0.3, -0.4],
        )

        batch = rollout.as_batch()
        flattened = batch.flattened()

        torch.testing.assert_close(
            flattened.observations, torch.tensor([[1.0], [2.0], [5.0], [6.0]])
        )
        torch.testing.assert_close(flattened.actions, torch.tensor([3, 4, 7, 8]))
        torch.testing.assert_close(flattened.values, torch.tensor([0.5, 0.6, 0.7, 0.8]))
        torch.testing.assert_close(
            flattened.next_values, torch.zeros(4)
        )


if __name__ == "__main__":
    unittest.main()
