import unittest

import torch

from arena_ai.agents.action import ACTION_COUNT, HEAD_NAMES, HEAD_SIZES, LOGIT_COUNT, decode_parts
from arena_ai.agents.distribution import FactoredCategorical, sample_with_log_prob


class SampleWithLogProbTests(unittest.TestCase):
    def test_log_prob_matches_factored_categorical(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(64, LOGIT_COUNT)
        action, log_prob = sample_with_log_prob(logits)
        self.assertTrue(((action >= 0) & (action < ACTION_COUNT)).all())
        torch.testing.assert_close(
            log_prob, FactoredCategorical(logits).log_prob(action)
        )

    def test_samples_follow_the_head_probabilities(self) -> None:
        torch.manual_seed(0)
        logits = torch.zeros(20000, LOGIT_COUNT)
        # Strongly favour attack = 1 (the attack head's second option).
        attack_head = HEAD_NAMES.index('attack')
        logits[:, sum(HEAD_SIZES[:attack_head]) + 1] = 3.0
        action, _ = sample_with_log_prob(logits)
        attack_rate = decode_parts(action)[attack_head].float().mean().item()
        self.assertAlmostEqual(attack_rate, 20.09 / 21.09, delta=0.02)


if __name__ == "__main__":
    unittest.main()
