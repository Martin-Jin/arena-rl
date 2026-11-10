import unittest

import torch

from arena_ai.agents.policy import ActorCritic, PolicyAdapter
from arena_ai.config import load_config
from arena_ai.sim.observation_vector import FEATURES_PER_FIGHTER
from arena_ai.training.ppo import GraphedTrainStep, train_step

CONFIG = load_config()
OPPONENTS = CONFIG.game.observation.max_opponents
OBSERVATION_SIZE = FEATURES_PER_FIGHTER * (OPPONENTS + 1) + OPPONENTS
POLICY = CONFIG.agent.policy
BATCH = 64
SAMPLES = BATCH * 3 + 20  # Three full minibatches and a short one.


def make_data(device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(0)
    observations = torch.rand(SAMPLES, OBSERVATION_SIZE, generator=generator)
    observations[:, -OPPONENTS:] = (observations[:, -OPPONENTS:] > 0.5).float()
    actions = torch.randint(0, 24, (SAMPLES,), generator=generator)
    log_probs = -torch.rand(SAMPLES, generator=generator) * 3
    advantages = torch.randn(SAMPLES, generator=generator)
    returns = torch.randn(SAMPLES, generator=generator)
    return tuple(t.to(device) for t in (observations, actions, log_probs, advantages, returns))


def make_model(device: torch.device):
    torch.manual_seed(1)
    model = ActorCritic(CONFIG).to(device)
    return model, PolicyAdapter(model, device)


def eager_step(policy, optimizer, tensors, indices) -> dict[str, torch.Tensor]:
    return train_step(
        policy.forward,
        optimizer,
        *(tensor[indices] for tensor in tensors),
        POLICY.clip_range,
        POLICY.value_loss_coef,
        POLICY.entropy_coef,
    )


def graphed(policy, optimizer, tensors) -> GraphedTrainStep:
    example = tuple(tensor[:BATCH] for tensor in tensors)
    return GraphedTrainStep(
        policy.forward,
        optimizer,
        example,
        POLICY.clip_range,
        POLICY.value_loss_coef,
        POLICY.entropy_coef,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA graphs need a CUDA device")
class GraphedTrainStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda")
        self.tensors = make_data(self.device)
        self.indices = torch.randperm(SAMPLES, device=self.device).split(BATCH)  # 3 full, 1 short

    def test_building_it_leaves_the_weights_and_optimizer_untouched(self) -> None:
        model, policy = make_model(self.device)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=POLICY.learning_rate)
        graphed(policy, optimizer, self.tensors)
        for parameter, saved in zip(model.parameters(), before):
            torch.testing.assert_close(parameter, saved, atol=0, rtol=0)
        for state in optimizer.state.values():
            self.assertEqual(float(state["step"]), 0.0)
            self.assertEqual(float(state["exp_avg"].abs().max()), 0.0)
            self.assertEqual(float(state["exp_avg_sq"].abs().max()), 0.0)

    def test_it_restores_an_optimizer_that_already_has_state(self) -> None:
        model, policy = make_model(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=POLICY.learning_rate)
        eager_step(policy, optimizer, self.tensors, self.indices[0])  # A "resumed" optimizer.
        saved = {
            parameter: {name: value.clone() for name, value in optimizer.state[parameter].items()}
            for parameter in model.parameters()
        }
        saved_weights = [parameter.detach().clone() for parameter in model.parameters()]
        graphed(policy, optimizer, self.tensors)
        for parameter, weights in zip(model.parameters(), saved_weights):
            torch.testing.assert_close(parameter, weights, atol=0, rtol=0)
            for name, value in optimizer.state[parameter].items():
                # The step counter moves to the GPU for the graph, so compare on the CPU.
                torch.testing.assert_close(
                    value.float().cpu(), saved[parameter][name].float().cpu(), atol=0, rtol=0
                )

    def test_steps_match_ordinary_training_steps(self) -> None:
        graph_model, graph_policy = make_model(self.device)
        eager_model, eager_policy = make_model(self.device)
        graph_optimizer = torch.optim.Adam(graph_model.parameters(), lr=POLICY.learning_rate)
        eager_optimizer = torch.optim.Adam(eager_model.parameters(), lr=POLICY.learning_rate)
        step = graphed(graph_policy, graph_optimizer, self.tensors)
        totals: dict[str, torch.Tensor] = {}
        for indices in self.indices:  # Includes the short one, which runs as an ordinary step.
            step.step(self.tensors, indices)
            stats = eager_step(eager_policy, eager_optimizer, self.tensors, indices)
            for name, value in stats.items():
                totals[name] = totals[name] + value if name in totals else value
        for graph_parameter, eager_parameter in zip(
            graph_model.parameters(), eager_model.parameters()
        ):
            torch.testing.assert_close(graph_parameter, eager_parameter, atol=1e-4, rtol=1e-3)
        means = step.means()
        self.assertEqual(set(means), set(totals))
        for name, value in means.items():
            expected = totals[name] / len(self.indices)
            torch.testing.assert_close(value, expected, atol=1e-4, rtol=1e-3)

    def test_means_start_over_after_each_call(self) -> None:
        model, policy = make_model(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=POLICY.learning_rate)
        step = graphed(policy, optimizer, self.tensors)
        step.step(self.tensors, self.indices[0])
        first = step.means()
        step.step(self.tensors, self.indices[1])
        second = step.means()
        self.assertNotEqual(float(first["loss"]), float(second["loss"]))
        self.assertEqual(step.count, 0)


if __name__ == "__main__":
    unittest.main()
