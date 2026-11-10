import unittest

from arena_ai.agents.action import Action
from arena_ai.agents.scripted import approach_and_attack
from arena_ai.config import load_config
from arena_ai.sim.observation import get_observation
from arena_ai.sim.observation_vector import encode_observation
from arena_ai.sim.state import FighterState

CONFIG = load_config()


def action_for(learner: FighterState, opponent: FighterState) -> Action:
    structured = get_observation(learner, {"opponent": opponent}, CONFIG)
    observation = encode_observation(structured, CONFIG)
    return Action(approach_and_attack(observation, CONFIG))


class ApproachTests(unittest.TestCase):
    def test_it_keeps_walking_toward_a_learner_in_front_during_its_swing(self) -> None:
        # Regression: it stood still once the learner was inside reach, so a learner at the
        # edge of reach could step back out of the swing's hitbox. Pushing on to body
        # contact leaves no such gap. Here the learner is inside reach but not touching.
        learner = FighterState(x=5.0, facing_dir=1)
        opponent = FighterState(x=5.8, facing_dir=-1, attack_frame=3)
        self.assertEqual(action_for(learner, opponent), Action.MOVE_LEFT)

    def test_it_does_not_turn_during_its_swing_for_a_learner_behind_it(self) -> None:
        # Turning would also turn the swing.
        learner = FighterState(x=5.0, facing_dir=1)
        opponent = FighterState(x=5.8, facing_dir=1, attack_frame=3)
        self.assertEqual(action_for(learner, opponent), Action.IDLE)


if __name__ == "__main__":
    unittest.main()
