import unittest

from arena_ai.agents.action import Action, to_fighter_action
from arena_ai.agents.scripted import approach_and_attack, dodging_approacher
from arena_ai.config import AttackType, load_config
from arena_ai.sim.observation import get_observation
from arena_ai.sim.observation_vector import encode_observation
from arena_ai.sim.simulate import FighterAction, step_game
from arena_ai.sim.state import FighterState

CONFIG = load_config()
COMBAT = CONFIG.game.combat
DODGE = CONFIG.game.dodge
LAST_WINDUP = COMBAT.attack_windup_frames  # One wind-up frame left.


def observe(learner: FighterState, opponent: FighterState) -> list[float]:
    return encode_observation(get_observation(learner, {"opponent": opponent}, CONFIG), CONFIG)


def action_for(learner: FighterState, opponent: FighterState) -> Action:
    return Action(dodging_approacher(observe(learner, opponent), CONFIG))


def swinging_learner(frame: int = LAST_WINDUP) -> FighterState:
    return FighterState(x=5.0, facing_dir=1, attack_frame=frame)


class DodgerTests(unittest.TestCase):
    def test_it_dodges_toward_a_learner_whose_swing_is_about_to_land(self) -> None:
        opponent = FighterState(x=5.5, facing_dir=-1)
        self.assertEqual(action_for(swinging_learner(), opponent), Action.MOVE_LEFT_DODGE)

    def test_it_waits_while_the_swing_is_too_early_to_be_covered(self) -> None:
        opponent = FighterState(x=5.5, facing_dir=-1)
        early = LAST_WINDUP - DODGE.duration_frames  # Many wind-up frames left.
        action = action_for(swinging_learner(early), opponent)
        self.assertNotIn(action, (Action.MOVE_LEFT_DODGE, Action.MOVE_RIGHT_DODGE))

    def test_it_does_not_dodge_a_swing_that_cannot_reach_it(self) -> None:
        far = FighterState(x=8.0, facing_dir=-1)
        self.assertNotIn(action_for(swinging_learner(), far), (Action.MOVE_LEFT_DODGE,))
        behind = FighterState(x=4.0, facing_dir=1)  # The swing points away from it.
        self.assertNotIn(action_for(swinging_learner(), behind), (Action.MOVE_RIGHT_DODGE,))

    def test_it_cannot_dodge_without_stamina_or_in_the_air(self) -> None:
        tired = FighterState(x=5.5, facing_dir=-1, stamina=DODGE.cost - 1)
        self.assertEqual(action_for(swinging_learner(), tired), Action.MOVE_LEFT_ATTACK)
        airborne = FighterState(x=5.5, y=0.5, facing_dir=-1, on_ground=False)
        self.assertNotEqual(action_for(swinging_learner(), airborne), Action.MOVE_LEFT_DODGE)

    def test_without_a_swing_it_acts_like_the_approach_opponent(self) -> None:
        learner, opponent = FighterState(x=5.0, facing_dir=1), FighterState(x=9.0, facing_dir=-1)
        observation = observe(learner, opponent)
        self.assertEqual(
            dodging_approacher(observation, CONFIG), approach_and_attack(observation, CONFIG)
        )

    def test_a_swing_at_it_does_no_damage(self) -> None:
        fighters = {
            "agent": FighterState(x=5.0, facing_dir=1),
            "opponent": FighterState(x=5.5, facing_dir=-1),
        }
        swing = FighterAction(attack=True)
        for _ in range(COMBAT.swing_total_frames(AttackType.SLASH)):
            observation = observe(fighters["agent"], fighters["opponent"])
            opponent_action = to_fighter_action(dodging_approacher(observation, CONFIG))
            step_game(fighters, {"agent": swing, "opponent": opponent_action}, CONFIG)
            swing = FighterAction()
        self.assertEqual(fighters["opponent"].health, 100)
        self.assertLess(fighters["opponent"].stamina, DODGE.max_stamina)


if __name__ == "__main__":
    unittest.main()
