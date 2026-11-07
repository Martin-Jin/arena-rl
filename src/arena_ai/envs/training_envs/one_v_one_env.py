"""Define the one-learning-agent versus one-opponent training environment."""

from collections.abc import Callable, Mapping
from functools import partial

from arena_ai.agents.action import to_fighter_action
from arena_ai.agents.scripted import approach_and_attack
from arena_ai.config import Config
from arena_ai.envs.arena_env import ArenaEnv
from arena_ai.sim.simulate import FighterAction
from arena_ai.sim.state import FighterState, create_initial_fighters


class OneVOneEnv(ArenaEnv):
    """Run a two-fighter episode with one learning agent and one opponent."""

    def __init__(
        self,
        config: Config,
        opponent_policy: Callable[[list[float]], int] | None = None,
        max_episode_steps: int | None = None,
    ) -> None:
        """Initialize the one-versus-one scenario and its optional opponent."""
        super().__init__(config, max_episode_steps)
        self.opponent_policy: Callable[[list[float]], int] = opponent_policy or partial(
            approach_and_attack, config=config
        )

    def create_initial_fighters(self) -> dict[str, FighterState]:
        """Create the deterministic one-versus-one starting state."""
        return create_initial_fighters(self.config.game)

    def get_actions(
        self,
        agent_action: int,
        observation: list[float],
    ) -> Mapping[str, FighterAction]:
        """Convert learner and opponent decisions into simulation actions."""
        opponent_action = self.opponent_policy(observation)
        return {
            "agent": to_fighter_action(agent_action),
            "opponent": to_fighter_action(opponent_action),
        }
