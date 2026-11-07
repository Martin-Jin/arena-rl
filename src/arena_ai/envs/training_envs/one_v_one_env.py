"""Define the one-learning-agent versus one-opponent training environment."""

from collections.abc import Callable, Mapping
from functools import partial
import random

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
        self.rng = random.Random()
        self.opponent_policy: Callable[[list[float]], int] = opponent_policy or partial(
            approach_and_attack, config=config
        )

    def reset(self, seed: int | None = None):
        """Reseed the spawn RNG only when a seed is given, so later
        auto-resets keep drawing new positions."""
        if seed is not None:
            self.rng = random.Random(seed)
        return super().reset(seed)

    def create_initial_fighters(self) -> dict[str, FighterState]:
        """Create the one-versus-one starting state, optionally randomized."""
        fighters = create_initial_fighters(self.config.game)
        if not self.config.training.randomize_spawns:
            return fighters
        world = self.config.game.world
        margin = self.config.game.fighter.width_m / 2
        for fighter in fighters.values():
            fighter.x = self.rng.uniform(margin, world.arena_width_m - margin)
        agent, opponent = fighters["agent"], fighters["opponent"]
        agent.facing_dir = 1 if opponent.x >= agent.x else -1
        opponent.facing_dir = -agent.facing_dir
        return fighters

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
