"""Define the one-learning-agent versus several-opponents training environment."""

from collections.abc import Callable, Mapping, Sequence
import random

from arena_ai.agents import selfplay
from arena_ai.agents.action import to_fighter_action
from arena_ai.agents.scripted.registry import OpponentMixture
from arena_ai.agents.scripted.view import team_observation
from arena_ai.config import Config
from arena_ai.envs.arena_env import ArenaEnv
from arena_ai.sim.observation_vector import fighter_rows
from arena_ai.sim.simulate import FighterAction
from arena_ai.sim.state import (
    AGENT_ID,
    FighterState,
    create_initial_fighters,
    opponent_ids,
)

OpponentPolicy = Callable[[list[float]], int]


def matchup_label(names: Sequence[str]) -> str:
    """Return the label stats are kept under: the sorted names joined with "+"."""
    return "+".join(sorted(names))


class OneVManyEnv(ArenaEnv):
    """Run an episode of one learning agent against one or more scripted opponents.

    ``opponents`` is a list of ``(name, policy)``. The opponents are allies: they never
    hurt each other. Each policy sees the learner-perspective observation of the agent, itself
    and its living allies (``team_observation``). The episode ends when the agent or every
    opponent is dead. With a ``mixture`` the opponents are drawn again at every reset (and
    ``opponents`` is ignored), so one environment cycles through different matchups.
    """

    def __init__(
        self,
        config: Config,
        opponents: Sequence[tuple[str, OpponentPolicy]],
        max_episode_steps: int | None = None,
        mixture: OpponentMixture | None = None,
    ) -> None:
        if mixture is None and not 1 <= len(opponents) <= config.game.observation.max_opponents:
            raise ValueError(
                f"an environment needs 1 to {config.game.observation.max_opponents} "
                f"opponents, got {len(opponents)}"
            )
        super().__init__(config, max_episode_steps)
        self.rng = random.Random()
        self.mixture = mixture
        if mixture is not None:
            opponents = self._draw_mixture()
        self._set_opponents(opponents)

    def _draw_mixture(self) -> list[tuple[str, OpponentPolicy]]:
        """Draw this episode's opponents from the mixture."""
        return [
            (name, selfplay.make_opponent_policy(name, self.config))
            for name in self.mixture.draw(self.rng)
        ]

    def _set_opponents(self, opponents: Sequence[tuple[str, OpponentPolicy]]) -> None:
        self.opponent_names = [name for name, _ in opponents]
        self.opponent_policies = dict(zip(opponent_ids(len(opponents)), [p for _, p in opponents]))
        # Reported in info["opponent"] so training can track stats per matchup. A mixture
        # reports its own label, so its drawn matchups share one line of stats.
        self.label = self.mixture.label if self.mixture else matchup_label(self.opponent_names)

    def reset(self, seed: int | None = None):
        """Reseed the RNG only when a seed is given, so later auto-resets keep drawing new
        positions, opponents and options. Each ``EpisodePolicy`` redraws its options."""
        if seed is not None:
            self.rng = random.Random(seed)
        if self.mixture is not None:
            self._set_opponents(self._draw_mixture())
        for policy in self.opponent_policies.values():
            if hasattr(policy, "resample"):
                policy.resample(self.rng)
        return super().reset(seed)

    def set_opponent_weights(self, weights) -> None:
        """Give the self-play opponents the learner's weights (see ``agents/selfplay.py``).

        The network is shared by the whole process, so one call serves every environment in it.
        """
        selfplay.load_weights(weights, self.config)

    def step(self, action: int):
        """Advance one step and tag the info mapping with the matchup."""
        observation, reward, terminated, truncated, info = super().step(action)
        info["opponent"] = self.label
        return observation, reward, terminated, truncated, info

    def create_initial_fighters(self) -> dict[str, FighterState]:
        """Create the starting state, optionally randomized."""
        rng = self.rng if self.config.training.randomize_spawns else None
        return create_initial_fighters(self.config.game, rng, len(self.opponent_policies))

    def get_actions(self, agent_action: int) -> Mapping[str, FighterAction]:
        """Convert the learner's and every living opponent's decision into actions.

        Each scripted opponent builds its own view from the fighters, so that it is the second
        fighter in it, followed by its allies. The fighters are encoded once for all views. A
        self-play opponent sees the game from its own side instead (``own_observation``).
        """
        assert self.fighters is not None
        actions = {AGENT_ID: to_fighter_action(agent_action)}
        rows = fighter_rows(self.fighters, self.config)
        for fighter_id, policy in self.opponent_policies.items():
            fighter = self.fighters[fighter_id]
            if fighter.health <= 0:
                actions[fighter_id] = FighterAction()
                continue
            if isinstance(policy, selfplay.SelfPlayPolicy):
                view = selfplay.own_observation(self.fighters, fighter_id, self.config)
            else:
                view = team_observation(self.fighters, fighter_id, self.config, rows)
            actions[fighter_id] = to_fighter_action(policy(view))
        return actions
