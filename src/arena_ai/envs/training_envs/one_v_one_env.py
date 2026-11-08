"""Define the one-learning-agent versus one-opponent training environment."""

from collections.abc import Callable
from functools import partial

from arena_ai.agents.scripted import approach_and_attack
from arena_ai.config import Config
from arena_ai.envs.training_envs.one_v_many_env import OneVManyEnv


class OneVOneEnv(OneVManyEnv):
    """Run a two-fighter episode with one learning agent and one opponent."""

    def __init__(
        self,
        config: Config,
        opponent_policy: Callable[[list[float]], int] | None = None,
        max_episode_steps: int | None = None,
        opponent_name: str = "approach",
    ) -> None:
        """Initialize the one-versus-one scenario and its optional opponent."""
        policy = opponent_policy or partial(approach_and_attack, config=config)
        super().__init__(config, [(opponent_name, policy)], max_episode_steps)
