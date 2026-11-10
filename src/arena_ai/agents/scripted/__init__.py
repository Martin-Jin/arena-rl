"""Non-learning opponents for training and the viewer.

Each policy is a function ``policy(observation, config) -> action index``. ``registry`` holds
``OPPONENTS`` (by name) and ``training_opponents``. ``view`` says how a policy reads its
observation and ``teamwork`` makes opponents on the same side help each other.
"""

from arena_ai.agents.scripted.approach import approach_and_attack
from arena_ai.agents.scripted.dodger import dodging_approacher
from arena_ai.agents.scripted.episode import EpisodePolicy
from arena_ai.agents.scripted.parrier import HUNTER_ATTACK_CHANCE, parrying_hunter
from arena_ai.agents.scripted.registry import (
    CONTEXT_SIZE,
    OPPONENT_TYPES,
    OPPONENTS,
    SELF_PLAY,
    OpponentMixture,
    opponent_context,
    random_opponent_pool,
    training_opponents,
    uses_self_play,
)
from arena_ai.agents.scripted.runner import runner
from arena_ai.agents.scripted.simple import idle, random_action

__all__ = [
    "CONTEXT_SIZE",
    "EpisodePolicy",
    "HUNTER_ATTACK_CHANCE",
    "OPPONENTS",
    "OPPONENT_TYPES",
    "OpponentMixture",
    "SELF_PLAY",
    "approach_and_attack",
    "dodging_approacher",
    "idle",
    "opponent_context",
    "parrying_hunter",
    "random_action",
    "random_opponent_pool",
    "runner",
    "training_opponents",
    "uses_self_play",
]
