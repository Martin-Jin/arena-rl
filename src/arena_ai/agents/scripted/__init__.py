"""Non-learning opponents for training and the viewer.

Each policy is a function ``policy(observation, config) -> action index``. ``registry`` holds
``OPPONENTS`` (by name) and ``training_opponents``. ``view`` says how a policy reads its
observation and ``teamwork`` makes opponents on the same side help each other.
"""

from arena_ai.agents.scripted.approach import approach_and_attack
from arena_ai.agents.scripted.episode import EpisodePolicy
from arena_ai.agents.scripted.parrier import HUNTER_ATTACK_CHANCE, parrying_hunter
from arena_ai.agents.scripted.registry import OPPONENTS, training_opponents
from arena_ai.agents.scripted.runner import runner
from arena_ai.agents.scripted.simple import idle, random_action

__all__ = [
    "EpisodePolicy",
    "HUNTER_ATTACK_CHANCE",
    "OPPONENTS",
    "approach_and_attack",
    "idle",
    "parrying_hunter",
    "random_action",
    "runner",
    "training_opponents",
]
