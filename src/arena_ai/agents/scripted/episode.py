"""Scripted opponents whose behavior is redrawn each training episode.

**Why:** a fixed script is a fixed target. The learner found a swing distance and a wiggle
that beat it every time, so training gives each episode's opponent slightly different
numbers. **What:** ``EpisodePolicy`` wraps a policy from ``OPPONENTS`` and, when the
environment resets, draws that policy's options from the environment's seeded generator.
**Limits:** only ``approach``, ``parrier`` and ``random`` have options. The others are
wrapped but never change. The viewer uses the plain policies, so it is not randomized.
"""

from collections.abc import Callable

from arena_ai.agents.scripted.registry import OPPONENTS
from arena_ai.config import Config


def _draw_approach(rng, config: Config) -> dict:
    low, high = config.training.swing_start_offset_m
    return {"swing_start_offset_m": rng.uniform(low, high)}


def _draw_parrier(rng, config: Config) -> dict:
    low, high = config.training.hunter_attack_chance
    return {**_draw_approach(rng, config), "attack_chance": rng.uniform(low, high)}


# Options drawn on every reset when ``training.randomize_opponents`` is on.
RANDOMIZED_OPTIONS: dict[str, Callable] = {
    "approach": _draw_approach,
    "parrier": _draw_parrier,
}
# Opponents that take the environment's generator, so their dice are seeded either way.
SEEDED = {"parrier", "random"}


class EpisodePolicy:
    """A scripted policy with options redrawn by ``resample`` at the start of an episode."""

    def __init__(self, name: str, config: Config) -> None:
        self.name = name
        self.policy = OPPONENTS[name]
        self.config = config
        self.options: dict = {}

    def resample(self, rng) -> None:
        """Draw this episode's options from ``rng`` (the environment's seeded generator)."""
        self.options = {}
        if self.config.training.randomize_opponents and self.name in RANDOMIZED_OPTIONS:
            self.options.update(RANDOMIZED_OPTIONS[self.name](rng, self.config))
        if self.name in SEEDED:
            self.options["rng"] = rng

    def __call__(self, observation: list[float]) -> int:
        return self.policy(observation, self.config, **self.options)
