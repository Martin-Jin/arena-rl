"""The scripted opponents by name, and the check of a training opponent list."""

from dataclasses import dataclass

from arena_ai.agents.scripted.approach import approach_and_attack
from arena_ai.agents.scripted.dodger import dodging_approacher
from arena_ai.agents.scripted.parrier import parrying_hunter
from arena_ai.agents.scripted.runner import runner
from arena_ai.agents.scripted.simple import idle, random_action

# Every scripted opponent, for the training loop (``training.opponents``) and the viewer.
# ``parrier`` parries every attack it can see from the front, so a learner cannot win
# against it by attacking head on. Mix it with easier opponents so there is still a
# winning signal, and so the learner has to find another way past it. ``dodger`` is the
# approach opponent that rolls through every swing it can see, until its stamina runs out.
OPPONENTS = {
    "approach": approach_and_attack,
    "parrier": parrying_hunter,
    "dodger": dodging_approacher,
    "idle": idle,
    "runner": runner,
    "random": random_action,
}


@dataclass(frozen=True)
class OpponentMixture:
    """The opponents of a training environment where some are drawn again each episode.

    **Why:** a fixed opponent list is a fixed target. **What:** ``slots`` is the entry of
    ``training.opponents``: a name fixes that slot, ``None`` draws it from ``pool`` every
    episode, and ``slots=None`` also draws how many opponents there are (1 to
    ``max_opponents``). **Limits:** the draw comes from the environment's seeded generator,
    so training stays reproducible. Stats are kept under ``label``, not under what was drawn,
    so a mixture stays one line in the report.
    """

    slots: tuple[str | None, ...] | None
    pool: tuple[str, ...]
    max_opponents: int

    @property
    def label(self) -> str:
        """Return the name stats are kept under: the fixed names and ``*`` for each drawn one."""
        if self.slots is None:
            return "random"
        return "+".join(name or "*" for name in self.slots)

    def draw(self, rng) -> list[str]:
        """Return this episode's opponent names."""
        if self.slots is None:
            return [rng.choice(self.pool) for _ in range(rng.randint(1, self.max_opponents))]
        return [name or rng.choice(self.pool) for name in self.slots]


def random_opponent_pool(names: list) -> list[str]:
    """Return the opponents a drawn slot can be: ``names``, or every opponent when empty.

    Raises ``ValueError`` for a name that is not in ``OPPONENTS``.
    """
    if not names:
        return sorted(OPPONENTS)
    unknown = [name for name in names if name not in OPPONENTS]
    if unknown:
        raise ValueError(
            f"Unknown opponents {unknown} in training.random_opponent_pool; "
            f"choose from {sorted(OPPONENTS)}"
        )
    return list(names)


def training_opponents(
    entries: list, max_envs: int, max_opponents: int
) -> list[list[str | None] | None]:
    """Return the opponents of each training environment, in the order of ``entries``.

    Each entry lists the opponents of one environment (a bare name means one opponent). A
    ``None`` in the list is a slot drawn again each episode, and a ``None`` entry is a whole
    mixture, with the number of opponents drawn too (see ``OpponentMixture``).
    Raises ``ValueError`` for no entries, more entries than ``max_envs``, an empty entry or one
    with more than ``max_opponents`` slots, or a name that is not in ``OPPONENTS``, so a bad
    ``training.opponents`` stops training before it starts.
    """
    if not entries:
        raise ValueError("training.opponents must name at least one opponent")
    if len(entries) > max_envs:
        raise ValueError(
            f"training.opponents lists {len(entries)} environments but training.num_envs "
            f"allows at most {max_envs}"
        )
    environments = []
    for entry in entries:
        if entry is None:
            environments.append(None)
            continue
        names = [entry] if isinstance(entry, str) else list(entry)
        if not 1 <= len(names) <= max_opponents:
            raise ValueError(
                f"each training.opponents entry needs 1 to {max_opponents} opponents "
                f"(observation.max_opponents), got {names}"
            )
        unknown = [name for name in names if name is not None and name not in OPPONENTS]
        if unknown:
            raise ValueError(
                f"Unknown training opponents {unknown}; choose from {sorted(OPPONENTS)}"
            )
        environments.append(names)
    return environments
