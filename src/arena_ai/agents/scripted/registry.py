"""The scripted opponents by name, and the check of a training opponent list."""

from arena_ai.agents.scripted.approach import approach_and_attack
from arena_ai.agents.scripted.parrier import parrying_hunter
from arena_ai.agents.scripted.runner import runner
from arena_ai.agents.scripted.simple import idle, random_action

# Every scripted opponent, for the training loop (``training.opponents``) and the viewer.
# ``parrier`` parries every attack it can see from the front, so a learner cannot win
# against it by attacking head on. Mix it with easier opponents so there is still a
# winning signal, and so the learner has to find another way past it.
OPPONENTS = {
    "approach": approach_and_attack,
    "parrier": parrying_hunter,
    "idle": idle,
    "runner": runner,
    "random": random_action,
}


def training_opponents(entries: list, max_envs: int, max_opponents: int) -> list[list[str]]:
    """Return the opponent names of each training environment, in the order of ``entries``.

    Each entry lists the opponents of one environment (a bare name means one opponent).
    Raises ``ValueError`` for no entries, more entries than ``max_envs``, an entry with no
    names or more than ``max_opponents``, or a name that is not in ``OPPONENTS``, so a bad
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
        names = [entry] if isinstance(entry, str) else list(entry)
        if not 1 <= len(names) <= max_opponents:
            raise ValueError(
                f"each training.opponents entry needs 1 to {max_opponents} opponents "
                f"(observation.max_opponents), got {names}"
            )
        unknown = [name for name in names if name not in OPPONENTS]
        if unknown:
            raise ValueError(
                f"Unknown training opponents {unknown}; choose from {sorted(OPPONENTS)}"
            )
        environments.append(names)
    return environments
