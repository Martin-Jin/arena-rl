"""Names for the fighters of a match, shared by the match, its HUD and the main menu."""

from collections import Counter

from arena_ai.sim.state import AGENT_ID, opponent_ids
from arena_ai.viewer.settings import OPPONENT_OPTIONS, PLAYER_OPTIONS, label_for


def opponent_labels(kinds: list[str]) -> list[str]:
    """Return one label per opponent, numbering repeats: "Scripted: approach #2"."""
    labels = [label_for(OPPONENT_OPTIONS, kind) for kind in kinds]
    totals = Counter(labels)
    seen: Counter[str] = Counter()
    result = []
    for label in labels:
        seen[label] += 1
        result.append(f"{label} #{seen[label]}" if totals[label] > 1 else label)
    return result


def fighter_labels(player_kind: str, opponent_kinds: list[str]) -> dict[str, str]:
    """Return the label of every fighter, keyed by fighter id."""
    labels = dict(zip(opponent_ids(len(opponent_kinds)), opponent_labels(opponent_kinds)))
    return {AGENT_ID: label_for(PLAYER_OPTIONS, player_kind), **labels}


def matchup_text(player_kind: str, opponent_kinds: list[str]) -> str:
    """Return "You vs A, B, C", or "You · Practice (solo)" with no opponents."""
    player = label_for(PLAYER_OPTIONS, player_kind)
    if not opponent_kinds:
        return f"{player}  ·  Practice (solo)"
    return f"{player}  vs  {', '.join(opponent_labels(opponent_kinds))}"
