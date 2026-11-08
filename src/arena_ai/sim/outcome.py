# outcome.py
# This module decides when an episode or match is over and who won it.

from collections.abc import Mapping

from arena_ai.sim.state import AGENT_ID, FighterState, enemies_of


def is_over(fighters: Mapping[str, FighterState]) -> bool:
    """Return whether the agent is dead or every opponent is dead.

    With no opponents at all (solo practice) only the agent's death ends it.
    """
    if fighters[AGENT_ID].health <= 0:
        return True
    opponents = enemies_of(AGENT_ID, fighters)
    return bool(opponents) and all(f.health <= 0 for f in opponents.values())


def winner(fighters: Mapping[str, FighterState], timed_out: bool = False) -> str | None:
    """Return "agent", "opponent" or None for a draw or a match still running.

    The agent wins when all opponents are dead and it is alive. The opponents win when
    the agent is dead and at least one is alive. If both sides die on the same step it
    is a draw. On a time-out (``timed_out``), the agent wins if its health is above the
    mean opponent health, loses if below and draws if equal. Dead opponents count as 0.
    """
    agent = fighters[AGENT_ID]
    opponents = list(enemies_of(AGENT_ID, fighters).values())
    if is_over(fighters):
        opponents_alive = any(f.health > 0 for f in opponents)
        if agent.health > 0:
            return AGENT_ID
        return "opponent" if opponents_alive else None
    if timed_out and opponents:
        mean_health = sum(max(0, f.health) for f in opponents) / len(opponents)
        if agent.health != mean_health:
            return AGENT_ID if agent.health > mean_health else "opponent"
    return None
