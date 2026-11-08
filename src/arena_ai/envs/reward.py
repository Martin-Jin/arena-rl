"""Reward calculation for training environments."""

from collections.abc import Mapping

from arena_ai.config import Config
from arena_ai.sim.state import AGENT_ID, FighterState, enemies_of


def wall_distance(agent: FighterState, config: Config) -> float:
    """Return the distance in meters from the agent to the nearest wall."""
    return min(agent.x, config.game.world.arena_width_m - agent.x)


def edge_penalty(agent: FighterState, config: Config) -> float:
    """Return a penalty that grows linearly as the agent nears a wall."""
    scoring = config.scoring
    distance = wall_distance(agent, config)
    if distance >= scoring.edge_margin:
        return 0.0
    return scoring.edge_penalty_scale * (1 - distance / scoring.edge_margin)


def calculate_reward(
    previous_health: Mapping[str, int],
    fighters: Mapping[str, FighterState],
    config: Config,
) -> float:
    """Reward damage, parries and counter hits; penalise damage taken, misses, walls and time.

    The victory bonus is paid once every opponent is dead. The time-out bonus is added by
    ``ArenaEnv.step``, since only it knows the episode timed out.
    """
    scoring = config.scoring
    agent = fighters[AGENT_ID]
    opponents = enemies_of(AGENT_ID, fighters)
    opponent_damage = sum(previous_health[fid] - f.health for fid, f in opponents.items())
    reward = scoring.damage_reward_scale * (
        opponent_damage - (previous_health[AGENT_ID] - agent.health)
    )
    reward -= edge_penalty(agent, config)
    reward -= scoring.time_penalty_per_step
    reward += scoring.parry_reward * agent.parried
    reward += scoring.counter_reward * agent.counter_hit
    reward -= scoring.missed_parry_penalty * agent.missed_parry
    reward -= scoring.missed_attack_penalty * agent.missed_attack
    if agent.health <= 0:
        reward -= scoring.defeat_penalty
    elif opponents and all(f.health <= 0 for f in opponents.values()):
        reward += scoring.victory_reward
    return reward
