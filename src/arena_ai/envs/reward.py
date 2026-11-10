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
    time_left: float = 0.0,
) -> float:
    """Reward damage, parries and counter hits; penalise damage taken, misses and walls.

    The victory bonus is paid once every opponent is dead, with a bonus for the agent's
    remaining health and for ``time_left`` (the share of the episode time limit still
    left, from 0 to 1). Neither extra is paid on a loss or a time-out, so nothing rewards
    ending a fight sooner unless the agent wins it. The time-out bonus is added by
    ``ArenaEnv.step``, since only it knows the episode timed out.
    """
    scoring = config.scoring
    agent = fighters[AGENT_ID]
    opponents = enemies_of(AGENT_ID, fighters)
    opponent_damage = sum(previous_health[fid] - f.health for fid, f in opponents.items())
    reward = scoring.damage_reward_scale * (
        opponent_damage - (previous_health[AGENT_ID] - agent.health)
    )
    # Each opponent that reached 0 health this step. Damage is paid the same however it is
    # spread, so this is what rewards finishing an attacker off instead of wearing all down.
    kills = sum(previous_health[fid] > 0 and f.health <= 0 for fid, f in opponents.items())
    reward += scoring.kill_reward * kills
    reward -= edge_penalty(agent, config)
    reward += scoring.parry_reward * agent.parried
    reward += scoring.counter_reward * agent.counter_hit
    reward -= scoring.missed_parry_penalty * agent.missed_parry
    reward -= scoring.missed_attack_penalty * agent.missed_attack
    if agent.health <= 0:
        reward -= scoring.defeat_penalty
    elif opponents and all(f.health <= 0 for f in opponents.values()):
        reward += scoring.victory_reward
        reward += scoring.health_reward * agent.health / config.game.fighter.max_health
        reward += scoring.victory_time_bonus * time_left
    return reward
