"""Reward calculation for training environments."""

from collections.abc import Mapping

from arena_ai.config import Config
from arena_ai.sim.state import FighterState


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
    """Reward damage dealt, damage received, terminal outcomes; penalise walls."""
    scoring = config.scoring
    agent = fighters["agent"]
    opponents = [f for fid, f in fighters.items() if fid != "agent"]
    opponent_damage = sum(
        previous_health[fid] - f.health for fid, f in fighters.items() if fid != "agent"
    )
    reward = scoring.damage_reward_scale * (
        opponent_damage - (previous_health["agent"] - agent.health)
    )
    reward -= edge_penalty(agent, config)
    if agent.health <= 0:
        reward -= scoring.defeat_penalty
    elif any(f.health <= 0 for f in opponents):
        reward += scoring.victory_reward
    return reward
