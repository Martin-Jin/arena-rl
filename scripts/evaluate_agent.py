"""Play a trained checkpoint against each scripted opponent and print how it does.

Run from the repository root: ``py scripts/evaluate_agent.py [checkpoint.pt] [episodes]``.
The checkpoint is a file name in ``checkpoints/`` (default: the viewer's automatic choice).
Use it to see whether a change to a scripted opponent still beats a trained agent, and by
how much health, before retraining.
"""

import random
import sys

from arena_ai.agents.action import to_fighter_action
from arena_ai.agents.scripted import OPPONENTS
from arena_ai.config import load_config
from arena_ai.sim.simulate import step_game
from arena_ai.sim.state import FighterState
from arena_ai.viewer.controllers import make_controller

MAX_STEPS = 1800  # 30 seconds


def play_episode(opponent: str, seed: int, greedy: bool, checkpoint: str, config) -> tuple:
    """Return (agent health, opponent health, steps) after one fight from fixed starts."""
    random.seed(seed)  # The hunter's attack chance uses the global random module.
    fighters = {
        "agent": FighterState(x=3.0, facing_dir=1),
        "opponent": FighterState(x=9.0, facing_dir=-1),
    }
    agent = make_controller("trained", config, greedy, "agent", checkpoint)
    scripted = make_controller(opponent, config, True, "opponent")
    steps = 0
    while steps < MAX_STEPS and all(f.health > 0 for f in fighters.values()):
        actions = {
            "agent": to_fighter_action(agent.action(fighters)),
            "opponent": to_fighter_action(scripted.action(fighters)),
        }
        step_game(fighters, actions, config)
        steps += 1
    return fighters["agent"].health, fighters["opponent"].health, steps


def main() -> None:
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else ""
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    config = load_config()
    for name in OPPONENTS:
        if name == "idle":
            continue
        for greedy in (True, False):
            results = [
                play_episode(name, seed, greedy, checkpoint, config) for seed in range(episodes)
            ]
            wins = sum(opponent_hp <= 0 for _, opponent_hp, _ in results)
            agent_hp, opponent_hp, length = (sum(col) / episodes for col in zip(*results))
            print(
                f"{name:9s} {'greedy' if greedy else 'sampled':7s} win {wins}/{episodes}  "
                f"agent hp {agent_hp:.0f}  opponent hp {opponent_hp:.0f}  length {length:.0f}"
            )


if __name__ == "__main__":
    main()
