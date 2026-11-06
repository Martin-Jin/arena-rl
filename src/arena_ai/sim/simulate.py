# simulate.py
# This module defines what occurs at each step of the game,
# used for headless simulation and training the AI agent.

# Steps forward one timestep in the game simulation, updating the fighter's state.
def step_game(fighter, move_dir, jump, config):
    """Update the game state by one timestep."""
    from arena_ai.sim.rules import step_fighter

    step_fighter(fighter, move_dir, jump, config.game)
