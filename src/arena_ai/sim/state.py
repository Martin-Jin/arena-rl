# state.py
# This module defines the sprite states that are updated in the game loop.


from dataclasses import dataclass


@dataclass
class FighterState:
    x: float = 6.0  # meters
    y: float = 0.0
    vx: float = 0.0  # m/s
    vy: float = 0.0
    on_ground: bool = True
