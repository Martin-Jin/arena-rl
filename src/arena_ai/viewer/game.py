# game.py
# This module defines the main game loop that runs the
# simulation and renders the game using Pygame.

import pygame

from arena_ai.config import load_config
from arena_ai.sim.rules import step_fighter
from arena_ai.sim.state import FighterState

# Load all application configuration from the YAML files in configs/.
config = load_config()
DT = config.game.world.dt_seconds

PIXELS_PER_METER = 100  # viewer-only setting
FLOOR_PX = 650  # screen y-coordinate of the floor


# Convert meters to pixels for rendering.
def to_screen(x_m: float, y_m: float) -> tuple[float, float]:
    return x_m * PIXELS_PER_METER, FLOOR_PX - y_m * PIXELS_PER_METER


# Initialize Pygame and run the main game loop.
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
accumulator = 0.0
fighter = FighterState()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    frame_time = (
        clock.tick(60) / 1000
    )  # This is the actual elapsed time since the previous frame, in seconds.
    accumulator += min(
        frame_time, 0.25
    )  # Cap the maximum frame time to avoid a spiral of death.

    keys = pygame.key.get_pressed()
    # -1 for right, 0 for no movement, 1 for left
    move_dir = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
    jump = keys[pygame.K_w]
    attack = keys[pygame.K_SPACE]

    # Simulate enough fixed timesteps to catch up after a lag spike.
    while accumulator >= DT:
        step_fighter(fighter, move_dir, jump, config.game, attack=attack)
        accumulator -= DT

    # render the player
    screen.fill("purple")
    h_px = config.game.fighter.height_m * PIXELS_PER_METER
    w_px = h_px * 0.3
    x_px, y_px = to_screen(fighter.x, fighter.y)
    pygame.draw.rect(screen, "red", (x_px - w_px / 2, y_px - h_px, w_px, h_px))
    pygame.display.flip()

pygame.quit()
