# game.py
# This module defines the main game loop that runs the
# simulation and renders the game using Pygame.

import pygame

from arena_ai.config import load_config
from arena_ai.sim.rules import step_fighter
from arena_ai.sim.state import FighterState
from arena_ai.viewer.annimation import PIXELS_PER_METER, draw_fighter
from arena_ai.viewer.audio import AudioManager

# Load all application configuration from the YAML files in configs/.
config = load_config()
DT = config.game.world.dt_seconds

PIXELS_PER_METER = config.viewer.rendering.pixels_per_meter
FLOOR_PX = config.viewer.rendering.floor_y_px


# Convert meters to pixels for rendering.
def to_screen(x_m: float, y_m: float) -> tuple[float, float]:
    return x_m * PIXELS_PER_METER, FLOOR_PX - y_m * PIXELS_PER_METER


def render(
    screen: pygame.Surface,
    fighter: FighterState,
    tick: int,
    turning: bool = False,
) -> None:
    """Render the current game frame."""
    screen.fill("white")
    if not draw_fighter(screen, fighter, tick, to_screen, turning=turning):
        h_px = config.game.fighter.height_m * PIXELS_PER_METER
        w_px = h_px * 0.3
        x_px, y_px = to_screen(fighter.x, fighter.y)
        pygame.draw.rect(screen, "red", (x_px - w_px / 2, y_px - h_px, w_px, h_px))
    pygame.display.flip()


# Initialize Pygame and run the main game loop.
pygame.init()
screen = pygame.display.set_mode(
    (config.viewer.rendering.screen_width, config.viewer.rendering.screen_height)
)
clock = pygame.time.Clock()
audio = AudioManager(config.viewer.audio)
audio.start_music()
running = True
accumulator = 0.0
render_tick = 0
turn_timer = 0
last_move_dir = 0
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

    # Key bindings overview:
    #   W: jump
    #   A: move left
    #   D: move right
    #   SPACE: attack
    keys = pygame.key.get_pressed()
    # -1 for right, 0 for no movement, 1 for left
    move_dir = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
    if move_dir and move_dir != last_move_dir:
        turn_timer = 6
    if move_dir:
        last_move_dir = move_dir
    jump = keys[pygame.K_w]
    attack = keys[pygame.K_SPACE]

    # Simulate enough fixed timesteps to catch up after a lag spike.
    while accumulator >= DT:
        # TODO: Track the generalized support-contact transition when platforms
        # are added so jump and landing effects work on every walkable surface.
        was_on_ground = fighter.on_ground
        previous_attack_frame = fighter.attack_frame
        step_fighter(fighter, move_dir, jump, config.game, attack=attack)
        audio.update_fighter(
            fighter,
            move_dir,
            was_on_ground,
            previous_attack_frame,
        )
        accumulator -= DT

    render(screen, fighter, render_tick, turning=turn_timer > 0)
    turn_timer = max(0, turn_timer - 1)
    render_tick += 1

audio.close()
pygame.quit()
