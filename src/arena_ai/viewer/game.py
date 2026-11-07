# game.py
# This module defines the main game loop that runs the
# simulation and renders the game using Pygame.

import pygame
import torch

from arena_ai.agents.action import to_fighter_action
from arena_ai.agents.policy import ActorCritic
from arena_ai.config import load_config
from arena_ai.agents.scripted import approach_and_attack
from arena_ai.sim.observation import get_observation
from arena_ai.sim.observation_vector import encode_observation
from arena_ai.sim.simulate import step_game
from arena_ai.sim.state import FighterState, create_initial_fighters
from arena_ai.viewer.annimation import draw_fighter
from arena_ai.viewer.audio import AudioManager
from arena_ai.training.checkpoints import load_checkpoint

# Switch this to False to run the original single-fighter practice viewer.
VIEW_OPPONENT = True
# When enabled, the opponent uses the latest trained checkpoint instead of the
# scripted policy. VIEW_OPPONENT must also be enabled.
PLAY_AGAINST_TRAINED_AGENT = False
TRAINED_AGENT_CHECKPOINT = "checkpoints/latest.pt"

# Load all application configuration from the YAML files in configs/.
config = load_config()
DT = config.game.world.dt_seconds
trained_model: ActorCritic | None = None
if PLAY_AGAINST_TRAINED_AGENT:
    if not VIEW_OPPONENT:
        raise ValueError("PLAY_AGAINST_TRAINED_AGENT requires VIEW_OPPONENT")
    trained_model = ActorCritic(config)
    load_checkpoint(TRAINED_AGENT_CHECKPOINT, trained_model)
    trained_model.eval()

PIXELS_PER_METER = config.viewer.rendering.pixels_per_meter
FLOOR_PX = config.viewer.rendering.floor_y_px


# Convert meters to pixels for rendering.
def to_screen(x_m: float, y_m: float) -> tuple[float, float]:
    return x_m * PIXELS_PER_METER, FLOOR_PX - y_m * PIXELS_PER_METER


def trained_opponent_action() -> int:
    """Choose a greedy action for the trained model from the opponent view."""
    if trained_model is None:
        raise RuntimeError("trained opponent model has not been loaded")
    structured = get_observation(
        fighters["opponent"],
        {"agent": fighters["agent"]},
        config,
    )
    observation = torch.tensor(
        encode_observation(structured, config), dtype=torch.float32
    ).unsqueeze(0)
    with torch.no_grad():
        logits, _ = trained_model(observation)
    return int(torch.argmax(logits, dim=-1).item())


def render(
    screen: pygame.Surface,
    fighters: dict[str, FighterState],
    tick: int,
    turning: bool = False,
) -> None:
    """Render the current game frame."""
    screen.fill(config.viewer.ui.background_color)
    for fighter_id, fighter in fighters.items():
        is_opponent = fighter_id == "opponent"
        if not draw_fighter(
            screen,
            fighter,
            tick,
            to_screen,
            turning=turning and not is_opponent,
            shade=config.viewer.ui.opponent_tint if is_opponent else None,
        ):
            h_px = config.game.fighter.height_m * PIXELS_PER_METER
            w_px = h_px * 0.3
            x_px, y_px = to_screen(fighter.x, fighter.y)
            color = (
                config.viewer.ui.fallback_opponent_color
                if is_opponent
                else config.viewer.ui.fallback_agent_color
            )
            pygame.draw.rect(screen, color, (x_px - w_px / 2, y_px - h_px, w_px, h_px))


# Initialize Pygame and run the main game loop.
pygame.init()
screen = pygame.display.set_mode(
    (config.viewer.rendering.screen_width, config.viewer.rendering.screen_height)
)
clock = pygame.time.Clock()
font = pygame.font.Font(None, config.viewer.ui.font_size)
button_font = pygame.font.Font(None, config.viewer.ui.button_font_size)
audio = AudioManager(config.viewer.audio)
audio.start_music()
running = True
accumulator = 0.0
render_tick = 0
turn_timer = 0
last_move_dir = 0
fighters = (
    create_initial_fighters(config.game) if VIEW_OPPONENT else {"agent": FighterState()}
)
game_over = False
winner = ""
restart_button: pygame.Rect | None = None


def reset_game() -> None:
    """Reset the viewer to a fresh episode."""
    global fighters, game_over, winner, accumulator, render_tick, turn_timer
    fighters = (
        create_initial_fighters(config.game)
        if VIEW_OPPONENT
        else {"agent": FighterState()}
    )
    game_over = False
    winner = ""
    accumulator = 0.0
    render_tick = 0
    turn_timer = 0


def draw_health_ui() -> None:
    """Draw learner and opponent health bars."""
    ui = config.viewer.ui
    bar_width = ui.health_bar_width
    bar_height = ui.health_bar_height
    margin = ui.health_bar_margin
    y = ui.health_bar_y
    for index, (fighter_id, label, color) in enumerate(
        (
            ("agent", ui.health_label, ui.agent_health_color),
            ("opponent", ui.opponent_health_label, ui.opponent_health_color),
        )
    ):
        if fighter_id not in fighters:
            continue
        fighter = fighters[fighter_id]
        x = margin if index == 0 else screen.get_width() - margin - bar_width
        background = pygame.Rect(x, y, bar_width, bar_height)
        health_ratio = max(
            0.0, min(1.0, fighter.health / config.game.fighter.max_health)
        )
        fill = pygame.Rect(x, y, round(bar_width * health_ratio), bar_height)
        pygame.draw.rect(screen, ui.health_background_color, background)
        pygame.draw.rect(screen, color, fill)
        pygame.draw.rect(screen, ui.text_color, background, ui.health_border_width)
        text = font.render(f"{label}: {fighter.health}", True, ui.text_color)
        screen.blit(text, (x, y + bar_height + 5))


def draw_end_screen() -> pygame.Rect:
    """Draw the end-game overlay and return the restart button bounds."""
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    ui = config.viewer.ui
    overlay.fill((*ui.overlay_color, ui.overlay_alpha))
    screen.blit(overlay, (0, 0))
    title = button_font.render(winner, True, ui.text_color)
    screen.blit(
        title,
        title.get_rect(
            center=(
                screen.get_width() // 2,
                screen.get_height() // 2 - ui.end_title_y_offset,
            )
        ),
    )
    button = pygame.Rect(
        screen.get_width() // 2 - ui.restart_button_width // 2,
        screen.get_height() // 2,
        ui.restart_button_width,
        ui.restart_button_height,
    )
    pygame.draw.rect(
        screen,
        ui.restart_button_color,
        button,
        border_radius=ui.restart_button_radius,
    )
    pygame.draw.rect(
        screen,
        ui.restart_button_border_color,
        button,
        ui.restart_button_border_width,
        border_radius=ui.restart_button_radius,
    )
    text = button_font.render(ui.restart_label, True, ui.restart_button_border_color)
    screen.blit(text, text.get_rect(center=button.center))
    return button


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif game_over and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if restart_button is not None and restart_button.collidepoint(event.pos):
                reset_game()
                restart_button = None

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
    move_dir = 0 if game_over else int(keys[pygame.K_d]) - int(keys[pygame.K_a])
    if move_dir and move_dir != last_move_dir:
        turn_timer = 6
    if move_dir:
        last_move_dir = move_dir
    jump = not game_over and keys[pygame.K_w]
    attack = not game_over and keys[pygame.K_SPACE]

    # Simulate enough fixed timesteps to catch up after a lag spike.
    while accumulator >= DT and not game_over:
        agent = fighters["agent"]
        actions = {
            "agent": to_fighter_action(
                3
                if jump and move_dir == 0
                else 4
                if jump and move_dir < 0
                else 5
                if jump and move_dir > 0
                else 1
                if move_dir < 0
                else 2
                if move_dir > 0
                else 6
                if attack
                else 0
            )
        }
        if VIEW_OPPONENT:
            structured = get_observation(
                agent,
                {"opponent": fighters["opponent"]},
                config,
            )
            opponent_action = (
                trained_opponent_action()
                if PLAY_AGAINST_TRAINED_AGENT
                else approach_and_attack(
                    encode_observation(structured, config),
                    config,
                )
            )
            actions["opponent"] = to_fighter_action(opponent_action)
        was_on_ground = agent.on_ground
        previous_attack_frame = agent.attack_frame
        step_game(fighters, actions, config)
        if any(fighter.health <= 0 for fighter in fighters.values()):
            game_over = True
            winner = (
                config.viewer.ui.win_message
                if fighters["agent"].health > 0
                else config.viewer.ui.loss_message
            )
        audio.update_fighter(
            agent,
            move_dir,
            was_on_ground,
            previous_attack_frame,
        )
        accumulator -= DT

    render(screen, fighters, render_tick, turning=turn_timer > 0)
    draw_health_ui()
    if game_over:
        restart_button = draw_end_screen()
    pygame.display.flip()
    turn_timer = max(0, turn_timer - 1)
    render_tick += 1

audio.close()
pygame.quit()
