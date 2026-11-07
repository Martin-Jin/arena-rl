"""Animation loading and drawing helpers for the arena viewer."""

import pygame
from pathlib import Path

from arena_ai.config import load_config

config = load_config()
PIXELS_PER_METER = config.viewer.rendering.pixels_per_meter
TARGET_FIGHTER_HEIGHT_PX = config.game.fighter.height_m * PIXELS_PER_METER


class Animation:
    """A simple animation class that cycles through a list of frames."""

    def __init__(
        self, frames: list[pygame.Surface], frame_duration: int = 3, loop: bool = True
    ):
        self.frames = frames
        self.frame_duration = frame_duration  # sim ticks per image
        self.loop = loop

    def get_frame(self, tick: int) -> pygame.Surface:
        if not self.frames:
            raise RuntimeError("Cannot draw an animation without any frames")
        i = tick // self.frame_duration
        if self.loop:
            i %= len(self.frames)
        else:
            i = min(i, len(self.frames) - 1)
        return self.frames[i]


def load_frames(
    folder: str | Path,
    source_body_height_px: int,
    target_body_height_px: float = TARGET_FIGHTER_HEIGHT_PX,
) -> list[pygame.Surface]:
    """Load all PNG frames from a folder and return them as a list of Pygame surfaces."""
    frames = []
    for p in sorted(Path(folder).glob("*.png")):
        img = pygame.image.load(p)
        if pygame.display.get_surface() is not None:
            img = img.convert_alpha()
        scale = target_body_height_px / source_body_height_px
        img = pygame.transform.scale_by(img, scale)
        frames.append(img)
    return frames


ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets" / "sprites" / "fighter"

anims = {
    name: Animation(
        load_frames(
            ASSET_ROOT / name, config.viewer.animation.source_body_height_px[name]
        ),
        config.viewer.animation.frame_duration[name],
        loop=config.viewer.animation.loop[name],
    )
    for name in config.viewer.animation.source_body_height_px
}


def draw_fighter(
    screen,
    fighter,
    tick,
    to_screen,
    turning=False,
    shade: tuple[int, int, int] | None = None,
):
    """Draw a fighter animation and return whether sprite frames were available."""
    if fighter.attack_frame:
        animation_name = "attack"
        animation_tick = fighter.attack_frame
    elif turning:
        animation_name = "turn"
        animation_tick = 0
    # TODO: Use the simulation's support-contact state here once platforms
    # exist; airborne animation should depend on contact, not floor height.
    # TODO: "falling" is not in viewer.yaml yet, so airborne fighters currently
    # fall back to the idle animation.
    elif not fighter.on_ground:
        falling_animation = anims.get("falling")
        animation_name = (
            "falling"
            if falling_animation is not None and falling_animation.frames
            else "idle"
        )
        animation_tick = tick
    elif fighter.vx:
        animation_name = "run"
        animation_tick = tick
    else:
        animation_name = "idle"
        animation_tick = tick

    animation = anims[animation_name]
    if not animation.frames:
        return False

    if animation_name == "run" and fighter.vx < 0:
        frame_index = (animation_tick // animation.frame_duration) % len(
            animation.frames
        )
        frame_index = len(animation.frames) - 1 - frame_index
        img = animation.frames[frame_index]
    else:
        img = animation.get_frame(animation_tick)
    if shade is not None:
        img = img.copy()
        img.fill((*shade, 255), special_flags=pygame.BLEND_RGBA_MULT)
    if fighter.facing_dir < 0 and animation_name != "turn":
        img = pygame.transform.flip(img, True, False)
    x_px, y_px = to_screen(fighter.x, fighter.y)
    screen.blit(img, img.get_rect(midbottom=(x_px, y_px)))
    return True
