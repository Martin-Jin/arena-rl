"""Animation loading and drawing helpers for the arena viewer."""

import pygame
from pathlib import Path


class Animation:
    """A simple animation class that cycles through a list of frames."""

    def __init__(self, frames: list[pygame.Surface], frame_duration: int = 3, loop: bool = True):
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


def load_frames(folder: str | Path, scale: float = 1) -> list[pygame.Surface]:
    """Load all PNG frames from a folder and return them as a list of Pygame surfaces."""
    frames = []
    for p in sorted(Path(folder).glob("*.png")):
        img = pygame.image.load(p)
        if pygame.display.get_surface() is not None:
            img = img.convert_alpha()
        if scale != 1:
            img = pygame.transform.scale_by(img, scale)
        frames.append(img)
    return frames


ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets" / "sprites" / "fighter"

anims = {
    "idle": Animation(load_frames(ASSET_ROOT / "idle"), 6, loop=True),
    "run": Animation(load_frames(ASSET_ROOT / "run"), 4, loop=True),
    "attack": Animation(load_frames(ASSET_ROOT / "attack"), 3, loop=False),
}


def draw_fighter(screen, fighter, tick, to_screen):
    """Draw a fighter animation and return whether sprite frames were available."""
    if fighter.attack_frame:
        animation_name = "attack"
        animation_tick = fighter.attack_frame
    elif fighter.vx:
        animation_name = "run"
        animation_tick = tick
    else:
        animation_name = "idle"
        animation_tick = tick

    animation = anims[animation_name]
    if not animation.frames:
        return False

    img = animation.get_frame(animation_tick)
    if fighter.facing_dir < 0:
        img = pygame.transform.flip(img, True, False)
    x_px, y_px = to_screen(fighter.x, fighter.y)
    screen.blit(img, img.get_rect(midbottom=(x_px, y_px)))
    return True
