"""Animation loading and drawing helpers for the arena viewer."""

import math
from pathlib import Path

import pygame

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


ASSET_ROOT = Path(__file__).resolve().parents[4] / "assets" / "sprites" / "fighter"

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


BUBBLE_ARC_POINTS = 40
PARTICLES_PER_EMISSION = 2  # Particles shed from each end of the arc on each frame.


def _jitter(*values: int) -> float:
    """Return a repeatable pseudo-random number in [0, 1) for the given integers."""
    mixed = 0
    for value, prime in zip(values, (73856093, 19349663, 83492791)):
        mixed ^= (value + 2) * prime
    return (mixed % 1000) / 1000


def bubble_span_degrees(frame: int, window: int) -> float:
    """Return the angle the bubble covers on a frame of the parry window.

    It starts a little over a half circle and burns away from both ends toward the
    front, so it is smallest when the window closes.
    """
    ui = config.viewer.ui
    progress = (frame - 1) / (window - 1) if window > 1 else 0.0
    return ui.parry_bubble_arc_degrees * (1 - ui.parry_bubble_burn * progress)


def bubble_particles(frame: int, window: int, radius: float) -> list[tuple[float, float, float]]:
    """Return the burning particles as ``(dx, dy, strength)`` from the bubble's center.

    The positions are for a fighter facing right (flip dx for the other side). Each frame
    the two ends of the arc shed particles. They then flow along the bubble's surface,
    backward from the facing direction, drifting outward and fading. Each position is a
    function of the frame, so nothing has to be stored between frames.
    """
    ui = config.viewer.ui
    life = ui.parry_particle_life_frames
    particles = []
    for edge in (-1, 1):  # The top and the bottom end of the arc.
        for age in range(life):
            born = frame - age
            if born < 1:
                break
            for index in range(PARTICLES_PER_EMISSION):
                jitter = _jitter(born, edge, index)
                degrees = edge * (
                    bubble_span_degrees(born, window) / 2 + ui.parry_particle_speed_deg * age
                )
                reach = radius * (1 + ui.parry_particle_drift * age * (0.5 + jitter))
                angle = math.radians(degrees)
                particles.append(
                    (reach * math.cos(angle), reach * math.sin(angle), 1 - age / life)
                )
    return particles


def bubble_outline(
    frame: int, window: int, radius: float
) -> list[tuple[float, float, float, float]]:
    """Return the bubble's uneven edge as ``(dx, dy, fraction, flame)`` from its center.

    The positions are for a fighter facing right (flip dx for the other side). The edge
    is not a clean arc: its radius wobbles and throws out flame-like tongues that move
    from frame to frame, more so toward the ends of the arc and as the window runs out.
    ``fraction`` runs from -1 to 1 along the arc and ``flame`` (0 to 1) is how far a point
    sits in a tongue. Each point is a function of the frame, so nothing is stored.
    """
    ui = config.viewer.ui
    progress = (frame - 1) / (window - 1) if window > 1 else 0.0
    half_span = math.radians(bubble_span_degrees(frame, window) / 2)
    phase = frame * ui.parry_flame_speed
    points = []
    for step in range(BUBBLE_ARC_POINTS + 1):
        fraction = 2 * step / BUBBLE_ARC_POINTS - 1
        wobble = (
            0.5 * math.sin(6 * fraction + phase)
            + 0.3 * math.sin(13 * fraction - 1.7 * phase + 1.3)
            + 0.2 * math.sin(29 * fraction + 2.3 * phase)
        )
        tongue = max(0.0, math.sin(17 * fraction + 2.1 * phase)) ** 3
        end_weight = 0.25 + 0.75 * fraction**2  # The ends burn hardest.
        amplitude = ui.parry_flame_amplitude * (1 + 1.5 * progress) * end_weight
        reach = radius * (1 + amplitude * (0.6 * wobble + tongue))
        angle = half_span * fraction
        points.append(
            (reach * math.cos(angle), reach * math.sin(angle), fraction, tongue * end_weight)
        )
    return points


def draw_parry_bubble(
    screen: pygame.Surface,
    fighter,
    x_px: float,
    y_px: float,
    frame: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a bubble outline with a burning edge on the side a fighter faces.

    It is centered on the fighter's body and drawn only while its parry window is open
    (``frame`` counts from the press), so it never shows protection the fighter does not
    have. ``y_px`` is the fighter's feet and ``color`` is the fighter's side color. See
    ``bubble_span_degrees``, ``bubble_outline`` and ``bubble_particles`` for how it burns.
    """
    ui = config.viewer.ui
    body_px = TARGET_FIGHTER_HEIGHT_PX
    radius = body_px * ui.parry_bubble_radius_scale
    border = ui.parry_bubble_border_px
    hot = ui.colors["on_accent"]  # The tips of the flames are lighter than the rest.
    window = fighter.parry_window
    reach = radius * (1 + ui.parry_particle_drift * ui.parry_particle_life_frames * 1.5)
    reach *= 1 + 3 * ui.parry_flame_amplitude
    side = math.ceil(2 * (reach + border + ui.parry_particle_size_px))
    layer = pygame.Surface((side, side), pygame.SRCALPHA)
    center = side / 2
    sign = 1 if fighter.facing_dir >= 0 else -1
    outline = [(center + sign * dx, center + dy, fraction, flame)
               for dx, dy, fraction, flame in bubble_outline(frame, window, radius)]
    # The outline is solid with no fill, and does not fade: it is there for the whole
    # window and gone as soon as the window ends.
    for (x0, y0, _, flame0), (x1, y1, _, flame1) in zip(outline, outline[1:]):
        flame = (flame0 + flame1) / 2
        tint = tuple(round(c + (h - c) * 0.6 * flame) for c, h in zip(color, hot))
        pygame.draw.line(layer, (*tint, 255), (x0, y0), (x1, y1), border)
    for dx, dy, particle_strength in bubble_particles(frame, window, radius):
        pygame.draw.circle(
            layer,
            (*color, round(255 * particle_strength)),
            (center + sign * dx, center + dy),
            ui.parry_particle_size_px / 2,
        )
    screen.blit(layer, (x_px - center, y_px - body_px / 2 - center))


def parry_frame_index(parry_frame: int, window: int, frame_count: int) -> int | None:
    """Return the shield frame for a parry, or None once the protected window is over.

    The shield rises and lowers within the protected window, stretched to it however far
    spamming shrank it. The recovery is not drawn as a parry, so the shield is only
    visible while the fighter can actually parry.
    """
    if not 0 < parry_frame <= window:
        return None
    progress = (parry_frame - 1) / (window - 1) if window > 1 else 0.5
    # A triangle that is steep enough to hold the fully raised frame at its peak.
    raised = min(1.0, 1.5 * (1 - abs(2 * progress - 1)))
    return round(raised * (frame_count - 1))


def deflect_frame_index(fighter, animation: Animation) -> int:
    """Return the attack frame for a parried fighter: the swing played backwards.

    It runs from the frame the swing had reached back to the start, stretched over
    the stagger, so the sword looks deflected for exactly as long as the fighter
    cannot attack.
    """
    start = min(fighter.stagger_from_frame // animation.frame_duration, len(animation.frames) - 1)
    stagger = config.game.combat.parried_stagger_frames
    return round(start * fighter.stagger_frames / stagger)


def draw_fighter(
    screen,
    fighter,
    tick,
    to_screen,
    turning=False,
    shade: tuple[int, int, int] | None = None,
    hurt: float = 0.0,
    effect_color: tuple[int, int, int] | None = None,
):
    """Draw a fighter animation and return whether sprite frames were available.

    ``effect_color`` is the fighter's side color for its parry bubble (the accent color
    when None).

    ``hurt`` (0 to 1) flashes the fighter: it is tinted toward the damage color and made
    partly transparent, by that fraction.
    """
    effect_color = effect_color or config.viewer.ui.colors["accent"]
    # After a successful parry the shield effects play on to the end of the window, even
    # while the fighter counters. Pressing parry again restarts them. The fighter's own
    # pose switches to the attack as soon as it swings.
    parry_frame = fighter.parry_frame or fighter.parry_pose_frame
    parry_visible = 0 < parry_frame <= fighter.parry_window
    deflected = False
    if parry_visible and not fighter.attack_frame:
        animation_name = "parry"
        animation_tick = 0
    elif fighter.attack_frame:
        animation_name = "attack"
        # A counter has a shorter wind-up: skip the start of the swing to match it.
        animation_tick = config.game.combat.normal_swing_frame(
            fighter.attack_frame, fighter.attack_type
        )
    elif fighter.stagger_frames:
        animation_name = "attack"
        animation_tick = 0
        deflected = True
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

    if animation_name == "parry":
        index = parry_frame_index(parry_frame, fighter.parry_window, len(animation.frames))
        img = animation.frames[index]
    elif deflected:
        img = animation.frames[deflect_frame_index(fighter, animation)]
    elif animation_name == "run" and fighter.vx < 0:
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
    # After the flip, which would not keep the surface opacity.
    if hurt > 0:
        ui = config.viewer.ui
        img = img.copy()
        tint = tuple(round(255 + (c - 255) * hurt) for c in ui.colors["damage_taken"])
        img.fill((*tint, 255), special_flags=pygame.BLEND_RGBA_MULT)
        img.set_alpha(round(255 - (255 - ui.hurt_flash_min_alpha) * hurt))
    x_px, y_px = to_screen(fighter.x, fighter.y)
    rect = img.get_rect(midbottom=(x_px, y_px))
    screen.blit(img, rect)
    if parry_visible:
        draw_parry_bubble(screen, fighter, x_px, y_px, parry_frame, effect_color)
    return True
