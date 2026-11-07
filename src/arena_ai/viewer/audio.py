"""Sound effect and music loading for the pygame viewer."""

from array import array
import math
from pathlib import Path

import pygame

from arena_ai.config import AudioConfig
from arena_ai.sim.state import FighterState


ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets" / "audio"
EFFECT_ROOT = ASSET_ROOT / "effects"
MUSIC_ROOT = ASSET_ROOT / "music"


def _normalize_sound(
    sound: pygame.mixer.Sound,
    target_rms: float,
    peak_ceiling: float,
) -> pygame.mixer.Sound:
    """Return a normalized copy of a decoded sound without changing its file."""
    mixer_info = pygame.mixer.get_init()
    if mixer_info is None:
        raise RuntimeError("The pygame mixer must be initialized before loading audio")
    _, mixer_format, _ = mixer_info
    if mixer_format not in (-8, -16, -32):
        raise RuntimeError(f"Unsupported pygame mixer format: {mixer_format}")

    sample_width = abs(mixer_format) // 8
    typecode = {1: "b", 2: "h", 4: "i"}[sample_width]
    samples = array("B" if mixer_format == -8 else typecode)
    samples.frombytes(sound.get_raw())
    if not samples:
        return sound
    if mixer_format == -8:
        samples = array("b", (value - 128 for value in samples))

    maximum = (1 << (sample_width * 8 - 1)) - 1
    peak = max(abs(value) for value in samples) / maximum
    rms = math.sqrt(sum(value * value for value in samples) / len(samples)) / maximum
    if rms == 0.0:
        return sound

    gain = min(target_rms / rms, peak_ceiling / peak) if peak else 1.0
    normalized = array(
        typecode,
        (
            max(-maximum, min(maximum, round(value * gain)))
            for value in samples
        ),
    )
    if mixer_format == -8:
        normalized = array("B", (value + 128 for value in normalized))
    return pygame.mixer.Sound(buffer=normalized.tobytes())


class AudioManager:
    """Load and play the viewer's music and fighter sound effects."""

    def __init__(self, config: AudioConfig) -> None:
        self.effects = {
            name: _normalize_sound(
                pygame.mixer.Sound(EFFECT_ROOT / filename),
                config.normalization_rms,
                config.normalization_peak,
            )
            for name, filename in config.effects.items()
        }
        self.set_effects_volume(config.effects_volume)
        pygame.mixer.set_reserved(2)
        self.running_channels = (
            pygame.mixer.Channel(0),
            pygame.mixer.Channel(1),
        )
        self.running_channel_index = 0
        self.running_fade_ms = config.running_fade_ms
        self._running = False
        self._running_direction = 0
        self.music_volume = config.music_volume
        self.music_filename = config.music

    def set_effects_volume(self, volume: float) -> None:
        """Set the master volume of every sound effect (0 to 1)."""
        self.effects_volume = volume
        for effect in self.effects.values():
            effect.set_volume(volume)

    def set_music_volume(self, volume: float) -> None:
        """Set the music volume (0 to 1), including music that is playing."""
        self.music_volume = volume
        pygame.mixer.music.set_volume(volume)

    def silence(self) -> None:
        """Fade out the running loop, for example when the game is paused."""
        if self._running:
            self._stop_running()

    def start_music(self) -> None:
        """Start the battle music and loop it until stopped."""
        pygame.mixer.music.load(MUSIC_ROOT / self.music_filename)
        pygame.mixer.music.set_volume(self.music_volume)
        pygame.mixer.music.play(loops=-1)

    def stop_music(self) -> None:
        """Stop the battle music."""
        pygame.mixer.music.stop()

    def play(self, effect: str) -> None:
        """Play a one-shot effect by name."""
        self.effects[effect].play()

    def _start_running(self, move_dir: int) -> None:
        """Start the running loop with a short fade-in."""
        inactive_index = 1 - self.running_channel_index
        inactive_channel = self.running_channels[inactive_index]
        inactive_channel.stop()
        inactive_channel.play(
            self.effects["running"],
            loops=-1,
            fade_ms=self.running_fade_ms,
        )
        self.running_channel_index = inactive_index
        self._running = True
        self._running_direction = move_dir

    def _stop_running(self) -> None:
        """Fade out the active running loop."""
        self.running_channels[self.running_channel_index].fadeout(
            self.running_fade_ms
        )
        self._running = False
        self._running_direction = 0

    def update_fighter(
        self,
        fighter: FighterState,
        move_dir: int,
        was_on_ground: bool,
        previous_attack_frame: int,
    ) -> None:
        """Play effects for transitions observed during one simulation step."""
        # TODO: Gate running audio on walkable support contact, not a
        # floor-specific flag, when platform collision state is available.
        moving = move_dir != 0 and fighter.on_ground
        if moving and (
            not self._running or move_dir != self._running_direction
        ):
            if self._running:
                self._stop_running()
            self._start_running(move_dir)
        elif not moving and self._running:
            self._stop_running()

        if previous_attack_frame == 0 and fighter.attack_frame > 0:
            self.play("attack")
        if was_on_ground and not fighter.on_ground:
            self.play("jump")
        elif not was_on_ground and fighter.on_ground:
            self.play("landing")

    def close(self) -> None:
        """Stop active audio channels."""
        for channel in self.running_channels:
            channel.stop()
        self.stop_music()
