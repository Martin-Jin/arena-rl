"""Sound effect and music loading for the pygame viewer."""

from array import array
from dataclasses import dataclass
import math
from pathlib import Path
import random

import pygame

from arena_ai.config import AudioConfig
from arena_ai.sim.state import FighterState


ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets" / "audio"
EFFECT_ROOT = ASSET_ROOT / "effects"
MUSIC_ROOT = ASSET_ROOT / "music"
# Each fighter gets its own pair of channels for its looping footsteps.
MAX_FIGHTERS = 2
MIXER_CHANNELS = 16


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


@dataclass(frozen=True)
class FighterSnapshot:
    """The parts of a fighter's state that sound effects react to changing."""

    on_ground: bool
    attack_frame: int
    health: int
    parry_frame: int

    @classmethod
    def of(cls, fighter: FighterState) -> "FighterSnapshot":
        return cls(fighter.on_ground, fighter.attack_frame, fighter.health, fighter.parry_frame)


class _RunningLoop:
    """One fighter's looping footsteps, faded across two channels.

    Starting on the idle channel while the other fades out is what makes a
    direction change sound smooth instead of cutting off.
    """

    def __init__(
        self,
        sound: pygame.mixer.Sound,
        channels: tuple[pygame.mixer.Channel, pygame.mixer.Channel],
        fade_ms: int,
    ) -> None:
        self._sound = sound
        self._channels = channels
        self._fade_ms = fade_ms
        self._index = 0
        self.running = False
        self.direction = 0

    def update(self, moving: bool, move_dir: int) -> None:
        """Start, restart or fade out the loop to match the fighter's movement."""
        if moving and (not self.running or move_dir != self.direction):
            if self.running:
                self.stop()
            self._start(move_dir)
        elif not moving and self.running:
            self.stop()

    def _start(self, move_dir: int) -> None:
        inactive_index = 1 - self._index
        channel = self._channels[inactive_index]
        channel.stop()
        channel.play(self._sound, loops=-1, fade_ms=self._fade_ms)
        self._index = inactive_index
        self.running = True
        self.direction = move_dir

    def stop(self) -> None:
        """Fade the loop out."""
        self._channels[self._index].fadeout(self._fade_ms)
        self.running = False
        self.direction = 0

    def halt(self) -> None:
        """Cut the loop off immediately."""
        for channel in self._channels:
            channel.stop()
        self.running = False
        self.direction = 0


class AudioManager:
    """Load and play the viewer's music and sound effects for every fighter."""

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
        pygame.mixer.set_num_channels(MIXER_CHANNELS)
        pygame.mixer.set_reserved(2 * MAX_FIGHTERS)
        self.running_fade_ms = config.running_fade_ms
        self.match_end_fade_ms = config.match_end_fade_ms
        # Parries alternate between their sounds, starting with a random one.
        self._parry_sounds = sorted(name for name in self.effects if name.startswith("parry"))
        self._next_parry = random.randrange(len(self._parry_sounds))
        self._loops: dict[str, _RunningLoop] = {}
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
        """Fade out every running loop, for example when the game is paused."""
        for loop in self._loops.values():
            if loop.running:
                loop.stop()

    def cut_footsteps(self) -> None:
        """Stop every footstep loop at once, for the pause menu.

        ``silence`` only fades the active channel, so a loop still fading in or out
        when the game paused could keep playing under the menu.
        """
        for loop in self._loops.values():
            loop.halt()

    def cut_effects(self) -> None:
        """Stop every sound effect at once, footsteps included. The music keeps playing.

        Use it when a round ends or a new one starts. ``silence`` only fades out the
        footsteps, so a long one-shot such as a parry sound would otherwise carry on
        into the next round.
        """
        pygame.mixer.stop()
        for loop in self._loops.values():
            loop.halt()

    def start_music(self) -> None:
        """Start the battle music and loop it until stopped."""
        pygame.mixer.music.load(MUSIC_ROOT / self.music_filename)
        pygame.mixer.music.set_volume(self.music_volume)
        pygame.mixer.music.play(loops=-1)

    def ensure_music(self) -> None:
        """Start the battle music if it is not playing, for example after a match end cut it."""
        if not pygame.mixer.music.get_busy():
            self.start_music()

    def fade_out_effects(self) -> None:
        """Fade every sound effect that is still playing out over ``match_end_fade_ms``."""
        self.silence()
        pygame.mixer.fadeout(self.match_end_fade_ms)

    def stop_music(self) -> None:
        """Stop the battle music."""
        pygame.mixer.music.stop()

    def play(self, effect: str) -> None:
        """Play a one-shot effect by name."""
        self.effects[effect].play()

    def _loop_for(self, fighter_id: str) -> _RunningLoop:
        """Return the footstep loop for a fighter, giving it channels on first use."""
        if fighter_id not in self._loops:
            pair = len(self._loops)
            if pair >= MAX_FIGHTERS:
                raise ValueError(f"At most {MAX_FIGHTERS} fighters can have sound")
            channels = (
                pygame.mixer.Channel(2 * pair),
                pygame.mixer.Channel(2 * pair + 1),
            )
            self._loops[fighter_id] = _RunningLoop(
                self.effects["running"], channels, self.running_fade_ms
            )
        return self._loops[fighter_id]

    def update_fighter(
        self,
        fighter_id: str,
        fighter: FighterState,
        previous: FighterSnapshot,
    ) -> None:
        """Play the effects for what one fighter did during one simulation step.

        Call it for every fighter each step, with ``previous`` taken before the
        step, so the opponent's sounds play as well as the player's.
        """
        # TODO: Gate running audio on walkable support contact, not a
        # floor-specific flag, when platform collision state is available.
        # Footsteps follow the fighter's actual speed, not the key held: a fighter that is
        # parrying, or pressed against nothing, makes no steps.
        moving = fighter.vx != 0 and fighter.on_ground
        self._loop_for(fighter_id).update(moving, (fighter.vx > 0) - (fighter.vx < 0))

        if previous.attack_frame == 0 and fighter.attack_frame > 0:
            self.play("counter" if fighter.counter_started else "attack")
        if previous.parry_frame == 0 and fighter.parry_frame > 0:
            self.play("shield_raise")
        if previous.on_ground and not fighter.on_ground:
            self.play("jump")
        elif not previous.on_ground and fighter.on_ground:
            self.play("landing")
        if fighter.health < previous.health:
            self.play("hit")
        if fighter.parried:
            self.play(self._parry_sounds[self._next_parry])
            self._next_parry = (self._next_parry + 1) % len(self._parry_sounds)

    def close(self) -> None:
        """Stop active audio channels."""
        for loop in self._loops.values():
            loop.halt()
        self.stop_music()
