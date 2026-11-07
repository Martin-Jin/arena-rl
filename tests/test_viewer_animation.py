import math
import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from arena_ai.sim.state import FighterState
from arena_ai.config import load_config
from arena_ai.viewer.render import animation
from arena_ai.viewer.render.scene import draw_scene
from arena_ai.viewer.ui.hud import Hud
from arena_ai.viewer.ui.theme import Theme


class ParryAnimationTests(unittest.TestCase):
    def setUp(self) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))

    def tearDown(self) -> None:
        pygame.quit()

    def test_the_shield_shows_only_inside_the_protected_window(self) -> None:
        for window in (8, 3):
            frames = [animation.parry_frame_index(frame, window, 5) for frame in range(1, 25)]
            self.assertTrue(all(index is not None for index in frames[:window]))
            self.assertTrue(all(index is None for index in frames[window:]))
            # It rises to the fully raised frame and is lowered again by the last frame.
            self.assertEqual(max(frames[:window]), 4)
            self.assertEqual(frames[window - 1], 0)

    def test_turning_mid_attack_flips_the_attack_sprite(self) -> None:
        surface = pygame.Surface((200, 200), pygame.SRCALPHA)

        def draw(facing_dir: int) -> bytes:
            surface.fill((0, 0, 0, 0))
            fighter = FighterState(attack_frame=17, facing_dir=facing_dir)
            animation.draw_fighter(surface, fighter, 0, lambda x, y: (100, 150))
            return pygame.image.tobytes(surface, "RGBA")

        self.assertNotEqual(draw(1), draw(-1))

    def test_the_bubble_shows_only_while_the_parry_window_is_open(self) -> None:
        config = load_config()
        to_screen = lambda x, y: (100 + 100 * x, 150 - 100 * y)

        def scene(**state) -> bytes:
            surface = pygame.Surface((400, 200))
            fighters = {"agent": FighterState(x=1.0, **state)}
            draw_scene(surface, config, to_screen, fighters, 0)
            return pygame.image.tobytes(surface, "RGB")

        rested = scene()
        self.assertNotEqual(scene(parry_frame=3, parry_window=8), rested)
        recovering = scene(parry_frame=12, parry_window=8)
        self.assertEqual(recovering, scene())  # Recovery draws neither a shield nor a guard.

    def test_the_bubble_covers_the_side_the_fighter_faces(self) -> None:
        def painted_columns(facing_dir: int) -> tuple[int, int]:
            surface = pygame.Surface((400, 200), pygame.SRCALPHA)
            fighter = FighterState(facing_dir=facing_dir, parry_window=8)
            animation.draw_parry_bubble(surface, fighter, 200, 150, 1, (59, 110, 235))
            box = surface.get_bounding_rect()
            return box.left, box.right

        right_left, right_right = painted_columns(1)
        left_left, left_right = painted_columns(-1)
        self.assertGreater(right_right, 230)  # Extends to the side it faces...
        self.assertGreaterEqual(right_left, 180)  # ...and only slightly past the fighter.
        self.assertLess(left_left, 170)  # Mirrored for a fighter facing left.
        self.assertLessEqual(left_right, 220)

    def test_a_hurt_fighter_is_tinted_and_see_through_then_recovers(self) -> None:
        surface = pygame.Surface((200, 200), pygame.SRCALPHA)

        def draw(hurt: float) -> bytes:
            surface.fill((255, 255, 255, 255))
            fighter = FighterState(facing_dir=1)
            animation.draw_fighter(surface, fighter, 0, lambda x, y: (100, 150), hurt=hurt)
            return pygame.image.tobytes(surface, "RGBA")

        normal, flashing = draw(0.0), draw(1.0)
        self.assertNotEqual(normal, flashing)
        self.assertEqual(draw(0.0), normal)  # Nothing is left changed afterwards.

    def test_hurt_strength_rises_then_falls_over_the_flash(self) -> None:
        config = load_config()
        hud = Hud(config, Theme(config.viewer.ui), lambda x, y: (0, 0))
        fighters = {"agent": FighterState()}
        hud.reset(fighters)
        self.assertEqual(hud.hurt_strength("agent"), 0.0)
        fighters["agent"].health -= 10
        seconds = config.viewer.ui.hurt_flash_seconds
        hud.update(0.0, fighters)
        hud.update(seconds / 2, fighters)
        self.assertAlmostEqual(hud.hurt_strength("agent"), 1.0)
        hud.update(seconds, fighters)
        self.assertEqual(hud.hurt_strength("agent"), 0.0)

    def test_the_bubble_is_a_little_more_than_a_half_circle_and_burns_from_the_ends(self) -> None:
        ui = load_config().viewer.ui
        first = animation.bubble_span_degrees(1, 8)
        last = animation.bubble_span_degrees(8, 8)
        self.assertGreater(first, 180)
        self.assertAlmostEqual(last, first * (1 - ui.parry_bubble_burn))
        spans = [animation.bubble_span_degrees(frame, 8) for frame in range(1, 9)]
        self.assertEqual(spans, sorted(spans, reverse=True))

    def test_the_bubble_edge_is_uneven_and_flickers(self) -> None:
        radius = 60.0
        edge = animation.bubble_outline(4, 8, radius)
        distances = [math.hypot(dx, dy) for dx, dy, _, _ in edge]
        self.assertGreater(max(distances) - min(distances), 0.05 * radius)  # Not a clean arc.
        self.assertNotEqual(edge, animation.bubble_outline(5, 8, radius))  # It moves.
        self.assertEqual(edge, animation.bubble_outline(4, 8, radius))  # But is repeatable.
        # The ends are burnt more unevenly than the middle.
        def spread(points):
            values = [math.hypot(dx, dy) for dx, dy, _, _ in points]
            return max(values) - min(values)
        ends = edge[:6] + edge[-6:]
        middle = edge[17:24]
        self.assertGreater(spread(ends), spread(middle))

    def test_the_opponents_bubble_is_drawn_in_its_own_color(self) -> None:
        def painted(color) -> bytes:
            surface = pygame.Surface((400, 200), pygame.SRCALPHA)
            fighter = FighterState(facing_dir=1, parry_window=8)
            animation.draw_parry_bubble(surface, fighter, 200, 150, 3, color)
            return pygame.image.tobytes(surface, "RGBA")

        self.assertNotEqual(painted((59, 110, 235)), painted((222, 76, 76)))

    def test_particles_come_from_the_ends_and_flow_backward(self) -> None:
        self.assertEqual(animation.bubble_particles(1, 8, 60.0)[0][2], 1.0)  # Just born.
        radius = 60.0
        later = animation.bubble_particles(8, 8, radius)
        self.assertGreater(len(later), len(animation.bubble_particles(1, 8, radius)))
        # Some have flowed past the middle line to the back of the fighter (dx < 0), and
        # older ones are fainter than new ones.
        self.assertTrue(any(dx < 0 for dx, _, _ in later))
        strengths = {round(strength, 3) for _, _, strength in later}
        self.assertGreater(len(strengths), 1)
        self.assertEqual(animation.bubble_particles(8, 8, radius), later)  # Repeatable.

    def test_a_parry_adds_a_flash_that_expires(self) -> None:
        config = load_config()
        hud = Hud(config, Theme(config.viewer.ui), lambda x, y: (100 + 100 * x, 150 - 100 * y))
        fighters = {"agent": FighterState(x=1.0)}
        hud.reset(fighters)
        hud.parried("agent", fighters["agent"])
        surface = pygame.Surface((400, 200))
        surface.fill((0, 0, 0))
        hud.update(0.05, fighters)
        hud.draw(surface, fighters, {"agent": "You"}, 0)
        self.assertNotEqual(pygame.image.tobytes(surface, "RGB"), bytes(400 * 200 * 3))
        hud.update(config.viewer.ui.parry_flash_seconds, fighters)
        self.assertEqual(hud._flashes, [])

    def _counter_hud(self):
        config = load_config()
        hud = Hud(config, Theme(config.viewer.ui), lambda x, y: (100 + 100 * x, 150 - 100 * y))
        fighters = {
            "agent": FighterState(x=1.0, facing_dir=1, attack_frame=1, attack_counter=True),
            "opponent": FighterState(x=2.0, facing_dir=-1),
        }
        hud.reset(fighters)
        hud.countered("agent", fighters["agent"])
        return config, hud, fighters

    def test_effects_use_the_color_of_the_fighters_side(self) -> None:
        config, hud, fighters = self._counter_hud()
        hud.countered("opponent", fighters["opponent"])
        hud.parried("opponent", fighters["opponent"])
        opponent = config.viewer.ui.colors["opponent_health"]
        accent = config.viewer.ui.colors["accent"]
        self.assertEqual(hud._slashes[0].color, accent)  # The agent's counter.
        self.assertEqual(hud._slashes[1].color, opponent)
        self.assertEqual(hud._flashes[0].color, opponent)
        self.assertTrue(all(f.color == opponent for f in hud._floaters[-2:]))

    def test_the_counter_line_follows_the_fighter(self) -> None:
        _, hud, fighters = self._counter_hud()
        fighters["agent"].x = 1.6
        hud.update(0.02, fighters)
        slash = hud._slashes[0]
        self.assertEqual((slash.start_x, slash.end_x, slash.ended_for), (1.0, 1.6, None))

    def test_the_counter_line_carries_through_the_opponent_on_a_hit(self) -> None:
        config, hud, fighters = self._counter_hud()
        fighters["agent"].x = 1.9
        fighters["opponent"].health -= 10
        hud.update(0.02, fighters)
        slash = hud._slashes[0]
        carry = config.viewer.ui.counter_line_carry_m
        self.assertAlmostEqual(slash.end_x, 2.0 + carry)
        self.assertEqual(slash.ended_for, 0.0)

    def test_the_counter_line_stops_at_the_parrying_fighter(self) -> None:
        config, hud, fighters = self._counter_hud()
        fighters["agent"].x = 1.8
        fighters["agent"].attack_frame = 0  # A parry cancels the swing and staggers it.
        fighters["agent"].attack_counter = False
        fighters["agent"].stagger_frames = 20
        hud.update(0.02, fighters)
        slash = hud._slashes[0]
        edge = 2.0 - config.game.fighter.width_m / 2
        self.assertAlmostEqual(slash.end_x, edge)  # Not carried through.
        self.assertEqual(slash.ended_for, 0.0)

    def test_the_counter_line_is_drawn_and_then_fades_away(self) -> None:
        config, hud, fighters = self._counter_hud()
        fighters["agent"].x = 1.5
        surface = pygame.Surface((400, 200))
        surface.fill((0, 0, 0))
        hud.update(0.02, fighters)
        hud.draw(surface, fighters, {"agent": "You", "opponent": "Bot"}, 0)
        painted = surface.get_bounding_rect()
        self.assertGreater(painted.width, 0)
        fighters["agent"].attack_counter = False  # The swing ended without a hit.
        hud.update(0.02, fighters)
        hud.update(config.viewer.ui.counter_effect_seconds, fighters)
        self.assertEqual(hud._slashes, [])

    def test_the_shield_bubble_stays_while_the_fighter_counters(self) -> None:
        surface = pygame.Surface((400, 300), pygame.SRCALPHA)

        def draw(fighter: FighterState) -> bytes:
            surface.fill((0, 0, 0, 0))
            animation.draw_fighter(surface, fighter, 0, lambda x, y: (200, 250))
            return pygame.image.tobytes(surface, "RGBA")

        countering = FighterState(attack_frame=3, attack_counter=True)
        with_bubble = FighterState(
            attack_frame=3, attack_counter=True, parry_pose_frame=3, parry_window=8
        )
        self.assertNotEqual(draw(with_bubble), draw(countering))  # The bubble is still drawn.
        # The sprite is the attack pose, not the shield pose.
        self.assertNotEqual(draw(countering), draw(FighterState(parry_frame=3, parry_window=8)))

    def test_a_fighter_in_parry_recovery_is_not_drawn_with_the_shield(self) -> None:
        surface = pygame.Surface((200, 200), pygame.SRCALPHA)

        def draw(fighter: FighterState) -> bytes:
            surface.fill((0, 0, 0, 0))
            animation.draw_fighter(surface, fighter, 0, lambda x, y: (100, 150))
            return pygame.image.tobytes(surface, "RGBA")

        recovering = FighterState(parry_frame=12, parry_window=8)
        self.assertEqual(draw(recovering), draw(FighterState()))
        self.assertNotEqual(draw(FighterState(parry_frame=4, parry_window=8)), draw(FighterState()))


if __name__ == "__main__":
    unittest.main()
