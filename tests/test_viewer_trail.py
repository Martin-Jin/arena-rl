import unittest

from arena_ai.config import load_config
from arena_ai.sim.state import FighterState
from arena_ai.viewer.render.trail import DodgeTrail

CONFIG = load_config()
UI = CONFIG.viewer.ui


class DodgeTrailTests(unittest.TestCase):
    def test_a_dodge_leaves_a_copy_that_fades_out(self) -> None:
        trail = DodgeTrail(CONFIG)
        fighters = {"agent": FighterState(dodge_frames_left=3)}
        trail.update(0.016, fighters)
        self.assertEqual(len(trail.ghosts()), 1)
        self.assertEqual(trail.ghosts()[0][2], 1.0)
        fighters["agent"].dodge_frames_left = 0
        trail.update(UI.dodge_trail_seconds + 0.01, fighters)
        self.assertEqual(trail.ghosts(), [])

    def test_no_dodge_leaves_nothing(self) -> None:
        trail = DodgeTrail(CONFIG)
        trail.update(1.0, {"agent": FighterState()})
        self.assertEqual(trail.ghosts(), [])


if __name__ == "__main__":
    unittest.main()
