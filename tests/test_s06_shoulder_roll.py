"""S06 — shoulder roll: shoulders-only visibility + vertical motion over time."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posture_detector import validate_stretch_form, _reset_s06_roll_state


class _Lm:
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def _shoulders_only_pose(ly=0.36, ry=0.36, x_shift=0.0):
    lm = [_Lm(visibility=0.0) for _ in range(33)]
    lm[11] = _Lm(0.44 + x_shift, ly, visibility=1.0)
    lm[12] = _Lm(0.56 + x_shift, ry, visibility=1.0)
    return lm


class TestS06ShoulderRoll(unittest.TestCase):
    def setUp(self):
        _reset_s06_roll_state()

    def test_no_elbows_required(self):
        lm = _shoulders_only_pose()
        for _ in range(12):
            ok, fb = validate_stretch_form("S06", lm)
        self.assertFalse(ok)
        self.assertGreater(len(fb), 0)

    def test_static_shoulders_never_green(self):
        lm = _shoulders_only_pose(0.36, 0.36)
        for _ in range(30):
            ok, _ = validate_stretch_form("S06", lm)
        self.assertFalse(ok)

    def test_vertical_motion_eventually_green(self):
        ok = False
        for i in range(20):
            dy = 0.022 * (i % 5)
            lm = _shoulders_only_pose(0.36 + dy, 0.36 + dy)
            ok, fb = validate_stretch_form("S06", lm)
            if ok:
                self.assertGreater(len(fb), 0)
                break
        self.assertTrue(ok)

    def test_missing_shoulder_fails(self):
        lm = _shoulders_only_pose()
        lm[11] = _Lm(0.44, 0.36, visibility=0.1)
        ok, fb = validate_stretch_form("S06", lm)
        self.assertFalse(ok)
        self.assertIn("shoulder", fb.lower())

    def test_side_to_side_body_turn_rejects_despite_vertical_motion(self):
        """Large mid-shoulder x drift (turning) must not go green even if y oscillates."""
        last_fb = ""
        for i in range(25):
            x_shift = 0.13 if (i % 2 == 0) else 0.0
            dy = 0.022 * (i % 5)
            lm = _shoulders_only_pose(0.36 + dy, 0.36 + dy, x_shift=x_shift)
            ok, last_fb = validate_stretch_form("S06", lm)
            self.assertFalse(ok, f"frame {i}: should reject horizontal sway; got ok={ok!r} {last_fb!r}")
        self.assertIn("facing", last_fb.lower())


if __name__ == "__main__":
    unittest.main()
