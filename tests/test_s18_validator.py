"""
S18 — Seated Ankle Mobility: lifted foot + visible ankle rotation (angle variation).
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posture_detector import validate_stretch_form, _reset_s18_ankle_state


class _Lm:
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def _base_seated_lower_body():
    lm = [_Lm() for _ in range(33)]
    lm[0] = _Lm(0.5, 0.22)
    lm[11] = _Lm(0.44, 0.36)
    lm[12] = _Lm(0.56, 0.36)
    lm[23] = _Lm(0.46, 0.62)
    lm[24] = _Lm(0.54, 0.62)
    lm[25] = _Lm(0.46, 0.82)
    lm[26] = _Lm(0.54, 0.82)
    lm[27] = _Lm(0.46, 0.93)
    lm[28] = _Lm(0.54, 0.93)
    lm[29] = _Lm(0.46, 0.97)
    lm[30] = _Lm(0.54, 0.97)
    lm[31] = _Lm(0.44, 0.99)
    lm[32] = _Lm(0.56, 0.99)
    return lm


def _clone(lm):
    return [_Lm(p.x, p.y, p.z, p.visibility) for p in lm]


class TestS18SeatedAnkleMobility(unittest.TestCase):
    def setUp(self):
        _reset_s18_ankle_state()

    def test_static_lift_never_green_without_rotation(self):
        lm = _clone(_base_seated_lower_body())
        lm[27] = _Lm(0.46, 0.88)
        lm[28] = _Lm(0.54, 0.93)
        for _ in range(25):
            ok, _ = validate_stretch_form("S18", lm)
        self.assertFalse(ok, "Same pose repeatedly should not reach green without angle span.")

    def test_rotation_sequence_eventually_green(self):
        lm = _clone(_base_seated_lower_body())
        lm[27] = _Lm(0.46, 0.88)
        lm[28] = _Lm(0.54, 0.93)
        ok = False
        for i in range(24):
            lm[31] = _Lm(0.32 + 0.018 * i, 0.88 + 0.012 * (i % 5))
            ok, feedback = validate_stretch_form("S18", lm)
            if ok:
                self.assertGreater(len(feedback), 0)
                break
        self.assertTrue(ok, "Varying toe position should produce enough angle span for green.")

    def test_incorrect_both_feet_down(self):
        lm = _base_seated_lower_body()
        ok, feedback = validate_stretch_form("S18", lm)
        self.assertFalse(ok)
        self.assertGreater(len(feedback), 0)

    def test_incorrect_standing_straight_legs(self):
        lm = _clone(_base_seated_lower_body())
        lm[25] = _Lm(0.46, 0.55)
        lm[26] = _Lm(0.54, 0.55)
        lm[27] = _Lm(0.46, 0.93)
        lm[28] = _Lm(0.54, 0.88)
        ok, _ = validate_stretch_form("S18", lm)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
