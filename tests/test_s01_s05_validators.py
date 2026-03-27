"""
tests/test_s01_s05_validators.py
---------------------------------
Per-stretch tests for S01-S05 (neck group).

Each test provides:
  - A CORRECT-form landmark set  → validator must return (True, non-empty str)
  - An INCORRECT-form landmark set → validator must return (False, non-empty str)

Landmark coordinate system: MediaPipe normalised [0, 1], y increases downward.
Key indices used here:
  0  nose        7  left ear       8  right ear
  2  left eye    5  right eye
  11 left shoulder  12 right shoulder
  13 left elbow     14 right elbow
  15 left wrist     16 right wrist
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posture_detector import validate_stretch_form


class _Lm:
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def _neutral_landmarks():
    """
    Anatomically-plausible sitting pose, all landmarks visible.
    All 33 slots initialised to (0.5, 0.5) then key points overridden.
    """
    lm = [_Lm() for _ in range(33)]
    # Head
    lm[0]  = _Lm(0.50, 0.22)          # nose
    lm[2]  = _Lm(0.47, 0.20)          # left eye
    lm[5]  = _Lm(0.53, 0.20)          # right eye
    lm[7]  = _Lm(0.45, 0.22)          # left ear  – ear-to-shoulder y-dist = 0.14
    lm[8]  = _Lm(0.55, 0.22)          # right ear – ear-to-shoulder y-dist = 0.14
    # Shoulders / arms
    lm[11] = _Lm(0.44, 0.36)
    lm[12] = _Lm(0.56, 0.36)
    lm[13] = _Lm(0.40, 0.48)
    lm[14] = _Lm(0.60, 0.48)
    lm[15] = _Lm(0.37, 0.58)
    lm[16] = _Lm(0.63, 0.58)
    lm[19] = _Lm(0.35, 0.60)
    lm[20] = _Lm(0.65, 0.60)
    # Hips / knees (needed for full-body validators in later tests)
    lm[23] = _Lm(0.46, 0.62)
    lm[24] = _Lm(0.54, 0.62)
    lm[25] = _Lm(0.46, 0.82)
    lm[26] = _Lm(0.54, 0.82)
    return lm


def _clone(lm):
    """Return a shallow copy of the landmark list so tests don't interfere."""
    return [_Lm(p.x, p.y, p.z, p.visibility) for p in lm]


class TestS01NeckSideTilt(unittest.TestCase):
    """
    S01 – Neck Side Tilt (lateral cervical flexion).

    Science: A therapeutic lateral flexion of ~20-30° closes the vertical
    ear-to-shoulder gap to ≈ 0.07-0.09 in normalised coords.
    Threshold: ear-to-shoulder y-distance < 0.10
    Before fix: < 0.18 → neutral pose (dist = 0.14) passed as correct (false positive).
    After fix:  < 0.10 → neutral fails; a genuine tilt (dist ≈ 0.08) passes.
    """

    def test_correct_form_left_tilt(self):
        lm = _clone(_neutral_landmarks())
        # Left ear drops toward left shoulder: y 0.22 → 0.28
        # ear-to-shoulder dist = |0.28 - 0.36| = 0.08  < 0.10  ✓
        lm[7] = _Lm(0.45, 0.28)
        ok, feedback = validate_stretch_form("S01", lm)
        self.assertTrue(ok, f"Expected correct form; got feedback: {feedback!r}")
        self.assertIsInstance(feedback, str)
        self.assertGreater(len(feedback), 0)

    def test_incorrect_form_neutral(self):
        lm = _neutral_landmarks()
        # ear-to-shoulder dist = 0.14  > 0.10  ✗
        ok, feedback = validate_stretch_form("S01", lm)
        self.assertFalse(ok, "Neutral pose should NOT pass as a neck side tilt.")
        self.assertGreater(len(feedback), 0)


class TestS02ChinTuck(unittest.TestCase):
    """
    S02 – Chin Tuck (cervical retraction).

    Science: A chin tuck involves ~10° of capital flexion, moving the nose
    down by ≈ 0.02-0.04 normalised units relative to the eye midpoint.
    Threshold: nose_y - eye_mid_y > 0.04
    Before: > 0.05 was too strict for a subtle but visible tuck.
    After:  > 0.04 catches a clinically appropriate tuck while neutral (0.02) still fails.
    """

    def test_correct_form_chin_tucked(self):
        lm = _clone(_neutral_landmarks())
        # Nose drops from 0.22 → 0.25; eye_mid stays at 0.20
        # delta = 0.25 - 0.20 = 0.05  > 0.04  ✓
        lm[0] = _Lm(0.50, 0.25)
        ok, feedback = validate_stretch_form("S02", lm)
        self.assertTrue(ok, f"Expected correct form; got feedback: {feedback!r}")
        self.assertGreater(len(feedback), 0)

    def test_incorrect_form_neutral(self):
        lm = _neutral_landmarks()
        # delta = 0.22 - 0.20 = 0.02  < 0.04  ✗
        ok, feedback = validate_stretch_form("S02", lm)
        self.assertFalse(ok, "Neutral pose should NOT pass as a chin tuck.")
        self.assertGreater(len(feedback), 0)


class TestS03HeadRotation(unittest.TestCase):
    """
    S03 – Head Rotation (cervical axial rotation).

    Science: Full therapeutic cervical rotation is ~45°. In a frontal-facing
    normalised frame this shifts the nose ≈ 0.09-0.12 off the shoulder midline.
    Threshold: |nose_x - shoulder_mid_x| > 0.09
    Before: > 0.06 triggered on minor asymmetry / camera angle variation.
    After:  > 0.09 requires a deliberate, visible rotation.
    """

    def test_correct_form_rotated_left(self):
        lm = _clone(_neutral_landmarks())
        # Nose shifts left: 0.50 → 0.40; shoulder midpoint stays at 0.50
        # offset = |0.40 - 0.50| = 0.10  > 0.09  ✓
        lm[0] = _Lm(0.40, 0.22)
        ok, feedback = validate_stretch_form("S03", lm)
        self.assertTrue(ok, f"Expected correct form; got feedback: {feedback!r}")
        self.assertGreater(len(feedback), 0)

    def test_incorrect_form_neutral(self):
        lm = _neutral_landmarks()
        # offset = |0.50 - 0.50| = 0.00  < 0.09  ✗
        ok, feedback = validate_stretch_form("S03", lm)
        self.assertFalse(ok, "Neutral pose should NOT pass as a head rotation.")
        self.assertGreater(len(feedback), 0)

    def test_incorrect_form_small_offset(self):
        lm = _clone(_neutral_landmarks())
        # Nose barely off-centre (e.g. camera angle artefact): offset = 0.07
        # Should still fail under the tightened threshold.
        lm[0] = _Lm(0.43, 0.22)   # |0.43 - 0.50| = 0.07  < 0.09  ✗
        ok, feedback = validate_stretch_form("S03", lm)
        self.assertFalse(ok, "Small asymmetry (0.07) should not trigger S03.")
        self.assertGreater(len(feedback), 0)


class TestS04HeadTiltShouldersDown(unittest.TestCase):
    """
    S04 – Levator Scapulae Stretch (tilt + shoulders depressed).

    Science: Combines S01's lateral tilt with the requirement that neither
    shoulder shrugs upward. Shoulder Y delta < 0.08 ensures level shoulders.
    Threshold: ear-to-shoulder y-dist < 0.10 AND shoulder_delta < 0.08
    Before fix: ear threshold was 0.18 → neutral (0.14) passed as correct.
    After fix:  < 0.10 eliminates the false positive.
    """

    def test_correct_form_tilt_with_level_shoulders(self):
        lm = _clone(_neutral_landmarks())
        # Left ear tilts toward left shoulder (same geometry as S01 correct)
        # ear-to-shoulder dist = 0.08  < 0.10 ✓
        # Shoulders stay level: shoulder_delta = 0.0  < 0.08 ✓
        lm[7] = _Lm(0.45, 0.28)
        ok, feedback = validate_stretch_form("S04", lm)
        self.assertTrue(ok, f"Expected correct form; got feedback: {feedback!r}")
        self.assertGreater(len(feedback), 0)

    def test_incorrect_form_neutral(self):
        lm = _neutral_landmarks()
        # ear-to-shoulder dist = 0.14  > 0.10  ✗
        ok, feedback = validate_stretch_form("S04", lm)
        self.assertFalse(ok, "Neutral pose should NOT pass as S04.")
        self.assertGreater(len(feedback), 0)

    def test_incorrect_form_tilt_with_shrugged_shoulder(self):
        lm = _clone(_neutral_landmarks())
        # Left ear close to shoulder but left shoulder is clearly shrugged up
        lm[7]  = _Lm(0.45, 0.28)   # ear tilted — ear-to-shoulder dist passes
        lm[11] = _Lm(0.44, 0.26)   # left shoulder shrugged high
        # shoulder_delta = |0.26 - 0.36| = 0.10  > 0.08  → fails shoulder check ✗
        ok, feedback = validate_stretch_form("S04", lm)
        self.assertFalse(ok, "Tilted head with shrugged shoulder should not pass S04.")
        self.assertGreater(len(feedback), 0)


class TestS05NeckRoll(unittest.TestCase):
    """
    S05 – Neck Roll Arc.

    Science: A neck roll passes through chin-to-chest (forward bow) and lateral
    tilt. Two detectable positions are used as proxies:
      (a) Forward bow: nose-to-shoulder y-dist < 0.12 (head bowed forward)
      (b) Lateral position: nose x-offset from shoulder midline > 0.10
    Threshold: condition (a) OR condition (b)
    Before fix: < 0.35 passed trivially for every pose including neutral (dist=0.14).
    After fix:  neutral (y-dist=0.14, x-offset=0.0) correctly fails both conditions.
    """

    def test_correct_form_forward_bow(self):
        lm = _clone(_neutral_landmarks())
        # Nose drops toward shoulder midpoint: y 0.22 → 0.28
        # y-dist = |0.28 - 0.36| = 0.08  < 0.12  ✓  (chin-to-chest arc position)
        lm[0] = _Lm(0.50, 0.28)
        ok, feedback = validate_stretch_form("S05", lm)
        self.assertTrue(ok, f"Forward bow should pass S05; got: {feedback!r}")
        self.assertGreater(len(feedback), 0)

    def test_correct_form_lateral_tilt(self):
        lm = _clone(_neutral_landmarks())
        # Nose shifts to the side during the arc: x-offset = 0.11  > 0.10  ✓
        lm[0] = _Lm(0.39, 0.22)
        ok, feedback = validate_stretch_form("S05", lm)
        self.assertTrue(ok, f"Lateral tilt should pass S05; got: {feedback!r}")
        self.assertGreater(len(feedback), 0)

    def test_incorrect_form_neutral(self):
        lm = _neutral_landmarks()
        # y-dist = 0.14  NOT < 0.12
        # x-offset = 0.0  NOT > 0.10
        # Both conditions fail ✗
        ok, feedback = validate_stretch_form("S05", lm)
        self.assertFalse(ok, "Neutral pose should NOT pass as S05 (was broken before fix).")
        self.assertGreater(len(feedback), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
