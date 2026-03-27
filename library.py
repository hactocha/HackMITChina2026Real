import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posture_detector import (
    validate_stretch_form,
    start_routine,
    stop_routine,
    get_routine_status,
)


class _Lm:
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def _neutral_landmarks():
    lm = [_Lm() for _ in range(33)]
    # Head
    lm[0] = _Lm(0.5, 0.22, -0.03)
    lm[2] = _Lm(0.47, 0.20, -0.02)
    lm[5] = _Lm(0.53, 0.20, -0.02)
    lm[7] = _Lm(0.45, 0.22, -0.02)
    lm[8] = _Lm(0.55, 0.22, -0.02)
    # Shoulders / arms
    lm[11] = _Lm(0.44, 0.36, 0.00)
    lm[12] = _Lm(0.56, 0.36, 0.00)
    lm[13] = _Lm(0.40, 0.48, 0.00)
    lm[14] = _Lm(0.60, 0.48, 0.00)
    lm[15] = _Lm(0.37, 0.58, 0.00)
    lm[16] = _Lm(0.63, 0.58, 0.00)
    lm[19] = _Lm(0.35, 0.60, 0.00)
    lm[20] = _Lm(0.65, 0.60, 0.00)
    # Hips / knees
    lm[23] = _Lm(0.46, 0.62, 0.01)
    lm[24] = _Lm(0.54, 0.62, 0.01)
    lm[25] = _Lm(0.46, 0.82, 0.02)
    lm[26] = _Lm(0.54, 0.82, 0.02)
    return lm


class RoutineValidatorSmokeTests(unittest.TestCase):
    def test_all_stretch_validators_return_bool_and_feedback(self):
        lm = _neutral_landmarks()
        for i in range(1, 21):
            sid = f"S{i:02d}"
            is_correct, feedback = validate_stretch_form(sid, lm)
            self.assertIsInstance(is_correct, bool, sid)
            self.assertIsInstance(feedback, str, sid)
            self.assertGreater(len(feedback), 0, sid)

    def test_routine_start_and_stop_state(self):
        stretches = [{"id": f"S{i:02d}", "duration_seconds": 5} for i in range(1, 6)]
        start_routine(stretches)
        status = get_routine_status()
        self.assertEqual(status["state"], "running")
        self.assertEqual(status["total"], 5)
        self.assertEqual(status["current_stretch"]["id"], "S01")

        stop_routine()
        status = get_routine_status()
        self.assertEqual(status["state"], "inactive")
        self.assertEqual(status["total"], 0)


if __name__ == "__main__":
    unittest.main()
