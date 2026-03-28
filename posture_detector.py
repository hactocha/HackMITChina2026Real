"""
posture_detector.py
--------------------
Real-time posture analysis using MediaPipe Pose + OpenCV.

Detects:
  - Forward Head Tilt  (FORWARD_HEAD)
  - Rounded Shoulders  (ROUNDED_SHOULDERS)
  - Slouching          (SPINE_CURVE)
  - Wrist Strain Risk  (WRIST_POSITION sustained 30+ frames)

Exposes:
  - current_posture        : module-level str (thread-safe via _posture_lock)
  - get_current_posture()  : returns current_posture
  - generate_frames()      : yields MJPEG-encoded frames for Flask streaming

Architecture: a single background camera thread reads from the webcam and
broadcasts to all Flask clients via a shared latest-frame buffer. This
prevents multiple VideoCapture instances competing for the same camera.

Run standalone:
  python posture_detector.py

Install deps:
  pip install mediapipe opencv-python numpy
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import platform
import threading
import time
import numpy as np
import cv2
import mediapipe as mp


def _camera_device_index() -> int:
    try:
        return int(os.environ.get("CAMERA_INDEX", "0"))
    except ValueError:
        return 0


def _video_capture(index: int = 0) -> cv2.VideoCapture:
    """OpenCV capture with a backend that works reliably on macOS (often fixes all-black frames)."""
    if platform.system() == "Darwin":
        # AVFoundation avoids many cases where the default backend opens but returns black frames.
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index)


def _warmup_capture(cap: cv2.VideoCapture, n: int = 15) -> None:
    """Discard initial frames; built-in / Continuity cameras often need a moment before valid pixels."""
    for _ in range(n):
        cap.read()

# ── MediaPipe setup ───────────────────────────────────────────────────────────
_mp_pose = mp.solutions.pose
_mp_draw = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles

# ── Module-level shared state ─────────────────────────────────────────────────
current_posture: str = "Good Posture"
_posture_lock = threading.Lock()

# Shared latest JPEG frame — written by camera thread, read by Flask clients
_latest_frame: bytes = b""
_frame_lock = threading.Lock()
_frame_event = threading.Event()   # signals that a new frame is ready

# Camera thread lifecycle
_camera_thread: threading.Thread | None = None
_camera_running = False

# Rolling counter: consecutive frames where a wrist is raised above its elbow
_wrist_high_frames: int = 0

# Smoothing: last N posture labels — output only changes when majority agrees
_SMOOTH_WINDOW = 7
_label_history: list = []

# Guided routine state (backend-authoritative)
_routine_lock = threading.Lock()
_routine_state = {
    "state": "inactive",                 # inactive | running | finished
    "stretches": [],
    "current_index": 0,
    "is_correct_now": False,
    "hold_elapsed_s": 0.0,
    "hold_target_s": 0.0,
    "last_feedback_text": "No routine running.",
    "completed_ids": [],
    "done_message": "",
}
_last_routine_tick: float | None = None


def _landmark_is_visible(lm, idx: int, min_visibility: float = 0.35) -> bool:
    try:
        return lm[idx].visibility >= min_visibility
    except Exception:
        return True


def _safe_point(lm, idx: int):
    return np.array([lm[idx].x, lm[idx].y, lm[idx].z], dtype=float)


def _midpoint(lm, a: int, b: int):
    return (_safe_point(lm, a) + _safe_point(lm, b)) / 2.0


def _distance(lm, a: int, b: int) -> float:
    return float(np.linalg.norm(_safe_point(lm, a) - _safe_point(lm, b)))


def _angle_from_idx(lm, a: int, b: int, c: int) -> float:
    return calculate_angle(_safe_point(lm, a), _safe_point(lm, b), _safe_point(lm, c))


def _require_core_upper_body(lm) -> tuple[bool, str]:
    required = [0, 11, 12, 13, 14, 15, 16]
    if any(not _landmark_is_visible(lm, idx) for idx in required):
        return False, "Move fully into frame so head, shoulders and arms are visible."
    return True, ""


def _require_core_full_body(lm) -> tuple[bool, str]:
    required = [11, 12, 23, 24, 25, 26]
    if any(not _landmark_is_visible(lm, idx) for idx in required):
        return False, "Step back so torso and legs are fully visible."
    return True, ""


def _head_tilt_metrics(lm):
    ear_l_sh = abs(lm[7].y - lm[11].y)
    ear_r_sh = abs(lm[8].y - lm[12].y)
    shoulder_delta = abs(lm[11].y - lm[12].y)
    return ear_l_sh, ear_r_sh, shoulder_delta


def _wrist_extended_forward(lm, side: str) -> bool:
    if side == "left":
        return abs(lm[15].x - lm[11].x) > 0.20 and abs(lm[13].x - lm[11].x) > 0.08
    return abs(lm[16].x - lm[12].x) > 0.20 and abs(lm[14].x - lm[12].x) > 0.08

#CHECKED ------
def validate_S01(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    ear_l_sh, ear_r_sh, _ = _head_tilt_metrics(lm)
    good = min(ear_l_sh, ear_r_sh) < 0.18
    return good, "Tilt one ear closer to your shoulder while keeping shoulders relaxed."

def validate_S02(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    eye_mid_y = (lm[2].y + lm[5].y) / 2.0
    chin_tucked = (lm[0].y - eye_mid_y) > 0.06
    return chin_tucked, "Tuck your chin down gently as if making a double chin."

def validate_S03(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    shoulder_mid_x = (lm[11].x + lm[12].x) / 2.0
    rotated = abs(lm[0].x - shoulder_mid_x) > 0.09
    return rotated, "Rotate your head to one side while keeping shoulders still."

def validate_S04(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    ear_l_sh, ear_r_sh, shoulder_delta = _head_tilt_metrics(lm)
    good = min(ear_l_sh, ear_r_sh) < 0.18 and shoulder_delta < 0.30 and ((_angle_from_idx(lm, 14, 12, 11) > 75 and _angle_from_idx(lm, 14, 12, 11) < 105) or (_angle_from_idx(lm, 12, 11, 13) > 75 and _angle_from_idx(lm, 12, 11, 13) < 105))
    return good, "Tilt your head to one side and keep shoulders down."

#NEED TO BE CHECKED ------
def validate_S05(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    shoulder_mid_x = (lm[11].x + lm[12].x) / 2.0
    shoulder_mid_y = (lm[11].y + lm[12].y) / 2.0
    neck_bent = (abs(lm[0].y - shoulder_mid_y) < 0.12) or (abs(lm[0].x - shoulder_mid_x) > 0.10)
    return neck_bent, "Move through a gentle head arc with chin lowered slightly."


def validate_S06(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    shoulder_raise = ((lm[11].y + lm[12].y) / 2.0) < ((lm[23].y + lm[24].y) / 2.0) - 0.22
    return shoulder_raise, "Lift shoulders up and roll backward in a smooth circle."


def validate_S07(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    left_elbow = _angle_from_idx(lm, 11, 13, 15)
    right_elbow = _angle_from_idx(lm, 12, 14, 16)
    good = 75 < left_elbow < 115 and 75 < right_elbow < 115 and abs(lm[13].y - lm[14].y) < 0.12
    return good, "Raise both arms to shoulder height with elbows around 90 degrees."


def validate_S08(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    shoulder_width = abs(lm[11].x - lm[12].x)
    elbow_width = abs(lm[13].x - lm[14].x)
    squeezed = elbow_width > shoulder_width * 0.9
    return squeezed, "Pull shoulder blades back and open your chest."


def validate_S09(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    left_across = lm[15].x > lm[0].x and abs(lm[13].y - lm[11].y) < 0.15
    right_across = lm[16].x < lm[0].x and abs(lm[14].y - lm[12].y) < 0.15
    return (left_across or right_across), "Bring one arm straight across your chest at shoulder height."


def validate_S10(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    wrists_back = (lm[15].y > lm[11].y and lm[16].y > lm[12].y and abs(lm[15].x - lm[16].x) < 0.20)
    return wrists_back, "Clasp hands behind your back and lift chest gently."


def validate_S11(lm):
    ok, msg = _require_core_full_body(lm)
    if not ok:
        return False, msg
    shoulder_mid_x = (lm[11].x + lm[12].x) / 2.0
    hip_mid_x = (lm[23].x + lm[24].x) / 2.0
    twisted = abs(shoulder_mid_x - hip_mid_x) > 0.04
    return twisted, "Twist your torso to one side while keeping hips stable."


def validate_S12(lm):
    ok, msg = _require_core_full_body(lm)
    if not ok:
        return False, msg
    shoulder_mid_y = (lm[11].y + lm[12].y) / 2.0
    hip_mid_y = (lm[23].y + lm[24].y) / 2.0
    folded = shoulder_mid_y > hip_mid_y - 0.03
    return folded, "Fold your torso forward toward your thighs."


def validate_S13(lm):
    ok, msg = _require_core_full_body(lm)
    if not ok:
        return False, msg
    spine_angle = _angle_from_idx(lm, 11, 23, 25)
    active = spine_angle < 165 or spine_angle > 195
    return active, "Alternate between arching and rounding your spine."


def validate_S14(lm):
    ok, msg = _require_core_full_body(lm)
    if not ok:
        return False, msg
    shoulder_mid_z = (lm[11].z + lm[12].z) / 2.0
    hip_mid_z = (lm[23].z + lm[24].z) / 2.0
    extended = shoulder_mid_z > hip_mid_z + 0.03
    return extended, "Lean gently backward from the lower back."


def validate_S15(lm):
    ok, msg = _require_core_full_body(lm)
    if not ok:
        return False, msg
    one_hip_open = abs(lm[25].x - lm[23].x) > 0.06 or abs(lm[26].x - lm[24].x) > 0.06
    torso_upright = abs(lm[11].y - lm[23].y) > 0.16
    return one_hip_open and torso_upright, "Keep torso upright and shift hips to stretch one hip flexor."


def validate_S16(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    left_pose = _wrist_extended_forward(lm, "left") and _angle_from_idx(lm, 13, 15, 19) < 145
    right_pose = _wrist_extended_forward(lm, "right") and _angle_from_idx(lm, 14, 16, 20) < 145
    return (left_pose or right_pose), "Extend one arm palm-up and bend wrist downward."


def validate_S17(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    left_pose = _wrist_extended_forward(lm, "left") and _angle_from_idx(lm, 13, 15, 19) > 160
    right_pose = _wrist_extended_forward(lm, "right") and _angle_from_idx(lm, 14, 16, 20) > 160
    return (left_pose or right_pose), "Extend one arm palm-down and press the hand downward."


def validate_S18(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    left_out = abs(lm[15].x - lm[11].x) > 0.22 and abs(lm[15].y - lm[11].y) < 0.18
    right_out = abs(lm[16].x - lm[12].x) > 0.22 and abs(lm[16].y - lm[12].y) < 0.18
    return (left_out or right_out), "Hold one arm out at shoulder height and rotate forearm."


def validate_S19(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    wrists_together = _distance(lm, 15, 16) < 0.10
    hands_center = abs(((lm[15].x + lm[16].x) / 2.0) - lm[0].x) < 0.12
    return wrists_together and hands_center, "Press palms together in front of your chest."


def validate_S20(lm):
    ok, msg = _require_core_upper_body(lm)
    if not ok:
        return False, msg
    hands_up = lm[15].y < lm[11].y + 0.25 and lm[16].y < lm[12].y + 0.25
    arms_open = abs(lm[15].x - lm[16].x) > 0.25
    return hands_up and arms_open, "Hold hands up and spread fingers wide like a fan."


_STRETCH_VALIDATORS = {
    "S01": validate_S01, "S02": validate_S02, "S03": validate_S03, "S04": validate_S04, "S05": validate_S05,
    "S06": validate_S06, "S07": validate_S07, "S08": validate_S08, "S09": validate_S09, "S10": validate_S10,
    "S11": validate_S11, "S12": validate_S12, "S13": validate_S13, "S14": validate_S14, "S15": validate_S15,
    "S16": validate_S16, "S17": validate_S17, "S18": validate_S18, "S19": validate_S19, "S20": validate_S20,
}


def validate_stretch_form(stretch_id: str, lm):
    validator = _STRETCH_VALIDATORS.get(stretch_id)
    if validator is None:
        return False, "No validator available for this stretch."
    return validator(lm)


def _current_stretch_locked():
    stretches = _routine_state["stretches"]
    idx = _routine_state["current_index"]
    if not stretches or idx < 0 or idx >= len(stretches):
        return None
    return stretches[idx]


def _set_next_stretch_locked():
    stretches = _routine_state["stretches"]
    _routine_state["current_index"] += 1
    _routine_state["hold_elapsed_s"] = 0.0
    _routine_state["is_correct_now"] = False
    _routine_state["last_feedback_text"] = "Prepare for the next stretch."
    if _routine_state["current_index"] >= len(stretches):
        _routine_state["state"] = "finished"
        _routine_state["done_message"] = "Great work. You completed the full stretch routine."
        return
    nxt = _current_stretch_locked()
    _routine_state["hold_target_s"] = float(nxt.get("duration_seconds", 20))


def _update_routine_with_landmarks(lm, dt: float):
    with _routine_lock:
        if _routine_state["state"] != "running":
            return
        stretch = _current_stretch_locked()
        if stretch is None:
            _routine_state["state"] = "finished"
            _routine_state["done_message"] = "Great work. Routine complete."
            return

        stretch_id = stretch.get("id", "")
        is_correct, feedback = validate_stretch_form(stretch_id, lm)
        _routine_state["is_correct_now"] = bool(is_correct)
        _routine_state["last_feedback_text"] = feedback
        _routine_state["hold_target_s"] = float(stretch.get("duration_seconds", 20))

        if is_correct:
            _routine_state["hold_elapsed_s"] += max(0.0, dt)

        if _routine_state["hold_elapsed_s"] >= _routine_state["hold_target_s"]:
            _routine_state["completed_ids"].append(stretch_id)
            _set_next_stretch_locked()


def _mark_routine_landmarks_missing():
    with _routine_lock:
        if _routine_state["state"] != "running":
            return
        _routine_state["is_correct_now"] = False
        _routine_state["last_feedback_text"] = "Move your full body into view to continue the timer."


def start_routine(stretches: list[dict]):
    global _last_routine_tick
    if not stretches:
        raise ValueError("Routine requires at least one stretch.")
    with _routine_lock:
        _routine_state["state"] = "running"
        _routine_state["stretches"] = stretches[:]
        _routine_state["current_index"] = 0
        _routine_state["is_correct_now"] = False
        _routine_state["hold_elapsed_s"] = 0.0
        _routine_state["hold_target_s"] = float(stretches[0].get("duration_seconds", 20))
        _routine_state["last_feedback_text"] = "Get into position and hold the stretch."
        _routine_state["completed_ids"] = []
        _routine_state["done_message"] = ""
    _last_routine_tick = time.monotonic()


def stop_routine():
    global _last_routine_tick
    with _routine_lock:
        _routine_state["state"] = "inactive"
        _routine_state["stretches"] = []
        _routine_state["current_index"] = 0
        _routine_state["is_correct_now"] = False
        _routine_state["hold_elapsed_s"] = 0.0
        _routine_state["hold_target_s"] = 0.0
        _routine_state["last_feedback_text"] = "Routine stopped."
        _routine_state["completed_ids"] = []
        _routine_state["done_message"] = ""
    _last_routine_tick = None


def get_routine_status():
    with _routine_lock:
        stretch = _current_stretch_locked()
        return {
            "state": _routine_state["state"],
            "current_stretch": stretch,
            "index": _routine_state["current_index"],
            "total": len(_routine_state["stretches"]),
            "is_correct": _routine_state["is_correct_now"],
            "feedback": _routine_state["last_feedback_text"],
            "hold_elapsed_s": round(float(_routine_state["hold_elapsed_s"]), 1),
            "hold_target_s": round(float(_routine_state["hold_target_s"]), 1),
            "completed_count": len(_routine_state["completed_ids"]),
            "completed_ids": _routine_state["completed_ids"][:],
            "done_message": _routine_state["done_message"],
        }


# ── Helper: angle at vertex b formed by points a-b-c ─────────────────────────
def calculate_angle(a, b, c) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# ── Core detection: analyse a single set of 33 landmarks ─────────────────────
def _analyse_landmarks(lm) -> str:
    global _wrist_high_frames

    # 1. FORWARD HEAD
    # Lateral (side-on): noticeable drift, not a tiny camera angle
    ear_mid_x      = (lm[7].x + lm[8].x) / 2
    shoulder_mid_x = (lm[11].x + lm[12].x) / 2
    lateral_fwd    = abs(ear_mid_x - shoulder_mid_x) > 0.15

    # Depth (front-facing): ears must be clearly forward of shoulders
    ear_mid_z      = (lm[7].z + lm[8].z) / 2
    shoulder_mid_z = (lm[11].z + lm[12].z) / 2
    depth_fwd      = (ear_mid_z - shoulder_mid_z) < -0.20

    forward_head = lateral_fwd or depth_fwd

    # 2. ROUNDED SHOULDERS
    # Shoulders must be clearly narrower than hips (heavy hunching only)
    shoulder_width = abs(lm[11].x - lm[12].x)
    hip_width      = abs(lm[23].x - lm[24].x)
    if hip_width > 1e-4:
        rounded_shoulders = (shoulder_width / hip_width) < 0.70
    else:
        rounded_shoulders = False

    # 3. SLOUCHING
    # Allow a reasonable natural curve — only flag pronounced slouching
    shoulder_mid = [(lm[11].x + lm[12].x) / 2,
                    (lm[11].y + lm[12].y) / 2,
                    (lm[11].z + lm[12].z) / 2]
    hip_mid      = [(lm[23].x + lm[24].x) / 2,
                    (lm[23].y + lm[24].y) / 2,
                    (lm[23].z + lm[24].z) / 2]
    knee_mid     = [(lm[25].x + lm[26].x) / 2,
                    (lm[25].y + lm[26].y) / 2,
                    (lm[25].z + lm[26].z) / 2]
    spine_angle  = calculate_angle(shoulder_mid, hip_mid, knee_mid)
    slouching    = abs(spine_angle - 180.0) > 28.0

    # 4. WRIST STRAIN — sustained raised wrists only
    left_wrist_high  = lm[15].y < lm[13].y
    right_wrist_high = lm[16].y < lm[14].y
    if left_wrist_high or right_wrist_high:
        _wrist_high_frames += 1
    else:
        _wrist_high_frames = 0
    wrist_strain = _wrist_high_frames >= 18

    if slouching:
        return "Slouching"
    if forward_head:
        return "Forward Head Tilt"
    if rounded_shoulders:
        return "Rounded Shoulders"
    if wrist_strain:
        return "Wrist Strain Risk"
    return "Good Posture"


# ── Frame processor ───────────────────────────────────────────────────────────
def _smooth_label(raw_label: str) -> str:
    """Return the majority label over the last _SMOOTH_WINDOW frames."""
    global _label_history
    _label_history.append(raw_label)
    if len(_label_history) > _SMOOTH_WINDOW:
        _label_history.pop(0)
    return max(set(_label_history), key=_label_history.count)


def _process_frame(frame: np.ndarray, pose) -> np.ndarray:
    global current_posture, _last_routine_tick

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        now = time.monotonic()
        if _last_routine_tick is None:
            _last_routine_tick = now
        dt = now - _last_routine_tick
        _last_routine_tick = now

        _update_routine_with_landmarks(lm, dt)

        routine = get_routine_status()
        if routine["state"] == "running" and routine["is_correct"]:
            good_spec = _mp_draw.DrawingSpec(color=(0, 220, 0), thickness=2, circle_radius=2)
            _mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                _mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=good_spec,
                connection_drawing_spec=good_spec,
            )
        else:
            _mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                _mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=_mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        raw   = _analyse_landmarks(lm)
        label = _smooth_label(raw)
        with _posture_lock:
            current_posture = label
    else:
        _mark_routine_landmarks_missing()
        with _posture_lock:
            label = current_posture

    color = (0, 255, 0) if label == "Good Posture" else (0, 0, 255)
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

    routine = get_routine_status()
    if routine["state"] == "running":
        timer_txt = f"Routine {routine['index'] + 1}/{routine['total']}  {routine['hold_elapsed_s']:.1f}s/{routine['hold_target_s']:.1f}s"
        status_txt = "Correct form" if routine["is_correct"] else "Adjust form"
        timer_color = (0, 220, 0) if routine["is_correct"] else (0, 165, 255)
        cv2.putText(frame, timer_txt, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2, cv2.LINE_AA)
        cv2.putText(frame, status_txt, (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2, cv2.LINE_AA)
    elif routine["state"] == "finished":
        cv2.putText(frame, "Routine complete", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)

    return frame


# ── Single camera thread — the only place VideoCapture is opened ──────────────
def _camera_loop() -> None:
    """
    Open the webcam once, run MediaPipe pose, encode each frame as JPEG and
    store it in _latest_frame. All Flask clients read from that shared buffer.
    """
    global _latest_frame, _camera_running

    idx = _camera_device_index()
    cap = _video_capture(idx)
    if not cap.isOpened():
        _camera_running = False
        return
    # Keep frames lightweight for lower detection latency on CPU-only setups.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if platform.system() != "Darwin":
        # On macOS, buffer size 1 can worsen black-frame issues with some cameras.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    _warmup_capture(cap)

    with _mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while _camera_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = _process_frame(frame, pose)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            jpeg = buffer.tobytes()
            with _frame_lock:
                _latest_frame = jpeg
            _frame_event.set()
            _frame_event.clear()

    cap.release()


# ── Public API ────────────────────────────────────────────────────────────────

def start_camera() -> bool:
    """
    Start the background camera thread. Returns True if the camera opened
    successfully, False otherwise. Safe to call multiple times.
    """
    global _camera_thread, _camera_running, _latest_frame

    if _camera_thread is not None and _camera_thread.is_alive():
        return True  # already running

    # Quick probe — don't hold the camera open
    probe = _video_capture(_camera_device_index())
    if not probe.isOpened():
        probe.release()
        return False
    _warmup_capture(probe, n=8)
    probe.release()

    _camera_running = True
    _latest_frame   = b""
    _camera_thread  = threading.Thread(
        target=_camera_loop, daemon=True, name="posture-camera"
    )
    _camera_thread.start()

    # Wait up to 3 s for the first frame
    deadline = time.time() + 3.0
    while time.time() < deadline:
        with _frame_lock:
            if _latest_frame:
                return True
        time.sleep(0.05)

    return bool(_latest_frame)


def stop_camera() -> None:
    """Signal the camera thread to stop."""
    global _camera_running
    _camera_running = False


def get_current_posture() -> str:
    """Return the most recently detected posture classification (thread-safe)."""
    with _posture_lock:
        return current_posture


def generate_frames():
    """
    Generator that yields MJPEG-encoded frames for a Flask streaming response.
    Reads from the shared _latest_frame buffer — no new VideoCapture needed.
    """
    last_frame = b""
    while True:
        with _frame_lock:
            frame = _latest_frame

        if not frame:
            time.sleep(0.05)
            continue

        if frame is last_frame:
            # No new frame yet — wait briefly
            time.sleep(0.03)
            continue

        last_frame = frame
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


# ── Standalone entry-point ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting camera… press 'q' to quit.")
    if not start_camera():
        print("ERROR: Could not open webcam.")
        raise SystemExit(1)

    cap = cv2.VideoCapture(0)  # separate window display
    if not cap.isOpened():
        print("ERROR: Could not open webcam for display.")
        raise SystemExit(1)

    with _mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated = _process_frame(frame, pose)
            cv2.imshow("Posture Detector", annotated)
            print(f"\rPosture: {get_current_posture():<25}", end="", flush=True)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    stop_camera()
    print()
