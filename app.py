import json
import logging
import os

from flask import Flask, Response, jsonify, request, render_template
from flask_cors import CORS

from posture_detector import (
    generate_frames,
    get_current_posture,
    start_camera,
    start_routine,
    stop_routine,
    get_routine_status,
)
from library import STRETCH_LIBRARY, fallback_stretches
from claude_client import recommend_stretches

# Enrich STRETCH_LIBRARY with image URLs from stretches.json
_json_path = os.path.join(os.path.dirname(__file__), "stretches.json")
try:
    with open(_json_path, "r") as _f:
        _json_data = {s["id"]: s for s in json.load(_f)}
    for _s in STRETCH_LIBRARY:
        _entry = _json_data.get(_s["id"], {})
        if "gif_url" in _entry:
            _s["gif_url"] = _entry["gif_url"]
        if "image_url" in _entry:
            _s["image_url"] = _entry["image_url"]
except Exception:
    pass

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_webcam_available = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "webcam": _webcam_available})


@app.route("/video_feed")
def video_feed():
    if not _webcam_available:
        return jsonify({"error": "Webcam not available"}), 503
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/posture")
def posture():
    return jsonify({"result": get_current_posture()})


@app.route("/stretches")
def stretches():
    return jsonify(STRETCH_LIBRARY)


@app.route("/routine/status")
def routine_status():
    return jsonify(get_routine_status())


@app.route("/routine/stop", methods=["POST"])
def routine_stop():
    stop_routine()
    return jsonify({"ok": True, "status": get_routine_status()})


@app.route("/routine/start", methods=["POST"])
def routine_start():
    try:
        data = request.get_json(force=True, silent=True) or {}
        by_id = {s["id"]: s for s in STRETCH_LIBRARY}

        stretches = data.get("stretches")
        stretch_ids = data.get("stretch_ids")

        chosen = []
        if isinstance(stretches, list) and stretches:
            for s in stretches:
                sid = str((s or {}).get("id", "")).strip()
                if sid in by_id:
                    chosen.append(by_id[sid])
        elif isinstance(stretch_ids, list) and stretch_ids:
            for sid in stretch_ids:
                sid = str(sid).strip()
                if sid in by_id:
                    chosen.append(by_id[sid])

        # Deduplicate while preserving order
        deduped = []
        seen = set()
        for s in chosen:
            sid = s["id"]
            if sid not in seen:
                deduped.append(s)
                seen.add(sid)

        if len(deduped) != 5:
            return jsonify({"error": "Routine requires exactly 5 valid stretches."}), 400

        start_routine(deduped)
        return jsonify({"ok": True, "status": get_routine_status()})
    except Exception as exc:
        logger.error("Error in /routine/start: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json(force=True, silent=True) or {}

        posture_label = data.get("posture", "").strip()
        pain_text     = data.get("pain_text", "").strip()

        if not posture_label:
            return jsonify({"error": "Missing required field: posture"}), 400
        if not pain_text:
            return jsonify({"error": "Missing required field: pain_text"}), 400

        result = recommend_stretches(posture_label, pain_text)
        return jsonify({"stretches": result})

    except Exception as exc:
        logger.error("Error in /recommend: %s", exc)
        return jsonify({"error": str(exc), "stretches": fallback_stretches()}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _webcam_available = start_camera()
    if _webcam_available:
        logger.info("Camera started — posture detection active.")
    else:
        logger.warning("Webcam not available — video feed disabled.")

    app.run(debug=True, threaded=True, use_reloader=False)

"""

cd "/Users/kwankaochuaphanich/Desktop/Home/External School Work/Everything/China HackMIT/Github/HackMITChina2026Real"
source .venv/bin/activate
python3 app.py

"""