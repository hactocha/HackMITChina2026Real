import json
import logging
import os
import urllib.error
import urllib.request
from urllib.parse import urlparse, unquote

from flask import Flask, Response, abort, jsonify, request, render_template
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

# Enrich STRETCH_LIBRARY with media URLs from stretches.json
_json_path = os.path.join(os.path.dirname(__file__), "stretches.json")
_STRETCH_JSON_BY_ID = {}
try:
    with open(_json_path, "r", encoding="utf-8") as _f:
        _STRETCH_JSON_BY_ID = {s["id"]: s for s in json.load(_f) if isinstance(s, dict) and s.get("id")}
    for _s in STRETCH_LIBRARY:
        _entry = _STRETCH_JSON_BY_ID.get(_s["id"], {})
        if "gif_url" in _entry:
            _s["gif_url"] = _entry["gif_url"]
        if "image_url" in _entry:
            _s["image_url"] = _entry["image_url"]
        if "body_region" in _entry:
            _s["body_region"] = _entry["body_region"]
except Exception as exc:
    logging.getLogger(__name__).warning("Failed to load stretches.json media metadata: %s", exc)

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


def _unsafe_image_hostname(hostname: str) -> bool:
    """Basic SSRF guard: block private / loopback hosts."""
    h = (hostname or "").lower().strip()
    if not h:
        return True
    if h in ("localhost", "0.0.0.0", "::1"):
        return True
    if h.endswith(".localhost") or h.endswith(".local") or h.endswith(".internal"):
        return True
    parts = h.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        a, b, c, d = (int(p) for p in parts)
        if a in (0, 10, 127) or (a == 169 and b == 254):
            return True
        if a == 172 and 16 <= b <= 31:
            return True
        if a == 192 and b == 168:
            return True
    return False


def _sniff_image_mimetype(data: bytes) -> str | None:
    """Infer image/* type from magic bytes when headers are wrong/missing."""
    if len(data) < 12:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM":
        return "image/bmp"
    return None


def _image_proxy_headers() -> dict[str, str]:
    """Some CDNs reject bare agents; send browser-like headers."""
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


@app.route("/image-proxy")
def image_proxy():
    """Fetch remote image server-side to avoid browser-side hotlink restrictions."""
    raw = request.args.get("url", "")
    try:
        url = unquote(raw)
    except Exception:
        url = raw
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        abort(400)
    if _unsafe_image_hostname(parsed.hostname or ""):
        abort(400)

    req = urllib.request.Request(url, headers=_image_proxy_headers(), method="GET")
    max_bytes = 4 * 1024 * 1024

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                abort(413)
            header_ct = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            sniffed = _sniff_image_mimetype(data)
            sample = data[:256].lstrip()

            if header_ct.startswith("image/"):
                mimetype = header_ct
            elif sniffed:
                mimetype = sniffed
            elif sample.startswith(b"<?xml") or sample.startswith(b"<svg"):
                mimetype = "image/svg+xml"
            else:
                abort(415)

            return Response(data, mimetype=mimetype)
    except urllib.error.HTTPError as exc:
        logger.warning("image-proxy HTTP error for %s: %s", url, exc)
        abort(502)
    except Exception as exc:
        logger.warning("image-proxy failed for %s: %s", url, exc)
        abort(502)


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
    status = get_routine_status()
    cur = status.get("current_stretch")
    if isinstance(cur, dict):
        sid = cur.get("id")
        if sid:
            entry = _STRETCH_JSON_BY_ID.get(sid, {})
            if isinstance(entry, dict):
                merged = dict(cur)
                if entry.get("gif_url"):
                    merged["gif_url"] = entry.get("gif_url")
                if entry.get("image_url"):
                    merged["image_url"] = entry.get("image_url")
                status["current_stretch"] = merged
    return jsonify(status)


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

        if len(chosen) != 5:
            return jsonify({"error": "Routine requires exactly 5 valid stretches."}), 400

        start_routine(chosen)
        return jsonify({"ok": True, "status": get_routine_status()})
    except Exception as exc:
        logger.error("Error in /routine/start: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/fallback")
def fallback():
    return jsonify({"stretches": fallback_stretches()})


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