"""
claude_client.py
----------------
Uses a local Ollama model to recommend stretches based on detected
posture and the user's self-reported pain description.

Exports:
  recommend_stretches(posture_label, pain_text) -> list[dict]
"""

import json
import logging
import urllib.request
import urllib.error

from library import STRETCH_LIBRARY, fallback_stretches

logger = logging.getLogger(__name__)

OLLAMA_URL  = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"

_STRETCH_SUMMARY = "\n".join(
    f"- {s['id']}: {s['name']} | targets: {s['target_muscle']} | "
    f"posture tags: {', '.join(s['posture_tags'])} | "
    f"pain keywords: {', '.join(s['pain_keywords'])}"
    for s in STRETCH_LIBRARY
)

_SYSTEM_PROMPT = f"""You are StretchAI, a workplace wellness assistant.
Given a detected posture issue and the user's pain description, select exactly 5 stretches from the library below.

Respond with ONLY a valid JSON array of exactly 5 stretch IDs. No explanation, no markdown, just the array.
Example: ["S01", "S06", "S11", "S16", "S08"]

Stretch library:
{_STRETCH_SUMMARY}
"""


def _normalize_to_five(stretches: list[dict]) -> list[dict]:
    """Return exactly 5 unique stretches, topping up from fallback if needed."""
    ordered = []
    seen = set()
    for s in stretches:
        sid = s.get("id")
        if sid and sid not in seen:
            ordered.append(s)
            seen.add(sid)

    if len(ordered) < 5:
        for s in fallback_stretches():
            sid = s.get("id")
            if sid and sid not in seen:
                ordered.append(s)
                seen.add(sid)
            if len(ordered) == 5:
                break

    if len(ordered) < 5:
        for s in STRETCH_LIBRARY:
            sid = s.get("id")
            if sid and sid not in seen:
                ordered.append(s)
                seen.add(sid)
            if len(ordered) == 5:
                break

    return ordered[:5]


def recommend_stretches(posture_label: str, pain_text: str) -> list:
    """
    Ask the local Ollama model to pick 5 stretches for the given posture and pain.

    Returns a list of stretch dicts from STRETCH_LIBRARY.
    Falls back to fallback_stretches() on any error.
    """
    user_message = (
        f"Detected posture issue: {posture_label}\n"
        f"User pain description: {pain_text}\n\n"
        "Reply with only a JSON array of 5 stretch IDs."
    )

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,
    }).encode()

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())

        raw = body["message"]["content"].strip()

        # Extract JSON array even if there's surrounding text
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON array found in response: {raw!r}")
        ids = json.loads(raw[start:end])

        lookup = {s["id"]: s for s in STRETCH_LIBRARY}
        result = [lookup[sid] for sid in ids if sid in lookup]

        if not result:
            raise ValueError(f"No valid IDs in response: {ids}")

        return _normalize_to_five(result)

    except Exception as exc:
        logger.error("Ollama recommendation failed: %s", exc)
        return _normalize_to_five(fallback_stretches())
