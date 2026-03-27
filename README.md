# StretchAI

StretchAI is a webcam-assisted posture and stretch coach built with Flask, MediaPipe, and OpenCV.

## Features

- Live posture detection with annotated webcam feed.
- Stretch recommendations based on posture + pain text.
- Guided 5-stretch routine:
  - Starts from the recommended list.
  - Validates current stretch form with rule-based landmark checks.
  - Counts hold time only when form is correct.
  - Auto-advances through all five stretches.
  - Shows completion message at the end.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install flask flask-cors mediapipe opencv-python numpy
```

If you use local Ollama recommendations, ensure Ollama is running and the `llama3.2` model is available.

## Run

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

## Routine API

- `POST /routine/start`
  - body: `{ "stretch_ids": ["S01","S06","S11","S16","S08"] }`
- `POST /routine/stop`
- `GET /routine/status`

## Manual verification checklist

1. Open app and confirm webcam feed appears.
2. Enter pain text and click **Recommend stretches**.
3. Confirm exactly 5 stretches are rendered.
4. Click **Start stretch routine**.
5. For current stretch:
   - Hold incorrect form -> timer stays paused, feedback asks to adjust.
   - Move into valid form -> status turns correct and timer increases.
6. Keep correct form until timer reaches target duration.
7. Confirm routine auto-advances to the next stretch.
8. Repeat until all 5 are complete and final completion message appears.
