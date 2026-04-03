# StretchAI

StretchAI is a webcam-assisted posture and stretch coach built with Flask, MediaPipe, and OpenCV.

## Features

- Live posture detection with annotated webcam feed.
- Stretch recommendations based on posture + pain text (AI-powered via Ollama, or falls back to curated defaults).
- Guided 5-stretch routine:
  - Starts from the recommended list.
  - Validates current stretch form with rule-based landmark checks.
  - Counts hold time only when form is correct.
  - Shows a live countdown timer while holding.
  - Auto-advances through all five stretches.
  - Shows completion message at the end.
  - Stop button to exit the routine at any time.
- Selection of your own routine from the existing library of stretches

## Setup

```bash
python -m venv .venv

# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### Optional: AI recommendations via Ollama

Without Ollama, the app falls back to a built-in set of curated stretches — everything still works. To enable AI-powered recommendations:

1. Install [Ollama](https://ollama.com)
2. Run: `ollama pull llama3.2`
3. Start the Ollama server: `ollama serve`

## Run

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Webcam not appearing | Check browser camera permissions; only one app can use the webcam at a time |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in your virtual environment |
| AI recommendations time out | Ollama may not be running — the app will use fallback stretches automatically |
| MediaPipe download hangs | First run downloads the pose model; wait or check your internet connection |
| Port 5000 in use (Mac) | Mac AirPlay uses port 5000. Run `python app.py` with `PORT=5001` or disable AirPlay Receiver in System Settings |

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server + webcam status |
| `/video_feed` | GET | MJPEG webcam stream |
| `/posture` | GET | Current posture classification |
| `/stretches` | GET | Full stretch library (20 stretches) |
| `/recommend` | POST | AI stretch recommendations — body: `{ "posture": "...", "pain_text": "..." }` |
| `/routine/start` | POST | Start routine — body: `{ "stretch_ids": ["S01","S06",...] }` |
| `/routine/stop` | POST | Stop current routine |
| `/routine/status` | GET | Routine progress and form feedback |

## Manual verification checklist

1. Open app and confirm webcam feed appears.
2. Enter pain text and click **Recommend stretches**.
3. Confirm 5 stretches are rendered (or fallback stretches if Ollama unavailable).
4. Click **Start stretch routine** — verify the Stop button appears.
5. For current stretch:
   - Hold incorrect form → timer stays paused, countdown hidden, feedback asks to adjust.
   - Move into valid form → status turns correct, green countdown appears, timer increases.
6. Keep correct form until countdown reaches 0.
7. Confirm routine auto-advances to the next stretch.
8. Repeat until all 5 are complete and "Routine complete" message appears.
9. Click **Stop routine** mid-routine to confirm it resets cleanly.
