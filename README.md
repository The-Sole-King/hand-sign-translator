# Sign Reader

A desktop program that reads **ASL fingerspelling** from your webcam and types it out. Everything runs locally on your computer, and no video is uploaded anywhere.

## Install

Needs Python 3.9+ and a webcam.

```bash
cd sign-reader
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python -m sign_reader
```

On first launch it downloads MediaPipe's hand model (~8 MB) into `~/.sign_reader/`.
Options: `--camera 1` to use a different camera, `--samples path.json` to use a different set of trained samples.

## Use it

It learns from **your** hands, so train it first:

1. Press **Tab** to switch to **TRAIN** mode.
2. Press a letter key (e.g. `a`). After a 3-second countdown, hold the handshape while it records 30 samples. Move your hand slightly as it records so it learns small variations.
3. Repeat for each letter you want. `1` and `2` record optional gestures for **space** and **delete**. **Shift + key** deletes that sign's samples.
4. Press **Tab** to go back to **READ** mode and start fingerspelling.

| Read mode | |
|---|---|
| Hold a sign | Types it once the bar fills (0.6 s) |
| Drop your hand | Lets you repeat a letter (the "LL" in HELLO) |
| Keep your hand down 1.5 s | Adds a space |
| Space / Backspace | Edit the text by keyboard |
| Enter | Read the text aloud |
| C | Clear the text |
| Tab / Esc | Train mode / quit |

Samples are saved to `~/.sign_reader/samples.json` after each recording. Copy that file to back up your training or move it to another computer.

**Tips:** letters that look alike (M/N, A/S/T, U/V) need more samples, so record them 2–3 times from slightly different angles. J and Z involve movement, so record their final handshape.

## How it works

1. **Hand tracking**: [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) finds 21 3-D points on the hand in each frame.
2. **Normalize**: the points are re-centred on the wrist, scaled by palm size, and mirrored for left hands. Position, distance and which hand you use don't matter, but the direction the hand points does (ASL relies on it: H vs U, K vs P).
3. **Classify**: k-nearest-neighbours (k = 5) compares the pose with your recorded samples.
4. **Debounce**: frame-by-frame guesses are smoothed so each held sign types exactly once.

| File | What it does |
|---|---|
| `sign_reader/core.py` | Features, classifier, debouncing, saving. No camera or GUI code |
| `sign_reader/app.py` | Camera, MediaPipe, window and keyboard controls |
| `tests/test_core.py` | Unit tests: `python -m unittest discover -s tests` |

Text-to-speech uses the system voice: `say` on macOS, built-in speech on Windows, and `spd-say` or `espeak` on Linux.

## Limitations

- Static handshapes only. J and Z are matched by their final pose, and full ASL signs (which use motion, facial expression and two hands) aren't recognized.
- Accuracy depends on your training. Good lighting and a plain background help.
