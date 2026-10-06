# Sign Reader

A desktop program that reads **ASL fingerspelling** from your webcam and types it out. Everything runs locally on your computer, and no video is uploaded anywhere.

## Get the app (double-click)

Download the build for your computer from the repo's **Actions** tab (open the latest *Build app* run and scroll to **Artifacts**), or from [**Releases**](../../releases) once a version has been published:

| OS | File | Open it |
|---|---|---|
| Windows | `SignReader-windows.zip` | Unzip and double-click **SignReader.exe**. The first time, Windows SmartScreen may say "Windows protected your PC": click **More info → Run anyway**. |
| macOS (Apple Silicon) | `SignReader-macos.zip` | Unzip, drag **SignReader.app** to Applications, then **right-click → Open** the first time (the app isn't signed by Apple). Allow camera access when asked. |
| Linux | `SignReader-linux.tar.gz` | Extract and double-click **SignReader** (or run `./SignReader`). |

The hand and face models are built in, so the app works offline (except for opening websites). It takes a few seconds to start because it unpacks itself.

## Run from source

Needs Python 3.9+ and a webcam.

```bash
git clone https://github.com/The-Sole-King/hand-sign-translator.git
cd hand-sign-translator
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m sign_reader
```

On first launch it downloads MediaPipe's hand model (~8 MB) into `~/.sign_reader/`.
Options: `--camera 1` to use a different camera, `--samples path.json` to use a different set of trained samples.

## Build the app yourself

Each OS has to build its own app:

```bash
pip install -r requirements.txt pyinstaller
python packaging/build.py
```

The result is in `dist/`: `SignReader.exe` on Windows, `SignReader.app` on macOS, `SignReader` on Linux. `dist/SignReader… --self-test` checks that the build can load the model. The [GitHub Actions workflow](.github/workflows/build.yml) does this on all three OSes on every push. Pushing a tag like `v1.0.0` publishes the builds as a Release.

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
| Tab / Esc | Next mode (read → train → jutsu) / quit |

Samples are saved to `~/.sign_reader/samples.json` after each recording. Copy that file to back up your training or move it to another computer.

**Tips:** letters that look alike (M/N, A/S/T, U/V) need more samples, so record them 2–3 times from slightly different angles. J and Z involve movement, so record their final handshape.

## Jutsu mode: hand-seal combos

Press **Tab** until the panel says **JUTSU**. Now signs don't type. Instead, a short sequence of signs (a "combo", like Naruto hand seals) triggers something:

| Combo | Signs | What happens |
|---|---|---|
| Rasengan | **B** then **C** | A swirling ball of chakra appears in your palm and follows your hand for 12 s |
| Kakashi Mask | **M** then **K** | A cloth mask covers your nose and mouth (sign it again to take it off) |
| Firefox | **F** then **O** | Opens Firefox |

Make each sign in order, holding it until the bar fills, within 3 seconds of the previous one. Train those letters first (seals you haven't trained show in red). **X** clears effects.

### Your own combos

Press **E** in jutsu mode to open `~/.sign_reader/combos.json`, edit it, save it, then press **R** to reload. Each combo has a name, a list of signs, and **one** action:

```json
[
  {"name": "Rasengan",     "signs": ["B", "C"],      "effect": "rasengan"},
  {"name": "Kakashi Mask", "signs": ["M", "K"],      "effect": "mask"},
  {"name": "Firefox",      "signs": ["F", "O"],      "open": "firefox"},
  {"name": "Spotify",      "signs": ["S", "P"],      "open": "spotify"},
  {"name": "YouTube",      "signs": ["Y", "T", "B"], "url": "https://youtube.com"}
]
```

- `effect`: `rasengan` or `mask`
- `open`: an installed app by name (`firefox`, `spotify`, `notepad`, `calc`, …; on macOS use the app's name, e.g. `"Visual Studio Code"`)
- `url`: a website, opened in your default browser

Combos can be any length. If one combo is the end of another (`B C` and `A B C`), the longer one wins when you sign it.

## How it works

1. **Hand tracking**: [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) finds 21 3-D points on the hand in each frame.
2. **Normalize**: the points are re-centred on the wrist, scaled by palm size, and mirrored for left hands. Position, distance and which hand you use don't matter, but the direction the hand points does (ASL relies on it: H vs U, K vs P).
3. **Classify**: k-nearest-neighbours (k = 5) compares the pose with your recorded samples.
4. **Debounce**: frame-by-frame guesses are smoothed so each held sign types exactly once.

| File | What it does |
|---|---|
| `sign_reader/core.py` | Features, classifier, debouncing, saving. No camera or GUI code |
| `sign_reader/app.py` | Camera, MediaPipe, window and keyboard controls |
| `sign_reader/jutsu.py` | Combo file format and sequence matching. No camera or GUI code |
| `sign_reader/effects.py` | Rasengan and mask drawing, launching apps and URLs |
| `tests/test_core.py` | Unit tests: `python -m unittest discover -s tests` |
| `packaging/` | PyInstaller spec and build script for the double-click app |

Text-to-speech uses the system voice: `say` on macOS, built-in speech on Windows, and `spd-say` or `espeak` on Linux.

## Limitations

- Static handshapes only. J and Z are matched by their final pose, and full ASL signs (which use motion, facial expression and two hands) aren't recognized.
- Accuracy depends on your training. Good lighting and a plain background help.
