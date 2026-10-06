"""Desktop window: webcam + MediaPipe hand tracking + the k-NN reader.

Run with `python -m sign_reader`. See README.md for controls.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .core import (
    DELETE,
    LABELS,
    SPACE,
    Stabilizer,
    apply_event,
    classify,
    landmarks_to_features,
    load_samples,
    save_samples,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DATA_DIR = Path.home() / ".sign_reader"
SAMPLES_PER_RECORDING = 30
SAMPLE_INTERVAL_S = 0.08
COUNTDOWN_S = 3.0

WINDOW = "Sign Reader"
CAM_W, CAM_H = 640, 480
PANEL_W = 360
TEXT_H = 110

# BGR colours matching the dark theme.
BG = (15, 10, 10)
SURFACE = (28, 20, 20)
SURFACE_2 = (38, 28, 28)
BORDER = (51, 38, 38)
TEXT = (238, 231, 231)
MUTED = (174, 154, 154)
ACCENT = (255, 124, 124)
ACCENT_2 = (238, 211, 34)
RED = (90, 90, 240)
FONT = cv2.FONT_HERSHEY_SIMPLEX

KEY_ESC, KEY_TAB, KEY_ENTER, KEY_RETURN = 27, 9, 13, 10
KEY_BACKSPACE = (8, 127)
# Keys that train the control gestures (letters train themselves).
CONTROL_KEYS = {ord("1"): SPACE, ord("2"): DELETE}


def short(label: str) -> str:
    return {SPACE: "SPC", DELETE: "DEL"}.get(label, label)


def ensure_model(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model (~8 MB) to {path} ...")
    try:
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    except Exception as e:  # noqa: BLE001 - surface any network error plainly
        sys.exit(
            f"Couldn't download the hand model: {e}\n"
            f"Download it manually from\n  {MODEL_URL}\nand save it as\n  {path}"
        )
    return path


def speak(text: str) -> None:
    """Read text aloud with the OS's built-in voice, without blocking."""
    text = text.strip()
    if not text:
        return
    system = platform.system()
    if system == "Darwin":
        cmd = ["say", text]
    elif system == "Windows":
        safe = text.replace("'", "''")
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')",
        ]
    else:
        tool = shutil.which("spd-say") or shutil.which("espeak-ng") or shutil.which("espeak")
        if not tool:
            print("Text-to-speech needs spd-say or espeak on Linux.")
            return
        cmd = [tool, text]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        print(f"Couldn't speak: {e}")


@dataclass
class Recording:
    label: str
    start_at: float
    last_at: float = 0.0
    captured: list = field(default_factory=list)


class App:
    def __init__(self, camera: int, samples_path: Path, model_path: Path):
        self.samples_path = samples_path
        self.samples = load_samples(samples_path)
        self.stabilizer = Stabilizer()
        self.mode = "read"  # read | train
        self.recording: Recording | None = None
        self.text = ""
        self.prediction = None
        self.message = ""
        self.message_until = 0.0

        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        if not self.cap.isOpened():
            sys.exit(
                f"Couldn't open camera {camera}. Check it's connected, not in use by "
                "another app, and that this terminal has camera permission."
            )

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        self.t0 = time.monotonic()
        self.last_ts = -1

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    self.flash("Camera stopped sending frames")
                    frame = np.zeros((CAM_H, CAM_W, 3), np.uint8)
                else:
                    frame = cv2.resize(cv2.flip(frame, 1), (CAM_W, CAM_H))
                    self.process(frame)
                cv2.imshow(WINDOW, self.render(frame))
                key = cv2.waitKey(1)
                if key != -1 and not self.handle_key(key & 0xFF):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.cap.release()
            self.landmarker.close()
            cv2.destroyAllWindows()

    def process(self, frame) -> None:
        now = time.monotonic()
        # MediaPipe VIDEO mode needs strictly increasing timestamps.
        ts = max(int((now - self.t0) * 1000), self.last_ts + 1)
        self.last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
        )
        self.draw_hand(frame, result)

        features = None
        if result.hand_landmarks:
            hand = (result.hand_world_landmarks or result.hand_landmarks)[0]
            handedness = result.handedness[0][0].category_name if result.handedness else "Right"
            features = landmarks_to_features(hand, handedness)

        rec = self.recording
        if rec:
            self.prediction = None
            if now >= rec.start_at and features and now - rec.last_at >= SAMPLE_INTERVAL_S:
                rec.last_at = now
                rec.captured.append(features)
                if len(rec.captured) >= SAMPLES_PER_RECORDING:
                    self.samples.setdefault(rec.label, []).extend(rec.captured)
                    self.save()
                    self.stabilizer.suppress(rec.label)
                    self.flash(f"Saved {len(rec.captured)} samples for {short(rec.label)}")
                    self.recording = None
            return

        self.prediction = classify(self.samples, features) if features else None
        if self.mode == "read":
            event = self.stabilizer.update(self.prediction, now)
            if event:
                self.text = apply_event(self.text, event)

    # -- input --------------------------------------------------------------

    def handle_key(self, key: int) -> bool:
        """Returns False to quit."""
        if key == KEY_ESC:
            if self.recording:
                self.recording = None
                self.flash("Recording cancelled")
                return True
            return False
        if key == KEY_TAB:
            self.mode = "train" if self.mode == "read" else "read"
            self.recording = None
            self.stabilizer.reset()
            return True

        if self.mode == "read":
            if key == ord(" "):
                self.text = apply_event(self.text, SPACE)
            elif key in KEY_BACKSPACE:
                self.text = apply_event(self.text, DELETE)
            elif key in (KEY_ENTER, KEY_RETURN):
                speak(self.text)
            elif key in (ord("c"), ord("C")):
                self.text = ""
            return True

        # train mode
        if self.recording:
            return True
        if ord("a") <= key <= ord("z"):
            self.start_recording(chr(key).upper())
        elif ord("A") <= key <= ord("Z"):  # Shift+letter
            self.clear_label(chr(key))
        elif key in CONTROL_KEYS:
            self.start_recording(CONTROL_KEYS[key])
        elif key in (ord("!"), ord("@")):  # Shift+1 / Shift+2
            self.clear_label(SPACE if key == ord("!") else DELETE)
        return True

    def start_recording(self, label: str) -> None:
        self.recording = Recording(label, start_at=time.monotonic() + COUNTDOWN_S)

    def clear_label(self, label: str) -> None:
        if self.samples.pop(label, None):
            self.save()
            self.flash(f"Cleared samples for {short(label)}")

    def save(self) -> None:
        try:
            save_samples(self.samples_path, self.samples)
        except OSError as e:
            self.flash(f"Couldn't save: {e}")

    def flash(self, message: str, seconds: float = 2.5) -> None:
        self.message = message
        self.message_until = time.monotonic() + seconds

    # -- drawing ------------------------------------------------------------

    def draw_hand(self, frame, result) -> None:
        for hand in result.hand_landmarks:
            pts = [(int(p.x * CAM_W), int(p.y * CAM_H)) for p in hand]
            for c in self.connections:
                cv2.line(frame, pts[c.start], pts[c.end], ACCENT, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 4, ACCENT_2, -1, cv2.LINE_AA)

    def render(self, frame):
        now = time.monotonic()
        canvas = np.full((CAM_H + TEXT_H, CAM_W + PANEL_W, 3), BG, np.uint8)
        canvas[:CAM_H, :CAM_W] = frame
        self.render_overlay(canvas, now)
        self.render_panel(canvas, CAM_W, now)
        self.render_transcript(canvas, CAM_H)
        return canvas

    def render_overlay(self, canvas, now) -> None:
        rec = self.recording
        if rec:
            shade = canvas[:CAM_H, :CAM_W].copy()
            cv2.rectangle(shade, (0, 0), (CAM_W, CAM_H), (0, 0, 0), -1)
            cv2.addWeighted(shade, 0.45, canvas[:CAM_H, :CAM_W], 0.55, 0, canvas[:CAM_H, :CAM_W])
            centered(canvas, short(rec.label), CAM_W // 2, CAM_H // 2 - 10, 4.0, TEXT, 8)
            if now < rec.start_at:
                line = f"Get ready... {int(rec.start_at - now) + 1}"
            else:
                line = f"Recording {len(rec.captured)}/{SAMPLES_PER_RECORDING} - move your hand slightly"
            centered(canvas, line, CAM_W // 2, CAM_H // 2 + 60, 0.7, TEXT, 2)
            centered(canvas, "Esc to cancel", CAM_W // 2, CAM_H // 2 + 95, 0.5, MUTED, 1)
            return

        # Prediction badge
        cv2.rectangle(canvas, (12, 12), (232, 84), SURFACE, -1)
        p = self.prediction
        cv2.putText(canvas, short(p.label) if p else "-", (24, 68), FONT, 1.6, TEXT, 3, cv2.LINE_AA)
        if p:
            info = f"{round(p.confidence * 100)}% sure"
        elif self.samples:
            info = "No hand"
        else:
            info = "Train first (Tab)"
        cv2.putText(canvas, info, (104, 42), FONT, 0.5, MUTED, 1, cv2.LINE_AA)
        if self.mode == "read":
            bar = int(116 * self.stabilizer.progress(now))
            cv2.rectangle(canvas, (104, 56), (220, 64), SURFACE_2, -1)
            if bar:
                cv2.rectangle(canvas, (104, 56), (104 + bar, 64), ACCENT_2, -1)

        if self.message and now < self.message_until:
            (w, _), _ = cv2.getTextSize(self.message, FONT, 0.6, 1)
            cv2.rectangle(canvas, (12, CAM_H - 46), (36 + w, CAM_H - 12), SURFACE, -1)
            cv2.putText(canvas, self.message, (24, CAM_H - 22), FONT, 0.6, TEXT, 1, cv2.LINE_AA)

    def render_panel(self, canvas, x0, now) -> None:
        x = x0 + 20
        mode = "TRAIN" if self.mode == "train" else "READ"
        cv2.putText(canvas, "SIGN READER", (x, 36), FONT, 0.75, TEXT, 2, cv2.LINE_AA)
        (w, _), _ = cv2.getTextSize(mode, FONT, 0.5, 1)
        cv2.rectangle(canvas, (x0 + PANEL_W - 36 - w, 18), (x0 + PANEL_W - 20, 44), ACCENT, -1)
        cv2.putText(canvas, mode, (x0 + PANEL_W - 28 - w, 37), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        trained = sum(1 for label in LABELS if self.samples.get(label))
        total = sum(len(v) for v in self.samples.values())
        cv2.putText(canvas, f"{trained}/{len(LABELS)} signs trained, {total} samples",
                    (x, 66), FONT, 0.45, MUTED, 1, cv2.LINE_AA)

        # Letter grid: 7 columns, a dot marks trained signs.
        cell, gap, top = 42, 4, 82
        for i, label in enumerate(LABELS):
            cx, cy = x + (i % 7) * (cell + gap), top + (i // 7) * (cell + gap)
            count = len(self.samples.get(label, []))
            active = self.recording and self.recording.label == label
            cv2.rectangle(canvas, (cx, cy), (cx + cell, cy + cell),
                          ACCENT if active else (SURFACE_2 if count else SURFACE), -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cell, cy + cell), BORDER, 1)
            centered(canvas, short(label), cx + cell // 2, cy + cell // 2 + 6,
                     0.55 if len(label) == 1 else 0.38, TEXT if count else MUTED, 1)
            if count:
                cv2.circle(canvas, (cx + cell - 6, cy + 6), 3, ACCENT_2, -1, cv2.LINE_AA)

        help_y = top + 4 * (cell + gap) + 20
        if self.mode == "read":
            lines = [
                "Hold a sign until the bar fills.",
                "Drop your hand for double letters;",
                "keep it down to add a space.",
                "",
                "Space / Backspace  edit text",
                "Enter  speak    C  clear",
                "Tab  train mode    Esc  quit",
            ]
        else:
            lines = [
                "Press a letter key to record it",
                f"({SAMPLES_PER_RECORDING} samples after a 3s countdown).",
                "1 / 2  record SPC / DEL gestures",
                "Shift+key  delete that sign's samples",
                "Record tricky letters 2-3 times.",
                "J, Z: hold the final handshape.",
                "Tab  read mode    Esc  quit",
            ]
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (x, help_y + i * 22), FONT, 0.45, MUTED, 1, cv2.LINE_AA)

    def render_transcript(self, canvas, y0) -> None:
        cv2.rectangle(canvas, (0, y0), (CAM_W + PANEL_W, y0 + TEXT_H), SURFACE, -1)
        cv2.line(canvas, (0, y0), (CAM_W + PANEL_W, y0), BORDER, 1)
        cv2.putText(canvas, "TRANSCRIPT", (20, y0 + 28), FONT, 0.45, MUTED, 1, cv2.LINE_AA)
        shown = self.text
        max_w = CAM_W + PANEL_W - 60
        while shown and cv2.getTextSize(shown + "_", FONT, 1.1, 2)[0][0] > max_w:
            shown = shown[1:]  # keep the end of long text visible
        cursor = "_" if int(time.monotonic() * 2) % 2 else " "
        if self.text:
            cv2.putText(canvas, shown + cursor, (20, y0 + 80), FONT, 1.1, TEXT, 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "Signed letters appear here...", (20, y0 + 80), FONT, 0.9, MUTED, 1, cv2.LINE_AA)


def centered(img, text, cx, cy, scale, color, thickness) -> None:
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.putText(img, text, (cx - w // 2, cy + h // 2 - 4), FONT, scale, color, thickness, cv2.LINE_AA)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Read ASL fingerspelling from your webcam.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--samples", type=Path, default=DATA_DIR / "samples.json",
                        help="where trained samples are stored (default ~/.sign_reader/samples.json)")
    parser.add_argument("--model", type=Path, default=DATA_DIR / "hand_landmarker.task",
                        help="MediaPipe hand model; downloaded on first run if missing")
    args = parser.parse_args(argv)
    App(args.camera, args.samples, ensure_model(args.model)).run()


if __name__ == "__main__":
    main()
