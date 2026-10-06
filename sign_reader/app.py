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
from .effects import draw_mask, draw_rasengan, open_app, open_url, palm_center
from .jutsu import Combo, ComboMatcher, load_combos

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DATA_DIR = Path.home() / ".sign_reader"
# Set when running as a PyInstaller-built app; bundled files live under it.
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "frozen", False) else None
SAMPLES_PER_RECORDING = 30
SAMPLE_INTERVAL_S = 0.08
COUNTDOWN_S = 3.0
RASENGAN_S = 12.0  # how long a Rasengan lasts
BANNER_S = 1.6

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
ORANGE = (40, 150, 255)
RED = (90, 90, 240)
FONT = cv2.FONT_HERSHEY_SIMPLEX

KEY_ESC, KEY_TAB, KEY_ENTER, KEY_RETURN = 27, 9, 13, 10
KEY_BACKSPACE = (8, 127)
# Keys that train the control gestures (letters train themselves).
CONTROL_KEYS = {ord("1"): SPACE, ord("2"): DELETE}


def short(label: str) -> str:
    return {SPACE: "SPC", DELETE: "DEL"}.get(label, label)


def log(message: str) -> None:
    # A windowed app on Windows has no console, so stdout can be None.
    if sys.stdout:
        print(message, flush=True)


def fail(message: str) -> None:
    """Report a fatal error and exit, with a dialog when there's no terminal."""
    log(message)
    if BUNDLE_DIR is not None:
        try:
            import tkinter
            from tkinter import messagebox

            root = tkinter.Tk()
            root.withdraw()
            messagebox.showerror(WINDOW, message)
            root.destroy()
        except Exception:  # noqa: BLE001 - the message was already logged
            pass
    sys.exit(1)


def default_model_path(filename: str = "hand_landmarker.task") -> Path:
    if BUNDLE_DIR is not None:
        bundled = BUNDLE_DIR / "assets" / filename
        if bundled.exists():
            return bundled
    return DATA_DIR / filename


def ensure_model(path: Path, url: str = MODEL_URL, what: str = "hand model (~8 MB)") -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {what} to {path} ...")
    try:
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    except Exception as e:  # noqa: BLE001 - surface any network error plainly
        fail(
            f"Couldn't download the {what}: {e}\n\n"
            f"Download it manually from\n{url}\nand save it as\n{path}"
        )
    return path


# CPU is plenty for one hand or face, and avoids MediaPipe's macOS Metal path,
# which aborts on machines without full GPU access.
def _base_options(model_path: Path) -> BaseOptions:
    return BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU)


def create_landmarker(model_path: Path):
    options = vision.HandLandmarkerOptions(
        base_options=_base_options(model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


def create_face_landmarker(model_path: Path):
    options = vision.FaceLandmarkerOptions(
        base_options=_base_options(model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def self_test(model_path: Path, face_model_path: Path) -> None:
    """Load the models and run one detection each, for checking a packaged build."""
    blank = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((CAM_H, CAM_W, 3), np.uint8))
    with create_landmarker(model_path) as landmarker:
        assert not landmarker.detect_for_video(blank, 0).hand_landmarks
    with create_face_landmarker(face_model_path) as landmarker:
        assert not landmarker.detect_for_video(blank, 0).face_landmarks
    log("self-test ok")


def open_file(path: Path) -> None:
    """Open a file in the OS's default app (used for editing combos)."""
    system = platform.system()
    if system == "Windows":
        import os

        os.startfile(path)
    elif system == "Darwin":
        subprocess.Popen(["open", "-t", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)], start_new_session=True)


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
            log("Text-to-speech needs spd-say or espeak on Linux.")
            return
        cmd = [tool, text]
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Don't flash a PowerShell window on Windows.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as e:
        log(f"Couldn't speak: {e}")


@dataclass
class Recording:
    label: str
    start_at: float
    last_at: float = 0.0
    captured: list = field(default_factory=list)


MODES = ("read", "train", "jutsu")


class App:
    def __init__(self, camera: int, samples_path: Path, model_path: Path,
                 face_model_path: Path, combos_path: Path):
        self.samples_path = samples_path
        self.samples = load_samples(samples_path)
        self.stabilizer = Stabilizer()
        self.mode = "read"  # one of MODES
        self.combos_path = combos_path
        self.matcher = ComboMatcher([])
        self.reload_combos()
        self.face_model_path = face_model_path
        self.face_landmarker = None  # created the first time the mask is used
        self.rasengan_at: float | None = None  # when the current Rasengan started
        self.mask_on = False
        self.banner = ""
        self.banner_until = 0.0
        self.recording: Recording | None = None
        self.text = ""
        self.prediction = None
        self.message = ""
        self.message_until = 0.0

        self.landmarker = create_landmarker(model_path)
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        if not self.cap.isOpened():
            self.landmarker.close()
            fail(
                f"Couldn't open camera {camera}.\n\nCheck it's connected, not in use by "
                "another app, and that Sign Reader is allowed to use the camera "
                "(macOS: System Settings > Privacy & Security > Camera)."
            )
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
            if self.face_landmarker:
                self.face_landmarker.close()
            cv2.destroyAllWindows()

    def process(self, frame) -> None:
        now = time.monotonic()
        # MediaPipe VIDEO mode needs strictly increasing timestamps.
        ts = max(int((now - self.t0) * 1000), self.last_ts + 1)
        self.last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, ts)
        if self.mode == "jutsu":
            self.draw_effects(frame, image, ts, result, now)
        else:
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
        elif self.mode == "jutsu":
            event = self.stabilizer.update(self.prediction, now)
            combo = self.matcher.feed(event, now) if event else None
            if combo:
                self.trigger(combo, now)

    # -- jutsu --------------------------------------------------------------

    def reload_combos(self) -> None:
        try:
            self.matcher = ComboMatcher(load_combos(self.combos_path))
            n = len(self.matcher.combos)
            self.flash(f"Loaded {n} combo{'s' * (n != 1)}")
        except (OSError, ValueError) as e:
            self.flash(f"Combos file error: {e}", seconds=8)

    def trigger(self, combo: Combo, now: float) -> None:
        self.banner = combo.name.upper() + "!"
        self.banner_until = now + BANNER_S
        try:
            if combo.action == "effect" and combo.target == "rasengan":
                self.rasengan_at = now
            elif combo.action == "effect" and combo.target == "mask":
                self.mask_on = not self.mask_on
            elif combo.action == "open":
                open_app(combo.target)
            elif combo.action == "url":
                open_url(combo.target)
        except OSError as e:
            self.flash(f"Couldn't {combo.describe}: {e}", seconds=5)

    def draw_effects(self, frame, image, ts: int, result, now: float) -> None:
        if self.mask_on:
            if self.face_landmarker is None:
                path = ensure_model(self.face_model_path, FACE_MODEL_URL, "face model (~4 MB)")
                self.face_landmarker = create_face_landmarker(path)
            faces = self.face_landmarker.detect_for_video(image, ts).face_landmarks
            if faces:
                draw_mask(frame, faces[0])

        if self.rasengan_at is not None and now - self.rasengan_at > RASENGAN_S:
            self.rasengan_at = None
        if self.rasengan_at is not None and result.hand_landmarks:
            center, radius = palm_center(result.hand_landmarks[0], CAM_W, CAM_H)
            draw_rasengan(frame, center, radius, now, grow=(now - self.rasengan_at) / 0.5)
        elif self.rasengan_at is None:
            self.draw_hand(frame, result)

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
            self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
            self.recording = None
            self.stabilizer.reset()
            self.matcher.reset()
            return True

        if self.mode == "jutsu":
            if key in (ord("x"), ord("X")):
                self.rasengan_at = None
                self.mask_on = False
                self.flash("Effects cleared")
            elif key in (ord("e"), ord("E")):
                try:
                    open_file(self.combos_path)
                except OSError as e:
                    self.flash(f"Couldn't open combos file: {e}", seconds=5)
            elif key in (ord("r"), ord("R")):
                self.reload_combos()
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
        if self.mode in ("read", "jutsu"):
            bar = int(116 * self.stabilizer.progress(now))
            cv2.rectangle(canvas, (104, 56), (220, 64), SURFACE_2, -1)
            if bar:
                cv2.rectangle(canvas, (104, 56), (104 + bar, 64), ACCENT_2, -1)

        if self.mode == "jutsu" and now < self.banner_until:
            # Pop in large, then settle.
            age = 1 - (self.banner_until - now) / BANNER_S
            scale = 1.6 + 0.6 * max(0.0, 1 - age * 4)
            centered(canvas, self.banner, CAM_W // 2 + 3, 123, scale, (0, 0, 0), 9)
            centered(canvas, self.banner, CAM_W // 2, 120, scale, ORANGE, 5)

        if self.message and now < self.message_until:
            (w, _), _ = cv2.getTextSize(self.message, FONT, 0.6, 1)
            cv2.rectangle(canvas, (12, CAM_H - 46), (36 + w, CAM_H - 12), SURFACE, -1)
            cv2.putText(canvas, self.message, (24, CAM_H - 22), FONT, 0.6, TEXT, 1, cv2.LINE_AA)

    def render_panel(self, canvas, x0, now) -> None:
        x = x0 + 20
        mode = self.mode.upper()
        cv2.putText(canvas, "SIGN READER", (x, 36), FONT, 0.75, TEXT, 2, cv2.LINE_AA)
        (w, _), _ = cv2.getTextSize(mode, FONT, 0.5, 1)
        cv2.rectangle(canvas, (x0 + PANEL_W - 36 - w, 18), (x0 + PANEL_W - 20, 44), ACCENT, -1)
        cv2.putText(canvas, mode, (x0 + PANEL_W - 28 - w, 37), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        trained = sum(1 for label in LABELS if self.samples.get(label))
        total = sum(len(v) for v in self.samples.values())
        cv2.putText(canvas, f"{trained}/{len(LABELS)} signs trained, {total} samples",
                    (x, 66), FONT, 0.45, MUTED, 1, cv2.LINE_AA)

        cell, gap, top = 42, 4, 82
        if self.mode == "jutsu":
            self.render_combos(canvas, x, top, now)
            return

        # Letter grid: 7 columns, a dot marks trained signs.
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
                "Tab  next mode    Esc  quit",
            ]
        else:
            lines = [
                "Press a letter key to record it",
                f"({SAMPLES_PER_RECORDING} samples after a 3s countdown).",
                "1 / 2  record SPC / DEL gestures",
                "Shift+key  delete that sign's samples",
                "Record tricky letters 2-3 times.",
                "J, Z: hold the final handshape.",
                "Tab  next mode    Esc  quit",
            ]
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (x, help_y + i * 22), FONT, 0.45, MUTED, 1, cv2.LINE_AA)

    def render_combos(self, canvas, x, top, now) -> None:
        progress = self.matcher.progress(now)
        y = top + 4
        if not self.matcher.combos:
            cv2.putText(canvas, "No combos. Press E to edit.", (x, y + 14), FONT, 0.5, MUTED, 1, cv2.LINE_AA)
        for combo in self.matcher.combos[:6]:
            done = progress.get(combo.name, 0)
            cv2.putText(canvas, combo.name, (x, y + 14), FONT, 0.55, TEXT, 1, cv2.LINE_AA)
            sx = x
            for i, sign in enumerate(combo.signs):
                label = short(sign)
                (w, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)
                color = ACCENT_2 if i < done else (MUTED if self.samples.get(sign) else RED)
                cv2.rectangle(canvas, (sx, y + 22), (sx + w + 12, y + 44), SURFACE_2, -1)
                cv2.putText(canvas, label, (sx + 6, y + 39), FONT, 0.5, color, 1, cv2.LINE_AA)
                sx += w + 18
            cv2.putText(canvas, combo.describe, (sx + 4, y + 39), FONT, 0.42, MUTED, 1, cv2.LINE_AA)
            y += 56
        lines = [
            "Sign each seal in order (3s apart max).",
            "Red seal = not trained yet.",
            "X  clear effects   E  edit combos",
            "R  reload combos   Tab  next mode",
        ]
        y = max(y + 6, 300)
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (x, y + i * 22), FONT, 0.45, MUTED, 1, cv2.LINE_AA)

    def render_transcript(self, canvas, y0) -> None:
        cv2.rectangle(canvas, (0, y0), (CAM_W + PANEL_W, y0 + TEXT_H), SURFACE, -1)
        cv2.line(canvas, (0, y0), (CAM_W + PANEL_W, y0), BORDER, 1)
        if self.mode == "jutsu":
            cv2.putText(canvas, "SEALS", (20, y0 + 28), FONT, 0.45, MUTED, 1, cv2.LINE_AA)
            seals = "  >  ".join(short(s) for s, _ in self.matcher.history)
            cv2.putText(canvas, seals or "Make a hand seal...", (20, y0 + 80), FONT,
                        1.1 if seals else 0.9, TEXT if seals else MUTED, 2 if seals else 1, cv2.LINE_AA)
            return
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
    parser.add_argument("--model", type=Path, default=default_model_path(),
                        help="MediaPipe hand model; downloaded on first run if missing")
    parser.add_argument("--face-model", type=Path, default=default_model_path("face_landmarker.task"),
                        help="MediaPipe face model for the mask effect; downloaded when first needed")
    parser.add_argument("--combos", type=Path, default=DATA_DIR / "combos.json",
                        help="hand-seal combos for jutsu mode (created with defaults if missing)")
    parser.add_argument("--self-test", action="store_true",
                        help="load the model, run one detection and exit (checks a build)")
    args = parser.parse_args(argv)
    model = ensure_model(args.model)
    if args.self_test:
        self_test(model, ensure_model(args.face_model, FACE_MODEL_URL, "face model (~4 MB)"))
        return
    App(args.camera, args.samples, model, args.face_model, args.combos).run()


if __name__ == "__main__":
    main()
