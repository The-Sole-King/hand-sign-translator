"""On-screen effects and app launching for jutsu combos."""

from __future__ import annotations

import math
import platform
import subprocess
import webbrowser

import cv2
import numpy as np

# Palm landmarks: wrist + the four knuckles.
PALM = (0, 5, 9, 13, 17)
# Face-mesh points outlining a Kakashi-style mask: cheek, down around the jaw
# and chin to the other cheek, then back up over the bridge of the nose.
MASK_OUTLINE = (
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
    # back over the face, below the eyes, through the nose bridge
    447, 345, 352, 280, 330, 6, 101, 50, 123, 116, 227,
)
MASK_FOLDS = ((205, 2, 425), (207, 0, 427))  # soft cloth creases


def palm_center(hand, w: int, h: int) -> tuple[tuple[int, int], int]:
    """Pixel centre of the palm and a radius that scales with hand size."""
    xs = [hand[i].x * w for i in PALM]
    ys = [hand[i].y * h for i in PALM]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    size = math.hypot((hand[9].x - hand[0].x) * w, (hand[9].y - hand[0].y) * h)
    return (int(cx), int(cy)), max(12, int(size * 0.75))


def soft(img, sigma: float):
    """Gaussian blur done at quarter resolution: large glows look the same
    but cost a fraction of a full-size blur."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / 4))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def draw_rasengan(frame, center, radius: int, t: float, grow: float = 1.0) -> None:
    """Draw a swirling ball of chakra, additively blended into `frame`."""
    r = max(4, int(radius * (0.3 + 0.7 * min(1.0, grow))))
    h, w = frame.shape[:2]
    pad = r * 3
    x0, y0 = max(0, center[0] - pad), max(0, center[1] - pad)
    x1, y1 = min(w, center[0] + pad), min(h, center[1] + pad)
    if x1 <= x0 or y1 <= y0:
        return
    c = (center[0] - x0, center[1] - y0)
    layer = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)

    # Faint aura, a blue body and a small white-hot core.
    cv2.circle(layer, c, int(r * 1.3), (90, 40, 0), -1, cv2.LINE_AA)
    cv2.circle(layer, c, r, (200, 110, 20), -1, cv2.LINE_AA)
    cv2.circle(layer, c, int(r * 0.35), (255, 220, 150), -1, cv2.LINE_AA)
    cv2.circle(layer, c, int(r * 0.15), (255, 255, 255), -1, cv2.LINE_AA)
    glow = soft(layer, max(2, r * 0.35))

    # Spinning swirl streaks drawn on top, kept sharp so they read as motion.
    streaks = np.zeros_like(layer)
    for i in range(12):
        rr = int(r * (0.3 + 0.06 * i))
        start = (t * (480 + 50 * i) + i * 47) % 360
        sweep = 60 + (i * 23) % 70
        tilt = (i * 31 + t * 120) % 180
        color = (255, 255, 255) if i % 3 == 0 else (255, 200, 90)
        cv2.ellipse(streaks, c, (rr, int(rr * 0.7)), tilt, start, start + sweep,
                    color, 2 if r > 30 else 1, cv2.LINE_AA)

    # Wind particles orbiting the edge.
    for i in range(18):
        a = t * (5 + i % 4) + i * 2.1
        d = r * (1.05 + 0.3 * math.sin(t * 3 + i))
        p = (int(c[0] + d * math.cos(a)), int(c[1] + d * math.sin(a) * 0.8))
        cv2.circle(streaks, p, 2, (255, 230, 180), -1, cv2.LINE_AA)

    roi = frame[y0:y1, x0:x1]
    # Darken behind the ball a little so it pops against bright backgrounds.
    shade = soft((layer[..., 2] > 0).astype(np.uint8) * 90, max(2, r * 0.3))
    cv2.multiply(roi, cv2.cvtColor(255 - shade, cv2.COLOR_GRAY2BGR), roi, scale=1 / 255)
    cv2.add(roi, glow, roi)
    # A solid sphere body under the streaks, with a soft edge.
    body = np.zeros_like(layer)
    cv2.circle(body, c, int(r * 0.9), (150, 80, 10), -1, cv2.LINE_AA)
    cv2.circle(body, c, int(r * 0.55), (210, 140, 40), -1, cv2.LINE_AA)
    cv2.add(roi, soft(body, max(1.5, r * 0.12)), roi)
    cv2.add(roi, cv2.GaussianBlur(streaks, (0, 0), 1.2), roi)
    cv2.add(roi, streaks // 2, roi)


def draw_mask(frame, face) -> None:
    """Draw a dark cloth mask over the nose, mouth and jaw."""
    h, w = frame.shape[:2]
    pts = np.array([(int(face[i].x * w), int(face[i].y * h)) for i in MASK_OUTLINE], np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (58, 40, 32), cv2.LINE_AA)
    for fold in MASK_FOLDS:
        line = np.array([(int(face[i].x * w), int(face[i].y * h)) for i in fold], np.int32)
        cv2.polylines(overlay, [line], False, (42, 28, 22), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
    cv2.polylines(frame, [pts], True, (30, 20, 16), 2, cv2.LINE_AA)


def open_app(name: str) -> None:
    """Launch an installed app by name (e.g. "firefox")."""
    system = platform.system()
    if system == "Windows":
        import os

        os.startfile(name)  # resolves registered apps like firefox.exe
    elif system == "Darwin":
        subprocess.Popen(["open", "-a", name])
    else:
        subprocess.Popen([name], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def open_url(url: str) -> None:
    webbrowser.open(url)
