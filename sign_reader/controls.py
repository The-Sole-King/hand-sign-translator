"""Computer control from hands and face.

The detectors here are pure logic over MediaPipe landmarks/blendshapes (no
camera, GUI or OS access), so they're unit-tested in tests/test_controls.py.
`InputBackend` at the bottom is the only part that touches the real mouse
and keyboard, via pynput.
"""

from __future__ import annotations

import math
import platform

# -- keys ---------------------------------------------------------------------

# Friendly names -> pynput Key attribute names.
KEY_ALIASES = {
    "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "option": "alt", "shift": "shift",
    "win": "cmd", "windows": "cmd", "cmd": "cmd", "command": "cmd", "super": "cmd",
    "tab": "tab", "enter": "enter", "return": "enter", "esc": "esc", "escape": "esc",
    "space": "space", "backspace": "backspace", "delete": "delete", "del": "delete",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "home": "home", "end": "end", "page_up": "page_up", "pgup": "page_up",
    "page_down": "page_down", "pgdn": "page_down", "print_screen": "print_screen",
    "play_pause": "media_play_pause", "next_track": "media_next",
    "prev_track": "media_previous", "previous_track": "media_previous",
    "volume_up": "media_volume_up", "volume_down": "media_volume_down",
    "mute": "media_volume_mute",
    **{f"f{i}": f"f{i}" for i in range(1, 13)},
}


def parse_keys(spec: str) -> list[str]:
    """Parse "ctrl+shift+t" / "play_pause" into key names (pynput Key names or
    single characters). Raises ValueError on anything unknown."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("empty key combination")
    keys = []
    for part in spec.lower().replace(" ", "").split("+"):
        if part in KEY_ALIASES:
            keys.append(KEY_ALIASES[part])
        elif len(part) == 1 and part.isprintable():
            keys.append(part)
        else:
            raise ValueError(f"unknown key \"{part}\"")
    return keys


# -- geometry helpers -----------------------------------------------------------

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP = 0, 4, 5, 8, 9
FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))  # (tip, middle joint)


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def palm_size(hand) -> float:
    return _dist(hand[WRIST], hand[MIDDLE_MCP]) or 1e-6


def fingers_extended(hand) -> list[bool]:
    """Index, middle, ring, pinky: is each straightened away from the wrist?"""
    w = hand[WRIST]
    return [_dist(w, hand[tip]) > _dist(w, hand[pip]) * 1.15 for tip, pip in FINGERS]


def pinch_ratio(hand) -> float:
    return _dist(hand[THUMB_TIP], hand[INDEX_TIP]) / palm_size(hand)


# -- hand mouse -------------------------------------------------------------------

class HandMouse:
    """Turns hand landmarks (normalized, mirrored camera frame) into mouse
    actions: ("move", x, y), ("press",), ("release",), ("scroll", steps).

    - The cursor follows the point between thumb tip and index tip, which stays
      still while you pinch (both fingers close in on it).
    - Pinch thumb + index = mouse button down; open = up. So a quick pinch is a
      click and pinch-move-open is a drag.
    - Index + middle up, other fingers folded = scroll by moving the hand up/down.
    - Only the middle part of the camera image maps to the screen, so you can
      reach the edges without your hand leaving the frame.
    """

    PRESS, RELEASE = 0.28, 0.45  # pinch ratio hysteresis
    BOX = (0.2, 0.15, 0.8, 0.75)  # x0, y0, x1, y1 of the active area
    SCROLL_GAIN = 40  # scroll steps per full frame height

    def __init__(self, screen_w: int, screen_h: int):
        self.screen = (screen_w, screen_h)
        self.reset()

    def reset(self) -> None:
        self.pos: tuple[float, float] | None = None
        self.pressed = False
        self.scroll_y: float | None = None
        self.scroll_acc = 0.0
        self.mode = "idle"  # idle | point | pinch | scroll

    def update(self, hand) -> list[tuple]:
        actions: list[tuple] = []
        if hand is None:
            if self.pressed:
                actions.append(("release",))
            self.reset()
            return actions

        ratio = pinch_ratio(hand)
        if self.pressed and ratio > self.RELEASE:
            self.pressed = False
            actions.append(("release",))
        elif not self.pressed and ratio < self.PRESS:
            self.pressed = True
            actions.append(("press",))

        index, middle, ring, pinky = fingers_extended(hand)
        if not self.pressed and index and middle and not ring and not pinky:
            self.mode = "scroll"
            y = (hand[INDEX_TIP].y + hand[12].y) / 2
            if self.scroll_y is not None:
                # Hand up = scroll up (positive), like dragging a page.
                self.scroll_acc += (self.scroll_y - y) * self.SCROLL_GAIN
                steps = int(self.scroll_acc)
                if steps:
                    self.scroll_acc -= steps
                    actions.append(("scroll", steps))
            self.scroll_y = y
            return actions
        self.scroll_y = None
        self.scroll_acc = 0.0
        self.mode = "pinch" if self.pressed else "point"

        tx, ty = hand[THUMB_TIP], hand[INDEX_TIP]
        x0, y0, x1, y1 = self.BOX
        nx = min(1.0, max(0.0, ((tx.x + ty.x) / 2 - x0) / (x1 - x0)))
        ny = min(1.0, max(0.0, ((tx.y + ty.y) / 2 - y0) / (y1 - y0)))
        target = (nx * (self.screen[0] - 1), ny * (self.screen[1] - 1))
        if self.pos is None:
            self.pos = target
        else:
            # Smooth more when nearly still (kills jitter), less when moving fast.
            speed = math.dist(self.pos, target) / max(self.screen)
            a = min(0.85, 0.2 + speed * 8)
            self.pos = (self.pos[0] + a * (target[0] - self.pos[0]),
                        self.pos[1] + a * (target[1] - self.pos[1]))
        actions.append(("move", int(self.pos[0]), int(self.pos[1])))
        return actions


# -- swipes -----------------------------------------------------------------------

SWIPES = ("swipe_left", "swipe_right", "swipe_up", "swipe_down")


class SwipeDetector:
    """A fast sweep of the whole hand across the camera image."""

    def __init__(self, distance: float = 0.25, window: float = 0.45, cooldown: float = 0.8):
        self.distance, self.window, self.cooldown = distance, window, cooldown
        self.history: list[tuple[float, float, float]] = []
        self.blocked_until = 0.0

    def update(self, hand, now: float) -> str | None:
        if hand is None:
            # Fast motion blurs the hand and tracking drops a frame or two
            # mid-swipe; keep the history (it expires with the window anyway).
            return None
        x = sum(hand[i].x for i in (0, 5, 9, 13, 17)) / 5
        y = sum(hand[i].y for i in (0, 5, 9, 13, 17)) / 5
        self.history.append((now, x, y))
        while self.history and now - self.history[0][0] > self.window:
            self.history.pop(0)
        if now < self.blocked_until or len(self.history) < 3:
            return None
        _, x0, y0 = self.history[0]
        dx, dy = x - x0, y - y0
        if abs(dx) >= self.distance and abs(dx) > 2 * abs(dy):
            swipe = "swipe_right" if dx > 0 else "swipe_left"
        elif abs(dy) >= self.distance and abs(dy) > 2 * abs(dx):
            swipe = "swipe_down" if dy > 0 else "swipe_up"
        else:
            return None
        self.history.clear()
        self.blocked_until = now + self.cooldown
        return swipe


# -- face gestures -----------------------------------------------------------------

# name -> (how to score it from blendshapes/roll, threshold, seconds to hold)
FACE_GESTURES = {
    "mouth_open": (lambda b, roll: b.get("jawOpen", 0), 0.55, 0.35),
    "eyebrows_up": (lambda b, roll: b.get("browInnerUp", 0), 0.65, 0.35),
    "smile": (lambda b, roll: (b.get("mouthSmileLeft", 0) + b.get("mouthSmileRight", 0)) / 2, 0.7, 0.5),
    # Long, deliberate blink; normal blinks are much shorter.
    "long_blink": (lambda b, roll: min(b.get("eyeBlinkLeft", 0), b.get("eyeBlinkRight", 0)), 0.6, 0.7),
    "head_tilt_left": (lambda b, roll: -roll / 20, 1.0, 0.35),
    "head_tilt_right": (lambda b, roll: roll / 20, 1.0, 0.35),
}


def head_roll(face) -> float:
    """Head tilt in degrees from the line between the outer eye corners
    (positive = tilted toward the right side of the image)."""
    a, b = face[33], face[263]
    left, right = (a, b) if a.x < b.x else (b, a)
    return math.degrees(math.atan2(right.y - left.y, right.x - left.x))


class FaceGestureDetector:
    """Fires each face gesture once it's held, then waits for it to relax."""

    def __init__(self):
        self.since: dict[str, float] = {}
        self.fired: set[str] = set()

    def update(self, blendshapes: dict[str, float] | None, roll: float, now: float) -> list[str]:
        if blendshapes is None:
            self.since.clear()
            self.fired.clear()
            return []
        events = []
        for name, (score, threshold, hold) in FACE_GESTURES.items():
            value = score(blendshapes, roll)
            if value >= threshold:
                start = self.since.setdefault(name, now)
                if name not in self.fired and now - start >= hold:
                    self.fired.add(name)
                    events.append(name)
            elif value < threshold * 0.7:  # re-arm only once clearly relaxed
                self.since.pop(name, None)
                self.fired.discard(name)
        return events


GESTURES = SWIPES + tuple(FACE_GESTURES)


# -- real mouse / keyboard ------------------------------------------------------------

def screen_size() -> tuple[int, int]:
    """Primary screen size in the same units pynput uses for the cursor."""
    try:
        if platform.system() == "Windows":
            import ctypes

            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()  # real pixels, matching pynput
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        size = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return size
    except Exception:  # noqa: BLE001 - any failure: assume a common screen
        return 1920, 1080


class InputBackend:
    """Drives the real mouse and keyboard. pynput is imported on first use so
    the rest of the app still works where it can't load (e.g. Wayland)."""

    def __init__(self):
        from pynput import keyboard, mouse

        self._keyboard, self._mouse = keyboard, mouse
        self.kb = keyboard.Controller()
        self.ms = mouse.Controller()

    def run(self, action: tuple) -> None:
        kind = action[0]
        if kind == "move":
            self.ms.position = (action[1], action[2])
        elif kind == "press":
            self.ms.press(self._mouse.Button.left)
        elif kind == "release":
            self.ms.release(self._mouse.Button.left)
        elif kind == "scroll":
            self.ms.scroll(0, action[1])

    def keys(self, names: list[str]) -> None:
        keys = [getattr(self._keyboard.Key, n) if len(n) > 1 else n for n in names]
        for k in keys:
            self.kb.press(k)
        for k in reversed(keys):
            self.kb.release(k)
