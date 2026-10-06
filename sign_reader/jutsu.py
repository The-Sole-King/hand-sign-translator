"""Combos: triggers (hand-seal sequences, gestures, people arriving) mapped
to actions.

Kept free of OpenCV/MediaPipe so it can be unit-tested (see tests/test_jutsu.py).
Combos live in a JSON file the user can edit, e.g.

    [
      {"name": "Rasengan", "signs": ["B", "C"], "effect": "rasengan"},
      {"name": "Firefox",  "signs": ["F", "O"], "open": "firefox"},
      {"name": "Next",     "gesture": "swipe_right", "keys": "right"},
      {"name": "Pause",    "gesture": "mouth_open", "keys": "play_pause"},
      {"name": "Morning",  "arrive": "Rushd", "url": "https://calendar.google.com"}
    ]

Each combo has exactly one trigger (`signs`, `gesture` or `arrive`) and one
action: an on-screen `effect`, an app to `open`, a `url`, or `keys` to press.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .controls import GESTURES, parse_keys
from .core import LABELS, SPACE

EFFECTS = ("rasengan", "mask")
TRIGGERS = ("signs", "gesture", "arrive")
ACTIONS = ("effect", "open", "url", "keys")

# The first release's defaults; files still matching them get upgraded.
OLD_DEFAULT_COMBOS = [
    # Open palm, then cup the hand as if holding a ball.
    {"name": "Rasengan", "signs": ["B", "C"], "effect": "rasengan"},
    # Signing it again takes the mask off.
    {"name": "Kakashi Mask", "signs": ["M", "K"], "effect": "mask"},
    {"name": "Firefox", "signs": ["F", "O"], "open": "firefox"},
]

DEFAULT_COMBOS = [
    # Open palm, then cup the hand as if holding a ball.
    {"name": "Rasengan", "signs": ["B", "C"], "effect": "rasengan"},
    # Signing it again takes the mask off.
    {"name": "Kakashi Mask", "signs": ["M", "K"], "effect": "mask"},
    {"name": "Firefox", "signs": ["F", "O"], "open": "firefox"},
    {"name": "Next", "gesture": "swipe_right", "keys": "right"},
    {"name": "Back", "gesture": "swipe_left", "keys": "left"},
    {"name": "Louder", "gesture": "swipe_up", "keys": "volume_up"},
    {"name": "Quieter", "gesture": "swipe_down", "keys": "volume_down"},
    {"name": "Play/Pause", "gesture": "mouth_open", "keys": "play_pause"},
    {"name": "Mute", "gesture": "eyebrows_up", "keys": "mute"},
]


@dataclass(frozen=True)
class Combo:
    name: str
    trigger: str  # "signs" | "gesture" | "arrive"
    when: tuple[str, ...]  # the signs, or (gesture,), or (person,)
    action: str  # "effect" | "open" | "url" | "keys"
    target: str  # effect name, app name, URL or key combination

    @property
    def signs(self) -> tuple[str, ...]:
        return self.when if self.trigger == "signs" else ()

    @property
    def describe(self) -> str:
        if self.action == "effect":
            return self.target
        if self.action == "keys":
            return f"press {self.target}"
        return f"open {self.target}"


def parse_combos(data) -> list[Combo]:
    """Validate combos loaded from JSON. Raises ValueError with a readable
    message on the first bad entry, so the user can fix their file."""
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, list):
        raise ValueError("combos file must contain a JSON list")
    combos = []
    for i, item in enumerate(data, 1):
        where = f"combo #{i}"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{where} needs a \"name\"")
        where = f"combo \"{name}\""
        triggers = [t for t in TRIGGERS if t in item]
        if len(triggers) != 1:
            raise ValueError(f"{where} needs exactly one of \"signs\", \"gesture\" or \"arrive\"")
        trigger = triggers[0]
        if trigger == "signs":
            signs = item["signs"]
            if not isinstance(signs, list) or not signs:
                raise ValueError(f"{where} needs a non-empty \"signs\" list")
            signs = [s.upper() if isinstance(s, str) else s for s in signs]
            bad = [s for s in signs if s not in LABELS or s == SPACE]
            if bad:
                raise ValueError(f"{where} has unknown signs {bad}; use letters A-Z or DEL")
            when = tuple(signs)
        elif trigger == "gesture":
            gesture = item["gesture"]
            if gesture not in GESTURES:
                raise ValueError(f"{where}: unknown gesture \"{gesture}\" (choose from {', '.join(GESTURES)})")
            when = (gesture,)
        else:
            person = item["arrive"]
            if not isinstance(person, str) or not person.strip():
                raise ValueError(f"{where}: \"arrive\" must be a person's name")
            when = (person.strip(),)
        actions = [a for a in ACTIONS if a in item]
        if len(actions) != 1:
            raise ValueError(f"{where} needs exactly one of \"effect\", \"open\" or \"url\"")
        action = actions[0]
        target = item[action]
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"{where}: \"{action}\" must be a non-empty string")
        if action == "effect" and target not in EFFECTS:
            raise ValueError(f"{where}: unknown effect \"{target}\" (choose from {', '.join(EFFECTS)})")
        if action == "url" and not target.startswith(("http://", "https://")):
            raise ValueError(f"{where}: \"url\" must start with http:// or https://")
        if action == "keys":
            try:
                parse_keys(target)
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from None
        combos.append(Combo(name.strip(), trigger, when, action, target.strip()))
    return combos


def load_combos(path: Path) -> list[Combo]:
    """Load combos, creating the file with the defaults on first run."""
    if not path.exists() or _is_old_default(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULT_COMBOS, indent=2) + "\n")
    return parse_combos(path.read_text())


def _is_old_default(path: Path) -> bool:
    try:
        return json.loads(path.read_text()) == OLD_DEFAULT_COMBOS
    except (OSError, ValueError):
        return False


class ComboMatcher:
    """Matches the stream of signed letters against combos.

    Each sign in a combo must follow the previous one within `step_timeout`
    seconds. When several combos match (one ending another), the longest wins.
    """

    def __init__(self, combos: list[Combo], step_timeout: float = 3.0):
        self.all = combos
        self.combos = [c for c in combos if c.trigger == "signs"]
        self.step_timeout = step_timeout
        self.history: list[tuple[str, float]] = []

    def _expire(self, now: float) -> None:
        if self.history and now - self.history[-1][1] > self.step_timeout:
            self.history.clear()

    def feed(self, label: str, now: float) -> Combo | None:
        """Feed one emitted sign. Returns the combo it completes, if any."""
        if label == SPACE:  # idle gaps aren't seals
            return None
        self._expire(now)
        self.history.append((label, now))
        longest = max((len(c.signs) for c in self.combos), default=0)
        del self.history[:-longest or None]
        signs = [s for s, _ in self.history]
        done = [c for c in self.combos if signs[-len(c.signs):] == list(c.signs)]
        if not done:
            return None
        self.history.clear()
        return max(done, key=lambda c: len(c.signs))

    def on(self, trigger: str, value: str) -> list[Combo]:
        """Combos fired by a gesture or a person arriving."""
        return [c for c in self.all if c.trigger == trigger and c.when == (value,)]

    def progress(self, now: float) -> dict[str, int]:
        """How many leading signs of each combo have been made so far."""
        self._expire(now)
        signs = [s for s, _ in self.history]
        out = {}
        for c in self.combos:
            out[c.name] = next(
                (n for n in range(min(len(c.signs) - 1, len(signs)), 0, -1)
                 if signs[-n:] == list(c.signs[:n])),
                0,
            )
        return out

    def reset(self) -> None:
        self.history.clear()
