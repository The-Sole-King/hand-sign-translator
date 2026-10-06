"""Camera- and GUI-free core of the sign reader.

Turns MediaPipe hand landmarks into a pose feature vector, classifies it with
k-nearest-neighbours against samples the user recorded, and debounces the
frame-by-frame predictions into typed letters. Kept free of OpenCV/MediaPipe so
it can be unit-tested on its own (see tests/test_core.py).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
# Control gestures the user can teach alongside the alphabet.
SPACE = "SPACE"
DELETE = "DEL"
LABELS = LETTERS + [SPACE, DELETE]

FEATURE_SIZE = 60  # 20 landmarks (wrist dropped) x 3 coordinates
WRIST = 0
MIDDLE_MCP = 9


def landmarks_to_features(landmarks, handedness: str = "Right") -> list[float] | None:
    """Normalize 21 hand landmarks into a 60-number vector.

    Ignores where the hand is in frame, how far it is from the camera and which
    hand it is:
    - translate so the wrist is the origin (the wrist itself is then dropped)
    - scale by the wrist -> middle-knuckle distance (palm size)
    - mirror x for left hands so both hands share one set of samples

    Orientation is deliberately kept: ASL distinguishes e.g. H from U and P
    from K mainly by which way the hand points.

    `landmarks` is any sequence of 21 objects with .x, .y and (optional) .z.
    """
    if not landmarks or len(landmarks) != 21:
        return None
    o, m = landmarks[WRIST], landmarks[MIDDLE_MCP]
    oz, mz = getattr(o, "z", 0) or 0, getattr(m, "z", 0) or 0
    scale = math.dist((m.x, m.y, mz), (o.x, o.y, oz))
    if not scale:
        return None
    flip = -1 if handedness == "Left" else 1
    out: list[float] = []
    for p in landmarks[1:]:
        pz = getattr(p, "z", 0) or 0
        out += [flip * (p.x - o.x) / scale, (p.y - o.y) / scale, (pz - oz) / scale]
    return out


@dataclass
class Prediction:
    label: str
    confidence: float  # inverse-distance-weighted vote share among k neighbours
    distance: float  # distance to the nearest sample of the winning label


def classify(samples: dict[str, list[list[float]]], features, k: int = 5) -> Prediction | None:
    """k-NN over user-recorded samples ({label: [vector, ...]})."""
    if features is None:
        return None
    neighbours = sorted(
        (math.dist(v, features), label) for label, vectors in samples.items() for v in vectors
    )
    if not neighbours:
        return None
    top = neighbours[:k]
    votes: dict[str, float] = {}
    for d, label in top:
        votes[label] = votes.get(label, 0.0) + 1 / (d + 1e-6)
    best = max(votes, key=votes.get)
    return Prediction(
        label=best,
        confidence=votes[best] / sum(votes.values()),
        distance=next(d for d, label in top if label == best),
    )


class Stabilizer:
    """Debounces noisy per-frame predictions into discrete events.

    A label is emitted once it has been held steadily for `hold_s` seconds. The
    same label won't fire again until a different one fires or the hand leaves
    the frame (drop your hand briefly to sign a double letter like the "LL" in
    HELLO). After `idle_space_s` with no hand visible, one SPACE is emitted so
    words separate naturally.
    """

    def __init__(self, hold_s: float = 0.6, min_confidence: float = 0.6, idle_space_s: float = 1.5):
        self.hold_s = hold_s
        self.min_confidence = min_confidence
        self.idle_space_s = idle_space_s
        self.reset()

    def reset(self) -> None:
        self.candidate: str | None = None  # label currently being held
        self.since = 0.0  # when the candidate started
        self.fired: str | None = None  # label already emitted for this hold
        self.last_seen: float | None = None  # last time a hand was visible
        self.spaced = True  # whether the idle space was already emitted

    def suppress(self, label: str) -> None:
        """Treat `label` as already emitted, e.g. right after recording samples
        for it, so the pose still being held doesn't immediately get typed."""
        self.reset()
        self.fired = label

    def update(self, prediction: Prediction | None, now: float) -> str | None:
        """Feed one frame (None = no hand). Returns an emitted label or None."""
        if prediction is None:
            self.candidate = None
            self.fired = None
            if (
                not self.spaced
                and self.last_seen is not None
                and now - self.last_seen >= self.idle_space_s
            ):
                self.spaced = True
                return SPACE
            return None

        self.last_seen = now
        label = prediction.label if prediction.confidence >= self.min_confidence else None
        if label != self.candidate:
            # `fired` is only cleared once a different label actually fires or
            # the hand leaves, so a one-frame flicker (A -> B -> A) can't
            # re-emit A.
            self.candidate = label
            self.since = now
            return None
        if label and label != self.fired and now - self.since >= self.hold_s:
            self.fired = label
            if label != SPACE:
                self.spaced = False
            return label
        return None

    def progress(self, now: float) -> float:
        """0..1 progress of the current hold, for the on-screen bar."""
        if not self.candidate or self.candidate == self.fired:
            return 0.0
        return min(1.0, (now - self.since) / self.hold_s)


def apply_event(text: str, event: str) -> str:
    """Apply a stabilizer event to the transcript text."""
    if event == DELETE:
        return text[:-1]
    if event == SPACE:
        return text + " " if text and not text.endswith(" ") else text
    return text + event


def parse_samples(data) -> dict[str, list[list[float]]]:
    """Validate a loaded model, dropping anything malformed."""
    if isinstance(data, str):
        data = json.loads(data)
    out: dict[str, list[list[float]]] = {}
    if not isinstance(data, dict):
        return out
    for label in LABELS:
        vectors = data.get(label)
        if not isinstance(vectors, list):
            continue
        valid = [
            [float(n) for n in v]
            for v in vectors
            if isinstance(v, list)
            and len(v) == FEATURE_SIZE
            and all(isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n) for n in v)
        ]
        if valid:
            out[label] = valid
    return out


def load_samples(path: Path) -> dict[str, list[list[float]]]:
    try:
        return parse_samples(path.read_text())
    except (OSError, ValueError):
        return {}


def save_samples(path: Path, samples: dict[str, list[list[float]]]) -> None:
    """Write the model atomically, rounded to keep the file small."""
    compact = {
        label: [[round(n, 3) for n in v] for v in vectors]
        for label, vectors in samples.items()
        if vectors
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(compact))
    tmp.replace(path)
