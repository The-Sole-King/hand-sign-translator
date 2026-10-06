"""Face recognition: remember people by name and recognize them later.

Uses OpenCV's SFace model to turn a face into a 128-number embedding. Faces
are located with the MediaPipe face landmarks the app already tracks, so no
separate face detector is needed. Embeddings are stored locally in
~/.sign_reader/faces.json and never leave the computer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# SFace's own recommended cosine-similarity threshold for "same person".
MATCH_THRESHOLD = 0.363

# MediaPipe face-mesh points matching the 5 points SFace aligns on.
IRIS_A, IRIS_B, NOSE_TIP, MOUTH_A, MOUTH_B = 468, 473, 1, 61, 291


def face_box(face, w: int, h: int) -> np.ndarray:
    """Build the [x, y, w, h, 5 landmark pairs, score] row that
    FaceRecognizerSF.alignCrop expects, from MediaPipe face landmarks."""
    xs = np.array([p.x for p in face]) * w
    ys = np.array([p.y for p in face]) * h
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()

    def pt(i):
        return xs[i], ys[i]

    # SFace expects the image-left eye/mouth corner first.
    eyes = sorted([pt(IRIS_A), pt(IRIS_B)])
    mouth = sorted([pt(MOUTH_A), pt(MOUTH_B)])
    row = [x0, y0, x1 - x0, y1 - y0, *eyes[0], *eyes[1], *pt(NOSE_TIP), *mouth[0], *mouth[1], 1.0]
    return np.array([row], np.float32)


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float32).ravel(), np.asarray(b, np.float32).ravel()
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9))


class FaceBook:
    """Named face embeddings, persisted as JSON ({name: [[128 floats], ...]})."""

    def __init__(self, path: Path):
        self.path = path
        self.people: dict[str, list[list[float]]] = {}
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                self.people = {
                    str(name): [list(map(float, e)) for e in embs if isinstance(e, list) and len(e) == 128]
                    for name, embs in data.items() if isinstance(embs, list)
                }
                self.people = {n: e for n, e in self.people.items() if e}
        except (OSError, ValueError):
            pass

    def add(self, name: str, embeddings: list) -> None:
        self.people.setdefault(name, []).extend(
            [round(float(x), 5) for x in np.asarray(e).ravel()] for e in embeddings
        )
        self.save()

    def forget_all(self) -> None:
        self.people = {}
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.people))
        tmp.replace(self.path)

    def match(self, embedding) -> tuple[str | None, float]:
        """Best (name, similarity), or (None, best similarity) if nobody matches."""
        best, best_score = None, -1.0
        for name, embs in self.people.items():
            score = max(cosine(embedding, e) for e in embs)
            if score > best_score:
                best, best_score = name, score
        return (best if best_score >= MATCH_THRESHOLD else None), best_score


class FaceRecognizer:
    """Wraps cv2.FaceRecognizerSF."""

    def __init__(self, model_path: Path):
        import cv2

        self._cv2 = cv2
        self.model = cv2.FaceRecognizerSF.create(str(model_path), "")

    def embed(self, frame_bgr, face) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        aligned = self.model.alignCrop(frame_bgr, face_box(face, w, h))
        return self.model.feature(aligned).ravel().copy()
