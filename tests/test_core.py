import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as P

from sign_reader.core import (
    DELETE,
    SPACE,
    Prediction,
    Stabilizer,
    apply_event,
    classify,
    landmarks_to_features,
    load_samples,
    parse_samples,
    save_samples,
)


def hand(seed=0.0, dx=0.0, dy=0.0, s=1.0):
    return [
        P(x=dx + s * math.sin(i + seed), y=dy + s * math.cos(i * 1.3 + seed), z=s * 0.1 * i)
        for i in range(21)
    ]


def pred(label, confidence=0.9):
    return Prediction(label, confidence, 0.0)


class FeatureTests(unittest.TestCase):
    def test_ignores_position_and_scale(self):
        a = landmarks_to_features(hand(1))
        b = landmarks_to_features(hand(1, dx=5, dy=-3, s=2.5))
        self.assertEqual(len(a), 60)
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y)

    def test_left_hand_is_mirrored_onto_right(self):
        right = hand(2)
        left = [P(x=-p.x, y=p.y, z=p.z) for p in right]
        for x, y in zip(landmarks_to_features(right, "Right"), landmarks_to_features(left, "Left")):
            self.assertAlmostEqual(x, y)

    def test_rejects_bad_input(self):
        self.assertIsNone(landmarks_to_features(None))
        self.assertIsNone(landmarks_to_features(hand()[:5]))
        self.assertIsNone(landmarks_to_features([P(x=0, y=0, z=0)] * 21))


class ClassifyTests(unittest.TestCase):
    def test_picks_nearest_class(self):
        samples = {
            "A": [landmarks_to_features(hand(0)), landmarks_to_features(hand(0.05))],
            "B": [landmarks_to_features(hand(3)), landmarks_to_features(hand(3.05))],
        }
        r = classify(samples, landmarks_to_features(hand(0.02)), k=3)
        self.assertEqual(r.label, "A")
        self.assertGreater(r.confidence, 0.5)
        self.assertIsNone(classify({}, landmarks_to_features(hand())))


class StabilizerTests(unittest.TestCase):
    def test_fires_once_per_hold(self):
        s = Stabilizer(hold_s=0.1)
        self.assertIsNone(s.update(pred("A"), 0))
        self.assertIsNone(s.update(pred("A"), 0.05))
        self.assertEqual(s.update(pred("A"), 0.1), "A")
        self.assertIsNone(s.update(pred("A"), 0.5))
        # a flicker to another label and back doesn't re-fire
        s.update(pred("B"), 0.51)
        self.assertIsNone(s.update(pred("A"), 0.52))
        self.assertIsNone(s.update(pred("A"), 0.7))
        # a real change does
        s.update(pred("B"), 0.8)
        self.assertEqual(s.update(pred("B"), 0.95), "B")

    def test_low_confidence_rearm_and_idle_space(self):
        s = Stabilizer(hold_s=0.1, idle_space_s=1.0)
        s.update(pred("A", 0.3), 0)
        self.assertIsNone(s.update(pred("A", 0.3), 0.5))
        s.update(pred("L"), 0.6)
        self.assertEqual(s.update(pred("L"), 0.75), "L")
        self.assertIsNone(s.update(None, 0.76))
        s.update(pred("L"), 0.8)
        self.assertEqual(s.update(pred("L"), 0.95), "L")
        self.assertEqual(s.update(None, 1.95), SPACE)
        self.assertIsNone(s.update(None, 5))

    def test_suppress_after_training(self):
        s = Stabilizer(hold_s=0.1)
        s.suppress("V")
        s.update(pred("V"), 0)
        self.assertIsNone(s.update(pred("V"), 0.5))
        s.update(None, 0.6)
        s.update(pred("V"), 0.7)
        self.assertEqual(s.update(pred("V"), 0.85), "V")


class TranscriptAndStorageTests(unittest.TestCase):
    def test_apply_event(self):
        self.assertEqual(apply_event("HI", "A"), "HIA")
        self.assertEqual(apply_event("HI", DELETE), "H")
        self.assertEqual(apply_event("HI", SPACE), "HI ")
        self.assertEqual(apply_event("HI ", SPACE), "HI ")
        self.assertEqual(apply_event("", SPACE), "")

    def test_model_round_trips_and_drops_junk(self):
        v = landmarks_to_features(hand(1))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nested" / "model.json"
            save_samples(path, {"A": [v], "B": []})
            loaded = load_samples(path)
            self.assertEqual(list(loaded), ["A"])
            self.assertAlmostEqual(loaded["A"][0][0], v[0], places=3)
            self.assertEqual(load_samples(Path(d) / "missing.json"), {})
        self.assertEqual(parse_samples({"A": [[1, 2]], "ZZ": [v], "C": "x"}), {})
        self.assertEqual(parse_samples([1, 2]), {})


if __name__ == "__main__":
    unittest.main()
