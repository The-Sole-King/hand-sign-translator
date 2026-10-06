import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as P

import numpy as np

from sign_reader.controls import (
    FaceGestureDetector,
    HandMouse,
    SwipeDetector,
    fingers_extended,
    head_roll,
    parse_keys,
)
from sign_reader.faces import MATCH_THRESHOLD, FaceBook
from sign_reader.jutsu import DEFAULT_COMBOS, OLD_DEFAULT_COMBOS, ComboMatcher, load_combos, parse_combos


def hand(cx=0.5, cy=0.5, pinch=False, pose="open"):
    """A rough right hand: wrist below, fingers pointing up."""
    pts = [P(x=cx, y=cy, z=0) for _ in range(21)]
    pts[0] = P(x=cx, y=cy + 0.2, z=0)  # wrist
    pts[9] = P(x=cx, y=cy, z=0)  # middle knuckle
    for k, (tip, pip, mcp) in enumerate(((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17))):
        x = cx - 0.03 + k * 0.02
        up = pose == "open" or (pose == "two" and k < 2)
        pts[mcp] = P(x=x, y=cy, z=0)
        pts[pip] = P(x=x, y=cy - 0.05, z=0)
        pts[tip] = P(x=x, y=cy - (0.12 if up else 0.0), z=0)
    pts[4] = P(x=pts[8].x + (0.01 if pinch else 0.12), y=pts[8].y, z=0)  # thumb tip
    for i in (1, 2, 3):
        pts[i] = P(x=cx - 0.06, y=cy + 0.1, z=0)
    return pts


class KeyTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_keys("ctrl+shift+t"), ["ctrl", "shift", "t"])
        self.assertEqual(parse_keys("Win + D"), ["cmd", "d"])
        self.assertEqual(parse_keys("play_pause"), ["media_play_pause"])
        for bad in ("", "ctrl+banana", "f13"):
            with self.assertRaises(ValueError):
                parse_keys(bad)


class HandMouseTests(unittest.TestCase):
    def test_point_click_drag(self):
        m = HandMouse(1000, 500)
        (move,) = m.update(hand(0.5, 0.45))
        self.assertEqual(move[0], "move")
        self.assertAlmostEqual(move[1], 500, delta=60)
        acts = m.update(hand(0.5, 0.45, pinch=True))
        self.assertEqual(acts[0], ("press",))
        acts = m.update(hand(0.6, 0.45, pinch=True))  # drag
        self.assertEqual(acts[0][0], "move")
        self.assertEqual(m.update(hand(0.6, 0.45))[0], ("release",))

    def test_edges_clamp_and_lost_hand_releases(self):
        m = HandMouse(1000, 500)
        for _ in range(30):
            acts = m.update(hand(0.05, 0.05))
        self.assertEqual(acts[-1][1:], (0, 0))
        m.update(hand(0.5, 0.5, pinch=True))
        self.assertEqual(m.update(None), [("release",)])

    def test_two_finger_scroll(self):
        m = HandMouse(1000, 500)
        self.assertEqual(fingers_extended(hand(pose="two")), [True, True, False, False])
        m.update(hand(0.5, 0.6, pose="two"))
        acts = m.update(hand(0.5, 0.4, pose="two"))  # hand up
        self.assertEqual(m.mode, "scroll")
        self.assertEqual(acts[0][0], "scroll")
        self.assertGreater(acts[0][1], 0)


class SwipeTests(unittest.TestCase):
    def test_swipes_and_cooldown(self):
        d = SwipeDetector()
        events = [d.update(hand(0.3 + 0.1 * i, 0.5), i * 0.05) for i in range(5)]
        self.assertIn("swipe_right", events)
        self.assertIsNone(d.update(hand(0.2, 0.5), 0.3))  # cooldown
        d.update(None, 2)
        events = [d.update(hand(0.5, 0.8 - 0.1 * i), 2 + i * 0.05) for i in range(5)]
        self.assertIn("swipe_up", events)

    def test_dropped_frames_mid_swipe(self):
        d = SwipeDetector()
        frames = [hand(0.2, 0.5), None, hand(0.4, 0.5), None, None, hand(0.6, 0.5)]
        events = [d.update(f, i * 0.06) for i, f in enumerate(frames)]
        self.assertIn("swipe_right", events)

    def test_slow_motion_is_not_a_swipe(self):
        d = SwipeDetector()
        events = [d.update(hand(0.3 + 0.01 * i, 0.5), i * 0.1) for i in range(40)]
        self.assertEqual(set(events), {None})


class FaceGestureTests(unittest.TestCase):
    def test_hold_fire_once_and_rearm(self):
        d = FaceGestureDetector()
        self.assertEqual(d.update({"jawOpen": 0.9}, 0, 0), [])
        self.assertEqual(d.update({"jawOpen": 0.9}, 0, 0.4), ["mouth_open"])
        self.assertEqual(d.update({"jawOpen": 0.9}, 0, 1.0), [])
        d.update({"jawOpen": 0.1}, 0, 1.1)
        d.update({"jawOpen": 0.9}, 0, 1.2)
        self.assertEqual(d.update({"jawOpen": 0.9}, 0, 1.6), ["mouth_open"])

    def test_short_blink_ignored_tilt_detected(self):
        d = FaceGestureDetector()
        blink = {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9}
        d.update(blink, 0, 0)
        self.assertEqual(d.update({}, 0, 0.2), [])
        d.update({}, 30, 1)
        self.assertEqual(d.update({}, 30, 1.4), ["head_tilt_right"])

    def test_head_roll(self):
        face = [P(x=0, y=0)] * 264
        face = list(face)
        face[33], face[263] = P(x=0.4, y=0.5), P(x=0.6, y=0.6)
        self.assertAlmostEqual(head_roll(face), 26.57, places=1)


class FaceBookTests(unittest.TestCase):
    def test_add_match_persist(self):
        rng = np.random.default_rng(0)
        alice, bob = rng.normal(size=128), rng.normal(size=128)
        with tempfile.TemporaryDirectory() as d:
            book = FaceBook(Path(d) / "faces.json")
            book.add("Alice", [alice, alice + rng.normal(scale=0.1, size=128)])
            book.add("Bob", [bob])
            reloaded = FaceBook(Path(d) / "faces.json")
            name, score = reloaded.match(alice + rng.normal(scale=0.2, size=128))
            self.assertEqual(name, "Alice")
            self.assertGreater(score, MATCH_THRESHOLD)
            self.assertEqual(reloaded.match(rng.normal(size=128))[0], None)
            reloaded.forget_all()
            self.assertEqual(FaceBook(Path(d) / "faces.json").people, {})


class TriggerTests(unittest.TestCase):
    def test_gesture_and_arrive_combos(self):
        combos = parse_combos([
            {"name": "Next", "gesture": "swipe_right", "keys": "right"},
            {"name": "Hi", "arrive": "Rushd", "url": "https://example.com"},
            {"name": "R", "signs": ["B", "C"], "effect": "rasengan"},
        ])
        m = ComboMatcher(combos)
        self.assertEqual([c.name for c in m.on("gesture", "swipe_right")], ["Next"])
        self.assertEqual([c.name for c in m.on("arrive", "Rushd")], ["Hi"])
        self.assertEqual([c.name for c in m.combos], ["R"])  # only sign combos match seals
        for bad in (
            {"name": "x", "gesture": "wave", "keys": "a"},
            {"name": "x", "gesture": "smile", "keys": "ctrl+banana"},
            {"name": "x", "signs": ["A"], "gesture": "smile", "keys": "a"},
        ):
            with self.assertRaises(ValueError, msg=bad):
                parse_combos([bad])

    def test_defaults_valid_and_old_defaults_upgraded(self):
        parse_combos(DEFAULT_COMBOS)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "combos.json"
            path.write_text(json.dumps(OLD_DEFAULT_COMBOS))
            load_combos(path)
            self.assertEqual(json.loads(path.read_text()), DEFAULT_COMBOS)
            custom = [{"name": "x", "signs": ["A"], "keys": "a"}]
            path.write_text(json.dumps(custom))
            load_combos(path)  # user edits are never overwritten
            self.assertEqual(json.loads(path.read_text()), custom)


if __name__ == "__main__":
    unittest.main()
