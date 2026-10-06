import json
import tempfile
import unittest
from pathlib import Path

from sign_reader.core import SPACE
from sign_reader.jutsu import DEFAULT_COMBOS, ComboMatcher, load_combos, parse_combos

COMBOS = parse_combos([
    {"name": "Rasengan", "signs": ["B", "C"], "effect": "rasengan"},
    {"name": "Big", "signs": ["A", "B", "C"], "effect": "mask"},
    {"name": "Fox", "signs": ["F", "O"], "open": "firefox"},
])


class ParseTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        self.assertEqual(len(parse_combos(DEFAULT_COMBOS)), len(DEFAULT_COMBOS))

    def test_lowercase_signs_are_accepted(self):
        (c,) = parse_combos([{"name": "x", "signs": ["b", "c"], "url": "https://a.b"}])
        self.assertEqual(c.signs, ("B", "C"))

    def test_rejects_bad_entries(self):
        bad = [
            {},
            {"name": "x", "signs": [], "effect": "mask"},
            {"name": "x", "signs": ["1"], "effect": "mask"},
            {"name": "x", "signs": ["A"]},
            {"name": "x", "signs": ["A"], "effect": "mask", "open": "y"},
            {"name": "x", "signs": ["A"], "effect": "fireball"},
            {"name": "x", "signs": ["A"], "url": "file:///etc/passwd"},
        ]
        for entry in bad:
            with self.assertRaises(ValueError, msg=entry):
                parse_combos([entry])
        with self.assertRaises(ValueError):
            parse_combos({"not": "a list"})

    def test_load_creates_default_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "combos.json"
            combos = load_combos(path)
            self.assertEqual(json.loads(path.read_text()), DEFAULT_COMBOS)
            self.assertEqual([c.name for c in combos], [c["name"] for c in DEFAULT_COMBOS])


class MatcherTests(unittest.TestCase):
    def test_matches_sequence(self):
        m = ComboMatcher(COMBOS)
        self.assertIsNone(m.feed("F", 0))
        self.assertEqual(m.feed("O", 1).name, "Fox")
        # history is cleared after a match
        self.assertIsNone(m.feed("O", 1.5))

    def test_longest_combo_wins(self):
        m = ComboMatcher(COMBOS)
        m.feed("A", 0)
        m.feed("B", 1)
        self.assertEqual(m.feed("C", 2).name, "Big")

    def test_timeout_breaks_combo(self):
        m = ComboMatcher(COMBOS, step_timeout=3)
        m.feed("B", 0)
        self.assertIsNone(m.feed("C", 3.5))

    def test_ignores_spaces_and_noise(self):
        m = ComboMatcher(COMBOS)
        m.feed("X", 0)
        m.feed("B", 0.5)
        self.assertIsNone(m.feed(SPACE, 1))
        self.assertEqual(m.feed("C", 1.5).name, "Rasengan")

    def test_progress(self):
        m = ComboMatcher(COMBOS)
        m.feed("A", 0)
        m.feed("B", 1)
        p = m.progress(1.5)
        self.assertEqual(p, {"Rasengan": 1, "Big": 2, "Fox": 0})
        self.assertEqual(m.progress(10), {"Rasengan": 0, "Big": 0, "Fox": 0})


if __name__ == "__main__":
    unittest.main()
