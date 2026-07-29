"""Tests for the structured-JSON semantic parse-identity addendum."""
import unittest

from analysis.semantic_sj import analyze_semantic, canonical, parse_sj


class TestParseSj(unittest.TestCase):
    def test_strict(self):
        mode, sem = parse_sj('{"b": 1, "a": 2}')
        self.assertEqual(mode, "strict")
        self.assertEqual(sem, '{"a":2,"b":1}')

    def test_spacing_variants_parse_identical(self):
        _, compact = parse_sj('{"a":1,"b":true}')
        _, spaced = parse_sj('{\n  "a": 1,\n  "b": true\n}')
        self.assertEqual(compact, spaced)

    def test_fenced(self):
        mode, sem = parse_sj('```json\n{"a": 1}\n```')
        self.assertEqual(mode, "fenced")
        self.assertEqual(sem, '{"a":1}')

    def test_bare_fence(self):
        mode, sem = parse_sj('```\n{"a": 1}\n```')
        self.assertEqual(mode, "fenced")
        self.assertEqual(sem, '{"a":1}')

    def test_fail(self):
        mode, sem = parse_sj("I cannot produce JSON")
        self.assertEqual(mode, "fail")
        self.assertIsNone(sem)

    def test_canonical_sorts_and_compacts(self):
        self.assertEqual(canonical({"b": 1, "a": [1, 2]}), '{"a":[1,2],"b":1}')


def _rec(text, sha, task="structured_json", ok=True, stop="end_turn", window="w"):
    return {
        "meta_task": task,
        "ok": ok,
        "stop_reason": stop,
        "text": text,
        "text_sha256": sha,
        "cell": "m|structured_json|p|a",
        "window": window,
    }


class TestAnalyzeSemantic(unittest.TestCase):
    def test_byte_variants_semantically_identical(self):
        records = [
            _rec('{"a":1}', "s1"),
            _rec('{\n  "a": 1\n}', "s2"),
            _rec('```json\n{"a": 1}\n```', "s3"),
        ]
        cells = analyze_semantic(records)
        entry = cells["w::m|structured_json|p|a"]
        self.assertEqual(entry["byte_distinct"], 3)
        self.assertEqual(entry["semantic_distinct"], 1)
        self.assertEqual(entry["modes"], {"strict": 2, "fenced": 1})

    def test_field_value_flip_is_semantic_variance(self):
        records = [
            _rec('{"name":"Corvid CS-220"}', "s1"),
            _rec('{"name":"Item Corvid CS-220"}', "s2"),
        ]
        cells = analyze_semantic(records)
        entry = cells["w::m|structured_json|p|a"]
        self.assertEqual(entry["semantic_distinct"], 2)
        self.assertEqual(len(entry["semantic_variants"]), 2)

    def test_non_sj_and_invalid_records_excluded(self):
        records = [
            _rec('{"a":1}', "s1"),
            _rec('{"a":1}', "s2", task="classification"),
            _rec('{"a":1}', "s3", ok=False),
            _rec('{"a":1}', "s4", stop="max_tokens"),
        ]
        cells = analyze_semantic(records)
        self.assertEqual(cells["w::m|structured_json|p|a"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
