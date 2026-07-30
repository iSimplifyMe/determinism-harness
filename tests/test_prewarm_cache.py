"""Companion-D session-qualification prewarm. Stub openers — no network."""
import io
import json
import unittest

from harness.planes import LocalPlane
from harness.prewarm_cache import run_prewarm
from harness.runner import build_schedule


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _opener_with_prefills(prefills_ms):
    calls = {"i": 0}

    def opener(request, timeout=None):
        i = min(calls["i"], len(prefills_ms) - 1)
        calls["i"] += 1
        payload = {
            "model": "gpt-oss:20b",
            "message": {"role": "assistant", "content": f"out{prefills_ms[i]}"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 60,
            "eval_count": 5,
            "eval_duration": 1000,
            "prompt_eval_duration": int(prefills_ms[i] * 1e6),
        }
        return _FakeResponse(json.dumps(payload).encode())

    return opener


class TestRunPrewarm(unittest.TestCase):
    def test_qualifies_on_three_consecutive_below_threshold(self):
        opener = _opener_with_prefills([40, 38, 20, 19, 18, 17])
        plane = LocalPlane(opener=opener)
        result = run_prewarm(plane, threshold_ms=25.0, consecutive=3,
                             max_calls=40)
        self.assertTrue(result["qualified"])
        self.assertEqual(result["calls"], 5)  # 2 slow + 3 fast
        self.assertEqual(len(result["trajectory"]), 5)

    def test_streak_resets_on_slow_call(self):
        opener = _opener_with_prefills([20, 20, 40, 20, 20, 20])
        plane = LocalPlane(opener=opener)
        result = run_prewarm(plane, threshold_ms=25.0, consecutive=3,
                             max_calls=40)
        self.assertTrue(result["qualified"])
        self.assertEqual(result["calls"], 6)

    def test_cap_without_qualification(self):
        opener = _opener_with_prefills([40])
        plane = LocalPlane(opener=opener)
        result = run_prewarm(plane, threshold_ms=25.0, consecutive=3,
                             max_calls=7)
        self.assertFalse(result["qualified"])
        self.assertEqual(result["calls"], 7)

    def test_body_matches_the_ab_measured_cell(self):
        opener = _opener_with_prefills([20, 20, 20])
        plane = LocalPlane(opener=opener)
        result = run_prewarm(plane)
        items = build_schedule("study3-cache-ab", box="cuda", repeats=1)
        measured = next(it for it in items if it["meta"].get("arm"))
        self.assertEqual(result["request_sha256"], measured["sha"])

    def test_sha_sequence_recorded(self):
        opener = _opener_with_prefills([40, 20, 20, 20])
        plane = LocalPlane(opener=opener)
        result = run_prewarm(plane, threshold_ms=25.0, consecutive=3)
        self.assertEqual(len(result["distinct_shas"]), 2)  # out40 vs out20


if __name__ == "__main__":
    unittest.main()
