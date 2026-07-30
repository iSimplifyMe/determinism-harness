"""Companion-E timing A/B: schedule, engine sleep control, analyzer.
Stub data only — stdlib, no network."""
import hashlib
import io
import json
import os
import tempfile
import unittest

from harness.config import CACHE_TIMING, LOCAL_MODELS
from harness.planes import LocalPlane
from harness.runner import (
    COMPANION_MODES,
    Engine,
    FIXED_SCHEDULE_MODES,
    build_schedule,
)
from analysis.analyze_cache_timing import TIMING_CELL, build_timing_report

TAG = LOCAL_MODELS["gpt-oss-20b"]["tag"]


class TestTimingSchedule(unittest.TestCase):
    def test_counts_arms_sleeps_and_alternation(self):
        items = build_schedule("study3-cache-timing", box="cuda")
        n = CACHE_TIMING["n_per_arm"]
        self.assertEqual(len(items), 2 + 2 * n)
        self.assertEqual(items[0]["meta"].get("control"), "warmup")
        self.assertNotIn("pre_sleep_ms", items[0])
        self.assertEqual(items[1]["meta"].get("control"), "burnin")
        self.assertEqual(items[1]["pre_sleep_ms"], 0)
        measured = items[2:]
        shas = {it["sha"] for it in measured}
        self.assertEqual(len(shas), 1)
        block = CACHE_TIMING["mini_block"]
        for i, it in enumerate(measured):
            expected = "adjacent" if (i // block) % 2 == 0 else "gapped"
            self.assertEqual(it["meta"]["arm"], expected, i)
            if expected == "adjacent":
                self.assertEqual(it["pre_sleep_ms"], 0)
                self.assertTrue(it["cell"].endswith("|timing=adjacent"))
            else:
                self.assertEqual(it["pre_sleep_ms"], CACHE_TIMING["gap_ms"])
                self.assertTrue(it["cell"].endswith("|timing=gapped"))
        self.assertIn("study3-cache-timing", COMPANION_MODES)
        self.assertIn("study3-cache-timing", FIXED_SCHEDULE_MODES)
        with self.assertRaises(ValueError):
            build_schedule("study3-cache-timing", box="tpu")


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class TestEngineSleepControl(unittest.TestCase):
    def test_pre_sleep_overrides_jitter(self):
        def opener(request, timeout=None):
            return _FakeResponse(json.dumps({
                "model": TAG,
                "message": {"role": "assistant", "content": "x"},
                "done": True, "done_reason": "stop",
                "eval_count": 1, "eval_duration": 1000,
            }).encode())

        items = [
            {"cell": "a", "meta": {"arm": "adjacent"}, "plane": "local",
             "payload": b"{}", "sha": "s", "model_id": TAG, "repeat": 0,
             "pre_sleep_ms": 0},
            {"cell": "g", "meta": {"arm": "gapped"}, "plane": "local",
             "payload": b"{}", "sha": "s", "model_id": TAG, "repeat": 0,
             "pre_sleep_ms": 600},
            {"cell": "d", "meta": {}, "plane": "local",
             "payload": b"{}", "sha": "s", "model_id": TAG, "repeat": 0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(
                items, os.path.join(tmp, "o.jsonl"), 1, 7,
                plane_clients={"local": LocalPlane(opener=opener)},
            )
            slept = []
            engine._sleep = slept.append
            summary = engine.run()
        self.assertEqual(summary["done"], 3)
        # adjacent: no sleep · gapped: exactly 0.6 · default: jitter range
        self.assertEqual(len(slept), 2)
        self.assertIn(0.6, slept)
        jitter = [s for s in slept if s != 0.6]
        self.assertEqual(len(jitter), 1)
        self.assertTrue(0.25 <= jitter[0] <= 1.0)


ADJ_FAST = int(17e6)
GAP_SLOW = int(36e6)


def _timing_record(arm, text, prefill_ns, repeat, ok=True):
    return {
        "schema": 3, "box": "cuda",
        "cell": f"{TIMING_CELL}|timing={arm}",
        "repeat": repeat,
        "request_sha256": "sha-frozen", "wire_sha256": "sha-frozen",
        "meta_model": "gpt-oss-20b", "meta_task": "open_generation",
        "meta_sampling": "greedy", "meta_thinking": "effort_low",
        "meta_hardware": "cuda", "meta_arm": arm,
        "ok": ok, "stop_reason": "stop",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "usage": {"output_tokens": len(text),
                  "prompt_eval_duration_ns": prefill_ns},
        "response_model": TAG,
    }


class TestTimingReport(unittest.TestCase):
    def test_pass_case_arms_differ(self):
        records = (
            [_timing_record("adjacent", "AAA", ADJ_FAST, r) for r in range(10)]
            + [_timing_record("gapped", "GGG", GAP_SLOW, r) for r in range(10)]
        )
        report = build_timing_report(records)
        cuda = report["boxes"]["cuda"]
        self.assertTrue(cuda["gates"]["manipulation"]["pass"])
        self.assertEqual(cuda["arms"]["adjacent"]["metrics"]["modal_share"], 1.0)
        self.assertEqual(cuda["arms"]["gapped"]["metrics"]["modal_share"], 1.0)
        self.assertTrue(cuda["arms_differ"])

    def test_slow_adjacent_call_voids(self):
        records = (
            [_timing_record("adjacent", "AAA", ADJ_FAST, r) for r in range(9)]
            + [_timing_record("adjacent", "AAA", GAP_SLOW, 9)]
            + [_timing_record("gapped", "GGG", GAP_SLOW, r) for r in range(10)]
        )
        report = build_timing_report(records)
        gates = report["boxes"]["cuda"]["gates"]["manipulation"]
        self.assertFalse(gates["adjacent_all_below"])
        self.assertFalse(gates["pass"])
        self.assertIsNone(report["boxes"]["cuda"]["arms_differ"])

    def test_fast_gapped_call_voids(self):
        records = (
            [_timing_record("adjacent", "AAA", ADJ_FAST, r) for r in range(10)]
            + [_timing_record("gapped", "GGG", ADJ_FAST, r) for r in range(10)]
        )
        report = build_timing_report(records)
        gates = report["boxes"]["cuda"]["gates"]["manipulation"]
        self.assertFalse(gates["gapped_all_above"])
        self.assertFalse(gates["pass"])

    def test_history_matches_descriptive(self):
        records = (
            [_timing_record("adjacent", "AAA", ADJ_FAST, r) for r in range(5)]
            + [_timing_record("gapped", "GGG", GAP_SLOW, r) for r in range(5)]
        )
        history = {"cachedA": hashlib.sha256(b"AAA").hexdigest()}
        report = build_timing_report(records, history=history)
        cuda = report["boxes"]["cuda"]
        self.assertEqual(cuda["arms"]["adjacent"]["history_matches"], ["cachedA"])
        self.assertEqual(cuda["arms"]["gapped"]["history_matches"], [])


if __name__ == "__main__":
    unittest.main()
