"""Companion-C cache-state A/B (FOLLOWUP-COMPANIONS.md): schedule + report.
Stub data only — stdlib, no network."""
import hashlib
import json
import unittest

from harness.config import CACHE_AB, LOCAL_KEEP_ALIVE, LOCAL_MODELS, LOCAL_SAMPLING
from harness.request_builder import canonical_local_body, sha256_hex
from harness.runner import (
    COMPANION_MODES,
    FIXED_SCHEDULE_MODES,
    build_schedule,
    study3_run_settings,
)
from harness.tasks import TASKS
from analysis.analyze_cache_ab import CACHE_CELL, build_cache_report

TAG = LOCAL_MODELS["gpt-oss-20b"]["tag"]


class TestCacheSchedule(unittest.TestCase):
    def test_counts_structure_and_alternation(self):
        items = build_schedule("study3-cache-ab", box="cuda")
        n = CACHE_AB["n_per_arm"]
        # 1 warmup + cold blocks contribute 2 items per measured call
        self.assertEqual(len(items), 1 + 2 * n + n)
        self.assertEqual(items[0]["meta"].get("control"), "warmup")
        # first measured item is cold-arm and follows a flusher
        first_measured = items[2]
        self.assertEqual(items[1]["meta"].get("control"), "flusher")
        self.assertEqual(first_measured["meta"]["arm"], "cold")
        # every cold measured call is immediately preceded by a flusher;
        # every warm measured call by a measured open-generation call
        for i, item in enumerate(items):
            arm = item["meta"].get("arm")
            if arm == "cold":
                self.assertEqual(
                    items[i - 1]["meta"].get("control"), "flusher", i
                )
            elif arm == "warm":
                prev = items[i - 1]["meta"]
                self.assertIn(prev.get("arm"), ("cold", "warm"), i)
                self.assertIsNone(prev.get("control"), i)

    def test_measured_bodies_frozen_and_identical_across_arms(self):
        items = build_schedule("study3-cache-ab", box="cuda")
        measured = [it for it in items if it["meta"].get("arm")]
        shas = {it["sha"] for it in measured}
        self.assertEqual(len(shas), 1)
        frozen = canonical_local_body(
            TAG,
            TASKS["open_generation"]["prompt"],
            "effort_low",
            options=LOCAL_SAMPLING["greedy"],
            keep_alive=LOCAL_KEEP_ALIVE,
        )
        self.assertEqual(measured[0]["payload"], frozen)
        self.assertEqual(shas.pop(), sha256_hex(frozen))
        cold = [it for it in measured if it["meta"]["arm"] == "cold"]
        warm = [it for it in measured if it["meta"]["arm"] == "warm"]
        self.assertEqual(len(cold), CACHE_AB["n_per_arm"])
        self.assertEqual(len(warm), CACHE_AB["n_per_arm"])
        for it in cold:
            self.assertTrue(it["cell"].endswith("|prefill=cold"))
        for it in warm:
            self.assertTrue(it["cell"].endswith("|prefill=warm"))

    def test_flushers_use_the_frozen_flusher_prompt(self):
        items = build_schedule("study3-cache-ab", box="cuda")
        flushers = [
            it for it in items if it["meta"].get("control") == "flusher"
        ]
        self.assertEqual(len(flushers), CACHE_AB["n_per_arm"])
        body = json.loads(flushers[0]["payload"])
        self.assertIn(
            TASKS[CACHE_AB["flusher_task"]]["prompt"][:40],
            body["messages"][0]["content"],
        )

    def test_membership_and_overrides(self):
        self.assertIn("study3-cache-ab", COMPANION_MODES)
        self.assertIn("study3-cache-ab", FIXED_SCHEDULE_MODES)
        settings = study3_run_settings("study3-cache-ab")
        self.assertEqual(settings["concurrency"], 1)
        items = build_schedule("study3-cache-ab", box="cuda", repeats=2)
        measured = [it for it in items if it["meta"].get("arm")]
        self.assertEqual(len(measured), 4)
        with self.assertRaises(ValueError):
            build_schedule("study3-cache-ab", box="tpu")


WARM_PREFILL = int(17e6)
COLD_PREFILL = int(199e6)


def _cache_record(arm, text, prefill_ns, repeat, ok=True, stop="stop"):
    return {
        "schema": 3,
        "box": "cuda",
        "cell": f"{CACHE_CELL}|prefill={arm}",
        "repeat": repeat,
        "request_sha256": "sha-frozen",
        "wire_sha256": "sha-frozen",
        "meta_model": "gpt-oss-20b",
        "meta_task": "open_generation",
        "meta_sampling": "greedy",
        "meta_thinking": "effort_low",
        "meta_hardware": "cuda",
        "meta_arm": arm,
        "ok": ok,
        "stop_reason": stop,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "usage": {
            "output_tokens": len(text),
            "prompt_eval_duration_ns": prefill_ns,
        },
        "response_model": "gpt-oss:20b",
    }


def _flusher_record(repeat):
    return {
        "schema": 3, "box": "cuda", "cell": "flusher|gpt-oss-20b",
        "meta_control": "flusher", "meta_model": "gpt-oss-20b",
        "request_sha256": "sha-fl", "ok": True, "stop_reason": "stop",
        "text": "BILLING", "text_sha256": "f", "usage": {},
        "repeat": repeat,
    }


class TestCacheReport(unittest.TestCase):
    def test_predictions_hold_case(self):
        records = (
            [_cache_record("cold", "CCC", COLD_PREFILL, r) for r in range(10)]
            + [_cache_record("warm", "WWW", WARM_PREFILL, r) for r in range(10)]
            + [_flusher_record(r) for r in range(10)]
        )
        report = build_cache_report(records)
        cuda = report["boxes"]["cuda"]
        self.assertTrue(cuda["gates"]["manipulation"]["pass"])
        self.assertTrue(cuda["gates"]["cross_arm_negative_control"])
        self.assertEqual(cuda["arms"]["cold"]["metrics"]["modal_share"], 1.0)
        self.assertEqual(cuda["arms"]["warm"]["metrics"]["modal_share"], 1.0)
        self.assertTrue(cuda["arms_differ"])
        self.assertEqual(report["totals"]["flushers_excluded"], 10)

    def test_history_matching_is_descriptive(self):
        records = (
            [_cache_record("cold", "CCC", COLD_PREFILL, r) for r in range(5)]
            + [_cache_record("warm", "WWW", WARM_PREFILL, r) for r in range(5)]
        )
        history = {
            "confirmatory_modal": hashlib.sha256(b"CCC").hexdigest(),
            "companionA_cached": hashlib.sha256(b"ZZZ").hexdigest(),
        }
        report = build_cache_report(records, history=history)
        cuda = report["boxes"]["cuda"]
        self.assertEqual(
            cuda["arms"]["cold"]["history_matches"], ["confirmatory_modal"]
        )
        self.assertEqual(cuda["arms"]["warm"]["history_matches"], [])

    def test_warm_prefill_violation_voids_gate(self):
        records = (
            [_cache_record("cold", "CCC", COLD_PREFILL, r) for r in range(5)]
            + [_cache_record("warm", "WWW", COLD_PREFILL, r) for r in range(5)]
        )
        report = build_cache_report(records)
        gates = report["boxes"]["cuda"]["gates"]["manipulation"]
        self.assertFalse(gates["pass"])
        self.assertIsNone(report["boxes"]["cuda"]["arms_differ"])

    def test_cold_call_below_threshold_voids_gate(self):
        records = (
            [_cache_record("cold", "CCC", COLD_PREFILL, r) for r in range(4)]
            + [_cache_record("cold", "CCC", WARM_PREFILL, 4)]
            + [_cache_record("warm", "WWW", WARM_PREFILL, r) for r in range(5)]
        )
        report = build_cache_report(records)
        gates = report["boxes"]["cuda"]["gates"]["manipulation"]
        self.assertFalse(gates["cold_all_above"])
        self.assertFalse(gates["pass"])


if __name__ == "__main__":
    unittest.main()
