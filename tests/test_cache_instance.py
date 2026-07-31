"""Companion-F fresh-instance confirmation: schedule + analyzer.
Stub data only — stdlib, no network."""
import hashlib
import unittest

from harness.config import CACHE_INSTANCE, LOCAL_MODELS
from harness.runner import (
    COMPANION_MODES,
    FIXED_SCHEDULE_MODES,
    build_schedule,
)
from analysis.analyze_cache_instance import (
    INSTANCE_CELL,
    build_instance_report,
)

TAG = LOCAL_MODELS["gpt-oss-20b"]["tag"]

PURE_FAST = int(17e6)
CONT_SLOW = int(36e6)
BURN_FRESH = int(200e6)


class TestInstanceSchedule(unittest.TestCase):
    def test_cycle_layout_counts_and_resets(self):
        items = build_schedule("study3-cache-instance", box="cuda")
        cycles = CACHE_INSTANCE["n_cycles"]
        n = CACHE_INSTANCE["n_per_arm_per_cycle"]
        span = 2 * n + 2
        self.assertEqual(len(items), cycles * span)
        # No warmup anywhere: a warmup's different prompt is the
        # checkpoint trigger under test.
        self.assertFalse(
            any(it["meta"].get("control") == "warmup" for it in items)
        )
        measured_shas = set()
        flusher_shas = set()
        for c in range(cycles):
            block = items[c * span:(c + 1) * span]
            burnin = block[0]
            self.assertEqual(burnin["meta"].get("control"), "burnin")
            self.assertEqual(burnin["meta"].get("cycle"), c)
            self.assertTrue(burnin.get("pre_unload"))
            self.assertEqual(burnin["pre_sleep_ms"], 0)
            self.assertTrue(burnin["cell"].endswith("|instance=burnin"))
            measured_shas.add(burnin["sha"])
            for it in block[1:1 + n]:
                self.assertEqual(it["meta"]["arm"], "pure")
                self.assertEqual(it["meta"]["cycle"], c)
                self.assertNotIn("pre_unload", it)
                self.assertEqual(it["pre_sleep_ms"], 0)
                self.assertTrue(it["cell"].endswith("|instance=pure"))
                measured_shas.add(it["sha"])
            flusher = block[1 + n]
            self.assertEqual(flusher["meta"].get("control"), "flusher")
            self.assertEqual(flusher["meta"].get("cycle"), c)
            self.assertEqual(flusher["pre_sleep_ms"], 0)
            self.assertNotIn("pre_unload", flusher)
            flusher_shas.add(flusher["sha"])
            for it in block[2 + n:span]:
                self.assertEqual(it["meta"]["arm"], "contaminated")
                self.assertEqual(it["meta"]["cycle"], c)
                self.assertNotIn("pre_unload", it)
                self.assertEqual(it["pre_sleep_ms"], 0)
                self.assertTrue(
                    it["cell"].endswith("|instance=contaminated")
                )
                measured_shas.add(it["sha"])
        # Burn-ins and both arms share ONE request sha (the cross-arm
        # negative control); the flusher differs by design.
        self.assertEqual(len(measured_shas), 1)
        self.assertEqual(len(flusher_shas), 1)
        self.assertNotEqual(measured_shas, flusher_shas)
        self.assertIn("study3-cache-instance", COMPANION_MODES)
        self.assertIn("study3-cache-instance", FIXED_SCHEDULE_MODES)
        # Registered CUDA-only (non-production box).
        with self.assertRaises(ValueError):
            build_schedule("study3-cache-instance", box="metal")
        with self.assertRaises(ValueError):
            build_schedule("study3-cache-instance", box=None)


def _record(arm, text, prefill_ns, repeat, cycle, ok=True):
    return {
        "schema": 3, "box": "cuda",
        "cell": f"{INSTANCE_CELL}|instance={arm}",
        "repeat": repeat,
        "request_sha256": "sha-frozen", "wire_sha256": "sha-frozen",
        "meta_model": "gpt-oss-20b", "meta_task": "open_generation",
        "meta_sampling": "greedy", "meta_thinking": "effort_low",
        "meta_hardware": "cuda", "meta_arm": arm, "meta_cycle": cycle,
        "ok": ok, "stop_reason": "stop",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "usage": {"output_tokens": len(text),
                  "prompt_eval_duration_ns": prefill_ns},
        "response_model": TAG,
    }


def _burnin(cycle, confirmed=True, prefill_ns=BURN_FRESH):
    rec = _record("burnin", "BBB", prefill_ns, cycle, cycle)
    rec["cell"] = f"{INSTANCE_CELL}|instance=burnin"
    del rec["meta_arm"]
    rec["meta_control"] = "burnin"
    rec["pre_unload_confirmed"] = confirmed
    rec["unload_wait_ms"] = 900.0
    return rec


def _cycle_records(cycle, pure_text="AAA", cont_text="GGG",
                   confirmed=True, n=10):
    records = [_burnin(cycle, confirmed=confirmed)]
    records += [
        _record("pure", pure_text, PURE_FAST, cycle * n + r, cycle)
        for r in range(n)
    ]
    records += [
        _record("contaminated", cont_text, CONT_SLOW, cycle * n + r, cycle)
        for r in range(n)
    ]
    return records


class TestInstanceReport(unittest.TestCase):
    def test_pass_case_all_three_endpoints(self):
        records = []
        for c in range(5):
            records += _cycle_records(c)
        report = build_instance_report(records)
        cuda = report["boxes"]["cuda"]
        manipulation = cuda["gates"]["manipulation"]
        self.assertTrue(manipulation["pure_all_below"])
        self.assertTrue(manipulation["contaminated_all_above"])
        self.assertTrue(manipulation["resets_confirmed"])
        self.assertTrue(manipulation["pass"])
        self.assertTrue(cuda["gates"]["cross_arm_negative_control"])
        self.assertEqual(
            cuda["arms"]["pure"]["metrics"]["modal_share"], 1.0
        )
        self.assertEqual(
            cuda["arms"]["contaminated"]["metrics"]["modal_share"], 1.0
        )
        self.assertTrue(cuda["arms_differ"])
        self.assertTrue(cuda["endpoint3_flip_all_cycles"])
        self.assertEqual(len(cuda["cycles"]), 5)
        for entry in cuda["cycles"]:
            self.assertTrue(entry["flip_at_flusher"])
            self.assertTrue(entry["burnin_reset_confirmed"])

    def test_slow_pure_call_voids_gate(self):
        records = []
        for c in range(5):
            records += _cycle_records(c)
        records.append(_record("pure", "AAA", CONT_SLOW, 50, 4))
        report = build_instance_report(records)
        cuda = report["boxes"]["cuda"]
        manipulation = cuda["gates"]["manipulation"]
        self.assertFalse(manipulation["pure_all_below"])
        self.assertFalse(manipulation["pass"])
        self.assertIsNone(cuda["arms_differ"])
        self.assertIsNone(cuda["endpoint3_flip_all_cycles"])

    def test_unconfirmed_reset_voids_gate(self):
        records = []
        for c in range(5):
            records += _cycle_records(c, confirmed=(c != 2))
        report = build_instance_report(records)
        manipulation = report["boxes"]["cuda"]["gates"]["manipulation"]
        self.assertFalse(manipulation["resets_confirmed"])
        self.assertFalse(manipulation["pass"])

    def test_flip_not_at_flusher_fails_endpoint3(self):
        # Cycle 2's pure block already carries the contaminated bytes
        # (fast prefill, so the gate still passes): pooled arms differ,
        # but the flip did not sit at the interposed call in cycle 2.
        records = []
        for c in range(5):
            records += _cycle_records(
                c, pure_text=("GGG" if c == 2 else "AAA")
            )
        report = build_instance_report(records)
        cuda = report["boxes"]["cuda"]
        self.assertTrue(cuda["gates"]["manipulation"]["pass"])
        self.assertEqual(
            cuda["arms"]["pure"]["metrics"]["modal_share"], 0.8
        )
        self.assertTrue(cuda["arms_differ"])
        self.assertFalse(cuda["endpoint3_flip_all_cycles"])
        flips = {e["cycle"]: e["flip_at_flusher"] for e in cuda["cycles"]}
        self.assertFalse(flips[2])
        self.assertTrue(flips[0])

    def test_history_matches_descriptive(self):
        records = []
        for c in range(2):
            records += _cycle_records(c)
        history = {
            "full_kv": hashlib.sha256(b"AAA").hexdigest(),
            "checkpoint": hashlib.sha256(b"GGG").hexdigest(),
        }
        report = build_instance_report(records, history=history)
        cuda = report["boxes"]["cuda"]
        self.assertEqual(
            cuda["arms"]["pure"]["history_matches"], ["full_kv"]
        )
        self.assertEqual(
            cuda["arms"]["contaminated"]["history_matches"], ["checkpoint"]
        )


if __name__ == "__main__":
    unittest.main()
