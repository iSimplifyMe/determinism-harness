"""Study-4 runner integration: schedules, door dispatch, batch driver."""
import json
import os
import random
import tempfile
import unittest

from harness.config import (
    STUDY4_CODEX_BATCH_SIZE,
    STUDY4_REPEATS_EXPLORATORY,
    STUDY4_REPEATS_FULL,
)
from harness.request_builder import canonical_bytes, sha256_hex
from harness.runner import (
    Engine,
    _study4_item,
    build_schedule,
    study4_run_settings,
)
from scripts.run_codex_batches import completed_pairs, remaining_items


class ScheduleTest(unittest.TestCase):
    def test_full_grid_counts(self):
        items = build_schedule("study4-full")
        # 3 HTTP doors x 4 tasks x 2 effort arms x n
        self.assertEqual(len(items), 3 * 4 * 2 * STUDY4_REPEATS_FULL)
        self.assertEqual(len({it["cell"] for it in items}), 24)
        self.assertTrue(all(it["door"] != "codex_sub" for it in items))

    def test_codex_counts(self):
        items = build_schedule("study4-codex")
        self.assertEqual(len(items), 4 * 2 * STUDY4_REPEATS_FULL)
        self.assertEqual(len({it["cell"] for it in items}), 8)
        self.assertTrue(all(it["door"] == "codex_sub" for it in items))

    def test_q4q5_counts_and_controls(self):
        items = build_schedule("study4-q4q5")
        self.assertEqual(len(items), (4 + 6) * STUDY4_REPEATS_EXPLORATORY)
        self.assertEqual(len({it["cell"] for it in items}), 10)
        controls = {it["meta"]["control"] for it in items}
        self.assertEqual(controls, {"q4_routing", "q5_default"})
        # codex has no default arm; Q5 is API-doors only (v4 section 1).
        self.assertTrue(all(it["door"] != "codex_sub" for it in items))
        q4 = [it for it in items if it["meta"]["control"] == "q4_routing"]
        self.assertEqual({it["door"] for it in q4},
                         {"runtime_us", "runtime_global"})

    def test_repeats_override(self):
        items = build_schedule("study4-codex", repeats=3)
        self.assertEqual(len(items), 4 * 2 * 3)

    def test_run_settings(self):
        self.assertEqual(study4_run_settings("study4-codex")["concurrency"], 1)
        self.assertIsNone(study4_run_settings("study4-full"))
        self.assertIsNone(study4_run_settings("study2-full"))


class ItemTest(unittest.TestCase):
    def test_responses_item_hash_is_of_wire_bytes(self):
        item = _study4_item("openai_1p", "extraction", "none", 0)
        self.assertIsInstance(item["payload"], bytes)
        self.assertEqual(item["sha"], sha256_hex(item["payload"]))
        self.assertEqual(item["cell"], "openai_1p|extraction|none")
        self.assertEqual(item["meta"]["effort"], "none")

    def test_converse_item_hash_is_planned_request(self):
        item = _study4_item("runtime_us", "extraction", "high", 2)
        self.assertIsInstance(item["payload"], dict)
        self.assertEqual(
            item["sha"], sha256_hex(canonical_bytes(item["payload"]))
        )
        self.assertEqual(
            item["payload"]["additionalModelRequestFields"],
            {"reasoning": {"effort": "high"}},
        )

    def test_codex_item_argv_carries_pin(self):
        item = _study4_item("codex_sub", "extraction", "high", 0)
        self.assertIn("model_reasoning_effort=high", item["payload"])


class _StubDoor:
    """Scripted door: pops one canned result per invoke."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        return dict(self.results.pop(0))


def _ok(text="x"):
    return {
        "ok": True, "latency_ms": 1, "request_id": None,
        "response_id": "r", "response_model": "m", "stop_reason": "completed",
        "usage": {}, "text": text,
        "text_sha256": sha256_hex(text.encode()), "wire_sha256": "w",
    }


def _fail(retryable):
    return {
        "ok": False, "error_code": "http_500", "error_message": "boom",
        "status_code": 500, "retryable": retryable, "request_id": None,
        "wire_sha256": None,
    }


class EngineDispatchTest(unittest.TestCase):
    def _run(self, items, doors):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.jsonl")
            engine = Engine(
                items, out, concurrency=1, seed=1,
                run_info={"mode": "study4-full", "window": "control",
                          "run_name": "t"},
                max_attempts=3, door_clients=doors,
            )
            engine._sleep = lambda s: None  # skip anti-burst jitter
            summary = engine.run()
            records = [json.loads(l) for l in open(out)]
        return summary, records

    def test_retryable_failure_then_success(self):
        item = _study4_item("openai_1p", "extraction", "none", 0)
        stub = _StubDoor([_fail(retryable=True), _ok()])
        summary, records = self._run([item], {"openai_1p": stub})
        self.assertEqual(summary["failures"], 0)
        self.assertEqual(summary["retries"], 1)
        record = records[0]
        self.assertTrue(record["ok"])
        self.assertEqual(record["attempts"], 2)
        self.assertEqual(record["schema"], 4)
        self.assertEqual(record["meta_door"], "openai_1p")
        self.assertEqual(record["meta_effort"], "none")
        self.assertEqual(record["request_sha256"], item["sha"])

    def test_nonretryable_fails_once(self):
        item = _study4_item("mantle", "extraction", "none", 0)
        stub = _StubDoor([_fail(retryable=False)])
        summary, records = self._run([item], {"mantle": stub})
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(records[0]["attempts"], 1)
        self.assertFalse(records[0]["ok"])

    def test_retry_bound_exhausts_as_counted_exclusion(self):
        item = _study4_item("codex_sub", "extraction", "none", 0)
        stub = _StubDoor([_fail(True), _fail(True), _fail(True)])
        summary, records = self._run([item], {"codex_sub": stub})
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(records[0]["attempts"], 3)
        self.assertEqual(stub.calls, 3)


class BatchDriverTest(unittest.TestCase):
    def test_shuffle_is_seed_stable(self):
        one = build_schedule("study4-codex")
        two = build_schedule("study4-codex")
        random.Random(20260727).shuffle(one)
        random.Random(20260727).shuffle(two)
        self.assertEqual(
            [(it["cell"], it["repeat"]) for it in one],
            [(it["cell"], it["repeat"]) for it in two],
        )

    def test_completed_pairs_and_remaining(self):
        schedule = build_schedule("study4-codex", repeats=2)
        random.Random(20260727).shuffle(schedule)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "control-study4-codex-b01-x.jsonl")
            with open(path, "w") as fh:
                for it in schedule[:5]:
                    fh.write(json.dumps(
                        {"cell": it["cell"], "repeat": it["repeat"],
                         "ok": it["repeat"] % 2 == 0}  # failures count too
                    ) + "\n")
            done = completed_pairs(tmp)
        self.assertEqual(len(done), 5)
        todo = remaining_items(schedule, done)
        self.assertEqual(len(todo), len(schedule) - 5)
        self.assertEqual(
            (todo[0]["cell"], todo[0]["repeat"]),
            (schedule[5]["cell"], schedule[5]["repeat"]),
        )

    def test_batch_size_default_is_registered(self):
        self.assertEqual(STUDY4_CODEX_BATCH_SIZE, 40)
