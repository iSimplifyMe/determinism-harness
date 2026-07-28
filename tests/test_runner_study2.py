"""Study-2 runner integration: plane-aware scheduling and engine dispatch.
Stub planes only — no network, no SDK imports; must pass on a stdlib-only
interpreter (SDK/boto3 construction is reached only for planes missing from
an injected plane_clients dict, which these tests always provide)."""
import json
import os
import tempfile
import unittest

from harness.runner import Engine, build_schedule, summary_is_complete


class TestStudy2Schedule(unittest.TestCase):
    def test_pilot_counts(self):
        items = build_schedule("study2-pilot")
        cells = {it["cell"] for it in items}
        # (2 five-family models x 4 tasks x 2 arms + haiku x 4 tasks) x 3 planes
        self.assertEqual(len(cells), 60)
        self.assertEqual(len(items), 60 * 20)

    def test_full_counts(self):
        self.assertEqual(len(build_schedule("study2-full")), 60 * 100)

    def test_positive_control_per_plane(self):
        items = build_schedule("study2-positive-control")
        self.assertEqual(len(items), 3 * 100)
        self.assertEqual(len({it["cell"] for it in items}), 3)
        for it in items:
            if it["plane"] == "bedrock":
                self.assertIn(b'"temperature":0.7', it["payload"])
            else:
                self.assertEqual(it["payload"]["temperature"], 0.7)

    def test_payload_shapes_by_plane(self):
        for it in build_schedule("study2-pilot"):
            if it["plane"] == "bedrock":
                self.assertIsInstance(it["payload"], bytes)
                body = json.loads(it["payload"])
                self.assertIn("anthropic_version", body)
                self.assertNotIn("model", body)
                self.assertTrue(it["model_id"].startswith("us."))
            else:
                self.assertIsInstance(it["payload"], dict)
                self.assertNotIn("anthropic_version", it["payload"])
                self.assertEqual(it["payload"]["model"], it["model_id"])

    def test_planned_sha_constant_within_cell(self):
        by_cell = {}
        for it in build_schedule("study2-pilot"):
            by_cell.setdefault(it["cell"], set()).add(it["sha"])
        for cell, shas in by_cell.items():
            self.assertEqual(len(shas), 1, cell)

    def test_planned_sha_across_planes(self):
        """Bedrock bytes differ from Messages params by construction; the two
        Messages planes plan byte-identical requests (same bare/dated IDs)."""
        by_cell = {}
        for it in build_schedule("study2-pilot"):
            by_cell[it["cell"]] = it["sha"]
        bedrock = by_cell["opus-5|classification|bedrock|adaptive"]
        p_aws = by_cell["opus-5|classification|p_aws|adaptive"]
        first_party = by_cell["opus-5|classification|anthropic_api|adaptive"]
        self.assertNotEqual(bedrock, p_aws)
        self.assertEqual(p_aws, first_party)


class StubPlane:
    """Scriptable plane double: returns the next canned record per invoke."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def invoke(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.script.pop(0)


def _ok():
    return {
        "ok": True,
        "latency_ms": 1,
        "request_id": "req",
        "response_id": "msg",
        "response_model": "m",
        "stop_reason": "end_turn",
        "usage": {"output_tokens": 1},
        "text": "T",
        "text_sha256": "ts",
        "wire_sha256": "ws",
    }


def _fail(retryable, code="rate_limit_error", status=429):
    return {
        "ok": False,
        "error_code": code,
        "error_message": "boom",
        "status_code": status,
        "retryable": retryable,
        "request_id": None,
        "wire_sha256": "ws",
    }


def _items(n, plane="p_aws"):
    return [
        {
            "cell": f"cell{i}",
            "meta": {"model": "x", "plane": plane},
            "plane": plane,
            "payload": {"model": "y"},
            "sha": "planned",
            "model_id": "y",
            "repeat": 0,
        }
        for i in range(n)
    ]


class TestEnginePlaneDispatch(unittest.TestCase):
    def _run(self, items, clients, max_attempts=3):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            engine = Engine(
                items,
                path,
                concurrency=2,
                seed=1,
                run_info={"mode": "test"},
                max_attempts=max_attempts,
                plane_clients=clients,
            )
            summary = engine.run()
            with open(path, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh]
        finally:
            os.unlink(path)
        return summary, records

    def test_success_record_schema(self):
        stub = StubPlane([_ok(), _ok(), _ok()])
        summary, records = self._run(_items(3), {"p_aws": stub})
        self.assertTrue(summary_is_complete(summary, 3))
        self.assertEqual(summary["failures"], 0)
        for rec in records:
            self.assertEqual(rec["schema"], 2)
            self.assertTrue(rec["ok"])
            self.assertEqual(rec["plane"], "p_aws")
            self.assertEqual(rec["request_sha256"], "planned")
            self.assertEqual(rec["wire_sha256"], "ws")
            self.assertEqual(rec["attempts"], 1)
            self.assertEqual(rec["meta_model"], "x")
            self.assertIn("sent_at_utc", rec)
            self.assertIn("received_at_utc", rec)

    def test_retryable_then_ok(self):
        stub = StubPlane([_fail(True), _ok()])
        summary, records = self._run(_items(1), {"p_aws": stub})
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(summary["failures"], 0)
        self.assertTrue(records[0]["ok"])
        self.assertEqual(records[0]["attempts"], 2)

    def test_terminal_failure_not_retried(self):
        stub = StubPlane([_fail(False, code="invalid_request_error", status=400)])
        summary, records = self._run(_items(1), {"p_aws": stub})
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(len(stub.calls), 1)
        self.assertFalse(records[0]["ok"])
        self.assertEqual(records[0]["error_code"], "invalid_request_error")
        self.assertEqual(records[0]["attempts"], 1)

    def test_retries_exhausted_records_failure(self):
        stub = StubPlane([_fail(True), _fail(True), _fail(True)])
        summary, records = self._run(_items(1), {"p_aws": stub}, max_attempts=3)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["retries"], 2)
        self.assertEqual(records[0]["attempts"], 3)
        self.assertFalse(records[0]["ok"])

    def test_bedrock_items_pass_model_id_and_bytes(self):
        items = _items(1, plane="bedrock")
        items[0]["payload"] = b"BYTES"
        stub = StubPlane([_ok()])
        _, records = self._run(items, {"bedrock": stub})
        self.assertEqual(stub.calls[0], (("y", b"BYTES"), {"stream": False}))
        self.assertTrue(records[0]["ok"])

    def test_streaming_delivery_dispatches_stream_kwarg(self):
        items = _items(1)
        items[0]["delivery"] = "streaming"
        stub = StubPlane([_ok()])
        _, records = self._run(items, {"p_aws": stub})
        self.assertEqual(stub.calls[0][1], {"stream": True})
        self.assertTrue(records[0]["ok"])

    def test_no_legacy_client_for_plane_only_schedules(self):
        stub = StubPlane([_ok()])
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            engine = Engine(
                _items(1), path, 1, 1, run_info={}, plane_clients={"p_aws": stub}
            )
            self.assertIsNone(engine.client)
            engine.run()
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
