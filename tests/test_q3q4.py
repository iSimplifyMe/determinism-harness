"""Study-2 Q3 (streamed delivery) and Q4 (input-length ladder) arms, plus the
wire-hash negative-control gate. Stdlib except the Bedrock stream-accumulator
test, which needs botocore's exception types and is skipped on interpreters
without it (the suite still passes bare)."""
import json
import unittest

from analysis.analyze import gate_cell
from harness.request_builder import sha256_hex
from harness.runner import build_schedule
from harness.tasks import PAD_CHAR_TARGETS, TASKS, padded_prompt

try:
    import botocore  # noqa: F401 — presence probe only

    HAVE_BOTOCORE = True
except ImportError:
    HAVE_BOTOCORE = False


class TestPaddedPrompt(unittest.TestCase):
    def test_char_exact_and_deterministic(self):
        base = TASKS["extraction"]["prompt"]
        for label, target in PAD_CHAR_TARGETS.items():
            a = padded_prompt(label, base)
            b = padded_prompt(label, base)
            self.assertEqual(a, b)
            self.assertEqual(len(a), target + 2 + len(base), label)
            self.assertTrue(a.endswith(base))
            a.encode("ascii")  # must not raise

    def test_ladder_monotonic(self):
        base = TASKS["extraction"]["prompt"]
        lengths = [len(padded_prompt(label, base)) for label in ("1k", "10k", "50k")]
        self.assertEqual(lengths, sorted(lengths))
        self.assertLess(lengths[0], lengths[1])
        self.assertLess(lengths[1], lengths[2])


class TestQ3Schedule(unittest.TestCase):
    def test_counts_and_delivery(self):
        items = build_schedule("study2-q3-streaming")
        cells = {it["cell"] for it in items}
        self.assertEqual(len(cells), 12)  # 2 models x 2 tasks x 3 planes
        self.assertEqual(len(items), 12 * 100)
        for it in items:
            self.assertEqual(it["delivery"], "streaming")
            self.assertEqual(it["meta"]["delivery"], "streaming")
            self.assertTrue(it["cell"].endswith("|streamed"))

    def test_planned_sha_matches_nonstreamed_cell(self):
        """The streamed request is parameter-identical to the main grid's —
        only delivery differs, so the planned hash must match."""
        q3 = {it["cell"]: it["sha"] for it in build_schedule("study2-q3-streaming")}
        main = {it["cell"]: it["sha"] for it in build_schedule("study2-pilot")}
        self.assertEqual(
            q3["opus-5|structured_json|p_aws|adaptive|streamed"],
            main["opus-5|structured_json|p_aws|adaptive"],
        )


class TestQ4Schedule(unittest.TestCase):
    def test_counts_and_padding(self):
        items = build_schedule("study2-q4-lengths")
        cells = {it["cell"] for it in items}
        self.assertEqual(len(cells), 18)  # 2 models x 3 pads x 3 planes
        self.assertEqual(len(items), 18 * 25)
        base = TASKS["extraction"]["prompt"]
        for it in items:
            label = it["meta"]["pad"]
            expected_prompt = padded_prompt(label, base)
            if it["plane"] == "bedrock":
                body = json.loads(it["payload"])
                sent = body["messages"][0]["content"][0]["text"]
            else:
                sent = it["payload"]["messages"][0]["content"][0]["text"]
            self.assertEqual(sent, expected_prompt)
            self.assertEqual(it["meta"]["task"], f"extraction_pad_{label}")
            self.assertNotIn("delivery", it)


class FakeStreamClient:
    def __init__(self, events, request_id="rid"):
        self._events = events
        self._request_id = request_id

    def invoke_model_with_response_stream(self, **kwargs):
        return {
            "body": list(self._events),
            "ResponseMetadata": {"RequestId": self._request_id},
        }


def _ev(obj):
    return {"chunk": {"bytes": json.dumps(obj).encode("utf-8")}}


@unittest.skipUnless(HAVE_BOTOCORE, "botocore not installed")
class TestBedrockStreamAccumulator(unittest.TestCase):
    def test_stream_events_normalize(self):
        from harness.planes import BedrockPlane

        plane = BedrockPlane.__new__(BedrockPlane)
        plane.client = FakeStreamClient(
            [
                _ev({"type": "message_start", "message": {
                    "id": "msg1", "model": "claude-opus-5",
                    "usage": {"input_tokens": 10}}}),
                _ev({"type": "content_block_start", "index": 0,
                     "content_block": {"type": "text"}}),
                _ev({"type": "content_block_delta",
                     "delta": {"type": "text_delta", "text": "hel"}}),
                _ev({"type": "content_block_delta",
                     "delta": {"type": "thinking_delta", "thinking": "x"}}),
                _ev({"type": "content_block_delta",
                     "delta": {"type": "text_delta", "text": "lo"}}),
                _ev({"type": "message_delta",
                     "delta": {"stop_reason": "end_turn"},
                     "usage": {"output_tokens": 5}}),
                _ev({"type": "message_stop"}),
            ]
        )
        rec = plane.invoke("model-id", b"BODY", stream=True)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["text"], "hello")
        self.assertEqual(rec["stop_reason"], "end_turn")
        self.assertEqual(rec["usage"], {"input_tokens": 10, "output_tokens": 5})
        self.assertEqual(rec["response_id"], "msg1")
        self.assertEqual(rec["response_model"], "claude-opus-5")
        self.assertEqual(rec["request_id"], "rid")
        self.assertEqual(rec["wire_sha256"], sha256_hex(b"BODY"))
        self.assertTrue(rec["delivered_streaming"])


def _rec(wire, ok=True, planned="P"):
    return {
        "ok": ok,
        "request_sha256": planned,
        "wire_sha256": wire,
        "stop_reason": "end_turn" if ok else None,
        "text": "T",
        "response_model": "m",
    }


class TestWireGate(unittest.TestCase):
    def test_uniform_wire_valid(self):
        gate = gate_cell([_rec("w1"), _rec("w1"), _rec("w1")])
        self.assertFalse(gate["flags"]["negative_control_failed"])
        self.assertFalse(gate["flags"]["wire_mismatch"])
        self.assertEqual(len(gate["valid"]), 3)

    def test_mixed_wire_invalidates_cell(self):
        gate = gate_cell([_rec("w1"), _rec("w2")])
        self.assertTrue(gate["flags"]["negative_control_failed"])
        self.assertTrue(gate["flags"]["wire_mismatch"])
        self.assertEqual(gate["valid"], [])

    def test_legacy_records_without_wire_exempt(self):
        gate = gate_cell([_rec(None), _rec(None)])
        self.assertFalse(gate["flags"]["negative_control_failed"])
        self.assertEqual(len(gate["valid"]), 2)


if __name__ == "__main__":
    unittest.main()
