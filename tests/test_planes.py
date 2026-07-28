"""Tests for harness.planes and the study-2 plane config — the pure pieces
only. Client construction and live behavior belong to the per-plane smoke
(harness.smoke_planes); nothing here touches the network or the SDKs, and
this file must pass on a stdlib-only interpreter."""
import threading
import unittest

from harness.config import MESSAGES_MODEL_IDS, MODELS, PLANES, plane_model_id
from harness.planes import (
    _WireCapture,
    _error_record,
    normalize_message,
    retryable_status,
)
from harness.request_builder import sha256_hex


class TestPlaneConfig(unittest.TestCase):
    def test_planes_roster(self):
        self.assertEqual(PLANES, ("bedrock", "p_aws", "anthropic_api"))

    def test_every_model_has_a_messages_id(self):
        self.assertEqual(set(MESSAGES_MODEL_IDS), set(MODELS))

    def test_bedrock_pinned_to_us_profile(self):
        for key in MODELS:
            self.assertEqual(
                plane_model_id("bedrock", key), MODELS[key]["profiles"]["us"]
            )

    def test_messages_ids_bare_no_provider_prefix(self):
        for plane in ("p_aws", "anthropic_api"):
            for key in MODELS:
                mid = plane_model_id(plane, key)
                self.assertFalse(mid.startswith("us."))
                self.assertFalse(mid.startswith("global."))
                self.assertFalse(mid.startswith("anthropic."))

    def test_haiku_stays_dated_on_messages_planes(self):
        self.assertEqual(
            plane_model_id("p_aws", "haiku-4-5"), "claude-haiku-4-5-20251001"
        )

    def test_unknown_plane_raises(self):
        with self.assertRaises(ValueError):
            plane_model_id("vertex", "opus-5")


class TestRetryableStatus(unittest.TestCase):
    def test_retryable(self):
        for status in (408, 429, 500, 502, 503, 504, 529, 561):
            self.assertTrue(retryable_status(status), status)

    def test_terminal(self):
        for status in (400, 401, 403, 404, 413, 422):
            self.assertFalse(retryable_status(status), status)

    def test_none_is_not_a_status(self):
        self.assertFalse(retryable_status(None))


class TestNormalizeMessage(unittest.TestCase):
    PAYLOAD = {
        "id": "msg_01",
        "model": "claude-opus-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "content": [
            {"type": "thinking", "thinking": "..."},
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ],
    }

    def test_text_concatenates_text_blocks_only(self):
        rec = normalize_message(self.PAYLOAD, "req-1", 123, b"WIRE")
        self.assertEqual(rec["text"], "hello world")
        self.assertEqual(rec["text_sha256"], sha256_hex(b"hello world"))

    def test_record_fields(self):
        rec = normalize_message(self.PAYLOAD, "req-1", 123, b"WIRE")
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["request_id"], "req-1")
        self.assertEqual(rec["latency_ms"], 123)
        self.assertEqual(rec["response_id"], "msg_01")
        self.assertEqual(rec["response_model"], "claude-opus-5")
        self.assertEqual(rec["stop_reason"], "end_turn")
        self.assertEqual(rec["usage"], {"input_tokens": 10, "output_tokens": 5})
        self.assertEqual(rec["wire_sha256"], sha256_hex(b"WIRE"))

    def test_missing_wire_bytes_recorded_as_none(self):
        rec = normalize_message(self.PAYLOAD, None, 1, None)
        self.assertIsNone(rec["wire_sha256"])


class TestErrorRecord(unittest.TestCase):
    def test_terminal_400(self):
        rec = _error_record("invalid_request_error", "bad", 400, "req-2", b"W")
        self.assertFalse(rec["ok"])
        self.assertFalse(rec["retryable"])
        self.assertEqual(rec["status_code"], 400)
        self.assertEqual(rec["error_code"], "invalid_request_error")
        self.assertEqual(rec["wire_sha256"], sha256_hex(b"W"))

    def test_rate_limit_retryable(self):
        rec = _error_record("rate_limit_error", "slow down", 429, None, None)
        self.assertTrue(rec["retryable"])

    def test_connection_error_retryable(self):
        rec = _error_record("APIConnectionError", "boom", None, None, None)
        self.assertTrue(rec["retryable"])

    def test_message_truncated(self):
        rec = _error_record("x", "y" * 1000, 400, None, None)
        self.assertEqual(len(rec["error_message"]), 400)


class _FakeRequest:
    def __init__(self, content):
        self.content = content


class TestWireCapture(unittest.TestCase):
    def test_take_returns_and_clears(self):
        cap = _WireCapture()
        cap(_FakeRequest(b"BODY"))
        self.assertEqual(cap.take(), b"BODY")
        self.assertIsNone(cap.take())

    def test_thread_local_no_cross_talk(self):
        cap = _WireCapture()
        seen = {}

        def worker(name, body):
            cap(_FakeRequest(body))
            seen[name] = cap.take()

        threads = [
            threading.Thread(target=worker, args=(f"t{i}", f"B{i}".encode()))
            for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(seen, {f"t{i}": f"B{i}".encode() for i in range(8)})

    def test_unreadable_content_captures_none(self):
        class Exploding:
            @property
            def content(self):
                raise RuntimeError("streaming body")

        cap = _WireCapture()
        cap(Exploding())
        self.assertIsNone(cap.take())


if __name__ == "__main__":
    unittest.main()
