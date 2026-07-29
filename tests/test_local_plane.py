"""Tests for the study-3 local plane: canonical_local_body + LocalPlane.

All plane tests inject a fake opener — no network, no Ollama, bare stdlib.
"""
import io
import json
import unittest
import urllib.error

from harness.planes import LocalPlane, make_plane, normalize_local_message
from harness.request_builder import (
    canonical_bytes,
    canonical_local_body,
    sha256_hex,
)


class TestCanonicalLocalBody(unittest.TestCase):
    def test_greedy_body_shape_and_determinism(self):
        a = canonical_local_body(
            "gpt-oss:20b", "hello", "none",
            options={"temperature": 0, "seed": 42},
        )
        b = canonical_local_body(
            "gpt-oss:20b", "hello", "none",
            options={"seed": 42, "temperature": 0},  # insertion order differs
        )
        self.assertEqual(a, b)
        body = json.loads(a)
        self.assertEqual(body["model"], "gpt-oss:20b")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])
        self.assertIs(body["stream"], False)
        self.assertEqual(body["options"], {"temperature": 0, "seed": 42})
        self.assertNotIn("think", body)

    def test_canonical_serialization_is_sorted_compact_ascii(self):
        raw = canonical_local_body("m", "pé", "none").decode("ascii")
        self.assertNotIn(" ", raw)
        self.assertIn("\\u00e9", raw)
        keys = list(json.loads(raw))
        self.assertEqual(keys, sorted(keys))

    def test_thinking_arms_encode_per_family(self):
        for arm, expected in (
            ("think_on", True),
            ("think_off", False),
            ("effort_low", "low"),
            ("effort_high", "high"),
        ):
            body = json.loads(canonical_local_body("m", "p", arm))
            self.assertEqual(body["think"], expected, arm)

    def test_none_arm_omits_think_and_unknown_raises(self):
        body = json.loads(canonical_local_body("m", "p", "none"))
        self.assertNotIn("think", body)
        with self.assertRaises(ValueError):
            canonical_local_body("m", "p", "adaptive")  # study-2 arm, not local

    def test_keep_alive_and_extra_merge(self):
        body = json.loads(
            canonical_local_body(
                "m", "p", "none", keep_alive="10m", extra={"format": "json"}
            )
        )
        self.assertEqual(body["keep_alive"], "10m")
        self.assertEqual(body["format"], "json")

    def test_prompt_matches_messages_plane_semantics(self):
        # Same prompt text lands in both shapes; structure differs by
        # construction (plain string here, typed blocks on the API planes).
        body = json.loads(canonical_local_body("m", "same prompt", "none"))
        self.assertEqual(body["messages"][0]["content"], "same prompt")


def _ok_payload(**overrides):
    payload = {
        "model": "gpt-oss:20b",
        "created_at": "2026-07-29T00:00:00Z",
        "message": {"role": "assistant", "content": "hi there"},
        "done": True,
        "done_reason": "stop",
        "total_duration": 2_000_000_000,
        "load_duration": 100_000_000,
        "prompt_eval_count": 12,
        "prompt_eval_duration": 50_000_000,
        "eval_count": 34,
        "eval_duration": 1_500_000_000,
    }
    payload.update(overrides)
    return payload


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _opener_returning(payload):
    def opener(request, timeout=None):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))
    return opener


class TestNormalizeLocalMessage(unittest.TestCase):
    def test_success_record_shape(self):
        wire = canonical_local_body("gpt-oss:20b", "p", "none")
        record = normalize_local_message(_ok_payload(), 1234, wire)
        self.assertTrue(record["ok"])
        self.assertEqual(record["text"], "hi there")
        self.assertEqual(record["text_sha256"], sha256_hex(b"hi there"))
        self.assertEqual(record["wire_sha256"], sha256_hex(wire))
        self.assertEqual(record["latency_ms"], 1234)
        self.assertEqual(record["response_model"], "gpt-oss:20b")
        self.assertEqual(record["stop_reason"], "stop")
        self.assertIsNone(record["response_id"])
        self.assertEqual(record["usage"]["input_tokens"], 12)
        self.assertEqual(record["usage"]["output_tokens"], 34)
        self.assertEqual(record["usage"]["eval_duration_ns"], 1_500_000_000)
        self.assertEqual(record["usage"]["load_duration_ns"], 100_000_000)

    def test_thinking_text_recorded_as_size_covariate(self):
        payload = _ok_payload()
        payload["message"]["thinking"] = "let me think"
        record = normalize_local_message(payload, 1, b"x")
        self.assertEqual(record["usage"]["thinking_chars"], len("let me think"))
        # thinking text itself is not the endpoint; only content text hashes
        self.assertEqual(record["text"], "hi there")


class TestLocalPlaneInvoke(unittest.TestCase):
    def test_success_invoke(self):
        plane = LocalPlane(opener=_opener_returning(_ok_payload()))
        body = canonical_local_body("gpt-oss:20b", "p", "none")
        record = plane.invoke(body)
        self.assertTrue(record["ok"])
        self.assertIs(record["delivered_streaming"], False)
        self.assertEqual(record["wire_sha256"], sha256_hex(body))
        self.assertGreaterEqual(record["latency_ms"], 0)
        # raw payload retained for probes (logprobs exposure inspection)
        self.assertEqual(plane.last_payload["message"]["content"], "hi there")

    def test_stream_not_supported(self):
        plane = LocalPlane(opener=_opener_returning(_ok_payload()))
        with self.assertRaises(ValueError):
            plane.invoke(b"{}", stream=True)

    def test_http_404_not_retryable_with_ollama_error_text(self):
        def opener(request, timeout=None):
            raise urllib.error.HTTPError(
                "http://x", 404, "Not Found", None,
                io.BytesIO(b'{"error":"model \'nope\' not found"}'),
            )
        record = LocalPlane(opener=opener).invoke(b"{}")
        self.assertFalse(record["ok"])
        self.assertEqual(record["status_code"], 404)
        self.assertFalse(record["retryable"])
        self.assertEqual(record["error_code"], "http_404")
        self.assertIn("not found", record["error_message"])

    def test_http_500_retryable(self):
        def opener(request, timeout=None):
            raise urllib.error.HTTPError(
                "http://x", 500, "boom", None, io.BytesIO(b"server error")
            )
        record = LocalPlane(opener=opener).invoke(b"{}")
        self.assertFalse(record["ok"])
        self.assertTrue(record["retryable"])

    def test_connection_error_retryable_no_status(self):
        def opener(request, timeout=None):
            raise urllib.error.URLError("connection refused")
        record = LocalPlane(opener=opener).invoke(b"{}")
        self.assertFalse(record["ok"])
        self.assertIsNone(record["status_code"])
        self.assertTrue(record["retryable"])

    def test_timeout_retryable(self):
        def opener(request, timeout=None):
            raise TimeoutError("timed out")
        record = LocalPlane(opener=opener).invoke(b"{}")
        self.assertFalse(record["ok"])
        self.assertTrue(record["retryable"])

    def test_request_targets_chat_endpoint_with_wire_bytes(self):
        seen = {}
        def opener(request, timeout=None):
            seen["url"] = request.full_url
            seen["data"] = request.data
            seen["content_type"] = request.get_header("Content-type")
            return _FakeResponse(json.dumps(_ok_payload()).encode("utf-8"))
        plane = LocalPlane(base_url="http://192.168.1.245:11434", opener=opener)
        body = canonical_local_body("gpt-oss:20b", "p", "none")
        plane.invoke(body)
        self.assertEqual(seen["url"], "http://192.168.1.245:11434/api/chat")
        self.assertEqual(seen["data"], body)
        self.assertEqual(seen["content_type"], "application/json")


class TestLocalPlaneHelpers(unittest.TestCase):
    def test_engine_version(self):
        plane = LocalPlane(opener=_opener_returning({"version": "0.30.5"}))
        self.assertEqual(plane.engine_version(), "0.30.5")

    def test_box_state_snapshot(self):
        ps = {
            "models": [
                {"name": "gpt-oss:20b", "digest": "d1", "size": 13,
                 "size_vram": 13, "expires_at": "soon", "irrelevant": "x"},
            ]
        }
        plane = LocalPlane(opener=_opener_returning(ps))
        state = plane.box_state()
        self.assertEqual(state["resident_models"], [
            {"name": "gpt-oss:20b", "digest": "d1", "size": 13,
             "size_vram": 13, "expires_at": "soon"},
        ])
        self.assertIn("captured_utc", state)

    def test_box_state_empty_server(self):
        plane = LocalPlane(opener=_opener_returning({"models": []}))
        self.assertEqual(plane.box_state()["resident_models"], [])

    def test_model_digest_from_tags(self):
        tags = {
            "models": [
                {"name": "other:1b", "digest": "aaa"},
                {"name": "gpt-oss:20b", "digest": "17052f91a42edeadbeef"},
            ]
        }
        plane = LocalPlane(opener=_opener_returning(tags))
        self.assertEqual(plane.model_digest("gpt-oss:20b"), "17052f91a42edeadbeef")

    def test_model_digest_missing_model_raises(self):
        plane = LocalPlane(opener=_opener_returning({"models": []}))
        with self.assertRaises(KeyError):
            plane.model_digest("gpt-oss:20b")


class TestMakePlane(unittest.TestCase):
    def test_local_plane_constructible_by_name(self):
        plane = make_plane("local", opener=_opener_returning({"version": "x"}))
        self.assertIsInstance(plane, LocalPlane)
        self.assertEqual(plane.name, "local")

    def test_custom_name_for_per_box_instances(self):
        plane = LocalPlane(name="local_cuda", opener=_opener_returning({}))
        self.assertEqual(plane.name, "local_cuda")


class TestCanonicalBytesParity(unittest.TestCase):
    def test_wire_hash_is_planned_hash_by_construction(self):
        body = canonical_local_body("m", "p", "think_on", options={"seed": 7})
        self.assertEqual(sha256_hex(body), sha256_hex(canonical_bytes(json.loads(body))))


if __name__ == "__main__":
    unittest.main()
