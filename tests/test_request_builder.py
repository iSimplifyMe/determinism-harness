"""Tests for harness.request_builder — the negative control (identical request
bytes within a cell) is only as good as this module's determinism."""
import json
import unittest

from harness.request_builder import (
    canonical_body,
    canonical_bytes,
    canonical_messages_params,
    sha256_hex,
)

CFG_5FAMILY = {"max_tokens": 16000, "effort": "medium"}
CFG_HAIKU = {"max_tokens": 8192, "effort": None}


class TestCanonicalBody(unittest.TestCase):
    def test_shape_adaptive(self):
        body = json.loads(canonical_body(CFG_5FAMILY, "PROMPT", "adaptive"))
        self.assertEqual(body["anthropic_version"], "bedrock-2023-05-31")
        self.assertEqual(body["max_tokens"], 16000)
        self.assertEqual(body["thinking"], {"type": "adaptive"})
        self.assertEqual(body["output_config"], {"effort": "medium"})
        self.assertEqual(
            body["messages"],
            [{"role": "user", "content": [{"type": "text", "text": "PROMPT"}]}],
        )

    def test_shape_disabled(self):
        body = json.loads(canonical_body(CFG_5FAMILY, "PROMPT", "disabled"))
        self.assertEqual(body["thinking"], {"type": "disabled"})

    def test_none_omits_thinking(self):
        body = json.loads(canonical_body(CFG_HAIKU, "PROMPT", "none"))
        self.assertNotIn("thinking", body)

    def test_null_effort_omits_output_config(self):
        body = json.loads(canonical_body(CFG_HAIKU, "PROMPT", "none"))
        self.assertNotIn("output_config", body)

    def test_extra_params_included(self):
        body = json.loads(
            canonical_body(CFG_HAIKU, "PROMPT", "none", extra={"temperature": 0.7})
        )
        self.assertEqual(body["temperature"], 0.7)

    def test_deterministic_bytes(self):
        a = canonical_body(CFG_5FAMILY, "PROMPT", "adaptive")
        b = canonical_body(CFG_5FAMILY, "PROMPT", "adaptive")
        self.assertEqual(a, b)
        self.assertEqual(sha256_hex(a), sha256_hex(b))
        self.assertEqual(len(sha256_hex(a)), 64)

    def test_extra_key_order_irrelevant(self):
        e1 = {}
        e1["temperature"] = 0.7
        e1["top_p"] = 0.9
        e2 = {}
        e2["top_p"] = 0.9
        e2["temperature"] = 0.7
        a = canonical_body(CFG_HAIKU, "PROMPT", "none", extra=e1)
        b = canonical_body(CFG_HAIKU, "PROMPT", "none", extra=e2)
        self.assertEqual(a, b)

    def test_ascii_escaped(self):
        raw = canonical_body(CFG_HAIKU, "café", "none")
        self.assertIn(b"caf\\u00e9", raw)

    def test_compact_separators(self):
        raw = canonical_body(CFG_HAIKU, "no-spaces-here", "none")
        self.assertNotIn(b": ", raw)
        self.assertNotIn(b", ", raw)


class TestCanonicalMessagesParams(unittest.TestCase):
    def test_shape_adaptive(self):
        params = canonical_messages_params(
            CFG_5FAMILY, "claude-opus-5", "PROMPT", "adaptive"
        )
        self.assertEqual(params["model"], "claude-opus-5")
        self.assertNotIn("anthropic_version", params)
        self.assertEqual(params["max_tokens"], 16000)
        self.assertEqual(params["thinking"], {"type": "adaptive"})
        self.assertEqual(params["output_config"], {"effort": "medium"})
        self.assertEqual(
            params["messages"],
            [{"role": "user", "content": [{"type": "text", "text": "PROMPT"}]}],
        )

    def test_shape_disabled(self):
        params = canonical_messages_params(
            CFG_5FAMILY, "claude-sonnet-5", "PROMPT", "disabled"
        )
        self.assertEqual(params["thinking"], {"type": "disabled"})

    def test_none_omits_thinking_and_effort(self):
        params = canonical_messages_params(
            CFG_HAIKU, "claude-haiku-4-5-20251001", "PROMPT", "none"
        )
        self.assertNotIn("thinking", params)
        self.assertNotIn("output_config", params)

    def test_extra_params_included(self):
        params = canonical_messages_params(
            CFG_HAIKU, "claude-haiku-4-5-20251001", "PROMPT", "none",
            extra={"temperature": 0.7},
        )
        self.assertEqual(params["temperature"], 0.7)

    def test_unknown_arm_raises(self):
        with self.assertRaises(ValueError):
            canonical_messages_params(CFG_5FAMILY, "claude-opus-5", "P", "bogus")

    def test_deterministic_bytes(self):
        a = canonical_bytes(
            canonical_messages_params(CFG_5FAMILY, "claude-opus-5", "PROMPT", "adaptive")
        )
        b = canonical_bytes(
            canonical_messages_params(CFG_5FAMILY, "claude-opus-5", "PROMPT", "adaptive")
        )
        self.assertEqual(a, b)
        self.assertEqual(len(sha256_hex(a)), 64)

    def test_cross_plane_semantic_identity(self):
        """Prereg v2 section 4: the ONLY structural differences between the
        Bedrock body and the Messages params are `anthropic_version` (Bedrock
        only) and `model` (Messages only). Everything else must be equal."""
        for cfg, arm, extra in (
            (CFG_5FAMILY, "adaptive", None),
            (CFG_5FAMILY, "disabled", None),
            (CFG_HAIKU, "none", {"temperature": 0.7}),
        ):
            bedrock = json.loads(canonical_body(cfg, "PROMPT", arm, extra=extra))
            messages = dict(
                canonical_messages_params(cfg, "any-id", "PROMPT", arm, extra=extra)
            )
            del bedrock["anthropic_version"]
            del messages["model"]
            self.assertEqual(bedrock, messages)


if __name__ == "__main__":
    unittest.main()
