"""Study-4 (PREREGISTRATION-v4) builder and door-client invariants.

Pure unit tests: no network, no subprocess. The live counterpart is
harness/smoke_study4.py, whose evidence files are a freeze-checklist item.
"""
import json
import unittest

from harness.config import (
    STUDY4_DOORS,
    STUDY4_EFFORT_ARMS,
    STUDY4_GRID_DOORS,
    STUDY4_MAX_OUTPUT_TOKENS,
)
from harness.doors import CodexDoor, normalize_responses
from harness.request_builder import (
    canonical_responses_body,
    codex_argv,
    converse_request,
    sha256_hex,
)


class ResponsesBodyTest(unittest.TestCase):
    def test_pinned_arm_exact_bytes(self):
        body = canonical_responses_body("gpt-5.6-sol", "hello", "none", 16000)
        self.assertEqual(
            body,
            b'{"input":"hello","max_output_tokens":16000,'
            b'"model":"gpt-5.6-sol","reasoning":{"effort":"none"}}',
        )

    def test_default_arm_omits_reasoning_entirely(self):
        body = json.loads(
            canonical_responses_body("gpt-5.6-sol", "hi", "default", 16000)
        )
        self.assertNotIn("reasoning", body)

    def test_no_store_field_ever(self):
        # v4 section 3: 1P's store-by-default is disclosed, not neutralized.
        for arm in ("none", "high", "default"):
            body = json.loads(
                canonical_responses_body("gpt-5.6-sol", "x", arm, 16000)
            )
            self.assertNotIn("store", body)

    def test_unknown_arm_rejected(self):
        with self.assertRaises(ValueError):
            canonical_responses_body("gpt-5.6-sol", "x", "medium-ish", 16000)

    def test_1p_mantle_parity_modulo_model_alias(self):
        # The two Responses doors share one builder; their bodies may differ
        # in exactly the model value and nothing else. (Whether mantle also
        # accepts the bare alias — full byte-identity — is a smoke question.)
        one = json.loads(
            canonical_responses_body(
                STUDY4_DOORS["openai_1p"]["model_id"], "p", "high",
                STUDY4_MAX_OUTPUT_TOKENS,
            )
        )
        two = json.loads(
            canonical_responses_body(
                STUDY4_DOORS["mantle"]["model_id"], "p", "high",
                STUDY4_MAX_OUTPUT_TOKENS,
            )
        )
        one.pop("model")
        two.pop("model")
        self.assertEqual(one, two)

    def test_same_alias_means_byte_identity(self):
        a = canonical_responses_body("gpt-5.6-sol", "p", "high", 16000)
        b = canonical_responses_body("gpt-5.6-sol", "p", "high", 16000)
        self.assertEqual(sha256_hex(a), sha256_hex(b))


class ConverseRequestTest(unittest.TestCase):
    def test_effort_pin_rides_nested_in_amrf(self):
        kwargs = converse_request("us.openai.gpt-5.6-sol", "p", "high", 16000)
        self.assertEqual(
            kwargs["additionalModelRequestFields"],
            {"reasoning": {"effort": "high"}},
        )
        self.assertEqual(kwargs["inferenceConfig"], {"maxTokens": 16000})
        self.assertEqual(
            kwargs["messages"], [{"role": "user", "content": [{"text": "p"}]}]
        )

    def test_default_arm_omits_amrf(self):
        kwargs = converse_request("us.openai.gpt-5.6-sol", "p", "default", 16000)
        self.assertNotIn("additionalModelRequestFields", kwargs)

    def test_no_sampling_fields_anywhere(self):
        # supports_sampling: False is a registered door fact; the builder
        # must be incapable of expressing temperature/top_p/seed.
        kwargs = converse_request("us.openai.gpt-5.6-sol", "p", "none", 16000)
        flat = json.dumps(kwargs)
        for banned in ("temperature", "topP", "top_p", "seed"):
            self.assertNotIn(banned, flat)


class CodexArgvTest(unittest.TestCase):
    def test_exact_argv_and_pin(self):
        argv = codex_argv("gpt-5.6-sol", "the prompt", "none", "/tmp/wd")
        self.assertEqual(
            argv,
            [
                "codex", "exec", "--json", "--ephemeral",
                "-m", "gpt-5.6-sol", "-s", "read-only",
                "-C", "/tmp/wd", "--skip-git-repo-check",
                "-c", "model_reasoning_effort=none",
                "--", "the prompt",
            ],
        )

    def test_default_arm_is_illegal_on_codex(self):
        with self.assertRaises(ValueError):
            codex_argv("gpt-5.6-sol", "p", "default", "/tmp/wd")

    def test_receipt_argv_is_measured_argv_minus_json(self):
        from harness.request_builder import codex_receipt_argv

        measured = codex_argv("gpt-5.6-sol", "Reply with exactly: OK",
                              "high", "/tmp/wd")
        receipt = codex_receipt_argv("gpt-5.6-sol", "Reply with exactly: OK",
                                     "high", "/tmp/wd")
        measured.remove("--json")
        self.assertEqual(receipt, measured)


class NormalizeResponsesTest(unittest.TestCase):
    PAYLOAD = {
        "id": "resp_123",
        "model": "gpt-5.6-sol",
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "part one "},
                    {"type": "output_text", "text": "part two"},
                ],
            },
        ],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }

    def test_text_join_and_fields(self):
        record = normalize_responses(self.PAYLOAD, "req-1", 42, b"body")
        self.assertTrue(record["ok"])
        self.assertEqual(record["text"], "part one part two")
        self.assertEqual(record["response_id"], "resp_123")
        self.assertEqual(record["response_model"], "gpt-5.6-sol")
        self.assertEqual(record["stop_reason"], "completed")
        self.assertEqual(
            record["usage"]["output_tokens_details"]["reasoning_tokens"], 0
        )
        self.assertEqual(record["wire_sha256"], sha256_hex(b"body"))

    def test_incomplete_status_carries_reason(self):
        payload = dict(self.PAYLOAD)
        payload["status"] = "incomplete"
        payload["incomplete_details"] = {"reason": "max_output_tokens"}
        record = normalize_responses(payload, None, 1, b"x")
        self.assertEqual(record["stop_reason"], "incomplete:max_output_tokens")


class CodexParsingTest(unittest.TestCase):
    STDOUT = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "th_1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "16"},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 13234,
                        "cached_input_tokens": 9984,
                        "output_tokens": 98,
                        "reasoning_output_tokens": 91,
                    },
                }
            ),
        ]
    )
    STDERR = "\n".join(
        [
            "OpenAI Codex v0.147.0",
            "--------",
            "model: gpt-5.6-sol",
            "provider: openai",
            "reasoning effort: high",
            "--------",
        ]
    )

    def test_stdout_parse(self):
        text, usage, thread_id = CodexDoor._parse_stdout(self.STDOUT)
        self.assertEqual(text, "16")
        self.assertEqual(usage["reasoning_output_tokens"], 91)
        self.assertEqual(thread_id, "th_1")

    def test_banner_parse(self):
        receipts = CodexDoor._parse_banner(self.STDERR)
        self.assertEqual(receipts["model"], "gpt-5.6-sol")
        self.assertEqual(receipts["provider"], "openai")
        self.assertEqual(receipts["reasoning_effort"], "high")


class Study4ConfigTest(unittest.TestCase):
    def test_grid_doors_are_registered_doors(self):
        for door in STUDY4_GRID_DOORS:
            self.assertIn(door, STUDY4_DOORS)

    def test_runtime_global_is_q4_only(self):
        self.assertNotIn("runtime_global", STUDY4_GRID_DOORS)
        self.assertIn("runtime_global", STUDY4_DOORS)

    def test_confirmatory_arms_are_pinned_only(self):
        self.assertEqual(STUDY4_EFFORT_ARMS, ("none", "high"))
