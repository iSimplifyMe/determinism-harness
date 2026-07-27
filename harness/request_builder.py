"""Canonical Bedrock request bodies.

The negative control requires that all repeats in a cell send byte-identical
payloads: bodies are serialized with sorted keys, compact separators, and
ASCII escapes, and the bytes that are hashed are exactly the bytes sent.
"""
import hashlib
import json

BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"

THINKING_ARMS = ("adaptive", "disabled", "none")


def canonical_body(model_cfg, prompt, thinking_arm, extra=None):
    """Build the InvokeModel body as canonical bytes.

    model_cfg needs `max_tokens` and `effort` (None on models where the
    effort parameter is rejected, e.g. Haiku 4.5). thinking_arm is
    "adaptive" / "disabled" (Claude 5 family) or "none" (omit the field —
    the only legal state on Haiku 4.5). `extra` merges additional top-level
    fields, e.g. temperature for the positive control.
    """
    body = {
        "anthropic_version": BEDROCK_ANTHROPIC_VERSION,
        "max_tokens": model_cfg["max_tokens"],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }
    if thinking_arm == "adaptive":
        body["thinking"] = {"type": "adaptive"}
    elif thinking_arm == "disabled":
        body["thinking"] = {"type": "disabled"}
    elif thinking_arm != "none":
        raise ValueError(f"unknown thinking arm: {thinking_arm}")
    if model_cfg.get("effort"):
        body["output_config"] = {"effort": model_cfg["effort"]}
    if extra:
        for key, value in extra.items():
            body[key] = value
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()
