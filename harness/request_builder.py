"""Canonical request bodies, per serving plane.

The negative control requires that all repeats in a cell send byte-identical
payloads: bodies are serialized with sorted keys, compact separators, and
ASCII escapes. On Bedrock the bytes hashed are exactly the bytes sent. On the
Messages planes (Claude Platform on AWS, first-party API) the SDK owns the
wire serialization, so the canonical bytes here are the *planned-request*
hash; the wire bytes actually sent are captured and hashed separately in
harness.planes, and within-cell byte-identity is verified from the records.

Cross-plane request shapes differ by construction (prereg v2 section 4):
Bedrock InvokeModel carries `anthropic_version` and no `model`; the Messages
planes carry `model` and no `anthropic_version`. Cross-plane identity is
semantic (same prompts, same parameters); byte-identity claims are always
within-cell, within-plane.
"""
import hashlib
import json

BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"

THINKING_ARMS = ("adaptive", "disabled", "none")


def _apply_arm_fields(body, model_cfg, thinking_arm, extra):
    """Shared thinking / effort / extra application — the factor encoding
    must be identical across planes for the semantic-identity claim to hold."""
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
    return body


def canonical_body(model_cfg, prompt, thinking_arm, extra=None):
    """Build the Bedrock InvokeModel body as canonical bytes.

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
    _apply_arm_fields(body, model_cfg, thinking_arm, extra)
    return canonical_bytes(body)


def canonical_messages_params(model_cfg, model_id, prompt, thinking_arm, extra=None):
    """Build the Messages-API request params (dict) for the SDK planes.

    Same factor encoding as the Bedrock body; the structural differences are
    exactly the two the planes impose: `model` present, `anthropic_version`
    absent. The returned dict is passed to messages.create(**params); hash
    canonical_bytes() of it for the planned-request record.
    """
    params = {
        "model": model_id,
        "max_tokens": model_cfg["max_tokens"],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }
    return _apply_arm_fields(params, model_cfg, thinking_arm, extra)


def canonical_bytes(obj):
    """The one canonical serialization: sorted keys, compact, ASCII."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()
