"""Per-plane clients presenting one normalized invoke surface.

Three serving planes, one semantic request (PREREGISTRATION-v2 section 4):

- bedrock       boto3 bedrock-runtime InvokeModel — the study-1 path,
                pinned to the `us.` inference profile in study 2. The
                canonical body bytes are sent as-is; hashed == sent.
- p_aws         Claude Platform on AWS via AnthropicAWS. SigV4; requires an
                AWS region and ANTHROPIC_AWS_WORKSPACE_ID (constructor arg
                or environment). Bare model IDs.
- anthropic_api First-party Messages API via Anthropic. Run-scoped API key
                (never a global export — resolve from the process
                environment of the run). Bare model IDs.

On the Messages planes the SDK owns wire serialization, so byte-identity is
not assumed: a request event hook on the HTTP client captures the exact body
bytes sent, and each record carries their SHA-256 (`wire_sha256`) alongside
the planned-request hash. Within-cell identity is then a verifiable readout,
exactly like the Bedrock negative control.

Retry policy mirrors the study-1 runner: the SDK/boto3 retry layers are
disabled (max_retries=0); every attempt is the caller's to count. Timeouts
match study 1 (600s read / 10s connect); the explicit client timeout also
lifts the SDK's non-streaming large-max_tokens guard.

This module is a runner-only dependency; the analysis pipeline stays stdlib.
"""
import threading
import time

from harness.config import REGION
from harness.request_builder import sha256_hex

# Statuses the runner may retry, matching the spirit of the Bedrock
# RETRYABLE_CODES set: rate limits, overload, and server-side failures.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}


def retryable_status(status_code):
    if status_code is None:
        return False
    return status_code in RETRYABLE_STATUS or status_code >= 500


class _WireCapture:
    """httpx request event hook. Thread-local, so workers sharing a client
    cannot cross-attribute wire bytes: the hook runs synchronously in the
    calling thread, and take() is read immediately after the call returns."""

    def __init__(self):
        self._local = threading.local()

    def __call__(self, request):
        try:
            self._local.content = bytes(request.content)
        except Exception:
            self._local.content = None

    def take(self):
        content = getattr(self._local, "content", None)
        self._local.content = None
        return content


def normalize_message(payload_dict, request_id, latency_ms, wire_bytes):
    """Shared success-record shape across all three planes. `payload_dict`
    is the Messages-API response as a plain dict (Bedrock body JSON, or
    SDK Message.to_dict())."""
    text = "".join(
        block.get("text", "")
        for block in payload_dict.get("content", [])
        if block.get("type") == "text"
    )
    return {
        "ok": True,
        "latency_ms": latency_ms,
        "request_id": request_id,
        "response_id": payload_dict.get("id"),
        "response_model": payload_dict.get("model"),
        "stop_reason": payload_dict.get("stop_reason"),
        "usage": payload_dict.get("usage"),
        "text": text,
        "text_sha256": sha256_hex(text.encode("utf-8")),
        "wire_sha256": sha256_hex(wire_bytes) if wire_bytes else None,
    }


def _error_record(error_code, error_message, status_code, request_id, wire_bytes):
    return {
        "ok": False,
        "error_code": error_code,
        "error_message": (error_message or "")[:400],
        "status_code": status_code,
        "retryable": retryable_status(status_code)
        if status_code is not None
        else True,  # connection-level failures are retryable
        "request_id": request_id,
        "wire_sha256": sha256_hex(wire_bytes) if wire_bytes else None,
    }


class MessagesPlane:
    """Base for the two SDK planes. Subclasses supply _make_client()."""

    name = None

    def __init__(self):
        self._capture = _WireCapture()
        self.client = self._make_client()

    def _make_client(self):
        raise NotImplementedError

    def _http_client(self):
        from anthropic import DefaultHttpxClient

        return DefaultHttpxClient(event_hooks={"request": [self._capture]})

    def _timeout(self):
        import httpx

        return httpx.Timeout(600.0, connect=10.0)

    def invoke(self, params):
        """One Messages call; params from canonical_messages_params()."""
        import anthropic

        start = time.monotonic()
        try:
            msg = self.client.messages.create(**params)
            latency_ms = int((time.monotonic() - start) * 1000)
            return normalize_message(
                msg.to_dict(),
                getattr(msg, "_request_id", None),
                latency_ms,
                self._capture.take(),
            )
        except anthropic.APIStatusError as err:
            code = getattr(err, "type", None) or f"http_{err.status_code}"
            request_id = None
            if getattr(err, "response", None) is not None:
                request_id = err.response.headers.get("request-id")
            return _error_record(
                code, str(err), err.status_code, request_id, self._capture.take()
            )
        except anthropic.APIConnectionError as err:
            return _error_record(
                type(err).__name__, str(err), None, None, self._capture.take()
            )


class PAWSPlane(MessagesPlane):
    """Claude Platform on AWS: SigV4-signed Messages API, bare model IDs."""

    name = "p_aws"

    def __init__(self, aws_region=REGION, workspace_id=None):
        self._aws_region = aws_region
        self._workspace_id = workspace_id
        super().__init__()

    def _make_client(self):
        from anthropic import AnthropicAWS

        kwargs = {
            "aws_region": self._aws_region,
            "max_retries": 0,
            "timeout": self._timeout(),
            "http_client": self._http_client(),
        }
        if self._workspace_id:
            kwargs["workspace_id"] = self._workspace_id
        return AnthropicAWS(**kwargs)


class FirstPartyPlane(MessagesPlane):
    """First-party Anthropic API. The key resolves from the run's process
    environment (ANTHROPIC_API_KEY) unless passed explicitly."""

    name = "anthropic_api"

    def __init__(self, api_key=None):
        self._api_key = api_key
        super().__init__()

    def _make_client(self):
        from anthropic import Anthropic

        kwargs = {
            "max_retries": 0,
            "timeout": self._timeout(),
            "http_client": self._http_client(),
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return Anthropic(**kwargs)


class BedrockPlane:
    """Study-1 invocation path behind the shared record shape. Canonical
    body bytes are sent verbatim, so wire_sha256 is computed directly."""

    name = "bedrock"

    def __init__(self, region=REGION):
        import boto3
        from botocore.config import Config

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                read_timeout=600, connect_timeout=10, retries={"max_attempts": 0}
            ),
        )

    def invoke(self, model_id, body_bytes):
        import json

        from botocore.exceptions import (
            BotoCoreError,
            ClientError,
        )

        start = time.monotonic()
        try:
            resp = self.client.invoke_model(
                modelId=model_id,
                body=body_bytes,
                contentType="application/json",
                accept="application/json",
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            payload = json.loads(resp["body"].read())
            request_id = resp.get("ResponseMetadata", {}).get("RequestId")
            return normalize_message(payload, request_id, latency_ms, body_bytes)
        except ClientError as err:
            error = err.response.get("Error", {})
            meta = err.response.get("ResponseMetadata", {})
            return _error_record(
                error.get("Code", "ClientError"),
                str(err),
                meta.get("HTTPStatusCode"),
                meta.get("RequestId"),
                body_bytes,
            )
        except BotoCoreError as err:
            return _error_record(type(err).__name__, str(err), None, None, body_bytes)


def make_plane(plane_name, **kwargs):
    if plane_name == "bedrock":
        return BedrockPlane(**kwargs)
    if plane_name == "p_aws":
        return PAWSPlane(**kwargs)
    if plane_name == "anthropic_api":
        return FirstPartyPlane(**kwargs)
    raise ValueError(f"unknown plane: {plane_name}")
