"""Per-plane clients presenting one normalized invoke surface.

Three API serving planes, one semantic request (PREREGISTRATION-v2
section 4), plus the study-3 local plane (PREREGISTRATION-v3):

- bedrock       boto3 bedrock-runtime InvokeModel — the study-1 path,
                pinned to the `us.` inference profile in study 2. The
                canonical body bytes are sent as-is; hashed == sent.
- p_aws         Claude Platform on AWS via AnthropicAWS. SigV4; requires an
                AWS region and ANTHROPIC_AWS_WORKSPACE_ID (constructor arg
                or environment). Bare model IDs.
- anthropic_api First-party Messages API via Anthropic. Run-scoped API key
                (never a global export — resolve from the process
                environment of the run). Bare model IDs.
- local         Ollama /api/chat over stdlib urllib (study 3). The harness
                owns every wire byte — hashed == sent by construction. One
                instance per box, named per hardware arm (local_metal /
                local_cuda).

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

# Statuses the runner may retry on the Messages planes: rate limits,
# overload, and server-side failures.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}

# Bedrock retry semantics are CODE-NAME based — exactly study 1's set. Some
# retryable Bedrock codes ride non-retryable HTTP statuses (e.g.
# ServiceQuotaExceededException arrives as a 400), so status-only
# classification would silently change the Bedrock arm's behavior.
BEDROCK_RETRYABLE_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
    "ModelTimeoutException",
    "InternalServerException",
    "ServiceQuotaExceededException",
}


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

    def invoke(self, params, stream=False):
        """One Messages call; params from canonical_messages_params().

        With stream=True the response is delivered over SSE and accumulated
        to the same final message (the SDK stream helper), so the normalized
        record is shape-identical; delivered_streaming records the mode. The
        SDK adds the stream field to the wire body, so the wire hash differs
        from the non-streamed cell's by construction — within-cell identity
        is what the negative control checks, and that still holds.
        """
        import anthropic

        start = time.monotonic()
        try:
            if stream:
                with self.client.messages.stream(**params) as stream_obj:
                    msg = stream_obj.get_final_message()
            else:
                msg = self.client.messages.create(**params)
            latency_ms = int((time.monotonic() - start) * 1000)
            record = normalize_message(
                msg.to_dict(),
                getattr(msg, "_request_id", None),
                latency_ms,
                self._capture.take(),
            )
            record["delivered_streaming"] = stream
            return record
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

    @staticmethod
    def _client_error_record(err, body_bytes):
        error = err.response.get("Error", {})
        meta = err.response.get("ResponseMetadata", {})
        code = error.get("Code", "ClientError")
        record = _error_record(
            code,
            str(err),
            meta.get("HTTPStatusCode"),
            meta.get("RequestId"),
            body_bytes,
        )
        record["retryable"] = record["retryable"] or code in BEDROCK_RETRYABLE_CODES
        return record

    def invoke(self, model_id, body_bytes, stream=False):
        import json

        from botocore.exceptions import (
            ClientError,
            ConnectionClosedError,
            EndpointConnectionError,
            ReadTimeoutError,
        )

        if stream:
            return self._invoke_stream(model_id, body_bytes)
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
            record = normalize_message(payload, request_id, latency_ms, body_bytes)
            record["delivered_streaming"] = False
            return record
        except ClientError as err:
            return self._client_error_record(err, body_bytes)
        except (
            EndpointConnectionError,
            ReadTimeoutError,
            ConnectionClosedError,
        ) as err:
            return _error_record(type(err).__name__, str(err), None, None, body_bytes)

    def _invoke_stream(self, model_id, body_bytes):
        """InvokeModelWithResponseStream, accumulated to the same normalized
        record. The body is identical to the non-streamed call (Bedrock
        selects streaming by endpoint, not by a body field), so hashed ==
        sent holds unchanged. EventStreamError subclasses ClientError, so
        mid-stream service errors land in the same classification."""
        import json

        from botocore.exceptions import (
            ClientError,
            ConnectionClosedError,
            EndpointConnectionError,
            ReadTimeoutError,
        )

        start = time.monotonic()
        try:
            resp = self.client.invoke_model_with_response_stream(
                modelId=model_id,
                body=body_bytes,
                contentType="application/json",
                accept="application/json",
            )
            message = {}
            usage = {}
            text_parts = []
            stop_reason = None
            for event in resp["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                ctype = chunk.get("type")
                if ctype == "message_start":
                    message = chunk.get("message", {})
                    usage = dict(message.get("usage") or {})
                elif ctype == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text_parts.append(delta.get("text", ""))
                elif ctype == "message_delta":
                    if (chunk.get("delta") or {}).get("stop_reason"):
                        stop_reason = chunk["delta"]["stop_reason"]
                    for key, value in (chunk.get("usage") or {}).items():
                        usage[key] = value
            latency_ms = int((time.monotonic() - start) * 1000)
            payload = {
                "id": message.get("id"),
                "model": message.get("model"),
                "stop_reason": stop_reason,
                "usage": usage,
                "content": [{"type": "text", "text": "".join(text_parts)}],
            }
            request_id = resp.get("ResponseMetadata", {}).get("RequestId")
            record = normalize_message(payload, request_id, latency_ms, body_bytes)
            record["delivered_streaming"] = True
            return record
        except ClientError as err:
            return self._client_error_record(err, body_bytes)
        except (
            EndpointConnectionError,
            ReadTimeoutError,
            ConnectionClosedError,
        ) as err:
            return _error_record(type(err).__name__, str(err), None, None, body_bytes)


def normalize_local_message(payload_dict, latency_ms, wire_bytes):
    """Success-record shape for the local plane (Ollama /api/chat), mapped
    onto the shared record surface: done_reason -> stop_reason,
    prompt_eval_count / eval_count -> input / output tokens. Ollama returns
    no response id. Duration fields (ns) ride in usage — eval_duration is
    pure decode time, the cleanest per-token rate for the Q4 hardware
    calibration. Thinking text (think-enabled models) is recorded as a size
    covariate only; the endpoint hashes content text, as everywhere."""
    message = payload_dict.get("message") or {}
    text = message.get("content", "") or ""
    usage = {}
    for src, dst in (
        ("prompt_eval_count", "input_tokens"),
        ("eval_count", "output_tokens"),
        ("total_duration", "total_duration_ns"),
        ("load_duration", "load_duration_ns"),
        ("prompt_eval_duration", "prompt_eval_duration_ns"),
        ("eval_duration", "eval_duration_ns"),
    ):
        if payload_dict.get(src) is not None:
            usage[dst] = payload_dict[src]
    thinking = message.get("thinking")
    if thinking is not None:
        usage["thinking_chars"] = len(thinking)
    return {
        "ok": True,
        "latency_ms": latency_ms,
        "request_id": None,
        "response_id": None,
        "response_model": payload_dict.get("model"),
        "stop_reason": payload_dict.get("done_reason"),
        "usage": usage,
        "text": text,
        "text_sha256": sha256_hex(text.encode("utf-8")),
        "wire_sha256": sha256_hex(wire_bytes) if wire_bytes else None,
    }


class LocalPlane:
    """Study-3 local plane: Ollama HTTP over stdlib urllib — zero new
    dependencies, and the canonical body bytes are sent verbatim (the exact
    negative control). Instantiate one per box and name it per hardware arm
    so records self-identify. No streamed arm is registered for study 3, so
    invoke(stream=True) is a programming error, not a request option.

    engine_version() and model_digest() feed the upgraded drift control:
    engine release and weights digest recorded per run (prereg v3
    section 3)."""

    name = "local"

    def __init__(self, base_url="http://127.0.0.1:11434", name=None,
                 timeout=600.0, opener=None):
        self.base_url = base_url.rstrip("/")
        if name is not None:
            self.name = name
        self._timeout = timeout
        if opener is None:
            import urllib.request

            opener = urllib.request.urlopen
        self._opener = opener

    def _request(self, path, body_bytes=None):
        import urllib.request

        headers = {"Content-Type": "application/json"} if body_bytes else {}
        return urllib.request.Request(
            self.base_url + path,
            data=body_bytes,
            headers=headers,
            method="POST" if body_bytes else "GET",
        )

    def _get_json(self, path):
        import json

        with self._opener(self._request(path), timeout=self._timeout) as resp:
            return json.loads(resp.read())

    def engine_version(self):
        return self._get_json("/api/version").get("version")

    def model_digest(self, model_tag):
        """Weights digest for an exact model tag (drift control; the Q4
        cross-box identity check compares this value between boxes)."""
        for model in self._get_json("/api/tags").get("models", []):
            if model.get("name") == model_tag:
                return model.get("digest")
        raise KeyError(f"model not present: {model_tag}")

    def invoke(self, body_bytes, stream=False):
        import json
        import urllib.error

        if stream:
            raise ValueError(
                "study 3 registers no streamed arm on the local plane"
            )
        start = time.monotonic()
        try:
            with self._opener(
                self._request("/api/chat", body_bytes), timeout=self._timeout
            ) as resp:
                payload = json.loads(resp.read())
            latency_ms = int((time.monotonic() - start) * 1000)
            record = normalize_local_message(payload, latency_ms, body_bytes)
            record["delivered_streaming"] = False
            return record
        except urllib.error.HTTPError as err:
            detail = ""
            try:
                detail = err.read().decode("utf-8", "replace")
            except Exception:
                pass
            return _error_record(
                f"http_{err.code}", detail or str(err), err.code, None, body_bytes
            )
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            return _error_record(
                type(err).__name__, str(err), None, None, body_bytes
            )


def make_plane(plane_name, **kwargs):
    if plane_name == "bedrock":
        return BedrockPlane(**kwargs)
    if plane_name == "p_aws":
        return PAWSPlane(**kwargs)
    if plane_name == "anthropic_api":
        return FirstPartyPlane(**kwargs)
    if plane_name == "local":
        return LocalPlane(**kwargs)
    raise ValueError(f"unknown plane: {plane_name}")
