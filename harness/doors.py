"""Study-4 door clients (PREREGISTRATION-v4): one OpenAI model, five doors.

Every door presents the study-2 normalized record surface (harness.planes),
so the analysis pipeline reads all doors uniformly:

- openai_1p       api.openai.com Responses API. stdlib urllib, Bearer key
                  from the run's process environment — the harness owns the
                  wire bytes, hashed == sent by construction.
- mantle          Bedrock's OpenAI-compatible Responses door. Identical
                  client path to openai_1p (same builder, same transport);
                  only base URL, credential, and model alias differ.
- runtime_us /    Bedrock Converse via boto3. boto3 owns wire serialization,
  runtime_global  so a botocore before-send hook captures the exact bytes
                  sent (the SDK-plane pattern of study 2); the canonical
                  kwargs hash is the planned-request hash.
- codex_sub       HARNESS DOOR: codex exec under a ChatGPT subscription.
                  The harness cannot see the bytes codex sends — no wire
                  control, disclosed in v4 sections 1-3. Receipts are the
                  JSONL usage record plus any banner lines codex emits.

Retry policy mirrors studies 1-2: SDK/boto3 retry layers disabled; every
attempt is the caller's to count. This module is a runner-only dependency;
the analysis pipeline stays stdlib.
"""
import json
import re
import subprocess
import threading
import time

from harness.config import REGION, STUDY4_DOORS
from harness.planes import (
    BEDROCK_RETRYABLE_CODES,
    _error_record,
    retryable_status,
)
from harness.request_builder import sha256_hex

# stderr substrings that mark a codex failure as retryable (subscription
# rate windows, transient upstream errors). Case-insensitive.
CODEX_RETRYABLE_PATTERN = re.compile(
    r"rate.?limit|usage.?limit|too many|429|5\d\d|server had an error|"
    r"try again|temporarily",
    re.IGNORECASE,
)

_BANNER_KEYS = ("model", "provider", "reasoning effort", "sandbox", "approval")


def normalize_responses(payload, request_id, latency_ms, wire_bytes):
    """Success-record shape for the Responses doors (1P, mantle). Text is
    the concatenation of output_text parts across message items; reasoning
    items carry no text on these doors (encrypted/summary-less) and are
    counted only through usage.output_tokens_details."""
    text = "".join(
        part.get("text", "")
        for item in payload.get("output", []) or []
        if item.get("type") == "message"
        for part in item.get("content", []) or []
        if part.get("type") == "output_text"
    )
    status = payload.get("status")
    incomplete = payload.get("incomplete_details") or {}
    stop_reason = status
    if status == "incomplete" and incomplete.get("reason"):
        stop_reason = f"incomplete:{incomplete['reason']}"
    return {
        "ok": True,
        "latency_ms": latency_ms,
        "request_id": request_id,
        "response_id": payload.get("id"),
        "response_model": payload.get("model"),
        "stop_reason": stop_reason,
        "usage": payload.get("usage"),
        "text": text,
        "text_sha256": sha256_hex(text.encode("utf-8")),
        "wire_sha256": sha256_hex(wire_bytes) if wire_bytes else None,
    }


class ResponsesDoor:
    """1P and mantle: one client class, two instantiations. The API key
    resolves from the named environment variable of the run's process at
    construction (run-scoped; never a repo file, never an argument that
    could land in a manifest)."""

    def __init__(self, name, base_url, api_key_env, timeout=600.0,
                 opener=None):
        import os

        self.name = name
        self.base_url = base_url.rstrip("/")
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(
                f"{name}: {api_key_env} is not set in the run environment"
            )
        self._auth = f"Bearer {key}"
        self._timeout = timeout
        if opener is None:
            import urllib.request

            opener = urllib.request.urlopen
        self._opener = opener

    def _request(self, body_bytes):
        import urllib.request

        return urllib.request.Request(
            self.base_url + "/responses",
            data=body_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": self._auth,
            },
            method="POST",
        )

    def invoke(self, body_bytes):
        import urllib.error

        start = time.monotonic()
        try:
            with self._opener(
                self._request(body_bytes), timeout=self._timeout
            ) as resp:
                payload = json.loads(resp.read())
                request_id = resp.headers.get("x-request-id")
            latency_ms = int((time.monotonic() - start) * 1000)
            return normalize_responses(
                payload, request_id, latency_ms, body_bytes
            )
        except urllib.error.HTTPError as err:
            detail, code = "", None
            try:
                error_obj = json.loads(err.read()).get("error") or {}
                detail = error_obj.get("message", "")
                code = error_obj.get("code") or error_obj.get("type")
            except Exception:
                pass
            record = _error_record(
                code or f"http_{err.code}",
                detail or str(err),
                err.code,
                err.headers.get("x-request-id") if err.headers else None,
                body_bytes,
            )
            record["retryable"] = retryable_status(err.code)
            return record
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            return _error_record(
                type(err).__name__, str(err), None, None, body_bytes
            )


class _ConverseWireCapture:
    """botocore before-send hook: records the exact serialized bytes of the
    Converse request. Thread-local like the study-2 httpx hook; take() is
    read immediately after the call returns in the same thread."""

    def __init__(self):
        self._local = threading.local()

    def __call__(self, request, **kwargs):
        try:
            body = request.body
            if isinstance(body, str):
                body = body.encode("utf-8")
            self._local.content = bytes(body) if body else None
        except Exception:
            self._local.content = None

    def take(self):
        content = getattr(self._local, "content", None)
        self._local.content = None
        return content


class ConverseDoor:
    """runtime_us / runtime_global. Converse takes structured params (no
    raw-bytes path like study 1's InvokeModel), so wire truth comes from
    the before-send capture; smoke fails if the hook never fires."""

    def __init__(self, name, model_id, region=REGION):
        import boto3
        from botocore.config import Config

        self.name = name
        self.model_id = model_id
        self._capture = _ConverseWireCapture()
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                read_timeout=600, connect_timeout=10,
                retries={"max_attempts": 0},
            ),
        )
        self.client.meta.events.register(
            "before-send.bedrock-runtime.Converse", self._capture
        )

    def invoke(self, converse_kwargs):
        from botocore.exceptions import (
            ClientError,
            ConnectionClosedError,
            EndpointConnectionError,
            ReadTimeoutError,
        )

        start = time.monotonic()
        try:
            resp = self.client.converse(**converse_kwargs)
            latency_ms = int((time.monotonic() - start) * 1000)
            wire = self._capture.take()
            message = (resp.get("output") or {}).get("message") or {}
            text = "".join(
                block["text"]
                for block in message.get("content", []) or []
                if "text" in block
            )
            return {
                "ok": True,
                "latency_ms": latency_ms,
                "request_id": resp.get("ResponseMetadata", {}).get("RequestId"),
                "response_id": None,   # Converse returns no response id
                "response_model": None,  # ...and no served-model echo
                "stop_reason": resp.get("stopReason"),
                "usage": resp.get("usage"),
                "text": text,
                "text_sha256": sha256_hex(text.encode("utf-8")),
                "wire_sha256": sha256_hex(wire) if wire else None,
            }
        except ClientError as err:
            wire = self._capture.take()
            error = err.response.get("Error", {})
            meta = err.response.get("ResponseMetadata", {})
            code = error.get("Code", "ClientError")
            record = _error_record(
                code, str(err), meta.get("HTTPStatusCode"),
                meta.get("RequestId"), wire,
            )
            record["retryable"] = (
                record["retryable"] or code in BEDROCK_RETRYABLE_CODES
            )
            return record
        except (
            EndpointConnectionError,
            ReadTimeoutError,
            ConnectionClosedError,
        ) as err:
            return _error_record(
                type(err).__name__, str(err), None, None,
                self._capture.take(),
            )


class CodexDoor:
    """codex_sub, the harness door. One subprocess per call, --ephemeral,
    stdin closed (a non-TTY exec otherwise blocks on an additional-input
    read — discovery-receipted). No wire control exists on this door by
    construction; receipts are the JSONL usage event, the thread id, and
    any banner lines codex prints to stderr in --json mode (availability
    recorded at smoke)."""

    def __init__(self, name="codex_sub", workdir=None, timeout=600.0):
        import os

        self.name = name
        self.workdir = workdir or os.path.expanduser("~/.cache/gpts")
        os.makedirs(self.workdir, exist_ok=True)
        self._timeout = timeout

    @staticmethod
    def _parse_stdout(stdout_text):
        text, usage, thread_id = None, None, None
        for line in stdout_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                event.get("type") == "item.completed"
                and (event.get("item") or {}).get("type") == "agent_message"
            ):
                text = event["item"].get("text")
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            if event.get("thread_id"):
                thread_id = event["thread_id"]
        return text, usage, thread_id

    @staticmethod
    def _parse_banner(stderr_text):
        receipts = {}
        for line in stderr_text.splitlines():
            for key in _BANNER_KEYS:
                prefix = key + ": "
                if line.startswith(prefix):
                    receipts[key.replace(" ", "_")] = line[len(prefix):].strip()
        return receipts

    def receipt(self, model_id, effort_arm):
        """Per-batch effort-pin receipt (v4 section 3): a plain-mode probe
        whose stderr banner must state the pinned model and effort. Returns
        the parsed banner dict; the caller asserts
        receipts['reasoning_effort'] == effort_arm before the batch runs."""
        from harness.request_builder import codex_receipt_argv

        argv = codex_receipt_argv(
            model_id, "Reply with exactly: OK", effort_arm, self.workdir
        )
        try:
            proc = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                cwd=self.workdir,
                timeout=self._timeout,
            )
        except (subprocess.TimeoutExpired, OSError) as err:
            return {"error": str(err)[:200]}
        receipts = self._parse_banner(proc.stderr)
        receipts["exit_code"] = proc.returncode
        return receipts

    def invoke(self, argv):
        start = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                cwd=self.workdir,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as err:
            record = _error_record(
                "CodexTimeout", str(err), None, None, None
            )
            record["retryable"] = True
            return record
        except OSError as err:
            return _error_record(type(err).__name__, str(err), None, None, None)
        latency_ms = int((time.monotonic() - start) * 1000)
        text, usage, thread_id = self._parse_stdout(proc.stdout)
        receipts = self._parse_banner(proc.stderr)
        if proc.returncode != 0 or text is None:
            stderr_tail = proc.stderr[-400:]
            record = _error_record(
                f"codex_exit_{proc.returncode}",
                stderr_tail or "no agent_message in stdout",
                None, thread_id, None,
            )
            record["retryable"] = bool(
                CODEX_RETRYABLE_PATTERN.search(proc.stderr or "")
            )
            record["receipts"] = receipts
            return record
        return {
            "ok": True,
            "latency_ms": latency_ms,
            "request_id": None,
            "response_id": thread_id,
            "response_model": receipts.get("model"),
            "stop_reason": "agent_message",
            "usage": usage,
            "text": text,
            "text_sha256": sha256_hex(text.encode("utf-8")),
            "wire_sha256": None,  # harness door: no wire control (v4 s3)
            "receipts": receipts,
        }


def make_door(door_key, **overrides):
    cfg = STUDY4_DOORS[door_key]
    kind = cfg["kind"]
    if kind == "responses":
        return ResponsesDoor(
            door_key, cfg["base_url"], cfg["api_key_env"], **overrides
        )
    if kind == "converse":
        return ConverseDoor(door_key, cfg["model_id"], **overrides)
    if kind == "codex":
        return CodexDoor(door_key, **overrides)
    raise ValueError(f"unknown door kind: {kind}")
