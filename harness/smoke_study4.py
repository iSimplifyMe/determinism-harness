"""Study-4 per-door live smoke (PREREGISTRATION-v4 freeze checklist).

Verifies, on every door, that the builders' encoding is ACCEPTED and that
the registered rejections actually reject (temperature; effort `minimal`;
flat `reasoning_effort` on the Responses doors) — the expected-rejection
pattern of the study-2 smokes. Also answers two questions the prereg
records at freeze:

- does mantle accept the bare 1P alias (`gpt-5.6-sol`)? If yes, the two
  Responses doors can run byte-identical bodies including the model field.
- which receipts does codex emit in --json mode (banner on stderr, usage
  event, thread id)?

Writes evidence/smoke-study4.json. Exit 0 only if every REQUIRED
expectation holds. Spend: ~a dozen minimal calls.

Usage: run from the repo root with run-scoped credentials in the process
environment (OPENAI_API_KEY, BEDROCK_API_KEY):
    python3 -m harness.smoke_study4
"""
import json
import sys
from datetime import datetime, timezone

from harness.config import STUDY4_DOORS
from harness.doors import make_door
from harness.request_builder import (
    canonical_bytes,
    canonical_responses_body,
    codex_argv,
    converse_request,
)

SMOKE_MAX_TOKENS = 512
PROBE_PROMPT = "Reply with exactly: OK"


def _responses_result(record):
    if record["ok"]:
        usage = record.get("usage") or {}
        details = usage.get("output_tokens_details") or {}
        return {
            "verdict": "accept",
            "text": record["text"][:40],
            "usage_reasoning_tokens": details.get("reasoning_tokens"),
            "response_model": record.get("response_model"),
            "wire_sha256_present": bool(record.get("wire_sha256")),
            "latency_ms": record["latency_ms"],
        }
    return {
        "verdict": "reject",
        "error_code": record.get("error_code"),
        "error_message": record.get("error_message"),
        "status_code": record.get("status_code"),
    }


def _converse_result(record):
    if record["ok"]:
        return {
            "verdict": "accept",
            "text": record["text"][:40],
            "usage": record.get("usage"),
            "wire_sha256_present": bool(record.get("wire_sha256")),
            "latency_ms": record["latency_ms"],
        }
    return {
        "verdict": "reject",
        "error_code": record.get("error_code"),
        "error_message": record.get("error_message"),
        "status_code": record.get("status_code"),
    }


def smoke_responses_door(door_key):
    """Base accept + three registered rejections. Returns probe list."""
    cfg = STUDY4_DOORS[door_key]
    door = make_door(door_key)
    model = cfg["model_id"]
    probes = []

    body = canonical_responses_body(model, PROBE_PROMPT, "none", SMOKE_MAX_TOKENS)
    result = _responses_result(door.invoke(body))
    result["required"] = "accept"
    result["extra_checks"] = {
        "usage_details_present": result.get("usage_reasoning_tokens") is not None,
        "wire_hash": result.get("wire_sha256_present", False),
    }
    probes.append({"probe": "base_effort_none", **result})

    for name, mutation in (
        ("reject_temperature", {"temperature": 0}),
        ("reject_effort_minimal", {"reasoning": {"effort": "minimal"}}),
        ("reject_flat_reasoning_effort", {"reasoning_effort": "low"}),
    ):
        raw = json.loads(canonical_responses_body(model, PROBE_PROMPT, "none", SMOKE_MAX_TOKENS))
        raw.pop("reasoning", None)
        raw.update(mutation)
        result = _responses_result(door.invoke(canonical_bytes(raw)))
        result["required"] = "reject"
        probes.append({"probe": name, **result})
    return probes


def smoke_mantle_bare_alias():
    """Informational: does mantle serve the bare 1P alias? Decides whether
    the 1P/mantle byte-parity claim extends to the model field."""
    door = make_door("mantle")
    body = canonical_responses_body(
        "gpt-5.6-sol", PROBE_PROMPT, "none", SMOKE_MAX_TOKENS
    )
    result = _responses_result(door.invoke(body))
    result["required"] = "informational"
    return [{"probe": "mantle_bare_alias", **result}]


def smoke_converse_door(door_key):
    cfg = STUDY4_DOORS[door_key]
    door = make_door(door_key)
    model = cfg["model_id"]
    probes = []

    record = door.invoke(
        converse_request(model, PROBE_PROMPT, "none", SMOKE_MAX_TOKENS)
    )
    result = _converse_result(record)
    result["required"] = "accept"
    # The before-send hook is load-bearing: a silent event-name mismatch
    # would void the wire negative control, so its firing is REQUIRED.
    result["extra_checks"] = {"wire_hash": result.get("wire_sha256_present", False)}
    probes.append({"probe": "base_effort_none", **result})

    reject_kwargs = converse_request(model, PROBE_PROMPT, "default", SMOKE_MAX_TOKENS)
    reject_kwargs["inferenceConfig"]["temperature"] = 0.0
    result = _converse_result(door.invoke(reject_kwargs))
    result["required"] = "reject"
    probes.append({"probe": "reject_temperature", **result})

    reject_kwargs = converse_request(model, PROBE_PROMPT, "default", SMOKE_MAX_TOKENS)
    reject_kwargs["additionalModelRequestFields"] = {
        "reasoning": {"effort": "minimal"}
    }
    result = _converse_result(door.invoke(reject_kwargs))
    result["required"] = "reject"
    probes.append({"probe": "reject_effort_minimal", **result})
    return probes


def smoke_codex_door():
    door = make_door("codex_sub")
    probes = []
    model = STUDY4_DOORS["codex_sub"]["model_id"]
    for arm in ("none", "high"):
        receipts = door.receipt(model, arm)
        matches = receipts.get("reasoning_effort") == arm
        probes.append({
            "probe": f"receipt_effort_{arm}",
            "required": "accept",
            "verdict": "accept" if matches else "reject",
            "receipts": receipts,
        })
    for arm in ("none", "high"):
        argv = codex_argv(
            STUDY4_DOORS["codex_sub"]["model_id"], PROBE_PROMPT, arm, door.workdir
        )
        record = door.invoke(argv)
        if record["ok"]:
            usage = record.get("usage") or {}
            probes.append({
                "probe": f"base_effort_{arm}",
                "required": "accept",
                "verdict": "accept",
                "text": record["text"][:40],
                "usage_present": bool(usage),
                "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                "receipts": record.get("receipts") or {},
                "banner_available_in_json_mode": bool(record.get("receipts")),
                "thread_id_present": bool(record.get("response_id")),
                "latency_ms": record["latency_ms"],
            })
        else:
            probes.append({
                "probe": f"base_effort_{arm}",
                "required": "accept",
                "verdict": "reject",
                "error_code": record.get("error_code"),
                "error_message": record.get("error_message"),
            })
    return probes


def main():
    report = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "doors": {},
    }
    failures = []
    plan = [
        ("openai_1p", lambda: smoke_responses_door("openai_1p")),
        ("mantle", lambda: smoke_responses_door("mantle")),
        ("mantle_bare_alias", smoke_mantle_bare_alias),
        ("runtime_us", lambda: smoke_converse_door("runtime_us")),
        ("runtime_global", lambda: smoke_converse_door("runtime_global")),
        ("codex_sub", smoke_codex_door),
    ]
    for label, fn in plan:
        try:
            probes = fn()
        except Exception as err:  # credential/setup failures land here
            probes = [{
                "probe": "setup", "required": "accept",
                "verdict": "error", "error_message": str(err)[:300],
            }]
        report["doors"][label] = probes
        for probe in probes:
            required = probe.get("required")
            verdict = probe.get("verdict")
            ok = (
                required == "informational"
                or verdict == required
            )
            if ok and probe.get("extra_checks"):
                ok = all(probe["extra_checks"].values())
            status = "PASS" if ok else "FAIL"
            if not ok:
                failures.append(f"{label}/{probe['probe']}")
            print(f"{status} {label:>18} {probe['probe']:<28} {verdict}")

    report["all_passed"] = not failures
    with open("evidence/smoke-study4.json", "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"all_passed={not failures}")
    if failures:
        print("failures:", ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
