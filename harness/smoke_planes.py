"""Per-plane instrument smoke for the Messages planes (study 2).

Mirrors harness/smoke.py (the Bedrock smoke) on the two SDK planes: a
handful of live calls verifying access, the exact grid request shapes, and
the API-constraint claims the design rests on. Expected-error cases are as
load-bearing as happy paths: the grid assumes temperature is rejected on the
Claude 5 family and that thinking:disabled is illegal above effort high on
Opus 5 — study 1 verified both on Bedrock; the attribution claim needs them
re-verified per plane, not assumed from parity marketing.

Credential preconditions (prereg v2 section 4 — run-scoped, never global):
  p_aws          AWS credentials + ANTHROPIC_AWS_WORKSPACE_ID (env or
                 --workspace-id). NOTE: first-ever P-AWS call may mint a new
                 AWS Marketplace subscription — the billing-canary path.
  anthropic_api  ANTHROPIC_API_KEY in this process's environment.

Usage:
  ANTHROPIC_AWS_WORKSPACE_ID=... .venv/bin/python -m harness.smoke_planes --plane p_aws
  ANTHROPIC_API_KEY=... .venv/bin/python -m harness.smoke_planes --plane anthropic_api

Writes evidence/smoke-<plane>.json; exits nonzero if any expectation is
violated, 3 if the plane's credentials are not present.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from harness.config import MODELS, plane_model_id
from harness.planes import make_plane
from harness.request_builder import (
    canonical_bytes,
    canonical_messages_params,
    sha256_hex,
)
from harness.tasks import TASKS

CASES = [
    {"name": "opus5-adaptive-extraction", "model": "opus-5",
     "thinking": "adaptive", "task": "extraction", "expect": "ok"},
    {"name": "opus5-disabled-medium", "model": "opus-5",
     "thinking": "disabled", "task": "classification", "expect": "ok"},
    {"name": "sonnet5-adaptive", "model": "sonnet-5",
     "thinking": "adaptive", "task": "classification", "expect": "ok"},
    {"name": "haiku-none", "model": "haiku-4-5",
     "thinking": "none", "task": "classification", "expect": "ok"},
    {"name": "haiku-temp07-control-path", "model": "haiku-4-5",
     "thinking": "none", "task": "classification",
     "extra": {"temperature": 0.7}, "expect": "ok"},
    {"name": "opus5-temp07-must-reject", "model": "opus-5",
     "thinking": "adaptive", "task": "classification",
     "extra": {"temperature": 0.7}, "expect": "validation_error"},
    {"name": "opus5-disabled-xhigh-must-reject", "model": "opus-5",
     "thinking": "disabled", "task": "classification",
     "effort_override": "xhigh", "expect": "validation_error"},
]


def run_case(plane, case):
    mcfg = dict(MODELS[case["model"]])
    if "effort_override" in case:
        mcfg["effort"] = case["effort_override"]
    model_id = plane_model_id(plane.name, case["model"])
    params = canonical_messages_params(
        mcfg, model_id, TASKS[case["task"]]["prompt"], case["thinking"],
        extra=case.get("extra"),
    )
    record = {
        "name": case["name"],
        "plane": plane.name,
        "model_id": model_id,
        "planned_sha256": sha256_hex(canonical_bytes(params)),
        "expect": case["expect"],
        "sent_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    result = plane.invoke(params)
    if result["ok"]:
        record.update({
            "outcome": "ok",
            "latency_ms": result["latency_ms"],
            "request_id": result["request_id"],
            "response_model": result["response_model"],
            "stop_reason": result["stop_reason"],
            "usage": result["usage"],
            "wire_sha256": result["wire_sha256"],
            "text_head": result["text"][:120],
        })
        record["passed"] = case["expect"] == "ok"
    else:
        record.update({
            "outcome": "api_error",
            "error_code": result["error_code"],
            "error_message": result["error_message"],
            "status_code": result["status_code"],
            "request_id": result["request_id"],
            "wire_sha256": result["wire_sha256"],
        })
        record["passed"] = (
            case["expect"] == "validation_error"
            and result["status_code"] == 400
            and result["error_code"] == "invalid_request_error"
        )
    return record


def check_credentials(plane_name, workspace_id):
    if plane_name == "p_aws":
        if not (workspace_id or os.environ.get("ANTHROPIC_AWS_WORKSPACE_ID")):
            print(
                "MISSING CREDENTIAL: Claude Platform on AWS needs "
                "ANTHROPIC_AWS_WORKSPACE_ID (env or --workspace-id).",
                file=sys.stderr,
            )
            return False
    if plane_name == "anthropic_api":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "MISSING CREDENTIAL: the first-party plane needs "
                "ANTHROPIC_API_KEY in this process's environment "
                "(run-scoped — do not export globally).",
                file=sys.stderr,
            )
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Messages-plane smoke")
    parser.add_argument("--plane", required=True, choices=("p_aws", "anthropic_api"))
    parser.add_argument("--workspace-id", default=None)
    args = parser.parse_args()

    if not check_credentials(args.plane, args.workspace_id):
        return 3

    if args.plane == "p_aws":
        plane = make_plane("p_aws", workspace_id=args.workspace_id)
    else:
        plane = make_plane("anthropic_api")

    results = []
    for case in CASES:
        record = run_case(plane, case)
        results.append(record)
        status = "PASS" if record["passed"] else "FAIL"
        detail = record.get("stop_reason") or record.get("error_code")
        print(f"{status}  {record['name']}  ->  {record['outcome']} ({detail})",
              flush=True)
        time.sleep(0.5)

    all_passed = all(r["passed"] for r in results)
    out = {
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "plane": args.plane,
        "all_passed": all_passed,
        "cases": results,
    }
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evidence", f"smoke-{args.plane}.json",
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"all_passed={all_passed}")
    print(f"evidence -> {out_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
