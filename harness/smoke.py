"""Instrument smoke test: ~10 live calls that verify account access, request
shapes, and the API-constraint claims the design rests on.

Expected-error cases are as load-bearing as happy paths: the grid assumes
temperature is rejected on the Claude 5 family (relocated positive control)
and that thinking:disabled is illegal above effort high on Opus 5 (pinned
medium effort). If either expectation fails, the design assumptions are
wrong and the prereg must change before any window runs.

Writes evidence/smoke.json; exits nonzero if any expectation is violated.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from harness.config import MODELS, REGION
from harness.request_builder import canonical_body, sha256_hex
from harness.tasks import TASKS

CASES = [
    {"name": "opus5-us-adaptive-extraction", "model": "opus-5", "profile": "us",
     "thinking": "adaptive", "task": "extraction", "expect": "ok"},
    {"name": "opus5-us-disabled-medium", "model": "opus-5", "profile": "us",
     "thinking": "disabled", "task": "classification", "expect": "ok"},
    {"name": "opus5-global-adaptive", "model": "opus-5", "profile": "global",
     "thinking": "adaptive", "task": "classification", "expect": "ok"},
    {"name": "sonnet5-us-adaptive", "model": "sonnet-5", "profile": "us",
     "thinking": "adaptive", "task": "classification", "expect": "ok"},
    {"name": "sonnet5-global-adaptive", "model": "sonnet-5", "profile": "global",
     "thinking": "adaptive", "task": "classification", "expect": "ok"},
    {"name": "haiku-us-none", "model": "haiku-4-5", "profile": "us",
     "thinking": "none", "task": "classification", "expect": "ok"},
    {"name": "haiku-global-none", "model": "haiku-4-5", "profile": "global",
     "thinking": "none", "task": "classification", "expect": "ok"},
    {"name": "haiku-us-temp07-control-path", "model": "haiku-4-5", "profile": "us",
     "thinking": "none", "task": "classification",
     "extra": {"temperature": 0.7}, "expect": "ok"},
    {"name": "opus5-us-temp07-must-reject", "model": "opus-5", "profile": "us",
     "thinking": "adaptive", "task": "classification",
     "extra": {"temperature": 0.7}, "expect": "validation_error"},
    {"name": "opus5-us-disabled-xhigh-must-reject", "model": "opus-5",
     "profile": "us", "thinking": "disabled", "task": "classification",
     "effort_override": "xhigh", "expect": "validation_error"},
]


def run_case(client, case):
    mcfg = dict(MODELS[case["model"]])
    if "effort_override" in case:
        mcfg["effort"] = case["effort_override"]
    body = canonical_body(
        mcfg, TASKS[case["task"]]["prompt"], case["thinking"],
        extra=case.get("extra"),
    )
    record = {
        "name": case["name"],
        "model_id": mcfg["profiles"][case["profile"]],
        "request_sha256": sha256_hex(body),
        "expect": case["expect"],
        "sent_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    start = time.monotonic()
    try:
        resp = client.invoke_model(
            modelId=record["model_id"], body=body,
            contentType="application/json", accept="application/json",
        )
        payload = json.loads(resp["body"].read())
        text = "".join(
            b.get("text", "") for b in payload.get("content", [])
            if b.get("type") == "text"
        )
        record.update({
            "outcome": "ok",
            "latency_ms": int((time.monotonic() - start) * 1000),
            "aws_request_id": resp.get("ResponseMetadata", {}).get("RequestId"),
            "response_model": payload.get("model"),
            "stop_reason": payload.get("stop_reason"),
            "usage": payload.get("usage"),
            "text_head": text[:120],
        })
        record["passed"] = case["expect"] == "ok"
    except ClientError as err:
        code = err.response.get("Error", {}).get("Code", "ClientError")
        record.update({
            "outcome": "client_error",
            "error_code": code,
            "error_message": err.response.get("Error", {}).get("Message", "")[:300],
            "aws_request_id": err.response.get("ResponseMetadata", {}).get(
                "RequestId"
            ),
        })
        record["passed"] = (
            case["expect"] == "validation_error" and code == "ValidationException"
        )
    return record


def main():
    client = boto3.client(
        "bedrock-runtime", region_name=REGION,
        config=Config(read_timeout=600, connect_timeout=10,
                      retries={"max_attempts": 0}),
    )
    results = []
    for case in CASES:
        record = run_case(client, case)
        results.append(record)
        status = "PASS" if record["passed"] else "FAIL"
        detail = record.get("stop_reason") or record.get("error_code")
        print(f"{status}  {record['name']}  ->  {record['outcome']} ({detail})",
              flush=True)

    all_passed = all(r["passed"] for r in results)
    out = {
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "region": REGION,
        "all_passed": all_passed,
        "cases": results,
    }
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evidence", "smoke.json",
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"all_passed={all_passed}")
    print(f"evidence -> {out_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
