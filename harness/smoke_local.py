"""Local-plane instrument smoke (study 3).

Mirrors harness/smoke_planes.py on the Ollama plane: a handful of live calls
verifying the exact request shapes the grid will send, per model family.
Field acceptance is load-bearing: the design encodes Qwen hybrid thinking as
`think: true/false` and gpt-oss reasoning effort as `think: "low"/"high"` —
verified here on the pinned engine, not assumed from docs. A same-bytes
repeat pair is recorded informationally (byte match is study data, not an
instrument expectation, so it never gates the smoke).

Expectations: "ok" and "error" gate the exit code; "record" cases capture
the outcome without gating (used where acceptance itself is the question).

Usage (server started for the window, e.g. over an ssh-held session):
  python3 -m harness.smoke_local --base-url http://127.0.0.1:11435 \
      --model gpt-oss:20b --family gpt-oss --label cuda-4090

Writes evidence/smoke-local-<label>.json; exits 1 if any expectation is
violated, 3 if the server is unreachable.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

from harness.planes import LocalPlane
from harness.request_builder import canonical_local_body, sha256_hex

GREEDY_OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 64}
PROMPT = "Reply with exactly: LOCAL-PLANE-OK"

# keep_alive is PER-REQUEST and OVERWRITES the model's current expiry. On
# the production box the study models are resident with 24h keep-alives —
# smoking them with a short keep_alive would schedule their unload. Pass
# --keep-alive 24h there; the short default is for dedicated boxes.
DEFAULT_KEEP_ALIVE = "2m"

FAMILY_CASES = {
    "gpt-oss": [
        {"name": "greedy-none", "arm": "none", "expect": "ok"},
        {"name": "greedy-repeat-same-bytes", "arm": "none", "expect": "ok",
         "compare_to": "greedy-none"},
        {"name": "effort-low", "arm": "effort_low", "expect": "ok"},
        {"name": "effort-high", "arm": "effort_high", "expect": "ok"},
        {"name": "think-bool-acceptance-probe", "arm": "think_on",
         "expect": "record"},
    ],
    "qwen": [
        {"name": "greedy-none", "arm": "none", "expect": "ok"},
        {"name": "greedy-repeat-same-bytes", "arm": "none", "expect": "ok",
         "compare_to": "greedy-none"},
        {"name": "think-on", "arm": "think_on", "expect": "ok"},
        {"name": "think-off", "arm": "think_off", "expect": "ok"},
        {"name": "effort-level-acceptance-probe", "arm": "effort_low",
         "expect": "record"},
    ],
}

MISSING_MODEL_CASE = {
    "name": "missing-model-must-404",
    "model": "definitely-absent:1b",
    "arm": "none",
    "expect": "error",
}


def run_case(plane, model_tag, case, seen_text_hashes, keep_alive):
    body = canonical_local_body(
        case.get("model", model_tag), PROMPT, case["arm"],
        options=GREEDY_OPTIONS, keep_alive=keep_alive,
    )
    record = {
        "name": case["name"],
        "plane": plane.name,
        "model": case.get("model", model_tag),
        "arm": case["arm"],
        "planned_sha256": sha256_hex(body),
        "expect": case["expect"],
        "sent_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    result = plane.invoke(body)
    if result["ok"]:
        record.update({
            "ok": True,
            "latency_ms": result["latency_ms"],
            "stop_reason": result["stop_reason"],
            "usage": result["usage"],
            "text_sha256": result["text_sha256"],
            "wire_sha256": result["wire_sha256"],
            "wire_matches_planned": result["wire_sha256"] == record["planned_sha256"],
        })
        seen_text_hashes[case["name"]] = result["text_sha256"]
        if case.get("compare_to"):
            other = seen_text_hashes.get(case["compare_to"])
            record["informational_byte_match_vs"] = case["compare_to"]
            record["informational_byte_match"] = (
                other is not None and other == result["text_sha256"]
            )
    else:
        record.update({
            "ok": False,
            "error_code": result.get("error_code"),
            "status_code": result.get("status_code"),
            "retryable": result.get("retryable"),
            "error_message": result.get("error_message"),
        })
    if case["expect"] == "ok":
        record["pass"] = bool(record["ok"]) and record["wire_matches_planned"]
    elif case["expect"] == "error":
        record["pass"] = not record["ok"]
    else:  # record
        record["pass"] = True
    return record


def main():
    parser = argparse.ArgumentParser(description="Local-plane instrument smoke")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--family", choices=sorted(FAMILY_CASES), default="gpt-oss")
    parser.add_argument("--label", required=True,
                        help="box label for the evidence file, e.g. cuda-4090")
    parser.add_argument("--keep-alive", default=DEFAULT_KEEP_ALIVE,
                        help="per-request keep_alive; MUST be 24h when "
                             "smoking resident models on the production box")
    parser.add_argument("--out", default="evidence")
    args = parser.parse_args()

    plane = LocalPlane(base_url=args.base_url, name=f"local_{args.label}")
    try:
        version = plane.engine_version()
        digest = plane.model_digest(args.model)
    except Exception as err:
        print(f"UNREACHABLE: {err}")
        return 3

    box_state_start = plane.box_state()
    seen_text_hashes = {}
    cases = FAMILY_CASES[args.family] + [MISSING_MODEL_CASE]
    results = [
        run_case(plane, args.model, case, seen_text_hashes, args.keep_alive)
        for case in cases
    ]

    evidence = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "label": args.label,
        "engine_version": version,
        "model": args.model,
        "model_digest": digest,
        "greedy_options": GREEDY_OPTIONS,
        "keep_alive": args.keep_alive,
        "box_state_start": box_state_start,
        "cases": results,
        "box_state_end": plane.box_state(),
    }
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"smoke-local-{args.label}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, sort_keys=True)

    passed = sum(1 for r in results if r["pass"])
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        extra = ""
        if "informational_byte_match" in r:
            extra = f' byte_match={r["informational_byte_match"]}'
        print(f'{status} {r["name"]} expect={r["expect"]} ok={r["ok"]}{extra}')
    print(f"{passed}/{len(results)} PASS  engine={version} digest={digest}")
    print(f"evidence -> {out_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
