"""Logprobs exposure probe (prereg v3 freeze gate).

The first-divergence / logprob-margin endpoints are registered ONLY IF the
pinned engine exposes per-token logprobs (PREREGISTRATION-v3 section 3);
otherwise the fallback (llama.cpp) changes the engine and the decision must
be made once, before freeze. This probe answers the question empirically on
the pinned engine: it sends candidate field spellings and reports which (if
any) the server accepts AND which yield logprob content in the response.

Run it against a RESIDENT model on the production box with
--keep-alive 24h (keep_alive is per-request and overwrites expiry).

Usage:
  python3 -m harness.probe_logprobs --base-url http://127.0.0.1:11436 \
      --model qwen3.6:35b-a3b-q8_0 --label metal --keep-alive 24h

Writes evidence/logprobs-probe-<label>.json; exit 0 always unless the
server is unreachable (3) — a negative result is a valid answer.
"""
import argparse
import json
import os
from datetime import datetime, timezone

from harness.planes import LocalPlane
from harness.request_builder import canonical_local_body

PROBE_OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 8}
PROMPT = "Reply with exactly: PROBE-OK"

VARIANTS = (
    {"name": "top-level-logprobs", "extra": {"logprobs": True}},
    {"name": "top-level-with-top-logprobs",
     "extra": {"logprobs": True, "top_logprobs": 3}},
    {"name": "options-logprobs", "extra": None,
     "options_extra": {"logprobs": 3}},
)


def find_logprob_keys(payload):
    """Where (if anywhere) logprob-like content appears in a response."""
    hits = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if "logprob" in key.lower():
                    hits.append(f"{path}.{key}" if path else key)
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for i, value in enumerate(node[:3]):
                walk(value, f"{path}[{i}]")

    walk(payload, "")
    return sorted(set(hits))


def run_variant(plane, model_tag, variant, keep_alive):
    options = dict(PROBE_OPTIONS)
    if variant.get("options_extra"):
        options.update(variant["options_extra"])
    body = canonical_local_body(
        model_tag, PROMPT, "none", options=options,
        keep_alive=keep_alive, extra=variant.get("extra"),
    )
    result = plane.invoke(body)
    entry = {
        "name": variant["name"],
        "accepted": bool(result["ok"]),
        "sent_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if result["ok"]:
        raw = plane.last_payload or {}
        entry["logprob_keys"] = find_logprob_keys(raw)
        entry["exposes_logprobs"] = bool(entry["logprob_keys"])
    else:
        entry["status_code"] = result.get("status_code")
        entry["error_message"] = (result.get("error_message") or "")[:200]
        entry["exposes_logprobs"] = False
    return entry


def main():
    parser = argparse.ArgumentParser(description="Ollama logprobs probe")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--keep-alive", required=True,
                        help="24h on the production box (resident models)")
    parser.add_argument("--out", default="evidence")
    args = parser.parse_args()

    plane = LocalPlane(base_url=args.base_url, name=f"local_{args.label}")
    try:
        version = plane.engine_version()
        digest = plane.model_digest(args.model)
    except Exception as err:
        print(f"UNREACHABLE: {err}")
        return 3

    results = [
        run_variant(plane, args.model, variant, args.keep_alive)
        for variant in VARIANTS
    ]
    exposed = [r["name"] for r in results if r.get("exposes_logprobs")]
    evidence = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "label": args.label,
        "engine_version": version,
        "model": args.model,
        "model_digest": digest,
        "keep_alive": args.keep_alive,
        "variants": results,
        "verdict": {
            "exposes_logprobs": bool(exposed),
            "working_variants": exposed,
        },
    }
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"logprobs-probe-{args.label}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, sort_keys=True)
    for r in results:
        print(
            f'{r["name"]}: accepted={r["accepted"]} '
            f'exposes={r.get("exposes_logprobs")} '
            f'keys={r.get("logprob_keys", [])}'
        )
    print(f"VERDICT exposes_logprobs={bool(exposed)} via {exposed}")
    print(f"evidence -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
