"""Companion-D session-qualification prewarm (FOLLOWUP-COMPANIONS.md).

Repeatedly issues the EXACT frozen measured body of the cache-state A/B
cell and watches `prompt_eval_duration` until the cached-prefill state
engages (N consecutive calls below the threshold that separates the two
observed prefill classes: ~17 ms cached vs ~34-41 ms full prefill). The
prewarm is a GATE, never analysis data — its trajectory and output-sha
sequence are written to evidence/ so any mid-prewarm state transition is
captured, and the A/B that follows stands on its own records.

Exit codes: 0 qualified · 4 not qualified within the cap · 3 unreachable.

Usage:
  python3 -m harness.prewarm_cache --base-url http://127.0.0.1:11435 \
      --label cuda-d1 --out evidence
"""
import argparse
import json
import os
from datetime import datetime, timezone

from harness.config import (
    CACHE_AB,
    LOCAL_KEEP_ALIVE,
    LOCAL_MODELS,
    LOCAL_SAMPLING,
    local_pinned_arm,
)
from harness.planes import LocalPlane
from harness.request_builder import canonical_local_body, sha256_hex
from harness.tasks import TASKS


def run_prewarm(plane, model_key=None, task=None, threshold_ms=25.0,
                consecutive=3, max_calls=40):
    model_key = model_key or CACHE_AB["model"]
    task = task or CACHE_AB["task"]
    cfg = LOCAL_MODELS[model_key]
    body = canonical_local_body(
        cfg["tag"],
        TASKS[task]["prompt"],
        local_pinned_arm(cfg),
        options=LOCAL_SAMPLING[CACHE_AB["sampling"]],
        keep_alive=LOCAL_KEEP_ALIVE,
    )
    trajectory = []
    streak = 0
    qualified = False
    for index in range(max_calls):
        result = plane.invoke(body)
        if not result["ok"]:
            trajectory.append({
                "call": index, "ok": False,
                "error": result.get("error_code"),
            })
            streak = 0
            continue
        usage = result.get("usage") or {}
        prefill_ms = (usage.get("prompt_eval_duration_ns") or 0) / 1e6
        trajectory.append({
            "call": index,
            "ok": True,
            "prefill_ms": round(prefill_ms, 3),
            "load_s": round((usage.get("load_duration_ns") or 0) / 1e9, 3),
            "text_sha256": result["text_sha256"],
        })
        streak = streak + 1 if prefill_ms < threshold_ms else 0
        if streak >= consecutive:
            qualified = True
            break
    return {
        "qualified": qualified,
        "calls": len(trajectory),
        "threshold_ms": threshold_ms,
        "consecutive": consecutive,
        "max_calls": max_calls,
        "request_sha256": sha256_hex(body),
        "trajectory": trajectory,
        "distinct_shas": sorted({
            t["text_sha256"] for t in trajectory if t.get("ok")
        }),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Companion-D session-qualification prewarm"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--label", required=True)
    parser.add_argument("--threshold-ms", type=float, default=25.0)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--max-calls", type=int, default=40)
    parser.add_argument("--out", default="evidence")
    args = parser.parse_args()

    plane = LocalPlane(base_url=args.base_url, name=f"local_{args.label}")
    try:
        version = plane.engine_version()
        digest = plane.model_digest(LOCAL_MODELS[CACHE_AB["model"]]["tag"])
    except Exception as err:
        print(f"UNREACHABLE: {err}")
        return 3

    result = run_prewarm(
        plane,
        threshold_ms=args.threshold_ms,
        consecutive=args.consecutive,
        max_calls=args.max_calls,
    )
    evidence = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "label": args.label,
        "engine_version": version,
        "model_digest": digest,
        **result,
    }
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"prewarm-cache-{args.label}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, sort_keys=True)

    tail = [t.get("prefill_ms") for t in result["trajectory"][-5:]]
    print(
        f"PREWARM qualified={result['qualified']} calls={result['calls']} "
        f"last_prefills_ms={tail} shas={len(result['distinct_shas'])}"
    )
    print(f"evidence -> {out_path}")
    return 0 if result["qualified"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
