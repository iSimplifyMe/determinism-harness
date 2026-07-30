"""Companion-E analysis: timing-manipulated cache-state A/B.

Exploratory-directional; plan and analyzer committed before the data.
Deterministic code, stdlib only.

Gates:
- per-arm validity via analyze_study3.gate_cell3;
- cross-arm negative control: one request sha across both arms (the
  manipulation is the pre-call gap, never the measured bytes);
- manipulation gate on absolute prefill classes (three-session evidence:
  full-KV ~16-18 ms vs checkpoint ~34-41 ms): every adjacent-arm call
  below ADJACENT_MAX_MS AND every gapped-arm call above GAPPED_MIN_MS.

Registered endpoints (plan, companion E):
1. within-arm modal shares (prediction: 1.0 both arms);
2. arms_differ (prediction: true).

History matching stays descriptive, against the same on-record variants
as companion C/D (analyze_cache_ab.DEFAULT_HISTORY).

Usage:
  python3 -m analysis.analyze_cache_timing runs/local-study3-cache-timing-*.jsonl
"""
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from analysis.analyze_cache_ab import DEFAULT_HISTORY
from analysis.analyze_study3 import gate_cell3
from analysis.latency_fingerprint import percentile
from analysis.metrics import cell_metrics

TIMING_CELL = "gpt-oss-20b|open_generation|greedy|effort_low"
ARMS = ("adjacent", "gapped")
ADJACENT_MAX_MS = 25.0
GAPPED_MIN_MS = 30.0


def _median(values):
    values = [v for v in values if v is not None]
    return percentile(values, 50) if values else None


def build_timing_report(records, history=None):
    warmups = 0
    by_box_arm = defaultdict(list)
    for record in records:
        if record.get("meta_control") in ("warmup", "burnin"):
            warmups += 1
            continue
        arm = record.get("meta_arm")
        if arm not in ARMS:
            continue
        box = record.get("box") or record.get("meta_hardware")
        by_box_arm[(box, arm)].append(record)

    boxes = {}
    for box in sorted({b for b, _ in by_box_arm}):
        arms_out = {}
        shas = set()
        prefill = {}
        for arm in ARMS:
            recs = by_box_arm.get((box, arm), [])
            gate = gate_cell3(recs)
            valid = gate["valid"]
            shas.update(
                r.get("request_sha256") for r in recs if r.get("request_sha256")
            )
            prefill[arm] = [
                ((r.get("usage") or {}).get("prompt_eval_duration_ns") or 0)
                / 1e6
                for r in valid
            ]
            entry = {
                "n_raw": gate["n_raw"],
                "excluded": gate["excluded"],
                "flags": gate["flags"],
                "prefill_ms": {
                    "median": _median(prefill[arm]),
                    "min": min(prefill[arm]) if prefill[arm] else None,
                    "max": max(prefill[arm]) if prefill[arm] else None,
                },
            }
            if valid:
                metrics = cell_metrics([
                    {
                        "text": r["text"],
                        "output_tokens": (r.get("usage") or {}).get(
                            "output_tokens"
                        ),
                    }
                    for r in valid
                ])
                entry["metrics"] = metrics
                if history:
                    entry["history_matches"] = sorted(
                        label for label, sha in history.items()
                        if sha == metrics["modal_sha256"]
                    )
            arms_out[arm] = entry

        adjacent_values = prefill.get("adjacent", [])
        gapped_values = prefill.get("gapped", [])
        adjacent_all_below = (
            bool(adjacent_values)
            and all(v < ADJACENT_MAX_MS for v in adjacent_values)
        )
        gapped_all_above = (
            bool(gapped_values)
            and all(v > GAPPED_MIN_MS for v in gapped_values)
        )
        manipulation = {
            "adjacent_all_below": adjacent_all_below,
            "gapped_all_above": gapped_all_above,
            "adjacent_median_ms": _median(adjacent_values),
            "gapped_median_ms": _median(gapped_values),
            "adjacent_max_ms_threshold": ADJACENT_MAX_MS,
            "gapped_min_ms_threshold": GAPPED_MIN_MS,
            "pass": adjacent_all_below and gapped_all_above,
        }
        cross_arm = len(shas) == 1
        arms_differ = None
        if (
            manipulation["pass"]
            and cross_arm
            and "metrics" in arms_out["adjacent"]
            and "metrics" in arms_out["gapped"]
        ):
            arms_differ = (
                arms_out["adjacent"]["metrics"]["modal_sha256"]
                != arms_out["gapped"]["metrics"]["modal_sha256"]
            )
        boxes[box] = {
            "arms": arms_out,
            "gates": {
                "cross_arm_negative_control": cross_arm,
                "manipulation": manipulation,
            },
            "arms_differ": arms_differ,
        }

    return {
        "exploratory": True,
        "companion_plan": "FOLLOWUP-COMPANIONS.md (companion E)",
        "cell": TIMING_CELL,
        "endpoints": (
            "1: within-arm modal shares (predicted 1.0 both); "
            "2: arms_differ (predicted true); history matching descriptive"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "records_in": len(records),
            "warmups_excluded": warmups,
        },
        "boxes": boxes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Companion-E timing A/B analysis (exploratory)"
    )
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    records = []
    for path in args.paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if not records:
        print("ERROR: no records")
        return 2

    report = build_timing_report(records, history=DEFAULT_HISTORY)
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.out, f"cache-timing-report-{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    for box in sorted(report["boxes"]):
        entry = report["boxes"][box]
        manipulation = entry["gates"]["manipulation"]
        adjacent = entry["arms"]["adjacent"]
        gapped = entry["arms"]["gapped"]
        print(
            f"{box}: gate pass={manipulation['pass']} "
            f"(adj med {manipulation['adjacent_median_ms']}ms, gap med "
            f"{manipulation['gapped_median_ms']}ms); adjacent modal "
            f"{adjacent.get('metrics', {}).get('modal_share')} "
            f"[{', '.join(adjacent.get('history_matches', []) or ['-'])}], "
            f"gapped modal "
            f"{gapped.get('metrics', {}).get('modal_share')} "
            f"[{', '.join(gapped.get('history_matches', []) or ['-'])}]; "
            f"arms_differ={entry['arms_differ']}"
        )
    print(f"report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
