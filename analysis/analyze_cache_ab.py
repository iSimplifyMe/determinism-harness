"""Companion-C analysis: cache-state A/B (FOLLOWUP-COMPANIONS.md).

Exploratory-directional; plan and analyzer committed before the data.
Deterministic code, stdlib only.

Gates:
- per-arm validity via analyze_study3.gate_cell3;
- cross-arm negative control: one request sha across BOTH measured arms
  (the manipulation lives in the preceding call, not the measured bytes);
- manipulation gate on the recorded prefill discriminator: every
  cold-arm call's prompt_eval_duration_ns > COLD_FACTOR x the warm arm's
  median, and the warm median < cold median / COLD_FACTOR.

Registered endpoints (plan, companion C):
1. within-arm modal shares (prediction: 1.0 both arms);
2. arms_differ — do the two arms' modal outputs differ? (prediction: yes)

History matching is DESCRIPTIVE only (cross-session drift is on record):
each arm's modal sha is reported against supplied labeled historical shas.

Usage:
  python3 -m analysis.analyze_cache_ab runs/local-study3-cache-ab-*.jsonl
"""
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from analysis.analyze_study3 import gate_cell3
from analysis.latency_fingerprint import percentile
from analysis.metrics import cell_metrics

CACHE_CELL = "gpt-oss-20b|open_generation|greedy|effort_low"
ARMS = ("cold", "warm")
COLD_FACTOR = 3.0

# The four CUDA variants already on public record for this cell — labels
# for the descriptive matching (FOLLOWUP-COMPANIONS.md, companion C).
DEFAULT_HISTORY = {
    "confirmatory_modal_prev_session":
        "13bae41c4c90fdc394fcd28058fa1c4f261f1c77aaaf6674a04faf145ff7f5d2",
    "companionA_B1_warmup_era":
        "20310cddf7fa12c5003d402fa35406f43067ba078e13a10cb8ce2aac8d79500e",
    "companionA_cached_steady":
        "cf2c66c89f3c8f0bc07f707419e5b0823fdfa1cd45b6bd165c1f05081bb7679a",
    "companionA_fresh_load":
        "45e27daf2ae249dec5aab86a52af873bd43638db96ac7712bc2f4fd838448af8",
}


def _median(values):
    values = [v for v in values if v is not None]
    return percentile(values, 50) if values else None


def build_cache_report(records, history=None):
    warmups = 0
    flushers = 0
    by_box_arm = defaultdict(list)
    for record in records:
        control = record.get("meta_control")
        if control == "warmup":
            warmups += 1
            continue
        if control == "flusher":
            flushers += 1
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

        warm_median = _median(prefill.get("warm", []))
        cold_values = prefill.get("cold", [])
        cold_median = _median(cold_values)
        threshold = (
            warm_median * COLD_FACTOR if warm_median is not None else None
        )
        cold_all_above = (
            bool(cold_values)
            and threshold is not None
            and all(v > threshold for v in cold_values)
        )
        warm_below = (
            warm_median is not None
            and cold_median is not None
            and warm_median < cold_median / COLD_FACTOR
        )
        manipulation = {
            "cold_all_above": cold_all_above,
            "warm_median_below_cold_third": warm_below,
            "warm_median_ms": warm_median,
            "cold_median_ms": cold_median,
            "threshold_ms": threshold,
            "pass": cold_all_above and warm_below,
        }
        cross_arm = len(shas) == 1
        arms_differ = None
        if (
            manipulation["pass"]
            and cross_arm
            and "metrics" in arms_out["cold"]
            and "metrics" in arms_out["warm"]
        ):
            arms_differ = (
                arms_out["cold"]["metrics"]["modal_sha256"]
                != arms_out["warm"]["metrics"]["modal_sha256"]
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
        "companion_plan": "FOLLOWUP-COMPANIONS.md (companion C)",
        "cell": CACHE_CELL,
        "endpoints": (
            "1: within-arm modal shares (predicted 1.0 both); "
            "2: arms_differ (predicted true); history matching descriptive"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "records_in": len(records),
            "warmups_excluded": warmups,
            "flushers_excluded": flushers,
        },
        "boxes": boxes,
    }


def write_md(report, path):
    lines = ["# Companion C report — cache-state A/B (exploratory)", ""]
    lines.append(
        f"Generated {report['generated_utc']}. Cell: `{report['cell']}`. "
        f"Records in: {report['totals']['records_in']} — warmups excluded: "
        f"{report['totals']['warmups_excluded']}, flushers excluded: "
        f"{report['totals']['flushers_excluded']}. Plan: "
        f"{report['companion_plan']} (committed pre-data)."
    )
    for box in sorted(report["boxes"]):
        entry = report["boxes"][box]
        manipulation = entry["gates"]["manipulation"]
        lines.append("")
        lines.append(f"## {box}")
        lines.append("")
        lines.append(
            f"- cross-arm negative control: "
            f"{entry['gates']['cross_arm_negative_control']} · manipulation "
            f"gate pass={manipulation['pass']} (warm median "
            f"{manipulation['warm_median_ms']} ms, cold median "
            f"{manipulation['cold_median_ms']} ms)"
        )
        lines.append("")
        lines.append(
            "| arm | n | modal share | distinct | prefill ms (med) | "
            "history matches |"
        )
        lines.append("|---|---|---|---|---|---|")
        for arm in ARMS:
            arm_entry = entry["arms"].get(arm, {})
            metrics = arm_entry.get("metrics")
            if not metrics:
                lines.append(f"| {arm} | 0 | - | - | - | - |")
                continue
            lines.append(
                f"| {arm} | {metrics['n']} | {metrics['modal_share']:.3f} | "
                f"{metrics['distinct_count']} | "
                f"{arm_entry['prefill_ms']['median']:.0f} | "
                f"{', '.join(arm_entry.get('history_matches', [])) or 'none'} |"
            )
        lines.append("")
        lines.append(f"Arms differ (registered endpoint 2): {entry['arms_differ']}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Companion-C cache-state A/B analysis (exploratory)"
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

    report = build_cache_report(records, history=DEFAULT_HISTORY)
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(args.out, f"cache-ab-report-{stamp}.json")
    md_path = os.path.join(args.out, f"cache-ab-report-{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    write_md(report, md_path)

    for box in sorted(report["boxes"]):
        entry = report["boxes"][box]
        gate = entry["gates"]["manipulation"]["pass"]
        cold = entry["arms"]["cold"].get("metrics", {})
        warm = entry["arms"]["warm"].get("metrics", {})
        print(
            f"{box}: gate pass={gate}; cold modal "
            f"{cold.get('modal_share')} warm modal {warm.get('modal_share')} "
            f"arms_differ={entry['arms_differ']}"
        )
    print(f"report -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
