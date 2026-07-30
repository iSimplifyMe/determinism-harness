"""Companion-A analysis: reload-churn A/B (FOLLOWUP-COMPANIONS.md).

Exploratory, with the plan and this analyzer committed before the data.
Deterministic code, stdlib only, like every analysis module in the arc.

Gates, per the committed plan:
- per-arm validity via analyze_study3.gate_cell3 (errors / truncation /
  wire identity within the arm);
- cross-arm negative control: BOTH arms must share exactly one request
  sha256 — the manipulation lives outside the measured request;
- manipulation gate: every valid churn record confirmed absent from
  /api/ps before its call AND cold-loaded (> COLD_FACTOR x the blocked
  arm's median load), and the blocked arm's median load must be warm
  (< WARM_MAX_S). A failed gate voids the primary endpoint — the report
  says so instead of estimating anyway.

Primary endpoint per box: churn-minus-blocked modal-share difference
(Wald, CI95). Negative diff = churn reduced reproducibility.

Usage:
  python3 -m analysis.analyze_churn_ab runs/local-study3-churn-ab-*.jsonl \
      --confirmatory-report reports/study3-report-20260730T100556Z.json
"""
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from analysis.analyze_study3 import gate_cell3
from analysis.latency_fingerprint import percentile
from analysis.metrics import cell_metrics
from analysis.stats import wald_diff

CHURN_CELL = "gpt-oss-20b|open_generation|greedy|effort_low"
ARMS = ("blocked", "churn")
# 3x, not 10x: a confirmed unload/reload is disk-warm on macOS (page
# cache), so load time corroborates the manipulation rather than defining
# it — see the pre-data amendment in FOLLOWUP-COMPANIONS.md. Smoke
# reloads: 13.9x (CUDA), 6.2x (Metal).
COLD_FACTOR = 3.0
WARM_MAX_S = 1.0


def _median(values):
    values = [v for v in values if v is not None]
    return percentile(values, 50) if values else None


def build_churn_report(records, confirmatory=None):
    warmups = 0
    by_box_arm = defaultdict(list)
    for record in records:
        if record.get("meta_control") == "warmup":
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
        loads = {}
        for arm in ARMS:
            recs = by_box_arm.get((box, arm), [])
            gate = gate_cell3(recs)
            valid = gate["valid"]
            shas.update(
                r.get("request_sha256") for r in recs if r.get("request_sha256")
            )
            load_s = [
                ((r.get("usage") or {}).get("load_duration_ns") or 0) / 1e9
                for r in valid
            ]
            loads[arm] = load_s
            entry = {
                "n_raw": gate["n_raw"],
                "excluded": gate["excluded"],
                "flags": gate["flags"],
                "load_duration_s": {
                    "median": _median(load_s),
                    "max": max(load_s) if load_s else None,
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
                if confirmatory is not None:
                    modal = (
                        confirmatory.get("q1_cells", {})
                        .get(f"{box}::{CHURN_CELL}", {})
                        .get("metrics", {})
                        .get("modal_sha256")
                    )
                    entry["matches_confirmatory_modal"] = (
                        modal is not None
                        and metrics["modal_sha256"] == modal
                    )
            arms_out[arm] = entry

        blocked_median = _median(loads.get("blocked", []))
        churn_valid = [
            r for r in by_box_arm.get((box, "churn"), [])
            if r.get("ok") and r.get("stop_reason") == "stop"
        ]
        churn_all_confirmed = bool(churn_valid) and all(
            r.get("pre_unload_confirmed") is True for r in churn_valid
        )
        cold_threshold = (
            blocked_median * COLD_FACTOR if blocked_median is not None else None
        )
        churn_loads = loads.get("churn", [])
        churn_all_cold = (
            bool(churn_loads)
            and cold_threshold is not None
            and all(s > cold_threshold for s in churn_loads)
        )
        blocked_warm = blocked_median is not None and blocked_median < WARM_MAX_S
        manipulation = {
            "blocked_warm": blocked_warm,
            "churn_all_confirmed": churn_all_confirmed,
            "churn_all_cold": churn_all_cold,
            "blocked_median_load_s": blocked_median,
            "churn_median_load_s": _median(churn_loads),
            "cold_threshold_s": cold_threshold,
            "pass": blocked_warm and churn_all_confirmed and churn_all_cold,
        }
        cross_arm = len(shas) == 1
        diff = None
        if (
            manipulation["pass"]
            and cross_arm
            and "metrics" in arms_out["blocked"]
            and "metrics" in arms_out["churn"]
        ):
            blocked_m = arms_out["blocked"]["metrics"]
            churn_m = arms_out["churn"]["metrics"]
            diff = wald_diff(
                churn_m["modal_count"], churn_m["n"],
                blocked_m["modal_count"], blocked_m["n"],
            )
        boxes[box] = {
            "arms": arms_out,
            "gates": {
                "cross_arm_negative_control": cross_arm,
                "manipulation": manipulation,
            },
            "churn_minus_blocked": diff,
        }

    return {
        "exploratory": True,
        "companion_plan": "FOLLOWUP-COMPANIONS.md",
        "cell": CHURN_CELL,
        "estimator": (
            "churn-minus-blocked modal-share Wald difference, computed only "
            "when the manipulation and cross-arm gates pass"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "records_in": len(records),
            "warmups_excluded": warmups,
        },
        "boxes": boxes,
    }


def _fmt_ci(ci):
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def write_md(report, path):
    lines = ["# Companion A report — reload-churn A/B (exploratory)", ""]
    lines.append(
        f"Generated {report['generated_utc']}. Cell: `{report['cell']}`. "
        f"Records in: {report['totals']['records_in']} — warmups excluded: "
        f"{report['totals']['warmups_excluded']}. Plan: "
        f"{report['companion_plan']} (committed pre-data)."
    )
    for box in sorted(report["boxes"]):
        entry = report["boxes"][box]
        gates = entry["gates"]
        manipulation = gates["manipulation"]
        lines.append("")
        lines.append(f"## {box}")
        lines.append("")
        lines.append(
            f"- cross-arm negative control (one sha both arms): "
            f"{gates['cross_arm_negative_control']}"
        )
        lines.append(
            f"- manipulation gate: pass={manipulation['pass']} "
            f"(blocked_warm={manipulation['blocked_warm']}, "
            f"churn_all_confirmed={manipulation['churn_all_confirmed']}, "
            f"churn_all_cold={manipulation['churn_all_cold']}; "
            f"blocked median load "
            f"{manipulation['blocked_median_load_s']}s, churn median "
            f"{manipulation['churn_median_load_s']}s)"
        )
        lines.append("")
        lines.append("| arm | n | modal share | distinct | matches confirmatory modal |")
        lines.append("|---|---|---|---|---|")
        for arm in ARMS:
            arm_entry = entry["arms"].get(arm, {})
            metrics = arm_entry.get("metrics")
            if not metrics:
                lines.append(f"| {arm} | 0 | - | - | - |")
                continue
            lines.append(
                f"| {arm} | {metrics['n']} | {metrics['modal_share']:.3f} | "
                f"{metrics['distinct_count']} | "
                f"{arm_entry.get('matches_confirmatory_modal', '-')} |"
            )
        diff = entry["churn_minus_blocked"]
        lines.append("")
        if diff is None:
            lines.append(
                "Primary endpoint NOT computed (a gate failed — see above)."
            )
        else:
            lines.append(
                f"Churn-minus-blocked modal-share diff: {diff['diff']:+.4f} "
                f"(SE {diff['se']:.4f}, CI95 {_fmt_ci(diff['ci95'])})."
            )
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Companion-A reload-churn A/B analysis (exploratory)"
    )
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--confirmatory-report",
                        help="study3 report JSON for modal cross-checks")
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
    confirmatory = None
    if args.confirmatory_report:
        with open(args.confirmatory_report, encoding="utf-8") as fh:
            confirmatory = json.load(fh)

    report = build_churn_report(records, confirmatory=confirmatory)
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(args.out, f"churn-ab-report-{stamp}.json")
    md_path = os.path.join(args.out, f"churn-ab-report-{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    write_md(report, md_path)

    for box in sorted(report["boxes"]):
        entry = report["boxes"][box]
        diff = entry["churn_minus_blocked"]
        gate = entry["gates"]["manipulation"]["pass"]
        if diff is None:
            print(f"{box}: gate pass={gate}; primary NOT computed")
        else:
            print(
                f"{box}: gate pass={gate}; churn-minus-blocked "
                f"{diff['diff']:+.4f} CI95 {_fmt_ci(diff['ci95'])}"
            )
    print(f"report -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
