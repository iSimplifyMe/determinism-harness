"""Study-3 analysis: runs/*study3*.jsonl -> reports/study3-report-*.json/.md

REGISTERED estimators, per PREREGISTRATION-v3 — this module is committed
BEFORE any confirmatory study-3 data exists, the discipline adopted after
study 1's estimator miss. Deterministic code only; stdlib only.

- Q1  per-cell modal share of byte-identical text + Wilson 95, identical
      machinery to studies 1-2 (analysis.metrics.cell_metrics).
- Q2  concurrency effect per model = equal-weight stratified difference
      over matched task strata (c4 arm vs the core grid's single-flight
      comparators), per-stratum binomial variances; the MoE-minus-dense
      difference-of-differences propagates both stratified SEs.
- Q3  thinking ON-minus-OFF modal-share difference on structured JSON per
      model (Wald, CI95).
- Q4  cross-box identity on cells present on both boxes: modal-output
      match, variant-set overlap, record-level cross-coverage. Decode rate
      (eval_duration is pure decode time) reported as the EXPLORATORY
      hardware-calibration readout.

Gates: >1 wire hash in a cell = negative-control failure (cell voided);
errors and non-"stop" stop reasons excluded and counted (local plane maps
Ollama done_reason -> stop_reason, so truncation arrives as "length");
warmup records (meta_control=warmup) excluded like pilot data; response-
model constancy flagged; the temp07 sampling arm is the positive-control
analog and must fire distinct outputs on open_generation per model+box.

Usage:
  python3 -m analysis.analyze_study3 runs/local-study3-*.jsonl --out reports
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

from analysis.latency_fingerprint import percentile
from analysis.metrics import cell_metrics
from analysis.stats import Z95, stratified_diff, wald_diff, wilson_interval

Q2_MOE = ("qwen3.5-122b", "qwen3.6-35b")
Q2_DENSE = ("qwen3-vl-32b",)

# ON/HIGH arm -> its LOW/OFF comparator (the core grid's pinned arm).
OFF_FOR_ON = {"think_on": "think_off", "effort_high": "effort_low"}

LOCAL_VALID_STOP = "stop"  # Ollama done_reason for a natural finish


def gate_cell3(records):
    """Validity gates for one local cell. Mirrors analyze.gate_cell with the
    local stop-reason mapping; the wire gate is byte-exact here because the
    harness owns the request bytes end to end."""
    excluded = {"error": 0, "truncated_or_other_stop": 0}
    flags = {
        "negative_control_failed": False,
        "wire_mismatch": False,
        "model_drift": False,
        "response_models": [],
    }
    hashes = {r.get("request_sha256") for r in records}
    wire_hashes = {r.get("wire_sha256") for r in records if r.get("wire_sha256")}
    if len(hashes) > 1 or len(wire_hashes) > 1:
        flags["negative_control_failed"] = True
        flags["wire_mismatch"] = len(wire_hashes) > 1
        return {
            "valid": [],
            "excluded": excluded,
            "flags": flags,
            "n_raw": len(records),
        }
    valid = []
    for record in records:
        if not record.get("ok"):
            excluded["error"] += 1
            continue
        if record.get("stop_reason") != LOCAL_VALID_STOP:
            excluded["truncated_or_other_stop"] += 1
            continue
        valid.append(record)
    models = sorted(
        {r.get("response_model") for r in valid if r.get("response_model")}
    )
    flags["response_models"] = models
    flags["model_drift"] = len(models) > 1
    return {
        "valid": valid,
        "excluded": excluded,
        "flags": flags,
        "n_raw": len(records),
    }


def group_records(records):
    """Group by box::cell, dropping warmup records (counted, never analyzed).
    Hardware is a real factor, so the box is part of the group key — Q4
    compares identical cell keys across the two boxes."""
    grouped = defaultdict(list)
    warmups = 0
    for record in records:
        if record.get("meta_control") == "warmup":
            warmups += 1
            continue
        box = record.get("box") or record.get("meta_hardware")
        grouped[f'{box}::{record["cell"]}'].append(record)
    return grouped, warmups


def _meta_of(record):
    return {k[5:]: v for k, v in record.items() if k.startswith("meta_")}


def q1_cells(grouped):
    """Gate and measure every (box, cell) group — the Q1 surface."""
    cells = {}
    for key in sorted(grouped):
        recs = grouped[key]
        gate = gate_cell3(recs)
        meta = _meta_of(recs[0])
        entry = {
            "cell": recs[0]["cell"],
            "box": recs[0].get("box") or meta.get("hardware"),
            "meta": meta,
            "gate": {
                "n_raw": gate["n_raw"],
                "excluded": gate["excluded"],
                "flags": gate["flags"],
            },
        }
        valid = gate["valid"]
        if valid:
            measurable = [
                {
                    "text": r["text"],
                    "output_tokens": (r.get("usage") or {}).get("output_tokens"),
                }
                for r in valid
            ]
            metrics = cell_metrics(measurable)
            entry["metrics"] = metrics
            entry["wilson_ci"] = wilson_interval(metrics["modal_count"], metrics["n"])
            entry["text_hash_counts"] = dict(
                Counter(r["text_sha256"] for r in valid)
            )
        cells[key] = entry
    return cells


def positive_control_gate(cells):
    """The temp07 arm is the per-model sampling positive control: on
    open_generation it must produce >1 distinct output, else that model's
    greedy nulls are uninterpretable on that box."""
    per_model = []
    for key in sorted(cells):
        entry = cells[key]
        meta = entry.get("meta", {})
        if meta.get("sampling") != "temp07":
            continue
        if meta.get("task") != "open_generation":
            continue
        metrics = entry.get("metrics")
        fired = bool(metrics and metrics["distinct_count"] > 1)
        per_model.append({
            "box": entry["box"],
            "model": meta.get("model"),
            "fired": fired,
            "distinct": metrics["distinct_count"] if metrics else 0,
            "n": metrics["n"] if metrics else 0,
        })
    return {
        "per_model": per_model,
        "all_fired": bool(per_model) and all(g["fired"] for g in per_model),
    }


def _index(cells):
    idx = {}
    for entry in cells.values():
        meta = entry.get("meta", {})
        idx[(
            entry["box"],
            meta.get("model"),
            meta.get("task"),
            meta.get("sampling"),
            meta.get("thinking"),
            meta.get("concurrency"),
        )] = entry
    return idx


def _sorted_index_items(idx):
    return sorted(idx.items(), key=lambda kv: str(kv[0]))


def q2_concurrency(cells):
    """REGISTERED Q2 estimator: per model, equal-weight stratified difference
    (concurrency-4 minus single-flight) over matched task strata; then the
    MoE-minus-dense difference-of-differences with propagated SE."""
    idx = _index(cells)
    strata_by_model = defaultdict(list)
    skipped = 0
    for (box, model, task, sampling, thinking, conc), entry in _sorted_index_items(idx):
        if conc != 4:
            continue
        if "metrics" not in entry:
            skipped += 1
            continue
        comparator = idx.get((box, model, task, sampling, thinking, None))
        if not comparator or "metrics" not in comparator:
            skipped += 1
            continue
        m4, m1 = entry["metrics"], comparator["metrics"]
        strata_by_model[model].append(
            (m4["modal_count"], m4["n"], m1["modal_count"], m1["n"])
        )
    per_model = []
    for model in sorted(strata_by_model):
        est = stratified_diff(strata_by_model[model])
        per_model.append({
            "model": model,
            "arch": "moe" if model in Q2_MOE else "dense",
            "diff": est["diff"],
            "se": est["se"],
            "n_strata": est["n_strata"],
            "ci95": (est["diff"] - Z95 * est["se"], est["diff"] + Z95 * est["se"]),
        })
    moe_strata = [s for m in Q2_MOE for s in strata_by_model.get(m, [])]
    dense_strata = [s for m in Q2_DENSE for s in strata_by_model.get(m, [])]
    dod = None
    if moe_strata and dense_strata:
        moe = stratified_diff(moe_strata)
        dense = stratified_diff(dense_strata)
        diff = moe["diff"] - dense["diff"]
        se = math.sqrt(moe["se"] ** 2 + dense["se"] ** 2)
        dod = {
            "diff": diff,
            "se": se,
            "ci95": (diff - Z95 * se, diff + Z95 * se),
            "moe_strata": len(moe_strata),
            "dense_strata": len(dense_strata),
        }
    return {
        "estimator": (
            "equal-weight stratified diff over matched task strata "
            "(registered primary, prereg v3 Q2)"
        ),
        "per_model": per_model,
        "moe_minus_dense": dod,
        "skipped_strata": skipped,
    }


def q3_thinking(cells):
    """REGISTERED Q3 estimator: per model+box, ON-minus-OFF modal-share
    difference on structured JSON under greedy sampling (Wald, CI95)."""
    idx = _index(cells)
    per_model = []
    skipped = 0
    for (box, model, task, sampling, thinking, conc), entry in _sorted_index_items(idx):
        if thinking not in OFF_FOR_ON or conc is not None:
            continue
        if task != "structured_json" or sampling != "greedy":
            continue
        if "metrics" not in entry:
            skipped += 1
            continue
        off = idx.get((box, model, task, sampling, OFF_FOR_ON[thinking], None))
        if not off or "metrics" not in off:
            skipped += 1
            continue
        on_m, off_m = entry["metrics"], off["metrics"]
        diff = wald_diff(on_m["modal_count"], on_m["n"], off_m["modal_count"], off_m["n"])
        per_model.append({
            "box": box,
            "model": model,
            "on_arm": thinking,
            "on_modal_share": on_m["modal_share"],
            "off_modal_share": off_m["modal_share"],
            "diff": diff["diff"],
            "se": diff["se"],
            "ci95": diff["ci95"],
        })
    return {
        "estimator": (
            "ON-minus-OFF modal-share Wald difference on structured JSON, "
            "greedy (registered, prereg v3 Q3)"
        ),
        "per_model": per_model,
        "skipped": skipped,
    }


def q4_cross_box(cells):
    """Cross-box identity: for every non-concurrency cell present on both
    boxes, does CUDA produce Metal's modal bytes (and vice versa)? Overlap
    is reported at variant level and record level. Registered expectation
    (prereg v3 Q4): within-box high, cross-box NOT byte-identical."""
    idx = _index(cells)
    out = []
    matches = 0
    for (box, model, task, sampling, thinking, conc), entry in _sorted_index_items(idx):
        if box != "metal" or conc is not None or "metrics" not in entry:
            continue
        other = idx.get(("cuda", model, task, sampling, thinking, None))
        if not other or "metrics" not in other:
            continue
        metal_hashes = entry["text_hash_counts"]
        cuda_hashes = other["text_hash_counts"]
        shared = set(metal_hashes) & set(cuda_hashes)
        metal_n = sum(metal_hashes.values())
        cuda_n = sum(cuda_hashes.values())
        cell = {
            "cell": entry["cell"],
            "metal": {
                "n": metal_n,
                "modal_share": entry["metrics"]["modal_share"],
                "modal_sha256": entry["metrics"]["modal_sha256"],
            },
            "cuda": {
                "n": cuda_n,
                "modal_share": other["metrics"]["modal_share"],
                "modal_sha256": other["metrics"]["modal_sha256"],
            },
            "modal_match": (
                entry["metrics"]["modal_sha256"] == other["metrics"]["modal_sha256"]
            ),
            "overlap": {
                "shared_variants": len(shared),
                "metal_only_variants": len(set(metal_hashes) - shared),
                "cuda_only_variants": len(set(cuda_hashes) - shared),
                "metal_records_matched_in_cuda": (
                    sum(c for h, c in metal_hashes.items() if h in cuda_hashes)
                    / metal_n
                ),
                "cuda_records_matched_in_metal": (
                    sum(c for h, c in cuda_hashes.items() if h in metal_hashes)
                    / cuda_n
                ),
            },
        }
        if cell["modal_match"]:
            matches += 1
        out.append(cell)
    return {"cells": out, "cells_compared": len(out), "modal_matches": matches}


def q4_decode_rate(records):
    """EXPLORATORY hardware calibration: tokens/sec from eval_duration (pure
    decode time, no load/prefill) per box|model, over valid records."""
    grouped = defaultdict(list)
    for record in records:
        if record.get("meta_control") == "warmup" or not record.get("ok"):
            continue
        usage = record.get("usage") or {}
        eval_ns = usage.get("eval_duration_ns")
        tokens = usage.get("output_tokens")
        if not eval_ns or not tokens:
            continue
        box = record.get("box") or record.get("meta_hardware")
        grouped[f'{box}|{record.get("meta_model")}'].append(
            tokens / (eval_ns / 1e9)
        )
    out = {}
    for key in sorted(grouped):
        values = grouped[key]
        out[key] = {
            "n": len(values),
            "p25": round(percentile(values, 25), 1),
            "p50": round(percentile(values, 50), 1),
            "p75": round(percentile(values, 75), 1),
        }
    return out


def build_report(records):
    grouped, warmups = group_records(records)
    cells = q1_cells(grouped)
    negative_failed = sorted(
        k for k, e in cells.items()
        if e["gate"]["flags"]["negative_control_failed"]
    )
    drift = sorted(
        k for k, e in cells.items() if e["gate"]["flags"]["model_drift"]
    )
    return {
        "registered": True,
        "estimators_source": (
            "PREREGISTRATION-v3 (this module committed before any "
            "confirmatory study-3 data)"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "records_in": len(records),
            "warmups_excluded": warmups,
            "cells": len(cells),
        },
        "gates": {
            "negative_control_failed_cells": negative_failed,
            "model_drift_cells": drift,
            "positive_control": positive_control_gate(cells),
        },
        "q1_cells": cells,
        "q2_concurrency": q2_concurrency(cells),
        "q3_thinking": q3_thinking(cells),
        "q4_cross_box": q4_cross_box(cells),
        "q4_decode_rate_tokens_per_sec": q4_decode_rate(records),
    }


def _fmt_ci(ci):
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def _md_cell(text):
    """Cell keys carry literal pipes; escape them so GFM tables survive."""
    return str(text).replace("|", "\\|")


def write_md(report, path):
    lines = ["# Study 3 report (registered estimators)", ""]
    lines.append(
        f"Generated {report['generated_utc']}. Records in: "
        f"{report['totals']['records_in']} - warmups excluded: "
        f"{report['totals']['warmups_excluded']} - cells: "
        f"{report['totals']['cells']}."
    )
    gates = report["gates"]
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    lines.append(
        f"- negative-control failures: "
        f"{len(gates['negative_control_failed_cells'])} "
        f"{gates['negative_control_failed_cells'] or ''}"
    )
    lines.append(f"- model-drift cells: {len(gates['model_drift_cells'])}")
    pc = gates["positive_control"]
    lines.append(
        f"- positive control (temp07 open_generation): all_fired="
        f"{pc['all_fired']} over {len(pc['per_model'])} model-box pairs"
    )
    q1 = report["q1_cells"]
    measured = [e for e in q1.values() if "metrics" in e]
    at_ceiling = [e for e in measured if e["metrics"]["modal_share"] == 1.0]
    lines.append("")
    lines.append("## Q1 - the ceiling")
    lines.append("")
    lines.append(
        f"{len(at_ceiling)}/{len(measured)} measured cells at modal share "
        f"1.0. Cells below the ceiling:"
    )
    lines.append("")
    lines.append("| box::cell | n | modal share | Wilson 95 | distinct |")
    lines.append("|---|---|---|---|---|")
    for key in sorted(q1):
        entry = q1[key]
        metrics = entry.get("metrics")
        if not metrics or metrics["modal_share"] == 1.0:
            continue
        lo, hi = entry["wilson_ci"]
        lines.append(
            f"| {_md_cell(key)} | {metrics['n']} | "
            f"{metrics['modal_share']:.3f} | "
            f"[{lo:.3f}, {hi:.3f}] | {metrics['distinct_count']} |"
        )
    q2 = report["q2_concurrency"]
    lines.append("")
    lines.append("## Q2 - concurrency (registered stratified estimator)")
    lines.append("")
    lines.append("| model | arch | diff (c4-c1) | SE | CI95 | strata |")
    lines.append("|---|---|---|---|---|---|")
    for m in q2["per_model"]:
        lines.append(
            f"| {m['model']} | {m['arch']} | {m['diff']:+.4f} | "
            f"{m['se']:.4f} | {_fmt_ci(m['ci95'])} | {m['n_strata']} |"
        )
    dod = q2["moe_minus_dense"]
    if dod:
        lines.append("")
        lines.append(
            f"MoE-minus-dense DoD: {dod['diff']:+.4f} "
            f"(SE {dod['se']:.4f}, CI95 {_fmt_ci(dod['ci95'])}; "
            f"{dod['moe_strata']}+{dod['dense_strata']} strata)."
        )
    q3 = report["q3_thinking"]
    lines.append("")
    lines.append("## Q3 - thinking analog (structured JSON, greedy)")
    lines.append("")
    lines.append("| box | model | on-arm | on | off | diff | CI95 |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in q3["per_model"]:
        lines.append(
            f"| {m['box']} | {m['model']} | {m['on_arm']} | "
            f"{m['on_modal_share']:.3f} | {m['off_modal_share']:.3f} | "
            f"{m['diff']:+.4f} | {_fmt_ci(m['ci95'])} |"
        )
    q4 = report["q4_cross_box"]
    lines.append("")
    lines.append("## Q4 - cross-box identity (gpt-oss, Metal vs CUDA)")
    lines.append("")
    lines.append(
        f"{q4['modal_matches']}/{q4['cells_compared']} compared cells have "
        f"byte-identical modal outputs across boxes."
    )
    lines.append("")
    lines.append("| cell | metal share | cuda share | modal match | shared variants | metal->cuda coverage |")
    lines.append("|---|---|---|---|---|---|")
    for cell in q4["cells"]:
        lines.append(
            f"| {_md_cell(cell['cell'])} | "
            f"{cell['metal']['modal_share']:.3f} | "
            f"{cell['cuda']['modal_share']:.3f} | {cell['modal_match']} | "
            f"{cell['overlap']['shared_variants']} | "
            f"{cell['overlap']['metal_records_matched_in_cuda']:.3f} |"
        )
    rates = report["q4_decode_rate_tokens_per_sec"]
    lines.append("")
    lines.append("## Decode rate (exploratory calibration, tokens/sec)")
    lines.append("")
    lines.append("| box, model | n | p25 | p50 | p75 |")
    lines.append("|---|---|---|---|---|")
    for key, s in rates.items():
        lines.append(
            f"| {_md_cell(key)} | {s['n']} | {s['p25']} | {s['p50']} | "
            f"{s['p75']} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Study-3 analysis (registered estimators)"
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

    report = build_report(records)
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(args.out, f"study3-report-{stamp}.json")
    md_path = os.path.join(args.out, f"study3-report-{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    write_md(report, md_path)

    totals = report["totals"]
    pc = report["gates"]["positive_control"]
    print(
        f"records={totals['records_in']} cells={totals['cells']} "
        f"warmups_excluded={totals['warmups_excluded']} "
        f"neg_failures={len(report['gates']['negative_control_failed_cells'])} "
        f"pc_all_fired={pc['all_fired']}"
    )
    dod = report["q2_concurrency"]["moe_minus_dense"]
    if dod:
        print(f"Q2 MoE-minus-dense: {dod['diff']:+.4f} CI95 {_fmt_ci(dod['ci95'])}")
    for m in report["q3_thinking"]["per_model"]:
        print(f"Q3 {m['box']}|{m['model']}: {m['diff']:+.4f} CI95 {_fmt_ci(m['ci95'])}")
    q4 = report["q4_cross_box"]
    print(f"Q4 cross-box modal matches: {q4['modal_matches']}/{q4['cells_compared']}")
    print(f"report -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
