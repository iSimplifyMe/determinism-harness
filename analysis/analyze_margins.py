"""Companion-B analysis: logprob margins (FOLLOWUP-COMPANIONS.md).

Exploratory, descriptive — no hypothesis test. Deterministic, stdlib only.
Consumes study3-margins records (compact per-token rows
[token, chosen_logprob, top1_logprob, top2_logprob]) and reports, per
box::cell:

- distribution of per-call minimum top1-minus-top2 margins;
- pooled fraction of token positions with margin below 0.001 / 0.01 / 0.1;
- for multi-variant cells, the fork: the first token index where the two
  most frequent variants' token streams differ, with each variant's margin
  at that index — the direct near-tie readout;
- chosen_not_top1 totals (greedy anomaly counter).

The observer caveat from the freeze gate rides every report: short-output
cells probed byte-neutral; open-generation margins describe
logprobs-perturbed trajectories, never the frozen runs' trajectories.

Usage:
  python3 -m analysis.analyze_margins runs/local-study3-margins-*.jsonl
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

from analysis.latency_fingerprint import percentile

THRESHOLDS = (0.001, 0.01, 0.1)

OBSERVER_CAVEAT = (
    "logprobs request fields are NOT byte-neutral at generation length "
    "(freeze-gate probe): open_generation margins describe "
    "logprobs-perturbed trajectories, not the frozen runs'."
)


def _median(values):
    values = [v for v in values if v is not None]
    return percentile(values, 50) if values else None


def _row_margin(row):
    if row[2] is None or row[3] is None:
        return None
    return row[2] - row[3]


def _fork(recs):
    """First token index where the two most frequent variants' token
    streams differ, with each variant's margin at that index. Variant
    choice is deterministic: count desc, then sha asc."""
    variants = Counter(r["text_sha256"] for r in recs)
    if len(variants) < 2:
        return None
    ranked = sorted(variants.items(), key=lambda kv: (-kv[1], kv[0]))
    reps = []
    for sha, _count in ranked[:2]:
        rep = next(
            (r for r in recs
             if r["text_sha256"] == sha and r.get("logprob_margins")),
            None,
        )
        if rep is None:
            return None
        reps.append(rep["logprob_margins"]["tokens"])
    modal_rows, alternate_rows = reps
    limit = min(len(modal_rows), len(alternate_rows))
    fork_index = next(
        (i for i in range(limit)
         if modal_rows[i][0] != alternate_rows[i][0]),
        None,
    )
    if fork_index is None:
        if len(modal_rows) == len(alternate_rows):
            return None  # token streams identical; divergence not token-level
        return {"token_index": limit, "note": "length-fork",
                "modal": None, "alternate": None}

    def _at(rows):
        row = rows[fork_index]
        return {
            "token": row[0],
            "chosen_logprob": row[1],
            "margin": _row_margin(row),
        }

    return {
        "token_index": fork_index,
        "modal": _at(modal_rows),
        "alternate": _at(alternate_rows),
    }


def build_margins_report(records):
    warmups = 0
    excluded = {"error": 0, "truncated_or_other_stop": 0, "no_margins": 0}
    grouped = defaultdict(list)
    for record in records:
        if record.get("meta_control") == "warmup":
            warmups += 1
            continue
        if not record.get("ok"):
            excluded["error"] += 1
            continue
        if record.get("stop_reason") != "stop":
            excluded["truncated_or_other_stop"] += 1
            continue
        box = record.get("box") or record.get("meta_hardware")
        grouped[f'{box}::{record["cell"]}'].append(record)

    cells = {}
    for key in sorted(grouped):
        recs = grouped[key]
        margins = [r.get("logprob_margins") for r in recs]
        excluded["no_margins"] += sum(1 for m in margins if not m)
        min_margins = [
            m["min_top2_margin"] for m in margins
            if m and m["min_top2_margin"] is not None
        ]
        pooled_defined = 0
        below = {str(t): 0 for t in THRESHOLDS}
        chosen_not_top1 = 0
        for m in margins:
            if not m:
                continue
            chosen_not_top1 += m["chosen_not_top1"]
            for row in m["tokens"]:
                margin = _row_margin(row)
                if margin is None:
                    continue
                pooled_defined += 1
                for t in THRESHOLDS:
                    if margin < t:
                        below[str(t)] += 1
        variants = Counter(r["text_sha256"] for r in recs)
        cells[key] = {
            "n": len(recs),
            "records_with_margins": sum(1 for m in margins if m),
            "variants": len(variants),
            "variant_counts": sorted(variants.values(), reverse=True),
            "min_margin": {
                "min": min(min_margins) if min_margins else None,
                "median": _median(min_margins),
            },
            "defined_positions": pooled_defined,
            "positions_below": {
                str(t): (
                    below[str(t)] / pooled_defined if pooled_defined else None
                )
                for t in THRESHOLDS
            },
            "chosen_not_top1_total": chosen_not_top1,
            "fork": _fork(recs),
        }

    return {
        "exploratory": True,
        "companion_plan": "FOLLOWUP-COMPANIONS.md",
        "observer_caveat": OBSERVER_CAVEAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "records_in": len(records),
            "warmups_excluded": warmups,
            "excluded": excluded,
        },
        "cells": cells,
    }


def _md_cell(text):
    return str(text).replace("|", "\\|")


def write_md(report, path):
    lines = ["# Companion B report — logprob margins (exploratory)", ""]
    lines.append(
        f"Generated {report['generated_utc']}. Records in: "
        f"{report['totals']['records_in']} — warmups excluded: "
        f"{report['totals']['warmups_excluded']}. Plan: "
        f"{report['companion_plan']} (committed pre-data)."
    )
    lines.append("")
    lines.append(f"Observer caveat: {report['observer_caveat']}")
    lines.append("")
    lines.append(
        "| box::cell | n | variants | min margin (min/med) | "
        "pos < 0.01 | fork idx | fork margins (modal/alt) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for key in sorted(report["cells"]):
        cell = report["cells"][key]
        mm = cell["min_margin"]
        fork = cell["fork"]
        if fork and fork.get("modal"):
            fork_idx = fork["token_index"]
            fork_margins = (
                f"{fork['modal']['margin']:.4f} / "
                f"{fork['alternate']['margin']:.4f}"
            )
        elif fork:
            fork_idx = fork["token_index"]
            fork_margins = fork.get("note", "-")
        else:
            fork_idx = "-"
            fork_margins = "-"
        below = cell["positions_below"]["0.01"]

        def _fmt(value):
            return "-" if value is None else f"{value:.4f}"

        lines.append(
            f"| {_md_cell(key)} | {cell['n']} | {cell['variants']} | "
            f"{_fmt(mm['min'])} / {_fmt(mm['median'])} | "
            f"{_fmt(below)} | {fork_idx} | {fork_margins} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Companion-B logprob-margins analysis (exploratory)"
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

    report = build_margins_report(records)
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(args.out, f"margins-report-{stamp}.json")
    md_path = os.path.join(args.out, f"margins-report-{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    write_md(report, md_path)

    for key in sorted(report["cells"]):
        cell = report["cells"][key]
        print(
            f"{key}: n={cell['n']} variants={cell['variants']} "
            f"min_margin_min={cell['min_margin']['min']} "
            f"below_0.01={cell['positions_below']['0.01']}"
        )
    print(f"report -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
