"""Validator catchability: reanalysis of the arc's confirmatory records.

Question (zero new calls): of the byte-level instability the three studies
measured, how much would a deterministic output validator have caught
(rejected) or neutralized (accepted into the same canonical object), and
how much passes validation while semantically divergent — the class no
deterministic gate can see?

The validator model is stated, deterministic, and fail-closed:
- structured_json  parse (strict, or after stripping the forbidden fence —
                   the same normalization as analysis.semantic_sj) + exact
                   key set + type check against the task's fixed schema.
- classification   exact membership in the instructed label set (after
                   whitespace strip).
- extraction       format check on the instructed output shape
                   (PO-#####-AA); format-only by construction — a
                   format-valid wrong value passes, which is precisely the
                   invisible-divergence risk this report quantifies.
- open_generation  no deterministic validator exists: coverage
                   "unvalidatable"; byte-level metrics only.

Per cell and pooled: byte_modal_share (what the studies measured) vs
post_validator_repro (semantic modal share among accepted responses),
reject_rate (availability cost), invisible_divergence_rate (residual risk),
recovered (= post - byte, the fraction of instability a validator absorbs).

Stdlib only; no model in the loop. Exploratory reanalysis of committed
confirmatory data — feed it the confirmatory run files, never pilots.

Usage:
  python3 -m analysis.validator_catchability \
      runs/low-full-*.jsonl runs/mid-full-*.jsonl ... --out reports
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

from analysis.semantic_sj import parse_sj

CLASSIFICATION_LABELS = ("BILLING", "TECHNICAL", "ACCOUNT", "GENERAL")

EXTRACTION_RE = re.compile(r"\APO-\d{5}-[A-Z]{2}\Z")

SJ_STRING_KEYS = ("sku", "name")
SJ_NUMBER_KEYS = ("quantity_on_hand", "unit_price_usd", "reorder_threshold")
SJ_BOOL_KEYS = ("in_stock",)
SJ_KEYS = frozenset(SJ_STRING_KEYS + SJ_NUMBER_KEYS + SJ_BOOL_KEYS)

VALID_STOP = ("end_turn", "stop")


def task_family(task):
    """extraction_pad_* (the study-2 length ladder) shares extraction's
    instructed output shape, so it shares its validator."""
    if task.startswith("extraction"):
        return "extraction"
    return task


def _sj_schema_ok(obj):
    if not isinstance(obj, dict) or set(obj) != SJ_KEYS:
        return False
    for key in SJ_STRING_KEYS:
        if not isinstance(obj[key], str):
            return False
    for key in SJ_NUMBER_KEYS:
        if isinstance(obj[key], bool) or not isinstance(obj[key], (int, float)):
            return False
    for key in SJ_BOOL_KEYS:
        if not isinstance(obj[key], bool):
            return False
    return True


def validate_response(task, text):
    """One response through the deterministic validator for its task."""
    family = task_family(task)
    if family == "open_generation":
        return {"coverage": "unvalidatable", "accepted": None,
                "mode": None, "semantic": None, "reason": None}
    if family == "structured_json":
        mode, sem = parse_sj(text)
        if sem is None:
            return {"coverage": "validatable", "accepted": False,
                    "mode": None, "semantic": None, "reason": "parse_fail"}
        if not _sj_schema_ok(json.loads(sem)):
            return {"coverage": "validatable", "accepted": False,
                    "mode": mode, "semantic": None, "reason": "schema_fail"}
        return {"coverage": "validatable", "accepted": True,
                "mode": mode, "semantic": sem, "reason": None}
    stripped = text.strip()
    if family == "classification":
        if stripped in CLASSIFICATION_LABELS:
            return {"coverage": "validatable", "accepted": True,
                    "mode": "strict", "semantic": stripped, "reason": None}
        return {"coverage": "validatable", "accepted": False,
                "mode": None, "semantic": None, "reason": "format_fail"}
    if family == "extraction":
        if EXTRACTION_RE.match(stripped):
            return {"coverage": "validatable", "accepted": True,
                    "mode": "strict", "semantic": stripped, "reason": None}
        return {"coverage": "validatable", "accepted": False,
                "mode": None, "semantic": None, "reason": "format_fail"}
    raise ValueError(f"unknown task family: {family}")


def config_class(record):
    """sampled = the intentional-divergence arms (temp-0.7 positive
    controls, study-3 temp07 sampling); everything else ran a
    deterministic configuration."""
    if record.get("meta_control") == "positive":
        return "sampled"
    if record.get("meta_sampling") == "temp07":
        return "sampled"
    return "deterministic"


def cell_accounting(task, texts):
    """Validator accounting for one cell's response texts."""
    n = len(texts)
    byte_counts = Counter(texts)
    byte_modal = max(byte_counts.values()) if byte_counts else 0
    out = {
        "n": n,
        "byte_distinct": len(byte_counts),
        "byte_modal_share": byte_modal / n if n else None,
    }
    if task_family(task) == "open_generation":
        out["coverage"] = "unvalidatable"
        return out
    out["coverage"] = "validatable"
    semantics = Counter()
    rejected = 0
    reasons = Counter()
    modes = Counter()
    for text in texts:
        verdict = validate_response(task, text)
        if not verdict["accepted"]:
            rejected += 1
            reasons[verdict["reason"]] += 1
            continue
        modes[verdict["mode"]] += 1
        semantics[verdict["semantic"]] += 1
    accepted = n - rejected
    semantic_modal = max(semantics.values()) if semantics else 0
    out.update({
        "rejected": rejected,
        "reject_rate": rejected / n if n else None,
        "reject_reasons": dict(reasons),
        "accept_modes": dict(modes),
        "accepted": accepted,
        "semantic_distinct": len(semantics),
        "semantic_modal_count": semantic_modal,
        "post_validator_repro": (
            semantic_modal / accepted if accepted else None
        ),
        "invisible_divergence_rate": (
            (accepted - semantic_modal) / n if n else None
        ),
    })
    if out["post_validator_repro"] is not None:
        out["recovered"] = out["post_validator_repro"] - out["byte_modal_share"]
    return out


def _group_key(record):
    schema = record.get("schema", 1)
    if schema == 3:
        box = record.get("box") or record.get("meta_hardware")
        return f'study3::{box}::{record["cell"]}'
    return f'study{schema}::{record["cell"]}'


def build_validator_report(records):
    warmups = 0
    excluded = {"error": 0, "truncated_or_other_stop": 0}
    grouped = defaultdict(list)
    for record in records:
        if record.get("meta_control") == "warmup":
            warmups += 1
            continue
        if not record.get("ok"):
            excluded["error"] += 1
            continue
        if record.get("stop_reason") not in VALID_STOP:
            excluded["truncated_or_other_stop"] += 1
            continue
        grouped[_group_key(record)].append(record)

    cells = {}
    aggregates = defaultdict(lambda: {
        "n": 0, "byte_modal_count": 0, "rejected": 0, "accepted": 0,
        "semantic_modal_count": 0, "cells": 0,
    })
    analyzed = 0
    for key in sorted(grouped):
        recs = grouped[key]
        analyzed += len(recs)
        task = recs[0].get("meta_task", "")
        acct = cell_accounting(task, [r["text"] for r in recs])
        study = key.split("::", 1)[0]
        cls = config_class(recs[0])
        entry = {
            "task_family": task_family(task),
            "model": recs[0].get("meta_model"),
            "config_class": cls,
            **acct,
        }
        cells[key] = entry
        if acct["coverage"] != "validatable":
            continue
        agg = aggregates[f"{study}::{task_family(task)}::{cls}"]
        agg["n"] += acct["n"]
        agg["byte_modal_count"] += round(acct["byte_modal_share"] * acct["n"])
        agg["rejected"] += acct["rejected"]
        agg["accepted"] += acct["accepted"]
        agg["semantic_modal_count"] += acct["semantic_modal_count"]
        agg["cells"] += 1

    finalized = {}
    for key, agg in sorted(aggregates.items()):
        finalized[key] = {
            **agg,
            "byte_modal_pooled": (
                agg["byte_modal_count"] / agg["n"] if agg["n"] else None
            ),
            "post_validator_pooled": (
                agg["semantic_modal_count"] / agg["accepted"]
                if agg["accepted"] else None
            ),
            "reject_rate_pooled": (
                agg["rejected"] / agg["n"] if agg["n"] else None
            ),
            "invisible_rate_pooled": (
                (agg["accepted"] - agg["semantic_modal_count"]) / agg["n"]
                if agg["n"] else None
            ),
        }

    return {
        "exploratory": True,
        "reanalysis_of": "committed confirmatory runs (zero new calls)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "validator_model": (
            "structured_json: parse (strict|fenced) + exact schema; "
            "classification: label-set membership; extraction: format "
            "regex (format-only); open_generation: unvalidatable"
        ),
        "totals": {
            "records_in": len(records),
            "records_analyzed": analyzed,
            "warmups_excluded": warmups,
            "excluded": excluded,
        },
        "cells": cells,
        "aggregates": finalized,
    }


def _fmt(value):
    return "-" if value is None else f"{value:.4f}"


def write_md(report, path):
    lines = ["# Validator catchability (exploratory reanalysis)", ""]
    lines.append(
        f"Generated {report['generated_utc']}. "
        f"Records analyzed: {report['totals']['records_analyzed']} "
        f"(of {report['totals']['records_in']} in; warmups excluded: "
        f"{report['totals']['warmups_excluded']}; excluded: "
        f"{report['totals']['excluded']}). Zero new calls."
    )
    lines.append("")
    lines.append(f"Validator model: {report['validator_model']}")
    lines.append("")
    lines.append("## Pooled aggregates (validatable tasks)")
    lines.append("")
    lines.append(
        "| study::task::class | cells | n | byte modal | post-validator | "
        "reject | invisible |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for key, agg in sorted(report["aggregates"].items()):
        lines.append(
            f"| {key} | {agg['cells']} | {agg['n']} | "
            f"{_fmt(agg['byte_modal_pooled'])} | "
            f"{_fmt(agg['post_validator_pooled'])} | "
            f"{_fmt(agg['reject_rate_pooled'])} | "
            f"{_fmt(agg['invisible_rate_pooled'])} |"
        )
    lines.append("")
    lines.append("## Cells with validator-relevant variance")
    lines.append("")
    lines.append(
        "| cell | class | n | byte modal | post-validator | reject | "
        "invisible | recovered |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for key, cell in sorted(report["cells"].items()):
        if cell["coverage"] != "validatable":
            continue
        interesting = (
            (cell["byte_modal_share"] or 0) < 1.0
            or (cell["rejected"] or 0) > 0
        )
        if not interesting:
            continue
        escaped = key.replace("|", "\\|")
        lines.append(
            f"| {escaped} | {cell['config_class']} | "
            f"{cell['n']} | {_fmt(cell['byte_modal_share'])} | "
            f"{_fmt(cell['post_validator_repro'])} | {cell['rejected']} | "
            f"{_fmt(cell['invisible_divergence_rate'])} | "
            f"{_fmt(cell.get('recovered'))} |"
        )
    lines.append("")
    lines.append("## Unvalidatable coverage (open generation)")
    lines.append("")
    unval = [
        (key, cell) for key, cell in sorted(report["cells"].items())
        if cell["coverage"] == "unvalidatable"
    ]
    below = [
        (key, cell) for key, cell in unval
        if (cell["byte_modal_share"] or 0) < 1.0
    ]
    lines.append(
        f"{len(unval)} open-generation cells carry no deterministic "
        f"validator; {len(below)} are below byte ceiling — that instability "
        f"is invisible to any output gate by construction."
    )
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Validator catchability reanalysis (exploratory)"
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

    report = build_validator_report(records)
    report["inputs"] = sorted(args.paths)
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(args.out, f"validator-catchability-{stamp}.json")
    md_path = os.path.join(args.out, f"validator-catchability-{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    write_md(report, md_path)

    print(
        f"records_analyzed={report['totals']['records_analyzed']} "
        f"cells={len(report['cells'])}"
    )
    for key, agg in sorted(report["aggregates"].items()):
        print(
            f"  {key}: n={agg['n']} byte={_fmt(agg['byte_modal_pooled'])} "
            f"post={_fmt(agg['post_validator_pooled'])} "
            f"reject={_fmt(agg['reject_rate_pooled'])} "
            f"invisible={_fmt(agg['invisible_rate_pooled'])}"
        )
    print(f"report -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
