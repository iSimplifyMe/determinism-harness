"""Analysis pipeline: runs/*.jsonl -> report.json + report.md.

Deterministic code only — no model anywhere in the measurement loop. The
functional core (gate_cell, analyze_records, pool_success, comparisons) is
unit-tested; main() is thin file IO. This script is committed before any
confirmatory data exists, per PREREGISTRATION.md.

Grouping: records are grouped per (window, cell) so the same grid cell
observed in different load windows is never collapsed. Records without a
window field (unit tests, ad hoc replays) group by bare cell key.
"""
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from analysis.metrics import cell_metrics, pop_variance
from analysis.stats import diff_ci, two_prop_tost, wilson_interval

POSITIVE_CONTROL_MARKER = "|temp=0.7"


def gate_cell(records):
    """Apply validity gates to one cell's raw records.

    - Mixed request hashes: the harness introduced variance — the whole
      cell is invalid (negative control failure).
    - Errored calls and non-end_turn stops are excluded and counted.
    - More than one returned model ID among valid calls flags version
      drift; records stay valid and the drift is reported.
    """
    excluded = {"error": 0, "truncated_or_other_stop": 0}
    flags = {
        "negative_control_failed": False,
        "model_drift": False,
        "response_models": [],
    }
    hashes = {r.get("request_sha256") for r in records}
    if len(hashes) > 1:
        flags["negative_control_failed"] = True
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
        if record.get("stop_reason") != "end_turn":
            excluded["truncated_or_other_stop"] += 1
            continue
        valid.append(record)
    models = sorted(
        {r.get("response_model") for r in valid if r.get("response_model")}
    )
    flags["response_models"] = models
    flags["model_drift"] = len(models) > 1
    return {"valid": valid, "excluded": excluded, "flags": flags, "n_raw": len(records)}


def _meta_of(record):
    meta = {k[5:]: v for k, v in record.items() if k.startswith("meta_")}
    if record.get("window"):
        meta["window"] = record["window"]
    return meta


def _group_key(record):
    window = record.get("window")
    return f'{window}::{record["cell"]}' if window else record["cell"]


def analyze_records(records):
    """Gate and measure every (window, cell) group."""
    grouped = defaultdict(list)
    for record in records:
        grouped[_group_key(record)].append(record)

    cells = {}
    for key in sorted(grouped):
        recs = grouped[key]
        gate = gate_cell(recs)
        entry = {
            "meta": _meta_of(recs[0]),
            "cell": recs[0]["cell"],
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

            thinking = [
                ((r.get("usage") or {}).get("output_tokens_details") or {}).get(
                    "thinking_tokens"
                )
                for r in valid
            ]
            thinking = [t for t in thinking if t is not None]
            entry["thinking_tokens_mean"] = (
                sum(thinking) / len(thinking) if thinking else None
            )
            entry["thinking_tokens_variance"] = (
                pop_variance(thinking) if thinking else None
            )

            entry["cache_tokens_total"] = sum(
                ((r.get("usage") or {}).get("cache_read_input_tokens") or 0)
                + ((r.get("usage") or {}).get("cache_creation_input_tokens") or 0)
                for r in valid
            )
            entry["service_tiers"] = sorted(
                {
                    (r.get("usage") or {}).get("service_tier")
                    for r in valid
                    if (r.get("usage") or {}).get("service_tier")
                }
            )
        cells[key] = entry
    return {"cells": cells}


def pool_success(cells, meta_filter):
    """Pool (successes, n) across cells whose meta passes the filter.

    A call is a success when its text matches its own cell's modal
    response — reproduction is always measured within-cell.
    """
    x = n = 0
    for entry in cells.values():
        metrics = entry.get("metrics")
        if not metrics:
            continue
        if entry["cell"].endswith(POSITIVE_CONTROL_MARKER):
            continue
        if not meta_filter(entry.get("meta", {})):
            continue
        x += metrics["modal_count"]
        n += metrics["n"]
    return x, n


def comparisons(cells, delta=0.01):
    """Pre-registered pooled contrasts: Q2 (us vs global), Q3 (adaptive vs
    disabled), Q4 (per-window rates)."""
    out = {}
    models = sorted(
        {
            e["meta"].get("model")
            for e in cells.values()
            if e.get("meta", {}).get("model")
        }
    )
    for model in models:
        x_us, n_us = pool_success(
            cells,
            lambda m, mo=model: m.get("model") == mo and m.get("profile") == "us",
        )
        x_gl, n_gl = pool_success(
            cells,
            lambda m, mo=model: m.get("model") == mo and m.get("profile") == "global",
        )
        if n_us and n_gl:
            out[f"q2_profile__{model}"] = {
                "us": {"x": x_us, "n": n_us, "rate": x_us / n_us},
                "global": {"x": x_gl, "n": n_gl, "rate": x_gl / n_gl},
                "tost": two_prop_tost(x_us, n_us, x_gl, n_gl, delta),
                "diff_ci95": diff_ci(x_us, n_us, x_gl, n_gl),
            }
        x_ad, n_ad = pool_success(
            cells,
            lambda m, mo=model: m.get("model") == mo
            and m.get("thinking") == "adaptive",
        )
        x_di, n_di = pool_success(
            cells,
            lambda m, mo=model: m.get("model") == mo
            and m.get("thinking") == "disabled",
        )
        if n_ad and n_di:
            out[f"q3_thinking__{model}"] = {
                "adaptive": {"x": x_ad, "n": n_ad, "rate": x_ad / n_ad},
                "disabled": {"x": x_di, "n": n_di, "rate": x_di / n_di},
                "diff_ci95": diff_ci(x_ad, n_ad, x_di, n_di),
            }

    windows = sorted(
        {
            e["meta"].get("window")
            for e in cells.values()
            if e.get("meta", {}).get("window")
        }
    )
    for window in windows:
        x_w, n_w = pool_success(
            cells, lambda m, w=window: m.get("window") == w
        )
        if n_w:
            out[f"q4_window__{window}"] = {
                "x": x_w,
                "n": n_w,
                "rate": x_w / n_w,
                "wilson_ci": wilson_interval(x_w, n_w),
            }
    return out


def controls_summary(cells):
    negative_failures = [
        key
        for key, e in cells.items()
        if e["gate"]["flags"]["negative_control_failed"]
    ]
    drift = [key for key, e in cells.items() if e["gate"]["flags"]["model_drift"]]
    cache_nonzero = [
        key for key, e in cells.items() if e.get("cache_tokens_total")
    ]
    positive = {}
    for key, e in cells.items():
        if e["cell"].endswith(POSITIVE_CONTROL_MARKER) and e.get("metrics"):
            m = e["metrics"]
            threshold = max(2, -(-m["n"] // 10))  # ceil(n/10), min 2
            positive[key] = {
                "n": m["n"],
                "distinct_count": m["distinct_count"],
                "modal_share": m["modal_share"],
                "fired": m["distinct_count"] >= threshold,
            }
    return {
        "negative_control_failed_cells": negative_failures,
        "model_drift_cells": drift,
        "cache_tokens_nonzero_cells": cache_nonzero,
        "positive_control": positive,
    }


def _md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def render_markdown(result, generated_at):
    cells = result["cells"]
    parts = [
        "# Determinism harness report",
        f"Generated {generated_at} by analysis/analyze.py (deterministic; no model in the loop).",
        "## Controls",
        "```json",
        json.dumps(result["controls"], indent=2, sort_keys=True),
        "```",
        "## Per-cell results",
    ]
    rows = []
    for key, e in sorted(cells.items()):
        m = e.get("metrics")
        if not m:
            rows.append([key, e["gate"]["n_raw"], "-", "-", "-", "-", "INVALID"])
            continue
        lo, hi = e["wilson_ci"]
        flags = []
        if e["gate"]["flags"]["model_drift"]:
            flags.append("DRIFT")
        if e["gate"]["excluded"]["error"] or e["gate"]["excluded"][
            "truncated_or_other_stop"
        ]:
            flags.append(
                f'excl={e["gate"]["excluded"]["error"]}+{e["gate"]["excluded"]["truncated_or_other_stop"]}'
            )
        rows.append(
            [
                key,
                m["n"],
                f'{m["modal_share"]:.4f}',
                f"[{lo:.4f}, {hi:.4f}]",
                m["distinct_count"],
                f'{m["norm_distance_mean_all"]:.4f}',
                ";".join(flags) or "ok",
            ]
        )
    parts.append(
        _md_table(
            ["window::cell", "n", "modal share", "wilson 95%", "distinct", "mean norm dist", "flags"],
            rows,
        )
    )
    parts.append("## Pre-registered comparisons")
    parts.append("```json")
    parts.append(json.dumps(result["comparisons"], indent=2, sort_keys=True))
    parts.append("```")
    return "\n\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Analyze determinism run records")
    parser.add_argument("paths", nargs="+", help="runs/*.jsonl files")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--delta", type=float, default=0.02)  # prereg v1.0, section 5
    args = parser.parse_args()

    records = []
    for path in args.paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if not records:
        print("ERROR: zero records loaded — refusing to write an empty report")
        return 2

    result = analyze_records(records)
    result["comparisons"] = comparisons(result["cells"], delta=args.delta)
    result["controls"] = controls_summary(result["cells"])
    result["inputs"] = sorted(args.paths)
    result["delta"] = args.delta

    generated_at = datetime.now(timezone.utc).isoformat()
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(args.out, f"report-{stamp}.json")
    md_path = os.path.join(args.out, f"report-{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(result, generated_at))

    print(f"records={len(records)} groups={len(result['cells'])}")
    print(f"report -> {md_path}")
    print(f"report -> {json_path}")
    bad = result["controls"]["negative_control_failed_cells"]
    if bad:
        print(f"NEGATIVE CONTROL FAILED in {len(bad)} cell(s): {bad}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
