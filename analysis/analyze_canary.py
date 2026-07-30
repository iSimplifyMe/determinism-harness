"""Canary run evaluation against committed baselines.

Alarm semantics (design doc 2026-07-30):
- RED    something a consumer would page on: transport failures, a wrong
         extraction value, an off-set classification label, a structured
         object outside the baseline golden set, a parse/schema failure,
         or a response-model identity change (the dated-Haiku anchor and
         the 5-family name constancy).
- YELLOW distribution drift worth a look, not a page: fence rate outside
         the exact binomial band for the probe's n, a novel byte-variant
         that still parses to a golden object, a valid-but-nonmodal label
         on a cell whose baseline never flips, or a probed cell with no
         baseline.
- GREEN  everything within baseline behavior.

Haiku label flips are baseline behavior (flip_rate > 0 in the baselines),
so they record without alarming; a flip on a cell with baseline
flip_rate == 0 is YELLOW. Latency medians are recorded per cell for the
log's trend history; no latency alarm fires in v1.

Exit codes: 0 GREEN · 1 YELLOW · 2 RED (the daily script keys off this).

Usage:
  python3 -m analysis.analyze_canary runs/canary-canary-*.jsonl \
      --baselines canary/baselines.json --out canary/log
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

from analysis.latency_fingerprint import percentile
from analysis.semantic_sj import parse_sj
from analysis.validator_catchability import (
    CLASSIFICATION_LABELS,
    _sj_schema_ok,
)
from harness.canary_baselines import binom_band


def evaluate_canary(records, baselines):
    red = []
    yellow = []
    grouped = defaultdict(list)
    observed_models = defaultdict(set)
    transport_failures = 0
    for record in records:
        if not record.get("ok") or record.get("stop_reason") != "end_turn":
            transport_failures += 1
            continue
        grouped[record["cell"]].append(record)
        if record.get("response_model"):
            observed_models[record["meta_model"]].add(
                record["response_model"]
            )
    if transport_failures:
        red.append({
            "check": "transport",
            "detail": f"{transport_failures} failed/abnormal calls",
        })

    for model, observed in sorted(observed_models.items()):
        allowed = set(baselines.get("response_models", {}).get(model, []))
        unexpected = sorted(observed - allowed)
        if unexpected:
            red.append({
                "check": f"response_model:{model}",
                "detail": f"unexpected ids {unexpected} (allowed {sorted(allowed)})",
            })

    extraction_golden = baselines.get("extraction_golden")
    cells_out = {}
    for cell in sorted(grouped):
        recs = grouped[cell]
        task = recs[0].get("meta_task")
        baseline = baselines.get("cells", {}).get(cell)
        cell_out = {
            "n": len(recs),
            "latency_ms_p50": percentile(
                [r["latency_ms"] for r in recs if r.get("latency_ms")], 50
            ) if any(r.get("latency_ms") for r in recs) else None,
        }
        if baseline is None:
            yellow.append({
                "check": f"no_baseline:{cell}",
                "detail": "probed cell has no committed baseline",
            })
        if task == "extraction":
            wrong = [
                r["text"].strip() for r in recs
                if r["text"].strip() != extraction_golden
            ]
            cell_out["wrong_values"] = wrong
            if wrong:
                red.append({
                    "check": f"extraction:{cell}",
                    "detail": f"{len(wrong)} values != golden: {wrong[:3]}",
                })
        elif task == "classification":
            labels = Counter(r["text"].strip() for r in recs)
            cell_out["labels"] = dict(labels)
            off_set = sorted(
                label for label in labels
                if label not in CLASSIFICATION_LABELS
            )
            if off_set:
                red.append({
                    "check": f"classification_format:{cell}",
                    "detail": f"off-set outputs {off_set}",
                })
            golden = (baseline or {}).get("golden_label")
            baseline_flip = (baseline or {}).get("flip_rate", 0.0)
            flips = sum(
                count for label, count in labels.items()
                if label in CLASSIFICATION_LABELS and label != golden
            )
            cell_out["flips"] = flips
            if golden and flips and baseline_flip == 0.0:
                yellow.append({
                    "check": f"label_flip:{cell}",
                    "detail": (
                        f"{flips} valid-but-nonmodal labels on a cell whose "
                        f"baseline never flips"
                    ),
                })
        elif task == "structured_json":
            goldens = set((baseline or {}).get("sj_semantic_goldens", []))
            corpus = set((baseline or {}).get("byte_sha_corpus", []))
            fenced = 0
            parsed = 0
            novel = []
            for r in recs:
                mode, sem = parse_sj(r["text"])
                if sem is None:
                    red.append({
                        "check": f"sj_parse:{cell}",
                        "detail": "response failed to parse",
                    })
                    continue
                parsed += 1
                if mode == "fenced":
                    fenced += 1
                if not _sj_schema_ok(json.loads(sem)):
                    red.append({
                        "check": f"sj_schema:{cell}",
                        "detail": "parsed object violates the schema",
                    })
                    continue
                if goldens and sem not in goldens:
                    red.append({
                        "check": f"sj_semantic:{cell}",
                        "detail": "object outside the baseline golden set",
                    })
                elif corpus and r["text_sha256"] not in corpus:
                    novel.append(r["text_sha256"])
            cell_out["fenced"] = fenced
            cell_out["parsed"] = parsed
            if novel:
                yellow.append({
                    "check": f"novel_variant:{cell}",
                    "detail": (
                        f"{len(novel)} byte-variants outside the corpus "
                        f"(golden semantics): {sorted(set(novel))[:2]}"
                    ),
                })
            if baseline is not None and parsed:
                lo, hi = binom_band(parsed, baseline.get("fence_rate", 0.0))
                cell_out["fence_band"] = [lo, hi]
                if not lo <= fenced <= hi:
                    yellow.append({
                        "check": f"fence_rate:{cell}",
                        "detail": (
                            f"fenced {fenced}/{parsed} outside baseline band "
                            f"[{lo}, {hi}] (p={baseline.get('fence_rate'):.3f})"
                        ),
                    })
        cells_out[cell] = cell_out

    status = "RED" if red else ("YELLOW" if yellow else "GREEN")
    return {
        "status": status,
        "red": red,
        "yellow": yellow,
        "totals": {
            "records_in": len(records),
            "transport_failures": transport_failures,
            "cells_probed": len(grouped),
        },
        "cells": cells_out,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a canary run against committed baselines"
    )
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--baselines", default="canary/baselines.json")
    parser.add_argument("--out", default="canary/log")
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
    with open(args.baselines, encoding="utf-8") as fh:
        baselines = json.load(fh)

    result = evaluate_canary(records, baselines)
    result["generated_utc"] = datetime.now(timezone.utc).isoformat()
    result["inputs"] = sorted(args.paths)
    result["baselines_generated_utc"] = baselines.get("generated_utc")

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.out, f"canary-{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    print(
        f"CANARY {result['status']} — red={len(result['red'])} "
        f"yellow={len(result['yellow'])} over "
        f"{result['totals']['records_in']} calls "
        f"({result['totals']['transport_failures']} transport failures)"
    )
    for item in result["red"]:
        print(f"  RED    {item['check']}: {item['detail']}")
    for item in result["yellow"]:
        print(f"  YELLOW {item['check']}: {item['detail']}")
    print(f"log -> {out_path}")
    return {"GREEN": 0, "YELLOW": 1, "RED": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
