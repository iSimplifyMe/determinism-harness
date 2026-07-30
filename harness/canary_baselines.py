"""Canary baselines: computed from the committed study-2 confirmatory runs.

The daily canary (design doc 2026-07-30, owner-approved) compares each
probe run against per-cell baselines derived from public confirmatory
data — fence rates, semantic goldens, the known byte-variant corpus,
label distributions, response-model identities, and latency medians.
Deterministic code, stdlib only; the emitted canary/baselines.json is
committed, so every alarm threshold is itself checkable.

Inputs are the study-2 confirmatory full windows (non-streamed,
deterministic-config cells only; positive controls, streamed arms, and
padded-extraction cells are excluded).

Usage:
  python3 -m harness.canary_baselines \
      runs/low-study2-full-*.jsonl runs/peak-study2-full-*.jsonl \
      --out canary/baselines.json
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

from analysis.latency_fingerprint import percentile
from analysis.semantic_sj import parse_sj
from analysis.stats import binom_pmf

CANARY_TASKS = ("structured_json", "extraction", "classification")
CANARY_ARMS = ("adaptive", "none")


def binom_band(n, p, alpha=0.05):
    """Exact central binomial acceptance band [lo, hi]: each tail holds at
    most alpha/2 probability under Binomial(n, p)."""
    lo = 0
    cumulative = 0.0
    for k in range(n + 1):
        cumulative += binom_pmf(k, n, p)
        if cumulative <= alpha / 2:
            lo = k + 1
    hi = n
    cumulative = 0.0
    for k in range(n, -1, -1):
        cumulative += binom_pmf(k, n, p)
        if cumulative <= alpha / 2:
            hi = k - 1
    if hi < lo:
        hi = lo
    return lo, hi


def _eligible(record):
    if not record.get("ok") or record.get("stop_reason") != "end_turn":
        return False
    if record.get("meta_control") == "positive":
        return False
    if record.get("meta_task") not in CANARY_TASKS:
        return False
    if record.get("meta_thinking") not in CANARY_ARMS:
        return False
    if record.get("cell", "").endswith("|streamed"):
        return False
    return True


def build_baselines(records):
    grouped = defaultdict(list)
    response_models = defaultdict(set)
    extraction_texts = Counter()
    for record in records:
        if not _eligible(record):
            continue
        grouped[record["cell"]].append(record)
        if record.get("response_model"):
            response_models[record["meta_model"]].add(record["response_model"])
        if record["meta_task"] == "extraction":
            extraction_texts[record["text"].strip()] += 1

    cells = {}
    for cell in sorted(grouped):
        recs = grouped[cell]
        task = recs[0]["meta_task"]
        entry = {
            "n": len(recs),
            "task": task,
            "latency_ms_p50": percentile(
                [r["latency_ms"] for r in recs if r.get("latency_ms")], 50
            ),
        }
        if task == "structured_json":
            modes = Counter()
            goldens = Counter()
            corpus = set()
            for r in recs:
                mode, sem = parse_sj(r["text"])
                modes[mode] += 1
                corpus.add(r["text_sha256"])
                if sem is not None:
                    goldens[sem] += 1
            parsed = modes["strict"] + modes["fenced"]
            entry.update({
                "fence_rate": modes["fenced"] / parsed if parsed else 0.0,
                "parse_fail_count": modes["fail"],
                "sj_semantic_goldens": sorted(goldens),
                "byte_sha_corpus": sorted(corpus),
            })
        elif task == "classification":
            labels = Counter(r["text"].strip() for r in recs)
            golden, _count = labels.most_common(1)[0]
            entry.update({
                "label_counts": dict(labels),
                "golden_label": golden,
                "flip_rate": 1.0 - labels[golden] / len(recs),
            })
        cells[cell] = entry

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "extraction_golden": (
            extraction_texts.most_common(1)[0][0] if extraction_texts else None
        ),
        "response_models": {
            model: sorted(ids) for model, ids in response_models.items()
        },
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute canary baselines from study-2 confirmatory runs"
    )
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default="canary/baselines.json")
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

    baselines = build_baselines(records)
    baselines["generated_from"] = sorted(args.paths)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(baselines, fh, indent=2, sort_keys=True)

    print(
        f"baselines: {len(baselines['cells'])} cells, extraction golden "
        f"{baselines['extraction_golden']!r} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
