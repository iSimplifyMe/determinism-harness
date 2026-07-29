"""Semantic parse-identity addendum for structured-JSON cells.

Byte-distinct structured-JSON responses may still parse to the same object —
study 1 found exactly that on the 5-family (cosmetic serialization variance,
semantic identity). This deterministic addendum measures it for study-2
records: per sj cell, the number of distinct parsed objects under canonical
serialization, the parse mode each response needed (strict, or after
stripping a markdown fence the prompt explicitly forbade), and outright
failures. Stdlib only; no model in the loop.

Usage:
  python3 -m analysis.semantic_sj runs/low-*.jsonl runs/peak-*.jsonl \
      runs/control-study2-q3-streaming-*.jsonl --out reports
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

FENCE_RE = re.compile(r"\A\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*\Z", re.DOTALL)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def parse_sj(text):
    """Classify one response: (mode, canonical_json_or_None).

    strict  — the raw text is valid JSON (the instructed format)
    fenced  — valid JSON only after stripping a surrounding markdown fence
    fail    — neither
    """
    try:
        return "strict", canonical(json.loads(text))
    except (ValueError, TypeError):
        pass
    match = FENCE_RE.match(text)
    if match:
        try:
            return "fenced", canonical(json.loads(match.group(1)))
        except (ValueError, TypeError):
            pass
    return "fail", None


def group_key(record):
    window = record.get("window")
    return f'{window}::{record["cell"]}' if window else record["cell"]


def analyze_semantic(records):
    groups = defaultdict(list)
    for record in records:
        task = record.get("meta_task", "")
        if task != "structured_json":
            continue
        if not record.get("ok") or record.get("stop_reason") != "end_turn":
            continue
        groups[group_key(record)].append(record)

    cells = {}
    for key in sorted(groups):
        recs = groups[key]
        modes = Counter()
        semantics = Counter()
        byte_variants = set()
        for record in recs:
            mode, sem = parse_sj(record["text"])
            modes[mode] += 1
            byte_variants.add(record["text_sha256"])
            if sem is not None:
                semantics[sem] += 1
        entry = {
            "n": len(recs),
            "byte_distinct": len(byte_variants),
            "semantic_distinct": len(semantics),
            "modes": dict(modes),
            "byte_modal_share": None,
            "semantic_modal_share": (
                max(semantics.values()) / sum(semantics.values())
                if semantics
                else None
            ),
        }
        if entry["semantic_distinct"] > 1 and entry["semantic_distinct"] <= 6:
            entry["semantic_variants"] = [
                {"object": json.loads(sem), "count": count}
                for sem, count in semantics.most_common()
            ]
        cells[key] = entry
    return cells


def main():
    parser = argparse.ArgumentParser(
        description="Semantic parse-identity addendum for structured JSON"
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

    cells = analyze_semantic(records)
    if not cells:
        print("ERROR: no structured_json records found")
        return 2

    total = sum(e["n"] for e in cells.values())
    fails = sum(e["modes"].get("fail", 0) for e in cells.values())
    fenced = sum(e["modes"].get("fenced", 0) for e in cells.values())
    semantic_unstable = {
        k: e["semantic_distinct"] for k, e in cells.items()
        if e["semantic_distinct"] > 1
    }
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": sorted(args.paths),
        "totals": {
            "sj_responses": total,
            "parse_fail": fails,
            "fenced": fenced,
            "cells": len(cells),
            "cells_semantically_unstable": len(semantic_unstable),
        },
        "cells": cells,
    }

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.out, f"semantic-sj-{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    print(f"sj responses={total} cells={len(cells)} fenced={fenced} fail={fails}")
    for key, entry in sorted(cells.items()):
        marker = "" if entry["semantic_distinct"] == 1 else "  << SEMANTIC VARIANCE"
        print(
            f'  {key}: bytes={entry["byte_distinct"]} '
            f'semantic={entry["semantic_distinct"]} modes={entry["modes"]}{marker}'
        )
    print(f"addendum -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
