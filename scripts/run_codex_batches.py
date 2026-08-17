"""Drive the study4-codex schedule in registered batches (v4 sections 2, 7).

The codex door runs ONE window's worth (4 tasks x 2 arms x n=100 = 800
calls), batched under the subscription's rate windows. Each invocation:

1. rebuilds the full study4-codex schedule and shuffles it with the SAME
   seed the runner uses, so the order is identical across invocations;
2. scans prior batch records in --out and skips every (cell, repeat)
   already attempted — failed calls are counted exclusions, never re-run
   (PROTOCOL.md);
3. runs the per-batch effort receipts (v4 section 3) and ABORTS if any
   banner disagrees with its pinned arm;
4. executes the next --batch-size items through the Engine (concurrency 1,
   registered retry bound), writing a normal manifest + JSONL per batch.

Run repeatedly until it prints SCHEDULE COMPLETE:
    python3 -m scripts.run_codex_batches [--batch-size 40] [--out runs]
"""
import argparse
import glob
import json
import os
import random
import sys

from harness.config import (
    STUDY4_CODEX_BATCH_SIZE,
    STUDY4_DOORS,
    STUDY4_RETRY_MAX_ATTEMPTS,
)
from harness.doors import make_door
from harness.runner import (
    Engine,
    WINDOWS,
    build_schedule,
    git_head,
    schedule_digest,
    utc_now_iso,
    utc_stamp,
)

MODE = "study4-codex"
WINDOW = "control"  # codex is registered outside the HTTP window factor
DEFAULT_SEED = 20260727  # the runner's default; MUST match across batches


def completed_pairs(out_dir):
    """(cell, repeat) pairs already attempted in prior batch files."""
    pairs = set()
    pattern = os.path.join(out_dir, f"*-{MODE}-*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pairs.add((rec["cell"], rec["repeat"]))
    return pairs


def remaining_items(schedule, done):
    return [it for it in schedule if (it["cell"], it["repeat"]) not in done]


def main():
    parser = argparse.ArgumentParser(description="study4-codex batch driver")
    parser.add_argument("--batch-size", type=int,
                        default=STUDY4_CODEX_BATCH_SIZE)
    parser.add_argument("--out", default="runs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    schedule = build_schedule(MODE)
    random.Random(args.seed).shuffle(schedule)
    total = len(schedule)

    os.makedirs(args.out, exist_ok=True)
    done = completed_pairs(args.out)
    todo = remaining_items(schedule, done)
    if not todo:
        print(f"SCHEDULE COMPLETE: {len(done)}/{total} attempted")
        return 0
    batch = todo[: args.batch_size]
    batch_index = len(done) // args.batch_size + 1
    print(
        f"batch {batch_index}: {len(batch)} items "
        f"({len(done)}/{total} already attempted, {len(todo)} remaining)"
    )
    if args.dry_run:
        for it in batch[:5]:
            print(f"  {it['cell']}#{it['repeat']}")
        print("DRY RUN: no calls made")
        return 0

    door = make_door("codex_sub")
    model = STUDY4_DOORS["codex_sub"]["model_id"]
    arms = sorted({it["meta"]["effort"] for it in batch})
    receipts = {}
    for arm in arms:
        receipts[arm] = door.receipt(model, arm)
        if receipts[arm].get("reasoning_effort") != arm:
            print(f"RECEIPT MISMATCH for arm {arm!r}: {receipts[arm]}")
            print("ABORTING BATCH — no measured calls made")
            return 3

    stamp = utc_stamp()
    run_name = f"{WINDOW}-{MODE}-b{batch_index:02d}-{stamp}"
    manifest = {
        "run_name": run_name,
        "mode": MODE,
        "window": WINDOW,
        "window_definition": WINDOWS[WINDOW],
        "schema": 4,
        "doors": ["codex_sub"],
        "seed": args.seed,
        "concurrency": 1,
        "batch_index": batch_index,
        "batch_size": args.batch_size,
        "prior_attempted": len(done),
        "schedule_total": total,
        "n_items": len(batch),
        "n_cells": len({it["cell"] for it in batch}),
        "cells": sorted({it["cell"] for it in batch}),
        "schedule_sha256_full": schedule_digest(schedule),
        "schedule_sha256_batch": schedule_digest(batch),
        "request_sha_by_cell": sorted(
            {(it["cell"], it["sha"]) for it in batch}
        ),
        "effort_receipts": receipts,
        "retry_max_attempts": STUDY4_RETRY_MAX_ATTEMPTS,
        "git_head": git_head(),
        "created_utc": utc_now_iso(),
    }
    manifest_path = os.path.join(args.out, f"{run_name}.manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    out_path = os.path.join(args.out, f"{run_name}.jsonl")
    print(f"records -> {out_path}")
    engine = Engine(
        batch,
        out_path,
        concurrency=1,
        seed=args.seed,
        run_info={"window": WINDOW, "mode": MODE, "run_name": run_name},
        max_attempts=STUDY4_RETRY_MAX_ATTEMPTS,
        door_clients={"codex_sub": door},
    )
    summary = engine.run()
    summary_path = os.path.join(args.out, f"{run_name}.done.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {**summary, "finished_utc": utc_now_iso(),
             "run_name": run_name,
             "attempted_after_batch": len(done) + summary["done"],
             "schedule_total": total},
            fh, indent=2, sort_keys=True,
        )
    attempted = len(done) + summary["done"]
    print(f"finished: {summary}")
    print(f"progress: {attempted}/{total} attempted")
    if summary["done"] != len(batch) or summary["fatal_worker_errors"]:
        return 2
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
