"""One codex-sub dry batch (PREREGISTRATION-v4 freeze checklist).

Exercises the registered batch shape end to end on the live door:
per-batch effort receipts first, then STUDY4_CODEX_BATCH_SIZE measured
calls (structured_json, effort none) through CodexDoor. Verifies the
subscription rate window tolerates the registered batch size and records
timing for the confirmatory batch plan. Exploratory; no confirmatory
role (the frozen grid re-runs these cells properly windowed).

    python3 -m scripts.codex_dry_batch
"""
import json
import sys
import time
from datetime import datetime, timezone

from harness.config import STUDY4_CODEX_BATCH_SIZE, STUDY4_DOORS
from harness.doors import make_door
from harness.request_builder import codex_argv
from harness.tasks import TASKS

TASK = "structured_json"
EFFORT = "none"


def main():
    model = STUDY4_DOORS["codex_sub"]["model_id"]
    door = make_door("codex_sub")
    started = datetime.now(timezone.utc).isoformat()

    receipts = {arm: door.receipt(model, arm) for arm in ("none", "high")}
    receipt_ok = all(
        receipts[arm].get("reasoning_effort") == arm for arm in receipts
    )
    print(f"receipts ok={receipt_ok}")

    prompt = TASKS[TASK]["prompt"]
    argv = codex_argv(model, prompt, EFFORT, door.workdir)
    records, failures = [], 0
    batch_start = time.monotonic()
    for i in range(1, STUDY4_CODEX_BATCH_SIZE + 1):
        record = door.invoke(argv)
        record["repeat"] = i
        records.append(record)
        if not record["ok"]:
            failures += 1
            print(f"  call {i}: FAIL {record.get('error_code')} "
                  f"retryable={record.get('retryable')}")
            if not record.get("retryable"):
                break
        if i % 10 == 0:
            elapsed = time.monotonic() - batch_start
            print(f"  {i}/{STUDY4_CODEX_BATCH_SIZE} "
                  f"({elapsed:.0f}s, {elapsed / i:.1f}s/call)")
    wall_s = time.monotonic() - batch_start

    summary = {
        "started_utc": started,
        "batch_size": STUDY4_CODEX_BATCH_SIZE,
        "completed": len(records),
        "failures": failures,
        "wall_seconds": round(wall_s, 1),
        "seconds_per_call": round(wall_s / max(1, len(records)), 2),
        "receipts": receipts,
        "receipt_ok": receipt_ok,
        "distinct_texts": len({r.get("text_sha256") for r in records
                               if r.get("ok")}),
    }
    with open("evidence/codex-dry-batch-20260817.jsonl", "w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    with open("evidence/codex-dry-batch-20260817.summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if (receipt_ok and failures == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
