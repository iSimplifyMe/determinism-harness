# Run Protocol

Operational runbook. The scientific commitments live in PREREGISTRATION.md;
this file is how a window actually gets executed.

## Prerequisites

- AWS credentials for account 024033896674 (us-east-1), Bedrock model access
  granted for all three study models (verified 2026-07-27; `evidence/smoke.json`).
- Python 3.10+ and boto3 (`pip install -r requirements.txt`). The analysis
  pipeline and test suite are stdlib-only.
- Run everything from the repository root.

## Standing order of operations

```
0. python3 -m unittest discover -s tests -t .        # must be green
1. python3 -m harness.verify_profiles                # must exit 0; refreshes evidence
2. python3 -m harness.smoke                          # must print all_passed=True
3. Pilot (pre-freeze):
     python3 -m harness.runner --mode pilot --window pilot
     python3 -m analysis.analyze runs/pilot-*.jsonl --out reports
     -> finalize n / delta in PREREGISTRATION.md, bump to v1.0, tag prereg-v1
4. Confirmatory windows (post-freeze only), one at a time, compressed:
     python3 -m harness.runner --mode full --window low
     python3 -m harness.runner --mode full --window mid
     python3 -m harness.runner --mode full --window peak
5. Positive control (same day as a confirmatory window):
     python3 -m harness.runner --mode positive-control --window control
6. Follow-on effort sweep (after the main grid):
     python3 -m harness.runner --mode effort-sweep --window control
7. Analysis over everything confirmatory:
     python3 -m analysis.analyze runs/low-full-*.jsonl runs/mid-full-*.jsonl \
         runs/peak-full-*.jsonl runs/control-positive-control-*.jsonl --out reports
8. Commit runs/ manifests + JSONL + reports/ (raw data is part of the artifact).
```

## Window definitions (UTC)

| Window | UTC range | Rationale |
|---|---|---|
| low | 07:00–10:00 | US night |
| mid | 00:00–03:00 | US evening |
| peak | 15:00–19:00 | US business morning/midday |

Start a window's run inside its range and let it finish; runs are sized to
complete within the range at the default concurrency. Never run two windows
concurrently. Keep each run compressed — the shorter the run, the smaller the
chance an undetectable point-version roll lands inside it (the 5-family IDs
are undated; PREREGISTRATION.md section 6).

## Runner behavior worth knowing

- `--dry-run` builds the schedule and manifest without touching AWS.
- Default `--concurrency 4`; the scheduler never lets two calls from the same
  cell be in flight at once regardless of concurrency.
- boto3 auto-retries are disabled; the runner does its own bounded backoff and
  records the attempt count on every record. Throttling shows up in the
  progress line as `retries=`.
- Every run writes `<window>-<mode>-<stamp>.manifest.json` (seed, schedule
  hash, per-cell request hashes, git commit) before the first call, then the
  JSONL, then `<...>.done.json` with totals.
- A run that ends with `failures>0` exits nonzero; the failed calls are in the
  JSONL with error codes. Do not re-run individual calls; re-run the window or
  accept the counted exclusions.

## Cost and duration expectations

4,000 calls per full window at concurrency 4 with 0.25–1.0 s jitter and mostly
short outputs: expect roughly 1.5–3 hours per window, dominated by the
open-generation and adaptive-thinking cells. The pilot (800 calls) is a
sub-hour run. Rate-limit contention from other workloads in the account slows
runs but does not bias them; avoid scheduling next to heavy production bursts.
