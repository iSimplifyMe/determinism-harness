"""Grid runner.

Controls implemented here, per the spec and PREREGISTRATION.md:
- Ordering control: the full (cell, repeat) schedule is shuffled with a
  recorded seed, and no two calls from the same cell are ever in flight
  concurrently, so repeats of a cell cannot self-batch server-side.
- Negative control: request bodies are built once per cell as canonical
  bytes; the SHA-256 recorded per call is the hash of exactly the bytes sent.
- Version-drift control: the model ID returned in each response body is
  recorded, as is the AWS request ID for every call.
- No silent retries: boto3's automatic retry layer is disabled; every attempt
  is counted and recorded. Throttles back off exponentially; terminal errors
  are recorded as failures, never dropped.

Study 2 (cross-plane, PREREGISTRATION-v2): the study2-* modes schedule the
model x task x plane x thinking grid with per-plane payloads and dispatch
through harness.planes. Records carry schema 2 and add `plane` plus
`wire_sha256` (hash of the bytes actually sent — on Bedrock identical to
`request_sha256` by construction). Credentials are run-scoped process env:
ANTHROPIC_AWS_WORKSPACE_ID for p_aws, ANTHROPIC_API_KEY for anthropic_api.

Usage:
  python3 -m harness.runner --mode pilot --window pilot --dry-run
  python3 -m harness.runner --mode full --window peak
  python3 -m harness.runner --mode positive-control --window control
  python3 -m harness.runner --mode effort-sweep --window control
  python3 -m harness.runner --mode study2-pilot --window pilot --dry-run
  python3 -m harness.runner --mode study2-full --window peak
  python3 -m harness.runner --mode study2-positive-control --window control
"""
import argparse
import json
import os
import random
import subprocess
import threading
import time
from datetime import datetime, timezone

from harness.config import (
    EFFORT_SWEEP,
    MODELS,
    PLANES,
    POSITIVE_CONTROL,
    Q3_STREAMING,
    Q4_LENGTHS,
    REGION,
    REPEATS_FULL,
    REPEATS_PILOT,
    WINDOWS,
    cell_key,
    cell_key2,
    grid_cells,
    grid_cells_study2,
    plane_model_id,
)
from harness.planes import BEDROCK_RETRYABLE_CODES, make_plane
from harness.request_builder import (
    canonical_body,
    canonical_bytes,
    canonical_messages_params,
    sha256_hex,
)
from harness.tasks import TASKS, padded_prompt

# Single source of truth lives in harness.planes so the study-1 inline path
# and BedrockPlane can never drift apart.
RETRYABLE_CODES = BEDROCK_RETRYABLE_CODES

STUDY2_MODES = (
    "study2-pilot",
    "study2-full",
    "study2-positive-control",
    "study2-q3-streaming",
    "study2-q4-lengths",
)
MODES = ("pilot", "full", "positive-control", "effort-sweep") + STUDY2_MODES


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def worker_seed(seed, worker_index):
    """Integer per-worker RNG seed. random.Random rejects tuples on
    Python 3.11+ — the 2026-07-27 pilot died on exactly that."""
    return (seed * 1000003 + worker_index * 7919 + 17) & 0x7FFFFFFFFFFFFFFF


def summary_is_complete(summary, expected):
    """A run is complete only if every scheduled call produced a record and
    no worker died. 'failures' are recorded per-call outcomes and do not
    make a run incomplete; silent shortfall does."""
    return (
        summary.get("done") == expected
        and not summary.get("fatal_worker_errors")
    )


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _item(cell_id, meta, body, model_id, repeat):
    return {
        "cell": cell_id,
        "meta": meta,
        "body": body,
        "sha": sha256_hex(body),
        "model_id": model_id,
        "repeat": repeat,
    }


def _item2(cell_id, meta, plane, payload, sha, model_id, repeat):
    return {
        "cell": cell_id,
        "meta": meta,
        "plane": plane,
        "payload": payload,
        "sha": sha,
        "model_id": model_id,
        "repeat": repeat,
    }


def _study2_payload(model_cfg, plane, model_id, prompt, thinking, extra=None):
    """Per-plane payload + planned-request hash. Bedrock: canonical bytes,
    hashed == sent. Messages planes: the params dict; its hash is of the
    canonical serialization, and the wire hash is captured at send time."""
    if plane == "bedrock":
        payload = canonical_body(model_cfg, prompt, thinking, extra=extra)
        return payload, sha256_hex(payload)
    payload = canonical_messages_params(
        model_cfg, model_id, prompt, thinking, extra=extra
    )
    return payload, sha256_hex(canonical_bytes(payload))


def build_schedule(mode):
    """Deterministic, unshuffled schedule for a mode."""
    items = []
    if mode in ("pilot", "full"):
        repeats = REPEATS_PILOT if mode == "pilot" else REPEATS_FULL
        for cell in grid_cells():
            mcfg = MODELS[cell["model"]]
            body = canonical_body(mcfg, TASKS[cell["task"]]["prompt"], cell["thinking"])
            cid = cell_key(cell)
            model_id = mcfg["profiles"][cell["profile"]]
            for r in range(repeats):
                items.append(_item(cid, dict(cell), body, model_id, r))
    elif mode == "positive-control":
        pc = POSITIVE_CONTROL
        mcfg = MODELS[pc["model"]]
        body = canonical_body(
            mcfg, TASKS[pc["task"]]["prompt"], pc["thinking"], extra=pc["extra"]
        )
        meta = {
            "model": pc["model"],
            "task": pc["task"],
            "profile": pc["profile"],
            "thinking": pc["thinking"],
            "control": "positive",
        }
        cid = cell_key(meta) + "|temp=0.7"
        model_id = mcfg["profiles"][pc["profile"]]
        for r in range(pc["repeats"]):
            items.append(_item(cid, dict(meta), body, model_id, r))
    elif mode == "effort-sweep":
        es = EFFORT_SWEEP
        base = MODELS[es["model"]]
        model_id = base["profiles"][es["profile"]]
        for task_key in es["tasks"]:
            for effort in es["efforts"]:
                mcfg = dict(base)
                mcfg["effort"] = effort
                body = canonical_body(mcfg, TASKS[task_key]["prompt"], es["thinking"])
                meta = {
                    "model": es["model"],
                    "task": task_key,
                    "profile": es["profile"],
                    "thinking": es["thinking"],
                    "effort": effort,
                }
                cid = (
                    f'{es["model"]}|{task_key}|{es["profile"]}|{es["thinking"]}'
                    f"|effort={effort}"
                )
                for r in range(es["repeats"]):
                    items.append(_item(cid, meta, body, model_id, r))
    elif mode in ("study2-pilot", "study2-full"):
        repeats = REPEATS_PILOT if mode == "study2-pilot" else REPEATS_FULL
        for cell in grid_cells_study2():
            mcfg = MODELS[cell["model"]]
            model_id = plane_model_id(cell["plane"], cell["model"])
            payload, sha = _study2_payload(
                mcfg,
                cell["plane"],
                model_id,
                TASKS[cell["task"]]["prompt"],
                cell["thinking"],
            )
            cid = cell_key2(cell)
            for r in range(repeats):
                items.append(
                    _item2(cid, dict(cell), cell["plane"], payload, sha, model_id, r)
                )
    elif mode == "study2-positive-control":
        pc = POSITIVE_CONTROL
        mcfg = MODELS[pc["model"]]
        for plane in PLANES:
            model_id = plane_model_id(plane, pc["model"])
            payload, sha = _study2_payload(
                mcfg,
                plane,
                model_id,
                TASKS[pc["task"]]["prompt"],
                pc["thinking"],
                extra=pc["extra"],
            )
            meta = {
                "model": pc["model"],
                "task": pc["task"],
                "plane": plane,
                "thinking": pc["thinking"],
                "control": "positive",
            }
            cid = cell_key2(meta) + "|temp=0.7"
            for r in range(pc["repeats"]):
                items.append(_item2(cid, dict(meta), plane, payload, sha, model_id, r))
    elif mode == "study2-q3-streaming":
        q3 = Q3_STREAMING
        for model_key in q3["models"]:
            mcfg = MODELS[model_key]
            for task_key in q3["tasks"]:
                for plane in PLANES:
                    model_id = plane_model_id(plane, model_key)
                    payload, sha = _study2_payload(
                        mcfg, plane, model_id, TASKS[task_key]["prompt"], q3["thinking"]
                    )
                    meta = {
                        "model": model_key,
                        "task": task_key,
                        "plane": plane,
                        "thinking": q3["thinking"],
                        "delivery": "streaming",
                    }
                    cid = f'{model_key}|{task_key}|{plane}|{q3["thinking"]}|streamed'
                    for r in range(q3["repeats"]):
                        item = _item2(
                            cid, dict(meta), plane, payload, sha, model_id, r
                        )
                        item["delivery"] = "streaming"
                        items.append(item)
    elif mode == "study2-q4-lengths":
        q4 = Q4_LENGTHS
        for model_key in q4["models"]:
            mcfg = MODELS[model_key]
            for label in q4["labels"]:
                prompt = padded_prompt(label, TASKS["extraction"]["prompt"])
                for plane in PLANES:
                    model_id = plane_model_id(plane, model_key)
                    payload, sha = _study2_payload(
                        mcfg, plane, model_id, prompt, q4["thinking"]
                    )
                    task_label = f"extraction_pad_{label}"
                    meta = {
                        "model": model_key,
                        "task": task_label,
                        "plane": plane,
                        "thinking": q4["thinking"],
                        "pad": label,
                    }
                    cid = f'{model_key}|{task_label}|{plane}|{q4["thinking"]}'
                    for r in range(q4["repeats"]):
                        items.append(
                            _item2(cid, dict(meta), plane, payload, sha, model_id, r)
                        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    return items


def git_head():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def schedule_digest(items):
    joined = "\n".join(f'{it["cell"]}#{it["repeat"]}' for it in items)
    return sha256_hex(joined.encode("utf-8"))


class Engine:
    def __init__(
        self,
        items,
        out_path,
        concurrency,
        seed,
        run_info=None,
        max_attempts=6,
        plane_clients=None,
    ):
        self.items = items
        self.run_info = run_info or {}
        self.claimed = [False] * len(items)
        self.in_flight = set()
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.done = 0
        self.retries = 0
        self.failures = 0
        self.fatal = []
        self.concurrency = concurrency
        self.seed = seed
        self.max_attempts = max_attempts
        self.out = open(out_path, "a", encoding="utf-8")

        needed = {it.get("plane") for it in items}
        self.planes = dict(plane_clients or {})
        for plane_name in sorted(p for p in needed if p):
            if plane_name not in self.planes:
                self.planes[plane_name] = make_plane(plane_name)

        self.client = None
        if None in needed:  # study-1 items use the legacy inline Bedrock path
            import boto3
            from botocore.config import Config

            self.client = boto3.client(
                "bedrock-runtime",
                region_name=REGION,
                config=Config(
                    read_timeout=600, connect_timeout=10, retries={"max_attempts": 0}
                ),
            )

    def _next_index(self):
        with self.lock:
            if all(self.claimed):
                return None
            for idx, taken in enumerate(self.claimed):
                if taken:
                    continue
                if self.items[idx]["cell"] in self.in_flight:
                    continue
                self.claimed[idx] = True
                self.in_flight.add(self.items[idx]["cell"])
                return idx
            return -1  # all remaining items blocked by an in-flight sibling

    def _finish(self, item, failed):
        with self.lock:
            self.in_flight.discard(item["cell"])
            self.done += 1
            if failed:
                self.failures += 1
            done, retries, failures = self.done, self.retries, self.failures
        if done % 25 == 0 or done == len(self.items):
            print(
                f"[{utc_now_iso()}] {done}/{len(self.items)} done, "
                f"retries={retries}, failures={failures}",
                flush=True,
            )

    def _write(self, record):
        with self.write_lock:
            self.out.write(json.dumps(record, sort_keys=True) + "\n")
            self.out.flush()

    def _execute(self, item, rng, schedule_index):
        from botocore.exceptions import (
            ClientError,
            ConnectionClosedError,
            EndpointConnectionError,
            ReadTimeoutError,
        )

        base = {
            "schema": 1,
            **self.run_info,
            "schedule_index": schedule_index,
            "cell": item["cell"],
            "repeat": item["repeat"],
            "model_id_sent": item["model_id"],
            "request_sha256": item["sha"],
            **{f"meta_{k}": v for k, v in item["meta"].items()},
        }
        attempts = 0
        last_error = ("unknown", "no attempt made")
        while attempts < self.max_attempts:
            attempts += 1
            sent_at = utc_now_iso()
            start = time.monotonic()
            try:
                resp = self.client.invoke_model(
                    modelId=item["model_id"],
                    body=item["body"],
                    contentType="application/json",
                    accept="application/json",
                )
                latency_ms = int((time.monotonic() - start) * 1000)
                payload = json.loads(resp["body"].read())
                text = "".join(
                    block.get("text", "")
                    for block in payload.get("content", [])
                    if block.get("type") == "text"
                )
                meta = resp.get("ResponseMetadata", {})
                headers = meta.get("HTTPHeaders") or {}
                return {
                    **base,
                    "ok": True,
                    "attempts": attempts,
                    "sent_at_utc": sent_at,
                    "received_at_utc": utc_now_iso(),
                    "latency_ms": latency_ms,
                    "aws_request_id": meta.get("RequestId"),
                    "amzn_headers": {
                        k: v for k, v in headers.items() if k.startswith("x-amzn")
                    },
                    "response_id": payload.get("id"),
                    "response_model": payload.get("model"),
                    "stop_reason": payload.get("stop_reason"),
                    "usage": payload.get("usage"),
                    "text": text,
                    "text_sha256": sha256_hex(text.encode("utf-8")),
                }
            except ClientError as err:
                code = err.response.get("Error", {}).get("Code", "ClientError")
                message = str(err)[:400]
                last_error = (code, message)
                if code in RETRYABLE_CODES and attempts < self.max_attempts:
                    with self.lock:
                        self.retries += 1
                    time.sleep(min(60.0, 2.0 ** attempts) + rng.uniform(0, 1))
                    continue
                break
            except (
                EndpointConnectionError,
                ReadTimeoutError,
                ConnectionClosedError,
            ) as err:
                last_error = (type(err).__name__, str(err)[:400])
                if attempts < self.max_attempts:
                    with self.lock:
                        self.retries += 1
                    time.sleep(min(60.0, 2.0 ** attempts) + rng.uniform(0, 1))
                    continue
                break
        return {
            **base,
            "ok": False,
            "attempts": attempts,
            "sent_at_utc": utc_now_iso(),
            "error_code": last_error[0],
            "error_message": last_error[1],
        }

    def _execute_plane(self, item, rng, schedule_index):
        """Study-2 execution: dispatch through harness.planes; retry on the
        record's own retryable classification (Bedrock keeps study 1's
        code-name semantics inside BedrockPlane)."""
        plane = self.planes[item["plane"]]
        base = {
            "schema": 2,
            **self.run_info,
            "schedule_index": schedule_index,
            "cell": item["cell"],
            "repeat": item["repeat"],
            "plane": item["plane"],
            "model_id_sent": item["model_id"],
            "request_sha256": item["sha"],
            **{f"meta_{k}": v for k, v in item["meta"].items()},
        }
        attempts = 0
        result = {"ok": False, "error_code": "unknown", "error_message": "no attempt made", "retryable": False}
        while attempts < self.max_attempts:
            attempts += 1
            sent_at = utc_now_iso()
            streaming = item.get("delivery") == "streaming"
            if item["plane"] == "bedrock":
                result = plane.invoke(
                    item["model_id"], item["payload"], stream=streaming
                )
            else:
                result = plane.invoke(item["payload"], stream=streaming)
            if result["ok"]:
                return {
                    **base,
                    **result,
                    "attempts": attempts,
                    "sent_at_utc": sent_at,
                    "received_at_utc": utc_now_iso(),
                }
            if result["retryable"] and attempts < self.max_attempts:
                with self.lock:
                    self.retries += 1
                time.sleep(min(60.0, 2.0 ** attempts) + rng.uniform(0, 1))
                continue
            break
        return {
            **base,
            **result,
            "attempts": attempts,
            "sent_at_utc": utc_now_iso(),
        }

    def _worker(self, worker_index):
        try:
            rng = random.Random(worker_seed(self.seed, worker_index))
            while True:
                idx = self._next_index()
                if idx is None:
                    return
                if idx == -1:
                    time.sleep(0.05)
                    continue
                item = self.items[idx]
                time.sleep(rng.uniform(0.25, 1.0))  # anti-burst jitter
                if item.get("plane"):
                    record = self._execute_plane(item, rng, idx)
                else:
                    record = self._execute(item, rng, idx)
                self._write(record)
                self._finish(item, failed=not record["ok"])
        except Exception as err:  # engine bug: record loudly, let peers drain
            detail = f"worker {worker_index}: {type(err).__name__}: {err}"
            print(f"FATAL {detail}", flush=True)
            with self.lock:
                self.fatal.append(detail)

    def run(self):
        threads = [
            threading.Thread(target=self._worker, args=(i,), daemon=True)
            for i in range(self.concurrency)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.out.close()
        return {
            "done": self.done,
            "expected": len(self.items),
            "retries": self.retries,
            "failures": self.failures,
            "fatal_worker_errors": self.fatal,
        }


def main():
    parser = argparse.ArgumentParser(description="Bedrock determinism grid runner")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--window", required=True, choices=sorted(WINDOWS.keys()))
    parser.add_argument("--out", default="runs")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    schedule = build_schedule(args.mode)
    rng = random.Random(args.seed)
    rng.shuffle(schedule)

    cells = sorted({it["cell"] for it in schedule})
    stamp = utc_stamp()
    run_name = f"{args.window}-{args.mode}-{stamp}"
    os.makedirs(args.out, exist_ok=True)

    manifest = {
        "run_name": run_name,
        "mode": args.mode,
        "window": args.window,
        "window_definition": WINDOWS[args.window],
        "seed": args.seed,
        "concurrency": args.concurrency,
        "region": REGION,
        "n_items": len(schedule),
        "n_cells": len(cells),
        "cells": cells,
        "schedule_sha256": schedule_digest(schedule),
        "request_sha_by_cell": sorted(
            {(it["cell"], it["sha"]) for it in schedule}
        ),
        "git_head": git_head(),
        "created_utc": utc_now_iso(),
        "dry_run": bool(args.dry_run),
    }
    if args.mode in STUDY2_MODES:
        manifest["schema"] = 2
        manifest["planes"] = sorted({it["plane"] for it in schedule})

    if args.dry_run:
        manifest_path = os.path.join(args.out, f"{run_name}.dryrun.manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        print(f"DRY RUN: {len(schedule)} calls across {len(cells)} cells")
        print(f"schedule sha256: {manifest['schedule_sha256']}")
        print(f"manifest written: {manifest_path}")
        return 0

    if args.mode in STUDY2_MODES:
        planes_present = {it["plane"] for it in schedule}
        missing = []
        if "p_aws" in planes_present and not os.environ.get(
            "ANTHROPIC_AWS_WORKSPACE_ID"
        ):
            missing.append("ANTHROPIC_AWS_WORKSPACE_ID (Claude Platform on AWS)")
        if "anthropic_api" in planes_present and not os.environ.get(
            "ANTHROPIC_API_KEY"
        ):
            missing.append("ANTHROPIC_API_KEY (first-party plane, run-scoped)")
        if missing:
            for name in missing:
                print(f"MISSING CREDENTIAL: {name}", flush=True)
            return 3

    import boto3

    try:
        ident = boto3.client("sts").get_caller_identity()
        manifest["aws_account"] = ident.get("Account")
        manifest["caller_arn"] = ident.get("Arn")
    except Exception as err:  # provenance is best-effort, never blocking
        manifest["aws_account"] = None
        manifest["identity_error"] = str(err)[:200]

    manifest_path = os.path.join(args.out, f"{run_name}.manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    out_path = os.path.join(args.out, f"{run_name}.jsonl")
    print(f"run: {run_name}  items: {len(schedule)}  cells: {len(cells)}")
    print(f"records -> {out_path}")
    engine = Engine(
        schedule,
        out_path,
        args.concurrency,
        args.seed,
        run_info={"window": args.window, "mode": args.mode, "run_name": run_name},
    )
    summary = engine.run()

    complete = summary_is_complete(summary, len(schedule))
    summary_path = os.path.join(args.out, f"{run_name}.done.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                **summary,
                "complete": complete,
                "finished_utc": utc_now_iso(),
                "run_name": run_name,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    print(f"finished: {summary}")
    if not complete:
        print(
            f"INCOMPLETE RUN: done={summary['done']} of {len(schedule)} "
            f"expected, fatal={summary['fatal_worker_errors']}",
            flush=True,
        )
        return 2
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
