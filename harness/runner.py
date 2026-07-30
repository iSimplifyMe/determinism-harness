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
    CHURN_AB,
    EFFORT_SWEEP,
    GPT_OSS_120B,
    LOCAL_KEEP_ALIVE,
    LOCAL_MODELS,
    LOCAL_SAMPLING,
    LOCAL_SEED,
    MARGINS_BATTERY,
    MARGINS_LOGPROB_FIELDS,
    MODELS,
    PLANES,
    POSITIVE_CONTROL,
    Q2_LOCAL_CONCURRENCY,
    Q3_STREAMING,
    Q4_LENGTHS,
    REGION,
    REPEATS_FULL,
    REPEATS_PILOT,
    REPEATS_STUDY3_PILOT,
    WINDOWS,
    cell_key,
    cell_key2,
    cell_key3,
    grid_cells,
    grid_cells_study2,
    grid_cells_study3,
    local_model_cfg,
    local_models_for_box,
    local_on_arm,
    local_pinned_arm,
    plane_model_id,
)
from harness.logprob_capture import compact_margins
from harness.planes import BEDROCK_RETRYABLE_CODES, make_plane
from harness.request_builder import (
    canonical_body,
    canonical_bytes,
    canonical_local_body,
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
# Follow-up companion modes (FOLLOWUP-COMPANIONS.md): exploratory,
# plan-committed-pre-data. Their schedules are FIXED — the schedule IS the
# manipulation (churn) or the eviction-ordering constraint (margins) — so
# main() must not shuffle, re-block, or auto-prepend warmups.
COMPANION_MODES = ("study3-churn-ab", "study3-margins")
FIXED_SCHEDULE_MODES = COMPANION_MODES

STUDY3_MODES = (
    "study3-pilot",
    "study3-full",
    "study3-q3-thinking",
    "study3-q2-concurrency",
    "study3-120b-window",
) + COMPANION_MODES
MODES = (
    ("pilot", "full", "positive-control", "effort-sweep")
    + STUDY2_MODES
    + STUDY3_MODES
)


def study3_run_settings(mode):
    """Execution constraints per study-3 mode. Q1's registered condition is
    single-flight, so the core modes force concurrency 1; the Q2 arm runs
    same-cell parallel at the registered level. None for non-study3 modes."""
    if mode not in STUDY3_MODES:
        return None
    if mode == "study3-q2-concurrency":
        return {
            "concurrency": Q2_LOCAL_CONCURRENCY["level"],
            "allow_same_cell_concurrency": True,
        }
    return {"concurrency": 1, "allow_same_cell_concurrency": False}


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


def _study3_item(cell, cid, repeat, cfg=None):
    """Local-plane item: canonical bytes are the wire bytes (exact negative
    control), keep_alive pinned so residency is part of the frozen request.
    `cfg` overrides the LOCAL_MODELS lookup for models that live outside the
    core roster (the dedicated-window 120b arm)."""
    cfg = cfg or LOCAL_MODELS[cell["model"]]
    body = canonical_local_body(
        cfg["tag"],
        TASKS[cell["task"]]["prompt"],
        cell["thinking"],
        options=LOCAL_SAMPLING[cell["sampling"]],
        keep_alive=LOCAL_KEEP_ALIVE,
    )
    return _item2(
        cid, dict(cell), "local", body, sha256_hex(body), cfg["tag"], repeat
    )


def _study3_q3_cells(box):
    for model_key, cfg in local_models_for_box(box).items():
        if not cfg.get("q3_eligible", True):
            continue  # struck at freeze (prereg v3 s1) — non-hybrid model
        yield {
            "model": model_key,
            "task": "structured_json",
            "sampling": "greedy",
            "thinking": local_on_arm(cfg),
            "hardware": box,
        }


def apply_model_blocks(schedule, seed):
    """Sort a shuffled study-3 schedule into contiguous per-model blocks.

    Swap thrash dominated the metal pilot (92 min for 364 calls on an
    over-budget box), so confirmatory study-3 runs execute one model at a
    time. The stable sort preserves the shuffled within-model order — the
    per-cell ordering control is unchanged, since model was never a
    within-cell factor — and floats each model's prepended warmup to its
    block head. Block order is itself seed-derived."""
    models = sorted({it["model_id"] for it in schedule})
    rng = random.Random(worker_seed(seed, 999))
    order = {m: i for i, m in enumerate(rng.sample(models, len(models)))}
    schedule.sort(key=lambda it: order[it["model_id"]])
    return schedule


def build_warmup_items(items):
    """One tiny recorded call per distinct local model in the schedule,
    prepended (never shuffled) so every grid cell runs against a warm model
    (prereg v3 Q1). Records carry meta control=warmup; analysis excludes
    them like pilot data."""
    warmups = {}
    for it in items:
        if it.get("plane") != "local" or it["model_id"] in warmups:
            continue
        body = canonical_local_body(
            it["model_id"],
            "warmup",
            "none",
            options={"temperature": 0, "seed": LOCAL_SEED, "num_predict": 8},
            keep_alive=LOCAL_KEEP_ALIVE,
        )
        meta = {
            "model": it["meta"]["model"],
            "control": "warmup",
            "hardware": it["meta"].get("hardware"),
        }
        warmups[it["model_id"]] = _item2(
            f'warmup|{it["meta"]["model"]}',
            meta,
            "local",
            body,
            sha256_hex(body),
            it["model_id"],
            0,
        )
    return list(warmups.values())


def _companion_warmup(model_tag, model_key, box):
    """One tiny recorded warmup at a companion block head — same body shape
    as build_warmup_items, emitted inline because companion schedules are
    fixed and never pass through apply_model_blocks."""
    body = canonical_local_body(
        model_tag,
        "warmup",
        "none",
        options={"temperature": 0, "seed": LOCAL_SEED, "num_predict": 8},
        keep_alive=LOCAL_KEEP_ALIVE,
    )
    meta = {"model": model_key, "control": "warmup", "hardware": box}
    return _item2(
        f"warmup|{model_key}", meta, "local", body, sha256_hex(body),
        model_tag, 0,
    )


def build_schedule(mode, box=None, repeats=None):
    """Deterministic, unshuffled schedule for a mode. Study-3 modes require
    `box` (one run targets one box); `repeats` overrides the mode default
    (pilot-sizing of arms — the override is recorded in the manifest)."""
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
    elif mode in ("study3-pilot", "study3-full"):
        if box is None:
            raise ValueError("study3 modes require a box (metal/cuda)")
        n = repeats or (
            REPEATS_STUDY3_PILOT if mode == "study3-pilot" else REPEATS_FULL
        )
        cells = list(grid_cells_study3(box))
        if mode == "study3-pilot":  # pilot exercises the Q3 arm too
            cells += list(_study3_q3_cells(box))
        for cell in cells:
            cid = cell_key3(cell)
            for r in range(n):
                items.append(_study3_item(cell, cid, r))
    elif mode == "study3-q3-thinking":
        if box is None:
            raise ValueError("study3 modes require a box (metal/cuda)")
        n = repeats or REPEATS_FULL
        for cell in _study3_q3_cells(box):
            cid = cell_key3(cell)
            for r in range(n):
                items.append(_study3_item(cell, cid, r))
    elif mode == "study3-120b-window":
        cfg = GPT_OSS_120B
        if box != cfg["box"]:
            raise ValueError(
                f'study3-120b-window runs on {cfg["box"]} only (got {box})'
            )
        n = repeats or REPEATS_FULL
        cells = [
            {
                "model": cfg["key"],
                "task": task_key,
                "sampling": sampling,
                "thinking": cfg["thinking_arms"][0],
                "hardware": box,
            }
            for task_key in TASKS
            for sampling in sorted(LOCAL_SAMPLING)
        ]
        cells.append({
            "model": cfg["key"],
            "task": "structured_json",
            "sampling": "greedy",
            "thinking": cfg["thinking_arms"][1],
            "hardware": box,
        })
        for cell in cells:
            cid = cell_key3(cell)
            for r in range(n):
                items.append(_study3_item(cell, cid, r, cfg=cfg))
    elif mode == "study3-q2-concurrency":
        q2 = Q2_LOCAL_CONCURRENCY
        if box != q2["box"]:
            raise ValueError(
                f'study3-q2-concurrency runs on {q2["box"]} only (got {box})'
            )
        n = repeats or REPEATS_FULL
        for model_key in q2["models"]:
            cfg = LOCAL_MODELS[model_key]
            for task_key in TASKS:
                cell = {
                    "model": model_key,
                    "task": task_key,
                    "sampling": "greedy",
                    "thinking": cfg["thinking_arms"][0],
                    "hardware": box,
                    "concurrency": q2["level"],
                }
                cid = cell_key3(cell) + f'|c{q2["level"]}'
                for r in range(n):
                    items.append(_study3_item(cell, cid, r))
    elif mode == "study3-churn-ab":
        if box not in CHURN_AB["boxes"]:
            raise ValueError(
                f'study3-churn-ab runs on {CHURN_AB["boxes"]} (got {box})'
            )
        n = repeats or CHURN_AB["n_per_arm"]
        cfg = LOCAL_MODELS[CHURN_AB["model"]]
        core = {
            "model": CHURN_AB["model"],
            "task": CHURN_AB["task"],
            "sampling": CHURN_AB["sampling"],
            "thinking": local_pinned_arm(cfg),
            "hardware": box,
        }
        items.append(_companion_warmup(cfg["tag"], core["model"], box))
        # Alternating mini-blocks (B,C,B,C,...) for time balance; the
        # measured bodies are byte-identical across arms — the churn
        # manipulation is the pre_unload flag, executed out-of-band.
        block = CHURN_AB["mini_block"]
        counts = {"blocked": 0, "churn": 0}
        position = 0
        while counts["blocked"] < n or counts["churn"] < n:
            arm = "blocked" if position % 2 == 0 else "churn"
            position += 1
            take = min(block, n - counts[arm])
            if take <= 0:
                continue
            for _ in range(take):
                cell = dict(core, arm=arm)
                item = _study3_item(cell, cell_key3(core) + f"|arm={arm}",
                                    counts[arm])
                if arm == "churn":
                    item["pre_unload"] = True
                items.append(item)
                counts[arm] += 1
    elif mode == "study3-margins":
        if box not in MARGINS_BATTERY:
            raise ValueError(
                f"study3-margins runs on {sorted(MARGINS_BATTERY)} (got {box})"
            )
        last_model = None
        for spec in MARGINS_BATTERY[box]:
            cfg = local_model_cfg(spec["model"])
            if spec["model"] != last_model:
                items.append(
                    _companion_warmup(cfg["tag"], spec["model"], box)
                )
                last_model = spec["model"]
            thinking = spec.get("thinking") or cfg["thinking_arms"][0]
            cell = {
                "model": spec["model"],
                "task": spec["task"],
                "sampling": "greedy",
                "thinking": thinking,
                "hardware": box,
                "exploratory": "margins",
            }
            cid = cell_key3(cell) + "|logprobs"
            body = canonical_local_body(
                cfg["tag"],
                TASKS[spec["task"]]["prompt"],
                thinking,
                options=LOCAL_SAMPLING["greedy"],
                keep_alive=LOCAL_KEEP_ALIVE,
                extra=MARGINS_LOGPROB_FIELDS,
            )
            sha = sha256_hex(body)
            for r in range(repeats or spec["n"]):
                item = _item2(cid, dict(cell), "local", body, sha,
                              cfg["tag"], r)
                item["capture_logprobs"] = True
                items.append(item)
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
        allow_same_cell_concurrency=False,
    ):
        self.items = items
        self.run_info = run_info or {}
        self.claimed = [False] * len(items)
        self.in_flight = set()
        # Ordering control default: no two calls from one cell in flight.
        # The study-3 Q2 arm inverts this deliberately — same-cell parallel
        # load IS the manipulation (prereg v3 section 1).
        self.allow_same_cell_concurrency = allow_same_cell_concurrency
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
                if (
                    not self.allow_same_cell_concurrency
                    and self.items[idx]["cell"] in self.in_flight
                ):
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
        unload_info = None
        while attempts < self.max_attempts:
            attempts += 1
            # Companion-A churn arm: unload + confirm absence before EVERY
            # attempt (a retried attempt must still be a cold-load call).
            if item.get("pre_unload") and hasattr(plane, "unload"):
                unload_info = plane.unload(item["model_id"])
            sent_at = utc_now_iso()
            streaming = item.get("delivery") == "streaming"
            if item["plane"] == "bedrock":
                result = plane.invoke(
                    item["model_id"], item["payload"], stream=streaming
                )
            else:
                result = plane.invoke(item["payload"], stream=streaming)
            if result["ok"]:
                record = {
                    **base,
                    **result,
                    "attempts": attempts,
                    "sent_at_utc": sent_at,
                    "received_at_utc": utc_now_iso(),
                }
                if unload_info is not None:
                    record["pre_unload_confirmed"] = unload_info["unloaded"]
                    record["unload_wait_ms"] = unload_info["wait_ms"]
                if item.get("capture_logprobs"):
                    margins = compact_margins(
                        getattr(plane, "last_payload", None)
                    )
                    if margins is not None:
                        record["logprob_margins"] = margins
                return record
            if result["retryable"] and attempts < self.max_attempts:
                with self.lock:
                    self.retries += 1
                time.sleep(min(60.0, 2.0 ** attempts) + rng.uniform(0, 1))
                continue
            break
        record = {
            **base,
            **result,
            "attempts": attempts,
            "sent_at_utc": utc_now_iso(),
        }
        if unload_info is not None:
            record["pre_unload_confirmed"] = unload_info["unloaded"]
            record["unload_wait_ms"] = unload_info["wait_ms"]
        return record

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
    parser.add_argument("--box", choices=("metal", "cuda"),
                        help="study3 modes: which hardware arm this run is")
    parser.add_argument("--local-url", default="http://127.0.0.1:11434",
                        help="study3 modes: Ollama base URL for the box")
    parser.add_argument("--repeats", type=int,
                        help="study3 modes: override the mode's repeats "
                             "(recorded in the manifest)")
    args = parser.parse_args()

    schedule = build_schedule(args.mode, box=args.box, repeats=args.repeats)
    fixed = args.mode in FIXED_SCHEDULE_MODES
    rng = random.Random(args.seed)
    if not fixed:
        rng.shuffle(schedule)

    settings = study3_run_settings(args.mode)
    warmups = []
    if settings:
        if args.concurrency != settings["concurrency"]:
            print(
                f'{args.mode} enforces concurrency='
                f'{settings["concurrency"]} (--concurrency ignored)'
            )
        args.concurrency = settings["concurrency"]
        if fixed:
            # Companion schedules ship their own warmup heads and encode
            # their ordering constraint — count, never reorder.
            warmups = [
                it for it in schedule
                if it["meta"].get("control") == "warmup"
            ]
        else:
            warmups = build_warmup_items(schedule)
            schedule = warmups + schedule  # warm every model before its cells
            schedule = apply_model_blocks(schedule, args.seed)

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
    elif args.mode in STUDY3_MODES:
        manifest["schema"] = 3
        manifest["planes"] = ["local"]
        manifest["box"] = args.box
        manifest["local_url"] = args.local_url
        manifest["repeats_override"] = args.repeats
        manifest["warmup_items"] = len(warmups)
        manifest["run_settings"] = settings
        manifest["schedule_blocking"] = "fixed" if fixed else "per-model"
        manifest["schedule_fixed"] = fixed
        if args.mode in COMPANION_MODES:
            manifest["exploratory"] = True
            manifest["companion_plan"] = "FOLLOWUP-COMPANIONS.md"

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

    local_plane = None
    if args.mode in STUDY3_MODES:
        # Reachability + drift-control capture, fail-fast like credentials:
        # engine version and per-model weights digests go in the manifest.
        local_plane = make_plane(
            "local", base_url=args.local_url, name=f"local_{args.box}"
        )
        try:
            manifest["engine_version"] = local_plane.engine_version()
            manifest["model_digests"] = {
                tag: local_plane.model_digest(tag)
                for tag in sorted({it["model_id"] for it in schedule})
            }
            manifest["box_state_start"] = local_plane.box_state()
        except Exception as err:
            print(f"LOCAL SERVER NOT READY at {args.local_url}: {err}", flush=True)
            return 3
    else:
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
    run_info = {"window": args.window, "mode": args.mode, "run_name": run_name}
    engine_kwargs = {}
    if args.mode in STUDY3_MODES:
        run_info["schema"] = 3
        run_info["box"] = args.box
        engine_kwargs = {
            "plane_clients": {"local": local_plane},
            "allow_same_cell_concurrency": settings["allow_same_cell_concurrency"],
        }
    engine = Engine(
        schedule,
        out_path,
        args.concurrency,
        args.seed,
        run_info=run_info,
        **engine_kwargs,
    )
    summary = engine.run()

    complete = summary_is_complete(summary, len(schedule))
    box_state_end = None
    if local_plane is not None:  # best-effort: the box may have died mid-run
        try:
            box_state_end = local_plane.box_state()
        except Exception as err:
            box_state_end = {"error": str(err)[:200]}
    summary_path = os.path.join(args.out, f"{run_name}.done.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                **summary,
                "complete": complete,
                "finished_utc": utc_now_iso(),
                "run_name": run_name,
                **({"box_state_end": box_state_end} if box_state_end else {}),
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
