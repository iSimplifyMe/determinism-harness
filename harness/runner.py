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
    CACHE_AB,
    CACHE_INSTANCE,
    CACHE_TIMING,
    CANARY,
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
    STUDY4_DOORS,
    STUDY4_EFFORT_ARMS,
    STUDY4_MAX_OUTPUT_TOKENS,
    STUDY4_REPEATS_EXPLORATORY,
    STUDY4_REPEATS_FULL,
    STUDY4_RETRY_MAX_ATTEMPTS,
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
from harness.doors import make_door
from harness.logprob_capture import compact_margins
from harness.planes import BEDROCK_RETRYABLE_CODES, make_plane
from harness.request_builder import (
    canonical_body,
    canonical_bytes,
    canonical_local_body,
    canonical_messages_params,
    canonical_responses_body,
    codex_argv,
    converse_request,
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
    # The daily reproducibility canary reuses the study-2 plane machinery
    # verbatim (schema-2 records, credential fail-fast, per-plane payloads)
    # and exact study-2 cell keys, so baselines look up directly.
    "canary",
)
# Follow-up companion modes (FOLLOWUP-COMPANIONS.md): exploratory,
# plan-committed-pre-data. Their schedules are FIXED — the schedule IS the
# manipulation (churn) or the eviction-ordering constraint (margins) — so
# main() must not shuffle, re-block, or auto-prepend warmups.
COMPANION_MODES = (
    "study3-churn-ab",
    "study3-margins",
    "study3-cache-ab",
    "study3-cache-timing",
    "study3-cache-instance",
)
FIXED_SCHEDULE_MODES = COMPANION_MODES

STUDY3_MODES = (
    "study3-pilot",
    "study3-full",
    "study3-q3-thinking",
    "study3-q2-concurrency",
    "study3-120b-window",
) + COMPANION_MODES
# Study-4 modes (PREREGISTRATION-v4 sections 2 and 7): the HTTP grid runs
# per window; the codex door runs its single window's worth in batches
# (scripts/run_codex_batches.py drives study4-codex slices); Q4/Q5
# exploratory arms share one control-window run.
STUDY4_MODES = (
    "study4-full",
    "study4-codex",
    "study4-q4q5",
)
STUDY4_HTTP_GRID_DOORS = ("openai_1p", "mantle", "runtime_us")
STUDY5_MODES = (
    "study5-pilot-api",
    "study5-full-api",
    "study5-pilot-local",
    "study5-full-local",
)
STUDY5_LOCAL_MODES = ("study5-pilot-local", "study5-full-local")
MODES = (
    ("pilot", "full", "positive-control", "effort-sweep")
    + STUDY2_MODES
    + STUDY3_MODES
    + STUDY4_MODES
    + STUDY5_MODES
)


def study5_run_settings(mode):
    """Study-5 execution constraints. Local runs are single-flight (the
    study-3 registered condition; also avoids swap thrash) — API runs keep
    the CLI concurrency: every paraphrase cell is unique, so the engine's
    no-two-same-cell rule already serializes the only repeated cells (the
    resample arm). None for non-study5 modes."""
    if mode not in STUDY5_MODES:
        return None
    if mode in STUDY5_LOCAL_MODES:
        return {"concurrency": 1}
    return {}


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


def study4_run_settings(mode):
    """Execution constraints per study-4 mode. The codex door is one
    subprocess per call against a subscription rate window — strictly
    sequential; parallel exec bursts are exactly what the registered batch
    plan avoids. HTTP modes keep the default concurrency (the same-cell
    in-flight guard already serializes within a cell)."""
    if mode not in STUDY4_MODES:
        return None
    if mode == "study4-codex":
        return {"concurrency": 1, "allow_same_cell_concurrency": False}
    return None


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
    elif mode == "canary":
        for model_key in CANARY["models"]:
            mcfg = MODELS[model_key]
            arm = "adaptive" if mcfg["family"] == "claude-5" else "none"
            for task_key, n in (
                ("structured_json", CANARY["n_sj"]),
                ("extraction", CANARY["n_extraction"]),
                ("classification", CANARY["n_classification"]),
            ):
                for plane in PLANES:
                    model_id = plane_model_id(plane, model_key)
                    payload, sha = _study2_payload(
                        mcfg, plane, model_id,
                        TASKS[task_key]["prompt"], arm,
                    )
                    cell = {
                        "model": model_key,
                        "task": task_key,
                        "plane": plane,
                        "thinking": arm,
                        "canary": True,
                    }
                    cid = cell_key2(cell)
                    for r in range(n):
                        items.append(
                            _item2(cid, dict(cell), plane, payload, sha,
                                   model_id, r)
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
    elif mode == "study3-cache-ab":
        if box not in CACHE_AB["boxes"]:
            raise ValueError(
                f'study3-cache-ab runs on {CACHE_AB["boxes"]} (got {box})'
            )
        n = repeats or CACHE_AB["n_per_arm"]
        cfg = LOCAL_MODELS[CACHE_AB["model"]]
        core = {
            "model": CACHE_AB["model"],
            "task": CACHE_AB["task"],
            "sampling": CACHE_AB["sampling"],
            "thinking": local_pinned_arm(cfg),
            "hardware": box,
        }
        flusher_body = canonical_local_body(
            cfg["tag"],
            TASKS[CACHE_AB["flusher_task"]]["prompt"],
            local_pinned_arm(cfg),
            options={"temperature": 0, "seed": LOCAL_SEED, "num_predict": 16},
            keep_alive=LOCAL_KEEP_ALIVE,
        )
        items.append(_companion_warmup(cfg["tag"], core["model"], box))
        # Alternating C,W mini-blocks; cold measured calls each follow a
        # flusher; a cold block's last call is a measured open-generation
        # call, so every warm-block call follows an identical prompt.
        block = CACHE_AB["mini_block"]
        counts = {"cold": 0, "warm": 0}
        position = 0
        flusher_index = 0
        while counts["cold"] < n or counts["warm"] < n:
            arm = "cold" if position % 2 == 0 else "warm"
            position += 1
            take = min(block, n - counts[arm])
            if take <= 0:
                continue
            for _ in range(take):
                if arm == "cold":
                    flusher_meta = {
                        "model": core["model"],
                        "control": "flusher",
                        "hardware": box,
                    }
                    items.append(_item2(
                        f'flusher|{core["model"]}', flusher_meta, "local",
                        flusher_body, sha256_hex(flusher_body), cfg["tag"],
                        flusher_index,
                    ))
                    flusher_index += 1
                cell = dict(core, arm=arm)
                items.append(_study3_item(
                    cell, cell_key3(core) + f"|prefill={arm}", counts[arm]
                ))
                counts[arm] += 1
    elif mode == "study3-cache-timing":
        if box not in CACHE_TIMING["boxes"]:
            raise ValueError(
                f'study3-cache-timing runs on {CACHE_TIMING["boxes"]} '
                f"(got {box})"
            )
        n = repeats or CACHE_TIMING["n_per_arm"]
        cfg = LOCAL_MODELS[CACHE_TIMING["model"]]
        core = {
            "model": CACHE_TIMING["model"],
            "task": CACHE_TIMING["task"],
            "sampling": CACHE_TIMING["sampling"],
            "thinking": local_pinned_arm(cfg),
            "hardware": box,
        }
        items.append(_companion_warmup(cfg["tag"], core["model"], box))
        # Burn-in: the warmup's different prompt would force the first
        # adjacent call into the checkpoint state on a technicality, so
        # one excluded measured-body call precedes the first block.
        burnin = _study3_item(
            dict(core, control="burnin"), cell_key3(core) + "|burnin", 0
        )
        burnin["pre_sleep_ms"] = 0
        items.append(burnin)
        # Alternating A,G mini-blocks; timing is the ONLY manipulation:
        # adjacent calls sleep 0 (jitter suppressed), gapped calls sleep
        # gap_ms before invoking. Bodies byte-identical across arms.
        block = CACHE_TIMING["mini_block"]
        counts = {"adjacent": 0, "gapped": 0}
        position = 0
        while counts["adjacent"] < n or counts["gapped"] < n:
            arm = "adjacent" if position % 2 == 0 else "gapped"
            position += 1
            take = min(block, n - counts[arm])
            if take <= 0:
                continue
            for _ in range(take):
                cell = dict(core, arm=arm)
                item = _study3_item(
                    cell, cell_key3(core) + f"|timing={arm}", counts[arm]
                )
                item["pre_sleep_ms"] = (
                    0 if arm == "adjacent" else CACHE_TIMING["gap_ms"]
                )
                items.append(item)
                counts[arm] += 1
    elif mode == "study3-cache-instance":
        if box not in CACHE_INSTANCE["boxes"]:
            raise ValueError(
                f'study3-cache-instance runs on {CACHE_INSTANCE["boxes"]} '
                f"(got {box})"
            )
        cycles = CACHE_INSTANCE["n_cycles"]
        n = repeats or CACHE_INSTANCE["n_per_arm_per_cycle"]
        cfg = LOCAL_MODELS[CACHE_INSTANCE["model"]]
        core = {
            "model": CACHE_INSTANCE["model"],
            "task": CACHE_INSTANCE["task"],
            "sampling": CACHE_INSTANCE["sampling"],
            "thinking": local_pinned_arm(cfg),
            "hardware": box,
        }
        flusher_body = canonical_local_body(
            cfg["tag"],
            TASKS[CACHE_INSTANCE["flusher_task"]]["prompt"],
            local_pinned_arm(cfg),
            options={"temperature": 0, "seed": LOCAL_SEED, "num_predict": 16},
            keep_alive=LOCAL_KEEP_ALIVE,
        )
        # NO warmup anywhere: a warmup's different prompt is itself the
        # checkpoint trigger under test. Each cycle: unload-reset rides
        # the burn-in item (measured body, fresh load, excluded) -> arm P
        # -> ONE different-prompt flusher -> arm C. pre_sleep 0 throughout
        # (companion E falsified timing; jitter stays suppressed so
        # instance history is the only variable).
        for cycle in range(cycles):
            burnin = _study3_item(
                dict(core, control="burnin", cycle=cycle),
                cell_key3(core) + "|instance=burnin", cycle,
            )
            burnin["pre_unload"] = True
            burnin["pre_sleep_ms"] = 0
            items.append(burnin)
            for r in range(n):
                item = _study3_item(
                    dict(core, arm="pure", cycle=cycle),
                    cell_key3(core) + "|instance=pure", cycle * n + r,
                )
                item["pre_sleep_ms"] = 0
                items.append(item)
            flusher_meta = {
                "model": core["model"],
                "control": "flusher",
                "hardware": box,
                "cycle": cycle,
            }
            flusher = _item2(
                f'flusher|{core["model"]}', flusher_meta, "local",
                flusher_body, sha256_hex(flusher_body), cfg["tag"], cycle,
            )
            flusher["pre_sleep_ms"] = 0
            items.append(flusher)
            for r in range(n):
                item = _study3_item(
                    dict(core, arm="contaminated", cycle=cycle),
                    cell_key3(core) + "|instance=contaminated",
                    cycle * n + r,
                )
                item["pre_sleep_ms"] = 0
                items.append(item)
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
    elif mode == "study4-full":
        n = repeats or STUDY4_REPEATS_FULL
        for door_key in STUDY4_HTTP_GRID_DOORS:
            for task_key in TASKS:
                for effort in STUDY4_EFFORT_ARMS:
                    for r in range(n):
                        items.append(_study4_item(door_key, task_key, effort, r))
    elif mode == "study4-codex":
        n = repeats or STUDY4_REPEATS_FULL
        for task_key in TASKS:
            for effort in STUDY4_EFFORT_ARMS:
                for r in range(n):
                    items.append(_study4_item("codex_sub", task_key, effort, r))
    elif mode == "study4-q4q5":
        n = repeats or STUDY4_REPEATS_EXPLORATORY
        for door_key in ("runtime_us", "runtime_global"):
            for task_key in ("structured_json", "open_generation"):
                for r in range(n):
                    items.append(_study4_item(
                        door_key, task_key, "none", r, control="q4_routing"
                    ))
        for door_key in STUDY4_HTTP_GRID_DOORS:
            for task_key in ("structured_json", "open_generation"):
                for r in range(n):
                    items.append(_study4_item(
                        door_key, task_key, "default", r, control="q5_default"
                    ))
    elif mode in STUDY5_MODES:
        from harness.study5_fixtures import load_corpus as load_s5_corpus
        from harness.study5_schedule import (
            STUDY5_API_SUBSTRATES,
            STUDY5_LOCAL_SUBSTRATES_BY_BOX,
            build_study5_items,
            pilot_corpus,
        )

        if mode in STUDY5_LOCAL_MODES:
            if box not in STUDY5_LOCAL_SUBSTRATES_BY_BOX:
                raise ValueError(
                    f"{mode} requires --box "
                    f"{sorted(STUDY5_LOCAL_SUBSTRATES_BY_BOX)} (got {box})"
                )
            substrates = STUDY5_LOCAL_SUBSTRATES_BY_BOX[box]
        else:
            substrates = STUDY5_API_SUBSTRATES
        corpus = load_s5_corpus()
        if mode.startswith("study5-pilot"):
            corpus = pilot_corpus(corpus)
        items = build_study5_items(corpus, substrates=substrates)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return items


def _study4_item(door_key, task_key, effort, repeat, control=None):
    """Door item (PREREGISTRATION-v4 section 3). Responses doors: canonical
    bytes ARE the wire bytes (hashed == sent). Converse doors: the kwargs
    dict; its canonical hash is the planned-request hash, wire captured by
    the door's before-send hook. codex: the argv is the planned request; no
    wire control exists on the harness door by registration."""
    cfg = STUDY4_DOORS[door_key]
    prompt = TASKS[task_key]["prompt"]
    meta = {"door": door_key, "task": task_key, "effort": effort}
    if control:
        meta["control"] = control
    if cfg["kind"] == "responses":
        payload = canonical_responses_body(
            cfg["model_id"], prompt, effort, STUDY4_MAX_OUTPUT_TOKENS
        )
        sha = sha256_hex(payload)
    elif cfg["kind"] == "converse":
        payload = converse_request(
            cfg["model_id"], prompt, effort, STUDY4_MAX_OUTPUT_TOKENS
        )
        sha = sha256_hex(canonical_bytes(payload))
    else:  # codex
        payload = codex_argv(
            cfg["model_id"], prompt, effort,
            os.path.expanduser("~/.cache/gpts"),
        )
        sha = sha256_hex(canonical_bytes(payload))
    return {
        "cell": f"{door_key}|{task_key}|{effort}",
        "meta": meta,
        "door": door_key,
        "payload": payload,
        "sha": sha,
        "model_id": cfg["model_id"],
        "repeat": repeat,
    }


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
        door_clients=None,
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
        # Injectable for the timing-arm tests; per-item pre_sleep_ms
        # overrides the anti-burst jitter (companion E manipulates timing).
        self._sleep = time.sleep
        self.out = open(out_path, "a", encoding="utf-8")

        needed = {it.get("plane") for it in items}
        self.planes = dict(plane_clients or {})
        for plane_name in sorted(p for p in needed if p):
            if plane_name not in self.planes:
                self.planes[plane_name] = make_plane(plane_name)

        # Study-4 door clients: constructed up front so a missing credential
        # fails the run before the first call, never mid-schedule.
        self.doors = dict(door_clients or {})
        needed_doors = {it.get("door") for it in items}
        for door_key in sorted(d for d in needed_doors if d):
            if door_key not in self.doors:
                self.doors[door_key] = make_door(door_key)

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

    def _execute_door(self, item, rng, schedule_index):
        """Study-4 execution: dispatch through harness.doors; retry on the
        record's own retryable classification with the registered bound
        (max_attempts is set to STUDY4_RETRY_MAX_ATTEMPTS by main for
        study-4 modes; an exhausted item is a counted exclusion)."""
        door = self.doors[item["door"]]
        base = {
            "schema": 4,
            **self.run_info,
            "schedule_index": schedule_index,
            "cell": item["cell"],
            "repeat": item["repeat"],
            "door": item["door"],
            "model_id_sent": item["model_id"],
            "request_sha256": item["sha"],
            **{f"meta_{k}": v for k, v in item["meta"].items()},
        }
        attempts = 0
        result = {
            "ok": False, "error_code": "unknown",
            "error_message": "no attempt made", "retryable": False,
        }
        while attempts < self.max_attempts:
            attempts += 1
            sent_at = utc_now_iso()
            result = door.invoke(item["payload"])
            if result["ok"]:
                return {
                    **base,
                    **result,
                    "attempts": attempts,
                    "sent_at_utc": sent_at,
                    "received_at_utc": utc_now_iso(),
                }
            if result.get("retryable") and attempts < self.max_attempts:
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
                pre_sleep_ms = item.get("pre_sleep_ms")
                if pre_sleep_ms is None:
                    self._sleep(rng.uniform(0.25, 1.0))  # anti-burst jitter
                elif pre_sleep_ms > 0:
                    self._sleep(pre_sleep_ms / 1000.0)
                if item.get("door"):
                    record = self._execute_door(item, rng, idx)
                elif item.get("plane"):
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
    # Study-5 schedules ship their own per-substrate blocks + warmup heads
    # — fixed order, like the companion schedules.
    fixed = args.mode in FIXED_SCHEDULE_MODES or args.mode in STUDY5_MODES
    rng = random.Random(args.seed)
    if not fixed:
        rng.shuffle(schedule)

    settings5 = study5_run_settings(args.mode)
    if settings5 and settings5.get("concurrency") \
            and args.concurrency != settings5["concurrency"]:
        print(
            f'{args.mode} enforces concurrency='
            f'{settings5["concurrency"]} (--concurrency ignored)'
        )
        args.concurrency = settings5["concurrency"]

    settings4 = study4_run_settings(args.mode)
    if settings4 and args.concurrency != settings4["concurrency"]:
        print(
            f'{args.mode} enforces concurrency='
            f'{settings4["concurrency"]} (--concurrency ignored)'
        )
        args.concurrency = settings4["concurrency"]

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
    elif args.mode in STUDY4_MODES:
        manifest["schema"] = 4
        manifest["doors"] = sorted({it["door"] for it in schedule})
        manifest["repeats_override"] = args.repeats
        manifest["run_settings"] = settings4
        manifest["retry_max_attempts"] = STUDY4_RETRY_MAX_ATTEMPTS
        if args.mode == "study4-q4q5":
            manifest["exploratory"] = True
    elif args.mode in STUDY5_MODES:
        from harness.study5_fixtures import CORPUS_PATH as S5_CORPUS_PATH
        from harness.study5_fixtures import load_corpus as load_s5_corpus
        from harness.study5_schedule import (
            RESAMPLE_N,
            RESAMPLE_TEMPERATURE,
            RESAMPLE_TEMPLATE,
        )

        s5_corpus = load_s5_corpus()
        with open(S5_CORPUS_PATH, "rb") as fh:
            corpus_sha = sha256_hex(fh.read())
        manifest["schema"] = 5
        manifest["planes"] = sorted({it["plane"] for it in schedule})
        manifest["substrates"] = sorted(
            {it["meta"]["substrate"] for it in schedule}
        )
        manifest["pilot"] = args.mode.startswith("study5-pilot")
        manifest["corpus_sha256"] = corpus_sha
        manifest["corpus_frozen"] = s5_corpus["meta"]["frozen"]
        manifest["corpus_n_total"] = len(s5_corpus["items"])
        manifest["items_in_run"] = len(
            {it["meta"]["item_id"] for it in schedule
             if not it["meta"].get("control")}
        )
        manifest["resample"] = {
            "template": RESAMPLE_TEMPLATE,
            "n": RESAMPLE_N,
            "temperature": RESAMPLE_TEMPERATURE,
        }
        manifest["run_settings"] = settings5
        manifest["warmup_items"] = sum(
            1 for it in schedule if it["meta"].get("control") == "warmup"
        )
        if args.mode in STUDY5_LOCAL_MODES:
            manifest["box"] = args.box
            manifest["local_url"] = args.local_url

    if args.dry_run:
        manifest_path = os.path.join(args.out, f"{run_name}.dryrun.manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        print(f"DRY RUN: {len(schedule)} calls across {len(cells)} cells")
        print(f"schedule sha256: {manifest['schedule_sha256']}")
        print(f"manifest written: {manifest_path}")
        return 0

    if args.mode in STUDY2_MODES or (
        args.mode in STUDY5_MODES and args.mode not in STUDY5_LOCAL_MODES
    ):
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

    if args.mode in STUDY4_MODES:
        doors_present = {it["door"] for it in schedule}
        missing = []
        for door_key in sorted(doors_present):
            env_name = STUDY4_DOORS[door_key].get("api_key_env")
            if env_name and not os.environ.get(env_name):
                missing.append(f"{env_name} ({door_key}, run-scoped)")
        if missing:
            for name in missing:
                print(f"MISSING CREDENTIAL: {name}", flush=True)
            return 3

    local_plane = None
    if args.mode in STUDY3_MODES or args.mode in STUDY5_LOCAL_MODES:
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
    elif args.mode in STUDY5_MODES:
        run_info["schema"] = 5
        if args.mode in STUDY5_LOCAL_MODES:
            run_info["box"] = args.box
            # Study-5 local items carry the box-named plane
            # (local_cuda/local_metal) so records name their hardware.
            engine_kwargs = {
                "plane_clients": {f"local_{args.box}": local_plane},
            }
    elif args.mode in STUDY4_MODES:
        # The registered retry bound (v4 section 3): 3 bounded-backoff
        # attempts, then the item is a counted exclusion.
        engine_kwargs = {"max_attempts": STUDY4_RETRY_MAX_ATTEMPTS}
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
