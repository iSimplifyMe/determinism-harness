"""Companion-F analysis: fresh-instance manipulated confirmation.

Exploratory-directional; plan and analyzer committed before the data.
Deterministic code, stdlib only.

Gates:
- per-arm validity via analyze_study3.gate_cell3;
- cross-arm negative control: one request sha across burn-ins and both
  arms (flushers carry a different frozen prompt by design and are
  excluded);
- manipulation gate on absolute prefill classes (four-session evidence:
  full-KV ~16-18 ms vs checkpoint ~34-41 ms): every pure-arm call below
  PURE_MAX_MS AND every contaminated-arm call above CONTAMINATED_MIN_MS
  AND every cycle's unload-reset /api/ps-confirmed
  (pre_unload_confirmed on the burn-in record).

Registered endpoints (plan, companion F):
1. within-arm modal shares pooled across cycles (prediction: 1.0 both);
2. arms_differ (prediction: true);
3. per-cycle flip AT the interposed call: every cycle's pure calls all
   match the pooled pure modal and its contaminated calls all match the
   pooled contaminated modal (prediction: true in every cycle).

History matching stays descriptive, against the same on-record variants
as companions C/D/E (analyze_cache_ab.DEFAULT_HISTORY).

Usage:
  python3 -m analysis.analyze_cache_instance runs/local-study3-cache-instance-*.jsonl
"""
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from analysis.analyze_cache_ab import DEFAULT_HISTORY
from analysis.analyze_study3 import gate_cell3
from analysis.latency_fingerprint import percentile
from analysis.metrics import cell_metrics

INSTANCE_CELL = "gpt-oss-20b|open_generation|greedy|effort_low"
ARMS = ("pure", "contaminated")
PURE_MAX_MS = 25.0
CONTAMINATED_MIN_MS = 30.0


def _median(values):
    values = [v for v in values if v is not None]
    return percentile(values, 50) if values else None


def _prefill_ms(record):
    return ((record.get("usage") or {}).get("prompt_eval_duration_ns") or 0) / 1e6


def build_instance_report(records, history=None):
    controls = 0
    by_box_arm = defaultdict(list)
    burnins = defaultdict(list)
    for record in records:
        box = record.get("box") or record.get("meta_hardware")
        control = record.get("meta_control")
        if control == "burnin":
            controls += 1
            burnins[box].append(record)
            continue
        if control in ("warmup", "flusher"):
            controls += 1
            continue
        arm = record.get("meta_arm")
        if arm not in ARMS:
            continue
        by_box_arm[(box, arm)].append(record)

    boxes = {}
    for box in sorted({b for b, _ in by_box_arm}):
        arms_out = {}
        shas = set()
        prefill = {}
        valid_by_arm = {}
        for arm in ARMS:
            recs = by_box_arm.get((box, arm), [])
            gate = gate_cell3(recs)
            valid = gate["valid"]
            valid_by_arm[arm] = valid
            shas.update(
                r.get("request_sha256") for r in recs if r.get("request_sha256")
            )
            prefill[arm] = [_prefill_ms(r) for r in valid]
            entry = {
                "n_raw": gate["n_raw"],
                "excluded": gate["excluded"],
                "flags": gate["flags"],
                "prefill_ms": {
                    "median": _median(prefill[arm]),
                    "min": min(prefill[arm]) if prefill[arm] else None,
                    "max": max(prefill[arm]) if prefill[arm] else None,
                },
            }
            if valid:
                metrics = cell_metrics([
                    {
                        "text": r["text"],
                        "output_tokens": (r.get("usage") or {}).get(
                            "output_tokens"
                        ),
                    }
                    for r in valid
                ])
                entry["metrics"] = metrics
                if history:
                    entry["history_matches"] = sorted(
                        label for label, sha in history.items()
                        if sha == metrics["modal_sha256"]
                    )
            arms_out[arm] = entry

        # Burn-ins share the measured body: they join the negative
        # control and carry the per-cycle reset confirmation.
        box_burnins = burnins.get(box, [])
        shas.update(
            r.get("request_sha256")
            for r in box_burnins if r.get("request_sha256")
        )
        resets_by_cycle = {}
        burnin_prefill = {}
        for r in box_burnins:
            cycle = r.get("meta_cycle")
            resets_by_cycle[cycle] = bool(r.get("pre_unload_confirmed"))
            burnin_prefill[cycle] = _prefill_ms(r)

        cycle_numbers = sorted(
            {r.get("meta_cycle") for arm in ARMS for r in valid_by_arm[arm]}
            | set(resets_by_cycle)
        )
        resets_confirmed = bool(resets_by_cycle) and all(
            resets_by_cycle.get(c) for c in cycle_numbers
        )

        pure_values = prefill.get("pure", [])
        cont_values = prefill.get("contaminated", [])
        pure_all_below = (
            bool(pure_values)
            and all(v < PURE_MAX_MS for v in pure_values)
        )
        cont_all_above = (
            bool(cont_values)
            and all(v > CONTAMINATED_MIN_MS for v in cont_values)
        )
        manipulation = {
            "pure_all_below": pure_all_below,
            "contaminated_all_above": cont_all_above,
            "resets_confirmed": resets_confirmed,
            "resets_by_cycle": {
                str(c): resets_by_cycle.get(c) for c in cycle_numbers
            },
            "pure_median_ms": _median(pure_values),
            "contaminated_median_ms": _median(cont_values),
            "pure_max_ms_threshold": PURE_MAX_MS,
            "contaminated_min_ms_threshold": CONTAMINATED_MIN_MS,
            "pass": pure_all_below and cont_all_above and resets_confirmed,
        }
        cross_arm = len(shas) == 1
        arms_differ = None
        if (
            manipulation["pass"]
            and cross_arm
            and "metrics" in arms_out["pure"]
            and "metrics" in arms_out["contaminated"]
        ):
            arms_differ = (
                arms_out["pure"]["metrics"]["modal_sha256"]
                != arms_out["contaminated"]["metrics"]["modal_sha256"]
            )

        # Endpoint 3: within every cycle, all pure calls carry the pooled
        # pure modal and all contaminated calls the pooled contaminated
        # modal — the state flip sits exactly at the interposed call.
        cycles = []
        for c in cycle_numbers:
            entry = {
                "cycle": c,
                "burnin_reset_confirmed": resets_by_cycle.get(c),
                "burnin_prefill_ms": burnin_prefill.get(c),
            }
            uniform = {}
            for arm in ARMS:
                modal = arms_out[arm].get("metrics", {}).get("modal_sha256")
                texts = [
                    r["text"] for r in valid_by_arm[arm]
                    if r.get("meta_cycle") == c
                ]
                metrics_c = cell_metrics(
                    [{"text": t} for t in texts]
                ) if texts else None
                uniform[arm] = bool(
                    texts and metrics_c["modal_share"] == 1.0
                    and metrics_c["modal_sha256"] == modal
                )
                entry[f"{arm}_uniform"] = uniform[arm]
                entry[f"{arm}_n"] = len(texts)
            entry["flip_at_flusher"] = (
                uniform["pure"] and uniform["contaminated"]
                and arms_out["pure"].get("metrics", {}).get("modal_sha256")
                != arms_out["contaminated"].get("metrics", {}).get(
                    "modal_sha256"
                )
            )
            cycles.append(entry)

        endpoint3 = None
        if arms_differ is not None:
            endpoint3 = bool(cycles) and all(
                e["flip_at_flusher"] for e in cycles
            )

        boxes[box] = {
            "arms": arms_out,
            "gates": {
                "cross_arm_negative_control": cross_arm,
                "manipulation": manipulation,
            },
            "arms_differ": arms_differ,
            "cycles": cycles,
            "endpoint3_flip_all_cycles": endpoint3,
        }

    return {
        "exploratory": True,
        "companion_plan": "FOLLOWUP-COMPANIONS.md (companion F)",
        "cell": INSTANCE_CELL,
        "endpoints": (
            "1: within-arm modal shares pooled across cycles (predicted "
            "1.0 both); 2: arms_differ (predicted true); 3: per-cycle "
            "flip at the interposed call (predicted true every cycle); "
            "history matching descriptive"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "records_in": len(records),
            "controls_excluded": controls,
        },
        "boxes": boxes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Companion-F fresh-instance analysis (exploratory)"
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
    if not records:
        print("ERROR: no records")
        return 2

    report = build_instance_report(records, history=DEFAULT_HISTORY)
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.out, f"cache-instance-report-{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    for box in sorted(report["boxes"]):
        entry = report["boxes"][box]
        manipulation = entry["gates"]["manipulation"]
        pure = entry["arms"]["pure"]
        cont = entry["arms"]["contaminated"]
        flips = [e["flip_at_flusher"] for e in entry["cycles"]]
        print(
            f"{box}: gate pass={manipulation['pass']} "
            f"(pure med {manipulation['pure_median_ms']}ms, cont med "
            f"{manipulation['contaminated_median_ms']}ms, resets "
            f"{manipulation['resets_confirmed']}); pure modal "
            f"{pure.get('metrics', {}).get('modal_share')} "
            f"[{', '.join(pure.get('history_matches', []) or ['-'])}], "
            f"cont modal "
            f"{cont.get('metrics', {}).get('modal_share')} "
            f"[{', '.join(cont.get('history_matches', []) or ['-'])}]; "
            f"arms_differ={entry['arms_differ']}; "
            f"flip_per_cycle={flips}; "
            f"endpoint3={entry['endpoint3_flip_all_cycles']}"
        )
    print(f"report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
