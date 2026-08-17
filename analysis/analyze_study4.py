"""Study-4 (PREREGISTRATION-v4) estimators, committed before confirmatory
data — the Q1-Q5 readouts exactly as registered.

Stdlib only, like every analysis module here. Expected record shape
(the study-4 runner modes emit it; the study-2 flat-record convention):

    {"mode": "study4-...", "window": "...", "repeat": int, "ok": bool,
     "meta_door": "openai_1p|mantle|runtime_us|runtime_global|codex_sub",
     "meta_task": "...", "meta_effort": "none|high|default",
     "text": "...", "text_sha256": "...", "usage": {...}, ...}

Usage:
    python3 -m analysis.analyze_study4 runs/study4-*.jsonl --out reports
"""
import json
import math
import sys
from collections import Counter, defaultdict

from analysis.stats import (
    stratified_diff,
    stratified_tost,
    wald_diff,
    wilson_interval,
)

DELTA = 0.02          # Q2 equivalence margin, carried from study 2
PRIMARY_TASK = "structured_json"
Q1_EFFORT = "none"
HTTP_DOORS = ("openai_1p", "mantle", "runtime_us")  # + runtime_global in Q4
GRID_DOORS = ("openai_1p", "mantle", "runtime_us", "codex_sub")
Q4_DOORS = ("runtime_us", "runtime_global")
Q4_TASKS = ("structured_json", "open_generation")
Q5_TASKS = ("structured_json", "open_generation")


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]], min-likelihood
    ordering (the same two-sided convention as stats.binom_test)."""
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def log_hyper(k):
        return (
            math.lgamma(col1 + 1) - math.lgamma(k + 1)
            - math.lgamma(col1 - k + 1)
            + math.lgamma(n - col1 + 1) - math.lgamma(row1 - k + 1)
            - math.lgamma(n - col1 - row1 + k + 1)
            - (math.lgamma(n + 1) - math.lgamma(row1 + 1)
               - math.lgamma(n - row1 + 1))
        )

    lo, hi = max(0, row1 - (n - col1)), min(row1, col1)
    observed = math.exp(log_hyper(a))
    total = 0.0
    for k in range(lo, hi + 1):
        p = math.exp(log_hyper(k))
        if p <= observed * (1.0 + 1e-7):
            total += p
    return min(1.0, total)


def load_records(paths):
    """Read study-4 JSONL records; returns (records, exclusions)."""
    records, exclusions = [], []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not str(rec.get("mode", "")).startswith("study4"):
                    continue
                if rec.get("ok"):
                    records.append(rec)
                else:
                    exclusions.append(rec)
    return records, exclusions


def _key(rec):
    return (rec["meta_door"], rec["meta_task"], rec["meta_effort"])


def group_cells(records):
    """(door, task, effort) -> window -> [records]."""
    cells = defaultdict(lambda: defaultdict(list))
    for rec in records:
        cells[_key(rec)][rec.get("window", "?")].append(rec)
    return cells


def _pooled(cells, door, task, effort):
    out = []
    for recs in cells.get((door, task, effort), {}).values():
        out.extend(recs)
    return out


def _variant_counts(recs):
    counts = Counter(rec["text_sha256"] for rec in recs)
    snippets = {}
    for rec in recs:
        snippets.setdefault(rec["text_sha256"], rec.get("text", "")[:160])
    return counts, snippets


def q1_door_attribution(cells):
    """Per-door byte-variant distribution on the primary cell; pairwise
    contrasts on the study-wide modal variant. Registered direction: only
    the openai_1p vs codex_sub pair (v4 section 1)."""
    pooled_all = Counter()
    per_door = {}
    for door in GRID_DOORS:
        recs = _pooled(cells, door, PRIMARY_TASK, Q1_EFFORT)
        counts, snippets = _variant_counts(recs)
        pooled_all.update(counts)
        per_door[door] = {"counts": counts, "snippets": snippets,
                          "n": len(recs)}
    if not pooled_all:
        return {"note": "no Q1 records"}
    modal_variant = pooled_all.most_common(1)[0][0]
    doors_out = {}
    for door, data in per_door.items():
        k = data["counts"].get(modal_variant, 0)
        n = data["n"]
        doors_out[door] = {
            "n": n,
            "modal_variant_share": (k / n) if n else None,
            "wilson95": wilson_interval(k, n) if n else None,
            "variants": [
                {"sha": sha, "count": count,
                 "snippet": data["snippets"].get(sha, "")}
                for sha, count in data["counts"].most_common()
            ],
        }
    pairs = {}
    for i, one in enumerate(GRID_DOORS):
        for two in GRID_DOORS[i + 1:]:
            a = per_door[one]["counts"].get(modal_variant, 0)
            n1 = per_door[one]["n"]
            c = per_door[two]["counts"].get(modal_variant, 0)
            n2 = per_door[two]["n"]
            if not n1 or not n2:
                continue
            pairs[f"{one}_vs_{two}"] = {
                "diff": wald_diff(a, n1, c, n2),
                "fisher_p": fisher_exact(a, n1 - a, c, n2 - c),
                "registered_direction": {one, two}
                == {"openai_1p", "codex_sub"},
            }
    return {
        "primary_cell": f"{PRIMARY_TASK}|effort={Q1_EFFORT}",
        "modal_variant_sha": modal_variant,
        "doors": doors_out,
        "pairwise": pairs,
    }


def _modal_count(recs):
    if not recs:
        return 0, 0
    counts = Counter(rec["text_sha256"] for rec in recs)
    return counts.most_common(1)[0][1], len(recs)


def q2_equivalence(cells, tasks, efforts, windows):
    """Stratified TOST between HTTP-door pairs; strata are matched
    (task, effort, window) cells; exact-match = share of the cell's own
    modal text (study-2 endpoint). codex_sub excluded by registration."""
    out = {}
    for i, one in enumerate(HTTP_DOORS):
        for two in HTTP_DOORS[i + 1:]:
            strata = []
            for task in tasks:
                for effort in efforts:
                    for window in windows:
                        r1 = cells.get((one, task, effort), {}).get(window, [])
                        r2 = cells.get((two, task, effort), {}).get(window, [])
                        if not r1 or not r2:
                            continue
                        x1, n1 = _modal_count(r1)
                        x2, n2 = _modal_count(r2)
                        strata.append((x1, n1, x2, n2))
            if strata:
                out[f"{one}_vs_{two}"] = stratified_tost(strata, DELTA)
    return out


def q3_effort_analog(cells):
    """Per-door high-minus-none difference in structured-JSON modal share
    (the study-2 thinking-effect endpoint), plus cross-door DoD."""
    per_door = {}
    for door in GRID_DOORS:
        high = _pooled(cells, door, PRIMARY_TASK, "high")
        none = _pooled(cells, door, PRIMARY_TASK, "none")
        if not high or not none:
            continue
        xh, nh = _modal_count(high)
        xn, nn = _modal_count(none)
        per_door[door] = wald_diff(xh, nh, xn, nn)
    dod = {}
    doors = list(per_door)
    for i, one in enumerate(doors):
        for two in doors[i + 1:]:
            d1, d2 = per_door[one], per_door[two]
            diff = d1["diff"] - d2["diff"]
            se = math.sqrt(d1["se"] ** 2 + d2["se"] ** 2)
            dod[f"{one}_vs_{two}"] = {
                "dod": diff, "se": se,
                "ci95": (diff - 1.959963984540054 * se,
                         diff + 1.959963984540054 * se),
            }
    return {"per_door_high_minus_none": per_door,
            "cross_door_dod": dod}


def q4_routing(cells):
    """runtime_us vs runtime_global at effort none: per-task modal-share
    difference + the pooled stratified bound (study-1 style)."""
    per_task, strata = {}, []
    for task in Q4_TASKS:
        r_us = _pooled(cells, "runtime_us", task, Q1_EFFORT)
        r_gl = _pooled(cells, "runtime_global", task, Q1_EFFORT)
        if not r_us or not r_gl:
            continue
        x1, n1 = _modal_count(r_us)
        x2, n2 = _modal_count(r_gl)
        per_task[task] = wald_diff(x1, n1, x2, n2)
        strata.append((x1, n1, x2, n2))
    return {
        "per_task": per_task,
        "stratified": stratified_diff(strata) if strata else None,
    }


def _reasoning_tokens(rec):
    usage = rec.get("usage") or {}
    details = usage.get("output_tokens_details") or {}
    if "reasoning_tokens" in details:
        return details["reasoning_tokens"]
    if "reasoning_output_tokens" in usage:
        return usage["reasoning_output_tokens"]
    return None  # Converse: aggregate outputTokens only (door property)


def _quantiles(values):
    values = sorted(values)
    n = len(values)
    if not n:
        return None

    def q(frac):
        idx = frac * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return values[lo] + (values[hi] - values[lo]) * (idx - lo)

    return {"n": n, "min": values[0], "q1": q(0.25), "median": q(0.5),
            "q3": q(0.75), "max": values[-1]}


def q5_default_burn(cells):
    """Default-arm reasoning dispersion + the modal-vs-divergent spend
    association, descriptive only (v4 section 1)."""
    out = {}
    for (door, task, effort), windows in cells.items():
        if effort != "default" or task not in Q5_TASKS:
            continue
        recs = [rec for recs in windows.values() for rec in recs]
        tokens = [(_reasoning_tokens(rec), rec["text_sha256"]) for rec in recs]
        known = [(tok, sha) for tok, sha in tokens if tok is not None]
        entry = {"n": len(recs)}
        if known:
            counts = Counter(sha for _, sha in tokens)
            modal_sha = counts.most_common(1)[0][0]
            entry["reasoning_tokens"] = _quantiles([t for t, _ in known])
            modal = [t for t, sha in known if sha == modal_sha]
            divergent = [t for t, sha in known if sha != modal_sha]
            entry["association"] = {
                "modal_n": len(modal),
                "modal_median": _quantiles(modal)["median"] if modal else None,
                "divergent_n": len(divergent),
                "divergent_median": (
                    _quantiles(divergent)["median"] if divergent else None
                ),
            }
        else:
            # Converse cells: reasoning is invisible; report total output.
            totals = [
                (rec.get("usage") or {}).get("outputTokens")
                for rec in recs
            ]
            totals = [t for t in totals if t is not None]
            entry["output_tokens_total_only"] = _quantiles(totals)
        out[f"{door}|{task}"] = entry
    return out


def analyze(paths):
    records, exclusions = load_records(paths)
    cells = group_cells(records)
    tasks = sorted({key[1] for key in cells})
    efforts = sorted({key[2] for key in cells})
    windows = sorted({
        window for windows in cells.values() for window in windows
    })
    return {
        "inputs": list(paths),
        "n_records": len(records),
        "n_exclusions": len(exclusions),
        "exclusion_codes": dict(Counter(
            rec.get("error_code") for rec in exclusions
        )),
        "q1_door_attribution": q1_door_attribution(cells),
        "q2_equivalence": q2_equivalence(
            cells, tasks, [e for e in efforts if e != "default"], windows
        ),
        "q3_effort_analog": q3_effort_analog(cells),
        "q4_routing": q4_routing(cells),
        "q5_default_burn": q5_default_burn(cells),
    }


def main(argv):
    out_dir = "reports"
    paths = []
    it = iter(argv)
    for arg in it:
        if arg == "--out":
            out_dir = next(it)
        else:
            paths.append(arg)
    if not paths:
        print("usage: python3 -m analysis.analyze_study4 <jsonl...> [--out dir]")
        return 2
    report = analyze(paths)
    import os

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "study4-analysis.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=list)
    print(f"records={report['n_records']} exclusions={report['n_exclusions']}")
    q1 = report["q1_door_attribution"]
    for door, data in (q1.get("doors") or {}).items():
        share = data["modal_variant_share"]
        print(f"  Q1 {door}: modal-variant share "
              f"{share:.3f} (n={data['n']})" if share is not None else door)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
