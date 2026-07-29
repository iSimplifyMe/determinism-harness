"""Latency/throughput fingerprint per serving plane (study 2 addendum).

EXPLORATORY / UNREGISTERED. This analysis is post-hoc and descriptive: it was
not part of PREREGISTRATION-v2 and makes no confirmatory claims. Motivation:
study 2 found the thinking-mode reproducibility cost follows the AWS front
door (Bedrock = P-AWS at 2pp, both differ from first-party). Every record
carries `latency_ms`; if the two AWS doors are served by the same stack,
their latency *shapes* and per-token decode rates should agree with each
other more than either agrees with the first-party API. That is a
distribution comparison, not proof of any mechanism — network paths, load,
and regional pinning also differ per plane (see caveats in the report).

Method, stdlib only:
- per stratum (window x model x task x thinking x delivery), per plane:
  latency percentile ladder;
- cross-plane two-sample KS distances on raw latency AND on median-centered
  latency (centering removes the pure network/queue offset, leaving shape);
- per (plane, model, window) decode rate: weighted least squares over
  per-output-token-count median latencies (bin medians resist tail noise) —
  slope = ms per generated token, `usage.output_tokens` (includes thinking
  tokens on every plane, so it is total decoded work);
- per (plane, model) prefill rate from the Q4 input-length ladder;
- open-generation effective throughput (tokens/sec) as a cross-check.

Usage:
  python3 -m analysis.latency_fingerprint runs/low-study2-*.jsonl \
      runs/peak-study2-*.jsonl runs/control-study2-*.jsonl --out reports
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import median

PAIRS = (
    ("bedrock", "p_aws"),
    ("anthropic_api", "bedrock"),
    ("anthropic_api", "p_aws"),
)


def eligible(record):
    """Successful first-attempt calls with a latency reading."""
    return bool(
        record.get("ok")
        and record.get("attempts", 1) == 1
        and record.get("latency_ms") is not None
    )


def stratum_key(record):
    delivery = "streamed" if record.get("delivered_streaming") else "request"
    return (
        f'{record.get("window")}::{record.get("meta_model")}'
        f'|{record.get("meta_task")}|{record.get("meta_thinking")}|{delivery}'
    )


def collect_strata(records):
    """stratum_key -> plane -> [(latency_ms, output_tokens, input_tokens)]."""
    strata = defaultdict(lambda: defaultdict(list))
    for record in records:
        if not eligible(record):
            continue
        usage = record.get("usage") or {}
        strata[stratum_key(record)][record.get("plane")].append(
            (
                record["latency_ms"],
                usage.get("output_tokens"),
                usage.get("input_tokens"),
            )
        )
    return strata


def percentile(values, q):
    """Linear-interpolated percentile, q in [0, 100]."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q / 100 * (len(ordered) - 1)
    lower = int(pos)
    frac = pos - lower
    if lower + 1 >= len(ordered):
        return float(ordered[-1])
    return ordered[lower] * (1 - frac) + ordered[lower + 1] * frac


def summarize(values):
    n = len(values)
    mean = sum(values) / n
    out = {"n": n, "mean": round(mean, 1)}
    for q in (5, 25, 50, 75, 95):
        out[f"p{q}"] = round(percentile(values, q), 1)
    out["iqr"] = round(out["p75"] - out["p25"], 1)
    if mean:
        variance = sum((v - mean) ** 2 for v in values) / n
        out["cv"] = round((variance ** 0.5) / mean, 4)
    return out


def ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic, tie-correct. Descriptive only."""
    a = sorted(a)
    b = sorted(b)
    na, nb = len(a), len(b)
    i = j = 0
    d = 0.0
    while i < na and j < nb:
        x = a[i] if a[i] <= b[j] else b[j]
        while i < na and a[i] == x:
            i += 1
        while j < nb and b[j] == x:
            j += 1
        d = max(d, abs(i / na - j / nb))
    return d


def center(values):
    m = median(values)
    return [v - m for v in values]


def binned_median_regression(pairs):
    """Weighted least squares over per-x median y. x is discrete (token counts).

    Bin medians make the fit robust to the long latency tail; weights are bin
    sizes so dense bins dominate.
    """
    bins = defaultdict(list)
    for x, y in pairs:
        bins[x].append(y)
    points = [(x, median(ys), len(ys)) for x, ys in sorted(bins.items())]
    if len(points) < 2:
        return None
    total = sum(w for _, _, w in points)
    xbar = sum(x * w for x, _, w in points) / total
    ybar = sum(y * w for _, y, w in points) / total
    sxx = sum(w * (x - xbar) ** 2 for x, _, w in points)
    if not sxx:
        return None
    slope = sum(w * (x - xbar) * (y - ybar) for x, y, w in points) / sxx
    return {
        "slope_ms_per_token": round(slope, 4),
        "intercept_ms": round(ybar - slope * xbar, 1),
        "n": total,
        "n_bins": len(points),
        "token_min": points[0][0],
        "token_max": points[-1][0],
    }


def decode_slopes(records):
    """Per plane|model|window: latency ~ output_tokens over full-window,
    non-streamed records (input tokens constant there; tasks provide the
    output-length lever arm)."""
    grouped = defaultdict(list)
    for record in records:
        if not eligible(record) or record.get("mode") != "study2-full":
            continue
        if record.get("delivered_streaming"):
            continue
        usage = record.get("usage") or {}
        out_tok = usage.get("output_tokens")
        if out_tok is None:
            continue
        key = (
            f'{record.get("plane")}|{record.get("meta_model")}'
            f'|{record.get("window")}'
        )
        grouped[key].append((out_tok, record["latency_ms"]))
    fits = {}
    for key in sorted(grouped):
        fit = binned_median_regression(grouped[key])
        if fit:
            fits[key] = fit
    return fits


def prefill_slopes(records):
    """Per plane|model: latency ~ input_tokens over the Q4 length ladder."""
    grouped = defaultdict(list)
    for record in records:
        if not eligible(record) or record.get("mode") != "study2-q4-lengths":
            continue
        usage = record.get("usage") or {}
        in_tok = usage.get("input_tokens")
        if in_tok is None:
            continue
        key = f'{record.get("plane")}|{record.get("meta_model")}'
        grouped[key].append((in_tok, record["latency_ms"]))
    fits = {}
    for key in sorted(grouped):
        fit = binned_median_regression(grouped[key])
        if fit:
            fits[key] = fit
    return fits


def pair_distances(strata):
    """Per stratum, KS distance for each plane pair, raw and median-centered."""
    distances = {}
    for key in sorted(strata):
        planes = strata[key]
        entry = {}
        for left, right in PAIRS:
            if left not in planes or right not in planes:
                continue
            a = [t[0] for t in planes[left]]
            b = [t[0] for t in planes[right]]
            entry[f"{left}|{right}"] = {
                "ks_raw": round(ks_stat(a, b), 4),
                "ks_centered": round(ks_stat(center(a), center(b)), 4),
                "n_min": min(len(a), len(b)),
            }
        if entry:
            distances[key] = entry
    return distances


def aggregate_pair_distances(distances):
    """Median KS per pair across strata + closest-pair votes (centered)."""
    per_pair_raw = defaultdict(list)
    per_pair_centered = defaultdict(list)
    votes = Counter()
    strata = 0
    for entry in distances.values():
        if len(entry) < len(PAIRS):
            continue
        strata += 1
        for pair, d in entry.items():
            per_pair_raw[pair].append(d["ks_raw"])
            per_pair_centered[pair].append(d["ks_centered"])
        closest = min(entry, key=lambda pair: entry[pair]["ks_centered"])
        votes[closest] += 1
    return {
        "strata": strata,
        "median_ks_raw": {
            pair: round(median(vals), 4) for pair, vals in sorted(per_pair_raw.items())
        },
        "median_ks_centered": {
            pair: round(median(vals), 4)
            for pair, vals in sorted(per_pair_centered.items())
        },
        "closest_pair_votes_centered": dict(votes),
    }


def open_gen_throughput(records):
    """tokens/sec on open_generation full-window records (decode-dominated)."""
    grouped = defaultdict(list)
    for record in records:
        if not eligible(record) or record.get("mode") != "study2-full":
            continue
        if record.get("meta_task") != "open_generation":
            continue
        usage = record.get("usage") or {}
        out_tok = usage.get("output_tokens")
        if not out_tok or not record["latency_ms"]:
            continue
        key = (
            f'{record.get("plane")}|{record.get("meta_model")}'
            f'|{record.get("window")}'
        )
        grouped[key].append(out_tok / (record["latency_ms"] / 1000))
    return {key: summarize(vals) for key, vals in sorted(grouped.items())}


def covariates(records):
    per_plane = defaultdict(lambda: {
        "geo": Counter(), "tier": Counter(), "stop": Counter(),
        "eligible": 0, "not_ok": 0, "retried": 0,
    })
    for record in records:
        plane = record.get("plane")
        entry = per_plane[plane]
        if not record.get("ok"):
            entry["not_ok"] += 1
            continue
        if record.get("attempts", 1) > 1:
            entry["retried"] += 1
            continue
        entry["eligible"] += 1
        usage = record.get("usage") or {}
        entry["geo"][usage.get("inference_geo", "absent")] += 1
        entry["tier"][usage.get("service_tier", "absent")] += 1
        entry["stop"][record.get("stop_reason", "absent")] += 1
    return {
        plane: {
            "eligible": e["eligible"],
            "excluded_not_ok": e["not_ok"],
            "excluded_retried": e["retried"],
            "inference_geo": dict(e["geo"]),
            "service_tier": dict(e["tier"]),
            "stop_reason": dict(e["stop"]),
        }
        for plane, e in sorted(per_plane.items())
    }


def _category(mode):
    return {
        "study2-full": "full",
        "study2-q3-streaming": "streamed",
        "study2-q4-lengths": "q4_lengths",
        "study2-positive-control": "positive_control",
    }.get(mode, mode or "unknown")


def build_report(records):
    strata = collect_strata(records)
    stratum_category = {}
    for record in records:
        if eligible(record):
            stratum_category.setdefault(
                stratum_key(record), _category(record.get("mode"))
            )
    distances = pair_distances(strata)
    by_category = defaultdict(dict)
    for key, entry in distances.items():
        by_category[stratum_category.get(key, "unknown")][key] = entry

    eligible_n = sum(len(v) for planes in strata.values() for v in planes.values())
    report = {
        "exploratory": True,
        "registered": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "records_in": len(records),
            "eligible": eligible_n,
            "strata": len(strata),
        },
        "covariates": covariates(records),
        "decode_slopes": decode_slopes(records),
        "prefill_slopes": prefill_slopes(records),
        "open_gen_tokens_per_sec": open_gen_throughput(records),
        "stratum_summaries": {
            key: {plane: summarize([t[0] for t in vals])
                  for plane, vals in sorted(planes.items())}
            for key, planes in sorted(strata.items())
        },
        "pair_distances": distances,
        "pair_distances_aggregate": {
            category: aggregate_pair_distances(entries)
            for category, entries in sorted(by_category.items())
        },
    }
    return report


def ascii_hist(values, lo, hi, bins=18, width=32):
    counts = [0] * bins
    span = (hi - lo) or 1
    for v in values:
        idx = int((v - lo) / span * bins)
        counts[max(0, min(bins - 1, idx))] += 1
    peak = max(counts) or 1
    lines = []
    for i, n in enumerate(counts):
        left = lo + span * i / bins
        bar = "#" * max(1 if n else 0, round(n / peak * width))
        lines.append(f"{left:>8.0f}ms |{bar:<{width}}| {n}")
    return lines


ILLUSTRATIVE = (
    "peak::opus-5|structured_json|adaptive|request",
    "peak::sonnet-5|structured_json|adaptive|request",
    "peak::sonnet-5|open_generation|disabled|request",
)

CAVEATS = (
    "Exploratory and unregistered: written after the confirmatory results "
    "were known; no hypothesis was preregistered and no p-values are "
    "reported. Distances and slopes are descriptive.",
    "Raw latency mixes serving time with network path, TLS, and per-plane "
    "ingress overhead measured from one client in one city; raw-latency "
    "distances partly reflect geography, not serving hardware. "
    "Median-centered distances and per-token slopes are the informative "
    "readouts.",
    "P-AWS shed 85 calls (529) in the low window; its surviving low-window "
    "records may be load-biased (survivor bias). Bedrock and 1P had zero "
    "terminal failures.",
    "inference_geo covariate: both Messages planes (1P, P-AWS) report "
    "'global' on 5-family records while Bedrock was pinned to the us. "
    "inference profile - regional routing is not held constant across "
    "planes by the platforms themselves.",
    "One client, one region-pinning configuration, two windows on one day "
    "per window type. Latency shapes may vary by day and region.",
    "A latency fingerprint cannot separate 'same hardware' from 'same "
    "software build on similar hardware'; agreement is consistent with a "
    "shared serving stack, not proof of one.",
)


def write_md(report, path):
    lines = []
    lines.append("# Latency fingerprint per serving plane (study 2 addendum)")
    lines.append("")
    lines.append(
        "**EXPLORATORY / UNREGISTERED.** Post-hoc descriptive analysis of the "
        "study-2 confirmatory records' `latency_ms`; no confirmatory claims. "
        f"Generated {report['generated_utc']}."
    )
    lines.append("")
    totals = report["totals"]
    lines.append(
        f"Records in: {totals['records_in']} - eligible (ok, first attempt): "
        f"{totals['eligible']} - strata: {totals['strata']}."
    )
    lines.append("")
    lines.append("## Covariates per plane")
    lines.append("")
    lines.append("| plane | eligible | not ok | retried | inference_geo | service_tier |")
    lines.append("|---|---|---|---|---|---|")
    for plane, c in report["covariates"].items():
        lines.append(
            f"| {plane} | {c['eligible']} | {c['excluded_not_ok']} | "
            f"{c['excluded_retried']} | {c['inference_geo']} | {c['service_tier']} |"
        )
    lines.append("")
    lines.append("## Decode rate - ms per generated token (binned-median WLS)")
    lines.append("")
    lines.append(
        "Fit over full-window, non-streamed records; output length is the "
        "lever arm (tasks range from a few tokens to open generation). "
        "`usage.output_tokens` includes thinking tokens on every plane, so "
        "the slope prices total decoded work. tok/s = 1000/slope."
    )
    lines.append("")
    lines.append("| plane\\|model\\|window | ms/token | ~tok/s | intercept ms | n | bins | token range |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, fit in report["decode_slopes"].items():
        slope = fit["slope_ms_per_token"]
        tps = round(1000 / slope, 1) if slope else "-"
        lines.append(
            f"| {key} | {slope} | {tps} | {fit['intercept_ms']} | {fit['n']} | "
            f"{fit['n_bins']} | {fit['token_min']}-{fit['token_max']} |"
        )
    lines.append("")
    lines.append("## Prefill rate - ms per input token (Q4 length ladder)")
    lines.append("")
    lines.append("| plane\\|model | ms/token | intercept ms | n | bins | token range |")
    lines.append("|---|---|---|---|---|---|")
    for key, fit in report["prefill_slopes"].items():
        lines.append(
            f"| {key} | {fit['slope_ms_per_token']} | {fit['intercept_ms']} | "
            f"{fit['n']} | {fit['n_bins']} | {fit['token_min']}-{fit['token_max']} |"
        )
    lines.append("")
    lines.append("## Open-generation effective throughput (tokens/sec)")
    lines.append("")
    lines.append("| plane\\|model\\|window | n | p25 | p50 | p75 |")
    lines.append("|---|---|---|---|---|")
    for key, s in report["open_gen_tokens_per_sec"].items():
        lines.append(f"| {key} | {s['n']} | {s['p25']} | {s['p50']} | {s['p75']} |")
    lines.append("")
    lines.append("## Cross-plane distribution distances (two-sample KS)")
    lines.append("")
    lines.append(
        "`ks_centered` compares median-centered latencies - shape with the "
        "network/queue offset removed. `closest votes` counts strata where "
        "that pair is the most similar of the three (centered)."
    )
    for category, agg in report["pair_distances_aggregate"].items():
        lines.append("")
        lines.append(f"### {category} ({agg['strata']} strata with all three planes)")
        lines.append("")
        lines.append("| pair | median KS raw | median KS centered | closest votes |")
        lines.append("|---|---|---|---|")
        votes = agg["closest_pair_votes_centered"]
        for pair in agg["median_ks_raw"]:
            lines.append(
                f"| {pair} | {agg['median_ks_raw'][pair]} | "
                f"{agg['median_ks_centered'][pair]} | {votes.get(pair, 0)} |"
            )
    lines.append("")
    lines.append("## Illustrative strata - median-centered latency shape")
    for key in ILLUSTRATIVE:
        planes = report["stratum_summaries"].get(key)
        if not planes:
            continue
        lines.append("")
        lines.append(f"### {key}")
        lines.append("")
        lines.append("| plane | n | p5 | p25 | p50 | p75 | p95 | IQR | CV |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for plane, s in planes.items():
            lines.append(
                f"| {plane} | {s['n']} | {s['p5']} | {s['p25']} | {s['p50']} | "
                f"{s['p75']} | {s['p95']} | {s['iqr']} | {s.get('cv', '-')} |"
            )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    for caveat in CAVEATS:
        lines.append(f"- {caveat}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Exploratory latency fingerprint per serving plane"
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

    report = build_report(records)
    if not report["totals"]["eligible"]:
        print("ERROR: no eligible records")
        return 2

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(args.out, f"latency-fingerprint-{stamp}.json")
    md_path = os.path.join(args.out, f"latency-fingerprint-{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    write_md(report, md_path)

    print(
        f"eligible={report['totals']['eligible']} strata={report['totals']['strata']}"
    )
    for key, fit in report["decode_slopes"].items():
        print(f"  decode {key}: {fit['slope_ms_per_token']} ms/token")
    for category, agg in report["pair_distances_aggregate"].items():
        print(
            f"  KS[{category}] centered={agg['median_ks_centered']} "
            f"votes={agg['closest_pair_votes_centered']}"
        )
    print(f"fingerprint -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
