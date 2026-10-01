"""Study-5 pilot power projection (PREREGISTRATION-v5 section 6).

Turns the pilot's 2x2 counts into the three numbers the freeze needs:

- projected power of the registered H1 test (two-sided Fisher exact at
  alpha .05 with a positive risk difference) at the confirmatory n, per
  candidate primary substrate, and on the pooled near_tie+ambiguous
  strata (the registered fallback);
- the primary substrate, resolved by the section-6 rule;
- projected power of the H3 equivalence test (TOST on the cross-door
  detector's catch rate minus false-alarm rate) at each candidate bound.

Method, fixed before the pilot's result tables were read:

- The model is the design's own sampling model. Each item independently
  disagrees with probability q; a disagreeing item is wrong with
  probability p1 and an agreeing item with probability p0. Power is the
  exact probability, under that model, that the confirmatory table
  rejects - enumerated, not simulated, so the number has no seed.
- Pilot proportions enter as (x + 0.5) / (n + 1), the adjustment
  analysis/stats.py already uses to rescue a zero standard error. A
  pilot of 21 items produces empty cells; a raw 0 or 1 would project a
  certainty the pilot cannot support. The raw-proportion projection is
  reported alongside whenever it is defined; the decision rule reads the
  adjusted one.
- Enumeration drops outcomes whose probability factor is below 1e-9;
  the covered probability mass is reported with every power figure.

Stdlib only.
"""
import argparse
import json
import os
from datetime import datetime, timezone

from analysis import stats
from analysis.analyze_study5 import (
    CROSS_DOOR,
    REGISTERED_KSETS,
    cross_pair_analysis,
    kset_analysis,
    load_records,
    parse_response,
    record_meta,
)

ALPHA = 0.05
POWER_BAR = 0.80
PRIMARY_CANDIDATES = ("sonnet_1p", "haiku_1p")   # section-6 order
POOLED_STRATA = ("near_tie", "ambiguous")
H3_BOUNDS = (0.10, 0.15)                          # tried in this order
PRUNE = 1e-9


def adjusted(x, n):
    """(x + 0.5) / (n + 1): never exactly 0 or 1, defined at n == 0."""
    return (x + 0.5) / (n + 1.0)


def _pmf_row(n, p):
    return [stats.binom_pmf(k, n, p) for k in range(n + 1)]


def h1_power(n, q, p1, p0, alpha=ALPHA):
    """Exact P(Fisher two-sided p < alpha AND risk difference > 0) for a
    confirmatory table of n items under (q, p1, p0). Returns
    (power, mass_covered)."""
    power = 0.0
    covered = 0.0
    for d, w_d in enumerate(_pmf_row(n, q)):
        if w_d < PRUNE:
            continue
        agree = n - d
        row_a = _pmf_row(d, p1)
        row_c = _pmf_row(agree, p0)
        for a, w_a in enumerate(row_a):
            if w_a < PRUNE:
                continue
            for c, w_c in enumerate(row_c):
                if w_c < PRUNE:
                    continue
                weight = w_d * w_a * w_c
                covered += weight
                if d == 0 or agree == 0:
                    continue  # one row empty: no contrast, no rejection
                if a / d <= c / agree:
                    continue  # wrong direction or no difference
                if stats.fisher_exact(a, d - a, c, agree - c) < alpha:
                    power += weight
    return power, covered


def h3_power(n, w, r, delta, alpha=ALPHA):
    """Exact P(TOST declares equivalence) for the cross-door detector's
    catch rate minus false-alarm rate, when the truth is the registered
    null: wrong and right items disagree across doors at the same rate r.
    n_wrong ~ Binomial(n, w). A table with no wrong items or no right
    items cannot be tested and counts as not equivalent. Returns
    (power, mass_covered)."""
    power = 0.0
    covered = 0.0
    for m, w_m in enumerate(_pmf_row(n, w)):
        if w_m < PRUNE:
            continue
        right = n - m
        row_x1 = _pmf_row(m, r)
        row_x2 = _pmf_row(right, r)
        for x1, w_1 in enumerate(row_x1):
            if w_1 < PRUNE:
                continue
            for x2, w_2 in enumerate(row_x2):
                if w_2 < PRUNE:
                    continue
                weight = w_m * w_1 * w_2
                covered += weight
                if m == 0 or right == 0:
                    continue
                if stats.two_prop_tost(x1, m, x2, right, delta, alpha)["equivalent"]:
                    power += weight
    return power, covered


def table_from_rates(block):
    """analyze_study5._rates block -> (a, b, c, d) with rows
    disagree/agree and columns wrong/right."""
    a = block["caught"]
    b = block["false_alarms"]
    c = block["n_wrong"] - block["caught"]
    d = block["n_items"] - block["n_wrong"] - block["false_alarms"]
    return a, b, c, d


def project_h1(table, n):
    """Projection block for one pilot table at confirmatory size n."""
    a, b, c, d = table
    disagree, agree = a + b, c + d
    total = disagree + agree
    q, p1, p0 = adjusted(disagree, total), adjusted(a, disagree), adjusted(c, agree)
    power, covered = h1_power(n, q, p1, p0)
    out = {
        "pilot_table": {"wrong_disagree": a, "right_disagree": b,
                        "wrong_agree": c, "right_agree": d},
        "pilot_n": total,
        "n_projected": n,
        "adjusted": {"q": q, "p1": p1, "p0": p0},
        "power_adjusted": power,
        "mass_covered": covered,
        "power_raw": None,
    }
    if disagree > 0 and agree > 0:
        raw_power, raw_covered = h1_power(
            n, disagree / total, a / disagree, c / agree
        )
        out["raw"] = {"q": disagree / total, "p1": a / disagree, "p0": c / agree}
        out["power_raw"] = raw_power
        out["mass_covered_raw"] = raw_covered
    return out


def parse_pilot(records):
    """Paraphrase-arm parsed answers, keyed like the analyzer's index."""
    parsed = {}
    for record in records:
        meta = record_meta(record)
        if meta.get("control") or ("ok" in record and not record["ok"]):
            continue
        if meta.get("arm") == "resample":
            continue
        mode, obj = parse_response(record.get("text", record.get("response_text")))
        if mode == "fail":
            continue
        key = (meta["substrate"], meta["item_id"], meta["template_id"])
        parsed.setdefault(key, []).append(obj)
    return parsed


def build_projection(records, corpus, n_full=None):
    items = corpus["items"]
    n_full = n_full or len(items)
    n_pooled = sum(1 for it in items if it["gradient"] in POOLED_STRATA)
    parsed = parse_pilot(records)
    pair = REGISTERED_KSETS[0]
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "alpha": ALPHA, "power_bar": POWER_BAR,
        "n_full": n_full, "n_pooled_strata": n_pooled,
        "pair": list(pair),
        "h1_full": {}, "h1_pooled_strata": {},
    }
    for substrate in PRIMARY_CANDIDATES:
        block = kset_analysis(parsed, items, substrate, pair, lenient=False)
        out["h1_full"][substrate] = project_h1(table_from_rates(block), n_full)
        pooled = [0, 0, 0, 0]
        for gradient in POOLED_STRATA:
            sub = block["by_gradient"].get(gradient)
            if sub:
                pooled = [x + y for x, y in zip(pooled, table_from_rates(sub))]
        out["h1_pooled_strata"][substrate] = project_h1(tuple(pooled), n_pooled)

    # Section-6 rule, in its registered order.
    primary = None
    for substrate in PRIMARY_CANDIDATES:
        if out["h1_full"][substrate]["power_adjusted"] >= POWER_BAR:
            primary = {"substrate": substrate, "population": "full corpus",
                       "n": n_full}
            break
    if primary is None:
        for substrate in PRIMARY_CANDIDATES:
            if out["h1_pooled_strata"][substrate]["power_adjusted"] >= POWER_BAR:
                primary = {"substrate": substrate,
                           "population": "pooled near_tie+ambiguous strata",
                           "n": n_pooled}
                break
    out["primary"] = primary  # None = no branch reaches the bar: stop, do not freeze

    # H3: cross-door detector on t1, wrongness scored on sonnet_1p.
    door = cross_pair_analysis(parsed, items, CROSS_DOOR[0], CROSS_DOOR[1], "t1")
    n_door = door["n_items"]
    w = adjusted(door["n_wrong"], n_door)
    r = adjusted(door["n_disagree"], n_door)
    h3 = {
        "pilot": {"n_items": n_door, "n_wrong": door["n_wrong"],
                  "n_disagree": door["n_disagree"], "caught": door["caught"],
                  "false_alarms": door["false_alarms"]},
        "adjusted": {"wrong_rate": w, "door_disagree_rate": r},
        "bounds": {}, "delta": None,
    }
    for delta in H3_BOUNDS:
        power, covered = h3_power(n_full, w, r, delta)
        h3["bounds"][f"{delta:.2f}"] = {"power": power, "mass_covered": covered}
        if h3["delta"] is None and power >= POWER_BAR:
            h3["delta"] = delta
    out["h3"] = h3  # delta None = descriptive only, stated at freeze
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Study-5 pilot power projection")
    parser.add_argument("records", nargs="+", help="pilot run .jsonl file(s)")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args(argv)

    from harness.study5_fixtures import load_corpus

    projection = build_projection(load_records(args.records), load_corpus(args.corpus))
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.out, f"study5-power-{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(projection, handle, indent=2, sort_keys=True)
    for label in ("h1_full", "h1_pooled_strata"):
        for substrate, block in projection[label].items():
            raw = block["power_raw"]
            print(
                f"{label} {substrate}: pilot {block['pilot_table']} "
                f"-> power(adjusted)={block['power_adjusted']:.3f} "
                f"power(raw)={'n/a' if raw is None else f'{raw:.3f}'} "
                f"at n={block['n_projected']} (mass {block['mass_covered']:.6f})"
            )
    print(f"primary: {projection['primary']}")
    h3 = projection["h3"]
    print(f"h3 pilot {h3['pilot']} bounds {h3['bounds']} -> delta {h3['delta']}")
    print(f"projection -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
