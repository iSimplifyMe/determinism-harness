"""Study-5 registered estimators and the pilot power projection.

Reference values are published (Newcombe 1998), exact by combinatorics
(Fisher via math.comb, an independent path from the lgamma
implementation), or checked against a seeded simulation of the same
sampling model (an independent path from the enumeration)."""
import json
import math
import random
import unittest
from fractions import Fraction

from analysis.stats import fisher_exact, newcombe_diff_ci, two_prop_tost
from analysis.study5_power import (
    POWER_BAR,
    adjusted,
    build_projection,
    h1_power,
    h3_power,
    project_h1,
    table_from_rates,
)


def brute_fisher(a, b, c, d):
    """Two-sided Fisher by exact rational arithmetic."""
    row1, col1, n = a + b, a + c, a + b + c + d
    lo, hi = max(0, row1 + col1 - n), min(row1, col1)
    denom = math.comb(n, col1)

    def pmf(x):
        return Fraction(math.comb(row1, x) * math.comb(n - row1, col1 - x), denom)

    observed = pmf(a)
    return float(sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= observed))


class TestNewcombe(unittest.TestCase):
    def test_published_example(self):
        # Newcombe 1998, 56/70 vs 48/80, method 10: 0.0524 to 0.3339
        lower, upper = newcombe_diff_ci(56, 70, 48, 80)
        self.assertAlmostEqual(lower, 0.0524, places=4)
        self.assertAlmostEqual(upper, 0.3339, places=4)

    def test_antisymmetric_under_arm_swap(self):
        lower, upper = newcombe_diff_ci(9, 12, 20, 138)
        swapped_lower, swapped_upper = newcombe_diff_ci(20, 138, 9, 12)
        self.assertAlmostEqual(lower, -swapped_upper, places=12)
        self.assertAlmostEqual(upper, -swapped_lower, places=12)

    def test_defined_and_bounded_at_the_edges(self):
        lower, upper = newcombe_diff_ci(5, 5, 0, 40)
        self.assertGreater(lower, 0.0)
        self.assertLessEqual(upper, 1.0)
        lower, upper = newcombe_diff_ci(0, 6, 0, 40)
        self.assertLess(lower, 0.0)
        self.assertGreater(upper, 0.0)

    def test_rejects_empty_arm(self):
        with self.assertRaises(ValueError):
            newcombe_diff_ci(0, 0, 3, 10)


class TestFisherExact(unittest.TestCase):
    def test_classic_table(self):
        # [[3, 1], [1, 3]]: 34/70
        self.assertAlmostEqual(fisher_exact(3, 1, 1, 3), 34 / 70, places=12)

    def test_matches_exact_rational_enumeration(self):
        tables = [
            (8, 2, 1, 5), (9, 3, 20, 118), (0, 4, 30, 116), (12, 0, 18, 120),
            (1, 1, 1, 1), (5, 20, 6, 119), (7, 7, 7, 7), (2, 19, 0, 0),
            (15, 9, 21, 105), (3, 0, 0, 147),
        ]
        for table in tables:
            with self.subTest(table=table):
                self.assertAlmostEqual(
                    fisher_exact(*table), brute_fisher(*table), places=9
                )

    def test_symmetric_under_row_and_column_swap(self):
        base = fisher_exact(9, 3, 20, 118)
        self.assertAlmostEqual(fisher_exact(20, 118, 9, 3), base, places=12)
        self.assertAlmostEqual(fisher_exact(3, 9, 118, 20), base, places=12)

    def test_empty_row_or_column_is_one(self):
        self.assertEqual(fisher_exact(0, 0, 10, 140), 1.0)
        self.assertEqual(fisher_exact(0, 12, 0, 138), 1.0)

    def test_rejects_bad_tables(self):
        with self.assertRaises(ValueError):
            fisher_exact(0, 0, 0, 0)
        with self.assertRaises(ValueError):
            fisher_exact(-1, 2, 3, 4)


class TestH1Power(unittest.TestCase):
    def test_null_stays_under_alpha(self):
        power, covered = h1_power(60, 0.2, 0.3, 0.3)
        self.assertLess(power, 0.05)
        self.assertGreater(covered, 0.9999)

    def test_large_effect_is_near_certain(self):
        power, covered = h1_power(150, 0.2, 0.8, 0.05)
        self.assertGreater(power, 0.999)
        self.assertGreater(covered, 0.9999)

    def test_monotone_in_n(self):
        small, _ = h1_power(40, 0.15, 0.5, 0.1)
        large, _ = h1_power(120, 0.15, 0.5, 0.1)
        self.assertLess(small, large)

    def test_wrong_direction_never_counts(self):
        power, _ = h1_power(80, 0.2, 0.05, 0.6)
        self.assertLess(power, 1e-6)

    def test_matches_seeded_simulation(self):
        n, q, p1, p0 = 60, 0.2, 0.55, 0.12
        exact, _ = h1_power(n, q, p1, p0)
        rng = random.Random(20261001)
        draws, hits = 4000, 0
        for _ in range(draws):
            a = b = c = d = 0
            for _ in range(n):
                if rng.random() < q:
                    if rng.random() < p1:
                        a += 1
                    else:
                        b += 1
                elif rng.random() < p0:
                    c += 1
                else:
                    d += 1
            if a + b and c + d and a / (a + b) > c / (c + d) \
                    and fisher_exact(a, b, c, d) < 0.05:
                hits += 1
        self.assertAlmostEqual(exact, hits / draws, delta=0.03)


class TestH3Power(unittest.TestCase):
    def test_monotone_in_delta(self):
        tight, _ = h3_power(150, 0.2, 0.01, 0.05)
        loose, covered = h3_power(150, 0.2, 0.01, 0.15)
        self.assertLess(tight, loose)
        self.assertGreater(covered, 0.9999)

    def test_silent_detector_is_equivalent_at_a_wide_bound(self):
        power, _ = h3_power(150, 0.2, 0.002, 0.15)
        self.assertGreater(power, 0.95)

    def test_agrees_with_the_registered_tost_on_one_table(self):
        # n_wrong certain-ish at w ~ 1 is degenerate; check the predicate
        # the enumeration applies, on a concrete table.
        self.assertTrue(two_prop_tost(0, 30, 0, 120, 0.10)["equivalent"])
        self.assertFalse(two_prop_tost(6, 30, 0, 120, 0.10)["equivalent"])


def _item(item_id, gradient, name):
    return {
        "id": item_id, "gradient": gradient, "target_field": "item_name",
        "document": f"doc {item_id}",
        "ground_truth": {"item_name": name, "unit_price": 1.0,
                         "quantity_in_stock": 1},
        "acceptable_alternatives": [], "rationale": "r",
    }


def _record(substrate, item_id, template_id, name):
    return {
        "ok": True, "meta_substrate": substrate, "meta_arm": "paraphrase",
        "meta_item_id": item_id, "meta_template_id": template_id,
        "text": json.dumps({"item_name": name, "unit_price": 1.0,
                            "quantity_in_stock": 1}),
    }


def _pilot(spec):
    """spec: {substrate: [(gradient, t1_right, t2_same_as_t1), ...]} on a
    shared item list; sonnet_bedrock mirrors sonnet_1p (doors agree)."""
    n = len(next(iter(spec.values())))
    gradients = [g for g, _, _ in next(iter(spec.values()))]
    items = [_item(f"i{k}", gradients[k], f"Truth {k}") for k in range(n)]
    records = []
    for substrate, rows in spec.items():
        for k, (_, t1_right, same) in enumerate(rows):
            first = f"Truth {k}" if t1_right else f"Wrong {k}"
            second = first if same else f"Other {k}"
            records.append(_record(substrate, f"i{k}", "t1", first))
            records.append(_record(substrate, f"i{k}", "t2", second))
            if substrate == "sonnet_1p":
                records.append(_record("sonnet_bedrock", f"i{k}", "t1", first))
    return records, {"items": items, "meta": {}}


class TestProjection(unittest.TestCase):
    def test_adjusted_never_degenerate(self):
        self.assertEqual(adjusted(0, 0), 0.5)
        self.assertGreater(adjusted(0, 21), 0.0)
        self.assertLess(adjusted(21, 21), 1.0)

    def test_table_from_rates(self):
        block = {"n_items": 21, "n_wrong": 6, "caught": 4, "false_alarms": 1}
        self.assertEqual(table_from_rates(block), (4, 1, 2, 14))

    def test_project_h1_reports_raw_only_when_defined(self):
        self.assertIsNone(project_h1((0, 0, 3, 18), 150)["power_raw"])
        self.assertIsNotNone(project_h1((4, 1, 2, 14), 150)["power_raw"])

    def test_rule_prefers_sonnet_when_it_clears_the_bar(self):
        strong = [("near_tie", False, False)] * 5 + [("clean", True, True)] * 16
        records, corpus = _pilot({"sonnet_1p": strong, "haiku_1p": strong})
        corpus["items"] *= 1  # 21 pilot items stand in for the corpus
        out = build_projection(records, corpus, n_full=150)
        self.assertGreaterEqual(
            out["h1_full"]["sonnet_1p"]["power_adjusted"], POWER_BAR
        )
        self.assertEqual(out["primary"]["substrate"], "sonnet_1p")
        self.assertEqual(out["primary"]["population"], "full corpus")

    def test_rule_falls_to_haiku_then_to_none(self):
        silent = [("near_tie", True, True)] * 5 + [("clean", True, True)] * 16
        strong = [("near_tie", False, False)] * 5 + [("clean", True, True)] * 16
        records, corpus = _pilot({"sonnet_1p": silent, "haiku_1p": strong})
        out = build_projection(records, corpus, n_full=150)
        self.assertEqual(out["primary"]["substrate"], "haiku_1p")
        records, corpus = _pilot({"sonnet_1p": silent, "haiku_1p": silent})
        out = build_projection(records, corpus, n_full=150)
        self.assertIsNone(out["primary"])

    def test_h3_block_counts_the_door_pair(self):
        rows = [("near_tie", False, False)] * 5 + [("clean", True, True)] * 16
        records, corpus = _pilot({"sonnet_1p": rows, "haiku_1p": rows})
        out = build_projection(records, corpus, n_full=150)
        self.assertEqual(out["h3"]["pilot"]["n_items"], 21)
        self.assertEqual(out["h3"]["pilot"]["n_disagree"], 0)
        self.assertEqual(out["h3"]["pilot"]["n_wrong"], 5)
        self.assertIn("0.10", out["h3"]["bounds"])


if __name__ == "__main__":
    unittest.main()
