"""Tests for the exploratory latency-fingerprint addendum (study 2)."""
import json
import unittest

from analysis.latency_fingerprint import (
    aggregate_pair_distances,
    binned_median_regression,
    build_report,
    center,
    collect_strata,
    decode_slopes,
    eligible,
    ks_stat,
    pair_distances,
    percentile,
    prefill_slopes,
    stratum_key,
    summarize,
)


def _rec(latency=1000, out_tok=100, in_tok=233, plane="bedrock", model="opus-5",
         task="structured_json", thinking="adaptive", window="peak",
         mode="study2-full", streamed=False, ok=True, attempts=1):
    return {
        "ok": ok,
        "attempts": attempts,
        "latency_ms": latency,
        "plane": plane,
        "meta_model": model,
        "meta_task": task,
        "meta_thinking": thinking,
        "window": window,
        "mode": mode,
        "delivered_streaming": streamed,
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "service_tier": "standard",
        },
    }


class TestEligibility(unittest.TestCase):
    def test_ok_single_attempt_included(self):
        self.assertTrue(eligible(_rec()))

    def test_failed_call_excluded(self):
        self.assertFalse(eligible(_rec(ok=False)))

    def test_retried_call_excluded(self):
        self.assertFalse(eligible(_rec(attempts=2)))

    def test_missing_latency_excluded(self):
        record = _rec()
        record["latency_ms"] = None
        self.assertFalse(eligible(record))


class TestStratumKey(unittest.TestCase):
    def test_key_carries_all_factors(self):
        key = stratum_key(_rec())
        self.assertEqual(key, "peak::opus-5|structured_json|adaptive|request")

    def test_streamed_delivery_marked(self):
        key = stratum_key(_rec(streamed=True, window="control"))
        self.assertEqual(key, "control::opus-5|structured_json|adaptive|streamed")


class TestStats(unittest.TestCase):
    def test_percentile_endpoints_and_median(self):
        values = list(range(101))  # 0..100
        self.assertEqual(percentile(values, 0), 0)
        self.assertEqual(percentile(values, 50), 50)
        self.assertEqual(percentile(values, 100), 100)

    def test_ks_identical_is_zero(self):
        a = [1, 1, 2, 3, 5, 8]
        self.assertEqual(ks_stat(a, list(a)), 0.0)

    def test_ks_disjoint_is_one(self):
        self.assertEqual(ks_stat([1, 2, 3], [10, 11, 12]), 1.0)

    def test_ks_handles_ties(self):
        # Half of b sits exactly on a's values; D = 0.5, not an artifact of tie order.
        self.assertAlmostEqual(ks_stat([1, 1, 2, 2], [1, 2, 3, 4]), 0.5)

    def test_center_zeroes_median(self):
        centered = center([10, 20, 30, 40, 1000])
        centered.sort()
        self.assertEqual(centered[2], 0)

    def test_centered_ks_removes_pure_location_shift(self):
        a = [100, 110, 120, 130, 140, 200]
        b = [x + 5000 for x in a]
        self.assertEqual(ks_stat(a, b), 1.0)
        self.assertEqual(ks_stat(center(a), center(b)), 0.0)


class TestBinnedMedianRegression(unittest.TestCase):
    def test_recovers_line_through_bin_medians(self):
        pairs = []
        for x in (10, 50, 200):
            base = 500 + 20 * x
            # odd count per bin, one wild outlier: the bin median ignores it
            pairs += [
                (x, base - 2), (x, base - 1), (x, base),
                (x, base + 1), (x, base + 90000),
            ]
        fit = binned_median_regression(pairs)
        self.assertAlmostEqual(fit["slope_ms_per_token"], 20.0, places=6)
        self.assertAlmostEqual(fit["intercept_ms"], 500.0, places=6)
        self.assertEqual(fit["n"], 15)
        self.assertEqual(fit["n_bins"], 3)

    def test_single_bin_returns_none(self):
        self.assertIsNone(binned_median_regression([(50, 1), (50, 2)]))


class TestGrouping(unittest.TestCase):
    def test_collect_strata_groups_by_plane(self):
        records = [
            _rec(plane="bedrock"),
            _rec(plane="p_aws"),
            _rec(plane="bedrock", ok=False),  # ineligible, dropped
        ]
        strata = collect_strata(records)
        key = "peak::opus-5|structured_json|adaptive|request"
        self.assertEqual(set(strata[key]), {"bedrock", "p_aws"})
        self.assertEqual(len(strata[key]["bedrock"]), 1)

    def test_decode_slopes_use_full_mode_only(self):
        records = []
        for x, y in ((10, 700), (100, 2500), (400, 8500)):
            records.append(_rec(out_tok=x, latency=y))
        records.append(_rec(out_tok=999, latency=1, mode="study2-q4-lengths"))
        records.append(_rec(out_tok=999, latency=1, streamed=True))
        slopes = decode_slopes(records)
        self.assertEqual(list(slopes), ["bedrock|opus-5|peak"])
        self.assertAlmostEqual(slopes["bedrock|opus-5|peak"]["slope_ms_per_token"], 20.0)

    def test_prefill_slopes_use_q4_only(self):
        records = [
            _rec(in_tok=1000, latency=2000, mode="study2-q4-lengths"),
            _rec(in_tok=10000, latency=11000, mode="study2-q4-lengths"),
            _rec(in_tok=50000, latency=51000, mode="study2-q4-lengths"),
            _rec(in_tok=999999, latency=1, mode="study2-full"),
        ]
        slopes = prefill_slopes(records)
        self.assertEqual(list(slopes), ["bedrock|opus-5"])
        self.assertAlmostEqual(slopes["bedrock|opus-5"]["slope_ms_per_token"], 1.0)


class TestPairDistances(unittest.TestCase):
    def test_three_planes_give_three_pairs(self):
        records = (
            [_rec(plane="bedrock", latency=1000 + i) for i in range(20)]
            + [_rec(plane="p_aws", latency=1000 + i) for i in range(20)]
            + [_rec(plane="anthropic_api", latency=9000 + i * 40) for i in range(20)]
        )
        strata = collect_strata(records)
        distances = pair_distances(strata)
        key = "peak::opus-5|structured_json|adaptive|request"
        pairs = distances[key]
        self.assertEqual(
            set(pairs),
            {"bedrock|p_aws", "anthropic_api|bedrock", "anthropic_api|p_aws"},
        )
        self.assertEqual(pairs["bedrock|p_aws"]["ks_raw"], 0.0)
        self.assertEqual(pairs["anthropic_api|bedrock"]["ks_raw"], 1.0)
        # 1P differs in scale too, so centering must NOT erase the difference
        # (pure symmetric scale mismatch gives KS exactly 0.5 here)
        self.assertGreaterEqual(pairs["anthropic_api|bedrock"]["ks_centered"], 0.5)

    def test_two_plane_stratum_skips_missing_pair(self):
        records = [_rec(plane="bedrock"), _rec(plane="p_aws")]
        distances = pair_distances(collect_strata(records))
        key = "peak::opus-5|structured_json|adaptive|request"
        self.assertEqual(set(distances[key]), {"bedrock|p_aws"})

    def test_aggregate_reports_closest_pair_votes(self):
        distances = {
            "s1": {
                "bedrock|p_aws": {"ks_raw": 0.1, "ks_centered": 0.05, "n_min": 9},
                "anthropic_api|bedrock": {"ks_raw": 0.9, "ks_centered": 0.6, "n_min": 9},
                "anthropic_api|p_aws": {"ks_raw": 0.8, "ks_centered": 0.5, "n_min": 9},
            },
            "s2": {
                "bedrock|p_aws": {"ks_raw": 0.2, "ks_centered": 0.1, "n_min": 9},
                "anthropic_api|bedrock": {"ks_raw": 0.7, "ks_centered": 0.4, "n_min": 9},
                "anthropic_api|p_aws": {"ks_raw": 0.6, "ks_centered": 0.3, "n_min": 9},
            },
        }
        agg = aggregate_pair_distances(distances)
        self.assertEqual(agg["strata"], 2)
        self.assertEqual(agg["closest_pair_votes_centered"]["bedrock|p_aws"], 2)
        self.assertAlmostEqual(
            agg["median_ks_centered"]["bedrock|p_aws"], 0.075
        )


class TestBuildReport(unittest.TestCase):
    def test_end_to_end_synthetic(self):
        records = []
        for plane, base in (("bedrock", 1200), ("p_aws", 1250), ("anthropic_api", 700)):
            for task, tok in (("classification", 5), ("structured_json", 90),
                              ("open_generation", 600)):
                for i in range(12):
                    records.append(
                        _rec(plane=plane, task=task, out_tok=tok + i,
                             latency=base + 15 * (tok + i) + i)
                    )
        report = build_report(records)
        self.assertEqual(report["exploratory"], True)
        self.assertEqual(report["totals"]["eligible"], len(records))
        json.dumps(report)  # must be serializable as-is
        self.assertIn("decode_slopes", report)
        self.assertIn("pair_distances_aggregate", report)
        self.assertIn("covariates", report)


if __name__ == "__main__":
    unittest.main()
