"""Study-2 analysis: the stratified estimator (prereg v2 Q2 primary) and the
plane-aware comparisons. Includes a regression test that reproduces study 1's
recorded methodological miss — on heterogeneous strata the cross-stratum
pooled SE is a large multiple of the stratified SE, so the pooled TOST stays
inconclusive at a margin the stratified test plainly certifies."""
import unittest

from analysis.analyze import comparisons
from analysis.stats import stratified_diff, stratified_tost, two_prop_tost


class TestStratifiedDiff(unittest.TestCase):
    def test_known_value(self):
        strata = [(9, 10, 8, 10), (1, 10, 2, 10)]
        est = stratified_diff(strata)
        self.assertAlmostEqual(est["diff"], 0.0)
        self.assertAlmostEqual(est["se"], (0.05 ** 0.5) / 2, places=10)
        self.assertEqual(est["n_strata"], 2)

    def test_single_stratum_matches_wald(self):
        est = stratified_diff([(8, 10, 6, 10)])
        self.assertAlmostEqual(est["diff"], 0.2)
        expected_se = (0.8 * 0.2 / 10 + 0.6 * 0.4 / 10) ** 0.5
        self.assertAlmostEqual(est["se"], expected_se)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            stratified_diff([])


class TestStratifiedVsPooled(unittest.TestCase):
    """The study-1 lesson as an executable fact: cells near 0 and 1 make the
    pooled rate sit mid-range, inflating p(1-p) far beyond the true
    per-stratum sampling error."""

    STRATA = [(100, 100, 99, 100), (5, 100, 4, 100)]

    def test_pooled_se_is_inflated(self):
        strat = stratified_tost(self.STRATA, delta=0.05)
        pooled = two_prop_tost(105, 200, 103, 200, delta=0.05)
        self.assertAlmostEqual(strat["diff"], pooled["diff"], places=10)
        self.assertGreater(pooled["se"], 2 * strat["se"])

    def test_equivalence_verdicts_diverge(self):
        strat = stratified_tost(self.STRATA, delta=0.05)
        pooled = two_prop_tost(105, 200, 103, 200, delta=0.05)
        self.assertTrue(strat["equivalent"])
        self.assertFalse(pooled["equivalent"])


class TestStratifiedTost(unittest.TestCase):
    def test_all_degenerate_strata_rescued(self):
        result = stratified_tost([(10, 10, 10, 10), (10, 10, 10, 10)], delta=0.02)
        self.assertAlmostEqual(result["diff"], 0.0)
        self.assertGreater(result["se"], 0.0)

    def test_real_difference_never_equivalent(self):
        strata = [(95, 100, 60, 100), (90, 100, 55, 100)]
        result = stratified_tost(strata, delta=0.02)
        self.assertFalse(result["equivalent"])
        self.assertAlmostEqual(result["diff"], 0.35)

    def test_ci90_ordered_and_centered(self):
        result = stratified_tost([(50, 100, 48, 100)], delta=0.05)
        lo, hi = result["ci90"]
        self.assertLess(lo, result["diff"])
        self.assertGreater(hi, result["diff"])


def _cell(model, task, plane, thinking, modal, n, window="w1"):
    key = f"{window}::{model}|{task}|{plane}|{thinking}"
    return key, {
        "cell": f"{model}|{task}|{plane}|{thinking}",
        "meta": {
            "model": model,
            "task": task,
            "plane": plane,
            "thinking": thinking,
            "window": window,
        },
        "gate": {"n_raw": n, "excluded": {}, "flags": {}},
        "metrics": {"modal_count": modal, "n": n, "modal_share": modal / n},
    }


def _study2_cells():
    cells = {}
    spec = {
        # (task, plane, thinking): modal count out of 20
        ("structured_json", "bedrock", "adaptive"): 10,
        ("structured_json", "bedrock", "disabled"): 19,
        ("structured_json", "p_aws", "adaptive"): 11,
        ("structured_json", "p_aws", "disabled"): 19,
        ("classification", "bedrock", "adaptive"): 20,
        ("classification", "bedrock", "disabled"): 20,
        ("classification", "p_aws", "adaptive"): 20,
        ("classification", "p_aws", "disabled"): 19,
    }
    for (task, plane, thinking), modal in spec.items():
        key, entry = _cell("opus-5", task, plane, thinking, modal, 20)
        cells[key] = entry
    return cells


class TestPlaneComparisons(unittest.TestCase):
    def test_q1_attribution_shape(self):
        out = comparisons(_study2_cells(), delta=0.02)
        q1 = out["q1_attribution__opus-5"]
        self.assertEqual(sorted(q1["planes"]), ["bedrock", "p_aws"])
        bedrock = q1["planes"]["bedrock"]
        self.assertEqual(bedrock["adaptive"]["x"], 10)
        self.assertEqual(bedrock["disabled"]["x"], 19)
        self.assertAlmostEqual(bedrock["effect"]["diff"], (10 - 19) / 20)
        dod = q1["dod"]["bedrock_vs_p_aws"]
        # bedrock effect -0.45, p_aws effect -0.40 -> dod -0.05
        self.assertAlmostEqual(dod["dod"], -0.05)
        self.assertEqual(len(dod["ci95"]), 2)

    def test_q2_stratified_primary_present(self):
        out = comparisons(_study2_cells(), delta=0.02)
        q2 = out["q2_plane__opus-5__bedrock_vs_p_aws"]
        # 2 tasks x 2 thinking arms x 1 window, matched across both planes
        self.assertEqual(q2["tost_stratified_primary"]["n_strata"], 4)
        self.assertIn("tost_pooled_sensitivity", q2)
        self.assertEqual(q2["bedrock"]["n"], 80)
        self.assertEqual(q2["p_aws"]["n"], 80)

    def test_single_model_has_no_all_scope(self):
        out = comparisons(_study2_cells(), delta=0.02)
        self.assertNotIn("q2_plane__ALL__bedrock_vs_p_aws", out)

    def test_study1_shaped_cells_unaffected(self):
        cells = {}
        for profile in ("us", "global"):
            key = f"w1::opus-5|classification|{profile}|adaptive"
            cells[key] = {
                "cell": f"opus-5|classification|{profile}|adaptive",
                "meta": {
                    "model": "opus-5",
                    "task": "classification",
                    "profile": profile,
                    "thinking": "adaptive",
                    "window": "w1",
                },
                "gate": {"n_raw": 20, "excluded": {}, "flags": {}},
                "metrics": {"modal_count": 20, "n": 20, "modal_share": 1.0},
            }
        out = comparisons(cells, delta=0.02)
        self.assertIn("q2_profile__opus-5", out)
        self.assertFalse(any(k.startswith("q1_attribution") for k in out))
        self.assertFalse(any(k.startswith("q2_plane") for k in out))


if __name__ == "__main__":
    unittest.main()
