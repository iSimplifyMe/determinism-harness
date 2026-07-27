"""Tests for analysis.stats — every statistical claim in the paper flows through
these functions, so reference values are hand-derived or exact by combinatorics."""
import unittest

from analysis.stats import (
    Z95,
    binom_pmf,
    binom_test,
    diff_ci,
    normal_cdf,
    two_prop_tost,
    wilson_interval,
)


class TestNormalCdf(unittest.TestCase):
    def test_zero_is_half(self):
        self.assertAlmostEqual(normal_cdf(0.0), 0.5, places=12)

    def test_z95_quantile(self):
        # 1.959963984540054 is the 97.5th percentile of the standard normal
        self.assertAlmostEqual(normal_cdf(Z95), 0.975, places=9)

    def test_symmetry(self):
        self.assertAlmostEqual(normal_cdf(-1.2345) + normal_cdf(1.2345), 1.0, places=12)


class TestWilson(unittest.TestCase):
    def test_known_value_8_of_10(self):
        # Hand-derived: center 0.716740, half-width 0.226573
        lo, hi = wilson_interval(8, 10)
        self.assertAlmostEqual(lo, 0.490167, delta=5e-4)
        self.assertAlmostEqual(hi, 0.943313, delta=5e-4)

    def test_zero_successes_lower_bound_is_zero(self):
        lo, hi = wilson_interval(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=12)
        self.assertGreater(hi, 0.0)

    def test_all_successes_upper_bound_is_one(self):
        lo, hi = wilson_interval(10, 10)
        self.assertAlmostEqual(hi, 1.0, places=12)
        self.assertLess(lo, 1.0)

    def test_contains_point_estimate(self):
        for k, n in [(1, 7), (50, 100), (99, 100), (3, 20)]:
            lo, hi = wilson_interval(k, n)
            self.assertLessEqual(lo, k / n)
            self.assertGreaterEqual(hi, k / n)


class TestBinomial(unittest.TestCase):
    def test_pmf_exact_half(self):
        # C(10,5)/2^10 = 252/1024
        self.assertAlmostEqual(binom_pmf(5, 10, 0.5), 0.24609375, places=10)

    def test_pmf_nonhalf(self):
        # C(10,3) * 0.3^3 * 0.7^7 = 0.266827932
        self.assertAlmostEqual(binom_pmf(3, 10, 0.3), 0.266827932, places=8)

    def test_greater_tail_exact(self):
        # P(X >= 8 | n=10, p=0.5) = (45 + 10 + 1)/1024
        self.assertAlmostEqual(binom_test(8, 10, 0.5, "greater"), 0.0546875, places=10)

    def test_less_tail_exact(self):
        # P(X <= 0 | n=10, p=0.5) = 1/1024
        self.assertAlmostEqual(binom_test(0, 10, 0.5, "less"), 0.0009765625, places=12)

    def test_two_sided_symmetric(self):
        # At p=0.5 the distribution is symmetric: two-sided = 2 * one-sided
        self.assertAlmostEqual(binom_test(8, 10, 0.5, "two-sided"), 0.109375, places=9)


class TestTost(unittest.TestCase):
    def test_underpowered_null_is_not_equivalent(self):
        # Equal observed rates, n=10k per arm, delta=1pp: z = 1.414, p ~ 0.0786
        r = two_prop_tost(5000, 10000, 5000, 10000, delta=0.01)
        self.assertAlmostEqual(r["p"], 0.0786, delta=2e-3)
        self.assertFalse(r["equivalent"])

    def test_powered_null_is_equivalent(self):
        r = two_prop_tost(50000, 100000, 50000, 100000, delta=0.01)
        self.assertLess(r["p"], 1e-4)
        self.assertTrue(r["equivalent"])

    def test_real_difference_never_equivalent(self):
        # True diff 2pp > delta 1pp: must not claim equivalence at any n
        r = two_prop_tost(5200, 10000, 5000, 10000, delta=0.01)
        self.assertFalse(r["equivalent"])
        self.assertGreater(r["p_upper"], 0.5)

    def test_degenerate_all_successes_does_not_crash(self):
        r = two_prop_tost(1000, 1000, 1000, 1000, delta=0.01)
        self.assertTrue(r["equivalent"])
        self.assertAlmostEqual(r["diff"], 0.0, places=12)

    def test_ci90_present_and_ordered(self):
        r = two_prop_tost(5000, 10000, 5000, 10000, delta=0.01)
        lo, hi = r["ci90"]
        self.assertLess(lo, hi)


class TestDiffCi(unittest.TestCase):
    def test_known_value(self):
        # 0.8 vs 0.7, n=100 each: 0.10 +/- 1.96 * 0.060828
        lo, hi = diff_ci(80, 100, 70, 100)
        self.assertAlmostEqual(lo, -0.019222, delta=1e-3)
        self.assertAlmostEqual(hi, 0.219222, delta=1e-3)
        self.assertLess(lo, 0.10)
        self.assertGreater(hi, 0.10)


if __name__ == "__main__":
    unittest.main()
