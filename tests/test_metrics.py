"""Tests for analysis.metrics — deterministic measurement, no model in the loop."""
import unittest

from analysis.metrics import (
    cell_metrics,
    first_divergence_index,
    levenshtein_banded,
    modal_response,
    normalized_distance,
    pop_variance,
    token_divergence_index,
)


class TestModal(unittest.TestCase):
    def test_simple_majority(self):
        self.assertEqual(modal_response(["x", "x", "y"]), ("x", 2))

    def test_tie_breaks_lexicographically(self):
        self.assertEqual(modal_response(["b", "a", "a", "b"]), ("a", 2))

    def test_single(self):
        self.assertEqual(modal_response(["only"]), ("only", 1))


class TestDivergenceIndex(unittest.TestCase):
    def test_char_index(self):
        self.assertEqual(first_divergence_index("abc", "abd"), 2)

    def test_prefix_case(self):
        self.assertEqual(first_divergence_index("abc", "abcd"), 3)

    def test_equal_is_none(self):
        self.assertIsNone(first_divergence_index("abc", "abc"))

    def test_empty_vs_nonempty(self):
        self.assertEqual(first_divergence_index("", "x"), 0)

    def test_token_index(self):
        self.assertEqual(token_divergence_index("a b c", "a b d"), 2)
        self.assertIsNone(token_divergence_index("a b", "a b"))
        self.assertEqual(token_divergence_index("a b", "a b c"), 2)


class TestLevenshtein(unittest.TestCase):
    def test_classic(self):
        self.assertEqual(levenshtein_banded("kitten", "sitting"), (3, False))

    def test_empty(self):
        self.assertEqual(levenshtein_banded("", "abc"), (3, False))
        self.assertEqual(levenshtein_banded("", ""), (0, False))

    def test_equal(self):
        self.assertEqual(levenshtein_banded("same", "same"), (0, False))

    def test_shared_affixes_stripped(self):
        self.assertEqual(levenshtein_banded("xxxxabcyyyy", "xxxxabdyyyy"), (1, False))

    def test_cap_on_massive_divergence(self):
        a = "ab" * 1000
        b = "cd" * 1000
        self.assertEqual(levenshtein_banded(a, b, cap=512), (512, True))

    def test_cap_via_length_difference(self):
        self.assertEqual(levenshtein_banded("a" * 1000, "", cap=512), (512, True))

    def test_normalized(self):
        d, capped = normalized_distance("kitten", "sitting")
        self.assertAlmostEqual(d, 3 / 7, places=9)
        self.assertFalse(capped)


class TestPopVariance(unittest.TestCase):
    def test_constant(self):
        self.assertEqual(pop_variance([2, 2, 2]), 0.0)

    def test_two_point(self):
        self.assertAlmostEqual(pop_variance([1, 3]), 1.0, places=12)


class TestCellMetrics(unittest.TestCase):
    def _records(self):
        recs = [{"text": "X", "output_tokens": 5} for _ in range(9)]
        recs.append({"text": "Y", "output_tokens": 7})
        return recs

    def test_nine_of_ten_identical(self):
        m = cell_metrics(self._records())
        self.assertEqual(m["n"], 10)
        self.assertAlmostEqual(m["modal_share"], 0.9, places=12)
        self.assertEqual(m["distinct_count"], 2)
        self.assertFalse(m["all_identical"])
        # (C(9,2) + C(1,2)) / C(10,2) = 36/45
        self.assertAlmostEqual(m["pairwise_agreement"], 0.8, places=12)
        self.assertEqual(m["divergence_char_indices"], [0])
        self.assertAlmostEqual(m["norm_distance_max"], 1.0, places=12)
        self.assertAlmostEqual(m["norm_distance_mean_all"], 0.1, places=12)
        self.assertAlmostEqual(m["output_token_variance"], 0.36, places=12)

    def test_all_identical(self):
        recs = [{"text": "Z", "output_tokens": 4} for _ in range(5)]
        m = cell_metrics(recs)
        self.assertTrue(m["all_identical"])
        self.assertAlmostEqual(m["modal_share"], 1.0, places=12)
        self.assertEqual(m["distinct_count"], 1)
        self.assertAlmostEqual(m["pairwise_agreement"], 1.0, places=12)
        self.assertEqual(m["divergence_char_indices"], [])


if __name__ == "__main__":
    unittest.main()
