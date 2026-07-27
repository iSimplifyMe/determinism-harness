"""Regression tests for the 2026-07-27 pilot incident: every worker thread
crashed on a tuple RNG seed (unsupported since Python 3.11) and the runner
still exited 0 with done=0. A run must be loud about incompleteness —
exit code and summary must reflect 'did all scheduled work happen', not
just 'did any recorded call fail'."""
import random
import unittest

from harness.runner import summary_is_complete, worker_seed


class TestWorkerSeed(unittest.TestCase):
    def test_seed_type_is_accepted_by_random(self):
        for i in range(8):
            random.Random(worker_seed(20260727, i))  # must not raise

    def test_seeds_distinct_across_workers(self):
        seeds = {worker_seed(20260727, i) for i in range(8)}
        self.assertEqual(len(seeds), 8)

    def test_deterministic(self):
        self.assertEqual(worker_seed(1, 2), worker_seed(1, 2))
        self.assertNotEqual(worker_seed(1, 2), worker_seed(2, 2))


class TestSummaryCompleteness(unittest.TestCase):
    def test_all_done_is_complete(self):
        self.assertTrue(
            summary_is_complete({"done": 800, "failures": 0, "fatal_worker_errors": []}, 800)
        )

    def test_zero_done_is_incomplete(self):
        self.assertFalse(
            summary_is_complete({"done": 0, "failures": 0, "fatal_worker_errors": []}, 800)
        )

    def test_partial_is_incomplete(self):
        self.assertFalse(
            summary_is_complete({"done": 799, "failures": 0, "fatal_worker_errors": []}, 800)
        )

    def test_fatal_worker_error_is_incomplete_even_if_counts_match(self):
        self.assertFalse(
            summary_is_complete(
                {"done": 800, "failures": 0, "fatal_worker_errors": ["boom"]}, 800
            )
        )


if __name__ == "__main__":
    unittest.main()
