"""Tests for analysis.analyze — validity gating and arm pooling.

The gates are the controls: a cell with mixed request hashes is invalidated
outright (harness variance, not Bedrock variance); truncated or errored
calls are excluded and counted, never silently dropped; a response-model
change inside a cell is flagged as drift but does not invalidate the cell.
"""
import unittest

from analysis.analyze import analyze_records, gate_cell, pool_success


def rec(
    cell="opus-5|classification|us|adaptive",
    repeat=0,
    text="BILLING",
    ok=True,
    stop_reason="end_turn",
    sha="AAAA",
    response_model="claude-opus-5",
    output_tokens=6,
    thinking_tokens=0,
):
    return {
        "schema": 1,
        "cell": cell,
        "repeat": repeat,
        "ok": ok,
        "stop_reason": stop_reason if ok else None,
        "request_sha256": sha,
        "response_model": response_model if ok else None,
        "text": text if ok else None,
        "usage": {
            "output_tokens": output_tokens,
            "output_tokens_details": {"thinking_tokens": thinking_tokens},
        }
        if ok
        else None,
        "meta_model": cell.split("|")[0],
        "meta_task": cell.split("|")[1],
        "meta_profile": cell.split("|")[2],
        "meta_thinking": cell.split("|")[3],
    }


class TestGateCell(unittest.TestCase):
    def test_excludes_errors_and_truncation_with_counts(self):
        records = [rec(repeat=i) for i in range(10)]
        records.append(rec(repeat=10, ok=False))
        records.append(rec(repeat=11, stop_reason="max_tokens"))
        gate = gate_cell(records)
        self.assertEqual(len(gate["valid"]), 10)
        self.assertEqual(gate["excluded"]["error"], 1)
        self.assertEqual(gate["excluded"]["truncated_or_other_stop"], 1)
        self.assertFalse(gate["flags"]["negative_control_failed"])
        self.assertFalse(gate["flags"]["model_drift"])

    def test_hash_mismatch_invalidates_cell(self):
        records = [rec(repeat=0, sha="AAAA"), rec(repeat=1, sha="BBBB")]
        gate = gate_cell(records)
        self.assertEqual(gate["valid"], [])
        self.assertTrue(gate["flags"]["negative_control_failed"])

    def test_model_drift_flagged_not_excluded(self):
        records = [
            rec(repeat=0, response_model="claude-opus-5"),
            rec(repeat=1, response_model="claude-opus-5-v2"),
        ]
        gate = gate_cell(records)
        self.assertEqual(len(gate["valid"]), 2)
        self.assertTrue(gate["flags"]["model_drift"])
        self.assertEqual(
            gate["flags"]["response_models"],
            ["claude-opus-5", "claude-opus-5-v2"],
        )


class TestAnalyzeRecords(unittest.TestCase):
    def _two_cells(self):
        us = "opus-5|classification|us|adaptive"
        gl = "opus-5|classification|global|adaptive"
        records = []
        for i in range(9):
            records.append(rec(cell=us, repeat=i, text="BILLING"))
        records.append(rec(cell=us, repeat=9, text="TECHNICAL"))
        for i in range(10):
            records.append(rec(cell=gl, repeat=i, text="BILLING"))
        return records, us, gl

    def test_cell_metrics_and_thinking_stats(self):
        records, us, gl = self._two_cells()
        result = analyze_records(records)
        self.assertAlmostEqual(
            result["cells"][us]["metrics"]["modal_share"], 0.9, places=12
        )
        self.assertAlmostEqual(
            result["cells"][gl]["metrics"]["modal_share"], 1.0, places=12
        )
        self.assertIn("wilson_ci", result["cells"][us])
        self.assertEqual(result["cells"][us]["thinking_tokens_mean"], 0.0)

    def test_pool_success_by_profile(self):
        records, us, gl = self._two_cells()
        result = analyze_records(records)
        x_us, n_us = pool_success(
            result["cells"], lambda meta: meta["profile"] == "us"
        )
        x_gl, n_gl = pool_success(
            result["cells"], lambda meta: meta["profile"] == "global"
        )
        self.assertEqual((x_us, n_us), (9, 10))
        self.assertEqual((x_gl, n_gl), (10, 10))


if __name__ == "__main__":
    unittest.main()
