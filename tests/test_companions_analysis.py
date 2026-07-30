"""Companion analyses (FOLLOWUP-COMPANIONS.md): logprob capture, churn A/B
report, margins report. Synthetic records only — stdlib, no network."""
import unittest

from harness.logprob_capture import compact_margins
from analysis.analyze_churn_ab import CHURN_CELL, build_churn_report
from analysis.analyze_margins import build_margins_report


def _lp_entry(token, logprob, tops):
    entry = {"token": token, "logprob": logprob}
    if tops is not None:
        entry["top_logprobs"] = [
            {"token": t, "logprob": lp} for t, lp in tops
        ]
    return entry


class TestCompactMargins(unittest.TestCase):
    def test_no_logprobs_returns_none(self):
        self.assertIsNone(compact_margins(None))
        self.assertIsNone(compact_margins({"message": {"content": "x"}}))

    def test_rows_min_margin_and_argmin(self):
        payload = {"logprobs": [
            _lp_entry("o", -0.01, [("o", -0.01), ("x", -4.2)]),
            _lp_entry("ut", -0.5, [("ut", -0.5), ("n", -0.9)]),
        ]}
        m = compact_margins(payload)
        self.assertEqual(m["n_tokens"], 2)
        self.assertEqual(m["tokens"][0], ["o", -0.01, -0.01, -4.2])
        self.assertEqual(m["tokens"][1], ["ut", -0.5, -0.5, -0.9])
        self.assertAlmostEqual(m["min_top2_margin"], 0.4, places=9)
        self.assertEqual(m["argmin_index"], 1)
        self.assertEqual(m["chosen_not_top1"], 0)

    def test_chosen_not_top1_counted(self):
        payload = {"logprobs": [
            _lp_entry("b", -0.7, [("a", -0.6), ("b", -0.7)]),
        ]}
        m = compact_margins(payload)
        self.assertEqual(m["chosen_not_top1"], 1)

    def test_missing_or_short_top_logprobs(self):
        payload = {"logprobs": [
            _lp_entry("a", -0.1, None),
            _lp_entry("b", -0.2, [("b", -0.2)]),
        ]}
        m = compact_margins(payload)
        self.assertEqual(m["tokens"][0], ["a", -0.1, None, None])
        self.assertEqual(m["tokens"][1], ["b", -0.2, -0.2, None])
        self.assertIsNone(m["min_top2_margin"])
        self.assertIsNone(m["argmin_index"])


def _churn_record(box, arm, text, load_ns, repeat, confirmed=True, ok=True,
                  stop="stop"):
    import hashlib

    record = {
        "schema": 3,
        "box": box,
        "cell": f"{CHURN_CELL}|arm={arm}",
        "repeat": repeat,
        "request_sha256": "sha-frozen",
        "wire_sha256": "sha-frozen",
        "meta_model": "gpt-oss-20b",
        "meta_task": "open_generation",
        "meta_sampling": "greedy",
        "meta_thinking": "effort_low",
        "meta_hardware": box,
        "meta_arm": arm,
        "ok": ok,
        "stop_reason": stop,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "usage": {"output_tokens": len(text), "load_duration_ns": load_ns},
        "response_model": "gpt-oss:20b",
    }
    if arm == "churn":
        record["pre_unload_confirmed"] = confirmed
        record["unload_wait_ms"] = 900
    return record


WARM = int(0.26e9)
COLD = int(6e9)


def _clean_records(churn_texts):
    records = [
        _churn_record("metal", "blocked", "AAA", WARM, r) for r in range(10)
    ]
    records += [
        _churn_record("metal", "churn", t, COLD, r)
        for r, t in enumerate(churn_texts)
    ]
    return records


class TestChurnReport(unittest.TestCase):
    def test_clean_ab_reports_diff_and_gates(self):
        records = _clean_records(["AAA"] * 8 + ["BBB"] * 2)
        report = build_churn_report(records)
        metal = report["boxes"]["metal"]
        gates = metal["gates"]
        self.assertTrue(gates["cross_arm_negative_control"])
        self.assertTrue(gates["manipulation"]["pass"])
        self.assertTrue(gates["manipulation"]["blocked_warm"])
        self.assertTrue(gates["manipulation"]["churn_all_confirmed"])
        self.assertTrue(gates["manipulation"]["churn_all_cold"])
        self.assertEqual(metal["arms"]["blocked"]["metrics"]["modal_share"], 1.0)
        self.assertEqual(metal["arms"]["churn"]["metrics"]["modal_share"], 0.8)
        diff = metal["churn_minus_blocked"]
        self.assertAlmostEqual(diff["diff"], -0.2, places=9)
        self.assertLess(diff["ci95"][0], diff["ci95"][1])

    def test_modal_match_against_confirmatory_report(self):
        import hashlib

        records = _clean_records(["AAA"] * 10)
        confirmatory = {
            "q1_cells": {
                f"metal::{CHURN_CELL}": {
                    "metrics": {
                        "modal_sha256": hashlib.sha256(b"AAA").hexdigest()
                    }
                }
            }
        }
        report = build_churn_report(records, confirmatory=confirmatory)
        metal = report["boxes"]["metal"]
        self.assertTrue(metal["arms"]["blocked"]["matches_confirmatory_modal"])
        self.assertTrue(metal["arms"]["churn"]["matches_confirmatory_modal"])

    def test_unconfirmed_unload_voids_manipulation_gate(self):
        records = _clean_records(["AAA"] * 10)
        records[-1] = _churn_record(
            "metal", "churn", "AAA", COLD, 9, confirmed=False
        )
        report = build_churn_report(records)
        gates = report["boxes"]["metal"]["gates"]
        self.assertFalse(gates["manipulation"]["churn_all_confirmed"])
        self.assertFalse(gates["manipulation"]["pass"])
        self.assertIsNone(report["boxes"]["metal"]["churn_minus_blocked"])

    def test_warm_churn_loads_void_manipulation_gate(self):
        records = [
            _churn_record("metal", "blocked", "AAA", WARM, r) for r in range(10)
        ] + [
            _churn_record("metal", "churn", "AAA", WARM, r) for r in range(10)
        ]
        report = build_churn_report(records)
        gates = report["boxes"]["metal"]["gates"]
        self.assertFalse(gates["manipulation"]["churn_all_cold"])
        self.assertFalse(gates["manipulation"]["pass"])

    def test_hot_blocked_arm_fails_warm_gate(self):
        records = [
            _churn_record("metal", "blocked", "AAA", COLD, r) for r in range(10)
        ] + [
            _churn_record("metal", "churn", "AAA", COLD * 20, r)
            for r in range(10)
        ]
        report = build_churn_report(records)
        self.assertFalse(
            report["boxes"]["metal"]["gates"]["manipulation"]["blocked_warm"]
        )

    def test_errors_and_warmups_excluded(self):
        records = _clean_records(["AAA"] * 10)
        records.append(
            _churn_record("metal", "churn", "ZZZ", COLD, 10, ok=False)
        )
        records.append({
            "schema": 3, "box": "metal", "cell": "warmup|gpt-oss-20b",
            "meta_control": "warmup", "meta_model": "gpt-oss-20b",
            "request_sha256": "w", "ok": True, "stop_reason": "stop",
            "text": "hi", "text_sha256": "x", "usage": {},
        })
        report = build_churn_report(records)
        metal = report["boxes"]["metal"]
        self.assertEqual(metal["arms"]["churn"]["metrics"]["n"], 10)
        self.assertEqual(metal["arms"]["churn"]["excluded"]["error"], 1)
        self.assertEqual(report["totals"]["warmups_excluded"], 1)


def _margins_record(box, cell, text, rows, repeat):
    import hashlib

    margins = None
    if rows is not None:
        defined = [
            (i, r[2] - r[3]) for i, r in enumerate(rows)
            if r[2] is not None and r[3] is not None
        ]
        margins = {
            "n_tokens": len(rows),
            "tokens": rows,
            "min_top2_margin": min((m for _, m in defined), default=None),
            "argmin_index": (
                min(defined, key=lambda im: im[1])[0] if defined else None
            ),
            "chosen_not_top1": sum(
                1 for r in rows if r[2] is not None and r[1] != r[2]
            ),
        }
    import hashlib as h

    return {
        "schema": 3,
        "box": box,
        "cell": cell,
        "repeat": repeat,
        "request_sha256": "sha-m",
        "meta_model": cell.split("|")[0],
        "meta_task": cell.split("|")[1],
        "meta_sampling": "greedy",
        "meta_thinking": cell.split("|")[3],
        "meta_hardware": box,
        "meta_exploratory": "margins",
        "ok": True,
        "stop_reason": "stop",
        "text": text,
        "text_sha256": h.sha256(text.encode()).hexdigest(),
        "usage": {"output_tokens": len(rows or [])},
        "logprob_margins": margins,
    }


SJ_CELL = "gpt-oss-20b|structured_json|greedy|effort_low|logprobs"


class TestMarginsReport(unittest.TestCase):
    def test_single_variant_cell_summary(self):
        rows = [
            ["a", -0.01, -0.01, -4.2],
            ["b", -0.2, -0.2, -0.5],
        ]
        records = [
            _margins_record("cuda", SJ_CELL, "ab", rows, r) for r in range(3)
        ]
        report = build_margins_report(records)
        cell = report["cells"][f"cuda::{SJ_CELL}"]
        self.assertEqual(cell["n"], 3)
        self.assertEqual(cell["variants"], 1)
        self.assertAlmostEqual(cell["min_margin"]["min"], 0.3, places=9)
        self.assertAlmostEqual(cell["min_margin"]["median"], 0.3, places=9)
        pooled = cell["positions_below"]
        self.assertEqual(pooled["0.001"], 0.0)
        self.assertEqual(pooled["0.01"], 0.0)
        self.assertEqual(pooled["0.1"], 0.0)
        self.assertIsNone(cell["fork"])
        self.assertEqual(cell["chosen_not_top1_total"], 0)

    def test_two_variant_cell_reports_fork_margins(self):
        rows_a = [
            ["x", -0.1, -0.1, -3.0],
            ["y", -0.6931, -0.6931, -0.6971],
        ]
        rows_b = [
            ["x", -0.1, -0.1, -3.0],
            ["z", -0.6971, -0.6971, -0.7031],
        ]
        records = [
            _margins_record("metal", SJ_CELL, "xy", rows_a, 0),
            _margins_record("metal", SJ_CELL, "xy", rows_a, 1),
            _margins_record("metal", SJ_CELL, "xz", rows_b, 2),
        ]
        report = build_margins_report(records)
        cell = report["cells"][f"metal::{SJ_CELL}"]
        self.assertEqual(cell["variants"], 2)
        fork = cell["fork"]
        self.assertEqual(fork["token_index"], 1)
        self.assertEqual(fork["modal"]["token"], "y")
        self.assertEqual(fork["alternate"]["token"], "z")
        self.assertAlmostEqual(fork["modal"]["margin"], 0.004, places=9)
        self.assertAlmostEqual(fork["alternate"]["margin"], 0.006, places=9)
        # a near-tie fork shows up in the pooled threshold fractions
        self.assertGreater(cell["positions_below"]["0.01"], 0.0)

    def test_rows_with_undefined_margins_are_excluded_from_pooling(self):
        rows = [
            ["a", -0.1, None, None],
            ["b", -0.2, -0.2, -0.9],
        ]
        records = [_margins_record("cuda", SJ_CELL, "ab", rows, 0)]
        report = build_margins_report(records)
        cell = report["cells"][f"cuda::{SJ_CELL}"]
        self.assertEqual(cell["defined_positions"], 1)
        self.assertAlmostEqual(cell["min_margin"]["min"], 0.7, places=9)


if __name__ == "__main__":
    unittest.main()
