"""analyze_study4 estimator tests — synthetic records with known answers."""
import unittest

from analysis.analyze_study4 import (
    analyze,
    fisher_exact,
    group_cells,
    q1_door_attribution,
    q3_effort_analog,
    q5_default_burn,
)


def rec(door, task, effort, sha, window="low", ok=True, usage=None, text="t"):
    return {
        "mode": "study4-full", "window": window, "ok": ok,
        "meta_door": door, "meta_task": task, "meta_effort": effort,
        "text": text, "text_sha256": sha, "usage": usage or {},
    }


def burst(door, task, effort, sha, count, **kw):
    return [rec(door, task, effort, sha, **kw) for _ in range(count)]


class FisherTest(unittest.TestCase):
    def test_discovery_table(self):
        # The discovery 1P/codex table: 11/9 vs 3/17 -> p ~= 0.0187.
        p = fisher_exact(11, 9, 3, 17)
        self.assertAlmostEqual(p, 0.0187, places=3)

    def test_identical_tables_are_p1(self):
        self.assertAlmostEqual(fisher_exact(10, 10, 10, 10), 1.0, places=9)


class Q1Test(unittest.TestCase):
    def test_shares_and_registered_pair(self):
        records = (
            burst("openai_1p", "structured_json", "none", "AA", 11)
            + burst("openai_1p", "structured_json", "none", "BB", 9)
            + burst("codex_sub", "structured_json", "none", "AA", 3)
            + burst("codex_sub", "structured_json", "none", "BB", 17)
        )
        q1 = q1_door_attribution(group_cells(records))
        # Modal variant study-wide: BB (26) over AA (14).
        self.assertEqual(q1["doors"]["openai_1p"]["modal_variant_share"], 9 / 20)
        self.assertEqual(q1["doors"]["codex_sub"]["modal_variant_share"], 17 / 20)
        pair = q1["pairwise"]["openai_1p_vs_codex_sub"]
        self.assertTrue(pair["registered_direction"])
        self.assertAlmostEqual(pair["fisher_p"], 0.0187, places=3)


class Q3Test(unittest.TestCase):
    def test_dod_arithmetic(self):
        records = (
            # door A: high 50/100 modal, none 100/100 -> diff -0.5
            burst("mantle", "structured_json", "high", "AA", 50)
            + burst("mantle", "structured_json", "high", "BB", 50)
            + burst("mantle", "structured_json", "none", "AA", 100)
            # door B: high == none -> diff 0
            + burst("runtime_us", "structured_json", "high", "AA", 100)
            + burst("runtime_us", "structured_json", "none", "AA", 100)
        )
        q3 = q3_effort_analog(group_cells(records))
        self.assertAlmostEqual(
            q3["per_door_high_minus_none"]["mantle"]["diff"], -0.5
        )
        self.assertAlmostEqual(
            q3["per_door_high_minus_none"]["runtime_us"]["diff"], 0.0
        )
        self.assertAlmostEqual(
            q3["cross_door_dod"]["mantle_vs_runtime_us"]["dod"], -0.5
        )


class Q5Test(unittest.TestCase):
    def test_dispersion_and_association(self):
        usage = lambda n: {"output_tokens_details": {"reasoning_tokens": n}}
        records = (
            [rec("openai_1p", "open_generation", "default", "M",
                 usage=usage(t)) for t in (100, 110, 120)]
            + [rec("openai_1p", "open_generation", "default", "D1",
                   usage=usage(400))]
        )
        q5 = q5_default_burn(group_cells(records))
        entry = q5["openai_1p|open_generation"]
        self.assertEqual(entry["reasoning_tokens"]["n"], 4)
        self.assertEqual(entry["association"]["modal_n"], 3)
        self.assertEqual(entry["association"]["modal_median"], 110)
        self.assertEqual(entry["association"]["divergent_median"], 400)

    def test_converse_falls_back_to_totals(self):
        records = [
            rec("runtime_us", "open_generation", "default", "M",
                usage={"outputTokens": 600}),
            rec("runtime_us", "open_generation", "default", "M",
                usage={"outputTokens": 700}),
        ]
        q5 = q5_default_burn(group_cells(records))
        entry = q5["runtime_us|open_generation"]
        self.assertEqual(entry["output_tokens_total_only"]["n"], 2)
        self.assertNotIn("association", entry)


class AnalyzeEndToEndTest(unittest.TestCase):
    def test_loader_counts_and_exclusions(self):
        import json
        import os
        import tempfile

        records = (
            burst("openai_1p", "structured_json", "none", "AA", 5)
            + burst("mantle", "structured_json", "none", "AA", 5)
            + [dict(rec("mantle", "structured_json", "none", "AA"),
                    ok=False, error_code="http_500")]
            + [{"mode": "study2-full", "ok": True}]  # ignored: not study 4
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "study4-test.jsonl")
            with open(path, "w") as fh:
                for record in records:
                    fh.write(json.dumps(record) + "\n")
            report = analyze([path])
        self.assertEqual(report["n_records"], 10)
        self.assertEqual(report["n_exclusions"], 1)
        self.assertEqual(report["exclusion_codes"], {"http_500": 1})
