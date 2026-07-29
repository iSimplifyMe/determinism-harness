"""Study-3 analysis: gates, registered estimators, cross-box readout.

Synthetic records only; committed BEFORE any confirmatory study-3 data,
per the discipline recorded after study 1's estimator miss.
"""
import json
import unittest

from analysis.analyze_study3 import (
    build_report,
    gate_cell3,
    group_records,
    positive_control_gate,
    q1_cells,
    q2_concurrency,
    q3_thinking,
    q4_cross_box,
    q4_decode_rate,
)


def _rec(text="out", box="metal", model="qwen3.6-35b", task="structured_json",
         sampling="greedy", thinking="think_off", concurrency=None,
         control=None, ok=True, stop="stop", wire="w1", eval_ns=None,
         out_tok=None):
    cell = f"{model}|{task}|{sampling}|{thinking}"
    if concurrency:
        cell += f"|c{concurrency}"
    record = {
        "schema": 3,
        "box": box,
        "cell": cell,
        "ok": ok,
        "stop_reason": stop,
        "text": text,
        "text_sha256": f"sha-{text}",
        "request_sha256": wire,
        "wire_sha256": wire,
        "response_model": "tag",
        "meta_model": model,
        "meta_task": task,
        "meta_sampling": sampling,
        "meta_thinking": thinking,
        "meta_hardware": box,
        "usage": {},
    }
    if concurrency:
        record["meta_concurrency"] = concurrency
    if control:
        record["meta_control"] = control
    if eval_ns is not None:
        record["usage"]["eval_duration_ns"] = eval_ns
    if out_tok is not None:
        record["usage"]["output_tokens"] = out_tok
    return record


class TestGate(unittest.TestCase):
    def test_wire_mismatch_kills_cell(self):
        gate = gate_cell3([_rec(wire="a"), _rec(wire="b")])
        self.assertTrue(gate["flags"]["negative_control_failed"])
        self.assertEqual(gate["valid"], [])

    def test_length_stop_excluded_not_fatal(self):
        gate = gate_cell3([_rec(), _rec(stop="length"), _rec(ok=False)])
        self.assertEqual(len(gate["valid"]), 1)
        self.assertEqual(gate["excluded"]["truncated_or_other_stop"], 1)
        self.assertEqual(gate["excluded"]["error"], 1)

    def test_warmups_dropped_and_counted(self):
        grouped, warmups = group_records(
            [_rec(), _rec(control="warmup"), _rec(control="warmup")]
        )
        self.assertEqual(warmups, 2)
        self.assertEqual(len(grouped), 1)

    def test_grouping_is_per_box(self):
        grouped, _ = group_records([_rec(box="metal"), _rec(box="cuda",
                                                            model="gpt-oss-20b")])
        self.assertEqual(len(grouped), 2)


class TestQ1(unittest.TestCase):
    def test_modal_share_and_wilson(self):
        records = [_rec(text="a"), _rec(text="a"), _rec(text="b")]
        grouped, _ = group_records(records)
        cells = q1_cells(grouped)
        entry = list(cells.values())[0]
        self.assertAlmostEqual(entry["metrics"]["modal_share"], 2 / 3)
        self.assertIn("wilson_ci", entry)
        self.assertEqual(entry["meta"]["hardware"], "metal")


class TestPositiveControl(unittest.TestCase):
    def test_fires_on_distinct_and_fails_on_identical(self):
        varied = [
            _rec(text=f"v{i}", task="open_generation", sampling="temp07")
            for i in range(5)
        ]
        flat = [
            _rec(text="same", task="open_generation", sampling="temp07",
                 model="qwen3.5-122b")
            for _ in range(5)
        ]
        cells = q1_cells(group_records(varied + flat)[0])
        gate = positive_control_gate(cells)
        by_key = {(g["box"], g["model"]): g["fired"] for g in gate["per_model"]}
        self.assertTrue(by_key[("metal", "qwen3.6-35b")])
        self.assertFalse(by_key[("metal", "qwen3.5-122b")])
        self.assertFalse(gate["all_fired"])


def _q2_records():
    records = []
    # dense: no concurrency effect (10/10 both arms, 2 tasks)
    for task in ("extraction", "classification"):
        for _ in range(10):
            records.append(_rec(text="d", model="qwen3-vl-32b", task=task))
            records.append(_rec(text="d", model="qwen3-vl-32b", task=task,
                                concurrency=4))
    # moe models: c1 perfect, c4 modal 5/10 on both tasks -> diff -0.5
    for model in ("qwen3.5-122b", "qwen3.6-35b"):
        for task in ("extraction", "classification"):
            for _ in range(10):
                records.append(_rec(text="m", model=model, task=task))
            for i in range(10):
                records.append(_rec(text="m" if i < 5 else f"x{i}",
                                    model=model, task=task, concurrency=4))
    return records


class TestQ2(unittest.TestCase):
    def test_registered_stratified_estimator_and_dod(self):
        cells = q1_cells(group_records(_q2_records())[0])
        q2 = q2_concurrency(cells)
        per_model = {m["model"]: m for m in q2["per_model"]}
        self.assertAlmostEqual(per_model["qwen3-vl-32b"]["diff"], 0.0)
        self.assertAlmostEqual(per_model["qwen3.6-35b"]["diff"], -0.5)
        self.assertEqual(per_model["qwen3.6-35b"]["n_strata"], 2)
        dod = q2["moe_minus_dense"]
        self.assertAlmostEqual(dod["diff"], -0.5)
        self.assertEqual(dod["moe_strata"], 4)
        self.assertEqual(dod["dense_strata"], 2)
        self.assertIn("ci95", dod)

    def test_missing_comparator_stratum_skipped(self):
        records = [_rec(model="qwen3.6-35b", task="extraction", concurrency=4)]
        cells = q1_cells(group_records(records)[0])
        q2 = q2_concurrency(cells)
        self.assertEqual(q2["per_model"], [])
        self.assertEqual(q2["skipped_strata"], 1)


class TestQ3(unittest.TestCase):
    def test_on_minus_off_diff(self):
        records = []
        for i in range(10):
            records.append(_rec(text="s"))  # off arm: 10/10
            records.append(_rec(text="s" if i < 6 else f"v{i}",
                                thinking="think_on"))
        cells = q1_cells(group_records(records)[0])
        q3 = q3_thinking(cells)
        entry = {m["model"]: m for m in q3["per_model"]}["qwen3.6-35b"]
        self.assertAlmostEqual(entry["diff"], -0.4)
        self.assertIn("ci95", entry)
        self.assertEqual(entry["on_arm"], "think_on")


class TestQ4(unittest.TestCase):
    def test_cross_box_identity_and_overlap(self):
        records = []
        for _ in range(4):
            records.append(_rec(text="same", box="metal", model="gpt-oss-20b",
                                thinking="effort_low"))
            records.append(_rec(text="same", box="cuda", model="gpt-oss-20b",
                                thinking="effort_low"))
        for _ in range(4):
            records.append(_rec(text="m-only", box="metal", model="gpt-oss-20b",
                                task="extraction", thinking="effort_low"))
            records.append(_rec(text="c-only", box="cuda", model="gpt-oss-20b",
                                task="extraction", thinking="effort_low"))
        cells = q1_cells(group_records(records)[0])
        q4 = q4_cross_box(cells)
        by_task = {e["cell"]: e for e in q4["cells"]}
        match = by_task["gpt-oss-20b|structured_json|greedy|effort_low"]
        self.assertTrue(match["modal_match"])
        self.assertEqual(match["overlap"]["shared_variants"], 1)
        miss = by_task["gpt-oss-20b|extraction|greedy|effort_low"]
        self.assertFalse(miss["modal_match"])
        self.assertEqual(miss["overlap"]["shared_variants"], 0)
        self.assertAlmostEqual(
            miss["overlap"]["metal_records_matched_in_cuda"], 0.0
        )
        self.assertEqual(q4["cells_compared"], 2)
        self.assertEqual(q4["modal_matches"], 1)

    def test_decode_rate_per_box(self):
        records = [
            _rec(box="cuda", model="gpt-oss-20b", eval_ns=1_000_000_000,
                 out_tok=80, thinking="effort_low"),
            _rec(box="cuda", model="gpt-oss-20b", eval_ns=2_000_000_000,
                 out_tok=80, thinking="effort_low"),
        ]
        rates = q4_decode_rate(records)
        entry = rates["cuda|gpt-oss-20b"]
        self.assertEqual(entry["n"], 2)
        self.assertAlmostEqual(entry["p50"], 60.0)  # median of 80 and 40


class TestBuildReport(unittest.TestCase):
    def test_end_to_end_serializable(self):
        records = _q2_records() + [_rec(control="warmup")]
        report = build_report(records)
        json.dumps(report)
        self.assertEqual(report["registered"], True)
        self.assertEqual(report["totals"]["warmups_excluded"], 1)
        self.assertIn("q1_cells", report)
        self.assertIn("q2_concurrency", report)
        self.assertIn("q3_thinking", report)
        self.assertIn("q4_cross_box", report)
        self.assertIn("gates", report)

    def test_md_escapes_cell_pipes(self):
        import os
        import tempfile

        from analysis.analyze_study3 import write_md

        report = build_report(_q2_records())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.md")
            write_md(report, path)
            with open(path, encoding="utf-8") as fh:
                md = fh.read()
        self.assertIn("qwen3.5-122b\\|classification", md)


if __name__ == "__main__":
    unittest.main()
