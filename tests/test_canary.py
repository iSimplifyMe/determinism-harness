"""Reproducibility canary: schedule, baselines builder, run analyzer.
Synthetic records only — stdlib, no network."""
import hashlib
import json
import unittest

from harness.config import CANARY, MODELS, PLANES
from harness.runner import STUDY2_MODES, build_schedule
from harness.canary_baselines import binom_band, build_baselines
from analysis.analyze_canary import evaluate_canary

GOOD_SJ = json.dumps({
    "sku": "CS220-BDG-K",
    "name": "Corvid CS-220 badge printer",
    "quantity_on_hand": 17,
    "unit_price_usd": 349.50,
    "reorder_threshold": 6,
    "in_stock": True,
})
ALT_SJ = json.dumps({
    "sku": "CS220-BDG-K",
    "name": "Item Corvid CS-220 badge printer",
    "quantity_on_hand": 17,
    "unit_price_usd": 349.50,
    "reorder_threshold": 6,
    "in_stock": True,
})


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class TestCanarySchedule(unittest.TestCase):
    def test_counts_and_cell_key_parity_with_study2(self):
        items = build_schedule("canary", )
        per_pair = (
            CANARY["n_sj"] + CANARY["n_extraction"] + CANARY["n_classification"]
        )
        self.assertEqual(len(items), 9 * per_pair)
        self.assertIn("canary", STUDY2_MODES)
        cells = {it["cell"] for it in items}
        # exact study-2 cell keys, so baselines look up directly
        self.assertIn("opus-5|structured_json|bedrock|adaptive", cells)
        self.assertIn("haiku-4-5|structured_json|anthropic_api|none", cells)
        self.assertIn("sonnet-5|extraction|p_aws|adaptive", cells)
        for it in items:
            self.assertEqual(it["meta"].get("canary"), True)
            self.assertIn(it["plane"], PLANES)

    def test_arm_per_family(self):
        items = build_schedule("canary")
        for it in items:
            model = it["meta"]["model"]
            expected = "adaptive" if MODELS[model]["family"] == "claude-5" else "none"
            self.assertEqual(it["meta"]["thinking"], expected)


class TestBinomBand(unittest.TestCase):
    def test_band_contains_bulk_and_respects_tails(self):
        lo, hi = binom_band(10, 0.85)
        self.assertLessEqual(lo, 8)
        self.assertEqual(hi, 10)
        self.assertGreater(lo, 3)
        lo0, hi0 = binom_band(10, 0.0)
        self.assertEqual((lo0, hi0), (0, 0))
        lo1, hi1 = binom_band(10, 1.0)
        self.assertEqual((lo1, hi1), (10, 10))


def _s2_record(model, task, plane, thinking, text, ok=True):
    return {
        "schema": 2,
        "cell": f"{model}|{task}|{plane}|{thinking}",
        "plane": plane,
        "meta_model": model,
        "meta_task": task,
        "meta_thinking": thinking,
        "ok": ok,
        "stop_reason": "end_turn",
        "text": text,
        "text_sha256": _sha(text),
        "response_model": (
            "claude-haiku-4-5-20251001" if model == "haiku-4-5"
            else f"claude-{model.replace('-5', '')}-5"
        ),
        "latency_ms": 1500,
        "usage": {"output_tokens": 100},
    }


def _baseline_inputs():
    records = []
    for plane in PLANES:
        records += [
            _s2_record("opus-5", "structured_json", plane, "adaptive", GOOD_SJ)
        ] * 8
        records += [
            _s2_record("opus-5", "structured_json", plane, "adaptive",
                       f"```json\n{GOOD_SJ}\n```")
        ] * 2
        records += [
            _s2_record("haiku-4-5", "structured_json", plane, "none", GOOD_SJ)
        ] * 9
        records += [
            _s2_record("haiku-4-5", "structured_json", plane, "none", ALT_SJ)
        ]
        records += [
            _s2_record("opus-5", "extraction", plane, "adaptive", "PO-83614-QN")
        ] * 5
        records += [
            _s2_record("opus-5", "classification", plane, "adaptive", "BILLING")
        ] * 5
        records += [
            _s2_record("haiku-4-5", "extraction", plane, "none", "PO-83614-QN")
        ] * 5
        records += [
            _s2_record("haiku-4-5", "classification", plane, "none", "BILLING")
        ] * 5
    return records


class TestBuildBaselines(unittest.TestCase):
    def test_fence_rates_goldens_and_corpus(self):
        baselines = build_baselines(_baseline_inputs())
        cell = baselines["cells"]["opus-5|structured_json|bedrock|adaptive"]
        self.assertAlmostEqual(cell["fence_rate"], 0.2)
        self.assertEqual(len(cell["sj_semantic_goldens"]), 1)
        self.assertEqual(len(cell["byte_sha_corpus"]), 2)
        haiku = baselines["cells"]["haiku-4-5|structured_json|bedrock|none"]
        self.assertEqual(len(haiku["sj_semantic_goldens"]), 2)
        self.assertEqual(baselines["extraction_golden"], "PO-83614-QN")
        cls = baselines["cells"]["opus-5|classification|bedrock|adaptive"]
        self.assertEqual(cls["golden_label"], "BILLING")
        self.assertIn(
            "claude-haiku-4-5-20251001",
            baselines["response_models"]["haiku-4-5"],
        )


def _canary_record(model, task, plane, thinking, text, response_model=None):
    record = _s2_record(model, task, plane, thinking, text)
    record["meta_canary"] = True
    if response_model:
        record["response_model"] = response_model
    return record


class TestEvaluateCanary(unittest.TestCase):
    def setUp(self):
        self.baselines = build_baselines(_baseline_inputs())

    def _green_run(self):
        records = []
        for plane in PLANES:
            records += [
                _canary_record("opus-5", "structured_json", plane, "adaptive",
                               GOOD_SJ)
            ] * 8
            records += [
                _canary_record("opus-5", "structured_json", plane, "adaptive",
                               f"```json\n{GOOD_SJ}\n```")
            ] * 2
            records += [
                _canary_record("opus-5", "extraction", plane, "adaptive",
                               "PO-83614-QN")
            ] * 3
            records += [
                _canary_record("opus-5", "classification", plane, "adaptive",
                               "BILLING")
            ] * 3
        return records

    def test_green_run(self):
        result = evaluate_canary(self._green_run(), self.baselines)
        self.assertEqual(result["status"], "GREEN")
        self.assertFalse(result["red"])
        self.assertFalse(result["yellow"])

    def test_wrong_extraction_is_red(self):
        records = self._green_run()
        records.append(
            _canary_record("opus-5", "extraction", "bedrock", "adaptive",
                           "PO-99999-ZZ")
        )
        result = evaluate_canary(records, self.baselines)
        self.assertEqual(result["status"], "RED")
        self.assertTrue(any("extraction" in c["check"] for c in result["red"]))

    def test_semantic_sj_divergence_is_red(self):
        bad = json.dumps(dict(json.loads(GOOD_SJ), quantity_on_hand=99))
        records = self._green_run()
        records.append(
            _canary_record("opus-5", "structured_json", "bedrock", "adaptive",
                           bad)
        )
        result = evaluate_canary(records, self.baselines)
        self.assertEqual(result["status"], "RED")

    def test_novel_byte_variant_with_golden_semantics_is_yellow(self):
        records = self._green_run()
        records.append(
            _canary_record("opus-5", "structured_json", "bedrock", "adaptive",
                           GOOD_SJ + "\n")
        )
        result = evaluate_canary(records, self.baselines)
        self.assertEqual(result["status"], "YELLOW")
        self.assertTrue(any("novel" in c["check"] for c in result["yellow"]))

    def test_fence_rate_outside_band_is_yellow(self):
        records = []
        for plane in PLANES:
            fenced = [
                _canary_record("opus-5", "structured_json", plane, "adaptive",
                               f"```json\n{GOOD_SJ}\n```")
            ] * 10
            records += fenced
            records += [
                _canary_record("opus-5", "extraction", plane, "adaptive",
                               "PO-83614-QN")
            ] * 3
            records += [
                _canary_record("opus-5", "classification", plane, "adaptive",
                               "BILLING")
            ] * 3
        result = evaluate_canary(records, self.baselines)
        self.assertIn(result["status"], ("YELLOW", "RED"))
        self.assertTrue(any("fence" in c["check"] for c in result["yellow"]))

    def test_response_model_drift_is_red(self):
        records = self._green_run()
        records.append(
            _canary_record("opus-5", "classification", "bedrock", "adaptive",
                           "BILLING", response_model="claude-opus-5-20270101")
        )
        result = evaluate_canary(records, self.baselines)
        self.assertEqual(result["status"], "RED")
        self.assertTrue(any("response_model" in c["check"] for c in result["red"]))

    def test_frontier_label_flip_is_yellow(self):
        records = self._green_run()
        records.append(
            _canary_record("opus-5", "classification", "bedrock", "adaptive",
                           "TECHNICAL")
        )
        result = evaluate_canary(records, self.baselines)
        self.assertEqual(result["status"], "YELLOW")

    def test_transport_failures_are_red(self):
        records = self._green_run()
        failed = _canary_record("opus-5", "extraction", "bedrock", "adaptive",
                                "")
        failed["ok"] = False
        records.append(failed)
        result = evaluate_canary(records, self.baselines)
        self.assertEqual(result["status"], "RED")


if __name__ == "__main__":
    unittest.main()
