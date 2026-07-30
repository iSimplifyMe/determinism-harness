"""Validator-catchability reanalysis: what fraction of observed byte-level
instability would a deterministic output validator have caught or
neutralized, and what fraction passes validation while semantically
divergent? Synthetic records only — stdlib, no network."""
import hashlib
import json
import unittest

from analysis.validator_catchability import (
    CLASSIFICATION_LABELS,
    build_validator_report,
    cell_accounting,
    config_class,
    task_family,
    validate_response,
)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


GOOD_SJ = json.dumps({
    "sku": "CS220-BDG-K",
    "name": "Corvid CS-220 badge printer",
    "quantity_on_hand": 17,
    "unit_price_usd": 349.50,
    "reorder_threshold": 6,
    "in_stock": True,
})


class TestValidateResponse(unittest.TestCase):
    def test_sj_strict_accepted(self):
        v = validate_response("structured_json", GOOD_SJ)
        self.assertEqual(v["coverage"], "validatable")
        self.assertTrue(v["accepted"])
        self.assertEqual(v["mode"], "strict")
        self.assertIsNotNone(v["semantic"])

    def test_sj_fenced_accepted_same_semantic(self):
        fenced = f"```json\n{GOOD_SJ}\n```"
        strict = validate_response("structured_json", GOOD_SJ)
        v = validate_response("structured_json", fenced)
        self.assertTrue(v["accepted"])
        self.assertEqual(v["mode"], "fenced")
        self.assertEqual(v["semantic"], strict["semantic"])

    def test_sj_parse_fail_rejected(self):
        v = validate_response("structured_json", "not json at all")
        self.assertFalse(v["accepted"])
        self.assertEqual(v["reason"], "parse_fail")

    def test_sj_schema_violations_rejected(self):
        obj = json.loads(GOOD_SJ)
        missing = {k: v for k, v in obj.items() if k != "sku"}
        extra = dict(obj, surprise=1)
        wrong_type = dict(obj, quantity_on_hand="17")
        bool_as_number = dict(obj, reorder_threshold=True)
        for bad in (missing, extra, wrong_type, bool_as_number):
            v = validate_response("structured_json", json.dumps(bad))
            self.assertFalse(v["accepted"], bad)
            self.assertEqual(v["reason"], "schema_fail")

    def test_classification_membership(self):
        ok = validate_response("classification", "BILLING\n")
        self.assertTrue(ok["accepted"])
        self.assertEqual(ok["semantic"], "BILLING")
        for bad in ("billing", "BILLING.", "Category: BILLING"):
            v = validate_response("classification", bad)
            self.assertFalse(v["accepted"], bad)
        self.assertIn("BILLING", CLASSIFICATION_LABELS)

    def test_extraction_format_only(self):
        ok = validate_response("extraction", " PO-83614-QN\n")
        self.assertTrue(ok["accepted"])
        self.assertEqual(ok["semantic"], "PO-83614-QN")
        # format-valid but WRONG value still passes — the validator is
        # format-only, which is exactly the invisible-divergence risk
        wrong = validate_response("extraction", "PO-99999-ZZ")
        self.assertTrue(wrong["accepted"])
        prose = validate_response("extraction", "The PO is PO-83614-QN")
        self.assertFalse(prose["accepted"])

    def test_extraction_pad_family(self):
        self.assertEqual(task_family("extraction_pad_50k"), "extraction")
        v = validate_response("extraction_pad_50k", "PO-83614-QN")
        self.assertTrue(v["accepted"])

    def test_open_generation_unvalidatable(self):
        v = validate_response("open_generation", "some prose")
        self.assertEqual(v["coverage"], "unvalidatable")
        self.assertIsNone(v["accepted"])


class TestConfigClass(unittest.TestCase):
    def test_sampled_flags(self):
        self.assertEqual(
            config_class({"meta_control": "positive"}), "sampled"
        )
        self.assertEqual(
            config_class({"meta_sampling": "temp07"}), "sampled"
        )
        self.assertEqual(
            config_class({"meta_sampling": "greedy"}), "deterministic"
        )
        self.assertEqual(config_class({}), "deterministic")


class TestCellAccounting(unittest.TestCase):
    def test_cosmetic_variance_recovered(self):
        fenced = f"```json\n{GOOD_SJ}\n```"
        texts = [GOOD_SJ, GOOD_SJ, GOOD_SJ, fenced]
        acct = cell_accounting("structured_json", texts)
        self.assertEqual(acct["n"], 4)
        self.assertEqual(acct["byte_distinct"], 2)
        self.assertEqual(acct["byte_modal_share"], 0.75)
        self.assertEqual(acct["rejected"], 0)
        self.assertEqual(acct["semantic_distinct"], 1)
        self.assertEqual(acct["post_validator_repro"], 1.0)
        self.assertEqual(acct["invisible_divergence_rate"], 0.0)
        self.assertEqual(acct["recovered"], 0.25)

    def test_semantic_divergence_is_invisible(self):
        other = json.dumps(dict(json.loads(GOOD_SJ), quantity_on_hand=18))
        texts = [GOOD_SJ] * 3 + [other]
        acct = cell_accounting("structured_json", texts)
        self.assertEqual(acct["semantic_distinct"], 2)
        self.assertEqual(acct["post_validator_repro"], 0.75)
        self.assertEqual(acct["invisible_divergence_rate"], 0.25)
        self.assertEqual(acct["recovered"], 0.0)

    def test_rejects_counted_not_invisible(self):
        texts = [GOOD_SJ] * 3 + ["garbage"]
        acct = cell_accounting("structured_json", texts)
        self.assertEqual(acct["rejected"], 1)
        self.assertEqual(acct["reject_rate"], 0.25)
        self.assertEqual(acct["post_validator_repro"], 1.0)
        self.assertEqual(acct["invisible_divergence_rate"], 0.0)

    def test_unvalidatable_reports_bytes_only(self):
        acct = cell_accounting("open_generation", ["a", "a", "b"])
        self.assertEqual(acct["coverage"], "unvalidatable")
        self.assertAlmostEqual(acct["byte_modal_share"], 2 / 3)
        self.assertNotIn("post_validator_repro", acct)


def _record(schema, cell, task, text, box=None, ok=True, stop=None,
            control=None, sampling=None):
    stop = stop or ("stop" if schema == 3 else "end_turn")
    record = {
        "schema": schema,
        "cell": cell,
        "meta_model": cell.split("|")[0],
        "meta_task": task,
        "ok": ok,
        "stop_reason": stop,
        "text": text,
        "text_sha256": _sha(text),
    }
    if box:
        record["box"] = box
    if control:
        record["meta_control"] = control
    if sampling:
        record["meta_sampling"] = sampling
    return record


class TestBuildReport(unittest.TestCase):
    def test_groups_by_study_and_box_excludes_warmups(self):
        records = [
            _record(1, "opus-5|structured_json|us|adaptive",
                    "structured_json", GOOD_SJ),
            _record(3, "gpt-oss-20b|structured_json|greedy|effort_low",
                    "structured_json", GOOD_SJ, box="cuda",
                    sampling="greedy"),
            _record(3, "warmup|gpt-oss-20b", "warmup", "",
                    box="cuda", control="warmup"),
            _record(2, "opus-5|classification|bedrock|adaptive",
                    "classification", "BILLING"),
        ]
        report = build_validator_report(records)
        keys = set(report["cells"])
        self.assertIn("study1::opus-5|structured_json|us|adaptive", keys)
        self.assertIn(
            "study3::cuda::gpt-oss-20b|structured_json|greedy|effort_low",
            keys,
        )
        self.assertEqual(report["totals"]["warmups_excluded"], 1)
        self.assertEqual(report["totals"]["records_analyzed"], 3)

    def test_aggregates_pool_by_study_task_class(self):
        records = (
            [_record(1, "opus-5|structured_json|us|adaptive",
                     "structured_json", GOOD_SJ)] * 3
            + [_record(1, "opus-5|structured_json|us|adaptive",
                       "structured_json", f"```json\n{GOOD_SJ}\n```")]
        )
        report = build_validator_report(records)
        agg = report["aggregates"]["study1::structured_json::deterministic"]
        self.assertEqual(agg["n"], 4)
        self.assertEqual(agg["byte_modal_pooled"], 0.75)
        self.assertEqual(agg["post_validator_pooled"], 1.0)


if __name__ == "__main__":
    unittest.main()
